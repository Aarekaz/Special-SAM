"""Training loop for LLPM + SAM decoder joint fine-tuning.

The LLPM (Learnable Local Preprocessing Module) sits before SAM's frozen
ViT-H encoder. Unlike decoder-only training, pre-computed embeddings CANNOT
be used here — the encoder must run every forward pass so gradients flow
through the input back to LLPM.

Forward pass:
    image -> LLPM -> clamp(0,255) -> SAM normalize -> frozen encoder
    -> prompt encoder (no_grad) -> trainable decoder -> loss

Usage:
    python -m src.training.train_llpm --config configs/train_llpm.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data.image_dataset import CamoImageDataset
from src.models.decoder import freeze_encoder, unfreeze_decoder
from src.models.sam_loader import get_device, load_llpm_sam
from src.training.loss import bce_dice_loss


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sam_normalize(x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Apply SAM's pixel normalization.

    SAM registers pixel_mean and pixel_std as buffers with shape (3,1,1).
    We unsqueeze to (1,3,1,1) for broadcasting over (B,3,H,W).

    Args:
        x: Image tensor (B, 3, H, W) in [0, 255].
        model: SAM model with pixel_mean/pixel_std buffers.

    Returns:
        Normalized tensor.
    """
    pixel_mean = model.pixel_mean.unsqueeze(0)  # (1, 3, 1, 1)
    pixel_std = model.pixel_std.unsqueeze(0)    # (1, 3, 1, 1)
    return (x - pixel_mean) / pixel_std


def train_llpm(config: dict) -> None:
    """Run the LLPM + decoder joint training loop.

    Key differences from decoder-only training:
    - No pre-computed embeddings; encoder runs every forward pass.
    - Encoder params are frozen (requires_grad=False) but forward pass
      computes input gradients for LLPM backprop.
    - Mixed precision via autocast + GradScaler.
    - Prompt encoder is in torch.no_grad() (no path back to LLPM).

    Args:
        config: Training configuration dict (from train_llpm.yaml).
    """
    set_seed(config["seed"])
    device = get_device()

    # Load SAM + LLPM
    decoder_init = config["model"].get("decoder_init_path")
    llpm_cfg = config.get("llpm", {})

    model, llpm = load_llpm_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["checkpoint"],
        decoder_path=decoder_init,
        llpm_config=llpm_cfg,
        device=device,
    )

    # Freeze encoder + prompt encoder, unfreeze decoder
    freeze_encoder(model)
    unfreeze_decoder(model)
    model.train()
    llpm.train()

    print(f"LLPM parameters: {llpm.count_parameters():,}")
    decoder_params = sum(p.numel() for p in model.mask_decoder.parameters()
                         if p.requires_grad)
    print(f"Decoder parameters: {decoder_params:,}")

    # Optimizer: LLPM + decoder parameters
    trainable_params = list(llpm.parameters()) + list(model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    # Dataset
    train_ds = CamoImageDataset(
        img_dir=config["data"]["train_img_dir"],
        mask_dir=config["data"]["train_mask_dir"],
        target_size=config["preprocessing"]["target_size"],
        augment=config["preprocessing"]["augmentation"].get("horizontal_flip", True),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    epochs = config["training"]["epochs"]
    bce_w = config["training"]["loss"]["bce_weight"]
    dice_w = config["training"]["loss"]["dice_weight"]

    # LR scheduler
    scheduler = None
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    lr_sched_type = config["training"].get("lr_scheduler")
    if lr_sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )

    # Mixed precision
    scaler = GradScaler()

    print(f"\nTraining LLPM+decoder for {epochs} epochs on {len(train_ds)} samples...")
    if scheduler:
        print(f"  LR scheduler: {lr_sched_type}, warmup: {warmup_epochs} epochs")
    if decoder_init:
        print(f"  Decoder warm-start from: {decoder_init}")

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (image, gt_mask, point, label, box) in enumerate(train_loader):
            image = image.to(device)
            gt_mask = gt_mask.to(device)
            point = point.to(device)
            label = label.to(device)
            box = box.to(device)

            optimizer.zero_grad()

            with autocast():
                # LLPM preprocessing
                enhanced = llpm(image)
                enhanced = enhanced.clamp(0, 255)

                # SAM normalization
                normalized = sam_normalize(enhanced, model)

                # Frozen encoder — no torch.no_grad()! Gradients flow through
                # input back to LLPM even though encoder params are frozen.
                image_embedding = model.image_encoder(normalized)

                # Prompt encoder — no gradient path back to LLPM
                use_box = random.random() > 0.5
                with torch.no_grad():
                    if use_box:
                        sparse, dense = model.prompt_encoder(
                            points=None, boxes=box, masks=None
                        )
                    else:
                        sparse, dense = model.prompt_encoder(
                            points=(point, label), boxes=None, masks=None
                        )

                # Decoder
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )

                pred_mask = F.interpolate(
                    low_res_masks,
                    size=(gt_mask.shape[2], gt_mask.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

                loss = bce_dice_loss(pred_mask, gt_mask, bce_w, dice_w)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]["lr"]
        alpha_val = llpm.alpha.item()
        alpha_grad = llpm.alpha.grad
        alpha_grad_val = alpha_grad.item() if alpha_grad is not None else 0.0
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
            f"LR: {lr:.2e} | alpha: {alpha_val:.4f} | "
            f"alpha_grad: {alpha_grad_val:.6f}"
        )

        if scheduler and epoch >= warmup_epochs:
            scheduler.step()

    # Save checkpoints
    llpm_path = Path(config["output"]["llpm_path"])
    llpm_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(llpm.state_dict(), str(llpm_path))
    print(f"Saved LLPM to {llpm_path}")

    decoder_path = Path(config["output"]["decoder_path"])
    decoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.mask_decoder.state_dict(), str(decoder_path))
    print(f"Saved decoder to {decoder_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train LLPM + SAM decoder for camouflaged object detection"
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_llpm.yaml",
        help="Path to LLPM training config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_llpm(config)


if __name__ == "__main__":
    main()
