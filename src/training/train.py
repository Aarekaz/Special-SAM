"""Training loop for SAM decoder-only fine-tuning.

Freezes the image encoder and prompt encoder, trains only the mask decoder
using pre-computed embeddings with random point/box prompt switching.

Usage:
    python -m src.training.train --config configs/train.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.cod10k import CamoDataset
from src.models.decoder import freeze_encoder, unfreeze_decoder
from src.models.sam_loader import get_device, load_sam
from src.training.loss import bce_dice_loss


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict) -> None:
    """Run the full training loop.

    Loads the SAM model, freezes encoder layers, and trains only the
    mask decoder on pre-computed embeddings. Randomly switches between
    point prompts and box prompts to improve robustness.

    Args:
        config: Training configuration dict (from train.yaml).
    """
    set_seed(config["seed"])
    device = get_device()

    # Load model
    model = load_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["checkpoint"],
        device=device,
    )

    # Freeze encoder, train decoder only
    model.train()
    freeze_encoder(model)
    unfreeze_decoder(model)

    optimizer = torch.optim.AdamW(
        model.mask_decoder.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    # Load pre-computed embeddings
    csv_path = config["embeddings"]["metadata_csv"]
    train_df = pd.read_csv(csv_path)
    train_ds = CamoDataset(train_df, device=device)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    epochs = config["training"]["epochs"]
    bce_w = config["training"]["loss"]["bce_weight"]
    dice_w = config["training"]["loss"]["dice_weight"]

    # Learning rate scheduler
    scheduler = None
    lr_sched_type = config["training"].get("lr_scheduler")
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    if lr_sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )

    print(f"Training for {epochs} epochs on {len(train_ds)} samples...")
    if scheduler:
        print(f"  LR scheduler: {lr_sched_type}, warmup: {warmup_epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0.0

        for emb, gt_mask, point, label, box in train_loader:
            emb = emb.squeeze(1)

            # Prompt mixing: configurable via config
            prompt_mix = config["training"].get("prompt_mix", "mixed")
            if prompt_mix == "point_only":
                use_box = False
            elif prompt_mix == "box_only":
                use_box = True
            elif isinstance(prompt_mix, (int, float)):
                use_box = random.random() < float(prompt_mix)
            else:  # "mixed" or default
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

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=emb,
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if scheduler and epoch >= warmup_epochs:
            scheduler.step()

        # Save intermediate checkpoints
        save_every = config["training"].get("save_every_n_epochs")
        if save_every and (epoch + 1) % save_every == 0:
            ckpt_dir = Path(config["output"]["decoder_path"]).parent
            ckpt_path = ckpt_dir / f"decoder_epoch{epoch + 1}.pth"
            torch.save(model.mask_decoder.state_dict(), str(ckpt_path))
            print(f"  Saved intermediate checkpoint: {ckpt_path}")

    # Save decoder weights
    output_path = Path(config["output"]["decoder_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.mask_decoder.state_dict(), str(output_path))
    print(f"Saved decoder to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SAM decoder for camouflaged object detection"
    )
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
