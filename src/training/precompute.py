"""Pre-compute SAM image embeddings for efficient decoder training.

Running the frozen ViT-H image encoder on each training image is the
bottleneck (~2 hours on a V100). By pre-computing and caching embeddings
as .npy files, we eliminate this cost from the training loop entirely.

Usage:
    python -m src.training.precompute --config configs/train.yaml
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.data.cod10k import get_image_mask_pairs
from src.data.transforms import horizontal_flip, resize_image_mask
from src.models.sam_loader import get_device, load_sam, get_predictor


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def precompute_embeddings(config: dict) -> pd.DataFrame:
    """Pre-compute SAM embeddings for all training images.

    For each image, computes embeddings for both the original and a
    horizontally-flipped version (2x augmentation). Also extracts
    point prompts (random foreground pixel) and box prompts (bounding box).

    Args:
        config: Training configuration dict (from train.yaml).

    Returns:
        DataFrame with columns: embed_path, mask_path, prompt_x, prompt_y,
        box_xmin, box_ymin, box_xmax, box_ymax.
    """
    set_seed(config["seed"])
    device = get_device()
    target_size = config["preprocessing"]["target_size"]

    embed_dir = Path(config["embeddings"]["dir"])
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["checkpoint"],
        device=device,
    )
    predictor = get_predictor(model)

    # Get training pairs
    train_pairs = get_image_mask_pairs(
        config["data"]["train_img_dir"],
        config["data"]["train_mask_dir"],
    )

    metadata = []
    print(f"Pre-computing embeddings for {len(train_pairs)} images "
          f"(x2 with augmentation)...")

    for i, (img_path, mask_path) in enumerate(train_pairs):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_resized, mask_resized = resize_image_mask(img, mask, target_size)
        img_flipped, mask_flipped = horizontal_flip(img_resized, mask_resized)

        for aug_idx, (curr_img, curr_mask) in enumerate(
            [(img_resized, mask_resized), (img_flipped, mask_flipped)]
        ):
            aug_suffix = "_flip" if aug_idx == 1 else ""
            embed_path = embed_dir / f"train_{i:04d}{aug_suffix}_embed.npy"
            mask_save_path = embed_dir / f"train_{i:04d}{aug_suffix}_mask.npy"

            if not (embed_path.exists() and mask_save_path.exists()):
                predictor.set_image(curr_img)
                embedding = predictor.get_image_embedding().cpu().numpy()
                np.save(embed_path, embedding)
                np.save(mask_save_path, curr_mask)
            else:
                curr_mask = np.load(mask_save_path)

            # Extract prompts from mask
            ys, xs = np.where(curr_mask > 128)
            if len(xs) == 0:
                continue

            idx_pt = random.randint(0, len(xs) - 1)
            metadata.append({
                "embed_path": str(embed_path),
                "mask_path": str(mask_save_path),
                "prompt_x": int(xs[idx_pt]),
                "prompt_y": int(ys[idx_pt]),
                "box_xmin": int(np.min(xs)),
                "box_ymin": int(np.min(ys)),
                "box_xmax": int(np.max(xs)),
                "box_ymax": int(np.max(ys)),
            })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(train_pairs)}")

    df = pd.DataFrame(metadata)
    csv_path = config["embeddings"]["metadata_csv"]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved metadata for {len(df)} samples to {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute SAM image embeddings for training"
    )
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    precompute_embeddings(config)


if __name__ == "__main__":
    main()
