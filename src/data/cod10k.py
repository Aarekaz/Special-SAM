"""COD10K dataset loading utilities.

Provides functions to load image-mask pairs from the COD10K dataset structure,
and a Dataset class for loading pre-computed embeddings during training.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def get_image_mask_pairs(
    img_dir: str | Path,
    mask_dir: str | Path,
    max_samples: Optional[int] = None,
) -> list[tuple[str, str]]:
    """Get matching image and mask file pairs from COD10K dataset.

    Assumes filenames match (ignoring extension). For example:
        Image: COD10K-CAM-1-Aquatic-1-BatFish-1.jpg
        Mask:  COD10K-CAM-1-Aquatic-1-BatFish-1.png

    Args:
        img_dir: Directory containing test images.
        mask_dir: Directory containing ground truth masks.
        max_samples: Maximum number of samples to return (None = all).

    Returns:
        List of (image_path, mask_path) tuples as strings.

    Example:
        >>> pairs = get_image_mask_pairs(
        ...     "data/cod10k/Test/Image",
        ...     "data/cod10k/Test/GT_Object",
        ...     max_samples=200
        ... )
        >>> print(f"Found {len(pairs)} image-mask pairs")
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    # Get all image files (jpg or png)
    img_files = sorted(
        list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
    )
    print(f"Found {len(img_files)} images in {img_dir}")

    # Get all mask files and create lookup dict: stem -> path
    mask_files = sorted(
        list(mask_dir.rglob("*.png")) + list(mask_dir.rglob("*.jpg"))
    )
    mask_dict = {m.stem: m for m in mask_files}
    print(f"Found {len(mask_files)} masks in {mask_dir}")

    # Match images to masks
    pairs = []
    for img_path in img_files:
        stem = img_path.stem
        if stem in mask_dict:
            pairs.append((str(img_path), str(mask_dict[stem])))

    print(f"Matched {len(pairs)} image-mask pairs")

    # Limit samples if requested (0 or None = use all)
    if max_samples is not None and max_samples > 0:
        pairs = pairs[:max_samples]
        print(f"Using {len(pairs)} samples (max_samples={max_samples})")
    else:
        print(f"Using all {len(pairs)} samples")

    return pairs


class CamoDataset(Dataset):
    """Dataset for loading pre-computed SAM embeddings with prompts.

    Used by decoder-only training where the frozen encoder's outputs
    are cached as .npy files to avoid recomputing each epoch.

    Args:
        df: DataFrame with columns from precompute metadata CSV:
            embed_path, mask_path, prompt_x, prompt_y,
            box_xmin, box_ymin, box_xmax, box_ymax.
        device: Target device for tensors.
    """

    def __init__(self, df, device=None):
        self.df = df.reset_index(drop=True)
        self.device = device or torch.device("cpu")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        emb = torch.from_numpy(np.load(row["embed_path"])).to(self.device)
        mask = np.load(row["mask_path"])
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        point = torch.tensor(
            [[row["prompt_x"], row["prompt_y"]]], dtype=torch.float32
        ).to(self.device)
        label = torch.tensor([1], dtype=torch.int64).to(self.device)
        box = torch.tensor(
            [[row["box_xmin"], row["box_ymin"], row["box_xmax"], row["box_ymax"]]],
            dtype=torch.float32,
        ).to(self.device)

        return emb, mask, point, label, box
