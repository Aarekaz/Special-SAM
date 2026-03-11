"""Raw image dataset for LLPM training.

Unlike CamoDataset which loads pre-computed embeddings, this dataset loads
raw images so the LLPM module can process them before the live encoder pass.
Images are returned as float32 in [0, 255] — NOT normalized — because LLPM
expects unnormalized input and SAM normalization is applied after LLPM.
"""

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cod10k import get_image_mask_pairs
from src.data.transforms import pixel_shift, random_rotation, resize_image_mask


class CamoImageDataset(Dataset):
    """Dataset that loads raw images with masks and prompts for LLPM training.

    Returns the same 5-tuple signature as CamoDataset:
        (image, mask, point, label, box)
    but with a raw image tensor instead of a pre-computed embedding.

    Args:
        img_dir: Directory containing training images.
        mask_dir: Directory containing ground truth masks.
        target_size: Resize dimension (default 1024).
        max_samples: Limit number of samples (None = all).
        augment: Whether to apply random horizontal flip.
        rotation: Whether to apply random rotation.
        max_rotation: Maximum rotation angle in degrees.
        shift: Whether to apply random pixel shift.
        max_shift: Maximum pixel shift amount.
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        target_size: int = 1024,
        max_samples: int | None = None,
        augment: bool = True,
        rotation: bool = False,
        max_rotation: float = 5.0,
        shift: bool = False,
        max_shift: int = 16,
    ):
        self.pairs = get_image_mask_pairs(img_dir, mask_dir, max_samples)
        self.target_size = target_size
        self.augment = augment
        self.rotation = rotation
        self.max_rotation = max_rotation
        self.shift = shift
        self.max_shift = max_shift

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img, mask = resize_image_mask(img, mask, self.target_size)

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # Random rotation
        if self.rotation and random.random() > 0.5:
            img, mask = random_rotation(img, mask, self.max_rotation)

        # Random pixel shift
        if self.shift and random.random() > 0.5:
            img, mask = pixel_shift(img, mask, self.max_shift)

        # Binarize mask
        mask_bin = (mask > 128).astype(np.float32)

        # Extract point prompt: random foreground pixel
        ys, xs = np.where(mask_bin > 0.5)
        if len(xs) == 0:
            # Fallback to center if no foreground
            cx, cy = self.target_size // 2, self.target_size // 2
            bx_min, by_min = 0, 0
            bx_max, by_max = self.target_size - 1, self.target_size - 1
        else:
            pt_idx = random.randint(0, len(xs) - 1)
            cx, cy = int(xs[pt_idx]), int(ys[pt_idx])
            bx_min, by_min = int(np.min(xs)), int(np.min(ys))
            bx_max, by_max = int(np.max(xs)), int(np.max(ys))

        # Image: (H, W, 3) uint8 -> (3, H, W) float32 [0, 255]
        image_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()

        mask_t = torch.from_numpy(mask_bin).unsqueeze(0)  # (1, H, W)
        point = torch.tensor([[cx, cy]], dtype=torch.float32)
        label = torch.tensor([1], dtype=torch.int64)
        box = torch.tensor([[bx_min, by_min, bx_max, by_max]], dtype=torch.float32)

        return image_t, mask_t, point, label, box
