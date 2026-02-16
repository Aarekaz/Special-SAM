"""Image and mask transformation utilities.

Provides preprocessing functions for preparing images and masks for SAM.
"""

import cv2
import numpy as np


def resize_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and mask to target size for SAM input.

    SAM expects square images of size 1024x1024 by default.

    Args:
        image: Input image (H, W, 3) in RGB format.
        mask: Ground truth mask (H, W) with values 0-255.
        target_size: Target size for both dimensions (default 1024).

    Returns:
        Tuple of (resized_image, resized_mask).
        - resized_image: (target_size, target_size, 3) RGB
        - resized_mask: (target_size, target_size) with values 0-255

    Example:
        >>> img = cv2.imread("image.jpg")
        >>> img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        >>> mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
        >>> img_resized, mask_resized = resize_image_mask(img, mask, 1024)
    """
    # Resize image using bilinear interpolation
    img_resized = cv2.resize(
        image, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )

    # Resize mask using nearest neighbor to preserve binary values
    mask_resized = cv2.resize(
        mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )

    return img_resized, mask_resized


def horizontal_flip(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip image and mask horizontally for data augmentation.

    Args:
        image: Input image (H, W, 3) in RGB format.
        mask: Ground truth mask (H, W) with values 0-255.

    Returns:
        Tuple of (flipped_image, flipped_mask).

    Example:
        >>> img_flipped, mask_flipped = horizontal_flip(img, mask)
    """
    img_flipped = cv2.flip(image, 1)  # 1 = horizontal flip
    mask_flipped = cv2.flip(mask, 1)

    return img_flipped, mask_flipped
