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


def random_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    max_angle: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask by a random angle for data augmentation.

    Uses BORDER_REFLECT_101 for image and INTER_NEAREST for mask to
    preserve binary values.

    Args:
        image: Input image (H, W, 3) in RGB format.
        mask: Ground truth mask (H, W) with values 0-255.
        max_angle: Maximum rotation angle in degrees (symmetric).

    Returns:
        Tuple of (rotated_image, rotated_mask).
    """
    h, w = image.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    img_rot = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask_rot = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img_rot, mask_rot


def pixel_shift(
    image: np.ndarray,
    mask: np.ndarray,
    max_shift: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate image and mask by a random pixel offset.

    Shift up to half of ViT's 32px patch size to test sub-patch robustness.

    Args:
        image: Input image (H, W, 3) in RGB format.
        mask: Ground truth mask (H, W) with values 0-255.
        max_shift: Maximum shift in pixels (symmetric, both axes).

    Returns:
        Tuple of (shifted_image, shifted_mask).
    """
    h, w = image.shape[:2]
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    img_shifted = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask_shifted = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img_shifted, mask_shifted
