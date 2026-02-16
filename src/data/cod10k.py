"""COD10K dataset loading utilities.

Provides functions to load image-mask pairs from the COD10K dataset structure.
"""

from pathlib import Path
from typing import Optional


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

    # Limit samples if requested
    if max_samples is not None and max_samples > 0:
        pairs = pairs[:max_samples]
        print(f"Using {len(pairs)} samples (max_samples={max_samples})")

    return pairs
