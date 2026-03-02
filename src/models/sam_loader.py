"""SAM model loading utilities.

Handles downloading weights, loading base SAM, and loading specialized
(fine-tuned decoder) SAM models.
"""

import os
import urllib.request
from pathlib import Path

import torch
from segment_anything import sam_model_registry, SamPredictor


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_weights(path: str, url: str) -> Path:
    """Download SAM checkpoint if it doesn't exist locally.

    Args:
        path: Local path to save the checkpoint.
        url: URL to download from.

    Returns:
        Path to the downloaded checkpoint.
    """
    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM weights to {path}...")
    urllib.request.urlretrieve(url, str(path))
    print(f"Downloaded: {path}")
    return path


def load_sam(
    model_type: str = "vit_h",
    checkpoint: str = "weights/sam_vit_h_4b8939.pth",
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load base SAM model.

    Args:
        model_type: SAM variant ('vit_h', 'vit_l', 'vit_b').
        checkpoint: Path to model checkpoint.
        device: Target device. Auto-detected if None.

    Returns:
        Loaded SAM model on the specified device.
    """
    if device is None:
        device = get_device()

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device)
    return model


def load_specialized_sam(
    model_type: str = "vit_h",
    checkpoint: str = "weights/sam_vit_h_4b8939.pth",
    decoder_path: str = "checkpoints/camo_decoder_vith.pth",
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load SAM with a fine-tuned decoder.

    Loads the base SAM model then replaces the mask decoder weights
    with the specialized (fine-tuned) decoder state dict.

    Args:
        model_type: SAM variant ('vit_h', 'vit_l', 'vit_b').
        checkpoint: Path to base SAM checkpoint.
        decoder_path: Path to fine-tuned decoder state dict.
        device: Target device. Auto-detected if None.

    Returns:
        SAM model with specialized decoder loaded.
    """
    if device is None:
        device = get_device()

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.mask_decoder.load_state_dict(
        torch.load(decoder_path, map_location=device)
    )
    model.to(device)
    return model


def load_llpm_sam(
    model_type: str = "vit_h",
    checkpoint: str = "weights/sam_vit_h_4b8939.pth",
    decoder_path: str | None = None,
    llpm_path: str | None = None,
    llpm_config: dict | None = None,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load SAM with LLPM module for preprocessing.

    Creates an LLPM module and a SAM model. Optionally loads pre-trained
    weights for both the decoder and LLPM.

    Args:
        model_type: SAM variant ('vit_h', 'vit_l', 'vit_b').
        checkpoint: Path to base SAM checkpoint.
        decoder_path: Optional path to fine-tuned decoder state dict.
        llpm_path: Optional path to pre-trained LLPM state dict.
        llpm_config: Dict with 'edge_channels' and 'enhance_channels'.
        device: Target device. Auto-detected if None.

    Returns:
        Tuple of (sam_model, llpm_module).
    """
    from src.models.llpm import LLPM

    if device is None:
        device = get_device()

    if llpm_config is None:
        llpm_config = {}

    model = sam_model_registry[model_type](checkpoint=checkpoint)

    if decoder_path is not None:
        model.mask_decoder.load_state_dict(
            torch.load(decoder_path, map_location=device)
        )

    model.to(device)

    llpm = LLPM(
        edge_channels=llpm_config.get("edge_channels", 32),
        enhance_channels=llpm_config.get("enhance_channels", 64),
        alpha_init=llpm_config.get("alpha_init", 0.1),
    )

    if llpm_path is not None:
        llpm.load_state_dict(torch.load(llpm_path, map_location=device))

    llpm.to(device)

    return model, llpm


def get_predictor(model: torch.nn.Module) -> SamPredictor:
    """Wrap a SAM model in a SamPredictor for inference.

    Args:
        model: A loaded SAM model.

    Returns:
        SamPredictor wrapping the model.
    """
    return SamPredictor(model)
