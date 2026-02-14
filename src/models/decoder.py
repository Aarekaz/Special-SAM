"""Decoder freeze/unfreeze utilities for SAM fine-tuning.

During decoder-only fine-tuning, the image encoder and prompt encoder
are frozen while only the mask decoder parameters are trainable.
"""

import torch.nn as nn


def freeze_encoder(model: nn.Module) -> None:
    """Freeze image encoder and prompt encoder, leaving decoder trainable.

    This is the core strategy for decoder-only specialization:
    the heavy image encoder stays frozen (preserving SAM's learned
    representations) while only the lightweight decoder adapts.

    Args:
        model: A SAM model instance.
    """
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False


def unfreeze_decoder(model: nn.Module) -> None:
    """Ensure all mask decoder parameters are trainable.

    Args:
        model: A SAM model instance.
    """
    for p in model.mask_decoder.parameters():
        p.requires_grad = True


def get_trainable_params(model: nn.Module) -> list:
    """Return list of trainable parameters (for optimizer).

    Args:
        model: A SAM model instance.

    Returns:
        List of parameters with requires_grad=True.
    """
    return [p for p in model.parameters() if p.requires_grad]
