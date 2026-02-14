"""Loss functions for SAM decoder fine-tuning.

Combines BCE and Dice loss for binary segmentation. The BCE component
handles pixel-wise classification while Dice addresses class imbalance
(camouflaged objects often occupy small regions).
"""

import torch
import torch.nn.functional as F


def bce_dice_loss(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Combined BCE + Dice loss for binary mask prediction.

    Args:
        pred_logits: Raw model output before sigmoid (B, 1, H, W).
        gt_mask: Ground truth binary mask (B, 1, H, W).
        bce_weight: Weight for BCE component.
        dice_weight: Weight for Dice component.

    Returns:
        Scalar loss tensor.
    """
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_mask)

    pred = torch.sigmoid(pred_logits)
    intersection = (pred * gt_mask).sum()
    dice_loss = 1 - (2.0 * intersection + 1.0) / (
        pred.sum() + gt_mask.sum() + 1.0
    )

    return bce_weight * bce_loss + dice_weight * dice_loss
