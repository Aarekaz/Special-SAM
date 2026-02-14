"""Segmentation metrics for camouflaged object detection evaluation.

Includes standard metrics (IoU, Dice, F1, Boundary) extracted from the
original notebook, plus 4 standard COD evaluation metrics:
  - S-alpha (Structure measure, Fan et al. ICCV 2017)
  - E-phi (Enhanced alignment measure, Fan et al. IJCAI 2018)
  - F-beta-w (Weighted F-measure, Margolin et al. CVPR 2014)
  - MAE (Mean Absolute Error)

Reference implementations can be verified against py-sod-metrics.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


class SegmentationMetrics:
    """Collection of segmentation metrics for comprehensive evaluation."""

    # ── Existing metrics (from notebook) ────────────────────────────

    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """Intersection over Union (Jaccard Index).

        Args:
            pred: Binary prediction (H, W).
            gt: Binary ground truth (H, W).

        Returns:
            IoU score in [0, 1].
        """
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
        """Dice coefficient (Sorensen-Dice).

        Formula: 2 * |A ∩ B| / (|A| + |B|)

        Args:
            pred: Binary prediction (H, W).
            gt: Binary ground truth (H, W).

        Returns:
            Dice score in [0, 1].
        """
        intersection = np.logical_and(pred, gt).sum()
        denominator = pred.sum() + gt.sum()
        if denominator == 0:
            return 1.0 if intersection == 0 else 0.0
        return float((2.0 * intersection) / denominator)

    @staticmethod
    def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Pixel-wise F1 score (equivalent to Dice for binary masks).

        Args:
            pred: Binary prediction (H, W).
            gt: Binary ground truth (H, W).

        Returns:
            F1 score in [0, 1].
        """
        tp = np.logical_and(pred == 1, gt == 1).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))

    @staticmethod
    def boundary_precision(
        pred: np.ndarray,
        gt: np.ndarray,
        threshold_px: int = 5,
    ) -> dict:
        """Boundary precision, recall, and F1.

        Extracts boundary pixels via Canny edge detection, then measures
        how many predicted boundary pixels fall within threshold_px of
        ground truth boundaries (and vice versa).

        Args:
            pred: Binary prediction (H, W).
            gt: Binary ground truth (H, W).
            threshold_px: Distance threshold in pixels.

        Returns:
            Dict with keys: precision, recall, f1.
        """
        pred_boundary = cv2.Canny((pred * 255).astype(np.uint8), 100, 200) > 0
        gt_boundary = cv2.Canny((gt * 255).astype(np.uint8), 100, 200) > 0

        if not pred_boundary.any() or not gt_boundary.any():
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        dist_pred_to_gt = distance_transform_edt(~gt_boundary)
        dist_gt_to_pred = distance_transform_edt(~pred_boundary)

        pred_coords = np.where(pred_boundary)
        precision = float(np.mean(dist_pred_to_gt[pred_coords] <= threshold_px))

        gt_coords = np.where(gt_boundary)
        recall = float(np.mean(dist_gt_to_pred[gt_coords] <= threshold_px))

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": float(f1)}

    # ── New standard COD metrics ────────────────────────────────────

    @staticmethod
    def mae(pred: np.ndarray, gt: np.ndarray) -> float:
        """Mean Absolute Error between prediction and ground truth.

        Args:
            pred: Prediction map (H, W), values in [0, 1].
            gt: Ground truth map (H, W), values in [0, 1].

        Returns:
            MAE score (lower is better).
        """
        pred_f = pred.astype(np.float64)
        gt_f = gt.astype(np.float64)
        return float(np.mean(np.abs(pred_f - gt_f)))

    @staticmethod
    def s_alpha(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
        """Structure measure (S-alpha) from Fan et al., ICCV 2017.

        Combines object-aware (S_o) and region-aware (S_r) structural
        similarity to evaluate how well the prediction captures the
        structure of the ground truth.

        S_alpha = alpha * S_o + (1 - alpha) * S_r

        Args:
            pred: Binary prediction (H, W), values in {0, 1}.
            gt: Binary ground truth (H, W), values in {0, 1}.
            alpha: Weight balancing S_o and S_r (default 0.5).

        Returns:
            S-alpha score in [0, 1] (higher is better).
        """
        pred_f = pred.astype(np.float64)
        gt_f = gt.astype(np.float64)

        gt_mean = gt_f.mean()

        # Handle edge cases
        if gt_mean == 0:
            # No foreground in GT
            return 1.0 - pred_f.mean()
        if gt_mean == 1:
            # All foreground in GT
            return pred_f.mean()

        # Object-aware structural similarity (S_o)
        s_o = SegmentationMetrics._s_object(pred_f, gt_f)

        # Region-aware structural similarity (S_r)
        s_r = SegmentationMetrics._s_region(pred_f, gt_f)

        return float(alpha * s_o + (1 - alpha) * s_r)

    @staticmethod
    def _s_object(pred: np.ndarray, gt: np.ndarray) -> float:
        """Object-aware structural similarity component."""
        # Foreground similarity
        fg_gt = gt.copy()
        fg_pred = pred.copy()
        fg_pred[gt == 0] = 0  # zero out false positives for fg eval

        o_fg = SegmentationMetrics._object_score(fg_pred, fg_gt)

        # Background similarity
        bg_gt = 1.0 - gt
        bg_pred = 1.0 - pred
        bg_pred[gt == 1] = 0  # zero out false negatives for bg eval

        o_bg = SegmentationMetrics._object_score(bg_pred, bg_gt)

        gt_mean = gt.mean()
        return float(gt_mean * o_fg + (1 - gt_mean) * o_bg)

    @staticmethod
    def _object_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Score a single foreground/background object region."""
        pred_mean = pred.mean()
        gt_mean = gt.mean()

        if gt_mean == 0 and pred_mean == 0:
            return 1.0
        if gt_mean == 0 or pred_mean == 0:
            return 0.0

        pred_norm = pred - pred_mean
        gt_norm = gt - gt_mean

        n = pred.size
        cross = np.sum(pred_norm * gt_norm) / n
        sigma_pred = np.sqrt(np.sum(pred_norm ** 2) / n)
        sigma_gt = np.sqrt(np.sum(gt_norm ** 2) / n)

        if sigma_pred == 0 or sigma_gt == 0:
            return 0.0

        # SSIM-like formula
        c1 = (0.01 * 1) ** 2
        c2 = (0.03 * 1) ** 2

        ssim = ((2 * pred_mean * gt_mean + c1) * (2 * cross + c2)) / (
            (pred_mean ** 2 + gt_mean ** 2 + c1) * (sigma_pred ** 2 + sigma_gt ** 2 + c2)
        )
        return max(0.0, float(ssim))

    @staticmethod
    def _s_region(pred: np.ndarray, gt: np.ndarray) -> float:
        """Region-aware structural similarity component.

        Divides the image into 4 quadrants based on the centroid of GT
        and computes SSIM for each region.
        """
        h, w = gt.shape
        ys, xs = np.where(gt > 0.5)
        if len(xs) == 0:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))

        # Clamp to valid range
        cx = max(1, min(cx, w - 1))
        cy = max(1, min(cy, h - 1))

        # 4 quadrants
        gt_tl, pred_tl = gt[:cy, :cx], pred[:cy, :cx]
        gt_tr, pred_tr = gt[:cy, cx:], pred[:cy, cx:]
        gt_bl, pred_bl = gt[cy:, :cx], pred[cy:, :cx]
        gt_br, pred_br = gt[cy:, cx:], pred[cy:, cx:]

        # Weighted sum by region area
        total = h * w
        score = 0.0
        for g, p in [(gt_tl, pred_tl), (gt_tr, pred_tr),
                      (gt_bl, pred_bl), (gt_br, pred_br)]:
            if g.size == 0:
                continue
            weight = g.size / total
            score += weight * SegmentationMetrics._ssim_region(p, g)

        return float(score)

    @staticmethod
    def _ssim_region(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute SSIM for a single region."""
        if pred.size == 0 or gt.size == 0:
            return 0.0

        pred_mean = pred.mean()
        gt_mean = gt.mean()

        n = pred.size
        pred_norm = pred - pred_mean
        gt_norm = gt - gt_mean

        sigma_pred_sq = np.sum(pred_norm ** 2) / n
        sigma_gt_sq = np.sum(gt_norm ** 2) / n
        sigma_cross = np.sum(pred_norm * gt_norm) / n

        c1 = (0.01) ** 2
        c2 = (0.03) ** 2

        ssim = ((2 * pred_mean * gt_mean + c1) * (2 * sigma_cross + c2)) / (
            (pred_mean ** 2 + gt_mean ** 2 + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
        )
        return max(0.0, float(ssim))

    @staticmethod
    def e_phi(pred: np.ndarray, gt: np.ndarray) -> float:
        """Enhanced alignment measure (E-phi) from Fan et al., IJCAI 2018.

        Combines pixel-level matching and image-level matching to capture
        both local accuracy and global statistics.

        Args:
            pred: Binary prediction (H, W), values in {0, 1}.
            gt: Binary ground truth (H, W), values in {0, 1}.

        Returns:
            E-phi score in [0, 1] (higher is better).
        """
        pred_f = pred.astype(np.float64)
        gt_f = gt.astype(np.float64)

        h, w = gt_f.shape
        n = h * w

        if n == 0:
            return 0.0

        # Global mean
        pred_mean = pred_f.mean()
        gt_mean = gt_f.mean()

        # Alignment matrix
        align = (
            2 * (pred_f - pred_mean) * (gt_f - gt_mean)
            + 2 * pred_mean * gt_mean
        ) / (
            (pred_f - pred_mean) ** 2
            + (gt_f - gt_mean) ** 2
            + 2 * pred_mean * gt_mean
            + 1e-8
        )

        # Enhanced alignment: (1 + align)^2 / 4
        enhanced = ((1 + align) ** 2) / 4.0

        return float(enhanced.mean())

    @staticmethod
    def f_beta_w(
        pred: np.ndarray,
        gt: np.ndarray,
        beta_sq: float = 1.0,
    ) -> float:
        """Weighted F-measure (F-beta-w) from Margolin et al., CVPR 2014.

        Uses distance-weighted precision and recall to penalize errors
        far from object boundaries more than errors near boundaries.

        Args:
            pred: Binary prediction (H, W), values in {0, 1}.
            gt: Binary ground truth (H, W), values in {0, 1}.
            beta_sq: Beta squared parameter (default 1.0 for F1-weighted).

        Returns:
            Weighted F-measure in [0, 1] (higher is better).
        """
        pred_f = pred.astype(np.float64)
        gt_f = gt.astype(np.float64)

        if gt_f.sum() == 0:
            return 0.0 if pred_f.sum() > 0 else 1.0

        # Distance transform of the complement of GT (distance to nearest GT pixel)
        # Eroded GT gives distances from GT boundary inward
        dist = distance_transform_edt(1 - gt_f)

        # Create weight map: areas close to GT boundary get lower weight
        # This follows Margolin et al.'s weighting scheme
        et = gt_f.copy()

        # Weighted true positives, false positives, false negatives
        # Weight by inverse distance (closer to boundary = less certain)
        weight_map = 1.0 + 5.0 * np.abs(
            distance_transform_edt(gt_f) - distance_transform_edt(1 - gt_f)
        )
        weight_map = weight_map / weight_map.max() if weight_map.max() > 0 else weight_map

        tp_w = np.sum(pred_f * gt_f * weight_map)
        fp_w = np.sum(pred_f * (1 - gt_f) * weight_map)
        fn_w = np.sum((1 - pred_f) * gt_f * weight_map)

        precision_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
        recall_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0

        if precision_w + recall_w == 0:
            return 0.0

        return float(
            (1 + beta_sq) * precision_w * recall_w
            / (beta_sq * precision_w + recall_w)
        )
