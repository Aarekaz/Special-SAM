"""Generate qualitative comparison figure grid for the Special-SAM paper.

Creates a figure with N rows (default 4) and 4 columns:
  Image + prompt | GT (green contour) | Base SAM (red, + GT contour) | Ours (blue, + GT contour)

Prediction columns overlay a thin GT boundary contour so the reader can
compare prediction vs. ground truth boundaries without scanning back and
forth. IoU scores appear as badges in the lower-right corner.

Examples are selected for diversity across easy/medium/hard difficulty
buckets based on base SAM IoU.

Usage:
    python -m scripts.generate_qualitative_figures --config configs/eval.yaml
    python -m scripts.generate_qualitative_figures --num-examples 4 --output paper/figures/qualitative_comparison.png
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.cod10k import get_image_mask_pairs
from src.data.transforms import resize_image_mask
from src.evaluation.prompt_strategies import PromptStrategy
from src.models.sam_loader import (
    get_device,
    get_predictor,
    load_sam,
    load_specialized_sam,
)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        pred: Predicted binary mask.
        gt: Ground truth binary mask.

    Returns:
        IoU score as a float in [0, 1].
    """
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def run_prediction(predictor, image: np.ndarray, point_coords: np.ndarray,
                   point_labels: np.ndarray) -> np.ndarray:
    """Run SAM prediction and return the binary mask at original resolution.

    Args:
        predictor: SamPredictor instance with the model loaded.
        image: RGB image (H, W, 3), already resized for SAM.
        point_coords: Point prompt coordinates, shape (N, 2).
        point_labels: Point prompt labels, shape (N,).

    Returns:
        Binary prediction mask (H, W) as bool array.
    """
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    return masks[0] > 0.5


def collect_candidates(config: dict) -> list[dict]:
    """Run base SAM on all test images and collect per-image IoU scores.

    This first pass identifies easy/medium/hard examples based on base SAM
    performance, so we can select diverse rows for the figure.

    Args:
        config: Evaluation configuration dict.

    Returns:
        List of dicts, each containing image path, mask path, base IoU,
        the resized image, resized mask, and prompt point coordinates.
    """
    device = get_device()
    target_size = config["evaluation"]["target_size"]

    print("Loading base SAM for candidate selection...")
    base_model = load_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["base_checkpoint"],
        device=device,
    )
    base_model.eval()
    base_predictor = get_predictor(base_model)

    test_pairs = get_image_mask_pairs(
        config["data"]["test_img_dir"],
        config["data"]["test_mask_dir"],
    )

    print(f"Scoring {len(test_pairs)} test images with base SAM...")

    candidates = []
    for idx, (img_path, mask_path) in enumerate(test_pairs):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        img_resized, mask_resized = resize_image_mask(img, mask, target_size)

        point_coords, point_labels = PromptStrategy.center_of_mass(mask_resized)
        if point_coords is None:
            continue

        try:
            pred = run_prediction(base_predictor, img_resized,
                                  point_coords, point_labels)
            gt_binary = mask_resized > 128
            iou = compute_iou(pred, gt_binary)
        except Exception as e:
            print(f"  Skipping sample {idx}: {e}")
            continue

        candidates.append({
            "img_path": img_path,
            "mask_path": mask_path,
            "base_iou": iou,
            "img_resized": img_resized,
            "mask_resized": mask_resized,
            "point_coords": point_coords,
            "point_labels": point_labels,
        })

        if (idx + 1) % 100 == 0:
            print(f"  Scored {idx + 1}/{len(test_pairs)}")

    print(f"Collected {len(candidates)} valid candidates")

    # Free base model memory before loading both models for final rendering
    del base_predictor, base_model
    torch.cuda.empty_cache()

    return candidates


def select_diverse_examples(candidates: list[dict],
                            num_examples: int = 12) -> list[dict]:
    """Select diverse examples spanning easy, medium, and hard difficulty.

    Splits candidates into three IoU-based buckets and picks evenly from
    each, sorted by IoU within each bucket for a clean visual gradient.

    Args:
        candidates: List of candidate dicts with 'base_iou' key.
        num_examples: Total number of examples to select (should be
            divisible by 3 for even splits; remainder goes to medium).

    Returns:
        Selected subset of candidates, ordered hard -> medium -> easy
        (i.e. ascending IoU) so the figure shows improvement gradient.
    """
    easy = [c for c in candidates if c["base_iou"] > 0.7]
    medium = [c for c in candidates if 0.3 <= c["base_iou"] <= 0.7]
    hard = [c for c in candidates if c["base_iou"] < 0.3]

    per_bucket = num_examples // 3
    remainder = num_examples - 3 * per_bucket

    print(f"Candidate pool: {len(easy)} easy, {len(medium)} medium, "
          f"{len(hard)} hard")

    # Sort each bucket by IoU and pick evenly spaced examples
    def pick(bucket, n):
        if len(bucket) == 0 or n == 0:
            return []
        bucket_sorted = sorted(bucket, key=lambda c: c["base_iou"])
        if len(bucket_sorted) <= n:
            return bucket_sorted
        indices = np.linspace(0, len(bucket_sorted) - 1, n, dtype=int)
        return [bucket_sorted[i] for i in indices]

    selected_hard = pick(hard, per_bucket)
    selected_medium = pick(medium, per_bucket + remainder)
    selected_easy = pick(easy, per_bucket)

    # Fill any shortfalls from other buckets
    total_selected = len(selected_hard) + len(selected_medium) + len(selected_easy)
    if total_selected < num_examples:
        all_remaining = sorted(candidates, key=lambda c: c["base_iou"])
        already_selected = set(id(s) for s in selected_hard + selected_medium + selected_easy)
        for c in all_remaining:
            if id(c) not in already_selected:
                selected_medium.append(c)
                total_selected += 1
                if total_selected >= num_examples:
                    break

    # Order: hard (low IoU) -> medium -> easy (high IoU), top to bottom
    selected = selected_hard + selected_medium + selected_easy
    print(f"Selected {len(selected)} examples for figure")
    return selected


def _mask_contours(binary_mask: np.ndarray) -> list:
    """Extract contours from a binary mask using OpenCV."""
    mask_u8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    return contours


def _overlay_with_contour(img: np.ndarray, binary_mask: np.ndarray,
                          fill_color: tuple, contour_color: tuple,
                          fill_alpha: float = 0.25,
                          contour_thickness: int = 2) -> np.ndarray:
    """Draw a semi-transparent fill and solid contour for a mask on an image.

    Args:
        img: RGB image (H, W, 3) uint8.
        binary_mask: Boolean mask (H, W).
        fill_color: RGB tuple for the fill (0-255).
        contour_color: BGR tuple for the contour (OpenCV convention).
        fill_alpha: Opacity of the fill.
        contour_thickness: Contour line width in pixels.
    """
    canvas = img.copy()
    fill_arr = np.array(fill_color, dtype=np.uint8)
    canvas[binary_mask] = (
        (1 - fill_alpha) * canvas[binary_mask] + fill_alpha * fill_arr
    ).astype(np.uint8)

    contours = _mask_contours(binary_mask)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.drawContours(canvas_bgr, contours, -1, contour_color, contour_thickness)
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def _draw_gt_contour(img: np.ndarray, gt_mask: np.ndarray,
                     color: tuple = (0, 200, 0),
                     thickness: int = 1) -> np.ndarray:
    """Draw a thin GT boundary contour on an image (no fill).

    Overlaid on prediction columns so the reader can compare prediction
    vs. GT boundaries without scanning back and forth.
    """
    contours = _mask_contours(gt_mask)
    canvas_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.drawContours(canvas_bgr, contours, -1, color, thickness)
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def _draw_iou_badge(ax, iou: float) -> None:
    """Draw an IoU score badge in the lower-right corner of matplotlib axes."""
    ax.text(
        0.95, 0.05, f"IoU {iou:.3f}",
        transform=ax.transAxes,
        fontsize=9, fontweight="bold", color="white",
        ha="right", va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="black", alpha=0.7, edgecolor="none",
        ),
    )


def generate_figure(selected: list[dict], config: dict,
                    output_path: str) -> None:
    """Generate the qualitative comparison figure for the paper.

    Layout per row (4 columns):
      Image + prompt | GT (green contour + fill) |
      Base SAM (red contour + fill, thin green GT contour, IoU badge) |
      Ours (blue contour + fill, thin green GT contour, IoU badge)

    The GT contour appears in prediction columns so the reader can
    compare boundaries directly.
    """
    device = get_device()
    target_size = config["evaluation"]["target_size"]
    n_rows = len(selected)

    matplotlib.use("Agg")

    print("Loading base SAM for figure generation...")
    base_model = load_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["base_checkpoint"],
        device=device,
    )
    base_model.eval()
    base_predictor = get_predictor(base_model)

    print("Loading specialized SAM for figure generation...")
    spec_model = load_specialized_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["base_checkpoint"],
        decoder_path=config["model"]["specialized_decoder"],
        device=device,
    )
    spec_model.eval()
    spec_predictor = get_predictor(spec_model)

    col_width = 3.8
    row_height = 3.2
    fig_width = col_width * 4
    fig_height = row_height * n_rows + 0.6

    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Color palette (fill = RGB, contour = BGR for OpenCV)
    GT_FILL = (0, 210, 0)
    GT_CONTOUR_BGR = (0, 200, 0)
    BASE_FILL = (220, 80, 80)
    BASE_CONTOUR_BGR = (60, 60, 220)
    OURS_FILL = (80, 140, 255)
    OURS_CONTOUR_BGR = (255, 140, 80)
    GT_THIN_BGR = (0, 200, 0)

    for row_idx, candidate in enumerate(selected):
        img = candidate["img_resized"]
        gt_mask = candidate["mask_resized"]
        point_coords = candidate["point_coords"]
        point_labels = candidate["point_labels"]

        gt_binary = gt_mask > 128
        base_pred = run_prediction(base_predictor, img, point_coords, point_labels)
        spec_pred = run_prediction(spec_predictor, img, point_coords, point_labels)

        base_iou_val = compute_iou(base_pred, gt_binary)
        spec_iou_val = compute_iou(spec_pred, gt_binary)

        # Column 0: Image + prompt point
        ax = axes[row_idx, 0]
        ax.imshow(img)
        for pt in point_coords:
            ax.plot(
                pt[0], pt[1], marker="*", markersize=14,
                color="#FF4444", markeredgecolor="white",
                markeredgewidth=1.2, zorder=5,
            )
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 1: GT (green contour + light fill)
        ax = axes[row_idx, 1]
        gt_vis = _overlay_with_contour(img, gt_binary,
                                       fill_color=GT_FILL,
                                       contour_color=GT_CONTOUR_BGR,
                                       fill_alpha=0.25,
                                       contour_thickness=2)
        ax.imshow(gt_vis)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 2: Base SAM (red prediction overlay + thin GT contour)
        ax = axes[row_idx, 2]
        base_vis = _overlay_with_contour(img, base_pred,
                                         fill_color=BASE_FILL,
                                         contour_color=BASE_CONTOUR_BGR,
                                         fill_alpha=0.25,
                                         contour_thickness=2)
        base_vis = _draw_gt_contour(base_vis, gt_binary,
                                    color=GT_THIN_BGR, thickness=1)
        ax.imshow(base_vis)
        for pt in point_coords:
            ax.plot(
                pt[0], pt[1], marker="*", markersize=10,
                color="#FF4444", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5,
            )
        _draw_iou_badge(ax, base_iou_val)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 3: Ours (blue prediction overlay + thin GT contour)
        ax = axes[row_idx, 3]
        spec_vis = _overlay_with_contour(img, spec_pred,
                                         fill_color=OURS_FILL,
                                         contour_color=OURS_CONTOUR_BGR,
                                         fill_alpha=0.25,
                                         contour_thickness=2)
        spec_vis = _draw_gt_contour(spec_vis, gt_binary,
                                    color=GT_THIN_BGR, thickness=1)
        ax.imshow(spec_vis)
        for pt in point_coords:
            ax.plot(
                pt[0], pt[1], marker="*", markersize=10,
                color="#FF4444", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5,
            )
        _draw_iou_badge(ax, spec_iou_val)
        ax.set_xticks([])
        ax.set_yticks([])

    # Column headers
    titles = ["Image", "Ground Truth", "Base SAM", "Ours"]
    for col_idx, title in enumerate(titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Remove all spines for a cleaner look
    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.subplots_adjust(wspace=0.03, hspace=0.06)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)

    print(f"\nFigure saved to: {output}")
    print(f"  Resolution: {fig_width * 300:.0f} x {fig_height * 300:.0f} px")


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative comparison figure for Special-SAM paper"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--num-examples", type=int, default=4,
        help="Number of example rows in the figure (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default="paper/figures/qualitative_comparison.png",
        help="Output path for the figure PNG",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("Qualitative Figure Generation")
    print("=" * 70)
    print(f"  Config:       {args.config}")
    print(f"  Num examples: {args.num_examples}")
    print(f"  Output:       {args.output}")
    print()

    # Step 1: Score all test images with base SAM to find easy/medium/hard
    candidates = collect_candidates(config)

    if len(candidates) < args.num_examples:
        print(f"Warning: Only {len(candidates)} valid candidates found, "
              f"requested {args.num_examples}")

    # Step 2: Select diverse examples
    selected = select_diverse_examples(candidates, args.num_examples)

    # Step 3: Generate the figure with both models
    generate_figure(selected, config, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
