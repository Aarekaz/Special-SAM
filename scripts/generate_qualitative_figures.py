"""Generate qualitative comparison figure grid for the Special-SAM paper.

Creates a figure with N rows (default 12) and 4 columns:
  Original Image | Ground Truth Mask | Base SAM Prediction | Specialized SAM Prediction

Examples are selected for diversity: 4 "easy" (base SAM IoU > 0.7),
4 "medium" (IoU 0.3-0.7), and 4 "hard" (IoU < 0.3) cases.

The center-of-mass prompt point is overlaid on each image.

Usage:
    python -m scripts.generate_qualitative_figures --config configs/eval.yaml
    python -m scripts.generate_qualitative_figures --num-examples 12 --output paper/figures/qualitative_comparison.png
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


def generate_figure(selected: list[dict], config: dict,
                    output_path: str) -> None:
    """Generate the final qualitative comparison figure.

    Loads both base and specialized SAM, runs predictions on each
    selected example, and assembles the 4-column figure grid.

    Args:
        selected: List of selected candidate dicts.
        config: Evaluation configuration dict.
        output_path: Path to save the output PNG.
    """
    device = get_device()
    target_size = config["evaluation"]["target_size"]
    n_rows = len(selected)

    # Use non-interactive backend for HPC compatibility
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

    # Figure sizing: 4 columns, n_rows rows
    col_width = 3.5  # inches per column
    row_height = 3.0  # inches per row
    fig_width = col_width * 4
    fig_height = row_height * n_rows + 1.0  # extra space for column titles

    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    column_titles = [
        "Original Image",
        "Ground Truth",
        "Base SAM",
        "Specialized SAM",
    ]

    # Style: green overlay for masks, red point for prompt
    mask_color = np.array([0, 255, 0], dtype=np.uint8)  # green
    mask_alpha = 0.4
    point_color = "red"
    point_size = 80
    point_edge_color = "white"

    for row_idx, candidate in enumerate(selected):
        img = candidate["img_resized"]
        gt_mask = candidate["mask_resized"]
        point_coords = candidate["point_coords"]
        point_labels = candidate["point_labels"]
        base_iou = candidate["base_iou"]

        gt_binary = gt_mask > 128

        # Run predictions
        base_pred = run_prediction(base_predictor, img, point_coords, point_labels)
        spec_pred = run_prediction(spec_predictor, img, point_coords, point_labels)

        base_iou_val = compute_iou(base_pred, gt_binary)
        spec_iou_val = compute_iou(spec_pred, gt_binary)

        # Column 0: Original image with prompt point
        ax = axes[row_idx, 0]
        ax.imshow(img)
        for pt in point_coords:
            ax.scatter(
                pt[0], pt[1],
                c=point_color, s=point_size, marker="*",
                edgecolors=point_edge_color, linewidths=0.8,
                zorder=5,
            )
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 1: Ground truth mask overlay
        ax = axes[row_idx, 1]
        overlay = img.copy()
        overlay[gt_binary] = (
            (1 - mask_alpha) * overlay[gt_binary]
            + mask_alpha * mask_color
        ).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 2: Base SAM prediction overlay
        ax = axes[row_idx, 2]
        overlay = img.copy()
        overlay[base_pred] = (
            (1 - mask_alpha) * overlay[base_pred]
            + mask_alpha * mask_color
        ).astype(np.uint8)
        ax.imshow(overlay)
        for pt in point_coords:
            ax.scatter(
                pt[0], pt[1],
                c=point_color, s=point_size, marker="*",
                edgecolors=point_edge_color, linewidths=0.8,
                zorder=5,
            )
        ax.set_xlabel(f"IoU: {base_iou_val:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 3: Specialized SAM prediction overlay
        ax = axes[row_idx, 3]
        overlay = img.copy()
        overlay[spec_pred] = (
            (1 - mask_alpha) * overlay[spec_pred]
            + mask_alpha * mask_color
        ).astype(np.uint8)
        ax.imshow(overlay)
        for pt in point_coords:
            ax.scatter(
                pt[0], pt[1],
                c=point_color, s=point_size, marker="*",
                edgecolors=point_edge_color, linewidths=0.8,
                zorder=5,
            )
        ax.set_xlabel(f"IoU: {spec_iou_val:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row label indicating difficulty
        if row_idx < len(selected) // 3:
            difficulty = "Hard"
        elif row_idx < 2 * (len(selected) // 3):
            difficulty = "Medium"
        else:
            difficulty = "Easy"
        axes[row_idx, 0].set_ylabel(
            difficulty, fontsize=10, fontweight="bold", rotation=90,
            labelpad=10,
        )

    # Column titles
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()

    # Save high-resolution PNG
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=300, bbox_inches="tight", facecolor="white")
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
        "--num-examples", type=int, default=12,
        help="Number of example rows in the figure (default: 12)",
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
