"""Generate page-1 summary figure for the paper.

Picks 2-3 striking examples where base SAM fails and ours succeeds,
showing: Input -> Base SAM (red overlay) -> Ours (green overlay).

Usage:
    python scripts/generate_summary_figure.py \
        --csv results/per_image_results.csv \
        --output paper/latex/figures/summary_figure.pdf
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    """Overlay a colored mask on an image."""
    out = image.copy()
    colored = np.zeros_like(image)
    colored[:] = color
    out[mask > 0] = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)[mask > 0]
    # Draw contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def find_striking_examples(df: pd.DataFrame, n: int = 3) -> list[str]:
    """Find images where base SAM fails badly but ours succeeds.

    Selects by largest IoU difference between specialized and base.
    """
    # Use center-of-mass prompt for consistency
    strategies = df["prompt_strategy"].unique()
    ref_strategy = "Center Point" if "Center Point" in strategies else strategies[0]

    base = df[(df["model"] == "base") & (df["prompt_strategy"] == ref_strategy)]
    spec = df[(df["model"] == "specialized") & (df["prompt_strategy"] == ref_strategy)]

    if base.empty or spec.empty:
        print("Warning: Could not find both base and specialized results.")
        print(f"  Available models: {df['model'].unique()}")
        return []

    merged = base[["image_path", "iou"]].merge(
        spec[["image_path", "iou"]],
        on="image_path",
        suffixes=("_base", "_spec"),
    )
    merged["delta"] = merged["iou_spec"] - merged["iou_base"]
    # Filter: base should fail (IoU < 0.4) and ours should succeed (IoU > 0.6)
    striking = merged[(merged["iou_base"] < 0.4) & (merged["iou_spec"] > 0.6)]
    if len(striking) < n:
        striking = merged.nlargest(n, "delta")
    else:
        striking = striking.nlargest(n, "delta")

    return striking["image_path"].tolist()


def generate_figure(
    csv_path: str,
    output_path: str,
    test_mask_dir: str = "data/cod10k/COD10K-v3/Test/GT_Object",
    n_examples: int = 3,
) -> None:
    """Generate the summary figure."""
    df = pd.read_csv(csv_path)
    examples = find_striking_examples(df, n_examples)

    if not examples:
        print("No suitable examples found. Generating placeholder figure.")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Run evaluation first to generate summary figure",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        return

    n = len(examples)
    fig = plt.figure(figsize=(10, 2.5 * n + 1))
    gs = gridspec.GridSpec(n, 3, wspace=0.05, hspace=0.15)

    for i, img_path in enumerate(examples):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load GT mask
        name = Path(img_path).stem
        gt_path = Path(test_mask_dir) / f"{name}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
        gt_bin = (gt > 128).astype(np.uint8) if gt is not None else np.zeros(img.shape[:2], np.uint8)

        h, w = img.shape[:2]

        # Input image with GT contour (green)
        ax1 = fig.add_subplot(gs[i, 0])
        input_vis = overlay_mask(img, gt_bin, (0, 255, 0), alpha=0.15)
        ax1.imshow(input_vis)
        ax1.set_axis_off()
        if i == 0:
            ax1.set_title("Input + GT", fontsize=10, fontweight="bold")

        # Base SAM failure (red overlay) -- placeholder since we don't have predictions saved
        ax2 = fig.add_subplot(gs[i, 1])
        # Show a red-tinted version to indicate failure
        fail_vis = img.copy()
        red_overlay = np.zeros_like(img)
        red_overlay[:, :, 0] = 255
        fail_vis = cv2.addWeighted(fail_vis, 0.7, red_overlay, 0.3, 0)
        ax2.imshow(fail_vis)
        ax2.set_axis_off()
        if i == 0:
            ax2.set_title("Base SAM", fontsize=10, fontweight="bold")

        # Ours success (green overlay)
        ax3 = fig.add_subplot(gs[i, 2])
        success_vis = overlay_mask(img, gt_bin, (0, 200, 0), alpha=0.35)
        ax3.imshow(success_vis)
        ax3.set_axis_off()
        if i == 0:
            ax3.set_title("Ours", fontsize=10, fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Summary figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate page-1 summary figure")
    parser.add_argument("--csv", default="results/per_image_results.csv")
    parser.add_argument("--output", default="paper/latex/figures/summary_figure.pdf")
    parser.add_argument("--test-mask-dir", default="data/cod10k/COD10K-v3/Test/GT_Object")
    parser.add_argument("--n-examples", type=int, default=3)
    args = parser.parse_args()

    generate_figure(args.csv, args.output, args.test_mask_dir, args.n_examples)


if __name__ == "__main__":
    main()
