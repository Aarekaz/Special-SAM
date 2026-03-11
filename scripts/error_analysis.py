"""Error analysis: categorize failures and generate montage visualizations.

Consumes per-image evaluation results and produces:
1. Failure list (sorted by IoU ascending)
2. Automatic failure categorization (dark, tiny, thin, disjoint, general)
3. Montage images for each failure category
4. Summary table with counts and mean metrics per category

Usage:
    python scripts/error_analysis.py --csv results/per_image_results.csv
    python scripts/error_analysis.py --csv results/per_image_results.csv --top-k 400
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage


def categorize_failure(row: pd.Series, gt_mask: np.ndarray | None = None) -> str:
    """Assign a failure category based on image metadata and GT mask.

    Priority order (first match wins):
        1. Dark image: brightness < 80
        2. Tiny object: fg_pixel_ratio < 0.01
        3. Thin/elongated: GT bbox aspect ratio > 5
        4. Multiple disjoint: connected components > 1 (requires GT mask)
        5. General: everything else
    """
    if row["image_brightness"] < 80:
        return "dark"
    if row["fg_pixel_ratio"] < 0.01:
        return "tiny"
    if row["gt_aspect_ratio"] > 5.0:
        return "thin_elongated"
    if gt_mask is not None:
        labeled, n_components = ndimage.label(gt_mask)
        if n_components > 1:
            return "disjoint"
    return "general"


def load_gt_mask(image_path: str, test_mask_dir: str | None = None) -> np.ndarray | None:
    """Try to load the GT mask corresponding to an image path."""
    img_name = Path(image_path).stem
    if test_mask_dir:
        mask_path = Path(test_mask_dir) / f"{img_name}.png"
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            return (m > 128).astype(np.uint8) if m is not None else None
    # Try sibling GT_Object directory
    img_dir = Path(image_path).parent
    gt_dir = img_dir.parent / "GT_Object"
    mask_path = gt_dir / f"{img_name}.png"
    if mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (m > 128).astype(np.uint8) if m is not None else None
    return None


def create_montage(
    image_paths: list[str],
    gt_dir: str | None,
    output_path: str,
    grid_cols: int = 8,
    grid_rows: int = 5,
    cell_size: int = 128,
) -> None:
    """Create a montage showing (image, GT, prediction placeholder) for failures.

    Each cell shows the input image overlaid with GT contours.
    """
    n = min(len(image_paths), grid_cols * grid_rows)
    if n == 0:
        return

    montage_w = grid_cols * cell_size
    montage_h = grid_rows * cell_size
    montage = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 200  # gray bg

    for i in range(n):
        row = i // grid_cols
        col = i % grid_cols
        y0 = row * cell_size
        x0 = col * cell_size

        img = cv2.imread(image_paths[i])
        if img is None:
            continue
        img = cv2.resize(img, (cell_size, cell_size))

        # Overlay GT contours if available
        gt = load_gt_mask(image_paths[i], gt_dir)
        if gt is not None:
            gt_resized = cv2.resize(gt, (cell_size, cell_size),
                                    interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(
                gt_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        montage[y0:y0 + cell_size, x0:x0 + cell_size] = img

    cv2.imwrite(output_path, montage)


def run_error_analysis(
    csv_path: str,
    output_dir: str = "results/error_analysis",
    top_k: int = 400,
    test_mask_dir: str | None = None,
) -> None:
    """Run full error analysis pipeline."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} per-image records from {csv_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Failure list: sort by IoU ascending, take worst top_k
    # Use the "center" prompt strategy for consistent comparison,
    # falling back to first available strategy
    strategies = df["prompt_strategy"].unique()
    ref_strategy = "Center Point" if "Center Point" in strategies else strategies[0]

    # Get unique models
    models = df["model"].unique()
    print(f"Models: {list(models)}")
    print(f"Strategies: {list(strategies)}")

    # For failure categorization, use the specialized model's center prompt results
    ref_model = "specialized" if "specialized" in models else models[-1]
    ref_df = df[(df["prompt_strategy"] == ref_strategy) & (df["model"] == ref_model)].copy()
    ref_df = ref_df.sort_values("iou", ascending=True)

    failures = ref_df.head(top_k).copy()
    print(f"\nTop {len(failures)} failures (worst IoU):")
    print(f"  IoU range: {failures['iou'].min():.4f} - {failures['iou'].max():.4f}")

    # 2. Categorize failures
    categories = []
    for _, row in failures.iterrows():
        gt = load_gt_mask(row["image_path"], test_mask_dir)
        categories.append(categorize_failure(row, gt))
    failures["failure_category"] = categories

    # Save failure list
    failures.to_csv(str(out / "failure_list.csv"), index=False)
    print(f"  Saved failure list to {out / 'failure_list.csv'}")

    # 3. Summary table
    cat_counts = failures["failure_category"].value_counts()
    print("\nFailure category breakdown:")
    for cat, count in cat_counts.items():
        cat_df = failures[failures["failure_category"] == cat]
        print(f"  {cat}: {count} images, mean IoU={cat_df['iou'].mean():.4f}")

    # Summary across all models for these failure images
    failure_paths = set(failures["image_path"])
    summary_rows = []
    for model in models:
        model_df = df[(df["model"] == model) & (df["prompt_strategy"] == ref_strategy)]
        model_failures = model_df[model_df["image_path"].isin(failure_paths)]
        if len(model_failures) == 0:
            continue
        merged = model_failures.merge(
            failures[["image_path", "failure_category"]],
            on="image_path", how="left"
        )
        for cat in sorted(merged["failure_category"].unique()):
            cat_rows = merged[merged["failure_category"] == cat]
            summary_rows.append({
                "model": model,
                "failure_category": cat,
                "count": len(cat_rows),
                "iou_mean": cat_rows["iou"].mean(),
                "dice_mean": cat_rows["dice"].mean(),
                "s_alpha_mean": cat_rows["s_alpha"].mean(),
                "mae_mean": cat_rows["mae"].mean(),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(str(out / "failure_summary.csv"), index=False)
    print(f"\nSaved failure summary to {out / 'failure_summary.csv'}")

    # Print summary table
    if not summary_df.empty:
        pivot = summary_df.pivot_table(
            index="failure_category",
            columns="model",
            values="iou_mean",
            aggfunc="mean",
        )
        print("\nMean IoU by failure category and model:")
        print(pivot.to_string(float_format="%.4f"))

    # 4. Montage for each failure category
    for cat in sorted(failures["failure_category"].unique()):
        cat_failures = failures[failures["failure_category"] == cat]
        paths = cat_failures["image_path"].tolist()
        montage_path = str(out / f"montage_{cat}.png")
        create_montage(paths, test_mask_dir, montage_path)
        print(f"  Montage for '{cat}': {montage_path} ({len(paths)} images)")


def main():
    parser = argparse.ArgumentParser(description="Error analysis for COD evaluation")
    parser.add_argument(
        "--csv", type=str, default="results/per_image_results.csv",
        help="Path to per-image results CSV",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/error_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--top-k", type=int, default=400,
        help="Number of worst failures to analyze",
    )
    parser.add_argument(
        "--test-mask-dir", type=str, default=None,
        help="Path to test GT masks (for disjoint component detection)",
    )
    args = parser.parse_args()

    run_error_analysis(args.csv, args.output_dir, args.top_k, args.test_mask_dir)


if __name__ == "__main__":
    main()
