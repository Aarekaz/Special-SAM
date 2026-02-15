"""Visualization utilities for SAM evaluation results.

Generates side-by-side comparisons, bar charts, comparison tables,
and multi-point prompt visualizations for the paper.

Usage:
    python -m src.visualization.visualize --config configs/eval.yaml --type all
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.cod10k import get_image_mask_pairs
from src.data.transforms import resize_image_mask
from src.evaluation.metrics import SegmentationMetrics
from src.evaluation.prompt_strategies import PROMPT_CONFIGS, PromptStrategy
from src.models.sam_loader import (
    get_device,
    get_predictor,
    load_sam,
    load_specialized_sam,
)


def compare_side_by_side(
    config: dict,
    num_samples: int = 10,
    output_dir: str = "results/visualizations",
) -> None:
    """Generate side-by-side comparison: Input, GT, Base SAM, Specialized SAM.

    Selects diverse samples by grouping test images by instance ID to
    avoid showing multiple frames of the same animal.

    Args:
        config: Evaluation config dict.
        output_dir: Directory to save output images.
    """
    device = get_device()
    target_size = config["evaluation"]["target_size"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both models
    base_model = load_sam(
        config["model"]["type"],
        config["model"]["base_checkpoint"],
        device,
    )
    base_model.eval()
    base_predictor = get_predictor(base_model)

    spec_model = load_specialized_sam(
        config["model"]["type"],
        config["model"]["base_checkpoint"],
        config["model"]["specialized_decoder"],
        device,
    )
    spec_model.eval()
    spec_predictor = get_predictor(spec_model)

    all_pairs = get_image_mask_pairs(
        config["data"]["test_img_dir"],
        config["data"]["test_mask_dir"],
    )

    # Group by instance to get diverse samples
    random.seed(None)
    instances = {}
    for img_path, mask_path in all_pairs:
        stem = Path(img_path).stem
        instance_id = "-".join(stem.split("-")[:-1])
        if instance_id not in instances:
            instances[instance_id] = []
        instances[instance_id].append((img_path, mask_path))

    unique_instances = list(instances.keys())
    random.shuffle(unique_instances)

    count = 0
    for inst_id in unique_instances:
        if count >= num_samples:
            break

        img_path, mask_path = random.choice(instances[inst_id])

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = img.shape[:2]
        img_resized, mask_resized = resize_image_mask(img, mask, target_size)

        ys, xs = np.where(mask_resized > 128)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        # Base prediction
        base_predictor.set_image(img_resized)
        base_masks, _, _ = base_predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        base_pred = cv2.resize(
            base_masks[0].astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Specialized prediction
        spec_predictor.set_image(img_resized)
        spec_masks, _, _ = spec_predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        spec_pred = cv2.resize(
            spec_masks[0].astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Visualization
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(img)
        axes[0].scatter(
            cx * w // target_size, cy * h // target_size,
            c="red", marker="*", s=250, edgecolors="white", linewidths=2,
        )
        axes[0].set_title(f"Input\n({Path(img_path).name})", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground Truth", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(base_pred, alpha=0.6, cmap="jet")
        axes[2].set_title("Base SAM", fontsize=14)
        axes[2].axis("off")

        axes[3].imshow(img)
        axes[3].imshow(spec_pred, alpha=0.6, cmap="spring")
        axes[3].set_title("Specialized SAM (Ours)", fontsize=14)
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"comparison_{count + 1}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        count += 1

    print(f"Saved {count} side-by-side comparisons to {output_dir}")


def create_comparison_table(
    results_csv: str,
    output_dir: str = "results",
) -> None:
    """Generate bar chart comparisons from evaluation results CSV.

    Args:
        results_csv: Path to combined evaluation results CSV.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    strategies = df["prompt_strategy"].unique()
    x = np.arange(len(strategies))
    width = 0.35

    metrics = [
        ("iou_mean", "Mean IoU"),
        ("dice_mean", "Dice Coefficient"),
        ("boundary_f1_mean", "Boundary F1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        base_vals = [
            df[(df["model"] == "Base SAM ViT-H") &
               (df["prompt_strategy"] == s)][metric].values[0]
            for s in strategies
        ]
        spec_vals = [
            df[(df["model"] == "Specialized SAM ViT-H") &
               (df["prompt_strategy"] == s)][metric].values[0]
            for s in strategies
        ]

        bars1 = ax.bar(
            x - width / 2, base_vals, width,
            label="Base SAM", color="skyblue", alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2, spec_vals, width,
            label="Specialized SAM", color="lightcoral", alpha=0.8,
        )

        ax.set_xlabel("Prompt Strategy", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} by Prompt Type", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, height,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=8,
                )

    plt.tight_layout()
    plot_path = output_dir / "evaluation_comparison.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart to {plot_path}")


def visualize_multipoint_prompts(
    config: dict,
    num_samples: int = 10,
    output_dir: str = "results/visualizations",
) -> None:
    """Visualize different prompt strategies on sample images.

    For each sample, shows the image with prompt points overlaid,
    the model prediction, and per-strategy metrics.

    Args:
        config: Evaluation config dict.
        num_samples: Number of test images to visualize.
        output_dir: Directory to save output images.
    """
    device = get_device()
    target_size = config["evaluation"]["target_size"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_specialized_sam(
        config["model"]["type"],
        config["model"]["base_checkpoint"],
        config["model"]["specialized_decoder"],
        device,
    )
    model.eval()

    test_pairs = get_image_mask_pairs(
        config["data"]["test_img_dir"],
        config["data"]["test_mask_dir"],
    )

    random.seed(42)
    sample_indices = random.sample(
        range(len(test_pairs)), min(num_samples, len(test_pairs))
    )

    strategies = ["center", "edge_single", "multi_grid"]
    strategy_colors = {"center": "red", "edge_single": "yellow", "multi_grid": "lime"}
    strategy_markers = {"center": "*", "edge_single": "o", "multi_grid": "^"}
    strategy_names = {
        "center": "Center-of-Mass",
        "edge_single": "Edge Points",
        "multi_grid": "Multi-Point Grid",
    }

    predictor = get_predictor(model)

    for sample_idx, idx in enumerate(sample_indices):
        img_path, mask_path = test_pairs[idx]

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, mask = resize_image_mask(
            image,
            cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE),
            target_size,
        )

        fig, axes = plt.subplots(len(strategies), 3, figsize=(18, 6 * len(strategies)))
        fig.suptitle(
            f"Sample {sample_idx + 1}: Multi-Point Prompt Strategies",
            fontsize=16, fontweight="bold",
        )

        for s_idx, strategy in enumerate(strategies):
            cfg = PROMPT_CONFIGS[strategy]
            point_coords, point_labels = cfg["strategy"](
                mask, num_points=cfg["num_points"]
            )

            if point_coords is None:
                for col in range(3):
                    axes[s_idx, col].axis("off")
                continue

            # Image with prompts
            axes[s_idx, 0].imshow(image)
            for pt in point_coords:
                axes[s_idx, 0].plot(
                    pt[0], pt[1],
                    marker=strategy_markers[strategy],
                    color=strategy_colors[strategy],
                    markersize=20, markeredgewidth=2, markeredgecolor="black",
                )
            axes[s_idx, 0].set_title(
                f"{strategy_names[strategy]} Prompts", fontsize=12
            )
            axes[s_idx, 0].axis("off")

            # Prediction
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            pred_mask = masks[0].astype(np.uint8) * 255

            axes[s_idx, 1].imshow(image)
            axes[s_idx, 1].imshow(pred_mask, alpha=0.5, cmap="Blues")
            axes[s_idx, 1].set_title("Prediction", fontsize=12)
            axes[s_idx, 1].axis("off")

            # Metrics
            iou = SegmentationMetrics.iou(pred_mask > 128, mask > 128)
            dice = SegmentationMetrics.dice_coefficient(pred_mask > 128, mask > 128)
            boundary = SegmentationMetrics.boundary_precision(
                pred_mask > 128, mask > 128, threshold_px=5
            )

            metrics_text = (
                f"IoU: {iou:.4f}\n"
                f"Dice: {dice:.4f}\n"
                f"Boundary F1: {boundary['f1']:.4f}"
            )
            axes[s_idx, 2].text(
                0.1, 0.5, metrics_text, fontsize=11,
                verticalalignment="center", family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            axes[s_idx, 2].set_title("Metrics", fontsize=12)
            axes[s_idx, 2].axis("off")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"multipoint_sample_{sample_idx + 1}.png",
            dpi=100, bbox_inches="tight",
        )
        plt.close()

    print(f"Saved {len(sample_indices)} multi-point visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for SAM evaluation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--type", type=str, default="all",
        choices=["side_by_side", "table", "multipoint", "all"],
        help="Type of visualization to generate",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of samples for side-by-side and multipoint visualizations",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = config["output"].get("visualization_dir", "results/visualizations")

    if args.type in ("side_by_side", "all"):
        compare_side_by_side(config, args.num_samples, output_dir)

    if args.type in ("table", "all"):
        results_csv = config["output"]["results_csv"]
        create_comparison_table(results_csv, str(Path(output_dir).parent))

    if args.type in ("multipoint", "all"):
        visualize_multipoint_prompts(config, args.num_samples, output_dir)


if __name__ == "__main__":
    main()
