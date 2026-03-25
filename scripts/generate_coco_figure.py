"""Generate qualitative comparison figure on regular COCO images.

Tests whether our fine-tuned decoder still works on everyday (non-camouflage)
images, or whether specialization has broken general segmentation.

Fixed center-of-image prompt is used (no GT required).

Usage:
    python -m scripts.generate_coco_figure
    python -m scripts.generate_coco_figure --num-examples 8 --output paper/latex/figures/coco_comparison.png
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.models.sam_loader import (
    get_device,
    get_predictor,
    load_sam,
    load_specialized_sam,
)


def run_prediction(predictor, image, point_coords, point_labels):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    # Pick highest scoring mask
    best = np.argmax(scores)
    return masks[best] > 0.5


def generate_coco_figure(img_dir: str, config: dict, output_path: str,
                          num_examples: int = 8, seed: int = 42) -> None:
    random.seed(seed)
    matplotlib.use("Agg")

    device = get_device()
    target_size = config["evaluation"]["target_size"]

    # Sample random images
    all_images = sorted(Path(img_dir).rglob("*.jpg"))
    selected = random.sample(all_images, min(num_examples, len(all_images)))
    print(f"Selected {len(selected)} COCO images")

    print("Loading base SAM...")
    base_model = load_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["base_checkpoint"],
        device=device,
    )
    base_model.eval()
    base_predictor = get_predictor(base_model)

    print("Loading specialized SAM...")
    spec_model = load_specialized_sam(
        model_type=config["model"]["type"],
        checkpoint=config["model"]["base_checkpoint"],
        decoder_path=config["model"]["specialized_decoder"],
        device=device,
    )
    spec_model.eval()
    spec_predictor = get_predictor(spec_model)

    n_rows = len(selected)
    col_width = 3.5
    row_height = 3.0
    fig, axes = plt.subplots(n_rows, 3, figsize=(col_width * 3, row_height * n_rows + 1.0), squeeze=False)

    column_titles = ["Original Image", "Base SAM", "Specialized SAM (Ours)"]
    mask_color = np.array([0, 200, 255], dtype=np.uint8)  # cyan for COCO
    mask_alpha = 0.4
    point_color = "red"

    for row_idx, img_path in enumerate(selected):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # Fixed center prompt
        cx, cy = new_w // 2, new_h // 2
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1])

        base_pred = run_prediction(base_predictor, img_resized, point_coords, point_labels)
        spec_pred = run_prediction(spec_predictor, img_resized, point_coords, point_labels)

        # Col 0: original + prompt point
        ax = axes[row_idx, 0]
        ax.imshow(img_resized)
        ax.scatter(cx, cy, c=point_color, s=80, marker="*", edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 1: base SAM
        ax = axes[row_idx, 1]
        overlay = img_resized.copy()
        overlay[base_pred] = ((1 - mask_alpha) * overlay[base_pred] + mask_alpha * mask_color).astype(np.uint8)
        ax.imshow(overlay)
        ax.scatter(cx, cy, c=point_color, s=80, marker="*", edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 2: specialized SAM
        ax = axes[row_idx, 2]
        overlay = img_resized.copy()
        overlay[spec_pred] = ((1 - mask_alpha) * overlay[spec_pred] + mask_alpha * mask_color).astype(np.uint8)
        ax.imshow(overlay)
        ax.scatter(cx, cy, c=point_color, s=80, marker="*", edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved to: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--img-dir", type=str, default="data/coco_val")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--output", type=str, default="paper/latex/figures/coco_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("COCO Regular Image Figure Generation")
    print("=" * 60)

    generate_coco_figure(args.img_dir, config, args.output, args.num_examples, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
