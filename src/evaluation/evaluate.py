"""Comprehensive evaluation comparing base vs specialized SAM.

Evaluates both models across multiple prompt strategies and metrics,
producing a combined results DataFrame and CSV output.

Usage:
    python -m src.evaluation.evaluate --config configs/eval.yaml
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.data.cod10k import get_image_mask_pairs
from src.data.transforms import resize_image_mask
from src.evaluation.metrics import SegmentationMetrics
from src.evaluation.prompt_strategies import PROMPT_CONFIGS
from src.models.sam_loader import (
    get_device,
    get_predictor,
    load_llpm_sam,
    load_sam,
    load_specialized_sam,
)


def extract_category(filepath: str) -> str:
    """Extract super-category from COD10K filename.

    Filenames follow: COD10K-{CAM/NonCAM}-{super}-{Category}-{sub}-{Species}-{id}.jpg
    E.g. COD10K-CAM-1-Aquatic-1-BatFish-2.jpg -> Aquatic
    """
    name = Path(filepath).stem
    match = re.match(r"COD10K-\w+-\d+-(\w+)-", name)
    if match:
        return match.group(1)
    return "Unknown"


def predict_with_llpm(
    image: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    model: "torch.nn.Module",
    llpm: "torch.nn.Module",
    device: "torch.device",
) -> np.ndarray:
    """Run prediction through LLPM -> encoder -> decoder, bypassing SamPredictor.

    SamPredictor.set_image() applies its own normalization, so we bypass it
    and manually run the full pipeline.

    Args:
        image: RGB image (H, W, 3) uint8.
        point_coords: Point prompts (N, 2) as numpy array.
        point_labels: Point labels (N,) as numpy array.
        model: SAM model.
        llpm: LLPM module.
        device: Torch device.

    Returns:
        Binary mask (H, W) as numpy bool array.
    """
    import torch
    import torch.nn.functional as F

    h, w = image.shape[:2]

    # Image to tensor: (1, 3, H, W) float32 [0, 255]
    image_t = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # LLPM preprocessing
        enhanced = llpm(image_t).clamp(0, 255)

        # SAM normalization
        pixel_mean = model.pixel_mean.unsqueeze(0)
        pixel_std = model.pixel_std.unsqueeze(0)
        normalized = (enhanced - pixel_mean) / pixel_std

        # Encoder
        image_embedding = model.image_encoder(normalized)

        # Prompt encoder
        points_t = torch.tensor(
            point_coords, dtype=torch.float32, device=device
        ).unsqueeze(0)
        labels_t = torch.tensor(
            point_labels, dtype=torch.int64, device=device
        ).unsqueeze(0)

        sparse, dense = model.prompt_encoder(
            points=(points_t, labels_t), boxes=None, masks=None
        )

        # Decoder
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        # Upsample to original resolution
        pred = F.interpolate(
            low_res_masks,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        mask = (pred.squeeze().sigmoid() > 0.5).cpu().numpy()

    return mask


def evaluate_model(
    model_type: str,
    config: dict,
    prompt_strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate one model across all prompt strategies and metrics.

    Args:
        model_type: 'base' or 'specialized'.
        config: Evaluation configuration dict.
        prompt_strategies: List of strategy keys. Uses config default if None.

    Returns:
        DataFrame with one row per prompt strategy containing all metrics.
    """
    device = get_device()
    target_size = config["evaluation"]["target_size"]
    boundary_threshold = config["evaluation"]["boundary_threshold"]
    max_samples = config["evaluation"]["max_samples"]

    if prompt_strategies is None:
        prompt_strategies = config["evaluation"]["prompt_strategies"]

    # Load model
    llpm = None
    if model_type == "llpm":
        model, llpm = load_llpm_sam(
            model_type=config["model"]["type"],
            checkpoint=config["model"]["base_checkpoint"],
            decoder_path=config["model"].get("llpm_decoder"),
            llpm_path=config["model"].get("llpm_path"),
            llpm_config=config.get("llpm"),
            device=device,
        )
        model.eval()
        llpm.eval()
        predictor = None
    elif model_type == "specialized":
        model = load_specialized_sam(
            model_type=config["model"]["type"],
            checkpoint=config["model"]["base_checkpoint"],
            decoder_path=config["model"]["specialized_decoder"],
            device=device,
        )
        model.eval()
        predictor = get_predictor(model)
    else:
        model = load_sam(
            model_type=config["model"]["type"],
            checkpoint=config["model"]["base_checkpoint"],
            device=device,
        )
        model.eval()
        predictor = get_predictor(model)

    # Get test data
    test_pairs = get_image_mask_pairs(
        config["data"]["test_img_dir"],
        config["data"]["test_mask_dir"],
        max_samples=max_samples,
    )

    print(f"\nEvaluating {model_type.upper()} model on {len(test_pairs)} samples")

    results = []
    cat_results = []

    for strategy_key in prompt_strategies:
        if strategy_key not in PROMPT_CONFIGS:
            print(f"  Unknown strategy: {strategy_key}, skipping")
            continue

        cfg = PROMPT_CONFIGS[strategy_key]
        strategy_func = cfg["strategy"]
        num_points = cfg["num_points"]

        print(f"  Testing: {cfg['name']}")

        ious, dices, f1s = [], [], []
        boundary_precs, boundary_recalls, boundary_f1s = [], [], []
        s_alphas, e_phis, f_beta_ws, maes = [], [], [], []
        categories = []

        for idx, (img_path, mask_path) in enumerate(test_pairs):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h_orig, w_orig = img.shape[:2]
            img_resized, mask_resized = resize_image_mask(
                img, mask, target_size
            )

            point_coords, point_labels = strategy_func(
                mask_resized, num_points=num_points
            )
            if point_coords is None:
                continue

            try:
                if model_type == "llpm":
                    pred_binary = predict_with_llpm(
                        img_resized, point_coords, point_labels,
                        model, llpm, device,
                    )
                    # predict_with_llpm returns at input resolution (1024x1024),
                    # resize to original for metric computation
                    pred_resized = cv2.resize(
                        pred_binary.astype(np.uint8),
                        (w_orig, h_orig),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    predictor.set_image(img_resized)
                    masks, _, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False,
                    )
                    pred_resized = cv2.resize(
                        masks[0].astype(np.uint8),
                        (w_orig, h_orig),
                        interpolation=cv2.INTER_NEAREST,
                    )

                pred_binary = pred_resized > 0.5
                gt_binary = mask > 128

                # Compute all metrics
                ious.append(SegmentationMetrics.iou(pred_binary, gt_binary))
                dices.append(SegmentationMetrics.dice_coefficient(pred_binary, gt_binary))
                f1s.append(SegmentationMetrics.f1_score(pred_binary, gt_binary))

                boundary = SegmentationMetrics.boundary_precision(
                    pred_binary, gt_binary, boundary_threshold
                )
                boundary_precs.append(boundary["precision"])
                boundary_recalls.append(boundary["recall"])
                boundary_f1s.append(boundary["f1"])

                # New COD metrics
                s_alphas.append(SegmentationMetrics.s_alpha(pred_binary, gt_binary))
                e_phis.append(SegmentationMetrics.e_phi(pred_binary, gt_binary))
                f_beta_ws.append(SegmentationMetrics.f_beta_w(pred_binary, gt_binary))
                maes.append(SegmentationMetrics.mae(
                    pred_binary.astype(float), gt_binary.astype(float)
                ))
                categories.append(extract_category(img_path))

            except Exception as e:
                print(f"    Error on sample {idx}: {e}")
                continue

            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(test_pairs)}")

        if len(ious) > 0:
            result = {
                "prompt_strategy": cfg["name"],
                "iou_mean": np.mean(ious),
                "iou_std": np.std(ious),
                "dice_mean": np.mean(dices),
                "dice_std": np.std(dices),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
                "boundary_prec_mean": np.mean(boundary_precs),
                "boundary_prec_std": np.std(boundary_precs),
                "boundary_recall_mean": np.mean(boundary_recalls),
                "boundary_f1_mean": np.mean(boundary_f1s),
                "s_alpha_mean": np.mean(s_alphas),
                "e_phi_mean": np.mean(e_phis),
                "f_beta_w_mean": np.mean(f_beta_ws),
                "mae_mean": np.mean(maes),
                "num_samples": len(ious),
            }
            results.append(result)
            print(f"    mIoU={result['iou_mean']:.4f}, "
                  f"S-alpha={result['s_alpha_mean']:.4f}, "
                  f"MAE={result['mae_mean']:.4f}")

            # Per-category breakdown
            if categories:
                cat_arr = np.array(categories)
                iou_arr = np.array(ious)
                for cat in sorted(set(categories)):
                    mask = cat_arr == cat
                    cat_results.append({
                        "prompt_strategy": cfg["name"],
                        "category": cat,
                        "iou_mean": np.mean(iou_arr[mask]),
                        "dice_mean": np.mean(np.array(dices)[mask]),
                        "s_alpha_mean": np.mean(np.array(s_alphas)[mask]),
                        "mae_mean": np.mean(np.array(maes)[mask]),
                        "num_samples": int(mask.sum()),
                    })

    return pd.DataFrame(results), pd.DataFrame(cat_results)


def run_comparison(config: dict) -> pd.DataFrame:
    """Run evaluation on both base and specialized models, combine results.

    Args:
        config: Evaluation configuration dict.

    Returns:
        Combined DataFrame with results from both models.
    """
    base_results, base_cat = evaluate_model("base", config)
    base_results["model"] = "Base SAM ViT-H"
    base_cat["model"] = "Base SAM ViT-H"

    spec_results, spec_cat = evaluate_model("specialized", config)
    spec_results["model"] = "Specialized SAM ViT-H"
    spec_cat["model"] = "Specialized SAM ViT-H"

    all_results = [base_results, spec_results]
    all_cat = [base_cat, spec_cat]

    # Include LLPM evaluation if configured
    if config["model"].get("llpm_path"):
        llpm_results, llpm_cat = evaluate_model("llpm", config)
        llpm_results["model"] = "LLPM + SAM ViT-H"
        llpm_cat["model"] = "LLPM + SAM ViT-H"
        all_results.append(llpm_results)
        all_cat.append(llpm_cat)

    combined = pd.concat(all_results, ignore_index=True)

    # Save main CSV
    output_path = Path(config["output"]["results_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(str(output_path), index=False)
    print(f"\nResults saved to {output_path}")

    # Save per-category CSV
    combined_cat = pd.concat(all_cat, ignore_index=True)
    cat_path = output_path.parent / "per_category_results.csv"
    combined_cat.to_csv(str(cat_path), index=False)
    print(f"Per-category results saved to {cat_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base vs specialized SAM on COD10K"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to evaluation config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_comparison(config)


if __name__ == "__main__":
    main()
