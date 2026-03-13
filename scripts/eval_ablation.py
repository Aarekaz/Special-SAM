"""Ablation study evaluation: compare decoder checkpoints trained with
different loss function configurations.

Evaluates each decoder on the COD10K test set and produces a comparison
table summarizing key metrics across all ablation variants.

Usage:
    # Evaluate all ablation checkpoints and produce comparison table:
    python scripts/eval_ablation.py

    # Evaluate specific checkpoints:
    python scripts/eval_ablation.py \
        --decoders checkpoints/ablation/camo_decoder_bce_only.pth \
                   checkpoints/ablation/camo_decoder_dice_only.pth \
        --names bce_only dice_only

    # Quick evaluation on a subset:
    python scripts/eval_ablation.py --max-samples 500

    # Include the baseline (equal-weight) decoder for reference:
    python scripts/eval_ablation.py --include-baseline
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import evaluate_model


# Default ablation configurations
DEFAULT_ABLATION_CONFIGS = {
    "bce_only": {
        "decoder": "checkpoints/ablation/camo_decoder_bce_only.pth",
        "bce_weight": 1.0,
        "dice_weight": 0.0,
        "description": "BCE only (BCE=1.0, Dice=0.0)",
    },
    "dice_only": {
        "decoder": "checkpoints/ablation/camo_decoder_dice_only.pth",
        "bce_weight": 0.0,
        "dice_weight": 1.0,
        "description": "Dice only (BCE=0.0, Dice=1.0)",
    },
    "bce_heavy": {
        "decoder": "checkpoints/ablation/camo_decoder_bce_heavy.pth",
        "bce_weight": 0.7,
        "dice_weight": 0.3,
        "description": "BCE-heavy (BCE=0.7, Dice=0.3)",
    },
    "dice_heavy": {
        "decoder": "checkpoints/ablation/camo_decoder_dice_heavy.pth",
        "bce_weight": 0.3,
        "dice_weight": 0.7,
        "description": "Dice-heavy (BCE=0.3, Dice=0.7)",
    },
}

BASELINE_CONFIG = {
    "balanced": {
        "decoder": "checkpoints/camo_decoder_vith.pth",
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "description": "Balanced (BCE=0.5, Dice=0.5)",
    },
}


def build_eval_config(
    decoder_path: str,
    max_samples: int = 0,
    prompt_strategies: list[str] | None = None,
) -> dict:
    """Build an evaluation config dict for a given decoder checkpoint.

    Uses the same structure as eval.yaml but points to the specified
    decoder checkpoint.

    Args:
        decoder_path: Path to the decoder .pth file.
        max_samples: Max test samples (0 = all).
        prompt_strategies: Which prompt strategies to evaluate.

    Returns:
        Config dict compatible with evaluate_model().
    """
    if prompt_strategies is None:
        prompt_strategies = ["center", "multi_grid"]

    return {
        "model": {
            "type": "vit_h",
            "base_checkpoint": "weights/sam_vit_h_4b8939.pth",
            "specialized_decoder": decoder_path,
        },
        "data": {
            "test_img_dir": "data/cod10k/COD10K-v3/Test/Image",
            "test_mask_dir": "data/cod10k/COD10K-v3/Test/GT_Object",
        },
        "evaluation": {
            "max_samples": max_samples,
            "target_size": 1024,
            "boundary_threshold": 5,
            "prompt_strategies": prompt_strategies,
        },
    }


def evaluate_single_decoder(
    name: str,
    decoder_path: str,
    max_samples: int = 0,
    prompt_strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate a single decoder checkpoint.

    Args:
        name: Short name for this ablation variant (e.g. 'bce_only').
        decoder_path: Path to the decoder .pth file.
        max_samples: Max test samples (0 = all).
        prompt_strategies: Which prompt strategies to evaluate.

    Returns:
        DataFrame with evaluation results for this decoder.
    """
    decoder_file = Path(decoder_path)
    if not decoder_file.exists():
        print(f"WARNING: Decoder not found: {decoder_path} -- skipping '{name}'")
        return pd.DataFrame()

    config = build_eval_config(decoder_path, max_samples, prompt_strategies)

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  Decoder:  {decoder_path}")
    print(f"  Samples:  {max_samples if max_samples > 0 else 'all'}")
    print(f"{'='*60}")

    results_df, _, _ = evaluate_model("specialized", config, prompt_strategies)

    if results_df.empty:
        print(f"WARNING: No results for '{name}'")
        return pd.DataFrame()

    results_df["ablation_name"] = name
    results_df["decoder_path"] = decoder_path
    return results_df


def format_comparison_table(combined_df: pd.DataFrame, configs: dict) -> str:
    """Format a readable comparison table from combined results.

    Args:
        combined_df: Combined DataFrame from all ablation evaluations.
        configs: Dict mapping ablation names to their config info.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("ABLATION STUDY: Loss Function Configuration Comparison")
    lines.append("=" * 90)

    # Get unique prompt strategies
    strategies = combined_df["prompt_strategy"].unique()

    for strategy in strategies:
        lines.append(f"\nPrompt Strategy: {strategy}")
        lines.append("-" * 90)

        # Table header
        header = (
            f"{'Variant':<20} {'BCE':>5} {'Dice':>5} "
            f"{'mIoU':>7} {'Dice':>7} {'S-alpha':>8} {'E-phi':>7} "
            f"{'F-beta-w':>8} {'MAE':>7} {'BndF1':>7}"
        )
        lines.append(header)
        lines.append("-" * 90)

        strat_df = combined_df[combined_df["prompt_strategy"] == strategy]

        for _, row in strat_df.iterrows():
            name = row["ablation_name"]
            cfg = configs.get(name, {})
            bce_w = cfg.get("bce_weight", "?")
            dice_w = cfg.get("dice_weight", "?")

            line = (
                f"{name:<20} {bce_w:>5} {dice_w:>5} "
                f"{row['iou_mean']:>7.4f} {row['dice_mean']:>7.4f} "
                f"{row['s_alpha_mean']:>8.4f} {row['e_phi_mean']:>7.4f} "
                f"{row['f_beta_w_mean']:>8.4f} {row['mae_mean']:>7.4f} "
                f"{row.get('boundary_f1_mean', 0):>7.4f}"
            )
            lines.append(line)

        lines.append("")

    # Determine best variant per metric (using first strategy for summary)
    lines.append("=" * 90)
    lines.append("BEST VARIANTS (first prompt strategy)")
    lines.append("-" * 90)

    first_strat = strategies[0]
    strat_df = combined_df[combined_df["prompt_strategy"] == first_strat]

    metrics = {
        "mIoU": ("iou_mean", True),        # higher is better
        "Dice": ("dice_mean", True),
        "S-alpha": ("s_alpha_mean", True),
        "E-phi": ("e_phi_mean", True),
        "F-beta-w": ("f_beta_w_mean", True),
        "MAE": ("mae_mean", False),         # lower is better
    }

    for metric_name, (col, higher_better) in metrics.items():
        if col not in strat_df.columns:
            continue
        if higher_better:
            best_idx = strat_df[col].idxmax()
        else:
            best_idx = strat_df[col].idxmin()
        best_row = strat_df.loc[best_idx]
        lines.append(
            f"  {metric_name:<12}: {best_row['ablation_name']:<20} "
            f"({best_row[col]:.4f})"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ablation study decoder checkpoints"
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=None,
        help="Paths to decoder checkpoints to evaluate. "
             "If not specified, uses all default ablation checkpoints.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Short names for each decoder (must match --decoders count). "
             "If not specified, uses default ablation names.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max test samples per evaluation (0 = all). Default: 500 for speed.",
    )
    parser.add_argument(
        "--prompt-strategies",
        nargs="+",
        default=["center", "multi_grid"],
        help="Prompt strategies to evaluate. Default: center multi_grid.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include the baseline balanced (0.5/0.5) decoder for comparison.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/ablation/ablation_comparison.csv",
        help="Path to save the combined results CSV.",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="Path to eval.yaml to override default data/model paths.",
    )
    args = parser.parse_args()

    # Load eval config overrides if provided
    if args.eval_config:
        with open(args.eval_config) as f:
            eval_override = yaml.safe_load(f)
        print(f"Loaded eval config overrides from: {args.eval_config}")
    else:
        eval_override = None

    # Build the list of decoders to evaluate
    if args.decoders is not None:
        # User specified explicit decoder paths
        if args.names is not None:
            if len(args.names) != len(args.decoders):
                print("ERROR: --names count must match --decoders count")
                sys.exit(1)
            names = args.names
        else:
            # Derive names from filenames
            names = [Path(d).stem.replace("camo_decoder_", "") for d in args.decoders]

        configs = {}
        decoders_to_eval = []
        for name, decoder in zip(names, args.decoders):
            configs[name] = {
                "decoder": decoder,
                "bce_weight": "?",
                "dice_weight": "?",
                "description": name,
            }
            decoders_to_eval.append((name, decoder))
    else:
        # Use default ablation configurations
        configs = dict(DEFAULT_ABLATION_CONFIGS)
        decoders_to_eval = [
            (name, cfg["decoder"]) for name, cfg in configs.items()
        ]

    # Optionally include baseline
    if args.include_baseline:
        configs.update(BASELINE_CONFIG)
        decoders_to_eval.append(
            ("balanced", BASELINE_CONFIG["balanced"]["decoder"])
        )

    # Evaluate each decoder
    all_results = []

    for name, decoder_path in decoders_to_eval:
        result_df = evaluate_single_decoder(
            name=name,
            decoder_path=decoder_path,
            max_samples=args.max_samples,
            prompt_strategies=args.prompt_strategies,
        )
        if not result_df.empty:
            all_results.append(result_df)

    if not all_results:
        print("\nERROR: No decoder checkpoints were evaluated successfully.")
        print("Make sure training has been run first (see scripts/ablation.sh).")
        sys.exit(1)

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)

    # Save CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(str(output_path), index=False)
    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    table = format_comparison_table(combined, configs)
    print(table)

    # Also save the table as a text file
    table_path = output_path.parent / "ablation_comparison_table.txt"
    table_path.write_text(table)
    print(f"Comparison table saved to: {table_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
