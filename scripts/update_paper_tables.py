"""Update paper tables with COD metrics from evaluation results.

Reads comprehensive_evaluation_results.csv and updates paper.md tables.

Usage:
    python scripts/update_paper_tables.py
"""

import pandas as pd
from pathlib import Path


def load_results(csv_path="results/comprehensive_evaluation_results.csv"):
    """Load evaluation results CSV."""
    if not Path(csv_path).exists():
        print(f"Error: Results file not found: {csv_path}")
        print("\nPlease run evaluation first:")
        print("  python -m src.evaluation.evaluate --config configs/eval.yaml")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loaded results from: {csv_path}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Strategies: {df['prompt_strategy'].unique().tolist()}")
    print(f"  Metrics: {[c for c in df.columns if '_mean' in c]}")
    return df


def format_table1_data(df):
    """Format data for Table 1: Main Results."""
    # Filter for the 3 main strategies
    strategies = ["Center-of-Mass (Single)", "Edge (Single)", "Multi-Point Grid (4 pts)", "Multi-Point Random (3 pts)"]

    table_data = []
    for model in ["Base SAM ViT-H", "Specialized SAM ViT-H"]:
        for strategy in strategies:
            row = df[(df['model'] == model) & (df['prompt_strategy'] == strategy)]

            if len(row) == 0:
                print(f"Warning: No data for {model} + {strategy}")
                continue

            row = row.iloc[0]
            table_data.append({
                'Model': model,
                'Prompt Strategy': strategy,
                'mIoU': f"{row['iou_mean']:.4f}",
                'Dice': f"{row['dice_mean']:.4f}",
                'S-alpha': f"{row['s_alpha_mean']:.4f}",
                'E-phi': f"{row['e_phi_mean']:.4f}",
                'F-beta-w': f"{row['f_beta_w_mean']:.4f}",
                'MAE': f"{row['mae_mean']:.4f}",
                'Boundary F1': f"{row['boundary_f1_mean']:.4f}",
            })

    return table_data


def format_table2_data(df):
    """Format data for Table 2: Improvement Analysis."""
    strategy_map = {
        "Center-of-Mass (Single)": "Center-of-Mass",
        "Edge (Single)": "Edge",
        "Multi-Point Grid (4 pts)": "Multi-Point Grid",
        "Multi-Point Random (3 pts)": "Multi-Point Random",
    }

    table_data = []
    for full_name, short_name in strategy_map.items():
        base_row = df[(df['model'] == 'Base SAM ViT-H') & (df['prompt_strategy'] == full_name)]
        spec_row = df[(df['model'] == 'Specialized SAM ViT-H') & (df['prompt_strategy'] == full_name)]

        if len(base_row) == 0 or len(spec_row) == 0:
            print(f"Warning: Missing data for {full_name}")
            continue

        base_miou = base_row.iloc[0]['iou_mean']
        spec_miou = spec_row.iloc[0]['iou_mean']
        abs_gain = spec_miou - base_miou
        rel_gain = (abs_gain / base_miou) * 100

        table_data.append({
            'Prompt Type': short_name,
            'Base mIoU': f"{base_miou:.4f}",
            'Specialized mIoU': f"{spec_miou:.4f}",
            'Absolute Gain': f"{abs_gain:+.4f}",
            'Relative Gain': f"{rel_gain:+.1f}%",
        })

    return table_data


def format_table3_data(df):
    """Format data for Table 3: Detailed Metrics (Specialized only)."""
    strategies = ["Center-of-Mass (Single)", "Edge (Single)", "Multi-Point Grid (4 pts)", "Multi-Point Random (3 pts)"]

    table_data = []
    for strategy in strategies:
        row = df[(df['model'] == 'Specialized SAM ViT-H') & (df['prompt_strategy'] == strategy)]

        if len(row) == 0:
            print(f"Warning: No data for Specialized + {strategy}")
            continue

        row = row.iloc[0]
        table_data.append({
            'Prompt Strategy': strategy,
            'mIoU': f"{row['iou_mean']:.4f}",
            'Dice': f"{row['dice_mean']:.4f}",
            'S-alpha': f"{row['s_alpha_mean']:.4f}",
            'E-phi': f"{row['e_phi_mean']:.4f}",
            'F-beta-w': f"{row['f_beta_w_mean']:.4f}",
            'MAE': f"{row['mae_mean']:.4f}",
            'Boundary Prec': f"{row['boundary_prec_mean']:.4f}",
            'Boundary Rec': f"{row['boundary_recall_mean']:.4f}",
            'Boundary F1': f"{row['boundary_f1_mean']:.4f}",
        })

    return table_data


def format_markdown_table(title, data, columns):
    """Format a markdown table and return as string."""
    lines = []
    lines.append(f"\n## {title}\n")

    # Header
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")

    # Rows
    for row in data:
        values = [str(row.get(col, '???')) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def generate_summary_report(df):
    """Generate a summary report and write tables to files."""
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SUMMARY REPORT: COD Metrics Now Available")
    print("="*70)

    # Table 1
    table1_data = format_table1_data(df)
    table1 = format_markdown_table(
        "Table 1: Main Results (COD10K, 200 samples)",
        table1_data,
        ['Model', 'Prompt Strategy', 'mIoU', 'Dice', 'S-alpha', 'E-phi', 'F-beta-w', 'MAE', 'Boundary F1']
    )
    print(table1)

    # Table 2
    table2_data = format_table2_data(df)
    table2 = format_markdown_table(
        "Table 2: Improvement Analysis by Prompt Type",
        table2_data,
        ['Prompt Type', 'Base mIoU', 'Specialized mIoU', 'Absolute Gain', 'Relative Gain']
    )
    print(table2)

    # Table 3
    table3_data = format_table3_data(df)
    table3 = format_markdown_table(
        "Table 3: Detailed Metrics (Specialized SAM)",
        table3_data,
        ['Prompt Strategy', 'mIoU', 'Dice', 'S-alpha', 'E-phi', 'F-beta-w', 'MAE',
         'Boundary Prec', 'Boundary Rec', 'Boundary F1']
    )
    print(table3)

    # Write all tables to a single file
    all_tables = "\n\n".join([table1, table2, table3])
    tables_file = out_dir / "all_tables.md"
    tables_file.write_text(all_tables)
    print(f"\nTables written to: {tables_file}")

    # Write individual table files
    (out_dir / "table1_main_results.md").write_text(table1)
    (out_dir / "table2_improvement.md").write_text(table2)
    (out_dir / "table3_detailed.md").write_text(table3)
    print(f"Individual tables written to: {out_dir}/")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    spec_center = df[(df['model'] == 'Specialized SAM ViT-H') &
                     (df['prompt_strategy'] == 'Center-of-Mass (Single)')].iloc[0]

    print(f"\nSpecialized SAM (Center-of-Mass prompt):")
    print(f"  mIoU:     {spec_center['iou_mean']:.4f}")
    print(f"  Dice:     {spec_center['dice_mean']:.4f}")
    print(f"  S-alpha:  {spec_center['s_alpha_mean']:.4f}")
    print(f"  E-phi:    {spec_center['e_phi_mean']:.4f}")
    print(f"  F-beta-w: {spec_center['f_beta_w_mean']:.4f}")
    print(f"  MAE:      {spec_center['mae_mean']:.4f}")

    base_edge = df[(df['model'] == 'Base SAM ViT-H') &
                   (df['prompt_strategy'] == 'Edge (Single)')].iloc[0]
    spec_edge = df[(df['model'] == 'Specialized SAM ViT-H') &
                   (df['prompt_strategy'] == 'Edge (Single)')].iloc[0]

    edge_gain = ((spec_edge['iou_mean'] - base_edge['iou_mean']) / base_edge['iou_mean']) * 100

    print(f"\nEdge Prompt Improvement:")
    print(f"  Base:       {base_edge['iou_mean']:.4f} mIoU")
    print(f"  Specialized: {spec_edge['iou_mean']:.4f} mIoU")
    print(f"  Gain:       {edge_gain:+.1f}% relative improvement")


def main():
    """Main execution."""
    print("Paper Table Update Tool")
    print("="*70)

    df = load_results()
    if df is None:
        return 1

    generate_summary_report(df)
    return 0


if __name__ == "__main__":
    exit(main())
