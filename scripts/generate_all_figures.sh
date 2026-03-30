#!/bin/bash
#SBATCH --job-name=sam-allfigs
#SBATCH --output=logs/allfigs_%j.out
#SBATCH --error=logs/allfigs_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
mkdir -p paper/VisCon/figures/variants

echo "=========================================="
echo "Generating ALL qualitative figure variants"
echo "=========================================="
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ─────────────────────────────────────────────
# 1. BEST mode (largest IoU improvement)
# ─────────────────────────────────────────────

echo ">>> [1/6] Best mode, 4 rows"
python -m scripts.generate_qualitative_figures \
  --config configs/eval.yaml \
  --num-examples 4 \
  --best \
  --output paper/VisCon/figures/variants/best_4rows.png

echo ">>> [2/6] Best mode, 6 rows"
python -m scripts.generate_qualitative_figures \
  --config configs/eval.yaml \
  --num-examples 6 \
  --best \
  --output paper/VisCon/figures/variants/best_6rows.png

echo ">>> [3/6] Best mode, 1 row (teaser candidate)"
python -m scripts.generate_qualitative_figures \
  --config configs/eval.yaml \
  --num-examples 1 \
  --best \
  --output paper/VisCon/figures/variants/best_1row_teaser.png

# ─────────────────────────────────────────────
# 2. DIVERSE mode (hard/medium/easy mix)
# ─────────────────────────────────────────────

echo ">>> [4/6] Diverse mode, 4 rows"
python -m scripts.generate_qualitative_figures \
  --config configs/eval.yaml \
  --num-examples 4 \
  --diverse \
  --output paper/VisCon/figures/variants/diverse_4rows.png

echo ">>> [5/6] Diverse mode, 6 rows"
python -m scripts.generate_qualitative_figures \
  --config configs/eval.yaml \
  --num-examples 6 \
  --diverse \
  --output paper/VisCon/figures/variants/diverse_6rows.png

echo ">>> [6/6] Diverse mode, 8 rows (full spread)"
python -m scripts.generate_qualitative_figures \
  --config configs/eval.yaml \
  --num-examples 8 \
  --diverse \
  --output paper/VisCon/figures/variants/diverse_8rows.png

# ─────────────────────────────────────────────
# 3. Auto-crop teasers from each 4-row variant
# ─────────────────────────────────────────────

echo ">>> Cropping individual rows as teaser candidates..."
python -c "
from PIL import Image
from pathlib import Path

variants_dir = Path('paper/VisCon/figures/variants')

for src_name in ['best_4rows.png', 'diverse_4rows.png']:
    src_path = variants_dir / src_name
    if not src_path.exists():
        print(f'  Skipping {src_name} (not found)')
        continue

    img = Image.open(src_path)
    w, h = img.size
    n_rows = 4
    row_h = h // n_rows

    prefix = src_name.replace('_4rows.png', '')
    for i in range(n_rows):
        top = i * row_h
        bottom = (i + 1) * row_h
        row_img = img.crop((0, top, w, bottom))
        out_path = variants_dir / f'{prefix}_row{i}_teaser.png'
        row_img.save(out_path)
        print(f'  Saved {out_path.name} ({row_img.size[0]}x{row_img.size[1]})')

print('Done cropping teasers.')
"

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

echo ""
echo "=========================================="
echo "All variants generated:"
echo "=========================================="
ls -lh paper/VisCon/figures/variants/
echo ""
echo "Finished at $(date)"
echo ""
echo "NEXT STEPS:"
echo "  1. Browse the variants/ folder and pick your favorites"
echo "  2. Copy your chosen qualitative figure:"
echo "     cp paper/VisCon/figures/variants/diverse_4rows.png paper/VisCon/figures/qualitative_comparison.png"
echo "  3. Copy your chosen teaser:"
echo "     cp paper/VisCon/figures/variants/best_row0_teaser.png paper/VisCon/figures/teaser.png"
echo "  4. git add + commit + push"
