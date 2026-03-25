#!/bin/bash
#SBATCH --job-name=sam-coco-fig
#SBATCH --output=logs/coco_figure_%j.out
#SBATCH --error=logs/coco_figure_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

echo "=== COCO regular image figure generation ==="
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m scripts.generate_coco_figure \
    --config configs/eval.yaml \
    --img-dir data/coco_val \
    --num-examples 8 \
    --output paper/latex/figures/coco_comparison.png

echo "Finished at $(date)"
