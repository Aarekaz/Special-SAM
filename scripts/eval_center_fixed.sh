#!/bin/bash
#SBATCH --job-name=sam-center-fixed
#SBATCH --output=logs/center_fixed_%j.out
#SBATCH --error=logs/center_fixed_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs results

echo "=== Fixed center-of-image prompt evaluation ==="
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

python -m src.evaluation.evaluate --config configs/eval_center_fixed.yaml

echo "Finished at $(date)"
