#!/bin/bash
#SBATCH --job-name=sam-precompute
#SBATCH --output=logs/precompute_%j.out
#SBATCH --error=logs/precompute_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

module load python cuda

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

source venv/bin/activate 2>/dev/null || true

echo "Starting embedding pre-computation at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m src.training.precompute --config configs/train.yaml

echo "Finished at $(date)"
