#!/bin/bash
#SBATCH --job-name=sam-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00

module load python cuda

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

source venv/bin/activate 2>/dev/null || true

echo "Starting comprehensive evaluation at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m src.evaluation.evaluate --config configs/eval.yaml

echo "Finished at $(date)"
