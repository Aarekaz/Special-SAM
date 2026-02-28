#!/bin/bash
#SBATCH --job-name=sam-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

echo "Starting comprehensive evaluation at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

python -m src.evaluation.evaluate --config configs/eval.yaml

echo "Finished at $(date)"
