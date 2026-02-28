#!/bin/bash
#SBATCH --job-name=sam-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

module load miniconda/23.11.0-2
module load cuda/12.9

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

conda activate specialsam

echo "Starting decoder training at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m src.training.train --config configs/train.yaml

echo "Finished at $(date)"
