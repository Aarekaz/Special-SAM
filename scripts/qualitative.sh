#!/bin/bash
#SBATCH --job-name=sam-qualitative
#SBATCH --output=logs/qualitative_%j.out
#SBATCH --error=logs/qualitative_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

echo "Starting qualitative figure generation at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

python -m scripts.generate_qualitative_figures --config configs/eval.yaml

echo "Finished at $(date)"
