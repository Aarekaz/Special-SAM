#!/bin/bash
#SBATCH --job-name=precompute-aug
#SBATCH --output=logs/precompute_aug_%j.out
#SBATCH --error=logs/precompute_aug_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

# Precompute augmented embeddings (rotation + pixel shift)
# Run this before augmentation ablation training

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

echo "Starting augmented embedding precomputation at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Precompute for flip+rot config
python -m src.training.precompute --config configs/ablation/aug_flip_rot.yaml

# Precompute for flip+rot+shift config
python -m src.training.precompute --config configs/ablation/aug_flip_rot_shift.yaml

echo "Finished at $(date)"
