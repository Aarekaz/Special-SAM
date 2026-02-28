#!/bin/bash
#SBATCH --job-name=sam-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs checkpoints

echo "Starting decoder training at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

# Step 1: Pre-compute embeddings if not already done
if [ ! -d "data/embeddings/camo_embeddings_vith" ]; then
    echo "Pre-computing embeddings..."
    python -m src.training.precompute --config configs/train.yaml
fi

# Step 2: Train decoder
python -m src.training.train --config configs/train.yaml

echo "Finished at $(date)"
