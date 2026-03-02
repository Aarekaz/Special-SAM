#!/bin/bash
#SBATCH --job-name=sam-llpm
#SBATCH --output=logs/train_llpm_%j.out
#SBATCH --error=logs/train_llpm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs checkpoints

echo "Starting LLPM training at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

# Verify decoder-only checkpoint exists for warm-start
if [ ! -f "checkpoints/camo_decoder_vith.pth" ]; then
    echo "ERROR: Decoder checkpoint not found. Run decoder-only training first."
    echo "  sbatch scripts/train.sh"
    exit 1
fi

# Train LLPM + decoder
python -m src.training.train_llpm --config configs/train_llpm.yaml

echo "Finished at $(date)"
