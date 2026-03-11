#!/bin/bash
#SBATCH --job-name=aug-ablation
#SBATCH --output=logs/ablation_aug_%A_%a.out
#SBATCH --error=logs/ablation_aug_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-1

# ============================================================================
# Ablation Study: Data Augmentation
# ============================================================================
# Tests progressive augmentation strategies:
#   0 = flip + rotation
#   1 = flip + rotation + pixel shift
#
# Compare against flip-only baseline (camo_decoder_vith.pth)
#
# Each task: precompute augmented embeddings, then train decoder
# ============================================================================

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs checkpoints/ablation results/ablation

CONFIGS=(
    "configs/ablation/aug_flip_rot.yaml"
    "configs/ablation/aug_flip_rot_shift.yaml"
)

NAMES=(
    "flip_rot"
    "flip_rot_shift"
)

IDX=${SLURM_ARRAY_TASK_ID}
CONFIG=${CONFIGS[$IDX]}
NAME=${NAMES[$IDX]}

echo "============================================"
echo "Augmentation Ablation: ${NAME}"
echo "Config: ${CONFIG}"
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# Step 1: Precompute augmented embeddings
echo "[Step 1/3] Precomputing embeddings..."
python -m src.training.precompute --config "${CONFIG}"

# Step 2: Train decoder
echo "[Step 2/3] Training decoder..."
python -m src.training.train --config "${CONFIG}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed for ${NAME}"
    exit 1
fi

# Step 3: Evaluate
echo "[Step 3/3] Evaluating..."
DECODER="checkpoints/ablation/camo_decoder_${NAME}.pth"
python scripts/eval_ablation.py \
    --decoders "${DECODER}" \
    --names "${NAME}" \
    --max-samples 0 \
    --output-csv "results/ablation/${NAME}_results.csv" \
    --prompt-strategies center edge_single multi_grid multi_random

echo "Augmentation ablation ${NAME} finished at $(date)"
