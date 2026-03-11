#!/bin/bash
#SBATCH --job-name=prompt-ablation
#SBATCH --output=logs/ablation_prompt_%A_%a.out
#SBATCH --error=logs/ablation_prompt_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1

# ============================================================================
# Ablation Study: Prompt-Type Training
# ============================================================================
# Trains decoder variants with point-only vs box-only prompts.
# Compare against the default mixed-prompt model (camo_decoder_vith.pth).
#
# Array job mapping:
#   0 = point_only
#   1 = box_only
#
# Usage:
#   sbatch scripts/ablation_prompt.sh
# ============================================================================

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs checkpoints/ablation results/ablation

CONFIGS=(
    "configs/ablation/prompt_point_only.yaml"
    "configs/ablation/prompt_box_only.yaml"
)

NAMES=(
    "point_only"
    "box_only"
)

DECODERS=(
    "checkpoints/ablation/camo_decoder_point_only.pth"
    "checkpoints/ablation/camo_decoder_box_only.pth"
)

IDX=${SLURM_ARRAY_TASK_ID}
CONFIG=${CONFIGS[$IDX]}
NAME=${NAMES[$IDX]}
DECODER=${DECODERS[$IDX]}

echo "============================================"
echo "Prompt Ablation: ${NAME}"
echo "Config: ${CONFIG}"
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# Step 1: Train
echo "[Step 1/2] Training decoder: ${NAME}"
python -m src.training.train --config "${CONFIG}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed for ${NAME} (exit code ${TRAIN_EXIT})"
    exit 1
fi

# Step 2: Evaluate
echo "[Step 2/2] Evaluating decoder: ${NAME}"
python scripts/eval_ablation.py \
    --decoders "${DECODER}" \
    --names "${NAME}" \
    --max-samples 0 \
    --output-csv "results/ablation/${NAME}_results.csv" \
    --prompt-strategies center edge_single multi_grid multi_random

echo "Prompt ablation ${NAME} finished at $(date)"
