#!/bin/bash
#SBATCH --job-name=sam-ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-3

# ============================================================================
# Ablation Study: Loss Function Configurations
# ============================================================================
# Trains decoder variants with different BCE/Dice weight combinations,
# then evaluates each resulting checkpoint.
#
# Array job mapping:
#   0 = bce_only    (BCE=1.0, Dice=0.0)
#   1 = dice_only   (BCE=0.0, Dice=1.0)
#   2 = bce_heavy   (BCE=0.7, Dice=0.3)
#   3 = dice_heavy  (BCE=0.3, Dice=0.7)
#
# Usage:
#   sbatch scripts/ablation.sh          # submit all 4 as array job
#   sbatch --array=0 scripts/ablation.sh  # run only bce_only
# ============================================================================

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs checkpoints/ablation results/ablation

# Define ablation configurations
CONFIGS=(
    "configs/ablation/bce_only.yaml"
    "configs/ablation/dice_only.yaml"
    "configs/ablation/bce_heavy.yaml"
    "configs/ablation/dice_heavy.yaml"
)

NAMES=(
    "bce_only"
    "dice_only"
    "bce_heavy"
    "dice_heavy"
)

DECODERS=(
    "checkpoints/ablation/camo_decoder_bce_only.pth"
    "checkpoints/ablation/camo_decoder_dice_only.pth"
    "checkpoints/ablation/camo_decoder_bce_heavy.pth"
    "checkpoints/ablation/camo_decoder_dice_heavy.pth"
)

# Select config for this array task
IDX=${SLURM_ARRAY_TASK_ID}
CONFIG=${CONFIGS[$IDX]}
NAME=${NAMES[$IDX]}
DECODER=${DECODERS[$IDX]}

echo "============================================"
echo "Ablation Study: ${NAME}"
echo "Config: ${CONFIG}"
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
echo "============================================"

# ---- Step 1: Train decoder with this loss configuration ----
echo ""
echo "[Step 1/2] Training decoder: ${NAME}"
python -m src.training.train --config "${CONFIG}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed for ${NAME} (exit code ${TRAIN_EXIT})"
    exit 1
fi

if [ ! -f "${DECODER}" ]; then
    echo "ERROR: Expected decoder checkpoint not found: ${DECODER}"
    exit 1
fi

echo "Training complete. Decoder saved to: ${DECODER}"

# ---- Step 2: Evaluate the trained decoder ----
echo ""
echo "[Step 2/2] Evaluating decoder: ${NAME}"

# Use the ablation evaluation script with this single checkpoint
# Evaluate on 500 samples for speed (full test set can be run separately)
python scripts/eval_ablation.py \
    --decoders "${DECODER}" \
    --names "${NAME}" \
    --max-samples 500 \
    --output-csv "results/ablation/${NAME}_results.csv" \
    --prompt-strategies center multi_grid

EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "WARNING: Evaluation failed for ${NAME} (exit code ${EVAL_EXIT})"
    echo "Training succeeded; you can re-run evaluation separately."
fi

echo ""
echo "============================================"
echo "Ablation ${NAME} finished at $(date)"
echo "============================================"
