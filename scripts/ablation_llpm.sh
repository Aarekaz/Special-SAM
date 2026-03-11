#!/bin/bash
#SBATCH --job-name=llpm-ablation
#SBATCH --output=logs/ablation_llpm_%A_%a.out
#SBATCH --error=logs/ablation_llpm_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-2

# ============================================================================
# Ablation Study: LLPM Component Analysis
# ============================================================================
# Tests contribution of each LLPM branch:
#   0 = edge_only     (edge branch + alpha, no enhancement)
#   1 = enhance_only  (enhancement branch + alpha, no edge gating)
#   2 = no_gate       (both branches, alpha fixed to 1.0)
#
# Compare against full LLPM (llpm_vith.pth) and no-LLPM (camo_decoder_vith.pth)
#
# Usage:
#   sbatch scripts/ablation_llpm.sh
# ============================================================================

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs checkpoints/ablation results/ablation

CONFIGS=(
    "configs/ablation/llpm_edge_only.yaml"
    "configs/ablation/llpm_enhance_only.yaml"
    "configs/ablation/llpm_no_gate.yaml"
)

NAMES=(
    "llpm_edge_only"
    "llpm_enhance_only"
    "llpm_no_gate"
)

IDX=${SLURM_ARRAY_TASK_ID}
CONFIG=${CONFIGS[$IDX]}
NAME=${NAMES[$IDX]}

echo "============================================"
echo "LLPM Ablation: ${NAME}"
echo "Config: ${CONFIG}"
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# Train LLPM variant
python -m src.training.train_llpm --config "${CONFIG}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed for ${NAME} (exit code ${TRAIN_EXIT})"
    exit 1
fi

echo "LLPM ablation ${NAME} finished at $(date)"
