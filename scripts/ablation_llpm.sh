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

# ---- Step 1: Train LLPM variant ----
echo "[Step 1/2] Training LLPM variant: ${NAME}..."
python -m src.training.train_llpm --config "${CONFIG}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed for ${NAME} (exit code ${TRAIN_EXIT})"
    exit 1
fi

# ---- Step 2: Evaluate LLPM variant ----
LLPM_PATHS=(
    "checkpoints/ablation/llpm_edge_only.pth"
    "checkpoints/ablation/llpm_enhance_only.pth"
    "checkpoints/ablation/llpm_no_gate.pth"
)
DECODER_PATHS=(
    "checkpoints/ablation/camo_decoder_llpm_edge_only.pth"
    "checkpoints/ablation/camo_decoder_llpm_enhance_only.pth"
    "checkpoints/ablation/camo_decoder_llpm_no_gate.pth"
)

LLPM_CKPT=${LLPM_PATHS[$IDX]}
DECODER_CKPT=${DECODER_PATHS[$IDX]}

echo ""
echo "[Step 2/2] Evaluating LLPM variant: ${NAME}..."
echo "  LLPM:    ${LLPM_CKPT}"
echo "  Decoder: ${DECODER_CKPT}"

python -c "
import sys, yaml
sys.path.insert(0, '.')
from src.evaluation.evaluate import evaluate_model

config = {
    'model': {
        'type': 'vit_h',
        'base_checkpoint': 'weights/sam_vit_h_4b8939.pth',
        'llpm_path': '${LLPM_CKPT}',
        'llpm_decoder': '${DECODER_CKPT}',
    },
    'llpm': {
        'edge_channels': 32,
        'enhance_channels': 64,
    },
    'data': {
        'test_img_dir': 'data/cod10k/COD10K-v3/Test/Image',
        'test_mask_dir': 'data/cod10k/COD10K-v3/Test/GT_Object',
    },
    'evaluation': {
        'max_samples': 500,
        'target_size': 1024,
        'boundary_threshold': 5,
        'prompt_strategies': ['center', 'edge_single', 'multi_grid', 'multi_random'],
    },
}

agg_df, _, _ = evaluate_model('llpm', config)
if not agg_df.empty:
    agg_df.to_csv('results/ablation/${NAME}_results.csv', index=False)
    print('Results saved to results/ablation/${NAME}_results.csv')
else:
    print('WARNING: No results produced')
"
EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "WARNING: Eval failed for ${NAME} (exit code ${EVAL_EXIT}), but training succeeded"
fi

echo "LLPM ablation ${NAME} finished at $(date)"
