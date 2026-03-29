#!/bin/bash
#SBATCH --job-name=sam-crossdataset
#SBATCH --output=logs/crossdataset_%A_%a.out
#SBATCH --error=logs/crossdataset_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-1

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs results/crossdataset

DATASETS=("camo" "nc4k" "chameleon")
CONFIGS=("configs/eval_camo.yaml" "configs/eval_nc4k.yaml" "configs/eval_chameleon.yaml")

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "=== Cross-dataset evaluation: ${DATASET} ==="
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
echo "Config: ${CONFIG}"

python -m src.evaluation.evaluate --config "${CONFIG}"

echo "Finished ${DATASET} at $(date)"
