#!/bin/bash
#SBATCH --job-name=error-analysis
#SBATCH --output=logs/error_analysis_%j.out
#SBATCH --error=logs/error_analysis_%j.err
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# CPU-only job: runs error analysis on pre-computed per-image results

export PATH="/SEAS/home/g37014071/.conda/envs/specialsam/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs results/error_analysis

echo "Starting error analysis at $(date)"

python scripts/error_analysis.py \
    --csv results/per_image_results.csv \
    --output-dir results/error_analysis \
    --top-k 400 \
    --test-mask-dir data/cod10k/COD10K-v3/Test/GT_Object

echo "Finished at $(date)"
