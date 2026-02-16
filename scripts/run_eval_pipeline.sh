#!/bin/bash
#SBATCH --job-name=sam-eval-pipeline
#SBATCH --output=logs/eval_pipeline_%j.out
#SBATCH --error=logs/eval_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

module load python3/3.11.6

cd "$SLURM_SUBMIT_DIR" || exit 1

source venv/bin/activate

echo "=== Starting evaluation at $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m src.evaluation.evaluate --config configs/eval.yaml

echo "=== Evaluation finished at $(date) ==="
echo "=== Generating paper tables ==="

python scripts/update_paper_tables.py

echo "=== All done at $(date) ==="
