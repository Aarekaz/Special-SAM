#!/bin/bash
#SBATCH --job-name=sam-diagram
#SBATCH --output=logs/diagram_%j.out
#SBATCH --error=logs/diagram_%j.out
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# No GPU needed — just matplotlib rendering

echo "=== Architecture Diagram Generation ==="
echo "Started at $(date)"
echo "Node: $(hostname)"

# Activate conda
source ~/.bashrc
conda activate specialsam

cd ~/Special-SAM

python scripts/generate_architecture_diagram.py

echo "Finished at $(date)"
echo "Output: paper/figures/architecture_diagram.png"
echo "Output: paper/figures/architecture_diagram.pdf"
