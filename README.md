# Special-SAM

Special-SAM is a research codebase for specializing Segment Anything (SAM ViT-H) for camouflaged object detection.

The core idea is intentionally narrow: keep SAM's large image encoder frozen, then adapt the smaller mask-decoding path and lightweight preprocessing modules for camouflage-heavy scenes. The repo tracks the training pipeline, prompt strategies, evaluation scripts, ablations, paper figures, and experiment logs used for the project.

## What is here

- Decoder-only fine-tuning for SAM ViT-H on COD10K camouflage data.
- A Learnable Local Preprocessing Module (LLPM) path for improving difficult camouflage cases.
- Prompt evaluations for center-point, edge-point, grid, random multi-point, and box prompting.
- Cross-dataset evaluation over COD10K, CAMO, and NC4K experiment outputs.
- SLURM scripts and run logs for GWU Pegasus cluster experiments.
- Paper drafts, figures, tables, and reproducibility notes.

## Current status

The decoder-only baseline and LLPM experiments have both been run. The tracked logs and CSV outputs record the main ablations, cross-dataset evaluations, qualitative figures, and VisCon/CVPR-style paper artifacts.

This repo is research-first rather than package-first. The notebooks, logs, and paper folders are part of the project record.

## Repository map

```text
configs/        training, evaluation, and ablation configs
src/            dataset loading, SAM wrappers, training, evaluation, metrics
scripts/        SLURM jobs plus figure and analysis utilities
results/        tracked CSV summaries and ablation outputs
logs/           experiment and HPC run logs
paper/          paper drafts, figures, and LaTeX sources
plans/          design notes for LLPM and video-pipeline extensions
notebooks/      exploratory and full-training notebooks
SAM-Paper/      reference papers used during the project
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The base SAM checkpoint, COD datasets, cached embeddings, and trained checkpoints are intentionally not committed. See `.gitignore`, `PROJECT_LOG.md`, and `EXPERIMENTS.md` for the expected artifacts and run history.

## Common entry points

```bash
# Train the decoder-only specialization
bash scripts/train.sh

# Evaluate configured checkpoints/prompts
bash scripts/eval.sh

# Run ablation jobs
bash scripts/ablation.sh
bash scripts/ablation_prompt.sh
bash scripts/ablation_llpm.sh

# Regenerate paper figures
bash scripts/generate_all_figures.sh
```

Most scripts assume the project is running in the same data/checkpoint layout used on the research cluster. Adjust paths in `configs/*.yaml` before running locally.

## Results and notes

The project log is the best summary of the experiment state:

- `PROJECT_LOG.md` explains the method, headline metrics, paper artifacts, and next phases.
- `EXPERIMENTS.md` tracks SLURM jobs, checkpoint names, ablations, and evaluation outputs.
- `logs/HPC_RUN_LOG.md` is the canonical run log for cluster jobs.

## Related work

This repo is separate from `Aarekaz/CamoSet`. CamoSet is the dataset/pipeline side; Special-SAM is the SAM specialization and evaluation side. A useful bridge between them is a thin inference adapter that compares CamoSet motion/background masks against Special-SAM/SAM masks before any deeper integration.
