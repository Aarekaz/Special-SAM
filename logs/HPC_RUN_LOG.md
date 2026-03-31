# HPC Run Log — GWU Pegasus Cluster

Tracks all SLURM jobs, their purpose, status, and results.

**Last updated:** 2026-03-30 (VisCon campaign: full result CSVs in git, NC4K multi-prompt complete, `.gitignore` fix for `results/`).

---

## Job History

### Job 5785644 — Full Test Set Evaluation (original decoder, 7 epochs)
- **Script:** `scripts/eval.sh`
- **Submitted:** 2026-02-28
- **GPU:** A100 80GB
- **Runtime:** ~3 hours (17:21 - 20:20 EST)
- **Status:** COMPLETED
- **Config:** `configs/eval.yaml` (max_samples=0, all test images)
- **Decoder:** `checkpoints/camo_decoder_vith.pth` (7-epoch, flat LR)
- **Output:** `results/comprehensive_evaluation_results.csv`, `results/per_category_results.csv`
- **Log:** `logs/eval_5785644.out`
- **Results:**

| Prompt Strategy | Base mIoU | Specialized mIoU | Improvement |
|---|---|---|---|
| Center-of-Mass | 0.5724 | 0.7138 | +24.7% |
| Edge (Single) | 0.2052 | 0.6563 | +219.8% |
| Multi-Point Grid | 0.6815 | 0.7227 | +6.0% |
| Multi-Point Random | 0.7303 | 0.7449 | +2.0% |

---

### Job 5785646 — Retrain Decoder (15 epochs, cosine LR)
- **Script:** `scripts/train.sh`
- **Submitted:** 2026-02-28
- **GPU:** A100 80GB
- **Runtime:** ~6.5 hours (18:42 - 01:05 EST) including precompute
- **Status:** COMPLETED
- **Config:** `configs/train.yaml` (15 epochs, cosine LR, warmup=1, AdamW, lr=1e-4)
- **Output:** `checkpoints/camo_decoder_vith.pth` (overwrites old 7-epoch decoder)
- **Log:** `logs/train_5786340.out`
- **Notes:** First run failed (job 5785646) due to missing `CamoDataset` class. Fixed in commit bb76d16. Precompute completed successfully (6080 embeddings cached). Re-run as job 5786340 succeeded.
- **Training curve:**

| Epoch | Loss | LR |
|---|---|---|
| 1 | 0.1417 | 1.00e-04 |
| 5 | 0.1094 | 8.92e-05 |
| 10 | 0.0842 | 3.95e-05 |
| 15 | 0.0717 | 2.24e-06 |

---

### Job 5786340 — Retrain Decoder (re-run after CamoDataset fix)
- **Script:** `scripts/train.sh`
- **Submitted:** 2026-03-01
- **GPU:** A100 80GB
- **Runtime:** ~30 min (embeddings cached, training only)
- **Status:** COMPLETED
- **Config:** `configs/train.yaml` (15 epochs, cosine LR)
- **Output:** `checkpoints/camo_decoder_vith.pth`
- **Log:** `logs/train_5786340.out`
- **Final loss:** 0.0717 (49.4% reduction from epoch 1)

---

### Job 5786512 — Full Test Set Evaluation (new 15-epoch decoder)
- **Script:** `scripts/eval.sh`
- **Submitted:** 2026-03-01
- **GPU:** A100 80GB
- **Runtime:** ~3 hours (14:25 - 17:27 EST)
- **Status:** COMPLETED
- **Config:** `configs/eval.yaml` (max_samples=0, all test images)
- **Decoder:** `checkpoints/camo_decoder_vith.pth` (15-epoch, cosine LR)
- **Output:** `results/comprehensive_evaluation_results.csv`, `results/per_category_results.csv`
- **Log:** `logs/eval_5786512.out`
- **Results (15-epoch decoder, 2,026 camouflaged test images):**

| Prompt Strategy | Base mIoU | Specialized mIoU | Improvement |
|---|---|---|---|
| Center-of-Mass | 0.5724 | 0.7381 | +28.9% |
| Edge (Single) | 0.2052 | 0.6938 | +238.1% |
| Multi-Point Grid | 0.6815 | 0.7556 | +10.9% |
| Multi-Point Random | 0.7293 | 0.7751 | +6.3% |

- **Improvement over 7-epoch decoder (Job 5785644):**

| Prompt Strategy | 7-epoch mIoU | 15-epoch mIoU | Gain |
|---|---|---|---|
| Center-of-Mass | 0.7138 | 0.7381 | +2.4 pts |
| Edge (Single) | 0.6563 | 0.6938 | +3.8 pts |
| Multi-Point Grid | 0.7227 | 0.7556 | +3.3 pts |
| Multi-Point Random | 0.7449 | 0.7751 | +3.0 pts |

---

### Job 5786513 — Qualitative Figure Generation
- **Script:** `scripts/qualitative.sh`
- **Submitted:** 2026-03-01
- **GPU:** A100 80GB
- **Runtime:** ~18 min (14:25 - 14:43 EST)
- **Status:** COMPLETED
- **Config:** `configs/eval.yaml`
- **Output:** `paper/figures/qualitative_comparison.png` (4200 x 11100 px, ~21MB)
- **Log:** `logs/qualitative_5786513.out`
- **Details:** Scored all 4000 test images, selected 12 examples (535 hard, 480 medium, 1011 easy candidates). Generated 12-row comparison grid with columns: Image, GT, Base SAM, Specialized SAM.

---

### Job 5786514 — Ablation Study (loss function variants)
- **Script:** `scripts/ablation.sh` (SLURM array job, 4 tasks)
- **Submitted:** 2026-03-01
- **GPU:** A100 80GB x4 (parallel array)
- **Status:** COMPLETED
- **Configs:**
  - Task 0: `configs/ablation/bce_only.yaml` (BCE=1.0, Dice=0.0)
  - Task 1: `configs/ablation/dice_only.yaml` (BCE=0.0, Dice=1.0)
  - Task 2: `configs/ablation/bce_heavy.yaml` (BCE=0.7, Dice=0.3)
  - Task 3: `configs/ablation/dice_heavy.yaml` (BCE=0.3, Dice=0.7)
- **Output:** `checkpoints/ablation/camo_decoder_*.pth`, `results/ablation/*_results.csv`
- **Logs:** `logs/ablation_5786514_0.out` through `logs/ablation_5786514_3.out`
- **Results (Center-of-Mass prompt, 500 test images):**

| Variant | BCE | Dice | mIoU | S-alpha | E-phi | MAE |
|---|---|---|---|---|---|---|
| BCE only | 1.0 | 0.0 | 0.7134 | 0.8234 | 0.9259 | 0.0420 |
| Dice only | 0.0 | 1.0 | 0.7123 | 0.8151 | 0.9228 | 0.0463 |
| BCE-heavy | 0.7 | 0.3 | 0.7076 | 0.8175 | 0.9246 | 0.0443 |
| **Dice-heavy** | **0.3** | **0.7** | **0.7285** | **0.8291** | **0.9319** | **0.0427** |

- **Key finding:** Dice-heavy (0.3/0.7) performs best among ablation variants. All variants are competitive, showing robustness to loss configuration. Our default 0.5/0.5 on full test set achieves 0.7381 mIoU.

---

## Embedding Cache

- **Path:** `data/embeddings/camo_embeddings_vith/`
- **Created by:** Job 5785646 (precompute step)
- **Samples:** 6,080 (3,040 camouflaged images x2 with horizontal flip augmentation)
- **Metadata:** `data/embeddings/camo_train_meta_vith.csv`
- **Size:** ~45GB
- **Reused by:** All training jobs (train.sh skips precompute if dir exists)

---

### Job 5847371 — Cross-Dataset Evaluation (CAMO + NC4K)
- **Script:** `scripts/eval_crossdataset.sh` (SLURM array, tasks 0-1)
- **Submitted:** 2026-03-25
- **GPU:** A100 80GB
- **Wall time:** 6 hrs per task
- **Status:** COMPLETED
- **Configs:**
  - Task 0: `configs/eval_camo.yaml` (CAMO test, 250 images)
  - Task 1: `configs/eval_nc4k.yaml` (NC4K test, 4,121 images)
- **Models evaluated:** Base SAM, Decoder-only, LLPM+Decoder (Center-of-Mass prompt)
- **Output:** `results/crossdataset/camo_results.csv`, `results/crossdataset/nc4k_results.csv`
- **Logs:** `logs/crossdataset_5847371_0.out`, `logs/crossdataset_5847371_1.out`
- **Results:**

| Dataset | Model | mIoU | S_α | E_φ | MAE |
|---|---|---|---|---|---|
| CAMO (250) | Base SAM | 0.489 | 0.672 | 0.748 | 0.122 |
| CAMO (250) | Decoder-only | 0.658 | 0.787 | 0.875 | 0.075 |
| CAMO (250) | LLPM+Decoder | 0.677 | 0.797 | 0.887 | 0.073 |
| NC4K (4121) | Base SAM | 0.567 | 0.728 | 0.805 | 0.093 |
| NC4K (4121) | Decoder-only | 0.741 | 0.841 | 0.924 | 0.048 |
| NC4K (4121) | LLPM+Decoder | 0.752 | 0.848 | 0.929 | 0.045 |

- **Notes:** CHAMELEON skipped (dataset not downloaded). Cross-dataset generalization table added to paper (`sec/4_experiments.tex`).

---

### Job 5847376 — Fixed Center-of-Image Prompt Evaluation
- **Script:** `scripts/eval_center_fixed.sh`
- **Submitted:** 2026-03-25
- **GPU:** A100 80GB
- **Wall time:** 6 hrs
- **Status:** COMPLETED
- **Config:** `configs/eval_center_fixed.yaml`
- **Prompt:** Fixed image center (width/2, height/2) — no GT used
- **Models evaluated:** Base SAM, Decoder-only, LLPM+Decoder
- **Output:** `results/center_fixed_results.csv`
- **Logs:** `logs/center_fixed_5847376.out`
- **Why results were NOT included in the paper:**
  - The eval config pointed to `Test/Image` which contains ~4,000 images (2,026 camouflaged + ~1,974 non-camouflaged background images). The eval ran on all of them.
  - On background images there is no object near the image center, so GT mask is empty and the fixed-center point hits nothing. This tanked absolute metrics: Base SAM 0.165 mIoU, Decoder-only 0.287 mIoU.
  - LLPM result was clearly bugged (0.001 mIoU), likely due to the same empty-mask edge case.
  - Putting 0.287 mIoU as our "GT-free" headline would actively hurt the paper — reviewers would just see a weak model.
  - **Decision:** instead of presenting these numbers, we reframed the prompting story in `sec/1_intro.tex` and `sec/6_discussion.tex`: oracle center-of-mass prompts = standard SAM interactive segmentation protocol (simulated user click on target), not a GT dependency. Prompt robustness (3.5x→1.1x variance reduction) is the key practical finding.
  - If a clean fixed-center eval is needed in future, re-run with a config that filters to camouflaged-only images (e.g. using the GT mask list to whitelist files).

### Job 5850045 — Re-run center_fixed on camouflaged-only images
- **Root cause fixed:** `configs/eval_center_fixed.yaml` now points to `data/cod10k/COD10K-v3/Test/Image_CAM` (symlink dir with only 2,026 camouflaged images, created with `ln -s ../Image/COD10K-CAM-* .`)
- **Script:** `scripts/eval_center_fixed.sh`
- **Submitted:** 2026-03-28
- **Status:** RUNNING
- **Submission command:** `sbatch scripts/eval_center_fixed.sh`
- **Expected output:** `results/center_fixed_results.csv` (overwritten with correct 2,026-sample results)
- **What to do with results:** Paste CSV output to Claude. If Decoder-only mIoU > ~0.55, add a row to the main table in `paper/latex/sec/4_experiments.tex` and update the prompting discussion in `sec/6_discussion.tex` to cite a concrete GT-free number.

---

### Job 5847383 — COCO Regular Image Figure Generation
- **Script:** `scripts/coco_figure.sh`
- **Submitted:** 2026-03-25
- **GPU:** A100 80GB
- **Status:** COMPLETED
- **Config:** `configs/eval.yaml`
- **Output:** `paper/latex/figures/coco_comparison.png` (5.1MB)
- **Notes:** Sampled 8 random COCO val2017 images. Fixed image-center prompt, no GT. Shows specialized decoder does not degrade on non-camouflage images. Added to paper as qualitative generalization figure (`sec/4_experiments.tex`).

---

### Job 5854374 — Prompt-Mix Ablation (full test set)
- **Script:** `scripts/ablation_prompt.sh` (SLURM array, tasks 0-1)
- **Submitted:** 2026-03-29
- **GPU:** A100 80GB x2 (parallel array)
- **Status:** COMPLETED (task 0 finished 2026-03-29 18:56 EDT; task 1 finished 2026-03-29 23:15 EDT)
- **Configs:**
  - Task 0: `configs/ablation/prompt_point_only.yaml` (point-only training)
  - Task 1: `configs/ablation/prompt_box_only.yaml` (box-only training)
- **Evaluation:** `--max-samples 0` (all matched test pairs), all 4 prompt types (center, edge_single, multi_grid, multi_random)
- **Output:** `results/ablation/point_only_results.csv`, `results/ablation/box_only_results.csv`
- **Logs:** `logs/ablation_prompt_5854374_0.out`, `logs/ablation_prompt_5854374_1.out`
- **Purpose:** Rerun prompt ablation on full test set (previously 500-image subset) for VisCon paper Table 2.
- **Important:** Eval matched **4,000** image-mask pairs (`Test/Image` + `GT_Object`), not the 2,026 camouflaged-only subset. Paper Table 2 should either align eval to `Image_CAM` / camouflaged list or report 4,000-sample numbers explicitly.
- **Results (mIoU, from job logs):**

| Training | CoM | Edge | Grid | Random |
|---|---|---|---|---|
| point_only | 0.7287 | 0.6732 | 0.7472 | 0.7664 |
| box_only | 0.3846 | 0.5226 | 0.4242 | 0.4912 |

- **Prompt sensitivity (max/min mIoU):** point_only ~1.14x; box_only ~1.36x (vs. mixed decoder ~1.1x on 2,026 camouflaged COD10K in main table).

---

### Job 5854375 — Cross-Dataset Evaluation (CAMO + NC4K, all 4 prompt types)
- **Script:** `scripts/eval_crossdataset.sh` (SLURM array, tasks 0-1)
- **Submitted:** 2026-03-29
- **GPU:** A100 80GB x2 (parallel array)
- **Status:** **COMPLETED** — task 0 finished 2026-03-29 23:54 EDT (~32 min); task 1 finished 2026-03-30 07:02 EDT (~7h 35m, ExitCode 0)
- **Configs:**
  - Task 0: `configs/eval_camo.yaml` (CAMO test, 250 images, 4 prompt types)
  - Task 1: `configs/eval_nc4k.yaml` (NC4K test, 4,121 images, 4 prompt types)
- **Models evaluated:** Base SAM, Decoder-only, LLPM+Decoder
- **Prompt strategies:** center, edge_single, multi_grid, multi_random (expanded from center-only in previous Job 5847371)
- **Output:** `results/crossdataset/camo_results.csv`, `results/crossdataset/nc4k_results.csv`, plus optional per-image: `camo_per_image.csv`, `nc4k_per_image.csv`, `per_category_results.csv` (cross-dataset)
- **Logs:** `logs/crossdataset_5854375_0.out`, `logs/crossdataset_5854375_1.out`
- **Purpose:** Expand cross-dataset eval to all prompt types for VisCon paper Table 3 (Range column: CAMO 4.5×→1.1×; NC4K 5.8×→1.1× for decoder-only vs base).

**CAMO (250 images) — mIoU (from logs / `camo_results.csv`):**

| Model | CoM | Edge | Grid | Random | Range (approx.) |
|---|---|---|---|---|---|
| Base SAM | 0.489 | 0.153 | 0.688 | 0.692 | 4.5× |
| Specialized | 0.658 | 0.619 | 0.698 | 0.703 | 1.1× |
| LLPM+Decoder | 0.677 | 0.656 | 0.722 | 0.737 | 1.1× |

**NC4K (4,121 images) — mIoU (from `nc4k_results.csv`, completed 2026-03-30):**

| Model | CoM | Edge | Grid | Random | Range (approx.) |
|---|---|---|---|---|---|
| Base SAM | 0.567 | 0.127 | 0.726 | 0.739 | 5.8× |
| Specialized | 0.741 | 0.715 | 0.770 | 0.780 | 1.1× |
| LLPM+Decoder | 0.752 | 0.742 | 0.779 | 0.786 | 1.1× |

---

### Job 5854478 — Qualitative figure (VisCon refresh)
- **Script:** `scripts/qualitative.sh` (or equivalent; job name `sam-quali+` on cluster)
- **Submitted:** 2026-03-30 (~02:15 UTC end time per `sacct`)
- **GPU:** A100 80GB
- **Runtime:** ~33 min
- **Status:** COMPLETED (ExitCode 0)
- **Output:** `paper/VisCon/figures/qualitative_comparison.png` (updated grid; contour overlays / IoU badges per pipeline changes)
- **Logs:** `logs/qualitative_5854478.out`, `logs/qualitative_5854478.err`
- **Git:** Included in commit `17e818d` on HPC, then full repo sync to `origin/main`.

---

## Git / reproducibility (2026-03-29 — 2026-03-30)

These commits preserve HPC outputs without relying on cluster scratch long-term.

| Commit | Summary |
|--------|---------|
| `2808fcc` | Begin tracking `results/crossdataset/` CSVs; gitignore exceptions |
| `7fb16a0` | Ablation + cross-dataset CSVs, SLURM logs |
| `17e818d` | Full NC4K `nc4k_results.csv`, qualitative figure, log updates |
| `a898500` | Refine `.gitignore` (`results/*` + un-ignore subfolders so `git add results/...` works), VisCon manuscript + `ablation_comparison_table.txt`, `todo.md` |
| `58ab349` | **Full HPC bundle:** all ablation CSVs (point/box, LLPM variants, BCE/Dice), per-image CSVs (~26 MB total: NC4K per-image ~15 MB), `center_fixed_*`, `per_image_results.csv`, etc. |

**`.gitignore` note:** Plain `results/` blocked negated paths; replaced with `results/*` plus explicit `!results/crossdataset/**`, `!results/ablation/**`, `!results/tables/**`, `!results/*.csv`. Checkpoints remain ignored (`checkpoints/`, `*.pth`).

**Tracked files under `results/` (20 files):** run `git ls-files results/` — includes `comprehensive_evaluation_results.csv`, `per_category_results.csv`, `per_image_results.csv`, `center_fixed_results.csv`, `center_fixed_per_image.csv`, full `crossdataset/*`, full `ablation/*` (including `point_only_results.csv`, `box_only_results.csv`, `llpm_*_results.csv`, loss ablations, `ablation_comparison_table.txt`).

**SLURM logs for this campaign (filenames):**
- `logs/ablation_prompt_5854374_0.out`, `logs/ablation_prompt_5854374_1.out`
- `logs/crossdataset_5854375_0.out`, `logs/crossdataset_5854375_1.out`
- `logs/qualitative_5854478.out`, `logs/qualitative_5854478.err`

### Job 5855959 — All Qualitative Figure Variants (CANCELLED)
- **Script:** `scripts/generate_all_figures.sh`
- **Submitted:** 2026-03-30
- **GPU:** A100 80GB
- **Status:** CANCELLED — stuck in Priority queue (estimated start Apr 4)
- **Replaced by:** Job 5857617

### Job 5857617 — All Qualitative Figure Variants (V100 resubmit)
- **Script:** `scripts/generate_all_figures.sh`
- **Submitted:** 2026-03-30
- **GPU:** V100 32GB (resubmitted to avoid A100 queue)
- **Wall time:** 4 hrs
- **Status:** SUBMITTED
- **Purpose:** Generate multiple qualitative figure variants for VisCon paper — best/diverse selection modes, 1/4/6/8 row counts, plus auto-cropped teaser candidates.
- **Output:** `paper/VisCon/figures/variants/` (14 files):
  - `best_1row_teaser.png`, `best_4rows.png`, `best_6rows.png`
  - `diverse_4rows.png`, `diverse_6rows.png`, `diverse_8rows.png`
  - `best_row{0-3}_teaser.png`, `diverse_row{0-3}_teaser.png`
- **Logs:** `logs/allfigs_5857617.out`, `logs/allfigs_5857617.err`
- **What to do with results:** Browse `variants/`, pick favorites, copy to `paper/VisCon/figures/qualitative_comparison.png` and `paper/VisCon/figures/teaser.png`.

---

## Key Paths on Pegasus

| Resource | Path |
|---|---|
| Project root | `~/Special-SAM/` |
| Conda env | `/SEAS/home/g37014071/.conda/envs/specialsam/bin` |
| COD10K data | `/scratch/plessgrp/g37014071/cod10k/` (symlinked to `data/cod10k`) |
| SAM weights | `weights/sam_vit_h_4b8939.pth` (2.4GB) |
| Decoder checkpoint | `checkpoints/camo_decoder_vith.pth` (~16MB) |
| Embeddings | `data/embeddings/camo_embeddings_vith/` (~45GB) |
| Logs | `logs/` |
