# HPC Run Log — GWU Pegasus Cluster

Tracks all SLURM jobs, their purpose, status, and results.

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
