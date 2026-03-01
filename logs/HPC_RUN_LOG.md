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
- **Status:** RUNNING
- **Config:** `configs/eval.yaml` (max_samples=0, all test images)
- **Decoder:** `checkpoints/camo_decoder_vith.pth` (15-epoch, cosine LR)
- **Expected output:** `results/comprehensive_evaluation_results.csv`, `results/per_category_results.csv`
- **Expected log:** `logs/eval_5786512.out`
- **Purpose:** Evaluate improved decoder; expect better numbers than job 5785644

---

### Job 5786513 — Qualitative Figure Generation
- **Script:** `scripts/qualitative.sh`
- **Submitted:** 2026-03-01
- **GPU:** A100 80GB
- **Status:** RUNNING
- **Config:** `configs/eval.yaml`
- **Expected output:** `paper/figures/qualitative_comparison.png`
- **Expected log:** `logs/qualitative_5786513.out`
- **Purpose:** Generate 12-example comparison grid (4 hard, 4 medium, 4 easy)

---

### Job 5786514 — Ablation Study (loss function variants)
- **Script:** `scripts/ablation.sh` (SLURM array job, 4 tasks)
- **Submitted:** 2026-03-01
- **GPU:** A100 80GB x4 (parallel array)
- **Status:** RUNNING
- **Configs:**
  - Task 0: `configs/ablation/bce_only.yaml` (BCE=1.0, Dice=0.0)
  - Task 1: `configs/ablation/dice_only.yaml` (BCE=0.0, Dice=1.0)
  - Task 2: `configs/ablation/bce_heavy.yaml` (BCE=0.7, Dice=0.3)
  - Task 3: `configs/ablation/dice_heavy.yaml` (BCE=0.3, Dice=0.7)
- **Expected output:** `checkpoints/ablation/camo_decoder_*.pth`, `results/ablation/*_results.csv`
- **Expected log:** `logs/ablation_5786514_*.out`
- **Purpose:** Show which loss components matter; paper ablation table

---

## Embedding Cache

- **Path:** `data/embeddings/camo_embeddings_vith/`
- **Created by:** Job 5785646 (precompute step)
- **Samples:** 6,080 (6,000 images x2 with horizontal flip augmentation)
- **Metadata:** `data/embeddings/camo_train_meta_vith.csv`
- **Size:** ~45GB
- **Reused by:** All training jobs (train.sh skips precompute if dir exists)

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
