# Special-SAM: Project Log

Decoder-only specialization of SAM ViT-H for camouflaged object detection.
Target venue: CVPR 2026 SVC Workshop (deadline: March 20, 2026).

---

## Current Status

Phase 1 (decoder-only baseline) is complete. All experiments run, paper drafted in LaTeX.
Phase 2 (LLPM module) is planned, implementation not yet started.

---

## Completed Work

### Decoder-Only Fine-Tuning Pipeline

- Froze SAM ViT-H image encoder (632M params) and prompt encoder
- Trained only the mask decoder (~4M params) on COD10K
- Pre-computed and cached encoder embeddings (3,040 images * 2 with flip = 6,080 samples, ~45GB)
- Mixed-prompt training: 50% random foreground point, 50% bounding box
- Loss: 0.5 BCE + 0.5 Dice
- Optimizer: AdamW, lr=1e-4, cosine annealing, 1-epoch warmup, 15 epochs
- Hardware: NVIDIA A100 80GB on GWU Pegasus cluster
- Training time: ~30 min (with cached embeddings)
- Decoder checkpoint: ~16MB

### Evaluation (15-Epoch Decoder, 2,026 Camouflaged Test Images)

| Prompt Strategy      | Base mIoU | Ours mIoU | Rel. Improvement |
|----------------------|-----------|-----------|------------------|
| Center-of-Mass (1pt) | 0.5724    | 0.7381    | +28.9%           |
| Edge (1pt)           | 0.2052    | 0.6938    | +238.1%          |
| Multi-Point Grid (4) | 0.6815    | 0.7556    | +10.9%           |
| Multi-Point Random(3)| 0.7293    | 0.7751    | +6.3%            |

Full metrics (specialized model, multi-point random -- best config):
- S-alpha: 0.874, E-phi: 0.960, F-beta-w: 0.862, MAE: 0.021, Boundary F1: 0.739

### Per-Category Breakdown (Center-of-Mass Prompt)

| Category    | N   | Base   | Ours  | Improvement |
|-------------|-----|--------|-------|-------------|
| Terrestrial | 699 | 0.519  | 0.703 | +35.5%      |
| Aquatic     | 474 | 0.563  | 0.720 | +27.9%      |
| Flying      | 714 | 0.610  | 0.767 | +25.6%      |
| Amphibian   | 124 | 0.680  | 0.825 | +21.4%      |

### Ablation Study (Loss Function, 500 Test Images, Center-of-Mass)

| Variant             | BCE  | Dice | mIoU   | S-alpha |
|---------------------|------|------|--------|---------|
| Dice-heavy          | 0.3  | 0.7  | 0.7285 | 0.8291  |
| BCE only            | 1.0  | 0.0  | 0.7134 | 0.8234  |
| Dice only           | 0.0  | 1.0  | 0.7123 | 0.8151  |
| BCE-heavy           | 0.7  | 0.3  | 0.7076 | 0.8175  |
| Ours (0.5/0.5, full)| 0.5  | 0.5  | 0.7381 | 0.8468  |

Finding: model is robust to loss weighting. Dice-heavy marginally best among ablations.

### Paper Artifacts

- LaTeX paper: `paper/latex/main.tex` (CVPR 2026 SVC template, anonymous review mode)
- Architecture diagram: `paper/figures/architecture_diagram.pdf`
- Qualitative figure: `paper/figures/qualitative_comparison.png` (12 examples, 3 difficulty tiers)
- Bibliography: `paper/latex/main.bib` (16 references)

### Comparison with Published Methods (COD10K Test Set)

| Method         | S-alpha | E-phi | F-beta-w | MAE   |
|----------------|---------|-------|----------|-------|
| SINet (CVPR20) | .776    | .864  | .631     | .043  |
| PFNet (CVPR21) | .800    | .877  | .660     | .040  |
| SINet-V2 (TP22)| .815    | .887  | .680     | .037  |
| SegMaR (CVPR22)| .833    | .899  | .724     | .034  |
| ZoomNet(CVPR22)| .838    | .911  | .729     | .029  |
| SAM-Adapter    | .883    | .918  | .801     | .025  |
| COMPrompter    | .889    | .949  | .821     | .023  |
| Ours (Grid)    | .861    | .951  | .841     | .024  |

Our method surpasses all dedicated COD architectures on E-phi and F-beta-w.
SAM-Adapter and COMPrompter have higher S-alpha (they modify the encoder).

---

## Next Phase: LLPM Implementation

Detailed plan: `plans/llpm-video-pipeline-plan.md`

### Learnable Local Preprocessing Module

A lightweight module (~31K-500K params) placed before SAM's frozen encoder.
Two branches: edge detection (where to enhance) and feature enhancement (what to enhance).
Gated fusion with residual connection.

Key change: pre-computed embeddings cannot be used. The encoder runs in the forward pass
(frozen weights, but gradients flow through the input back to LLPM).

### Files to Create

| File                          | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `src/models/llpm.py`         | LLPM module (edge + enhancement branches)        |
| `src/data/image_dataset.py`  | Raw image dataset (not embeddings)               |
| `src/training/train_llpm.py` | Training loop with encoder in forward pass       |
| `configs/train_llpm.yaml`    | Training config                                  |
| `scripts/train_llpm.sh`      | SLURM job script                                 |

### Files to Modify

| File                          | Change                                           |
|-------------------------------|--------------------------------------------------|
| `src/models/sam_loader.py`   | Add `load_llpm_sam()` function                   |
| `src/evaluation/evaluate.py` | Add LLPM-aware evaluation path                   |
| `configs/eval.yaml`          | Add optional `llpm_path` config                  |

### Video Pipeline (Future Work)

Deferred to after LLPM. Uses SAM failure as proxy for camouflage difficulty scoring.
Will be mentioned as future work in the paper.

---

## Repository Structure

```
Special-SAM/
  configs/
    eval.yaml              # Evaluation config
    train.yaml             # Decoder-only training config
    ablation/              # 4 loss function ablation configs
  src/
    data/
      cod10k.py            # Dataset loading, CamoDataset class
      transforms.py        # Image preprocessing
    models/
      sam_loader.py        # SAM model loading utilities
      decoder.py           # Decoder utilities
    training/
      train.py             # Decoder-only training loop
    evaluation/
      evaluate.py          # Full evaluation pipeline
      metrics.py           # COD metrics (S-alpha, E-phi, F-beta-w, MAE)
      prompt_strategies.py # 4 prompt strategies
  scripts/
    train.sh               # SLURM: decoder training
    eval.sh                # SLURM: full evaluation
    ablation.sh            # SLURM: ablation array job
    qualitative.sh         # SLURM: qualitative figure generation
    architecture_diagram.sh # SLURM: architecture diagram
    generate_qualitative_figures.py
    generate_architecture_diagram.py
    eval_ablation.py
    setup_evaluation.py
    update_paper_tables.py
  paper/
    latex/
      main.tex             # Primary paper (CVPR 2026 SVC template)
      main.bib             # Bibliography
      cvpr.sty             # CVPR style file
      ieeenat_fullname.bst # Bibliography style
    figures/
      architecture_diagram.pdf
      architecture_diagram.png
      qualitative_comparison.png
  logs/
    HPC_RUN_LOG.md         # Detailed SLURM job tracker
    eval_*.out             # Evaluation logs
    train_*.out            # Training logs
    ablation_*.out         # Ablation logs
  results/
    comprehensive_evaluation_results.csv
    per_category_results.csv
    ablation/              # Ablation result CSVs
  plans/
    llpm-video-pipeline-plan.md  # LLPM + video pipeline design
  checkpoints/             # Decoder weights (gitignored except metadata)
  weights/                 # SAM base weights (gitignored)
  data/                    # COD10K dataset (gitignored, symlinked on HPC)
```

---

## HPC Environment

| Resource         | Path / Value                                              |
|------------------|-----------------------------------------------------------|
| Cluster          | GWU Pegasus (SLURM)                                      |
| Login            | `ssh G37014071@pegasus.arc.gwu.edu`                      |
| Project root     | `~/Special-SAM/`                                          |
| Conda env        | `module load miniconda/23.11.0-2 && conda activate specialsam` |
| Python           | 3.12 (PyTorch 2.10, torchvision, segment-anything)       |
| GPU              | NVIDIA A100 80GB PCIe                                     |
| COD10K data      | `/scratch/plessgrp/g37014071/cod10k/` -> `data/cod10k`   |
| SAM weights      | `weights/sam_vit_h_4b8939.pth` (2.4GB)                   |
| Decoder ckpt     | `checkpoints/camo_decoder_vith.pth` (~16MB)               |
| Embedding cache  | `data/embeddings/camo_embeddings_vith/` (~45GB, 6080 files) |

Job history: see `logs/HPC_RUN_LOG.md`

---

## Key Decisions

1. Decoder-only fine-tuning (not full fine-tuning or adapter-based) -- preserves encoder representations, fast training, small checkpoint.
2. Pre-computed embeddings for decoder training -- eliminates encoder forward pass, reduces epoch time from hours to minutes.
3. Mixed-prompt training (50/50 point/box) -- produces prompt-robust decoder that generalizes to unseen prompt types.
4. COD10K as primary dataset -- largest public COD benchmark, standard evaluation protocol.
5. 15 epochs with cosine annealing -- loss decreased 49.4% (0.1417 to 0.0717), consistent improvement over 7-epoch baseline.

---

## Detailed Methodology

### Why Camouflaged Object Detection

SAM was trained on 1 billion masks from 11 million general images (SA-1B). It performs
well on typical segmentation tasks but struggles with camouflaged objects because:
- Foreground and background share similar color, texture, and edge structure
- Standard segmentation relies on clear foreground-background contrast
- Camouflaged objects are specifically evolved to defeat visual detection systems

COD10K is the standard benchmark for this problem: 10,000 images across 78 sub-categories
of camouflaged animals (aquatic, terrestrial, flying, amphibian). We chose it because it is
the largest public COD dataset and all published methods report numbers on it.

### Why Decoder-Only Fine-Tuning

SAM has three components: image encoder (ViT-H, 632M params), prompt encoder (small),
and mask decoder (~4M params). The question was which to adapt.

Options considered:
1. Full fine-tuning: too expensive (632M params), risk of catastrophic forgetting
2. Encoder adapters (LoRA, SAM-Adapter): adds parameters inside the encoder, requires
   architectural changes
3. Decoder-only: trains 0.6% of parameters, no architectural changes, fast iteration

We chose decoder-only because the hypothesis was that the ViT-H encoder, trained on 1B masks,
already produces feature representations rich enough to detect camouflaged boundaries. The
problem is that the decoder was trained on general segmentation and doesn't know how to
interpret subtle features as object boundaries in camouflage scenarios.

### Why Pre-Computed Embeddings

Running the ViT-H encoder on a 1024x1024 image takes ~0.5 seconds on an A100. With 6,080
training samples and 15 epochs, that would be ~12.7 hours of just encoder forward passes.

By caching the encoder output once (~45GB of .npy files, one per image), we eliminate the
encoder from the training loop entirely. Each training epoch processes only the lightweight
decoder, which runs in seconds. Total training time: ~30 minutes for 15 epochs.

Trade-off: augmentation is limited to transforms that can be applied to both the image and
the embedding identically (horizontal flip works because it can be applied to the 64x64
embedding). Color jitter, rotation, and scale changes cannot be used with pre-computed
embeddings because they would change what the encoder produces.

### Why Mixed-Prompt Training

During training, each sample randomly gets either:
- A single foreground point (random pixel inside the GT mask), or
- A bounding box (tight box around the GT mask)

with 50/50 probability. This was a deliberate choice to prevent the decoder from overfitting
to any single prompt type. The result was better than expected: the decoder became robust to
prompt types it never saw during training (center-of-mass, edge points, multi-point grids).

We hypothesize this works because mixed-prompt training forces the decoder to learn the
actual object structure rather than a mapping from prompt-location to nearby mask pixels.

### Why BCE + Dice Loss

Binary cross-entropy (BCE) provides pixel-level supervision: every pixel gets an independent
loss signal. This is good for precise boundaries. Dice loss provides region-level supervision:
it optimizes the overlap between predicted and ground truth regions. This is good for shape
coherence and handling class imbalance (camouflaged objects can be very small).

The 0.5/0.5 weighting was chosen as a balanced default. The ablation study confirmed that
the model is robust to loss weighting -- all four variants (BCE-only, Dice-only, BCE-heavy,
Dice-heavy) performed within 2.1 mIoU points of each other.

### Why 15 Epochs with Cosine Annealing

Initial experiments used 7 epochs with a flat learning rate. Results were good but the
loss curve had not plateaued. We switched to 15 epochs with:
- 1-epoch linear warmup (lr: 0 -> 1e-4)
- Cosine annealing decay (lr: 1e-4 -> ~2.2e-6 at epoch 15)

This produced consistent improvement: every evaluation metric improved from the 7-epoch
to the 15-epoch checkpoint, confirming no overfitting. Training loss decreased 49.4%
(0.1417 to 0.0717).

### Evaluation Protocol

We test four prompt strategies of increasing difficulty:
1. Center-of-Mass: single point at GT mask centroid (easy, deterministic)
2. Edge: single point on the GT mask boundary contour (hard, ambiguous)
3. Multi-Point Grid: 4 points in a grid pattern inside the bounding box (medium)
4. Multi-Point Random: 3 random foreground points (medium, stochastic)

For each strategy, we compute 8 metrics: mIoU, Dice, S-alpha, E-phi, F-beta-w, MAE,
Boundary Precision, Boundary F1. S-alpha, E-phi, F-beta-w, and MAE are the four standard
COD evaluation metrics used by all published methods.

All evaluations run on the full 2,026 camouflaged test images (not a subset). The 4,000
total test images include ~1,974 non-camouflaged images which are excluded because they
have empty GT masks and are not relevant to COD evaluation.

---

## Timeline and Issues

### 2026-02-28: Initial Setup and First Runs

Configured GWU Pegasus HPC cluster. Created conda environment `specialsam` with Python 3.12,
PyTorch 2.10, segment-anything, opencv, scipy. Symlinked COD10K dataset from scratch storage
to project directory.

Submitted first eval job (5785644) and first training job (5785646).

### 2026-02-28: Training Failure (CamoDataset Bug)

**Issue:** Training job 5785646 completed the embedding precompute step (6,080 embeddings
cached successfully, ~6 hours) but crashed when starting the training loop with:

```
ImportError: cannot import name 'CamoDataset' from 'src.data.cod10k'
```

**Root cause:** The `CamoDataset` class was defined in a Jupyter notebook during prototyping
but was never added to the `src/data/cod10k.py` module. The training script (`train.py`)
imported it but the class did not exist.

**Fix:** Added the `CamoDataset` class to `src/data/cod10k.py`. This is a simple PyTorch
Dataset that loads pre-computed embeddings (.npy), ground truth masks, and prompt coordinates
from a metadata CSV.

**Impact:** Lost ~6 hours of GPU time on the precompute step, but embeddings were cached
to disk so the re-run (job 5786340) skipped precompute and completed training in ~30 minutes.

### 2026-03-01: Successful Training and Evaluation

Job 5786340 completed 15-epoch training. Final loss: 0.0717.
Submitted three parallel jobs:
- 5786512: Full evaluation with new decoder (~3 hours)
- 5786513: Qualitative figure generation (~18 minutes)
- 5786514: Ablation study (4 array tasks, each ~1.5 hours)

### 2026-03-01: Git Push Conflict

**Issue:** Pushed code changes from local machine while HPC had its own commits (log files).
Git push was rejected due to divergent branches.

**Fix:** `git pull --no-rebase` to merge, then `git push`. No data loss.

### 2026-03-01: Gitignore Blocking Results

**Issue:** `results/` and `*.csv` were in `.gitignore`, preventing result CSVs from being
tracked in git.

**Fix:** Added exception rules to `.gitignore`:
```
!results/*.csv
!results/tables/
!results/ablation/
!logs/
!logs/**
!paper/figures/*.png
!paper/figures/*.jpg
```

### 2026-03-01: 2,026 vs 4,000 Sample Count

**Issue:** Evaluation reported 2,026 samples instead of the expected 4,000 test images.
Initial concern that something was wrong.

**Explanation:** This is correct and standard. COD10K test set has 4,000 total images, but
only 2,026 are camouflaged (prefix COD10K-CAM-*). The remaining ~1,974 are non-camouflaged
(COD10K-NonCAM-*) with empty ground truth masks. All published COD methods evaluate on the
camouflaged subset only. The evaluation script processes all 4,000 images but only computes
metrics on the 2,026 that have non-empty foreground masks.

### 2026-03-01 to 03-02: All Jobs Complete

All HPC jobs completed successfully. Results committed and pushed.
LaTeX paper written with all tables, figures, and updated numbers.

### 2026-03-02: Cleanup

Consolidated 8 obsolete markdown files into single PROJECT_LOG.md. Fixed several
inaccuracies in the paper:
- Training set is 3,040 camouflaged images (not 6,000 total)
- Augmented to 6,080 (not 12,000)
- Training used random foreground points (not center-of-mass)
- PFNet venue was CVPR (not AAAI)
- Training time is <30 min (not <2 hours)

---

## Data Provenance

Every number in the paper traces back to a specific file in this repository:

| Paper Section | Claim | Source File | How to Verify |
|---|---|---|---|
| Abstract | mIoU 57.24% -> 73.81% | `results/comprehensive_evaluation_results.csv` | Row 1 (Base CoM) and Row 5 (Ours CoM), column iou_mean |
| Abstract | +238.1% edge improvement | Same CSV | (0.6938 - 0.2052) / 0.2052 = 2.381 |
| Abstract | +6.3% multi-point random | Same CSV | (0.7751 - 0.7293) / 0.7293 = 0.063 |
| Table 1 | All 8 rows, 8 columns | `results/comprehensive_evaluation_results.csv` | Direct mapping, rounded to 3 decimal places |
| Table 2 | Improvement percentages | Computed from Table 1 | (Ours - Base) / Base |
| Table 3 | Per-category mIoU | `results/per_category_results.csv` | Filter by Center-of-Mass strategy |
| Table 4 | Published method numbers | Referenced papers [6,7,8,9,10,11,12] | From published tables in cited papers |
| Table 5 | Ablation loss variants | `logs/ablation_5786514_*.out` | Parsed from evaluation output lines |
| Training loss | 0.1417 -> 0.0717 | `logs/train_5786340.out` | Epoch 1 and Epoch 15 lines |
| Training samples | 6,080 | `logs/train_5786340.out` | "Training for 15 epochs on 6080 samples" |
| Training time | ~30 min | `logs/train_5786340.out` | Timestamps: start to finish |
| Qualitative figure | 12 examples | `paper/figures/qualitative_comparison.png` | Generated by `scripts/generate_qualitative_figures.py` |
| Qualitative stats | 535 hard, 480 medium, 1011 easy | `logs/qualitative_5786513.out` | "Candidate pool:" line |

Published method numbers in Table 4 are taken from the respective papers' reported results
on the COD10K test set. These are standard benchmark numbers reproduced across multiple
publications. Sources: SINet [6], PFNet [7], SINet-V2 [14], SegMaR [16], ZoomNet [8],
SAM-Adapter [4], COMPrompter [5].

---

## Known Limitations

- Evaluation only on COD10K; cross-dataset generalization (CAMO, CHAMELEON, NC4K) not tested.
- Only ViT-H backbone; smaller backbones not evaluated.
- Requires user-provided prompts (not fully automatic like dedicated COD methods).
- Methods that modify the encoder (SAM-Adapter, COMPrompter) achieve higher S-alpha.
- No multi-seed evaluation; results are from single training run.
