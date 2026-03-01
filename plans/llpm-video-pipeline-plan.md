# Plan: Implement Professor's Two Research Directions

## Context

Professor shared two novel research concepts for the Special-SAM paper (targeting CVPR workshop):

1. **Learnable Local Preprocessing Module (LLPM)** — A lightweight module before SAM's frozen encoder that enhances boundary/edge features at high resolution, helping SAM handle camouflaged objects
2. **Video-Based Camouflage Dataset Pipeline** — Use SAM's segmentation failures on wildlife video as an automatic proxy for camouflage difficulty labeling

Both are **novel** (no prior work combines these). Related work: MAdapter/Self-Prompt SAM add adapters *inside* the encoder; EMSAM's LFEM works at 1/4 resolution *alongside* the encoder. Nobody places a learnable module *before* the encoder, and nobody uses foundation model failure to auto-label camouflage.

---

## Part 1: Learnable Local Preprocessing Module (LLPM)

### Architecture (~31K params, expandable to ~500K)

```
Input: (B, 3, 1024, 1024) unnormalized RGB

Edge Branch:          Conv(3->32,3) -> BN -> ReLU -> Conv(32->32,3) -> BN -> ReLU -> Conv(32->1,1) -> Sigmoid
                      Output: edge_map (B, 1, 1024, 1024)

Enhancement Branch:   Conv(3->64,3) -> BN -> ReLU -> Conv(64->64,3,groups=4) -> BN -> ReLU
                      -> Conv(64->64,3,groups=4) -> BN -> ReLU -> Conv(64->3,1)
                      Output: enhancement (B, 3, 1024, 1024)

Fusion:               output = x + alpha * (enhancement * edge_map)
                      alpha = nn.Parameter(0.1)  # learnable scale, starts small
```

**Key design**: Edge branch learns *where* boundaries are; enhancement branch learns *what* features to emphasize; multiplication gates enhancement to boundary regions. Residual connection ensures stable training start.

### Training Pipeline Change

With LLPM, **pre-computed embeddings can't be used** — the encoder must run in the forward pass (frozen params, but gradient flows through input to LLPM):

```
Image -> LLPM (trainable) -> SAM Normalization -> SAM Encoder (frozen, grad flows through)
-> Prompt Encoder (frozen, no_grad) -> Mask Decoder (trainable) -> Loss -> Backprop
```

Memory on A100 80GB with fp16: ~8-9 GB total — fits comfortably.

### Files to Create

| File | Purpose |
|------|---------|
| `src/models/llpm.py` | LLPM module (EdgeBranch, EnhancementBranch, LLPM class) |
| `src/data/image_dataset.py` | `CamoImageDataset` — loads raw images (not embeddings) with masks & prompts |
| `src/training/train_llpm.py` | Training loop: LLPM + frozen encoder + decoder, mixed precision, cosine LR |
| `configs/train_llpm.yaml` | Config: epochs=15, lr=5e-5, edge_channels=32, enhance_channels=64 |
| `scripts/train_llpm.sh` | SLURM: A100, 64GB RAM, 12hr time limit |

### Files to Modify

| File | Change |
|------|--------|
| `src/models/sam_loader.py` | Add `load_llpm_sam()` — loads SAM + LLPM + optional decoder weights |
| `src/models/decoder.py` | Add `freeze_all_sam()`, `count_parameters()` utilities |
| `src/data/cod10k.py` | Add missing `CamoDataset` class (imported by train.py but undefined) -- DONE |
| `src/evaluation/evaluate.py` | Add LLPM-aware eval path with `predict_with_llpm()` helper |
| `configs/eval.yaml` | Add optional `llpm_path` and `llpm` config section |

### Critical Implementation Details

1. **Gradient flow**: Set `requires_grad=False` on encoder params but do NOT wrap encoder call in `torch.no_grad()` — PyTorch computes input gradients even for frozen layers
2. **SAM normalization**: LLPM operates on unnormalized images (0-255 float); apply `(x - model.pixel_mean) / model.pixel_std` AFTER LLPM, before encoder
3. **Eval bypass**: Can't use `SamPredictor.set_image()` (it normalizes internally); call encoder directly after LLPM + normalization
4. **Separate checkpoints**: Save LLPM and decoder as separate state_dicts for modularity

---

## Part 2: Video-Based Camouflage Dataset Pipeline

### Pipeline Stages

```
Stage 1: Video -> Frames        (cv2, 1 FPS, SSIM dedup)
Stage 2: Frames -> Detections   (Faster R-CNN from torchvision, COCO animal classes)
Stage 3: Detections -> Scores   (Run SAM with box prompts, multi-signal difficulty scoring)
Stage 4: Scores -> Dataset      (COD10K-format output with difficulty labels)
Stage 5: Analysis              (Distribution plots, validation stats)
```

### Difficulty Scoring (the core novelty)

Without ground truth masks, composite difficulty from 5 proxy signals:

| Signal | Weight | Low = SAM succeeded | High = SAM failed |
|--------|--------|--------------------|--------------------|
| SAM's self-reported IoU prediction | 0.30 | High confidence | Low confidence |
| Mask-to-bbox area ratio (30-90% = good) | 0.20 | Normal ratio | Too small/large |
| Boundary compactness (4*pi*A/P^2) | 0.15 | Compact shape | Fragmented |
| SAM logit confidence (mean in mask) | 0.20 | High logits | Low logits |
| Detector confidence | 0.15 | Easy to detect | Hard to detect |

`difficulty = 1.0 - weighted_mean(normalized_signals)`
- Easy: difficulty < 0.3 (SAM succeeds -> not camouflaged)
- Hard: difficulty > 0.7 (SAM fails -> camouflaged)

### Video Sources (for paper)

1. **MoCA dataset** (Moving Camouflaged Animals) — 141 video sequences, public, has GT for validation
2. **Snapshot Serengeti** (Zooniverse) — millions of camera trap frames from Tanzania, research-friendly
3. **YouTube CC-BY wildlife clips** — supplemental, verify licensing per-video

### Files to Create

| File | Purpose |
|------|---------|
| `src/pipeline/__init__.py` | Package init |
| `src/pipeline/video_extract.py` | Stage 1: Frame extraction with SSIM dedup |
| `src/pipeline/animal_detector.py` | Stage 2: Faster R-CNN detection, COCO animal classes |
| `src/pipeline/sam_oracle.py` | Stage 3: SAM scoring with composite difficulty |
| `src/pipeline/dataset_builder.py` | Stage 4: Build COD10K-format dataset from scored frames |
| `src/pipeline/difficulty_scorer.py` | Stage 5: Analysis and validation |
| `src/data/combined_dataset.py` | `CombinedCamoDataset` — merges COD10K + video data |
| `configs/pipeline.yaml` | Full pipeline config (extraction, detection, scoring, thresholds) |
| `configs/train_combined.yaml` | Training on combined COD10K + video dataset |
| `scripts/pipeline.sh` | SLURM: A100, 64GB, 12hr for full pipeline |
| `scripts/download_videos.sh` | Download MoCA / video sources |

### Files to Modify

| File | Change |
|------|--------|
| `src/training/train.py` | Support loading multiple dataset CSVs (concat DataFrames) |
| `src/evaluation/evaluate.py` | Add difficulty-stratified evaluation |

### Output Format

Frames saved as: `VIDCOD-{difficulty}-{video_id}-{animal_class}-{frame_idx}.{jpg,png}`
Directory structure mirrors COD10K: `Train/Image/`, `Train/GT_Object/`, `Test/Image/`, `Test/GT_Object/`
Train/test split by *video* (not frame) to prevent leakage.

---

## Implementation Order

### Phase 1: Fix Foundation (prerequisite for both)
1. Add missing `CamoDataset` class to `src/data/cod10k.py` -- DONE

### Phase 2: LLPM Module
2. Create `src/models/llpm.py` — EdgeBranch, EnhancementBranch, LLPM
3. Create `src/data/image_dataset.py` — CamoImageDataset
4. Create `src/training/train_llpm.py` — training loop with encoder in forward pass
5. Modify `src/models/sam_loader.py` — add `load_llpm_sam()`
6. Modify `src/models/decoder.py` — add utilities
7. Create `configs/train_llpm.yaml` and `scripts/train_llpm.sh`
8. Modify `src/evaluation/evaluate.py` — LLPM eval path

### Phase 3: Video Pipeline
9. Create `src/pipeline/` package with all 5 stage modules
10. Create `configs/pipeline.yaml`
11. Create `scripts/pipeline.sh` and `scripts/download_videos.sh`
12. Create `src/data/combined_dataset.py`
13. Create `configs/train_combined.yaml`

### Phase 4: Integration
14. Modify `src/training/train.py` — combined dataset support
15. Modify `src/evaluation/evaluate.py` — difficulty-stratified eval
16. Update `configs/eval.yaml` — LLPM + video dataset options

---

## Verification Plan

1. **LLPM unit test**: Create LLPM, pass random tensor through, verify output shape matches input, gradients flow
2. **LLPM training**: Run 1 epoch on COD10K subset (50 images) locally or on A100, verify loss decreases
3. **Video pipeline**: Process 1 short video clip through all 5 stages, verify COD10K-format output
4. **Combined training**: Train decoder on COD10K + small video dataset, verify convergence
5. **LLPM eval**: Run eval with LLPM checkpoint, compare metrics against base & decoder-only
6. **Full HPC run**: Submit SLURM jobs for LLPM training (12hr) and pipeline (12hr)
