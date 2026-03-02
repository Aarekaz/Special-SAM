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
| PFNet (AAAI21) | .800    | .877  | .660     | .040  |
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

## Known Limitations

- Evaluation only on COD10K; cross-dataset generalization (CAMO, CHAMELEON, NC4K) not tested.
- Only ViT-H backbone; smaller backbones not evaluated.
- Requires user-provided prompts (not fully automatic like dedicated COD methods).
- Methods that modify the encoder (SAM-Adapter, COMPrompter) achieve higher S-alpha.
- No multi-seed evaluation; results are from single training run.
