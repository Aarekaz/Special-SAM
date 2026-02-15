# Paper Evolution Plan: Special-SAM -> Publication-Ready Research

## Goal
Evolve the current CVPR workshop draft into a rigorous, technically sound paper with honest analysis and strong experimental backing.

---

## Phase 1: Literature Deep Dive (1-2 days)

### 1.1 COD-Specific Papers (Top Priority)
- [ ] SINet-v2 (Fan et al., 2022) — Search, Identify, and segment
- [ ] ZoomNet (Pang et al., 2022) — Scale-aware COD
- [ ] FEDER (He et al., 2023) — Feature Decomposition and Erasure
- [ ] HitNet (Hu et al., 2023) — Hierarchical Interactive Network
- [ ] DGNet (Ji et al., 2023) — Deep Gradient Network
- [ ] FSPNet (Huang et al., 2023) — Frequency-Spatial Prior
- [ ] EVP (Liu et al., 2023) — Explicit Visual Prompting
- [ ] SAM-Adapter (Chen et al., 2024) — Adapter-based SAM fine-tuning
- [ ] SAM-HQ (Ke et al., 2024) — High-Quality SAM masks

### 1.2 SAM Adaptation Papers
- [ ] Medical SAM Adapter — decoder-only in medical imaging
- [ ] PerSAM — Personalized SAM with one-shot
- [ ] Matcher — SAM for few-shot segmentation
- [ ] RSPrompter — SAM for remote sensing
- [ ] SAMed — SAM for medical image segmentation

### 1.3 Deliverable
- Build a comparison table: Method | Backbone | Params Trained | COD10K mIoU | CAMO mIoU | NC4K mIoU | Key Technique
- Identify our unique positioning (decoder-only, prompt robustness, failure analysis)

---

## Phase 2: Strengthen Experiments (3-5 days)

### 2.1 Critical Fixes (Must Do)
- [ ] **Full test set evaluation**: Run on all 4,000 COD10K test images (not just 200)
- [ ] **Add standard COD metrics properly**: S-alpha, E-phi, F-beta-w, MAE (with proper implementations matching COD papers)
- [ ] **Statistical significance**: Run 3-5 seeds, report mean +/- std

### 2.2 Baseline Comparisons (Must Do)
- [ ] Base SAM ViT-H (already have)
- [ ] Base SAM ViT-B (smaller model comparison)
- [ ] SAM-Adapter (if code available)
- [ ] At least 2 dedicated COD methods (use reported numbers from their papers on COD10K)

### 2.3 Ablation Studies (Must Do)
- [ ] Loss function: BCE-only vs Dice-only vs BCE+Dice (vary weights)
- [ ] Prompt strategy: Point-only vs Box-only vs Mixed training
- [ ] Training epochs: 3, 5, 7, 10, 15 (find sweet spot vs overfitting)
- [ ] Learning rate: 1e-5, 5e-5, 1e-4, 5e-4
- [ ] Data augmentation: None vs HFlip vs HFlip+VFlip+ColorJitter
- [ ] Model capacity: MobileSAM vs ViT-B vs ViT-L vs ViT-H decoder

### 2.4 Cross-Dataset Generalization (Should Do)
- [ ] CAMO dataset (250 test images) — train on COD10K, test on CAMO
- [ ] CHAMELEON dataset (76 images) — zero-shot transfer
- [ ] NC4K dataset (4,121 images) — largest COD benchmark

### 2.5 Advanced Experiments (Nice to Have)
- [ ] Multi-scale inference (resize to 3 scales, merge predictions)
- [ ] Ensemble prompt strategies at inference time
- [ ] LoRA fine-tuning comparison (decoder + LoRA in encoder)
- [ ] Progressive unfreezing (train decoder, then unfreeze last encoder blocks)

---

## Phase 3: Code Improvements (2-3 days)

### 3.1 Training Pipeline
- [ ] Add proper validation split (80/20 from training set)
- [ ] Add early stopping based on validation loss
- [ ] Add learning rate scheduling (cosine annealing or step decay)
- [ ] Add proper logging (wandb or tensorboard)
- [ ] Support multi-GPU training (DataParallel at minimum)
- [ ] Add seed setting for full reproducibility
- [ ] Increase batch size > 1 with gradient accumulation

### 3.2 Evaluation Pipeline
- [ ] Evaluate on FULL test set (not 200 samples)
- [ ] Add per-category evaluation (aquatic vs terrestrial vs flying)
- [ ] Add qualitative failure case analysis
- [ ] Generate publication-quality figures (matplotlib with proper fonts/sizing)
- [ ] Add inference speed benchmarking (FPS, latency)

### 3.3 Code Quality
- [ ] Add type hints throughout
- [ ] Add docstrings to all public functions
- [ ] Add unit tests for metrics computation
- [ ] Add CLI entry points (argparse or click)
- [ ] requirements.txt with pinned versions

---

## Phase 4: Rewrite Paper (2-3 days)

### 4.1 Reframe the Contribution
**Option A (Recommended):** "Decoder-only fine-tuning as a practical specialization strategy"
- Emphasize: minimal parameters trained, fast training, prompt robustness
- Position against: full fine-tuning, adapter methods, dedicated architectures
- Strength: practical value + systematic study of when it works/fails

**Option B:** "When and why SAM specialization works"
- Emphasize: systematic study across 3 domains (wood, carpet, camo)
- The failures are as informative as the successes
- Derive conditions/guidelines for practitioners

**Option C:** "Prompt robustness through mixed-prompt decoder training"
- Emphasize: the +184% edge prompt improvement
- Novel finding: robustness transfers across prompt types
- Could be a focused, tight paper

### 4.2 Paper Structure (Revised)
1. **Abstract** — Tighten to 150 words, lead with the key finding
2. **Introduction** — 1 page, clear problem statement, specific contributions (3 bullet points)
3. **Related Work** — 0.75 pages, organized: (a) COD methods, (b) SAM adaptations, (c) efficient fine-tuning
4. **Method** — 1 page with formal math notation
   - Problem formulation with equations
   - Architecture diagram (proper figure)
   - Loss function with LaTeX equations
   - Training protocol (algorithm box)
5. **Experiments** — 2 pages
   - Setup (datasets, metrics, baselines, implementation details)
   - Main results table (vs baselines)
   - Ablation studies (2-3 tables)
   - Cross-dataset generalization
   - Prompt robustness analysis
6. **Discussion** — 0.5 pages
   - When specialization works vs fails (with the 3-experiment evidence)
   - Limitations (honest)
   - Practical guidelines
7. **Conclusion** — 0.25 pages

### 4.3 Figures Needed
- [ ] Fig 1: Architecture diagram (SAM with frozen encoder, trained decoder)
- [ ] Fig 2: Training loss curves (all 3 experiments overlaid)
- [ ] Fig 3: Qualitative results grid (6-8 examples: input, GT, base SAM, specialized)
- [ ] Fig 4: Prompt robustness bar chart (base vs specialized per prompt type)
- [ ] Fig 5: Failure cases (honest — where specialized model still struggles)
- [ ] Fig 6 (optional): Per-category performance breakdown

### 4.4 Tables Needed
- [ ] Table 1: Main results — our method vs baselines on COD10K (S_alpha, E_phi, F_beta_w, MAE)
- [ ] Table 2: Cross-dataset generalization (CAMO, CHAMELEON, NC4K)
- [ ] Table 3: Ablation — loss function variants
- [ ] Table 4: Ablation — prompt strategy variants
- [ ] Table 5: Parameter efficiency comparison (params trained vs performance)

---

## Phase 5: Iterate and Polish (ongoing)

### 5.1 Result-Driven Architecture Improvements
Based on Phase 2 results, pursue the most promising:
- If gap to SOTA is in boundaries: Add boundary-aware loss (boundary IoU)
- If gap is in small objects: Add multi-scale inference
- If gap is in texture: Add texture-aware augmentation
- If gap is overall: Try LoRA in encoder + decoder fine-tuning

### 5.2 Writing Polish
- [ ] Get feedback from advisor/colleagues on framing
- [ ] Proofread for technical accuracy
- [ ] Verify all numbers match code output exactly
- [ ] Check formatting against venue requirements (CVPR workshop template)
- [ ] Ensure reproducibility: all hyperparameters documented

### 5.3 Submission Checklist
- [ ] Paper formatted in correct template
- [ ] Supplementary material prepared (additional results, code link)
- [ ] Code cleaned and documented for release
- [ ] Anonymous GitHub repo or supplementary zip
- [ ] All figures are vector graphics (PDF) where possible
- [ ] References are complete and correctly formatted

---

## Timeline Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 1 | Literature review + full test eval | Comparison table, 4K results |
| Week 2 | Baselines + ablations | Complete experimental tables |
| Week 3 | Cross-dataset + code cleanup | Generalization results |
| Week 4 | Paper rewrite | Complete draft v2 |
| Week 5 | Architecture improvements | Updated results |
| Week 6 | Polish + submission | Final paper |

---

## Key Metrics to Beat (from Literature)

Target performance on COD10K test set (approximate SOTA):
- S-alpha: 0.82-0.86
- E-phi: 0.88-0.92
- F-beta-w: 0.75-0.80
- MAE: 0.030-0.040
- mIoU: ~0.78-0.82

Our current (200 samples): mIoU ~0.66, so there's a meaningful gap to close.

---

## Honest Assessment

**Strengths we should lean into:**
1. Parameter efficiency (4M/636M = 0.6% of params trained)
2. Prompt robustness finding (+184% on edge prompts)
3. Systematic study across 3 domains with failure analysis
4. Practical: 2-hour training, no architecture changes needed

**Weaknesses we must address:**
1. Raw performance vs dedicated COD methods (likely 10-15% gap)
2. Single dataset evaluation (currently)
3. No baselines beyond vanilla SAM
4. Limited ablation studies
5. Small evaluation subset (200/4000)

**Our best positioning:**
"We don't claim SOTA. We show that decoder-only fine-tuning is a surprisingly effective, practical strategy that creates emergent prompt robustness — and we provide the first systematic study of when SAM specialization succeeds vs fails."
