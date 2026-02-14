# Decoder-Only Specialization Improves SAM's Prompt Robustness for Camouflaged Object Detection

**CVPR 2026 Workshop: Subtle Visual Computing**

## Abstract

<!--
Key points to cover:
- SAM is powerful but struggles with camouflaged objects
- Simple decoder-only fine-tuning (no architectural changes) dramatically improves performance
- Critical finding: improvement is robust across ALL prompt types, not just training prompts
- +184% mIoU on edge prompts (hardest case)
- Implications: lightweight specialization as a general strategy
-->

## 1. Introduction

<!--
- Camouflaged object detection (COD) as a challenging visual task
- SAM's zero-shot capabilities and limitations on subtle boundaries
- Research question: Can simple decoder fine-tuning make SAM robust for COD?
- Key insight: decoder specialization transfers across prompt types
- Contributions: (1) decoder-only pipeline, (2) prompt robustness analysis, (3) comprehensive COD evaluation
-->

## 2. Related Work

<!--
- SAM and foundation models for segmentation
- Camouflaged object detection methods (SINet, PFNet, etc.)
- Fine-tuning strategies for vision foundation models
- Prompt engineering for segmentation models
-->

## 3. Method

<!--
3.1 Background: SAM Architecture
- Image encoder (ViT-H), prompt encoder, mask decoder
- Why decoder-only: encoder representations are already strong

3.2 Decoder-Only Fine-Tuning
- Freeze image encoder + prompt encoder
- Train mask decoder with BCE + Dice loss
- Mixed prompt training (random point/box switching)
- Pre-computed embeddings for efficiency

3.3 Evaluation Protocol
- Multiple prompt strategies (center, edge, grid, random)
- Comprehensive metrics (IoU, Dice, S-alpha, E-phi, F-beta-w, MAE, Boundary F1)
-->

## 4. Experiments

<!--
4.1 Dataset: COD10K
- 6,000 train / 4,000 test across 78 categories
- Ground truth masks with object boundaries

4.2 Implementation Details
- SAM ViT-H backbone
- 7 epochs, AdamW lr=1e-4, batch size 1
- 2x augmentation (horizontal flip)
- Training on pre-computed embeddings

4.3 Evaluation Setup
- 4 prompt strategies × 2 models × 200 test images
- 8 evaluation metrics
-->

## 5. Results

<!--
5.1 Main Results Table
- Base vs Specialized across all prompt types and metrics
- Highlight: edge prompt improvement (+184% mIoU)

5.2 Prompt Robustness Analysis
- Specialized model improves on ALL prompt types
- Largest gains on hardest prompts (edge, random)
- Smallest gap on easiest prompts (center)

5.3 Qualitative Results
- Side-by-side visualizations
- Multi-point prompt visualizations
-->

## 6. Discussion

<!--
- Why decoder-only works: encoder already captures texture/boundary features
- Prompt robustness as an emergent property of decoder specialization
- Practical implications: lightweight fine-tuning for domain adaptation
- Limitations: single domain (COD), single backbone (ViT-H)
-->

## 7. Conclusion

<!--
- Simple decoder fine-tuning dramatically improves SAM for COD
- Key finding: prompt robustness improves across all prompt types
- Future work: multi-domain specialization, other architectures
-->

## References

<!-- To be filled -->
