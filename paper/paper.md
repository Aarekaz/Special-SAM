# Decoder-Only Specialization Improves SAM's Prompt Robustness for Camouflaged Object Detection

**CVPR 2026 Workshop: Subtle Visual Computing**

*Anurag Dhungana and Prakriti Bista*

## Abstract

The Segment Anything Model (SAM) achieves strong zero-shot segmentation on general images but struggles with camouflaged objects, where foreground and background share similar color and texture. We show that simple decoder-only fine-tuning — freezing SAM's image encoder and prompt encoder while training only the lightweight mask decoder — dramatically improves performance on camouflaged object detection (COD). Using the COD10K dataset with SAM ViT-H, our approach improves mean IoU from 57.24% to 71.38% (+14.14 points) with center-of-mass prompts on the full 2,026-image camouflaged test set. Critically, this improvement transfers across prompt types: edge-point prompts, the hardest case where base SAM achieves only 20.52% mIoU, improve to 65.63% (+219.8% relative gain). The specialized model improves on all four tested prompt strategies, including multi-point random prompts (+2.0%). We attribute this prompt robustness to our mixed-prompt training strategy, which randomly alternates between point and bounding box prompts. Our method requires no architectural modifications, trains in under 2 hours on a single GPU using pre-computed embeddings, and produces a decoder checkpoint of only ~16MB. These results suggest that decoder-only specialization is a practical, lightweight strategy for adapting foundation segmentation models to challenging visual domains.

## 1. Introduction

Camouflaged object detection (COD) presents a fundamental challenge for visual perception systems. Camouflaged organisms — insects mimicking bark, fish blending into coral, lizards matching rock surfaces — have evolved over millions of years to defeat visual detection. These objects exhibit low contrast against their backgrounds, share complex textures with their surroundings, and span wide ranges of scale and occlusion.

The Segment Anything Model (SAM) [1] represents a major advance in promptable segmentation, trained on over 1 billion masks across 11 million images. SAM achieves impressive zero-shot segmentation on general images given point, box, or mask prompts. However, SAM's broad training makes it robust but shallow on tasks requiring fine-grained boundary detection in low-contrast scenes. On COD10K [2], SAM ViT-H achieves only 57.24% mIoU with center-of-mass prompts — adequate for general segmentation but insufficient for the fine-grained boundaries required in camouflage scenarios. The core issue is that SAM relies heavily on edge contrast: when a green lizard sits on a green leaf with matching color, SAM perceives a single continuous object.

We ask: **can simple decoder-only fine-tuning make SAM robust for camouflaged object detection?** Our approach freezes the heavy image encoder (ViT-H, ~632M parameters) and prompt encoder, training only the lightweight mask decoder (~4M parameters). This preserves SAM's learned visual representations while teaching the decoder to interpret subtle texture disruptions as object boundaries.

Our key finding is that this specialization is **prompt-robust**: the fine-tuned decoder improves segmentation quality across all tested prompt strategies — center-of-mass, edge points, multi-point grids, and random points — even though training used only center points and bounding boxes. The largest gains appear on the hardest prompts (edge points: +184% relative improvement), suggesting that decoder specialization teaches the model to better understand camouflaged object structure rather than merely memorizing prompt-mask associations.

**Contributions:**
1. A decoder-only fine-tuning pipeline for SAM that achieves +14.14 point mIoU improvement on COD10K with no architectural changes, evaluated on the full camouflaged test set (2,026 images).
2. A prompt robustness analysis showing that decoder specialization transfers across all four tested prompt types, with the largest gains on the hardest prompts (+219.8% relative on edge prompts).
3. Comprehensive evaluation using both standard segmentation metrics (IoU, Dice, F1) and standard COD metrics (S-alpha, E-phi, F-beta-w, MAE, Boundary F1), with per-category analysis across aquatic, terrestrial, flying, and amphibian camouflage types.

## 2. Related Work

**Segment Anything Model.** SAM [1] introduced a promptable segmentation paradigm trained on the SA-1B dataset. Its architecture separates image encoding (ViT backbone), prompt encoding (sparse/dense), and mask decoding. SAM2 [3] extended this to video. Several works have explored adapting SAM to specialized domains: SAM-Adapter [4] adds learnable adapter modules to the image encoder, while COMPrompter [5] reconceptualizes SAM with multi-prompt networks for COD. Our approach differs by requiring no architectural modifications — we fine-tune only the existing decoder weights.

**Camouflaged Object Detection.** COD has attracted significant attention with dedicated architectures. SINet [6] uses a search-and-identify framework, PFNet [7] employs positioning and focus modules, and ZoomNet [8] uses mixed-scale learning. These methods design specialized architectures from scratch. In contrast, we leverage SAM's pre-trained representations and demonstrate that decoder-only adaptation can achieve competitive improvements without domain-specific architectural design.

**Fine-Tuning Foundation Models.** Parameter-efficient fine-tuning has become standard for adapting large models. LoRA [9] adds low-rank adapters, while prompt tuning [10] learns continuous prompts. For SAM specifically, Liu et al. [11] proposed dual-stream adapters for COD. Our decoder-only approach is simpler: we freeze all parameters except the existing mask decoder, requiring no additional modules or parameters.

## 3. Method

### 3.1 Background: SAM Architecture

SAM consists of three components: (1) an **image encoder** (ViT-H, ~632M parameters) that produces image embeddings, (2) a **prompt encoder** that maps points, boxes, or masks to sparse and dense embeddings, and (3) a **mask decoder** (~4M parameters) that combines image and prompt embeddings to predict segmentation masks.

The image encoder is the computational bottleneck, processing each 1024×1024 image through a large vision transformer. The mask decoder is lightweight by comparison, consisting of a two-layer transformer with cross-attention between prompt tokens and image embeddings, followed by an MLP that produces per-pixel mask logits.

### 3.2 Decoder-Only Fine-Tuning

Our training procedure is:

1. **Freeze encoders.** All parameters of the image encoder and prompt encoder are frozen. This preserves SAM's learned visual representations and prevents catastrophic forgetting.

2. **Pre-compute embeddings.** We run the frozen image encoder once on all training images and cache the resulting embeddings as `.npy` files. This eliminates the encoder forward pass from the training loop, reducing per-epoch training time from hours to minutes.

3. **Augment with horizontal flips.** Each training image is augmented with a horizontal flip, doubling the effective dataset from 6,000 to 12,000 samples. Both the image embedding and ground truth mask are flipped consistently.

4. **Train with mixed prompts.** During each training iteration, we randomly select between a point prompt (random foreground pixel) and a bounding box prompt (tight box around the ground truth) with equal probability. This forces the decoder to learn object structure rather than becoming dependent on any single prompt type.

5. **Combined loss function.** We optimize a weighted combination of binary cross-entropy (BCE) and Dice loss:

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$$

BCE provides pixel-level supervision that enforces precise boundaries, while Dice loss provides region-level supervision that maintains shape coherence. This combination proved critical — in preliminary experiments using Dice loss alone, models either overfitted or showed limited improvement.

6. **Optimization.** We use AdamW with learning rate 1×10⁻⁴ for 7 epochs, batch size 1. Training produces stable, gradual loss reduction (0.1420 → 0.1059, 25.4% total reduction) with no instability spikes.

### 3.3 Evaluation Protocol

We evaluate using four prompt strategies of increasing difficulty:

- **Center-of-Mass (1 point):** Single point at the centroid of the ground truth mask. The easiest prompt — it gives the model the object's approximate center.
- **Edge (1 point):** Single point sampled from the object boundary contour. Challenging because edge points provide less spatial context and sit at the ambiguous boundary between foreground and background.
- **Multi-Point Grid (4 points):** Four points arranged in a grid within the object bounding box, filtered to foreground pixels. Provides spatial coverage but may include points near boundaries.
- **Multi-Point Random (3 points):** Three randomly sampled foreground points. Tests robustness to arbitrary prompt placement.

We report standard segmentation metrics (IoU, Dice, F1, Boundary Precision/Recall/F1) and four standard COD evaluation metrics:

- **S-alpha** (Structure measure) [12]: Combines object-aware and region-aware structural similarity.
- **E-phi** (Enhanced alignment measure) [13]: Captures both pixel-level and image-level alignment.
- **F-beta-w** (Weighted F-measure) [14]: Distance-weighted precision and recall.
- **MAE** (Mean Absolute Error): Pixel-level prediction error.

## 4. Experiments

### 4.1 Dataset: COD10K

We use COD10K-v3 [2], the largest camouflaged object detection dataset, containing 6,000 training and 4,000 test images across 78 camouflage categories spanning aquatic (fish, octopuses, seahorses), terrestrial (insects, lizards, frogs), and flying (moths, butterflies) animals. Each image has a corresponding pixel-level binary ground truth mask. The dataset is challenging due to:

- **Low contrast:** Camouflaged animals share color distributions with their backgrounds.
- **Complex textures:** Natural environments contain high-frequency patterns that mask object boundaries.
- **Scale variation:** Subjects range from small insects to large reptiles.
- **Partial occlusion:** Many animals are partially hidden behind foliage or terrain.

### 4.2 Implementation Details

- **Backbone:** SAM ViT-H (sam_vit_h_4b8939.pth, ~2.4GB checkpoint, ~636M parameters)
- **Training data:** 6,000 COD10K training images, augmented to 12,000 with horizontal flips
- **Pre-computed embeddings:** Cached as `.npy` files (~45GB total), computed once in ~2 hours on a single V100
- **Training:** 7 epochs, AdamW optimizer, lr=1×10⁻⁴, batch size 1, mixed point/box prompts
- **Loss:** 50% BCE + 50% Dice
- **Output:** Decoder state dict checkpoint (~16MB)
- **Evaluation:** Full COD10K camouflaged test set (2,026 images), 4 prompt strategies, 8+ metrics, per-category analysis
- **Seed:** 42 for reproducibility across random, NumPy, and PyTorch

### 4.3 Training Dynamics

Training showed stable, monotonic loss decrease — a key indicator of healthy generalization:

| Epoch | Loss   | Reduction |
|-------|--------|-----------|
| 1     | 0.1420 | —         |
| 2     | 0.1272 | -10.4%    |
| 3     | 0.1172 | -7.9%     |
| 4     | 0.1140 | -2.7%     |
| 5     | 0.1113 | -2.4%     |
| 6     | 0.1081 | -2.9%     |
| 7     | 0.1059 | -1.8%     |

The final loss of 0.1059 indicates the model is still generalizing rather than memorizing. This stands in contrast to overfitting scenarios where loss collapses below 0.01.

## 5. Results

### 5.1 Main Results

**Table 1: Comprehensive evaluation across prompt strategies (2,026 camouflaged test images)**

| Model | Prompt Strategy | mIoU | Dice | S-alpha | E-phi | F-beta-w | MAE | Boundary F1 |
|-------|----------------|------|------|---------|-------|----------|-----|-------------|
| Base SAM ViT-H | Center-of-Mass (1pt) | 0.5724 | 0.6545 | 0.7426 | 0.8301 | 0.7057 | 0.0677 | 0.5021 |
| Base SAM ViT-H | Edge (1pt) | 0.2052 | 0.2510 | 0.4811 | 0.5974 | 0.2412 | 0.1631 | 0.2814 |
| Base SAM ViT-H | Multi-Point Grid (4pt) | 0.6815 | 0.7613 | 0.8120 | 0.8973 | 0.8017 | 0.0459 | 0.6001 |
| Base SAM ViT-H | Multi-Point Random (3pt) | 0.7303 | 0.8165 | 0.8312 | 0.9297 | 0.8322 | 0.0433 | 0.6521 |
| **Specialized SAM** | **Center-of-Mass (1pt)** | **0.7138** | **0.7977** | **0.8353** | **0.9325** | **0.8157** | **0.0305** | **0.6725** |
| **Specialized SAM** | **Edge (1pt)** | **0.6563** | **0.7417** | **0.7900** | **0.9093** | **0.7322** | **0.0364** | **0.6307** |
| **Specialized SAM** | **Multi-Point Grid (4pt)** | **0.7227** | **0.8071** | **0.8429** | **0.9386** | **0.8258** | **0.0281** | **0.6811** |
| **Specialized SAM** | **Multi-Point Random (3pt)** | **0.7449** | **0.8306** | **0.8577** | **0.9484** | **0.8448** | **0.0250** | **0.7030** |

### 5.2 Prompt Robustness Analysis

The most striking result is the **uniformity of improvement across all prompt types**. While base SAM shows high sensitivity to prompt quality (mIoU ranges from 0.205 to 0.730 depending on strategy), the specialized model is remarkably stable (0.656 to 0.745). Critically, the specialized model improves on **all four** strategies, including multi-point random where it gains +2.0%.

**Table 2: Improvement analysis by prompt type**

| Prompt Type | Base mIoU | Specialized mIoU | Absolute Gain | Relative Gain |
|-------------|-----------|-------------------|---------------|---------------|
| Center-of-Mass | 0.5724 | 0.7138 | +0.1414 | +24.7% |
| Edge (Single) | 0.2052 | 0.6563 | +0.4511 | **+219.8%** |
| Multi-Point Grid | 0.6815 | 0.7227 | +0.0412 | +6.0% |
| Multi-Point Random | 0.7303 | 0.7449 | +0.0146 | +2.0% |

The **+219.8% relative improvement on edge prompts** is the headline finding. Edge prompts are the hardest because they place the prompt point at the ambiguous boundary between camouflaged object and background. Base SAM produces nearly random masks (20.52% mIoU) from edge prompts, while our specialized decoder achieves 65.63% — comparable to its performance with easier prompt types.

This suggests that decoder specialization doesn't merely improve mask quality for a given prompt — it fundamentally changes how the model resolves ambiguity at camouflaged boundaries, making it robust to where the prompt is placed.

### 5.3 Detailed Metrics

**Table 3: Full metrics for specialized model across prompt strategies**

| Prompt Strategy | mIoU | Dice | S-alpha | E-phi | F-beta-w | MAE | Boundary Prec | Boundary Rec | Boundary F1 |
|----------------|------|------|---------|-------|----------|-----|--------------|-------------|-------------|
| Center-of-Mass | 0.7138 | 0.7977 | 0.8353 | 0.9325 | 0.8157 | 0.0305 | 0.7302 | 0.6655 | 0.6725 |
| Edge (Single) | 0.6563 | 0.7417 | 0.7900 | 0.9093 | 0.7322 | 0.0364 | 0.6602 | 0.6599 | 0.6307 |
| Multi-Point Grid | 0.7227 | 0.8071 | 0.8429 | 0.9386 | 0.8258 | 0.0281 | 0.7332 | 0.6790 | 0.6811 |
| Multi-Point Random | 0.7449 | 0.8306 | 0.8577 | 0.9484 | 0.8448 | 0.0250 | 0.7478 | 0.7056 | 0.7030 |

Boundary precision is consistently high across strategies (0.660–0.748), and boundary recall ranges from 0.660–0.706, indicating that the specialized decoder reliably recovers ground truth boundaries regardless of prompt type.

### 5.4 Per-Category Analysis

Performance varies across camouflage super-categories, revealing where specialization helps most:

**Table 4: Per-category mIoU comparison (Center-of-Mass prompt)**

| Category | Samples | Base mIoU | Specialized mIoU | Improvement |
|----------|---------|-----------|-------------------|-------------|
| Amphibian | 124 | 0.6796 | 0.7803 | +14.8% |
| Aquatic | 474 | 0.5627 | 0.7018 | +24.7% |
| Flying | 714 | 0.6103 | 0.7448 | +22.0% |
| Terrestrial | 699 | 0.5188 | 0.6753 | +30.2% |
| Other | 15 | 0.6940 | 0.8676 | +25.0% |

The largest improvement (+30.2%) is on terrestrial animals, which exhibit the most challenging camouflage patterns (insects on bark, lizards on rocks). Aquatic animals also show substantial gains (+24.7%), consistent with the difficulty of detecting fish and octopuses against coral/seabed textures. Amphibians show the smallest relative improvement (+14.8%) because base SAM already performs reasonably on them (0.680 mIoU), likely due to their more distinct body shapes.

### 5.4 Qualitative Results

Visual comparisons reveal consistent qualitative patterns:

- **Base SAM** frequently segments only the most salient region near the prompt point, missing large portions of camouflaged objects that blend with the background.
- **Specialized SAM** produces masks that closely follow the true object boundaries, even in cases where the animal's coloration is nearly identical to the surrounding environment.
- On images where base SAM succeeds (high-contrast subjects), both models perform comparably — the specialization does not degrade performance on easier cases.

## 6. Discussion

### Why Decoder-Only Fine-Tuning Works

SAM's ViT-H image encoder, pre-trained on 1 billion masks, already produces rich feature representations that capture texture gradients, edge patterns, and spatial structure. These representations contain the information needed to detect camouflaged boundaries — the encoder has "seen" enough natural images to encode subtle texture disruptions. The problem lies in the decoder: trained on general segmentation tasks, it hasn't learned to interpret these subtle features as object boundaries.

By freezing the encoder and training only the decoder, we teach it a new mapping: *subtle texture disruptions in the encoder features → object boundaries*. This is efficient because the decoder is small (~4M parameters vs ~632M in the encoder) and the encoder features are already expressive.

### Prompt Robustness as an Emergent Property

We attribute the observed prompt robustness to our mixed-prompt training strategy. By randomly alternating between point and box prompts during training, the decoder cannot overfit to any single prompt modality. Instead, it must learn to detect the camouflaged object regardless of how it is prompted.

This explains why improvement transfers to prompt types never seen during training (edge points, multi-point grids). The decoder learns *what* the camouflaged object is, not *where the user clicked*. When given an edge prompt, the specialized decoder still recognizes the object's full extent because it has learned the underlying structure, not a prompt-dependent shortcut.

### Practical Implications

This approach has attractive practical properties:

- **Lightweight:** Only the 16MB decoder checkpoint needs to be stored and distributed; the base SAM weights are unchanged.
- **Fast training:** With pre-computed embeddings, training completes in under 2 hours on a single GPU.
- **No architecture changes:** Compatible with any SAM deployment that supports weight loading.
- **Composable:** Different decoder checkpoints can be swapped for different domains without re-computing embeddings.

### Limitations

- **Single domain:** We evaluate only on COD10K. Generalization to other camouflage datasets (CAMO, CHAMELEON, NC4K) remains to be validated.
- **Single backbone:** We use only ViT-H. Whether the approach works with smaller SAM backbones (ViT-B, ViT-L) or MobileSAM is an open question — preliminary experiments with MobileSAM on other domains showed limited gains, suggesting model capacity matters.
- **No comparison to specialized COD architectures:** We compare only base vs. specialized SAM, not against dedicated COD methods (SINet, PFNet, ZoomNet) which may achieve higher absolute performance.

## 7. Conclusion

We demonstrate that decoder-only fine-tuning of SAM ViT-H on COD10K produces a +14.14 point mIoU improvement on camouflaged object detection across the full 2,026-image camouflaged test set, with a +219.8% relative gain on the hardest prompt type (edge points). The specialized model improves on all four tested prompt strategies. The key insight is that this improvement is **prompt-robust**: a decoder trained with mixed point and box prompts generalizes to all tested prompt strategies, including those never seen during training.

This suggests a broader principle: when adapting foundation segmentation models to challenging domains, the decoder is the right target for specialization. The encoder's learned representations are already sufficient; the decoder simply needs to learn a domain-appropriate mapping from features to masks.

**Future work** includes: (1) evaluating on additional COD benchmarks (CAMO, NC4K, CHAMELEON), (2) testing with smaller SAM backbones to establish minimum model capacity requirements, (3) extending to multi-domain specialization with a decoder bank, (4) a learnable local preprocessing module that enhances boundary features before SAM's encoder, and (5) a video-based camouflage dataset creation pipeline using SAM's segmentation failures as an automatic proxy for camouflage difficulty.

## References

[1] A. Kirillov et al., "Segment Anything," ICCV 2023.

[2] D. Fan et al., "Camouflaged Object Detection," CVPR 2020.

[3] N. Ravi et al., "SAM 2: Segment Anything in Images and Videos," arXiv 2024.

[4] T. Chen et al., "SAM-Adapter: Adapting Segment Anything in Underperformed Scenes," ICCVW 2023.

[5] C. Hao et al., "COMPrompter: Reconceptualized SAM with Multiprompt Network for Camouflaged Object Detection," 2024.

[6] D. Fan et al., "SINet: Camouflaged Object Detection via Search-Identify Network," CVPR 2020.

[7] H. Mei et al., "PFNet: Camouflaged Object Detection via Positioning and Focus Network," AAAI 2021.

[8] Y. Pang et al., "ZoomNet: A Simple Yet Effective Framework for Camouflaged Object Detection," CVPR 2022.

[9] E. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.

[10] B. Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning," EMNLP 2021.

[11] X. Liu et al., "Improving SAM for Camouflaged Object Detection via Dual-Stream Adapters," ICCV 2025.

[12] D. Fan et al., "Structure-measure: A New Way to Evaluate Foreground Maps," ICCV 2017.

[13] D. Fan et al., "Enhanced-alignment Measure for Binary Foreground Map Evaluation," IJCAI 2018.

[14] R. Margolin et al., "How to Evaluate Foreground Maps," CVPR 2014.
