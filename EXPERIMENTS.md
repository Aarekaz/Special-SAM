# Experiment Tracking

## HPC Jobs (SLURM)

| Job ID    | Script                  | Description                    | Status    | Submitted           | Finished            | Notes                                                  |
|-----------|-------------------------|--------------------------------|-----------|---------------------|---------------------|--------------------------------------------------------|
| —         | scripts/precompute.sh   | Pre-compute ViT-H embeddings   | Completed | —                   | —                   | Cached to .npy files                                   |
| —         | scripts/train.sh        | Decoder-only fine-tuning        | Completed | —                   | —                   | 15 epochs, final loss 0.0717, checkpoint: camo_decoder_vith.pth |
| 5787790   | scripts/train_llpm.sh   | LLPM + decoder joint training   | Completed | 2026-03-02 13:48 EST | 2026-03-02 21:14 EST | 15 epochs, final loss 0.0421, alpha 0.1847             |
| 5787867   | scripts/eval.sh         | Full eval (decoder + LLPM)      | Completed | 2026-03-02 ~21:15 EST | 2026-03-03 02:41 EST | 3 models × 4 prompts × 2026 images                    |

## Planned Jobs (2026-03-11)

| Priority | Script                          | Description                        | Depends On | Est. Time | GPU? | Status  |
|----------|---------------------------------|------------------------------------|------------|-----------|------|---------|
| 1        | scripts/eval.sh                 | Re-eval with per-image CSV saving  | —          | ~6 hrs    | Yes  | **Completed** (5803621) |
| 1        | scripts/ablation_prompt.sh      | Prompt-type ablation (2 tasks)     | —          | ~8 hrs    | Yes  | **Completed** (5804070) |
| 2        | scripts/error_analysis.sh       | Failure categorization + montages  | Job P1-eval | ~30 min  | No   | **Completed** (5804069) |
| 3        | scripts/ablation_augmentation.sh| Augmentation ablation (2 tasks)    | —          | ~16 hrs   | Yes  | **Completed** (5807623, resubmit of 5804072) |
| 4        | scripts/ablation_llpm.sh        | LLPM component ablation (3 tasks)  | —          | ~16 hrs   | Yes  | **Submitted** (5808865) |

### Expected Outputs
- `results/per_image_results.csv` — per-image metrics + metadata (from eval)
- `results/error_analysis/` — failure_list.csv, failure_summary.csv, montage PNGs
- `results/ablation/point_only_results.csv`, `box_only_results.csv` (from prompt ablation)
- `checkpoints/ablation/camo_decoder_point_only.pth`, `camo_decoder_box_only.pth`
- `checkpoints/ablation/camo_decoder_flip_rot.pth`, `camo_decoder_flip_rot_shift.pth`

## Checkpoints

| File                                  | Description                          | Size   |
|---------------------------------------|--------------------------------------|--------|
| weights/sam_vit_h_4b8939.pth          | Base SAM ViT-H                       | ~2.4GB |
| checkpoints/camo_decoder_vith.pth     | Decoder-only fine-tuned              | ~16MB  |
| checkpoints/llpm_vith.pth             | LLPM module                          | ~124KB |
| checkpoints/camo_decoder_llpm_vith.pth| Decoder after LLPM joint training    | ~16MB  |

## Key Training Metrics

### Decoder-Only (15 epochs)
- Loss: 0.1417 → 0.0717 (49.4% reduction)
- mIoU (CoM): 57.24% → 73.81% (+16.57 pts)
- Edge prompt: 20.52% → 69.38% (+238.1% relative)

### LLPM + Decoder (15 epochs, job 5787790)
- Loss: 0.1111 → 0.0421 (62.1% reduction)
- Alpha: 0.1034 → 0.1847 (steady growth, stabilized)
- Alpha gradients near zero by epoch 14-15 (converged)
- Eval results: completed (job 5787867)

## Eval Results (job 5787867)

### LLPM vs Decoder-Only (mIoU)

| Prompt       | Base  | Decoder-only | LLPM  | LLPM vs Dec. |
|--------------|-------|-------------|-------|-------------|
| CoM          | .572  | .738        | .740  | +0.23       |
| Edge         | .205  | .694        | .719  | **+2.57**   |
| Grid         | .682  | .756        | .755  | -0.09       |
| Random       | .726  | .775        | .772  | -0.24       |

### Full LLPM Metrics

| Prompt | mIoU | Dice | S_α  | E_φ  | F_β^w | MAE  |
|--------|------|------|------|------|-------|------|
| CoM    | .740 | .820 | .847 | .938 | .820  | .028 |
| Edge   | .719 | .802 | .827 | .933 | .778  | .029 |
| Grid   | .755 | .833 | .856 | .946 | .833  | .025 |
| Random | .772 | .851 | .866 | .955 | .848  | .022 |

## Prompt Ablation Results (job 5804070, 500 test images)

| Prompt | Mixed (ours) | Point-Only | Box-Only |
|--------|-------------|------------|----------|
| CoM    | **.738**    | .699       | .376     |
| Edge   | **.694**    | .672       | .501     |
| Grid   | **.756**    | .725       | .389     |
| Random | **.775**    | .768       | .455     |

Finding: Mixed training outperforms point-only by 3-4 mIoU on every prompt.
Box-only is catastrophic on point prompts (CoM drops to 0.376).

## Augmentation Ablation Results (job 5807623, 500 test images)

| Prompt | Flip only (ours) | + Rotation | + Rot + Shift |
|--------|-----------------|------------|---------------|
| CoM    | **.738**        | .722       | .724          |
| Edge   | .694            | .708       | **.718**      |
| Grid   | **.756**        | .741       | .738          |
| Random | **.775**        | .768       | .765          |

Training details:
- Flip only: 6,080 samples, final loss 0.0717
- Flip + rotation: 9,120 samples, final loss 0.0683
- Flip + rot + shift: 12,160 samples, final loss 0.0638

Finding: More augmentation slightly hurts CoM/Grid/Random but helps Edge (+2.4 mIoU).
Differences are small (<2 pts). Flip-only is near-optimal.

## Error Analysis Results (job 5804069)

| Category       | N   | Base  | Specialized | LLPM  |
|----------------|-----|-------|-------------|-------|
| Disjoint       | 175 | .213  | .303        | **.371** |
| Dark           | 104 | .316  | .326        | **.366** |
| General        | 65  | .352  | .342        | **.413** |
| Tiny           | 54  | .289  | .261        | **.292** |
| Thin/elongated | 2   | .004  | **.346**    | .222  |

Finding: LLPM improves on all major categories. Disjoint objects (44% of failures) and
dark images (26%) dominate. Tiny objects remain hardest (mean IoU 0.29).
