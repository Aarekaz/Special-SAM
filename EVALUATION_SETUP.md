# Evaluation Setup Guide

This guide will help you set up everything needed to run the comprehensive evaluation with COD metrics.

## Quick Setup (Automated)

### Step 1: Install Dependencies

```bash
pip install kagglehub huggingface_hub requests tqdm
```

### Step 2: Run Setup Script

```bash
python scripts/setup_evaluation.py
```

This will download (~3GB total):
- ✅ COD10K dataset (from Kaggle)
- ✅ SAM ViT-H weights (2.4GB from Facebook)
- ✅ Trained decoder checkpoint (16MB from HuggingFace)

### Step 3: Run Evaluation

```bash
python -m src.evaluation.evaluate --config configs/eval.yaml
```

This will:
- Evaluate Base SAM and Specialized SAM on 200 test images
- Test 4 prompt strategies (center, edge, multi-grid, multi-random)
- Compute **all metrics** including COD metrics (S-alpha, E-phi, F-beta-w, MAE)
- Save results to: `results/comprehensive_evaluation_results.csv`

---

## Manual Setup (If Automated Fails)

### Dataset: COD10K

**Option A: Kaggle CLI**
```bash
# Configure Kaggle API first (requires account)
# Download your kaggle.json from: https://www.kaggle.com/settings/account
# Place at: ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<user>\.kaggle\kaggle.json (Windows)

kaggle datasets download -d aarekaz/cod10k
unzip cod10k.zip -d data/cod10k/
```

**Option B: Manual Download**
1. Go to: https://www.kaggle.com/datasets/aarekaz/cod10k
2. Download ZIP file
3. Extract to: `data/cod10k/COD10K-v3/`

**Expected structure:**
```
data/cod10k/COD10K-v3/
├── Train/
│   ├── Image/      (6000 images)
│   └── GT_Object/  (6000 masks)
└── Test/
    ├── Image/      (4000 images)
    └── GT_Object/  (4000 masks)
```

---

### Base SAM Weights

**Download SAM ViT-H (2.4GB):**
```bash
mkdir -p weights
wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**Or download manually:**
- URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- Save to: `weights/sam_vit_h_4b8939.pth`

---

### Trained Decoder Checkpoint

**Download from HuggingFace (16MB):**
```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download AAREKAZ/SpecialSAM camo_decoder_vith.pth --local-dir checkpoints/
```

**Or download manually:**
1. Go to: https://huggingface.co/AAREKAZ/SpecialSAM
2. Download: `camo_decoder_vith.pth`
3. Save to: `checkpoints/camo_decoder_vith.pth`

---

## Verification

Check that all files exist:

```bash
# On Linux/Mac:
ls -lh data/cod10k/COD10K-v3/Test/Image/ | wc -l  # Should show ~4000
ls -lh weights/sam_vit_h_4b8939.pth                # Should be ~2.4GB
ls -lh checkpoints/camo_decoder_vith.pth           # Should be ~16MB

# On Windows PowerShell:
(Get-ChildItem data\cod10k\COD10K-v3\Test\Image\).Count  # Should show ~4000
Get-Item weights\sam_vit_h_4b8939.pth | Select-Object Length
Get-Item checkpoints\camo_decoder_vith.pth | Select-Object Length
```

---

## Running Evaluation

### Full Evaluation (200 samples, ~15-20 minutes)

```bash
python -m src.evaluation.evaluate --config configs/eval.yaml
```

### Test on Small Subset First (10 samples, ~1 minute)

Edit `configs/eval.yaml` and change:
```yaml
evaluation:
  max_samples: 10  # Change from 200 to 10
```

Then run:
```bash
python -m src.evaluation.evaluate --config configs/eval.yaml
```

---

## Expected Output

### Console Output:
```
Evaluating BASE model on 200 samples
  Testing: Center-of-Mass (1 point)
    Processed 50/200
    Processed 100/200
    Processed 150/200
    Processed 200/200
    ✅ Results: mIoU=0.4752, S-alpha=0.7123, MAE=0.0823

  Testing: Edge (1 point)
    ...

Evaluating SPECIALIZED model on 200 samples
  ...

Results saved to results/comprehensive_evaluation_results.csv
```

### Output Files:
- `results/comprehensive_evaluation_results.csv` - Full metrics table

### CSV Columns:
- `model`: "Base SAM ViT-H" or "Specialized SAM ViT-H"
- `prompt_strategy`: center, edge_single, multi_grid, multi_random
- `iou_mean`, `iou_std`
- `dice_mean`, `dice_std`
- `f1_mean`, `f1_std`
- `boundary_prec_mean`, `boundary_recall_mean`, `boundary_f1_mean`
- **`s_alpha_mean`** ← COD metric
- **`e_phi_mean`** ← COD metric
- **`f_beta_w_mean`** ← COD metric
- **`mae_mean`** ← COD metric
- `num_samples`

---

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'src.data'

**Solution:** Make sure you're in the project root:
```bash
cd C:\Development\Special-SAM
python -m src.evaluation.evaluate --config configs/eval.yaml
```

### Issue: FileNotFoundError: dataset not found

**Solution:** Check dataset path in `configs/eval.yaml`:
```yaml
data:
  test_img_dir: data/cod10k/COD10K-v3/Test/Image
  test_mask_dir: data/cod10k/COD10K-v3/Test/GT_Object
```

### Issue: CUDA out of memory

**Solution:** Edit `configs/eval.yaml` to process fewer samples:
```yaml
evaluation:
  max_samples: 50  # Reduce from 200
```

Or use CPU (slower but no memory limit):
```python
# In src/models/sam_loader.py, force CPU:
device = torch.device("cpu")
```

### Issue: Kaggle authentication error

**Solution:** Set up Kaggle API:
1. Go to: https://www.kaggle.com/settings/account
2. Click "Create New Token" (downloads kaggle.json)
3. Place at:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
4. Set permissions (Linux/Mac only): `chmod 600 ~/.kaggle/kaggle.json`

---

## Next Steps After Evaluation

Once evaluation completes:

1. ✅ Check the CSV: `results/comprehensive_evaluation_results.csv`
2. ✅ Extract COD metric values
3. ✅ Update paper tables (paper/paper.md)
4. ✅ Check off Option B in `plans/paper-readiness-gaps.md`

See paper tables that need updating:
- Table 1 (Main results) - line 122
- Table 3 (Detailed metrics) - line 150
