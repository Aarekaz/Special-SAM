"""Setup script to download all required assets for evaluation.

This script downloads:
1. COD10K dataset from Kaggle (via kagglehub)
2. SAM ViT-H base weights from Facebook
3. Trained decoder checkpoint from HuggingFace

Usage:
    python scripts/setup_evaluation.py
"""

import os
import zipfile
from pathlib import Path

# Check if required packages are installed
try:
    import kagglehub
    from huggingface_hub import hf_hub_download
    import requests
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install kagglehub huggingface_hub requests tqdm")
    exit(1)


def download_cod10k():
    """Download COD10K dataset from Kaggle."""
    print("\n" + "="*60)
    print("STEP 1: Downloading COD10K Dataset")
    print("="*60)

    try:
        # Download via kagglehub (requires Kaggle API authentication)
        print("Downloading from Kaggle (requires authentication)...")
        path = kagglehub.dataset_download("aarekaz/cod10k")
        print(f"[OK] Dataset downloaded to: {path}")

        # Create symlink or copy to expected location
        data_root = Path("data/cod10k")
        data_root.mkdir(parents=True, exist_ok=True)

        source = Path(path) / "COD10K-v3"
        target = data_root / "COD10K-v3"

        if not target.exists():
            print(f"Creating link: {source} -> {target}")
            # On Windows, we might need to copy instead of symlink
            import shutil
            if source.exists():
                shutil.copytree(source, target, dirs_exist_ok=True)
                print(f"[OK] Dataset copied to: {target}")

        # Verify structure
        test_img = target / "Test" / "Image"
        test_mask = target / "Test" / "GT_Object"

        if test_img.exists() and test_mask.exists():
            img_count = len(list(test_img.glob("*.jpg")) + list(test_img.glob("*.png")))
            print(f"[OK] Found {img_count} test images")
            return True
        else:
            print("[ERROR] Dataset structure not as expected")
            return False

    except Exception as e:
        print(f"[ERROR] Error downloading COD10K: {e}")
        print("\nAlternative: Download manually from Kaggle:")
        print("  https://www.kaggle.com/datasets/aarekaz/cod10k")
        print("  Extract to: data/cod10k/COD10K-v3/")
        return False


def download_sam_weights():
    """Download SAM ViT-H base weights from Facebook."""
    print("\n" + "="*60)
    print("STEP 2: Downloading SAM ViT-H Weights")
    print("="*60)

    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    checkpoint_path = weights_dir / "sam_vit_h_4b8939.pth"

    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"[OK] SAM weights already exist ({size_mb:.1f} MB)")
        return True

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    try:
        print(f"Downloading SAM ViT-H weights (2.4 GB)...")
        print(f"URL: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"[OK] SAM weights downloaded to: {checkpoint_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error downloading SAM weights: {e}")
        print("\nAlternative: Download manually:")
        print(f"  wget -P weights {url}")
        return False


def download_decoder_checkpoint():
    """Download trained decoder checkpoint from HuggingFace."""
    print("\n" + "="*60)
    print("STEP 3: Downloading Trained Decoder Checkpoint")
    print("="*60)

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoints_dir / "camo_decoder_vith.pth"

    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Decoder checkpoint already exists ({size_mb:.1f} MB)")
        return True

    repo_id = "AAREKAZ/SpecialSAM"
    filename = "camo_decoder_vith.pth"

    try:
        print(f"Downloading from HuggingFace: {repo_id}/{filename}")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )

        # Copy to expected location
        import shutil
        shutil.copy(downloaded_path, checkpoint_path)

        print(f"[OK] Decoder checkpoint downloaded to: {checkpoint_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error downloading decoder checkpoint: {e}")
        print("\nAlternative: Download manually from HuggingFace:")
        print(f"  https://huggingface.co/{repo_id}")
        print(f"  Save to: {checkpoint_path}")
        return False


def verify_setup():
    """Verify all required files exist."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    checks = {
        "COD10K Test Images": Path("data/cod10k/COD10K-v3/Test/Image"),
        "COD10K Test Masks": Path("data/cod10k/COD10K-v3/Test/GT_Object"),
        "SAM Base Weights": Path("weights/sam_vit_h_4b8939.pth"),
        "Decoder Checkpoint": Path("checkpoints/camo_decoder_vith.pth"),
    }

    all_good = True
    for name, path in checks.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            print(f"[ERROR] {name}: {path} (NOT FOUND)")
            all_good = False

    if all_good:
        print("\nAll assets ready! You can now run evaluation:")
        print("   python -m src.evaluation.evaluate --config configs/eval.yaml")
    else:
        print("\nSome assets are missing. Please review the errors above.")

    return all_good


def main():
    """Run full setup."""
    print("Special-SAM Evaluation Setup")
    print("This script will download ~3GB of data\n")

    # Step 1: COD10K Dataset
    cod10k_ok = download_cod10k()

    # Step 2: SAM Weights
    sam_ok = download_sam_weights()

    # Step 3: Decoder Checkpoint
    decoder_ok = download_decoder_checkpoint()

    # Verify
    verify_setup()


if __name__ == "__main__":
    main()
