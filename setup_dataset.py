"""
One-click Galaxy Zoo 2 dataset setup
Downloads BOTH images + labels CSV from a single Kaggle dataset (no competition)
Dataset: https://www.kaggle.com/datasets/saurabhshahane/galaxy-zoo-2
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path


def install_kaggle():
    try:
        import kaggle
        print("✅ Kaggle API already installed")
    except ImportError:
        print("📦 Installing Kaggle API...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle', '-q'])
        print("✅ Kaggle API installed")


def setup_credentials():
    local_json  = Path('./kaggle.json')
    kaggle_dir  = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if local_json.exists():
        print("✅ Found kaggle.json in project folder")
        kaggle_dir.mkdir(exist_ok=True)
        shutil.copy(local_json, kaggle_json)
        try:
            os.chmod(kaggle_json, 0o600)
        except Exception:
            pass
        print(f"✅ Copied to {kaggle_json}")
        return True

    if kaggle_json.exists():
        print("✅ Kaggle credentials found")
        return True

    print("\n" + "="*60)
    print("  KAGGLE CREDENTIALS NEEDED")
    print("="*60)
    print("\n1. Go to: https://www.kaggle.com/account")
    print("2. Scroll to 'API' → click 'Create New API Token'")
    print("3. Place kaggle.json here:")
    print(f"   {Path('.').absolute()}")
    print("="*60)
    input("\nPress ENTER after placing kaggle.json in the project folder...")

    if local_json.exists():
        kaggle_dir.mkdir(exist_ok=True)
        shutil.copy(local_json, kaggle_json)
        print("✅ Credentials ready!")
        return True

    print("❌ kaggle.json not found.")
    return False


def download_dataset():
    """
    Download Galaxy Zoo 2 from the Kaggle competition:
    https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge
    Includes both images AND training_solutions_rev1.csv
    NOTE: You must accept the competition rules once at the URL above.
    """
    data_dir   = Path('./data')
    images_dir = data_dir / 'images_training_rev1'
    csv_file   = data_dir / 'training_solutions_rev1.csv'

    images_ok = images_dir.exists() and len(list(images_dir.glob('*.jpg'))) > 1000
    csv_ok    = csv_file.exists()

    if images_ok and csv_ok:
        print("✅ Dataset already complete — skipping download")
        return True

    data_dir.mkdir(exist_ok=True)

    print("\n📥 Downloading Galaxy Zoo 2 (images + labels)...")
    print("   Source : kaggle.com/c/galaxy-zoo-the-galaxy-challenge")
    print("   Size   : ~13 GB — takes 10-30 min depending on internet")
    print("   Sit back and relax ☕\n")

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            'galaxy-zoo-the-galaxy-challenge',
            path=str(data_dir),
            quiet=False
        )
        print("\n✅ Download complete!")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n── Fix ──────────────────────────────────────────────────")
        print("You need to accept the competition rules once:")
        print("1. Go to: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data")
        print("2. Scroll to the bottom → click 'I Understand and Accept'")
        print("3. Run this script again")
        print("─────────────────────────────────────────────────────────")
        return False


def extract_files():
    """Extract zip and normalise folder structure"""
    data_dir   = Path('./data')
    images_dir = data_dir / 'images_training_rev1'

    if images_dir.exists() and len(list(images_dir.glob('*.jpg'))) > 1000:
        print("✅ Images already extracted")
        return True

    print("\n📦 Extracting files...")
    zip_files = list(data_dir.glob('*.zip'))

    if not zip_files:
        print("❌ No zip files found in data/")
        print("   Download manually from:")
        print("   https://www.kaggle.com/datasets/saurabhshahane/galaxy-zoo-2")
        return False

    for zf in zip_files:
        print(f"   Extracting {zf.name} ...")
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(data_dir)
        print(f"   ✅ {zf.name} done")

    # Some zips nest images inside a subfolder — find and move them
    if not images_dir.exists() or len(list(images_dir.glob('*.jpg'))) == 0:
        print("\n🔍 Locating images inside extracted folders...")
        for root, dirs, files in os.walk(str(data_dir)):
            jpgs = [f for f in files if f.lower().endswith('.jpg')]
            if len(jpgs) > 100:
                found = Path(root)
                print(f"   Found {len(jpgs)} images in: {found.name}/")
                if found != images_dir:
                    images_dir.mkdir(exist_ok=True)
                    print("   Moving to images_training_rev1/ ...")
                    for jpg in found.glob('*.jpg'):
                        shutil.move(str(jpg), str(images_dir / jpg.name))
                    print("   ✅ Moved")
                break

    # Move CSV to expected location if it landed elsewhere
    csv_target = data_dir / 'training_solutions_rev1.csv'
    if not csv_target.exists():
        for root, dirs, files in os.walk(str(data_dir)):
            for f in files:
                if 'solution' in f.lower() and f.endswith('.csv'):
                    shutil.move(str(Path(root) / f), str(csv_target))
                    print(f"   ✅ Labels CSV moved to data/training_solutions_rev1.csv")
                    break

    return True


def verify():
    data_dir   = Path('./data')
    images_dir = data_dir / 'images_training_rev1'
    csv_file   = data_dir / 'training_solutions_rev1.csv'

    print("\n🔍 Verifying dataset...")

    if images_dir.exists():
        count = len(list(images_dir.glob('*.jpg')))
        print(f"   ✅ Images : {count} found")
    else:
        print("   ❌ images_training_rev1/ not found")

    if csv_file.exists():
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"   ✅ Labels : {len(df)} rows in CSV")
    else:
        print("   ⚠️  Labels CSV not found — pseudo-labels will be used")


def clean_old_models():
    files = [
        'baseline_model.pth', 'corrupted_model.pth', 'best_model.pth',
        'training_log.csv', 'results.csv', 'ga_unlearned_model.pth',
        'ff_unlearned_model.pth', 'retrained_model.pth'
    ]
    removed = []
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            removed.append(f)
    if removed:
        print(f"\n🗑️  Removed old models: {', '.join(removed)}")
    else:
        print("\n✅ No old model files found")


def main():
    print("=" * 60)
    print("  Galaxy Zoo 2 — One-Click Dataset Setup")
    print("=" * 60)

    install_kaggle()

    if not setup_credentials():
        sys.exit(1)

    if not download_dataset():
        sys.exit(1)

    if not extract_files():
        sys.exit(1)

    verify()

    print("\n" + "=" * 60)
    ans = input("Delete old trained models for a clean run? (y/n): ")
    if ans.lower() == 'y':
        clean_old_models()

    print("\n✅ All done! Now run:")
    print("   python main.py        ← train the model")
    print("   streamlit run app.py  ← launch the web app")
    print("=" * 60)


if __name__ == "__main__":
    main()
