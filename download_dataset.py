"""
Helper script to download Galaxy Zoo 2 dataset from Kaggle
Requires: kaggle API credentials
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def check_kaggle_api():
    """Check if Kaggle API is installed and configured"""
    try:
        import kaggle
        print("✅ Kaggle API installed")
        return True
    except ImportError:
        print("❌ Kaggle API not installed")
        print("\nInstall with: pip install kaggle")
        return False

def check_kaggle_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json.exists():
        print(f"✅ Kaggle credentials found at {kaggle_json}")
        return True
    else:
        print(f"❌ Kaggle credentials not found")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to:")
        print(f"   {kaggle_json}")
        print("\nWindows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("Linux/Mac: ~/.kaggle/kaggle.json")
        return False

def download_galaxy_zoo_2():
    """Download Galaxy Zoo 2 dataset from Kaggle"""
    
    print("\n" + "="*60)
    print("  Galaxy Zoo 2 Dataset Downloader")
    print("="*60 + "\n")
    
    # Check prerequisites
    if not check_kaggle_api():
        print("\n❌ Please install kaggle API first")
        return False
    
    if not check_kaggle_credentials():
        print("\n❌ Please configure Kaggle credentials first")
        return False
    
    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    print(f"\n📁 Data directory: {data_dir.absolute()}")
    
    # Download dataset
    print("\n📥 Downloading Galaxy Zoo 2 dataset...")
    print("   This may take 10-30 minutes depending on your connection...")
    
    try:
        # Download using Kaggle API
        competition = 'galaxy-zoo-the-galaxy-challenge'
        
        print(f"\n   Downloading from competition: {competition}")
        
        # Download files
        cmd = [
            'kaggle', 'competitions', 'download',
            '-c', competition,
            '-p', str(data_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            print("\nAlternative: Download manually from:")
            print(f"https://www.kaggle.com/c/{competition}/data")
            return False
        
        print("✅ Download complete!")
        
        # Extract files
        print("\n📦 Extracting files...")
        
        zip_files = list(data_dir.glob('*.zip'))
        
        for zip_file in zip_files:
            print(f"   Extracting {zip_file.name}...")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            print(f"   ✅ Extracted {zip_file.name}")
        
        # Verify extraction
        print("\n🔍 Verifying dataset...")
        
        images_dir = data_dir / 'images_training_rev1'
        csv_file = data_dir / 'training_solutions_rev1.csv'
        
        if images_dir.exists():
            num_images = len(list(images_dir.glob('*.jpg')))
            print(f"   ✅ Found {num_images} training images")
        else:
            print(f"   ⚠️  Images directory not found: {images_dir}")
        
        if csv_file.exists():
            print(f"   ✅ Found labels CSV")
        else:
            print(f"   ⚠️  Labels CSV not found: {csv_file}")
        
        print("\n" + "="*60)
        print("  Dataset Ready!")
        print("="*60)
        print("\nNext steps:")
        print("  python main.py          # Run full pipeline")
        print("  streamlit run app.py    # Launch web interface")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data")
        print("2. Download: images_training_rev1.zip")
        print("3. Download: training_solutions_rev1.csv")
        print("4. Extract to: ./data/")
        return False

def main():
    """Main function"""
    
    print("Galaxy Zoo 2 Dataset Downloader")
    print("="*60)
    print("\nThis script will download the Galaxy Zoo 2 dataset from Kaggle.")
    print("You need a Kaggle account and API credentials.")
    print("\nDataset size: ~13 GB")
    print("Download time: 10-30 minutes")
    
    response = input("\nContinue? (y/n): ")
    
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    success = download_galaxy_zoo_2()
    
    if not success:
        print("\n" + "="*60)
        print("  Alternative: Use Simulated Data")
        print("="*60)
        print("\nYou can run the project without downloading the dataset!")
        print("The code will automatically generate synthetic galaxy images.")
        print("\nJust run: python main.py")
        print("\nSee DATASET_SETUP.md for more information.")

if __name__ == "__main__":
    main()
