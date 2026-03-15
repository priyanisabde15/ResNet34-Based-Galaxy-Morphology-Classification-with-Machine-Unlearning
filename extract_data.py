"""
Quick extraction script — run once to unzip Galaxy Zoo 2 data
"""
import zipfile
import shutil
import os
from pathlib import Path

data_dir   = Path('./data')
images_dir = data_dir / 'images_training_rev1'
csv_target = data_dir / 'training_solutions_rev1.csv'

# Zips we care about
target_zips = [
    data_dir / 'images_training_rev1.zip',
    data_dir / 'training_solutions_rev1.zip',
]

for zf in target_zips:
    if not zf.exists():
        print(f"⚠️  Not found: {zf.name}")
        continue
    print(f"📦 Extracting {zf.name} ...")
    with zipfile.ZipFile(zf, 'r') as z:
        z.extractall(data_dir)
    print(f"   ✅ Done")

# Verify
if images_dir.exists():
    count = len(list(images_dir.glob('*.jpg')))
    print(f"\n✅ Images: {count} found in images_training_rev1/")
else:
    print("\n❌ images_training_rev1/ still not found")

if csv_target.exists():
    import pandas as pd
    df = pd.read_csv(csv_target)
    print(f"✅ Labels CSV: {len(df)} rows")
else:
    print("❌ training_solutions_rev1.csv not found")

print("\nDone! Now run:  python main.py")
