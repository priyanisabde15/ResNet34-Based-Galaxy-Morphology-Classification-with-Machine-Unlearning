"""
Quick diagnostic test - Run this to see what's wrong
"""

print("="*60)
print("QUICK DIAGNOSTIC TEST")
print("="*60)

# Test 1: Python version
print("\n1. Python Version:")
import sys
print(f"   {sys.version}")

# Test 2: PyTorch
print("\n2. PyTorch:")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__} installed")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Other dependencies
print("\n3. Other Dependencies:")
deps = ['numpy', 'pandas', 'matplotlib', 'streamlit', 'PIL', 'sklearn']
for dep in deps:
    try:
        if dep == 'PIL':
            import PIL
            print(f"   ✅ Pillow installed")
        elif dep == 'sklearn':
            import sklearn
            print(f"   ✅ scikit-learn installed")
        else:
            __import__(dep)
            print(f"   ✅ {dep} installed")
    except:
        print(f"   ❌ {dep} NOT installed")

# Test 4: Project files
print("\n4. Project Files:")
import os
files = ['data_loader.py', 'model.py', 'train.py', 'main.py', 'app.py']
for f in files:
    if os.path.exists(f):
        print(f"   ✅ {f}")
    else:
        print(f"   ❌ {f} missing")

# Test 5: Data directory
print("\n5. Data Directory:")
if os.path.exists('data'):
    print(f"   ✅ data/ folder exists")
    if os.path.exists('data/images_training_rev1'):
        num_images = len([f for f in os.listdir('data/images_training_rev1') if f.endswith('.jpg')])
        print(f"   ✅ Found {num_images} images")
    else:
        print(f"   ⚠️  No images found (will use simulated data)")
else:
    print(f"   ⚠️  data/ folder not found (will use simulated data)")

# Test 6: Try importing project modules
print("\n6. Project Modules:")
try:
    import data_loader
    print("   ✅ data_loader.py imports successfully")
except Exception as e:
    print(f"   ❌ data_loader.py error: {e}")

try:
    import model
    print("   ✅ model.py imports successfully")
except Exception as e:
    print(f"   ❌ model.py error: {e}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nIf you see errors above, please share them!")
print("Otherwise, you're ready to run: python main.py")
