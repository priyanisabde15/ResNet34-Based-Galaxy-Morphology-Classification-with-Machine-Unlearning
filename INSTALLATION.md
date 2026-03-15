# Complete Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Storage**: 2 GB free space (20 GB if using real dataset)
- **GPU**: None (CPU fallback available)

### Recommended Requirements
- **OS**: Windows 11 or Ubuntu 20.04+
- **Python**: 3.9 or 3.10
- **RAM**: 16 GB
- **Storage**: 20 GB free space
- **GPU**: NVIDIA RTX 3050 (4GB VRAM) or better
- **CUDA**: 11.8 or 12.1

## Installation Steps

### Step 1: Install Python

#### Windows
1. Download Python from https://www.python.org/downloads/
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Click "Install Now"

Verify:
```cmd
python --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv
```

#### macOS
```bash
brew install python@3.10
```

### Step 2: Clone/Download Project

```bash
# If using git
git clone <repository-url>
cd galaxy-morphology-unlearning

# Or download and extract ZIP
```

### Step 3: Create Virtual Environment (Recommended)

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch (with CUDA support if available)
- TorchVision
- Streamlit
- Plotly
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- And more...

**Installation time**: 5-10 minutes

### Step 5: Verify Installation

```bash
python test_setup.py
```

Expected output:
```
============================================================
  SETUP TEST - Galaxy Morphology Unlearning
============================================================

Testing imports...
✅ PyTorch 2.x.x
✅ TorchVision 0.x.x
✅ NumPy 1.x.x
...

Testing CUDA...
✅ CUDA Available
   Device: NVIDIA GeForce RTX 3050
...

✅ ALL TESTS PASSED!
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU with CUDA

#### Check GPU
```bash
nvidia-smi
```

#### Install CUDA Toolkit (if needed)

**Windows:**
1. Download from: https://developer.nvidia.com/cuda-downloads
2. Install CUDA 11.8 or 12.1
3. Restart computer

**Linux:**
```bash
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA website
```

#### Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:
```python
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### AMD GPU (ROCm)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

### Apple Silicon (M1/M2)

PyTorch with MPS (Metal Performance Shaders):
```bash
pip install torch torchvision
```

The code will automatically use MPS if available.

## Dataset Setup (Optional)

### Option A: Download Galaxy Zoo 2 (13 GB)

```bash
# Install Kaggle API
pip install kaggle

# Configure credentials (see DATASET_SETUP.md)

# Download dataset
python download_dataset.py
```

### Option B: Use Simulated Data (No Download)

Just run the code - it will auto-generate synthetic data!

```bash
python main.py
```

## Troubleshooting

### "Python not found"

**Windows:**
- Reinstall Python with "Add to PATH" checked
- Or add manually: System Properties → Environment Variables → Path

**Linux/Mac:**
- Use `python3` instead of `python`
- Add alias: `echo "alias python=python3" >> ~/.bashrc`

### "pip not found"

```bash
python -m ensurepip --upgrade
```

### "CUDA not available" (but you have NVIDIA GPU)

1. Check drivers:
```bash
nvidia-smi
```

2. Reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. Verify:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

### "Out of memory" during training

Edit `config.py`:
```python
BATCH_SIZE = 8  # Reduce from 16
```

Or use CPU:
```python
DEVICE = torch.device('cpu')
```

### "ModuleNotFoundError"

```bash
pip install -r requirements.txt --upgrade
```

### "Permission denied" (Linux/Mac)

```bash
chmod +x run_pipeline.sh
sudo chown -R $USER:$USER .
```

### Slow training on CPU

This is normal. GPU is 10-20x faster.

Options:
- Use Google Colab (free GPU)
- Reduce epochs in `config.py`
- Use smaller dataset

## Platform-Specific Notes

### Windows

- Use Command Prompt or PowerShell
- Paths use backslashes: `data\images\`
- Run batch file: `run_pipeline.bat`

### Linux

- Use Terminal
- Paths use forward slashes: `data/images/`
- Run shell script: `./run_pipeline.sh`
- May need `sudo` for some operations

### macOS

- Use Terminal
- Similar to Linux
- Apple Silicon (M1/M2) uses MPS instead of CUDA
- May need Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

## Docker Installation (Advanced)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t galaxy-unlearning .
docker run -it --gpus all galaxy-unlearning
```

## Cloud Setup (Google Colab)

1. Upload project to Google Drive
2. Open Colab notebook
3. Mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Install dependencies:
```python
!pip install -r requirements.txt
```

5. Run:
```python
!python main.py
```

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip list`)
- [ ] test_setup.py passes
- [ ] GPU detected (optional)
- [ ] Dataset ready or will use simulation
- [ ] Can import torch, streamlit, etc.

## Next Steps

After successful installation:

1. **Quick test** (2 minutes):
   ```bash
   python test_setup.py
   ```

2. **Run pipeline** (20-30 minutes):
   ```bash
   python main.py
   ```

3. **Launch web app**:
   ```bash
   streamlit run app.py
   ```

## Getting Help

- Check `README.md` for detailed documentation
- See `QUICKSTART.md` for quick start guide
- Review `DATASET_SETUP.md` for dataset issues
- Check GitHub issues (if available)

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows

# Remove downloaded data
rm -rf data plots *.pth *.csv
```

---

**Installation complete!** 🎉

Run `python main.py` to start training!
