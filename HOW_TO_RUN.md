# How to Run the Project - Step by Step

## 🎯 Recommended Way to Run

### Current project setup expects the real Galaxy Zoo 2 training dataset.

---

## 📦 Option A: With Real Galaxy Zoo 2 Dataset

### Step 1: Download Dataset

Go to: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data

Download:
- `images_training_rev1.zip` (13 GB)
- `training_solutions_rev1.csv` (5 MB)

### Step 2: Create Data Folder

In your project folder, create this structure:

```
your_project/
├── data/                          ← CREATE THIS FOLDER
│   ├── images_training_rev1/      ← EXTRACT ZIP HERE
│   │   ├── 100008.jpg
│   │   ├── 100023.jpg
│   │   └── ... (61,578 images)
│   └── training_solutions_rev1.csv ← COPY CSV HERE
├── data_loader.py
├── model.py
├── main.py
└── ... (other files)
```

### Step 3: Extract Files

**Windows:**
```cmd
mkdir data
cd data
:: Extract images_training_rev1.zip here (right-click → Extract All)
:: Copy training_solutions_rev1.csv here
cd ..
```

**Linux/Mac:**
```bash
mkdir -p data
cd data
unzip images_training_rev1.zip
cp ~/Downloads/training_solutions_rev1.csv .
cd ..
```

### Step 4: Verify Structure

Run this to check:
```bash
python -c "import os; print('Images:', len([f for f in os.listdir('data/images_training_rev1') if f.endswith('.jpg')]) if os.path.exists('data/images_training_rev1') else 'Folder not found'); print('CSV:', 'Found' if os.path.exists('data/training_solutions_rev1.csv') else 'Not found')"
```

Expected output:
```
Images: 61578
CSV: Found
```

### Step 5: Run the Project

```bash
python main.py
```

The code will automatically detect and use your real dataset!

---

## 🌐 See the UI (Web Interface)

### After Training Completes:

```bash
streamlit run app.py
```

This opens your browser to: `http://localhost:8501`

### What You'll See:

**Page 1: Galaxy Classifier**
- Upload button for galaxy images
- Real-time classification
- Confidence scores with bars
- Interactive charts

**Page 2: Unlearning Dashboard**
- "Run Unlearning Pipeline" button
- Progress bar
- Before/After comparison
- Interactive Plotly charts
- Method comparison table

**Page 3: Training Metrics**
- Training loss curves
- Validation accuracy
- Confusion matrices
- Full training log

**Sidebar:**
- Model status (Trained/Not Trained)
- Dataset info
- GPU/CPU status
- Real-time memory usage

---

## 🎬 Complete Workflow

### First Time Setup:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test installation
python test_setup.py

# 3. Run training
python train.py --device cuda --epochs 50 --batch-size 16 --lr 2e-4 --backbone resnet34 --target-acc 86 --patience 12

# 4. Launch web UI
streamlit run app.py
```

### What Gets Created:

After training and unlearning, you'll have:

```
project/
├── baseline_model.pth          ← Trained models
├── best_model.pth
├── ga_unlearned_model.pth
├── ff_unlearned_model.pth
├── retrained_model.pth
├── training_log.csv            ← Training metrics
├── unlearning_results.csv      ← Comparison results
└── plots/                      ← Visualizations
    ├── training_curves.png
    ├── accuracy_comparison.png
    ├── forget_accuracy.png
    ├── computation_time.png
    ├── confusion_matrix_before.png
    ├── confusion_matrix_after.png
    └── sample_predictions.png
```

---

## 🔍 How Dataset Integration Works

The code automatically detects your dataset:

```python
# In data_loader.py - it checks these locations:
1. data/images_training_rev1/  ← Kaggle structure
2. data/images_test_rev1/
3. data/images/
4. data/

# If found: Uses real Galaxy Zoo 2 images
# Current workflow is designed around the real dataset
```

**You don't need to change ANY code!** Just put files in `data/` folder.

---

## 📊 Expected Output

### Terminal Output:
```
🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌
  MACHINE UNLEARNING IN GALAXY MORPHOLOGY CLASSIFICATION
🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌

================================================================================
  HARDWARE INFORMATION
================================================================================

✅ GPU Available: NVIDIA GeForce RTX 3050
   Total VRAM: 4.00 GB
   ...

================================================================================
  STEP 1: DATA LOADING
================================================================================

📂 Found images in: data/images_training_rev1
📊 Loading labels from: data/training_solutions_rev1.csv
✅ Loaded 61578 labels from CSV
📸 Found 61578 images

📊 Class Distribution:
   Smooth: 26693 samples
   Featured: 34826 samples
   Artifact: 59 samples

...
```

### Web UI:
- Clean dark theme
- Interactive charts
- Upload functionality
- Real-time predictions
- Method comparison

---

## ⚡ Quick Commands

```bash
# Test everything works
python test_setup.py

# Run training
python train.py --device cuda --epochs 50 --batch-size 16 --lr 2e-4 --backbone resnet34 --target-acc 86 --patience 12

# Run unlearning from saved checkpoint
python unlearn.py --device cuda --checkpoint best_model.pth

# Launch web interface
streamlit run app.py

# Windows batch script
run_pipeline.bat

# Linux/Mac shell script
./run_pipeline.sh
```

---

## 🎯 Summary

**To integrate dataset:**
1. Create `data/` folder
2. Extract `images_training_rev1.zip` inside it
3. Copy `training_solutions_rev1.csv` inside it
4. Run training with `python train.py ...`

**To see UI:**
1. Run `python train.py ...` (trains model)
2. Run `streamlit run app.py` (opens browser)
3. Interact with the web interface!

---

Ready to see it in action? Run training, then `streamlit run app.py`
