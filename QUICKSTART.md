# Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

## Step 2: Train or Run the Pipeline

```bash
python train.py --device cuda --epochs 50 --batch-size 16 --lr 2e-4 --backbone resnet34 --target-acc 86 --patience 12
```

This will:
- train the current ResNet34 classifier on real Galaxy Zoo 2 data
- save `best_model.pth`
- write `training_log.csv`
- optionally continue into unlearning if `--no-unlearning` is not used

**Grab a coffee while it runs!** ☕

## Step 3: Launch Web App (30 seconds)

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

## What You'll See

### Terminal Output
```
🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌
  MACHINE UNLEARNING IN GALAXY MORPHOLOGY CLASSIFICATION
  Research Paper Implementation
🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌

================================================================================
  HARDWARE INFORMATION
================================================================================

✅ GPU Available: NVIDIA GeForce RTX 3050
   Total VRAM: 4.00 GB
   ...
```

### Generated Files
```
├── baseline_model.pth          # Clean model
├── best_model.pth              # Best ResNet34 checkpoint
├── ga_unlearned_model.pth      # After gradient ascent
├── ff_unlearned_model.pth      # After Fisher forgetting
├── retrained_model.pth         # Full retrain
├── unlearning_results.csv      # Comparison table
├── training_log.csv            # Training metrics
└── plots/                      # All visualizations
    ├── training_curves.png
    ├── accuracy_comparison.png
    ├── forget_accuracy.png
    ├── computation_time.png
    ├── confusion_matrix_before.png
    ├── confusion_matrix_after.png
    └── sample_predictions.png
```

### Latest Recorded Results
```
================================================================================
MACHINE UNLEARNING COMPARISON
================================================================================

Method               Test Acc     Forget Acc   Retain Acc   Time(s)  
--------------------------------------------------------------------------------
Gradient Ascent         70.80%       11.01%       76.39%      103.6
Fisher Forgetting       41.71%       28.36%       43.44%      213.8
Full Retrain            76.17%        8.36%       99.64%     2649.4
================================================================================
```

## Quick Test (2 mins)

Want to test without full training? Run individual modules:

```bash
# Test data loader
python data_loader.py

# Test model
python model.py

# Test evaluation
python evaluate.py
```

## Troubleshooting

### "Out of memory"
Edit `data_loader.py` line 200:
```python
batch_size=8  # Reduce from 16 to 8
```

### "No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "Slow training"
- Normal on CPU (30-60 mins)
- Fast on GPU (15-25 mins)

## Next Steps

1. ✅ View results: `cat unlearning_results.csv`
2. ✅ Check plots: Open `plots/` folder
3. ✅ Run web app: `streamlit run app.py`
4. ✅ Upload your own galaxy images in the web app
5. ✅ Experiment with different mislabel ratios in `main.py`

## Tips

- **Current setup**: Uses real Galaxy Zoo 2 data from `data/`
- **GPU**: Automatically detected and used
- **CPU fallback**: Works but slower
- **Reproducible**: Random seed = 42 everywhere

## Need Help?

Check `README.md` for detailed documentation!

---

**Happy Unlearning!** 🚀🌌
