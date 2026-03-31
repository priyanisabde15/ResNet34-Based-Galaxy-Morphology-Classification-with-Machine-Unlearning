# Machine Unlearning in Galaxy Morphology Classification

A complete implementation of machine unlearning techniques for correcting mislabeled astronomical data in CNN-based galaxy classification models.

## Overview

This project implements and compares three machine unlearning methods to efficiently remove the influence of mislabeled galaxy images from a trained ResNet34 classifier:

1. **Gradient Ascent Unlearning** - Maximizes loss on mislabeled samples
2. **Fisher Forgetting** - Adds calibrated noise based on Fisher Information Matrix
3. **Full Retrain** - Baseline method that retrains from scratch

The system is optimized for RTX 3050 4GB VRAM with mixed precision training and memory-efficient implementations.

## Features

- **3-Class Galaxy Classification**: Smooth, Featured/Disk, Artifact
- **Simulated Noisy Data**: Intentionally mislabels 12% of training data
- **Memory Optimized**: Batch size 16, mixed precision, RTX 3050 4GB friendly
- **Comprehensive Evaluation**: Test accuracy, forget accuracy, retain accuracy, computation time
- **Interactive Web UI**: Streamlit dashboard for classification and unlearning visualization
- **Automatic Fallback**: Uses synthetic galaxy data if real dataset unavailable

## Project Structure

```
.
├── data_loader.py          # Dataset loading and mislabeling injection
├── model.py                # ResNet34-based CNN architecture
├── train.py                # Training pipeline with mixed precision
├── unlearn.py              # Three unlearning method implementations
├── evaluate.py             # Comprehensive evaluation metrics
├── visualize.py            # Plotting and visualization utilities
├── main.py                 # Complete end-to-end pipeline
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Download Galaxy Zoo 2 dataset from Kaggle and organize as:
```
data/
├── Smooth/
├── Featured/
└── Artifact/
```

This project now expects the real Galaxy Zoo 2 training data in `data/`.

## Usage

### Running the Complete Pipeline

Execute the full training and unlearning pipeline:

```bash
python main.py
```

This will:
1. Check hardware and GPU availability
2. Load or generate galaxy dataset
3. Train baseline model on clean data
4. Inject mislabels (12% of training data)
5. Train model on corrupted data
6. Apply all three unlearning methods
7. Evaluate and compare results
8. Generate visualizations in `./plots/`

**Expected runtime**: training depends on epochs; unlearning plus full retrain can take substantially longer on RTX 3050 4GB.

### Running the Web Application

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The web app provides:
- **Galaxy Classifier**: Upload images for real-time classification
- **Unlearning Dashboard**: Run and compare unlearning methods
- **Training Metrics**: View training curves and confusion matrices

Access at: `http://localhost:8501`

## Hardware Requirements

### Minimum
- CPU: Any modern processor
- RAM: 8GB
- GPU: None (CPU fallback available)

### Recommended
- GPU: NVIDIA RTX 3050 (4GB VRAM) or better
- RAM: 16GB
- Storage: 2GB free space

### Memory Optimizations
- Batch size: 16 (safe for 4GB VRAM)
- Mixed precision training (FP16)
- Gradient checkpointing enabled
- Automatic GPU cache clearing
- DataLoader with 2 workers

## Results

The pipeline generates:

### Models
- `baseline_model.pth` - Trained on clean data
- `best_model.pth` - Best trained ResNet34 checkpoint
- `ga_unlearned_model.pth` - After gradient ascent
- `ff_unlearned_model.pth` - After Fisher forgetting
- `retrained_model.pth` - Full retrain baseline

### Logs
- `training_log.csv` - Epoch-by-epoch training metrics
- `unlearning_results.csv` - Latest comparison of unlearning methods

### Visualizations (in `./plots/`)
- `training_curves.png` - Loss and accuracy over epochs
- `accuracy_comparison.png` - Test accuracy across methods
- `forget_accuracy.png` - Forget set accuracy (lower is better)
- `computation_time.png` - Time comparison
- `confusion_matrix_before.png` - Before unlearning
- `confusion_matrix_after.png` - After unlearning
- `sample_predictions.png` - Example classifications

## Key Metrics

The evaluation compares methods on:

- **Test Accuracy**: Performance on clean test set
- **Forget Accuracy**: Accuracy on mislabeled samples (should decrease)
- **Retain Accuracy**: Accuracy on clean training samples (should stay high)
- **Computation Time**: Seconds required for unlearning
- **Memory Usage**: GPU/RAM consumption

## Latest Workspace Results

Latest recorded results in this workspace:

| Method            | Test Acc | Forget Acc | Retain Acc | Time(s) |
|-------------------|----------|------------|------------|---------|
| Gradient Ascent   | 70.80%   | 11.01%     | 76.39%     | 103.6   |
| Fisher Forgetting | 41.71%   | 28.36%     | 43.44%     | 213.8   |
| Full Retrain      | 76.17%   | 8.36%      | 99.64%     | 2649.4  |

Latest training peak from `training_log.csv`:
- Best validation accuracy: `85.58%`
- Best balanced accuracy: `87.56%`
- Best macro-F1: `76.41%`
- Epochs in latest run: `19`

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `data_loader.py` (try 8 or 4)
- Close other GPU applications
- Use CPU mode (automatic fallback)

### Slow Training
- Ensure CUDA is properly installed
- Check GPU utilization with `nvidia-smi`
- Reduce number of epochs for testing

### Module Not Found
```bash
pip install -r requirements.txt --upgrade
```

### Dataset Issues
- Ensure the real Galaxy Zoo 2 dataset is present in `data/`
- Use the saved checkpoint and logs for repeat demos; retraining is not required every time

## Code Quality

- Fully commented and documented
- Modular design with clear separation of concerns
- Comprehensive error handling
- Cross-platform compatible (Windows/Linux/Mac)
- Random seed (42) for reproducibility
- Type hints for better code clarity

## Research Context

This implementation demonstrates:
- Practical machine unlearning in astronomical data
- Handling noisy labels in scientific datasets
- Memory-efficient deep learning on consumer hardware
- Comparison of unlearning method trade-offs

## Citation

If you use this code in your research, please cite:

```
Machine Unlearning in Galaxy Morphology Classification: 
Efficient Correction of Mislabeled Data in CNN-Based Astronomical Models
```

## License

This project is provided for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository.

---

**Built with**: PyTorch, Streamlit, ResNet34, Mixed Precision Training

**Optimized for**: NVIDIA RTX 3050 4GB VRAM

**Status**: Production Ready ✅
