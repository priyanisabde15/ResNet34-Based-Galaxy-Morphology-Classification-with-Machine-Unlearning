"""
Configuration file for Galaxy Morphology Unlearning Project
Centralized settings for easy customization
"""

import torch

# ============================================================================
# HARDWARE SETTINGS
# ============================================================================

# Automatically detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Batch size (reduce if out of memory)
BATCH_SIZE = 16  # Safe for RTX 3050 4GB VRAM
# Try 8 or 4 if you get OOM errors

# Number of data loading workers
NUM_WORKERS = 2

# Enable mixed precision training (MANDATORY for 4GB VRAM)
USE_MIXED_PRECISION = True

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Number of galaxy morphology classes
NUM_CLASSES = 3

# Class names
CLASS_NAMES = ['Smooth', 'Featured/Disk', 'Artifact']

# Use pretrained ImageNet weights
USE_PRETRAINED = True

# Model architecture
MODEL_NAME = 'resnet34'

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

# Number of training epochs
NUM_EPOCHS = 30

# Learning rate
LEARNING_RATE = 0.0003

# Optimizer settings
OPTIMIZER = 'adam'  # 'adam' or 'sgd'
WEIGHT_DECAY = 0.0001

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Dataset directory
DATA_DIR = './data'

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image size
IMAGE_SIZE = 224

# Data augmentation settings
AUGMENTATION = {
    'horizontal_flip': True,
    'rotation_degrees': 15,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2
    }
}

# Use simulated data if real dataset not found
USE_SIMULATION_FALLBACK = True
SIMULATED_SAMPLES = 3000

# ============================================================================
# MISLABELING SETTINGS
# ============================================================================

# Percentage of training data to intentionally mislabel
MISLABEL_RATIO = 0.12  # 12%

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# UNLEARNING SETTINGS
# ============================================================================

# Gradient Ascent Unlearning
GRADIENT_ASCENT = {
    'num_epochs': 5,
    'learning_rate': 0.0001
}

# Fisher Forgetting
FISHER_FORGETTING = {
    'noise_scale': 0.01
}

# Full Retrain
FULL_RETRAIN = {
    'num_epochs': 15,
    'learning_rate': 0.001
}

# ============================================================================
# FILE PATHS
# ============================================================================

# Model checkpoints
BASELINE_MODEL_PATH = 'baseline_model.pth'
CORRUPTED_MODEL_PATH = 'corrupted_model.pth'
BEST_MODEL_PATH = 'best_model.pth'
GA_MODEL_PATH = 'ga_unlearned_model.pth'
FF_MODEL_PATH = 'ff_unlearned_model.pth'
RETRAIN_MODEL_PATH = 'retrained_model.pth'

# Logs
TRAINING_LOG_PATH = 'training_log.csv'
BASELINE_LOG_PATH = 'baseline_training_log.csv'
RESULTS_PATH = 'results.csv'

# Plots directory
PLOTS_DIR = 'plots'

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Plot style
PLOT_STYLE = 'whitegrid'

# Figure DPI
PLOT_DPI = 150

# Color scheme
COLORS = {
    'train': '#3498DB',
    'val': '#E74C3C',
    'test': '#2ECC71',
    'method1': '#FF6B6B',
    'method2': '#4ECDC4',
    'method3': '#45B7D1',
    'method4': '#96CEB4'
}

# ============================================================================
# STREAMLIT SETTINGS
# ============================================================================

# Page configuration
STREAMLIT_CONFIG = {
    'page_title': 'Galaxy Morphology Unlearning',
    'page_icon': '🌌',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Theme
STREAMLIT_THEME = 'dark'

# ============================================================================
# MEMORY OPTIMIZATION FLAGS
# ============================================================================

# Enable gradient checkpointing
USE_GRADIENT_CHECKPOINTING = True

# Clear GPU cache after each epoch
CLEAR_CACHE_AFTER_EPOCH = True

# Pin memory for data loaders (faster GPU transfer)
PIN_MEMORY = torch.cuda.is_available()

# Persistent workers for data loaders
PERSISTENT_WORKERS = NUM_WORKERS > 0

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Verbosity level
VERBOSE = True

# Progress bars
USE_PROGRESS_BARS = True

# Print frequency (batches)
PRINT_FREQ = 10

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Metrics to compute
METRICS = [
    'accuracy',
    'loss',
    'confusion_matrix',
    'classification_report'
]

# Save predictions
SAVE_PREDICTIONS = False

# Number of sample images to visualize
NUM_SAMPLE_IMAGES = 9

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary():
    """Print configuration summary"""
    print("\n" + "="*60)
    print("  CONFIGURATION SUMMARY")
    print("="*60)
    print(f"\nHardware:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
    
    print(f"\nTraining:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Optimizer: {OPTIMIZER}")
    
    print(f"\nData:")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Mislabel Ratio: {MISLABEL_RATIO*100:.1f}%")
    print(f"  Train/Val/Test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    print(f"\nUnlearning:")
    print(f"  Gradient Ascent Epochs: {GRADIENT_ASCENT['num_epochs']}")
    print(f"  Fisher Noise Scale: {FISHER_FORGETTING['noise_scale']}")
    
    print("="*60 + "\n")


def validate_config():
    """Validate configuration settings"""
    assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Split ratios must sum to 1.0"
    assert 0 < MISLABEL_RATIO < 1.0, "Mislabel ratio must be between 0 and 1"
    assert BATCH_SIZE > 0, "Batch size must be positive"
    assert NUM_EPOCHS > 0, "Number of epochs must be positive"
    assert LEARNING_RATE > 0, "Learning rate must be positive"
    
    print("✅ Configuration validated successfully")


if __name__ == "__main__":
    # Test configuration
    get_config_summary()
    validate_config()
