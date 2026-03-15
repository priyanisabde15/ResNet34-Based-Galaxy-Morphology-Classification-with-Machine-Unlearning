"""
Main Pipeline for Machine Unlearning in Galaxy Morphology Classification
Runs complete end-to-end workflow
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Import all modules
from data_loader import get_data_loaders, inject_mislabels
from model import create_model, load_model
from train import train_model
from unlearn import UnlearningMethods, create_forget_retain_loaders
from evaluate import comprehensive_evaluation, compare_all_methods, print_comparison_table, get_confusion_matrix
from visualize import generate_all_plots, plot_confusion_matrix, plot_sample_predictions

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def print_header(text: str):
    """Print a nice header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def check_hardware():
    """Check and print hardware information"""
    print_header("HARDWARE INFORMATION")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {device_name}")
        print(f"   Total VRAM: {total_memory:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        if total_memory <= 4.5:
            print(f"   ⚠️  Limited VRAM detected - Memory optimizations enabled")
    else:
        print("⚠️  No GPU detected - Using CPU")
        print("   Training will be slower on CPU")
    
    print(f"\n   PyTorch Version: {torch.__version__}")
    print(f"   Device: {'cuda' if cuda_available else 'cpu'}")
    
    return torch.device('cuda' if cuda_available else 'cpu')


def main():
    """Main pipeline execution"""
    
    print("\n" + "🌌"*40)
    print("  MACHINE UNLEARNING IN GALAXY MORPHOLOGY CLASSIFICATION")
    print("  Research Paper Implementation")
    print("🌌"*40)
    
    start_time = datetime.now()
    
    # Step 1: Check hardware
    device = check_hardware()
    
    # Step 2: Load and preprocess data
    print_header("STEP 1: DATA LOADING")
    
    train_loader, val_loader, test_loader, dataset, train_indices, class_weights = get_data_loaders(
        data_dir='./data',
        batch_size=16,
        num_workers=2
    )
    
    class_names = dataset.classes
    
    # Step 3: Train baseline model on clean data
    print_header("STEP 2: TRAINING BASELINE MODEL (Clean Data)")
    
    baseline_model = create_model(device=device)
    
    if not os.path.exists('baseline_model.pth'):
        print("Training baseline model on clean data...")
        baseline_model = train_model(
            baseline_model,
            train_loader,
            val_loader,
            num_epochs=30,
            learning_rate=0.0003,
            device=device,
            save_path='baseline_model.pth',
            log_path='baseline_training_log.csv',
            class_weights=class_weights,
            early_stopping_patience=10
        )
    else:
        print("✅ Baseline model already exists, loading...")
        baseline_model = load_model('baseline_model.pth', device=device)
    
    # Step 4: Inject mislabels
    print_header("STEP 3: INJECTING MISLABELS")
    
    mislabeled_indices, original_labels = inject_mislabels(
        dataset,
        mislabel_ratio=0.12
    )
    
    # Recreate data loaders with mislabeled data
    train_loader_mislabeled, val_loader, test_loader, dataset, train_indices, class_weights = get_data_loaders(
        data_dir='./data',
        batch_size=16,
        num_workers=2
    )
    
    # Step 5: Train model with mislabeled data
    print_header("STEP 4: TRAINING WITH MISLABELED DATA")
    
    corrupted_model = create_model(device=device)
    
    if not os.path.exists('corrupted_model.pth'):
        print("Training model on corrupted data...")
        corrupted_model = train_model(
            corrupted_model,
            train_loader_mislabeled,
            val_loader,
            num_epochs=30,
            learning_rate=0.0003,
            device=device,
            save_path='corrupted_model.pth',
            log_path='training_log.csv',
            class_weights=class_weights,
            early_stopping_patience=10
        )
    else:
        print("✅ Corrupted model already exists, loading...")
        corrupted_model = load_model('corrupted_model.pth', device=device)
    
    # Step 6: Create forget and retain sets
    print_header("STEP 5: PREPARING UNLEARNING DATASETS")
    
    forget_loader, retain_loader = create_forget_retain_loaders(
        dataset,
        train_indices,
        mislabeled_indices,
        batch_size=16
    )
    
    # Step 7: Apply unlearning methods
    print_header("STEP 6: APPLYING UNLEARNING METHODS")
    
    unlearner = UnlearningMethods(corrupted_model, device)
    
    all_results = []
    
    # Evaluate corrupted model (no unlearning)
    print("\n--- Evaluating: No Unlearning (Baseline) ---")
    no_unlearn_results = comprehensive_evaluation(
        corrupted_model,
        test_loader,
        forget_loader,
        retain_loader,
        device,
        "No Unlearning",
        0.0
    )
    all_results.append(no_unlearn_results)
    
    # Method 1: Gradient Ascent
    print("\n--- Method 1: Gradient Ascent Unlearning ---")
    ga_model, ga_time = unlearner.gradient_ascent_unlearning(
        forget_loader,
        num_epochs=5,
        learning_rate=0.0001
    )
    ga_results = comprehensive_evaluation(
        ga_model,
        test_loader,
        forget_loader,
        retain_loader,
        device,
        "Gradient Ascent",
        ga_time
    )
    all_results.append(ga_results)
    
    # Save gradient ascent model
    torch.save(ga_model.state_dict(), 'ga_unlearned_model.pth')
    
    # Method 2: Fisher Forgetting
    print("\n--- Method 2: Fisher Forgetting ---")
    ff_model, ff_time = unlearner.fisher_forgetting(
        retain_loader,
        noise_scale=0.01
    )
    ff_results = comprehensive_evaluation(
        ff_model,
        test_loader,
        forget_loader,
        retain_loader,
        device,
        "Fisher Forgetting",
        ff_time
    )
    all_results.append(ff_results)
    
    # Save fisher model
    torch.save(ff_model.state_dict(), 'ff_unlearned_model.pth')
    
    # Method 3: Full Retrain
    print("\n--- Method 3: Full Retrain (Gold Standard) ---")
    
    # Create clean training loader (exclude mislabeled samples)
    from torch.utils.data import Subset
    clean_train_indices = [idx for idx in train_indices if idx not in mislabeled_indices]
    clean_train_dataset = Subset(dataset, clean_train_indices)
    
    from torch.utils.data import DataLoader
    clean_train_loader = DataLoader(
        clean_train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    retrain_model, retrain_time = unlearner.full_retrain(
        clean_train_loader,
        val_loader,
        num_epochs=15,
        learning_rate=0.001
    )
    retrain_results = comprehensive_evaluation(
        retrain_model,
        test_loader,
        forget_loader,
        retain_loader,
        device,
        "Full Retrain",
        retrain_time
    )
    all_results.append(retrain_results)
    
    # Save retrained model
    torch.save(retrain_model.state_dict(), 'retrained_model.pth')
    
    # Step 8: Compare all methods
    print_header("STEP 7: RESULTS COMPARISON")
    
    results_df = compare_all_methods(all_results, save_path='results.csv')
    print_comparison_table(results_df)
    
    # Step 9: Generate visualizations
    print_header("STEP 8: GENERATING VISUALIZATIONS")
    
    generate_all_plots('results.csv', 'training_log.csv')
    
    # Generate confusion matrices
    print("\nGenerating confusion matrices...")
    
    cm_before, _ = get_confusion_matrix(corrupted_model, test_loader, device, class_names)
    plot_confusion_matrix(cm_before, class_names, 
                         'Confusion Matrix - Before Unlearning',
                         'plots/confusion_matrix_before.png')
    
    cm_after, _ = get_confusion_matrix(ga_model, test_loader, device, class_names)
    plot_confusion_matrix(cm_after, class_names,
                         'Confusion Matrix - After Gradient Ascent',
                         'plots/confusion_matrix_after.png')
    
    # Sample predictions
    plot_sample_predictions(ga_model, test_loader, class_names, device,
                           save_path='plots/sample_predictions.png')
    
    # Step 10: Final summary
    print_header("PIPELINE COMPLETE")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print(f"✅ All tasks completed successfully!")
    print(f"   Total execution time: {total_time/60:.1f} minutes")
    print(f"\n📁 Generated Files:")
    print(f"   • Models: baseline_model.pth, corrupted_model.pth, *_unlearned_model.pth")
    print(f"   • Results: results.csv, training_log.csv")
    print(f"   • Plots: ./plots/ directory")
    print(f"\n🚀 Next Steps:")
    print(f"   • Run Streamlit app: streamlit run app.py")
    print(f"   • View results: cat results.csv")
    print(f"   • Check plots: ls plots/")
    
    print("\n" + "🌌"*40 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
