"""
Evaluation metrics for machine unlearning
Measures test accuracy, forget accuracy, retain accuracy, time, and memory
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import psutil

from model import GalaxyCNN

# Set random seed
torch.manual_seed(42)


def evaluate_model(model: GalaxyCNN,
                  data_loader: DataLoader,
                  device: torch.device,
                  desc: str = "Evaluating") -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset
    
    Args:
        model: Galaxy CNN model
        data_loader: Data loader
        device: Device (cuda/cpu)
        desc: Description for progress bar
    
    Returns:
        accuracy, loss, all_predictions, all_labels
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    avg_loss = running_loss / len(data_loader)
    
    return accuracy, avg_loss, np.array(all_predictions), np.array(all_labels)


def get_memory_usage() -> Tuple[float, float]:
    """Get GPU and RAM memory usage in MB"""
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    
    ram_mem = psutil.Process().memory_info().rss / 1024 / 1024
    return gpu_mem, ram_mem


def comprehensive_evaluation(model: GalaxyCNN,
                            test_loader: DataLoader,
                            forget_loader: DataLoader,
                            retain_loader: DataLoader,
                            device: torch.device,
                            method_name: str,
                            computation_time: float) -> Dict:
    """
    Comprehensive evaluation of an unlearning method
    
    Args:
        model: Model to evaluate
        test_loader: Test set loader
        forget_loader: Forget set loader (mislabeled samples)
        retain_loader: Retain set loader (clean samples)
        device: Device (cuda/cpu)
        method_name: Name of unlearning method
        computation_time: Time taken for unlearning
    
    Returns:
        Dictionary with all metrics
    """
    print(f"\n📊 Evaluating: {method_name}")
    
    # Test set accuracy
    test_acc, test_loss, test_preds, test_labels = evaluate_model(
        model, test_loader, device, desc="Test Set"
    )
    
    # Forget set accuracy (should be LOW after unlearning)
    forget_acc, forget_loss, _, _ = evaluate_model(
        model, forget_loader, device, desc="Forget Set"
    )
    
    # Retain set accuracy (should stay HIGH after unlearning)
    retain_acc, retain_loss, _, _ = evaluate_model(
        model, retain_loader, device, desc="Retain Set"
    )
    
    # Memory usage
    gpu_mem, ram_mem = get_memory_usage()
    
    results = {
        'method': method_name,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'forget_accuracy': forget_acc,
        'forget_loss': forget_loss,
        'retain_accuracy': retain_acc,
        'retain_loss': retain_loss,
        'computation_time_seconds': computation_time,
        'gpu_memory_mb': gpu_mem,
        'ram_memory_mb': ram_mem
    }
    
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Forget Accuracy: {forget_acc:.2f}% (lower is better)")
    print(f"   Retain Accuracy: {retain_acc:.2f}% (higher is better)")
    print(f"   Time: {computation_time:.2f}s")
    print(f"   GPU Memory: {gpu_mem:.1f} MB")
    
    return results


def compare_all_methods(results_list: List[Dict],
                       save_path: str = 'results.csv') -> pd.DataFrame:
    """
    Compare all unlearning methods and save results
    
    Args:
        results_list: List of result dictionaries
        save_path: Path to save CSV
    
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results_list)
    
    # Reorder columns for better readability
    column_order = [
        'method',
        'test_accuracy',
        'forget_accuracy',
        'retain_accuracy',
        'computation_time_seconds',
        'gpu_memory_mb',
        'test_loss',
        'forget_loss',
        'retain_loss',
        'ram_memory_mb'
    ]
    
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\n💾 Results saved to {save_path}")
    
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print a nice comparison table"""
    print("\n" + "="*80)
    print("MACHINE UNLEARNING COMPARISON")
    print("="*80)
    print(f"\n{'Method':<20} {'Test Acc':<12} {'Forget Acc':<12} {'Retain Acc':<12} {'Time(s)':<10}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<20} {row['test_accuracy']:>10.2f}% "
              f"{row['forget_accuracy']:>10.2f}% {row['retain_accuracy']:>10.2f}% "
              f"{row['computation_time_seconds']:>8.1f}")
    
    print("="*80)
    print("\n📌 Key Insights:")
    print("   • Forget Accuracy: Lower is better (model forgot mislabeled data)")
    print("   • Retain Accuracy: Higher is better (model retained clean data)")
    print("   • Test Accuracy: Overall performance on clean test set")
    print("="*80 + "\n")


def get_confusion_matrix(model: GalaxyCNN,
                        data_loader: DataLoader,
                        device: torch.device,
                        class_names: List[str]) -> Tuple[np.ndarray, str]:
    """
    Generate confusion matrix and classification report
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device
        class_names: List of class names
    
    Returns:
        Confusion matrix and classification report string
    """
    _, _, predictions, labels = evaluate_model(model, data_loader, device)
    
    cm = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, target_names=class_names)
    
    return cm, report


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation module...")
    
    from model import create_model
    from data_loader import get_data_loaders, inject_mislabels
    from unlearn import create_forget_retain_loaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, test_loader, dataset, train_indices, class_weights = get_data_loaders()
    
    # Inject mislabels
    mislabeled_indices, _ = inject_mislabels(dataset, mislabel_ratio=0.1)
    
    # Create forget/retain loaders
    forget_loader, retain_loader = create_forget_retain_loaders(
        dataset, train_indices, mislabeled_indices
    )
    
    # Create dummy model
    model = create_model(device=device)
    
    # Test evaluation
    results = comprehensive_evaluation(
        model, test_loader, forget_loader, retain_loader,
        device, "Test Method", 10.5
    )
    
    print("\n✅ Evaluation test complete!")
    print(f"Results: {results}")
