"""
Visualization utilities for machine unlearning results
Generates plots for training curves, accuracy comparisons, and confusion matrices
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import torch
from torch.utils.data import DataLoader

from model import GalaxyCNN

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create plots directory
os.makedirs('plots', exist_ok=True)


def plot_training_curves(log_path: str = 'training_log.csv',
                         save_path: str = 'plots/training_curves.png'):
    """
    Plot training and validation loss curves
    
    Args:
        log_path: Path to training log CSV
        save_path: Path to save plot
    """
    if not os.path.exists(log_path):
        print(f"⚠️  Training log not found: {log_path}")
        return
    
    df = pd.read_csv(log_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(df['epoch'], df['val_accuracy'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to {save_path}")


def plot_accuracy_comparison(results_df: pd.DataFrame,
                            save_path: str = 'plots/accuracy_comparison.png'):
    """
    Bar chart comparing test accuracy before and after unlearning
    
    Args:
        results_df: DataFrame with results
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = results_df['method'].tolist()
    test_acc = results_df['test_accuracy'].tolist()
    
    x = np.arange(len(methods))
    bars = ax.bar(x, test_acc, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Accuracy comparison saved to {save_path}")


def plot_forget_accuracy_comparison(results_df: pd.DataFrame,
                                   save_path: str = 'plots/forget_accuracy.png'):
    """
    Bar chart comparing forget set accuracy (lower is better)
    
    Args:
        results_df: DataFrame with results
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = results_df['method'].tolist()
    forget_acc = results_df['forget_accuracy'].tolist()
    
    x = np.arange(len(methods))
    bars = ax.bar(x, forget_acc, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Forget Set Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Forget Set Accuracy (Lower = Better Unlearning)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Forget accuracy comparison saved to {save_path}")


def plot_computation_time(results_df: pd.DataFrame,
                         save_path: str = 'plots/computation_time.png'):
    """
    Bar chart comparing computation time
    
    Args:
        results_df: DataFrame with results
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = results_df['method'].tolist()
    times = results_df['computation_time_seconds'].tolist()
    
    x = np.arange(len(methods))
    bars = ax.bar(x, times, color=['#9B59B6', '#1ABC9C', '#E67E22', '#34495E'],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Computation time comparison saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         title: str = 'Confusion Matrix',
                         save_path: str = 'plots/confusion_matrix.png'):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {save_path}")


def plot_sample_predictions(model: GalaxyCNN,
                           data_loader: DataLoader,
                           class_names: List[str],
                           device: torch.device,
                           num_samples: int = 9,
                           save_path: str = 'plots/sample_predictions.png'):
    """
    Plot sample galaxy images with predictions
    
    Args:
        model: Trained model
        data_loader: Data loader
        class_names: List of class names
        device: Device
        num_samples: Number of samples to show
        save_path: Path to save plot
    """
    model.eval()
    
    # Get a batch
    images, labels = next(iter(data_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = outputs.max(1)
    
    predictions = predictions.cpu().numpy()
    labels = labels.numpy()
    
    # Denormalize images for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(min(num_samples, 9)):
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}',
                           color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sample predictions saved to {save_path}")


def plot_all_metrics(results_df: pd.DataFrame,
                    save_path: str = 'plots/all_metrics.png'):
    """
    Comprehensive plot with all key metrics
    
    Args:
        results_df: DataFrame with results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = results_df['method'].tolist()
    x = np.arange(len(methods))
    
    # Test Accuracy
    axes[0, 0].bar(x, results_df['test_accuracy'], color='#3498DB', edgecolor='black')
    axes[0, 0].set_title('Test Accuracy', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Forget Accuracy
    axes[0, 1].bar(x, results_df['forget_accuracy'], color='#E74C3C', edgecolor='black')
    axes[0, 1].set_title('Forget Accuracy (Lower = Better)', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods, rotation=15, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Retain Accuracy
    axes[1, 0].bar(x, results_df['retain_accuracy'], color='#2ECC71', edgecolor='black')
    axes[1, 0].set_title('Retain Accuracy (Higher = Better)', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, rotation=15, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Computation Time
    axes[1, 1].bar(x, results_df['computation_time_seconds'], color='#9B59B6', edgecolor='black')
    axes[1, 1].set_title('Computation Time', fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All metrics plot saved to {save_path}")


def generate_all_plots(results_csv: str = 'results.csv',
                      training_log: str = 'training_log.csv'):
    """
    Generate all visualization plots
    
    Args:
        results_csv: Path to results CSV
        training_log: Path to training log CSV
    """
    print("\n🎨 Generating all visualizations...")
    
    # Load results
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        
        plot_accuracy_comparison(results_df)
        plot_forget_accuracy_comparison(results_df)
        plot_computation_time(results_df)
        plot_all_metrics(results_df)
    else:
        print(f"⚠️  Results file not found: {results_csv}")
    
    # Training curves
    if os.path.exists(training_log):
        plot_training_curves(training_log)
    else:
        print(f"⚠️  Training log not found: {training_log}")
    
    print("\n✅ All visualizations generated in ./plots/")


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization module...")
    
    # Create dummy results
    dummy_results = pd.DataFrame({
        'method': ['No Unlearning', 'Gradient Ascent', 'Fisher Forgetting', 'Full Retrain'],
        'test_accuracy': [85.5, 84.2, 83.8, 86.1],
        'forget_accuracy': [78.3, 45.2, 52.1, 38.5],
        'retain_accuracy': [87.2, 86.5, 85.9, 87.8],
        'computation_time_seconds': [0, 45.2, 32.1, 180.5]
    })
    
    plot_accuracy_comparison(dummy_results)
    plot_forget_accuracy_comparison(dummy_results)
    plot_computation_time(dummy_results)
    plot_all_metrics(dummy_results)
    
    print("\n✅ Visualization test complete!")
