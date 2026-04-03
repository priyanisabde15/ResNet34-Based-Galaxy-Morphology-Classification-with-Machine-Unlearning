"""
Training script for Galaxy Morphology CNN
Optimized for RTX 3050 4GB VRAM with mixed precision training
"""

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional
import psutil
from sklearn.metrics import balanced_accuracy_score, f1_score

from model import GalaxyCNN, create_model, save_model, load_model
from data_loader import get_data_loaders, inject_mislabels
from unlearn import UnlearningMethods, create_forget_retain_loaders
from evaluate import evaluate_model, comprehensive_evaluation, compare_all_methods

def set_global_seed(seed: int = 42) -> None:
    """Keep runs as reproducible as practical across train/eval/unlearning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_global_seed(42)


class FocalLoss(nn.Module):
    """Cross-entropy variant that focuses learning on harder examples."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_memory_usage() -> Tuple[float, float]:
    """Get current GPU and RAM memory usage in MB"""
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    
    ram_mem = psutil.Process().memory_info().rss / 1024 / 1024
    return gpu_mem, ram_mem


def build_device(force_gpu: bool = False) -> torch.device:
    """Select device with optional strict GPU requirement."""
    if force_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Install CUDA-enabled PyTorch and start again.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📡 Using device: {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")
    if device.type == 'cuda':
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDNN enabled: {torch.backends.cudnn.enabled}")

    return device


def check_fit_conditions(history: dict, min_val_acc: float = 80.0) -> None:
    """Check for underfitting/overfitting risk from training history."""
    if len(history['train_loss']) == 0 or len(history['val_loss']) == 0:
        return

    recent_val = history['val_accuracy'][-1]
    if recent_val < min_val_acc:
        print(f"⚠️ Validation accuracy {recent_val:.2f}% < target {min_val_acc:.1f}%; consider more epochs/stronger augmentation.")

    if history['train_loss'][-1] < history['val_loss'][-1]:
        print("⚠️ Potential overfitting: train loss < val loss at final epoch.")
    elif history['train_loss'][-1] > history['val_loss'][-1]:
        print("⚠️ Potential underfitting: train loss > val loss at final epoch.")


def train_epoch(model: GalaxyCNN,
                train_loader,
                criterion,
                optimizer,
                device: torch.device,
                scaler: GradScaler,
                epoch: int) -> float:
    """
    Train for one epoch with mixed precision
    
    Args:
        model: Galaxy CNN model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        scaler: Gradient scaler for mixed precision
        epoch: Current epoch number
    
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training (GPU only for speed)
        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        
        # Clear cache periodically to prevent memory buildup
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model: GalaxyCNN,
            val_loader,
            criterion,
            device: torch.device) -> Tuple[float, float, float, float]:
    """
    Validate the model
    
    Args:
        model: Galaxy CNN model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device (cuda/cpu)
    
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            # Use mixed precision for inference too (GPU only)
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    balanced_acc = 100.0 * balanced_accuracy_score(all_labels, all_predictions)
    macro_f1 = 100.0 * f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    return avg_loss, accuracy, balanced_acc, macro_f1


def train_model(model: GalaxyCNN,
                train_loader,
                val_loader,
                num_epochs: int = 50,
                learning_rate: float = 1e-4,
                device: Optional[torch.device] = None,
                save_path: str = 'best_model.pth',
                log_path: str = 'training_log.csv',
                class_weights: Optional[torch.Tensor] = None,
                target_acc: float = 80.0,
                early_stopping_patience: int = 8,
                freeze_backbone_epochs: int = 3,
                use_focal_loss: bool = True) -> GalaxyCNN:
    """
    Complete training pipeline with logging
    
    Args:
        model: Galaxy CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device (cuda/cpu)
        save_path: Path to save best model
        log_path: Path to save training logs
        class_weights: Optional per-class weights for imbalanced dataset
        target_acc: Break training when this validation accuracy is reached
        early_stopping_patience: Number of epochs to wait for improvement
    
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)

    for _, param in model.backbone.named_parameters():
        param.requires_grad = True

    if freeze_backbone_epochs > 0:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = name.startswith('fc')
    
    # Loss function and optimizer
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=1.5)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Mixed precision scaler - active only on CUDA
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Learning rate scheduler (cycle/annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_balanced_accuracy': [],
        'val_macro_f1': [],
        'learning_rate': [],
        'gpu_memory_mb': [],
        'time_seconds': []
    }

    best_val_acc = 0.0
    best_score = float('-inf')
    epochs_no_improve = 0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Mixed precision: ✅ Enabled")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"   Target accuracy for auto-stop: {target_acc:.2f}%")
    print(f"   Frozen-backbone warmup epochs: {freeze_backbone_epochs}")
    print(f"   Loss: {'FocalLoss' if use_focal_loss else 'CrossEntropyLoss'}")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            for _, param in model.backbone.named_parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.2, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, num_epochs - freeze_backbone_epochs), eta_min=1e-6
            )
            print("\n🔓 Unfroze backbone for fine-tuning with a lower learning rate.")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Validate
        val_loss, val_acc, val_balanced_acc, val_macro_f1 = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Get memory usage
        gpu_mem, _ = get_memory_usage()
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_balanced_accuracy'].append(val_balanced_acc)
        history['val_macro_f1'].append(val_macro_f1)
        history['learning_rate'].append(current_lr)
        history['gpu_memory_mb'].append(gpu_mem)
        history['time_seconds'].append(epoch_time)
        
        print(f"\n📊 Epoch {epoch}/{num_epochs} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val Accuracy: {val_acc:.2f}%")
        print(f"   Val Balanced Accuracy: {val_balanced_acc:.2f}%")
        print(f"   Val Macro-F1: {val_macro_f1:.2f}%")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   GPU Memory: {gpu_mem:.1f} MB")
        
        # Save best model
        selection_score = (0.60 * val_acc) + (0.25 * val_balanced_acc) + (0.15 * val_macro_f1)
        if selection_score > best_score:
            best_score = selection_score
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_model(
                model, save_path, optimizer, epoch,
                {
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'val_balanced_accuracy': val_balanced_acc,
                    'val_macro_f1': val_macro_f1,
                    'selection_score': selection_score
                }
            )
            print(f"   ⭐ New best model saved! Selection score: {selection_score:.2f}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{early_stopping_patience} epochs")
        
        # Auto-stop when target reached
        if val_acc >= target_acc:
            print(f"\n🥳 Target accuracy reached ({val_acc:.2f}%), stopping early.")
            break
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n⛔ Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break
        
        # Clear GPU cache after each epoch - CRITICAL
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save training log
    df = pd.DataFrame(history)
    df.to_csv(log_path, index=False)
    print(f"\n💾 Training log saved to {log_path}")
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Galaxy morphology training + unlearning evaluation')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--require-gpu', action='store_true', help='Fail if GPU is not available')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--backbone', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet34')
    parser.add_argument('--target-acc', type=float, default=80.0)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--mislabel-ratio', type=float, default=0.12)
    parser.add_argument('--unlearn-results-path', type=str, default='unlearning_results.csv')
    parser.add_argument('--no-unlearning', action='store_true', help='Skip unlearning tests if set')

    args = parser.parse_args()

    if args.device == 'cuda':
        device = build_device(force_gpu=True)
    elif args.device == 'cpu':
        device = torch.device('cpu')
        print('\n📡 Using CPU by request.')
    else:
        device = build_device(force_gpu=args.require_gpu)

    # Data load
    print('\n📥 Loading data...')
    train_loader, val_loader, test_loader, dataset, train_indices, class_weights = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=2
    )

    # Model
    model = create_model(device=device, backbone=args.backbone, pretrained=True)

    # Train
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_path='best_model.pth',
        log_path='training_log.csv',
        class_weights=class_weights,
        target_acc=args.target_acc,
        early_stopping_patience=args.patience
    )

    # Always evaluate and unlearn from the best checkpoint, not just the last in-memory epoch.
    print('\n📦 Reloading best checkpoint for final evaluation and unlearning...')
    best_model = load_model('best_model.pth', device=device)

    # Evaluate final accuracy
    print('\n📈 Final model evaluation on test set...')
    test_acc, test_loss, _, _ = evaluate_model(best_model, test_loader, device, desc='Final Test')
    print(f"\n✅ Final Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    if test_acc < args.target_acc:
        print(f"⚠️ Warning: final test accuracy below target {args.target_acc:.1f}%")

    if not args.no_unlearning:
        print('\n🔐 Running unlearning suite...')
        mislabeled_indices, _ = inject_mislabels(dataset, mislabel_ratio=args.mislabel_ratio)
        forget_loader, retain_loader = create_forget_retain_loaders(dataset, train_indices, mislabeled_indices, batch_size=args.batch_size)

        unlearner = UnlearningMethods(best_model, device)
        results = []

        ga_model, ga_time = unlearner.gradient_ascent_unlearning(forget_loader, num_epochs=3, learning_rate=1e-4)
        results.append(comprehensive_evaluation(ga_model, test_loader, forget_loader, retain_loader, device, 'GradientAscent', ga_time))

        ff_model, ff_time = unlearner.fisher_forgetting(retain_loader, noise_scale=0.01)
        results.append(comprehensive_evaluation(ff_model, test_loader, forget_loader, retain_loader, device, 'FisherForgetting', ff_time))

        fr_model, fr_time = unlearner.full_retrain(train_loader, val_loader, num_epochs=10, learning_rate=args.lr)
        results.append(comprehensive_evaluation(fr_model, test_loader, forget_loader, retain_loader, device, 'FullRetrain', fr_time))

        compare_all_methods(results, save_path=args.unlearn_results_path)

    print('\n✅ Run complete. Training and unlearning workflows finished successfully.')
