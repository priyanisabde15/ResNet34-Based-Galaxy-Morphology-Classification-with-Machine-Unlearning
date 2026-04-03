"""
Machine Unlearning Methods for Galaxy Classification
Implements: Gradient Ascent, Fisher Forgetting, and Full Retrain
"""

import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import psutil
from typing import List, Tuple, Optional
from torchvision import transforms

from model import GalaxyCNN, create_model, load_model
from data_loader import GalaxySubset, get_data_loaders, inject_mislabels
from evaluate import comprehensive_evaluation, compare_all_methods

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


def _build_eval_transform():
    """Evaluation transform shared by unlearning loaders."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_memory_usage() -> Tuple[float, float]:
    """Get current GPU and RAM memory usage in MB"""
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    ram_mem = psutil.Process().memory_info().rss / 1024 / 1024
    return gpu_mem, ram_mem


class UnlearningMethods:
    """Collection of machine unlearning algorithms"""
    
    def __init__(self, model: GalaxyCNN, device: torch.device):
        """
        Args:
            model: Trained galaxy classification model
            device: Device (cuda/cpu)
        """
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def gradient_ascent_unlearning(self,
                                   forget_loader: DataLoader,
                                   num_epochs: int = 3,
                                   learning_rate: float = 0.0001) -> Tuple[GalaxyCNN, float]:
        """
        Gradient Ascent Unlearning: Maximize loss on forget set
        
        This method intentionally increases the loss on mislabeled samples,
        effectively "forgetting" the incorrect patterns learned from them.
        
        Args:
            forget_loader: DataLoader for samples to forget (mislabeled data)
            num_epochs: Number of unlearning epochs
            learning_rate: Learning rate for gradient ascent
        
        Returns:
            Unlearned model and computation time
        """
        print("\n🔄 Starting Gradient Ascent Unlearning...")
        start_time = time.time()
        
        # Create a copy to avoid modifying original
        unlearned_model = copy.deepcopy(self.model)
        unlearned_model.train()
        
        optimizer = optim.Adam(unlearned_model.parameters(), lr=learning_rate)
        scaler = GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            
            pbar = tqdm(forget_loader, desc=f'Unlearn Epoch {epoch}/{num_epochs}')
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    outputs = unlearned_model(images)
                    loss = self.criterion(outputs, labels)
                    
                    # KEY: Negate loss for gradient ASCENT (maximize loss)
                    ascent_loss = -loss
                
                # Backward pass
                scaler.scale(ascent_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(forget_loader)
            print(f"   Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        computation_time = time.time() - start_time
        print(f"✅ Gradient Ascent complete in {computation_time:.2f}s")
        
        return unlearned_model, computation_time
    
    def fisher_forgetting(self,
                         retain_loader: DataLoader,
                         noise_scale: float = 0.01) -> Tuple[GalaxyCNN, float]:
        """
        Fisher Forgetting: Add calibrated noise to weights based on Fisher Information
        
        This method computes the Fisher Information Matrix (diagonal approximation)
        on the retain set, then adds noise inversely proportional to importance.
        
        Args:
            retain_loader: DataLoader for samples to retain (clean data)
            noise_scale: Scale of noise to add
        
        Returns:
            Unlearned model and computation time
        """
        print("\n🔄 Starting Fisher Forgetting...")
        start_time = time.time()
        
        # Create a copy
        unlearned_model = copy.deepcopy(self.model)
        unlearned_model.eval()
        
        # Step 1: Compute Fisher Information Matrix (diagonal)
        print("   Computing Fisher Information Matrix...")
        fisher_dict = {}
        
        for name, param in unlearned_model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        num_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(retain_loader, desc='Computing Fisher'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Enable gradients temporarily
                unlearned_model.zero_grad()
                
                with torch.enable_grad():
                    with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                        outputs = unlearned_model(images)
                        loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                
                # Accumulate squared gradients (Fisher approximation)
                for name, param in unlearned_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_dict[name] += param.grad.pow(2)
                
                num_samples += images.size(0)
        
        # Average Fisher information
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        
        # Step 2: Add calibrated noise to weights
        print("   Adding calibrated noise to weights...")
        
        with torch.no_grad():
            for name, param in unlearned_model.named_parameters():
                if name in fisher_dict:
                    # Noise inversely proportional to Fisher information
                    # High Fisher = important for retain set = less noise
                    fisher_info = fisher_dict[name] + 1e-8  # Avoid division by zero
                    noise = torch.randn_like(param) * noise_scale / torch.sqrt(fisher_info)
                    param.add_(noise)
        
        computation_time = time.time() - start_time
        print(f"✅ Fisher Forgetting complete in {computation_time:.2f}s")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return unlearned_model, computation_time
    
    def full_retrain(self,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    num_epochs: int = 10,
                    learning_rate: float = 0.0002) -> Tuple[GalaxyCNN, float]:
        """
        Full Retrain: Train from scratch on clean data (gold standard)
        
        This is the baseline method - retrain the model completely
        excluding the mislabeled samples.
        
        Args:
            train_loader: Clean training data (mislabeled samples removed)
            val_loader: Validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Retrained model and computation time
        """
        print("\n🔄 Starting Full Retrain (Baseline)...")
        start_time = time.time()
        
        # Create fresh model
        from model import create_model
        retrained_model = create_model(device=self.device, pretrained=True)
        retrained_model.train()
        
        optimizer = optim.Adam(retrained_model.parameters(), lr=learning_rate)
        scaler = GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Training
            retrained_model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Retrain Epoch {epoch}/{num_epochs}')
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    outputs = retrained_model(images)
                    loss = self.criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            retrained_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                        outputs = retrained_model(images)
                        loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            print(f"   Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        computation_time = time.time() - start_time
        print(f"✅ Full Retrain complete in {computation_time:.2f}s")
        
        return retrained_model, computation_time


def create_forget_retain_loaders(dataset,
                                 train_indices: List[int],
                                 mislabeled_indices: List[int],
                                 batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for forget set (mislabeled) and retain set (clean)
    
    Args:
        dataset: Full dataset
        train_indices: All training indices
        mislabeled_indices: Indices of mislabeled samples
        batch_size: Batch size
    
    Returns:
        forget_loader, retain_loader
    """
    # Forget set: mislabeled samples
    forget_indices = [idx for idx in mislabeled_indices if idx in train_indices]
    
    # Retain set: clean samples (all train samples except mislabeled)
    retain_indices = [idx for idx in train_indices if idx not in mislabeled_indices]
    
    eval_transform = _build_eval_transform()
    forget_dataset = GalaxySubset(dataset, forget_indices, transform=eval_transform)
    retain_dataset = GalaxySubset(dataset, retain_indices, transform=eval_transform)
    
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    
    print(f"\n📊 Unlearning Data Split:")
    print(f"   Forget set (mislabeled): {len(forget_indices)} samples")
    print(f"   Retain set (clean): {len(retain_indices)} samples")
    
    return forget_loader, retain_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine unlearning from a saved checkpoint")
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet34')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--mislabel-ratio', type=float, default=0.10)
    parser.add_argument('--ga-epochs', type=int, default=3)
    parser.add_argument('--ga-lr', type=float, default=1e-4)
    parser.add_argument('--fisher-noise', type=float, default=0.01)
    parser.add_argument('--retrain-epochs', type=int, default=10)
    parser.add_argument('--retrain-lr', type=float, default=2e-4)
    parser.add_argument('--results-path', type=str, default='unlearning_results.csv')
    args = parser.parse_args()

    if args.device == 'cuda':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n📡 Running unlearning on: {device}")
    print("\n📥 Loading data...")
    train_loader, val_loader, test_loader, dataset, train_indices, class_weights = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=2
    )

    print(f"\n💥 Injecting mislabels at ratio: {args.mislabel_ratio:.2f}")
    mislabeled_indices, _ = inject_mislabels(dataset, mislabel_ratio=args.mislabel_ratio)
    forget_loader, retain_loader = create_forget_retain_loaders(
        dataset, train_indices, mislabeled_indices, batch_size=args.batch_size
    )

    if os.path.exists(args.checkpoint):
        print(f"\n📦 Loading checkpoint: {args.checkpoint}")
        model = load_model(args.checkpoint, device=device)
    else:
        print(f"\n⚠️ Checkpoint not found: {args.checkpoint}")
        print("   Falling back to a fresh model, which is useful only for smoke testing.")
        model = create_model(device=device, backbone=args.backbone, pretrained=True)

    unlearner = UnlearningMethods(model, device)
    results = []

    print("\n--- Method 1: Gradient Ascent ---")
    ga_model, ga_time = unlearner.gradient_ascent_unlearning(
        forget_loader,
        num_epochs=args.ga_epochs,
        learning_rate=args.ga_lr
    )
    results.append(comprehensive_evaluation(
        ga_model, test_loader, forget_loader, retain_loader, device, 'GradientAscent', ga_time
    ))
    torch.save(ga_model.state_dict(), 'ga_unlearned_model.pth')

    print("\n--- Method 2: Fisher Forgetting ---")
    ff_model, ff_time = unlearner.fisher_forgetting(
        retain_loader,
        noise_scale=args.fisher_noise
    )
    results.append(comprehensive_evaluation(
        ff_model, test_loader, forget_loader, retain_loader, device, 'FisherForgetting', ff_time
    ))
    torch.save(ff_model.state_dict(), 'ff_unlearned_model.pth')

    print("\n--- Method 3: Full Retrain ---")
    clean_train_indices = [idx for idx in train_indices if idx not in mislabeled_indices]
    clean_train_dataset = GalaxySubset(dataset, clean_train_indices, transform=_build_eval_transform())
    clean_train_loader = DataLoader(
        clean_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    retrain_model, retrain_time = unlearner.full_retrain(
        clean_train_loader,
        val_loader,
        num_epochs=args.retrain_epochs,
        learning_rate=args.retrain_lr
    )
    results.append(comprehensive_evaluation(
        retrain_model, test_loader, forget_loader, retain_loader, device, 'FullRetrain', retrain_time
    ))
    torch.save(retrain_model.state_dict(), 'retrained_model.pth')

    compare_all_methods(results, save_path=args.results_path)
    print(f"\n✅ Unlearning complete. Results saved to {args.results_path}")
