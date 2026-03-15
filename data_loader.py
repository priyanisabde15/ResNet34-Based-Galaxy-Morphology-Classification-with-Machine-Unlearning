"""
Galaxy Zoo 2 Data Loader with Intentional Mislabeling
Handles official Galaxy Zoo 2 dataset from Kaggle with proper preprocessing
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import random
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class GalaxyDataset(Dataset):
    """
    Custom dataset for Galaxy Zoo 2 images with 3 morphology classes
    
    Dataset Structure (Kaggle Galaxy Zoo 2):
    - images_training_rev1/ or images_test_rev1/ - Galaxy images
    - training_solutions_rev1.csv - Labels with probabilities
    
    Classes based on GZ2 decision tree:
    - Class 0 (Smooth): Elliptical/smooth galaxies
    - Class 1 (Featured/Disk): Spiral/disk galaxies with features
    - Class 2 (Artifact): Star/artifact (edge-on disk or artifact)
    """
    
    def __init__(self, data_dir: str, csv_file: str = None, transform=None, use_simulation: bool = False):
        """
        Args:
            data_dir: Path to dataset directory containing images
            csv_file: Path to CSV file with labels (for training set)
            transform: Image transformations
            use_simulation: If True, generate synthetic galaxy-like images (disabled for production)
        """
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.use_simulation = False  # force real dataset path
        
        # Class mapping: 0=Smooth, 1=Featured/Disk, 2=Artifact
        self.classes = ['Smooth', 'Featured', 'Artifact']
        self.num_classes = len(self.classes)
        
        if use_simulation:
            raise ValueError("Synthetic simulation mode is disabled. Provide real Galaxy Zoo data in data_dir.")

        # Load real Galaxy Zoo 2 data
        self.samples = self._load_galaxy_zoo_2()
        print(f"✅ Loaded {len(self.samples)} real Galaxy Zoo 2 images")
    
    def _load_galaxy_zoo_2(self) -> List[Tuple]:
        """
        Load official Galaxy Zoo 2 dataset from Kaggle
        
        Expected structure:
        data_dir/
        ├── images_training_rev1/ (or images_test_rev1/)
        │   ├── 100008.jpg
        │   ├── 100023.jpg
        │   └── ...
        └── training_solutions_rev1.csv (optional)
            Columns: GalaxyID, Class1.1, Class1.2, Class1.3, ...
        """
        samples = []
        
        # Look for image directories
        possible_image_dirs = [
            os.path.join(self.data_dir, 'images_training_rev1'),
            os.path.join(self.data_dir, 'images_test_rev1'),
            os.path.join(self.data_dir, 'images'),
            self.data_dir
        ]
        
        image_dir = None
        for dir_path in possible_image_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # Check if it contains image files
                files = os.listdir(dir_path)
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    image_dir = dir_path
                    break
        
        if image_dir is None:
            raise FileNotFoundError(f"No image directory found in {self.data_dir}")
        
        print(f"📂 Found images in: {image_dir}")
        
        # Load labels from CSV if available
        labels_dict = {}
        if self.csv_file and os.path.exists(self.csv_file):
            print(f"📊 Loading labels from: {self.csv_file}")
            df = pd.read_csv(self.csv_file)
            df['GalaxyID'] = df['GalaxyID'].astype(str)
            
            # Galaxy Zoo 2 uses probability-based labels
            # We'll use the decision tree to classify:
            # - Class1.1 (smooth) > 0.469 → Smooth (0)
            # - Class1.2 (features/disk) > 0.430 → Featured (1)
            # - Class1.3 (star/artifact) > 0.5 → Artifact (2)
            
            for _, row in df.iterrows():
                galaxy_id = row['GalaxyID']
                
                # Get probabilities for main morphology
                smooth_prob = row.get('Class1.1', 0)
                featured_prob = row.get('Class1.2', 0)
                artifact_prob = row.get('Class1.3', 0)
                
                # Assign class based on highest probability
                probs = [smooth_prob, featured_prob, artifact_prob]
                class_idx = np.argmax(probs)
                
                labels_dict[galaxy_id] = class_idx
            
            print(f"✅ Loaded {len(labels_dict)} labels from CSV")
        
        if not labels_dict:
            raise FileNotFoundError(
                "No valid label CSV found. training_solutions_rev1.csv is required for supervised training."
            )

        # Load all images
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"No image files found in {image_dir}")
        
        print(f"📸 Found {len(image_files)} images")
        
        # Create samples
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            
            # Extract galaxy ID from filename (e.g., "100008.jpg" -> "100008")
            galaxy_id = os.path.splitext(img_file)[0]
            
            # Only keep images with real labels from the CSV.
            if galaxy_id in labels_dict:
                label = labels_dict[galaxy_id]
                samples.append((img_path, label))

        if len(samples) == 0:
            raise ValueError("No training images matched labels in the CSV file.")

        matched_ratio = 100.0 * len(samples) / len(image_files)
        print(f"✅ Matched labels for {len(samples)} / {len(image_files)} images ({matched_ratio:.2f}%)")
        
        return samples
    
    def _generate_simulated_data(self, num_samples: int = 300) -> List[Tuple]:
        """Generate galaxy image tensors fast using numpy vectorized ops"""
        print(f"   Generating {num_samples} simulated galaxy images...")
        samples = []
        samples_per_class = num_samples // self.num_classes

        for class_idx in range(self.num_classes):
            for i in range(samples_per_class):
                img = self._create_galaxy_image(class_idx)
                samples.append((img, class_idx))
                if (i + 1) % 50 == 0:
                    print(f"   Class {self.classes[class_idx]}: {i+1}/{samples_per_class}")

        print(f"   ✅ Generated {len(samples)} simulated images")
        return samples
    
    def _create_galaxy_image(self, class_idx: int) -> np.ndarray:
        """Create synthetic galaxy-like image — fully vectorized, no loops"""
        # Build coordinate grids once
        y, x = np.mgrid[0:224, 0:224].astype(np.float32)
        cy, cx = 112.0, 112.0
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        if class_idx == 0:  # Smooth elliptical
            intensity = np.clip(255 - dist * 3, 0, 255)
            noise = np.random.randint(-20, 20, (224, 224))
            channel = np.clip(intensity + noise, 0, 255).astype(np.uint8)
            img = np.stack([channel, channel, channel], axis=2)

        elif class_idx == 1:  # Featured/Disk spiral
            angle = np.arctan2(y - cy, x - cx)
            spiral = np.sin(angle * 3 + dist * 0.1) * 50
            intensity = np.clip(200 - dist * 2 + spiral, 0, 255)
            noise = np.random.randint(-15, 15, (224, 224))
            channel = np.clip(intensity + noise, 0, 255).astype(np.uint8)
            img = np.stack([channel, channel, channel], axis=2)

        else:  # Artifact
            img = np.random.randint(0, 150, (224, 224, 3), dtype=np.uint8)
            img[100:124, :] = np.random.randint(200, 255)

        return img
    
    def _load_real_data(self) -> List[Tuple]:
        """
        DEPRECATED: Use _load_galaxy_zoo_2() instead
        Legacy method for simple folder structure
        """
        samples = []
        
        if not os.path.exists(self.data_dir):
            return self._generate_simulated_data()
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))
        
        if len(samples) == 0:
            return self._generate_simulated_data()
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.use_simulation:
            img_array, label = self.samples[idx]
            img = Image.fromarray(img_array)
        else:
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class GalaxySubset(Dataset):
    """Split-specific dataset wrapper so each split can have its own transform."""

    def __init__(self, base_dataset: GalaxyDataset, indices: List[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx = self.indices[idx]
        img_path, label = self.base_dataset.samples[sample_idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def _compute_effective_class_weights(class_counts: List[int], beta: float = 0.9999, max_weight: float = 25.0) -> torch.Tensor:
    """Compute smoothed class weights for extreme imbalance without exploding the loss."""
    counts = np.array(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    raw_weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    raw_weights = raw_weights / raw_weights.mean()
    raw_weights = np.clip(raw_weights, 0.1, max_weight)
    return torch.tensor(raw_weights, dtype=torch.float32)


def inject_mislabels(dataset: Dataset, mislabel_ratio: float = 0.12) -> Tuple[List[int], List[int]]:
    """
    Intentionally mislabel a portion of the dataset to simulate noisy data
    
    Args:
        dataset: Original dataset
        mislabel_ratio: Fraction of samples to mislabel (default 12%)
    
    Returns:
        mislabeled_indices: List of indices that were mislabeled
        original_labels: Original correct labels for mislabeled samples
    """
    num_samples = len(dataset)
    num_mislabel = int(num_samples * mislabel_ratio)
    
    # Randomly select indices to mislabel
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)
    mislabeled_indices = all_indices[:num_mislabel]
    
    original_labels = []
    
    print(f"💥 Injecting mislabels into {num_mislabel} samples ({mislabel_ratio*100:.1f}%)...")
    
    for idx in mislabeled_indices:
        if dataset.use_simulation:
            _, original_label = dataset.samples[idx]
            original_labels.append(original_label)
            
            # Change to a different random class
            new_label = original_label
            while new_label == original_label:
                new_label = random.randint(0, dataset.num_classes - 1)
            
            img_data = dataset.samples[idx][0]
            dataset.samples[idx] = (img_data, new_label)
        else:
            img_path, original_label = dataset.samples[idx]
            original_labels.append(original_label)
            
            new_label = original_label
            while new_label == original_label:
                new_label = random.randint(0, dataset.num_classes - 1)
            
            dataset.samples[idx] = (img_path, new_label)
    
    print(f"✅ Mislabeling complete. {len(mislabeled_indices)} samples corrupted.")
    return mislabeled_indices, original_labels


def get_data_loaders(data_dir: str = './data', 
                     csv_file: str = None,
                     batch_size: int = 16,
                     num_workers: int = 2) -> Tuple:
    """
    Create train, validation, and test data loaders for Galaxy Zoo 2
    
    Args:
        data_dir: Path to dataset (should contain images_training_rev1/ or images/)
        csv_file: Path to training_solutions_rev1.csv (optional)
        batch_size: Batch size (max 16 for 4GB VRAM)
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader, full_dataset, train_indices
    """
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for val/test
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Auto-detect CSV file if not provided
    if csv_file is None:
        possible_csv = [
            os.path.join(data_dir, 'training_solutions_rev1.csv'),
            os.path.join(data_dir, 'labels.csv'),
            os.path.join(data_dir, 'training_labels.csv')
        ]
        for csv_path in possible_csv:
            if os.path.exists(csv_path):
                csv_file = csv_path
                break
    
    # Ensure real Galaxy Zoo 2 data exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Please download Galaxy Zoo data and point data_dir correctly.")

    dataset = GalaxyDataset(data_dir, csv_file=csv_file, transform=None, use_simulation=False)

    # Print true class distribution before splitting
    full_labels = [label for _, label in dataset.samples]
    print(f"\n📊 Class Distribution:")
    for i, class_name in enumerate(dataset.classes):
        print(f"   {class_name}: {sum(label == i for label in full_labels)} samples")

    indices = np.arange(len(dataset))
    labels = np.array(full_labels)

    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=42
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=temp_labels,
        random_state=42
    )

    train_dataset = GalaxySubset(dataset, train_indices.tolist(), transform=train_transform)
    val_dataset = GalaxySubset(dataset, val_indices.tolist(), transform=eval_transform)
    test_dataset = GalaxySubset(dataset, test_indices.tolist(), transform=eval_transform)

    train_class_counts = np.bincount(train_labels, minlength=dataset.num_classes)
    smoothed_sampling_weights = np.power(np.maximum(train_class_counts, 1), -0.5)
    sample_weights = [float(smoothed_sampling_weights[labels[idx]]) for idx in train_indices]
    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_indices),
        replacement=True
    )
    
    # Create data loaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Compute class weights from training subset to handle any imbalance
    class_counts = train_class_counts.tolist()
    class_weights_tensor = _compute_effective_class_weights(class_counts)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")
    print(f"   Classes: {dataset.classes}")
    print(f"   Class counts: {class_counts}")
    print(f"   Batch size: {batch_size} (RTX 3050 optimized)")
    print(f"   Class weights: {[float(w) for w in class_weights_tensor]}")
    if class_counts[-1] < 100:
        print("   ⚠️ Artifact class is extremely small; training will favor macro metrics and minority-aware loss.")

    return train_loader, val_loader, test_loader, dataset, train_indices, class_weights_tensor


if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")
    train_loader, val_loader, test_loader, dataset, _, class_weights = get_data_loaders()
    print(f"   Class weights loaded: {class_weights}")
    
    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Batch shape: {images.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print(f"✅ Data loader working correctly!")
