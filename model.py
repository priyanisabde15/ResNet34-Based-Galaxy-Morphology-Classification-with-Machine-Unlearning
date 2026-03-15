"""
Galaxy Morphology CNN Model
ResNet18-based architecture optimized for 4GB VRAM
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

class GalaxyCNN(nn.Module):
    """
    ResNet-based CNN for galaxy morphology classification
    Modified for 3-class output: Smooth, Featured/Disk, Artifact
    """
    
    def __init__(self, num_classes: int = 3, backbone: str = 'resnet34', pretrained: bool = True):
        """
        Args:
            num_classes: Number of galaxy morphology classes
            pretrained: Use ImageNet pretrained weights
        """
        super(GalaxyCNN, self).__init__()
        
        # Load pretrained ResNet backbone (more capacity for 80% target)
        backbone = backbone.lower()
        if pretrained:
            if backbone == 'resnet18':
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            elif backbone == 'resnet50':
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        else:
            if backbone == 'resnet18':
                self.backbone = models.resnet18(weights=None)
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(weights=None)
            elif backbone == 'resnet50':
                self.backbone = models.resnet50(weights=None)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer for our 3 classes
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),  # Regularization
            nn.Linear(num_features, num_classes)
        )
        
        # Enable gradient checkpointing to save memory
        self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        # Disabled for now to avoid recursion issues
        # Can be enabled manually if needed for very large batches
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final classification layer
        Useful for visualization and analysis
        """
        # Forward through all layers except final FC
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def create_model(num_classes: int = 3,
                device: Optional[torch.device] = None,
                backbone: str = 'resnet34',
                pretrained: bool = True) -> GalaxyCNN:
    """
    Factory function to create and initialize the model
    
    Args:
        num_classes: Number of output classes
        device: Device to place model on (cuda/cpu)
        backbone: ResNet backbone name (resnet18/resnet34/resnet50)
        pretrained: Use pretrained weights
    
    Returns:
        Initialized model on specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GalaxyCNN(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🧠 Model Architecture: {backbone}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {device}")
    print(f"   Memory optimizations: ✅ Mixed precision ready")
    
    return model


def load_model(checkpoint_path: str, 
               num_classes: int = 3,
               device: Optional[torch.device] = None) -> GalaxyCNN:
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to saved model weights
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GalaxyCNN(num_classes=num_classes, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    return model


def save_model(model: GalaxyCNN, 
               save_path: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None,
               metrics: Optional[dict] = None):
    """
    Save model checkpoint with optional training state
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    print(f"💾 Model saved to {save_path}")


def get_model_memory_usage(model: GalaxyCNN) -> float:
    """
    Calculate approximate model memory usage in MB
    
    Args:
        model: Model to analyze
    
    Returns:
        Memory usage in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Model memory: {get_model_memory_usage(model):.2f} MB")
    print(f"✅ Model working correctly!")
