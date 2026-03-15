"""
Quick setup test script
Verifies all dependencies and hardware are working correctly
"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
    except ImportError:
        print("❌ TorchVision not found")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not found")
        return False
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not found")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Available")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Total VRAM: {total_mem:.2f} GB")
        
        if total_mem < 4.5:
            print(f"   ⚠️  Limited VRAM - Memory optimizations will be used")
        
        return True
    else:
        print("⚠️  CUDA not available - Will use CPU")
        print("   Training will be slower but still functional")
        return True


def test_modules():
    """Test project modules"""
    print("\nTesting project modules...")
    
    try:
        import data_loader
        print("✅ data_loader.py")
    except Exception as e:
        print(f"❌ data_loader.py: {e}")
        return False
    
    try:
        import model
        print("✅ model.py")
    except Exception as e:
        print(f"❌ model.py: {e}")
        return False
    
    try:
        import train
        print("✅ train.py")
    except Exception as e:
        print(f"❌ train.py: {e}")
        return False
    
    try:
        import unlearn
        print("✅ unlearn.py")
    except Exception as e:
        print(f"❌ unlearn.py: {e}")
        return False
    
    try:
        import evaluate
        print("✅ evaluate.py")
    except Exception as e:
        print(f"❌ evaluate.py: {e}")
        return False
    
    try:
        import visualize
        print("✅ visualize.py")
    except Exception as e:
        print(f"❌ visualize.py: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        import torch
        from model import create_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(device=device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (2, 3), f"Expected shape (2, 3), got {output.shape}"
        
        print("✅ Model creation and forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_data_loader():
    """Test data loader"""
    print("\nTesting data loader...")
    
    try:
        from data_loader import get_data_loaders
        
        train_loader, val_loader, test_loader, dataset, _, class_weights = get_data_loaders(
            batch_size=4
        )
        
        # Get a batch
        images, labels = next(iter(train_loader))
        
        assert images.shape[0] <= 4, "Batch size incorrect"
        assert images.shape[1:] == (3, 224, 224), "Image shape incorrect"
        
        print("✅ Data loader working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data loader failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("  SETUP TEST - Galaxy Morphology Unlearning")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Some dependencies are missing!")
        print("   Run: pip install -r requirements.txt")
    
    # Test CUDA
    if not test_cuda():
        all_passed = False
    
    # Test modules
    if not test_modules():
        all_passed = False
        print("\n❌ Some project modules have errors!")
    
    # Test model
    if not test_model_creation():
        all_passed = False
    
    # Test data loader
    if not test_data_loader():
        all_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou're ready to go! Run:")
        print("  python main.py          # Full pipeline")
        print("  streamlit run app.py    # Web interface")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        sys.exit(1)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
