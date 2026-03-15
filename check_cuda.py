"""
Check CUDA version and provide installation instructions
"""

import subprocess
import sys

print("="*60)
print("CUDA VERSION CHECKER")
print("="*60)

print("\n1. Checking for NVIDIA GPU...")

try:
    # Run nvidia-smi command
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    
    if result.returncode == 0:
        output = result.stdout
        print("✅ NVIDIA GPU detected!\n")
        print(output)
        
        # Extract CUDA version
        for line in output.split('\n'):
            if 'CUDA Version' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                print(f"\n{'='*60}")
                print(f"Your CUDA Version: {cuda_version}")
                print(f"{'='*60}")
                
                # Recommend PyTorch version
                major_version = int(cuda_version.split('.')[0])
                
                print("\n2. RECOMMENDED PYTORCH INSTALLATION:\n")
                
                if major_version >= 12:
                    print("For CUDA 12.x, run these commands:")
                    print("-" * 60)
                    print("pip uninstall torch torchvision")
                    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                elif major_version == 11:
                    print("For CUDA 11.x, run these commands:")
                    print("-" * 60)
                    print("pip uninstall torch torchvision")
                    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                else:
                    print("For older CUDA, run:")
                    print("-" * 60)
                    print("pip uninstall torch torchvision")
                    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                
                print("-" * 60)
                break
    else:
        print("❌ nvidia-smi command failed")
        print("Your GPU drivers might not be installed correctly")
        
except FileNotFoundError:
    print("❌ nvidia-smi not found")
    print("\nThis means:")
    print("1. NVIDIA drivers are not installed, OR")
    print("2. nvidia-smi is not in your PATH")
    print("\nInstall NVIDIA drivers from:")
    print("https://www.nvidia.com/download/index.aspx")

print("\n" + "="*60)
print("3. AFTER INSTALLING PYTORCH WITH CUDA:")
print("="*60)
print("\nVerify GPU is working:")
print('python -c "import torch; print(torch.cuda.is_available())"')
print("\nThen run:")
print("python test_setup.py")
print("="*60)
