#!/usr/bin/env python3
"""
Quick CUDA/PyTorch compatibility test for Google Colab.
Run this BEFORE running heavy training cells to verify GPU setup.

Usage in Colab:
    !python test_cuda.py
"""

import subprocess
import sys

def run_cmd(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return None

def main():
    print("="*60)
    print("CUDA/PyTorch Quick Diagnostic")
    print("="*60)
    
    # Check GPU
    print("\n🖥️  GPU Info:")
    gpu_info = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null")
    if gpu_info:
        print(f"  {gpu_info}")
    else:
        print("  ❌ No GPU found or nvidia-smi failed")
        print("  → Go to Runtime > Change runtime type > Select GPU")
        return 1
    
    # Check CUDA version from system
    print("\n🔧 System CUDA:")
    cuda_version = run_cmd("nvcc --version 2>/dev/null | grep release | awk '{print $5}' | cut -d',' -f1")
    if cuda_version:
        print(f"  CUDA (nvcc): {cuda_version}")
    else:
        print("  CUDA (nvcc): not found")
    
    # Check PyTorch
    print("\n📦 PyTorch Installation:")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version (PyTorch): {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            
            # Test CUDA operations
            print("\n🧪 Testing CUDA operations...")
            try:
                x = torch.rand(100, 100).cuda()
                y = torch.rand(100, 100).cuda()
                z = x @ y
                print(f"  ✓ CUDA matrix multiplication works! Shape: {z.shape}")
                
                # Test set_device (the failing operation)
                print("  Testing torch.cuda.set_device(0)...")
                torch.cuda.set_device(0)
                print("  ✓ set_device works!")
                
                print("\n" + "="*60)
                print("✅ GPU is ready for YOLOX training!")
                print("="*60)
                return 0
                
            except Exception as e:
                print(f"  ❌ CUDA operation failed: {e}")
                print("\n🔧 FIX: Restart runtime (Ctrl+M .) and run this again")
                return 1
        else:
            print("\n❌ CUDA not available in PyTorch")
            print("   → Go to Runtime > Change runtime type > Select GPU")
            print("   → Then Runtime > Restart runtime")
            return 1
            
    except ImportError:
        print("  ❌ PyTorch not installed")
        return 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
