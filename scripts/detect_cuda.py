#!/usr/bin/env python3
"""
Simple CUDA detection script for SiloFlow
Safely detects CUDA version without installing GPU packages
"""

import subprocess
import sys
import os
from pathlib import Path

def check_nvidia_smi():
    """Check if nvidia-smi is available and get GPU info"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return None

def check_cuda_toolkit():
    """Check for CUDA toolkit installation"""
    possible_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA", 
        r"C:\cuda",
        r"C:\CUDA",
    ]
    
    found_versions = []
    for base_path in possible_paths:
        if os.path.exists(base_path):
            try:
                for item in os.listdir(base_path):
                    if item.startswith('v') and os.path.isdir(os.path.join(base_path, item)):
                        found_versions.append(item)
            except (PermissionError, OSError):
                continue
    
    return found_versions

def check_environment_variables():
    """Check CUDA-related environment variables"""
    cuda_vars = {}
    for var in ['CUDA_PATH', 'CUDA_HOME', 'CUDNN_PATH']:
        value = os.environ.get(var)
        if value:
            cuda_vars[var] = value
    return cuda_vars

def recommend_installation():
    """Provide installation recommendations"""
    print("\n" + "="*60)
    print("CUDA INSTALLATION RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. If you DON'T have an NVIDIA GPU:")
    print("   → Install: pip install -r requirements.txt")
    print("   → This gives you CPU acceleration with Polars")
    
    print("\n2. If you HAVE an NVIDIA GPU but no CUDA:")
    print("   → Download CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive")
    print("   → Then install: pip install cudf-cu11 cupy-cuda11x")
    
    print("\n3. If you want to try GPU packages without knowing CUDA version:")
    print("   → Try CUDA 11 first (most compatible):")
    print("     pip install cudf-cu11>=23.10 cupy-cuda11x>=12.0")
    print("   → If that fails, try CUDA 12:")
    print("     pip install cudf-cu12>=23.10 cupy-cuda12x>=12.0")
    
    print("\n4. Safe approach (recommended):")
    print("   → First install: pip install -r requirements.txt")
    print("   → Test the system works with CPU acceleration")
    print("   → Then add GPU packages incrementally")

def main():
    print("SiloFlow CUDA Detection Tool")
    print("="*40)
    
    # Check nvidia-smi
    print("\n1. Checking for NVIDIA GPU drivers...")
    nvidia_info = check_nvidia_smi()
    if nvidia_info:
        print("✅ NVIDIA GPU detected!")
        print("GPU Information:")
        # Extract relevant lines
        lines = nvidia_info.split('\n')
        for line in lines:
            if 'CUDA Version:' in line:
                print(f"   {line.strip()}")
            elif '|' in line and ('GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line or 'Quadro' in line):
                print(f"   GPU: {line.split('|')[1].strip()}")
    else:
        print("❌ No NVIDIA GPU drivers detected")
    
    # Check CUDA toolkit
    print("\n2. Checking for CUDA toolkit...")
    cuda_versions = check_cuda_toolkit()
    if cuda_versions:
        print(f"✅ Found CUDA versions: {', '.join(cuda_versions)}")
    else:
        print("❌ No CUDA toolkit found in standard locations")
    
    # Check environment variables
    print("\n3. Checking environment variables...")
    cuda_env = check_environment_variables()
    if cuda_env:
        print("✅ Found CUDA environment variables:")
        for var, value in cuda_env.items():
            print(f"   {var}: {value}")
    else:
        print("❌ No CUDA environment variables found")
    
    # Provide recommendations
    recommend_installation()
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Install basic requirements: pip install -r requirements.txt")
    print("2. Test SiloFlow with CPU acceleration")
    print("3. Add GPU packages if needed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
