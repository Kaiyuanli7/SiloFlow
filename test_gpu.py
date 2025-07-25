#!/usr/bin/env python3
"""Test GPU detection and usage chain functionality"""

import sys
import os
sys.path.append('service/scripts/testing')

# Import the SiloFlowTester class
from testingservice import SiloFlowTester

class MockVar:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value

def test_gpu_usage_chain():
    """Test the complete GPU detection and usage chain"""
    print("=== TESTING GPU USAGE CHAIN ===")
    
    # Create a minimal tester instance (without full initialization)
    tester = SiloFlowTester.__new__(SiloFlowTester)
    tester.batch_gpu_var = MockVar(True)  # Enable GPU
    
    print("\n1. GPU Hardware Detection:")
    gpu_info = tester._detect_gpus()
    print(f"   Has NVIDIA GPUs: {gpu_info['has_nvidia']}")
    print(f"   GPU Count: {gpu_info['nvidia_count']}")
    
    if gpu_info.get('error'):
        print(f"   Detection Error: {gpu_info['error']}")
    
    if gpu_info['nvidia_gpus']:
        print("\n   GPU Details:")
        for i, gpu in enumerate(gpu_info['nvidia_gpus']):
            print(f"     GPU {i}: {gpu['name']}")
            print(f"       Total Memory: {gpu['memory_total']:.1f} GB")
            print(f"       Free Memory: {gpu['memory_free']:.2f} GB")
            print(f"       Used Memory: {gpu['memory_used']:.2f} GB ({gpu['memory_percent']:.1f}%)")
    
    print("\n2. CUDA Availability:")
    cuda_info = tester._check_cuda_availability()
    print(f"   CUDA Available: {cuda_info['available']}")
    if cuda_info['available']:
        print(f"   CUDA Version: {cuda_info['version']}")
        print(f"   Device Count: {cuda_info['device_count']}")
        print(f"   Library Used: {cuda_info['library_used']}")
    else:
        if cuda_info.get('error'):
            print(f"   Error: {cuda_info['error']}")
    
    print("\n3. Final GPU Usage Decision:")
    use_gpu = tester.batch_gpu_var.get()
    final_gpu_usage = use_gpu and cuda_info['available']
    print(f"   GUI GPU Setting: {use_gpu}")
    print(f"   CUDA Available: {cuda_info['available']}")
    print(f"   Final GPU Usage: {final_gpu_usage}")
    
    if final_gpu_usage and gpu_info['has_nvidia']:
        primary_gpu = gpu_info['nvidia_gpus'][0] if gpu_info['nvidia_gpus'] else None
        if primary_gpu:
            print(f"   Selected GPU: {primary_gpu['name']} (GPU 0)")
            print(f"   Available VRAM: {primary_gpu['memory_free']:.2f} GB")
    
    print("\n4. Training Pipeline GPU Usage:")
    if final_gpu_usage:
        print("   ✅ The training pipeline WILL use GPU acceleration")
        print("   📋 Flow: testingservice.py -> streaming_processor.py -> multi_lgbm.py")
        print("   🔧 LightGBM will be configured with device='gpu'")
        print("   🎯 GPU parameters will be applied to LGBMRegressor")
    else:
        print("   ❌ The training pipeline will use CPU only")
        if use_gpu and not cuda_available['available']:
            print("   💡 Reason: CUDA not available despite GPU being present")
        elif not use_gpu:
            print("   💡 Reason: GPU disabled in GUI settings")
    
    print("\n=== GPU USAGE CHAIN TEST COMPLETE ===")
    
    return final_gpu_usage

if __name__ == "__main__":
    gpu_will_be_used = test_gpu_usage_chain()
    print(f"\n🏁 FINAL RESULT: GPU will be used for training: {gpu_will_be_used}")
