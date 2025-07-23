#!/usr/bin/env python3
"""
GPU Detection Utility for SiloFlow

This utility helps diagnose GPU availability and configuration for SiloFlow training.
Run this script to check your GPU setup before training models.
"""

import sys
from pathlib import Path

# Add service directory to path
sys.path.insert(0, str(Path(__file__).parent / "service"))

# Import GPU detection functions globally
try:
    from granary_pipeline import detect_available_gpus, detect_gpu_availability, select_best_gpu
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

def test_gpu_detection():
    """Test GPU detection capabilities."""
    print("🔍 SiloFlow GPU Detection Utility")
    print("=" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot import GPU detection functions")
        print("   Make sure you're running this from the SiloFlow root directory")
        print("   and that granary_pipeline.py exists in the service/ folder")
        return False
    
    try:
        # Test basic GPU detection
        print("1. Testing GPU Detection...")
        gpu_info = detect_available_gpus()
        
        if not gpu_info['gpus']:
            print("❌ No GPUs detected on this system")
            print("   - Check GPU drivers are installed")
            print("   - Install pynvml: pip install pynvml")
            print("   - For AMD/Intel GPUs, install pyopencl: pip install pyopencl")
            return False
        
        print(f"✅ Found {gpu_info['total_gpus']} GPU(s) via {gpu_info['detection_method']}")
        print()
        
        # Display detailed GPU information
        print("2. GPU Details:")
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"   GPU {gpu['id']} ({gpu['vendor']}):")
            print(f"      Name: {gpu['name']}")
            print(f"      Memory: {gpu['memory_total_mb']:.0f}MB total, {gpu['memory_free_mb']:.0f}MB free")
            print(f"      Utilization: {gpu['gpu_utilization']}% GPU, {gpu['memory_utilization']}% memory")
            print()
        
        # Test GPU selection
        print("3. Testing GPU Selection...")
        best_gpu = select_best_gpu(gpu_info['gpus'])
        best_gpu_info = gpu_info['gpus'][best_gpu]
        print(f"✅ Auto-selected GPU: {best_gpu} ({best_gpu_info['name']})")
        print(f"   Selection reason: {best_gpu_info['memory_free_mb']:.0f}MB free, {best_gpu_info['gpu_utilization']}% utilized")
        print()
        
        # Test LightGBM GPU compatibility
        print("4. Testing LightGBM GPU Compatibility...")
        gpu_available, gpu_info_msg, gpu_config = detect_gpu_availability(force_enable=True)
        
        if gpu_available:
            print(f"✅ LightGBM GPU test successful!")
            print(f"   {gpu_info_msg}")
            print(f"   GPU Config: {gpu_config}")
        else:
            print(f"❌ LightGBM GPU test failed:")
            print(f"   {gpu_info_msg}")
            return False
        
        # Test GPU data processing (cuDF)
        print()
        print("5. Testing GPU Data Processing (cuDF)...")
        try:
            from granarypredict.gpu_data_utils import get_gpu_data_backend_info, HAS_CUDF
            
            backend_info = get_gpu_data_backend_info()
            if backend_info['cudf_available']:
                print(f"✅ RAPIDS cuDF available for GPU data processing!")
                print(f"   Expected speedup: {backend_info['performance_estimates']['cudf_speedup']}")
                if 'gpu_memory_mb' in backend_info:
                    print(f"   GPU Memory: {backend_info['gpu_memory_mb']}MB")
                
                # Test basic cuDF operation
                import cudf
                import pandas as pd
                test_data = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
                    'temperature': range(1000),
                    'silo_id': [f"silo_{i%10}" for i in range(1000)]
                })
                
                gpu_df = cudf.from_pandas(test_data)
                result = gpu_df.groupby('silo_id').agg({'temperature': 'mean'})
                print(f"   ✅ cuDF basic test successful: {len(result)} groups processed")
                
            else:
                print(f"❌ RAPIDS cuDF not available")
                print(f"   Install with: conda install -c rapidsai -c conda-forge cudf")
                print(f"   Falling back to: {backend_info['performance_estimates']['polars_speedup'] if backend_info['polars_available'] else 'pandas'}")
        
        except ImportError:
            print(f"❌ GPU data processing utilities not available")
            print(f"   Make sure gpu_data_utils.py is in granarypredict/")
        except Exception as e:
            print(f"⚠️  cuDF test failed: {e}")
            print(f"   GPU data processing will fall back to Polars/pandas")
        
        print()
        print("5. Recommendations:")
        
        # Memory recommendations
        best_gpu_memory = best_gpu_info['memory_free_mb']
        if best_gpu_memory > 16000:
            print("✅ Excellent GPU memory (>16GB) - Can handle very large datasets")
            print("   Recommended: Use max_bin=511 for optimal performance")
            print("   Data Processing: cuDF will handle 1M+ rows efficiently")
        elif best_gpu_memory > 8000:
            print("✅ Good GPU memory (8-16GB) - Can handle large datasets")
            print("   Recommended: Use max_bin=255-511 depending on dataset size")
            print("   Data Processing: cuDF optimal for 100K-1M rows")
        elif best_gpu_memory > 4000:
            print("⚠️  Moderate GPU memory (4-8GB) - Suitable for medium datasets")
            print("   Recommended: Use max_bin=127-255, monitor memory usage")
            print("   Data Processing: cuDF good for 50K-500K rows")
        else:
            print("⚠️  Limited GPU memory (<4GB) - May need CPU for large datasets")
            print("   Recommended: Use max_bin=127, consider CPU for >100K rows")
            print("   Data Processing: Use Polars/pandas for large datasets")
        
        # Utilization recommendations
        if best_gpu_info['gpu_utilization'] < 30:
            print("✅ GPU is lightly utilized - Good for training and data processing")
        elif best_gpu_info['gpu_utilization'] < 70:
            print("⚠️  GPU is moderately utilized - Training possible but may be slower")
        else:
            print("🚨 GPU is heavily utilized - Consider using a different GPU or waiting")
        
        print()
        print("6. Usage Examples:")
        print("   # Auto-select best GPU:")
        print("   python granary_pipeline.py train --granary MyGranary --gpu")
        print()
        print(f"   # Use specific GPU {best_gpu}:")
        print(f"   python granary_pipeline.py train --granary MyGranary --gpu --gpu-id {best_gpu}")
        print()
        print("   # GPU data processing is automatic for large datasets (50K+ rows)")
        print("   # Install cuDF for maximum performance: conda install -c rapidsai cudf")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this from the SiloFlow root directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_multiple_gpu_scenarios():
    """Test scenarios with multiple GPUs."""
    print("\n" + "=" * 50)
    print("🎯 Multi-GPU Scenario Testing")
    print("=" * 50)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot test multi-GPU scenarios - imports not available")
        return
    
    try:
        # Test auto-selection
        print("Testing auto-selection...")
        available, info, config = detect_gpu_availability(force_enable=True)
        if available:
            print(f"✅ Auto-selection: {info}")
            print(f"   Config: GPU {config.get('gpu_device_id', 'unknown')}")
        
        # Test specific GPU selection
        gpu_info = detect_available_gpus()
        if len(gpu_info['gpus']) > 1:
            print(f"\nTesting specific GPU selection (GPU 1)...")
            available, info, config = detect_gpu_availability(force_enable=True, preferred_gpu_id=1)
            if available:
                print(f"✅ GPU 1 selection: {info}")
                print(f"   Config: GPU {config.get('gpu_device_id', 'unknown')}")
            
            print(f"\nTesting invalid GPU selection (GPU 99)...")
            available, info, config = detect_gpu_availability(force_enable=True, preferred_gpu_id=99)
            print(f"Result: {info}")
        
    except Exception as e:
        print(f"❌ Multi-GPU testing error: {e}")

if __name__ == "__main__":
    success = test_gpu_detection()
    
    if success:
        # Only test multi-GPU scenarios if basic detection works
        test_multiple_gpu_scenarios()
        
        print("\n" + "=" * 50)
        print("🎉 GPU Detection Test Complete!")
        print("=" * 50)
        print("Your system is ready for GPU-accelerated SiloFlow training.")
    else:
        print("\n" + "=" * 50)
        print("🚨 GPU Detection Test Failed")
        print("=" * 50)
        print("GPU acceleration is not available. SiloFlow will use CPU training.")
        print("See the error messages above for troubleshooting steps.")
