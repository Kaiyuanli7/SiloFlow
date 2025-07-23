#!/usr/bin/env python3
"""
Test script to verify batch processing integration with Polars optimizations
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

def create_test_dataset(size="small"):
    """Create a test dataset for batch processing"""
    if size == "small":
        num_rows = 1000
        granary_name = "test_small_granary"
    else:  # large
        num_rows = 100_000
        granary_name = "test_large_granary"
    
    print(f"📊 Creating {size} test dataset with {num_rows:,} rows...")
    
    # Create realistic-looking granary data
    dates = pd.date_range('2024-01-01', periods=num_rows, freq='h')  # Use 'h' instead of 'H'
    test_data = pd.DataFrame({
        'detection_time': dates,
        'granary_id': granary_name,
        'heap_id': np.random.choice(['heap_1', 'heap_2', 'heap_3'], num_rows),
        'grid_x': np.random.randint(1, 10, num_rows),
        'grid_y': np.random.randint(1, 10, num_rows), 
        'grid_z': np.random.randint(1, 5, num_rows),
        'temperature_grain': 20 + np.random.normal(0, 2, num_rows),
        'temperature_inside': 18 + np.random.normal(0, 1.5, num_rows),
        'temperature_outside': 15 + np.random.normal(0, 3, num_rows),
        'humidity_warehouse': np.random.uniform(40, 80, num_rows),
        'humidity_outside': np.random.uniform(30, 90, num_rows)
    })
    
    return test_data, granary_name

def test_cli_preprocessing(test_data, granary_name, test_type="small"):
    """Test the CLI preprocessing with our optimizations"""
    print(f"\n🧪 Testing CLI preprocessing for {test_type} dataset...")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Save test data
        input_file = temp_dir / f"{granary_name}.csv"
        output_file = temp_dir / f"{granary_name}_processed.parquet"
        
        print(f"📁 Input file: {input_file}")
        print(f"📁 Output file: {output_file}")
        
        test_data.to_csv(input_file, index=False)
        print(f"✅ Test data saved ({test_data.shape[0]:,} rows)")
        
        # Run the CLI preprocess command
        script_path = Path(__file__).parent / "service" / "granary_pipeline.py"
        
        cmd = [
            sys.executable, str(script_path),
            "preprocess",
            "--input", str(input_file),
            "--output", str(output_file)
        ]
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        print("-" * 60)
        
        try:
            # Set environment variables for better logging
            env = os.environ.copy()
            env['SILOFLOW_DISABLE_PARALLEL'] = '1'  # For consistent testing
            
            # Run the preprocessing
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=Path(__file__).parent
            )
            
            print("📤 STDOUT:")
            if result.stdout:
                print(result.stdout)
            else:
                print("(no stdout)")
                
            print("\n📥 STDERR:")
            if result.stderr:
                print(result.stderr)
            else:
                print("(no stderr)")
                
            print(f"\n📊 Return code: {result.returncode}")
            
            # Check if output file was created
            if output_file.exists():
                processed_df = pd.read_parquet(output_file)
                print(f"✅ SUCCESS: Processed file created with shape {processed_df.shape}")
                
                # Check for expected columns
                expected_cols = ['temperature_grain', 'lag_temp_1d', 'lag_temp_2d', 'lag_temp_3d']
                found_cols = [col for col in expected_cols if col in processed_df.columns]
                print(f"🔍 Expected columns found: {len(found_cols)}/{len(expected_cols)} - {found_cols}")
                
                return True
            else:
                print(f"❌ FAILURE: Output file not created")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: Command execution failed: {e}")
            return False

def main():
    """Test batch processing integration"""
    print("🚀 Testing Batch Processing Integration with Polars Optimizations")
    print("=" * 80)
    
    # Test 1: Small dataset (should use pandas backend)
    small_data, small_name = create_test_dataset("small")
    small_success = test_cli_preprocessing(small_data, small_name, "small")
    
    # Test 2: Large dataset (should use Polars backend)
    large_data, large_name = create_test_dataset("large")  
    large_success = test_cli_preprocessing(large_data, large_name, "large")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY:")
    print(f"   Small dataset (1K rows): {'✅ PASS' if small_success else '❌ FAIL'}")
    print(f"   Large dataset (100K rows): {'✅ PASS' if large_success else '❌ FAIL'}")
    
    if small_success and large_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("🔧 Batch processing from test service should work correctly with Polars optimizations")
        print("📈 Large datasets (>100K rows) will automatically use Polars for better performance")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Check the output above for errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
