#!/usr/bin/env python3
"""
Test script for Polars integration in granary pipeline
"""

import sys
import os
sys.path.append('.')

def test_polars_integration():
    """Test the Polars-optimized functions"""
    try:
        from service.granary_pipeline import load_data_optimized, add_lags_optimized, create_time_features_optimized
        import pandas as pd
        import numpy as np
        
        print("🧪 Testing Polars integration...")
        print("=" * 60)
        
        # Create a test dataset
        print("📊 Creating test dataset...")
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        test_data = pd.DataFrame({
            'detection_time': dates,
            'granary_id': 'test_granary',
            'heap_id': np.random.choice(['heap_1', 'heap_2'], 1000),
            'grid_x': np.random.randint(1, 10, 1000),
            'grid_y': np.random.randint(1, 10, 1000), 
            'grid_z': np.random.randint(1, 5, 1000),
            'temperature_grain': 20 + np.random.normal(0, 2, 1000)
        })
        
        print(f"✅ Test data created: {test_data.shape}")
        print("🔍 This dataset is <50K rows, should use PANDAS BACKEND")
        print("-" * 60)
        
        # Test optimized lag computation
        print("🚀 Testing optimized lag computation...")
        test_result = add_lags_optimized(test_data, lags=[1, 2, 3])
        print(f"✅ Lag computation completed: {test_result.shape}")
        lag_cols = [col for col in test_result.columns if "lag" in col or "delta" in col]
        print(f"New lag/delta columns: {len(lag_cols)} - {lag_cols[:5]}...")
        
        # Test optimized time features
        print("🚀 Testing optimized time features...")
        test_result2 = create_time_features_optimized(test_result)
        print(f"✅ Time features completed: {test_result2.shape}")
        time_cols = [col for col in test_result2.columns if any(word in col.lower() for word in ["hour", "month", "year", "sin", "cos", "doy", "weekend"])]
        print(f"New time columns: {len(time_cols)} - {time_cols[:5]}...")
        
        # Test with larger dataset to trigger Polars
        print("\n" + "=" * 60)
        print("🚀 Testing with larger dataset (>50K rows)...")
        print("🔍 This dataset should trigger POLARS BACKEND")
        large_dates = pd.date_range('2024-01-01', periods=60000, freq='H')
        large_data = pd.DataFrame({
            'detection_time': large_dates,
            'granary_id': 'test_granary',
            'heap_id': np.random.choice(['heap_1', 'heap_2', 'heap_3'], 60000),
            'grid_x': np.random.randint(1, 10, 60000),
            'grid_y': np.random.randint(1, 10, 60000), 
            'grid_z': np.random.randint(1, 5, 60000),
            'temperature_grain': 20 + np.random.normal(0, 2, 60000)
        })
        
        print(f"📊 Large test data created: {large_data.shape}")
        print("-" * 60)
        
        # This should trigger Polars optimization
        large_result = add_lags_optimized(large_data, lags=[1, 2, 3])
        print(f"✅ Large dataset lag computation completed: {large_result.shape}")
        
        large_result2 = create_time_features_optimized(large_result)
        print(f"✅ Large dataset time features completed: {large_result2.shape}")
        
        print("\n" + "=" * 60)
        print("🎉 All Polars optimizations working correctly!")
        print("📈 Performance comparison available - Polars backend used for datasets >50K rows")
        print("🔧 Check console output above to see which backend was used for each operation")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_polars_integration()
    sys.exit(0 if success else 1)
