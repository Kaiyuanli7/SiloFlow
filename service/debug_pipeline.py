#!/usr/bin/env python3
"""
Test granary_pipeline directly to debug data_paths error
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_granary_pipeline():
    try:
        print("Testing granary_pipeline import...")
        from granary_pipeline import run_complete_pipeline
        print("✓ Successfully imported run_complete_pipeline")
        
        # Check if the granary file exists
        granary_file = "data/granaries/中软粮情验证.parquet"
        if os.path.exists(granary_file):
            print(f"✓ Granary file exists: {granary_file}")
        else:
            print(f"✗ Granary file not found: {granary_file}")
            return
        
        print("Testing function call...")
        result = run_complete_pipeline(
            granary_csv=granary_file,
            granary_name='中软粮情验证',
            skip_train=True,
            force_retrain=False,
            changed_silos=None
        )
        
        print("✓ Function call succeeded!")
        print(f"Result keys: {list(result.keys())}")
        print(f"Success: {result.get('success', 'Unknown')}")
        
        if 'errors' in result and result['errors']:
            print(f"Errors: {result['errors']}")
        
        if 'steps_completed' in result:
            print(f"Steps completed: {result['steps_completed']}")
            
        # Check if processed file was created
        processed_files = []
        processed_dir = "data/processed"
        if os.path.exists(processed_dir):
            for f in os.listdir(processed_dir):
                if '中软粮情验证' in f:
                    processed_files.append(f)
        
        print(f"Processed files created: {processed_files}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_granary_pipeline()
