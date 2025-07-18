#!/usr/bin/env python3
"""
Run Data Streaming - Usage Examples
===================================

This script demonstrates how to use the SQL Data Streamer for different scenarios.
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stderr:
            print("Error output:", e.stderr)
        return False

def main():
    """Main function with usage examples."""
    print("SiloFlow SQL Data Streamer - Usage Examples")
    print("=" * 50)
    
    # Check if sql_data_streamer.py exists
    streamer_path = Path("sql_data_streamer.py")
    if not streamer_path.exists():
        print(f"ERROR: {streamer_path} not found!")
        print("Please ensure you're running this from the service directory.")
        sys.exit(1)
    
    # Example 1: Create default configuration
    print("\n1. Creating default configuration file...")
    if not run_command([
        sys.executable, "sql_data_streamer.py", 
        "--create-config", "streaming_config.json"
    ], "Create default configuration"):
        return
    
    # Example 2: Stream data for last month (data only, no pipeline)
    print("\n2. Example: Stream data for last month (data only)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"This would stream data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Command that would be run:")
    print(f"python sql_data_streamer.py --start-date {start_date.strftime('%Y-%m-%d')} --end-date {end_date.strftime('%Y-%m-%d')} --no-pipeline")
    
    # Example 3: Stream and process through pipeline
    print("\n3. Example: Stream and process through pipeline...")
    print("This would stream data AND process it through the automated pipeline:")
    print(f"python sql_data_streamer.py --start-date {start_date.strftime('%Y-%m-%d')} --end-date {end_date.strftime('%Y-%m-%d')}")
    
    # Example 4: Using custom configuration
    print("\n4. Example: Using custom configuration...")
    print("python sql_data_streamer.py --start-date 2024-12-01 --end-date 2025-01-01 --config streaming_config.json")
    
    # Show what files would be created
    print("\n" + "="*60)
    print("OUTPUT FILES CREATED:")
    print("="*60)
    print("1. data/streaming/granaries/[granary_name].parquet - Raw granary data")
    print("2. data/processed/[granary_name]_processed.parquet - Preprocessed data")
    print("3. models/[granary_name]_forecast_model.joblib - Trained models")
    print("4. sql_data_streamer.log - Processing logs")
    
    # Show recommended usage patterns
    print("\n" + "="*60)
    print("RECOMMENDED USAGE PATTERNS:")
    print("="*60)
    print("1. DEVELOPMENT/TESTING (small date range):")
    print("   python sql_data_streamer.py --start-date 2024-12-01 --end-date 2024-12-07 --no-pipeline")
    print()
    print("2. PRODUCTION (full processing):")
    print("   python sql_data_streamer.py --start-date 2024-12-01 --end-date 2025-01-01")
    print()
    print("3. DATA ONLY (no model training):")
    print("   python sql_data_streamer.py --start-date 2024-12-01 --end-date 2025-01-01 --no-pipeline")
    print()
    print("4. INCREMENTAL PROCESSING (daily):")
    print("   python sql_data_streamer.py --start-date 2025-01-01 --end-date 2025-01-02")
    
    # Performance tips
    print("\n" + "="*60)
    print("PERFORMANCE TIPS:")
    print("="*60)
    print("1. Start with small date ranges for testing")
    print("2. Monitor memory usage during processing")
    print("3. Use --no-pipeline for data collection only")
    print("4. Process models separately after data collection")
    print("5. Check logs for memory adjustment messages")
    
    print("\n" + "="*60)
    print("READY TO USE!")
    print("="*60)
    print("Configuration file created: streaming_config.json")
    print("You can now run the data streamer with your desired date range.")
    print("Start with a small date range to test the system.")

if __name__ == "__main__":
    main() 