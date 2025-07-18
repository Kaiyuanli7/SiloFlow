#!/usr/bin/env python3
"""
Test script to verify the output format of database scripts.
"""

import subprocess
import sys
from pathlib import Path

def test_list_granaries():
    """Test the list_granaries.py script output."""
    print("Testing list_granaries.py...")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "list_granaries.py"
    cmd = ["python", str(script_path), "--config", "streaming_config.json"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"Return code: {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_get_silos():
    """Test the get_silos_for_granary.py script output."""
    print("\nTesting get_silos_for_granary.py...")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "get_silos_for_granary.py"
    cmd = ["python", str(script_path), "--config", "streaming_config.json", "--granary", "中正粮食储备库"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"Return code: {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Testing database scripts output format...")
    print()
    
    # Test list_granaries
    granaries_output = test_list_granaries()
    
    # Test get_silos
    silos_output = test_get_silos()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"list_granaries.py: {'SUCCESS' if granaries_output else 'FAILED'}")
    print(f"get_silos_for_granary.py: {'SUCCESS' if silos_output else 'FAILED'}") 