#!/usr/bin/env python3
"""
Test script to debug the silo filtering logic.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Add service directory to path for imports
service_dir = Path(__file__).parent
sys.path.insert(0, str(service_dir))

from utils.silo_filtering import filter_silos_by_existing_files, get_simple_retrieval_directory

def debug_filename_parsing():
    """Debug what identifiers are being extracted from filenames"""
    simple_retrieval_dir = get_simple_retrieval_directory()
    if not simple_retrieval_dir:
        print("Simple retrieval directory not found")
        return
    
    existing_files = list(simple_retrieval_dir.glob("*.parquet"))
    print(f"Found {len(existing_files)} parquet files")
    
    # Test parsing a few specific files
    test_files = [
        "从化龙潭储备粮库_11仓-01廒-01堆_2023-01-17_to_2025-07-22.parquet",
        "从化龙潭储备粮库_2仓-01廒-01堆_2023-01-17_to_2025-07-22.parquet", 
        "贺岗粮库_P15-1堆_2023-12-07_to_2025-07-22.parquet"
    ]
    
    extracted_identifiers = set()
    
    for filename in test_files:
        file_path = simple_retrieval_dir / filename
        if file_path.exists():
            print(f"\nProcessing: {filename}")
            filename_stem = file_path.stem.lower()
            print(f"  Stem (lowercase): {filename_stem}")
            
            if "_to_" in filename_stem:
                before_to = filename_stem.split("_to_")[0]
                print(f"  Before '_to_': {before_to}")
                
                parts = before_to.split('_')
                print(f"  Parts: {parts}")
                
                # Look for date pattern working backwards
                date_start_index = -1
                for i in range(len(parts) - 1, -1, -1):
                    part = parts[i]
                    print(f"    Checking part {i}: '{part}'")
                    
                    if '-' in part:
                        date_parts = part.split('-')
                        print(f"      Date parts: {date_parts}")
                        
                        if (len(date_parts) == 3 and
                            len(date_parts[0]) == 4 and date_parts[0].isdigit() and
                            len(date_parts[1]) == 2 and date_parts[1].isdigit() and
                            len(date_parts[2]) == 2 and date_parts[2].isdigit() and
                            2000 <= int(date_parts[0]) <= 2100 and
                            1 <= int(date_parts[1]) <= 12 and
                            1 <= int(date_parts[2]) <= 31):
                            print(f"      Found date at index {i}")
                            date_start_index = i
                            break
                
                if date_start_index > 0:
                    granary_silo_parts = parts[:date_start_index]
                    granary_silo_identifier = '_'.join(granary_silo_parts)
                    print(f"  Extracted identifier: '{granary_silo_identifier}'")
                    extracted_identifiers.add(granary_silo_identifier)
                else:
                    print("  No date found or invalid date position")
            else:
                print("  No '_to_' pattern found")
        else:
            print(f"File {filename} not found")
    
    print(f"\nExtracted identifiers: {extracted_identifiers}")
    
    # Now test against sample silo data
    test_silos = [
        {
            'granary_name': '从化龙潭储备粮库',
            'silo_id': '11仓-01廒-01堆',
            'silo_name': '11仓-01廒-01堆',
            'start_date': '2023-01-17',
            'end_date': '2025-07-22',
            'has_data': True
        },
        {
            'granary_name': '从化龙潭储备粮库',
            'silo_id': '2仓-01廒-01堆',
            'silo_name': '2仓-01廒-01堆', 
            'start_date': '2023-01-17',
            'end_date': '2025-07-22',
            'has_data': True
        },
        {
            'granary_name': '贺岗粮库',
            'silo_id': 'P15-1堆',
            'silo_name': 'P15-1堆',
            'start_date': '2023-12-07', 
            'end_date': '2025-07-22',
            'has_data': True
        }
    ]
    
    print("\nTesting silo matching:")
    for silo in test_silos:
        granary_name = silo.get('granary_name', '').lower()
        silo_name = silo.get('silo_name', '').lower()
        silo_id = silo.get('silo_id', '').lower()
        
        possible_identifiers = [
            f"{granary_name}_{silo_name}",
            f"{granary_name}_{silo_id}",
        ]
        
        print(f"\nSilo: {silo['granary_name']} - {silo['silo_name']}")
        print(f"  Possible identifiers: {possible_identifiers}")
        
        found_match = False
        for identifier in possible_identifiers:
            if identifier in extracted_identifiers:
                print(f"  ✅ MATCH: {identifier}")
                found_match = True
                break
        
        if not found_match:
            print(f"  ❌ NO MATCH")

if __name__ == "__main__":
    debug_filename_parsing()
