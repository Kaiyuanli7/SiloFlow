#!/usr/bin/env python3

import sys
from pathlib import Path

# Add service directory to path
service_dir = Path(__file__).parent / "service"
sys.path.insert(0, str(service_dir))

# Test the filtering logic
try:
    from utils.silo_filtering import filter_silos_by_existing_files
    
    # Sample silo data from CSV
    test_silos = [
        {
            'granary_name': '从化龙潭储备粮库',
            'silo_name': '11仓-01廒-01堆',
            'silo_id': '4e70721a631944daae5f771589cdfe28',
            'start_date': '2023-01-17',
            'end_date': '2025-07-22',
            'has_data': True
        },
        {
            'granary_name': '从化龙潭储备粮库',
            'silo_name': '2仓-01廒-01堆',
            'silo_id': '08ce311a37c547769d47685bc1020474',
            'start_date': '2023-01-17',
            'end_date': '2025-07-22',
            'has_data': True
        },
        {
            'granary_name': '贺岗粮库',
            'silo_name': 'P15-1堆',
            'silo_id': 'cf9751d516204d1eb5ba2e69ca3322f1',
            'start_date': '2023-09-25',
            'end_date': '2025-07-22',
            'has_data': True
        }
    ]
    
    print("Testing filtering logic...")
    filtered_silos, skipped_silos = filter_silos_by_existing_files(test_silos)
    
    print(f"\nResults:")
    print(f"Original silos: {len(test_silos)}")
    print(f"Filtered (new): {len(filtered_silos)}")
    print(f"Skipped (existing): {len(skipped_silos)}")
    
    if skipped_silos:
        print(f"\nSkipped silos:")
        for silo in skipped_silos:
            print(f"  - {silo['granary_name']} - {silo['silo_name']}")
    
    if filtered_silos:
        print(f"\nNew silos to process:")
        for silo in filtered_silos:
            print(f"  - {silo['granary_name']} - {silo['silo_name']}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
