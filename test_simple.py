#!/usr/bin/env python3

import sys
from pathlib import Path

# Test the filename parsing logic directly
def test_filename_parsing():
    filename = "从化龙潭储备粮库_11仓-01廒-01堆_2023-01-17_to_2025-07-22.parquet"
    filename_stem = Path(filename).stem.lower()
    
    print(f"Original filename: {filename}")
    print(f"Filename stem: {filename_stem}")
    
    # Look for the "_to_" pattern
    if "_to_" in filename_stem:
        print("Found '_to_' pattern")
        before_to = filename_stem.split("_to_")[0]
        print(f"Before '_to_': {before_to}")
        parts = before_to.split('_')
        print(f"Parts: {parts}")
        
        # Find the last date pattern (YYYY-MM-DD format) 
        date_start_index = -1
        for i in range(len(parts) - 2, -1, -1):  # Work backwards
            part = parts[i]
            print(f"Checking part {i}: '{part}'")
            if len(part) == 4 and part.isdigit() and 2000 <= int(part) <= 2100:
                # Check if followed by month and day
                if i + 2 < len(parts):
                    month_part = parts[i + 1]
                    day_part = parts[i + 2]
                    print(f"  Month: '{month_part}', Day: '{day_part}'")
                    if (len(month_part) == 2 and month_part.isdigit() and 
                        len(day_part) == 2 and day_part.isdigit() and
                        1 <= int(month_part) <= 12 and 1 <= int(day_part) <= 31):
                        date_start_index = i
                        print(f"  Found date at index {i}")
                        break
        
        if date_start_index > 0:
            granary_silo_parts = parts[:date_start_index]
            granary_silo_identifier = '_'.join(granary_silo_parts)
            print(f"Extracted identifier: '{granary_silo_identifier}'")
        else:
            print("No date pattern found")

if __name__ == "__main__":
    test_filename_parsing()
