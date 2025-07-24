#!/usr/bin/env python3
"""
Simple test to debug filename parsing.
"""

def test_filename_parsing():
    # Test parsing logic directly
    filename = "从化龙潭储备粮库_11仓-01廒-01堆_2023-01-17_to_2025-07-22.parquet"
    print(f"Testing filename: {filename}")
    
    # Remove .parquet extension and convert to lowercase
    filename_stem = filename.replace('.parquet', '').lower()
    print(f"Stem: {filename_stem}")
    
    if "_to_" in filename_stem:
        before_to = filename_stem.split("_to_")[0]
        print(f"Before '_to_': {before_to}")
        
        parts = before_to.split('_')
        print(f"Parts: {parts}")
        
        # Look for date pattern working backwards
        date_start_index = -1
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            print(f"  Checking part {i}: '{part}'")
            
            if '-' in part:
                date_parts = part.split('-')
                print(f"    Date parts: {date_parts}")
                
                if (len(date_parts) == 3 and
                    len(date_parts[0]) == 4 and date_parts[0].isdigit() and
                    len(date_parts[1]) == 2 and date_parts[1].isdigit() and
                    len(date_parts[2]) == 2 and date_parts[2].isdigit() and
                    2000 <= int(date_parts[0]) <= 2100 and
                    1 <= int(date_parts[1]) <= 12 and
                    1 <= int(date_parts[2]) <= 31):
                    print(f"    ✅ Valid date found at index {i}")
                    date_start_index = i
                    break
                else:
                    print(f"    ❌ Invalid date format")
        
        if date_start_index > 0:
            granary_silo_parts = parts[:date_start_index]
            granary_silo_identifier = '_'.join(granary_silo_parts)
            print(f"Extracted identifier: '{granary_silo_identifier}'")
            return granary_silo_identifier
        else:
            print("No valid date found")
            return None
    else:
        print("No '_to_' pattern found")
        return None

def test_silo_matching():
    print("\n" + "="*50)
    print("Testing silo matching")
    
    # Expected identifier from filename parsing
    expected_identifier = "从化龙潭储备粮库_11仓-01廒-01堆"
    
    # Silo data from CSV
    silo = {
        'granary_name': '从化龙潭储备粮库',
        'silo_name': '11仓-01廒-01堆',
        'silo_id': '11仓-01廒-01堆'
    }
    
    granary_name = silo['granary_name'].lower()
    silo_name = silo['silo_name'].lower()
    silo_id = silo['silo_id'].lower()
    
    possible_identifiers = [
        f"{granary_name}_{silo_name}",
        f"{granary_name}_{silo_id}",
    ]
    
    print(f"Expected from filename: '{expected_identifier}'")
    print(f"Possible from silo data: {possible_identifiers}")
    
    for identifier in possible_identifiers:
        if identifier == expected_identifier:
            print(f"✅ MATCH: {identifier}")
            return True
        else:
            print(f"❌ NO MATCH: {identifier}")
    
    return False

if __name__ == "__main__":
    identifier = test_filename_parsing()
    test_silo_matching()
