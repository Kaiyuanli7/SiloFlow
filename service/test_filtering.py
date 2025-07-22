#!/usr/bin/env python3
"""
Test script to verify the silo filtering logic.
"""

import sys
from pathlib import Path
import pandas as pd

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent
sys.path.insert(0, str(service_dir))

from utils.silo_filtering import filter_silos_by_existing_files

def test_filtering():
    # Load silos from CSV
    csv_path = Path("data/simple_retrieval/granaries_silos_with_dates.csv")
    if not csv_path.exists():
        print(f"CSV file not found at: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    data_silos = df[df['data_available'] == 'Yes'].copy()
    
    silos_with_data = []
    for _, row in data_silos.iterrows():
        silos_with_data.append({
            'granary_name': str(row['granary_name']),
            'silo_id': str(row['silo_id']),
            'silo_name': str(row['silo_name']),
            'start_date': str(row['start_date']) if pd.notna(row['start_date']) else '',
            'end_date': str(row['end_date']) if pd.notna(row['end_date']) else '',
            'has_data': True
        })
    
    print(f"Total silos with data from CSV: {len(silos_with_data)}")
    
    # Run filtering
    filtered_silos, skipped_silos = filter_silos_by_existing_files(silos_with_data)
    
    print(f"Filtered silos (new): {len(filtered_silos)}")
    print(f"Skipped silos (existing): {len(skipped_silos)}")
    print(f"Total: {len(filtered_silos) + len(skipped_silos)}")
    
    # Show some examples of skipped silos
    print("\nFirst 5 skipped silos:")
    for i, silo in enumerate(skipped_silos[:5]):
        print(f"  {i+1}. {silo['granary_name']} - {silo['silo_name']}")
    
    print("\nFirst 5 new silos:")
    for i, silo in enumerate(filtered_silos[:5]):
        print(f"  {i+1}. {silo['granary_name']} - {silo['silo_name']}")

if __name__ == "__main__":
    test_filtering()
