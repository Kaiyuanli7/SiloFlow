#!/usr/bin/env python3
"""
Get Silos for Granary
=====================

This script retrieves all silos for a specific granary from the database.
"""

import argparse
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
import urllib.parse
import pandas as pd

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(service_dir))

from utils.database_utils import DatabaseManager, CLIUtils, ValidationUtils
from utils.database_utils import ValidationError

def get_silos_for_granary(connection, granary_name):
    """Get all silos for a specific granary."""
    try:
        # Query to get silos for the specified granary using the correct table structure
        # Handle both granary IDs and names by checking both columns
        query = """
        SELECT DISTINCT 
            store.store_id as silo_id,
            store.store_name as silo_name,
            store.storepoint_id as granary_id,
            locs.store_name as granary_name
        FROM cloud_server.v_store_list store
        INNER JOIN cloud_server.v_store_list locs 
            ON locs.store_id = store.storepoint_id AND locs.level = '1'
        WHERE (locs.store_name = %s OR locs.store_id = %s) AND store.level = '4'
        ORDER BY store.store_name
        """
        
        df = pd.read_sql(query, connection, params=[granary_name, granary_name])
        
        if df.empty:
            print(f"No silos found for granary: {granary_name}")
            return
        
        print(f"Silos for Granary: {df.iloc[0]['granary_name']} (ID: {df.iloc[0]['granary_id']})")
        print("=" * 60)
        print(f"Total silos found: {len(df)}")
        print()
        
        for _, row in df.iterrows():
            print(f"  Silo: {row['silo_name']} (ID: {row['silo_id']}):")
            print(f"    Granary ID: {row['granary_id']}")
            print(f"    Granary Name: {row['granary_name']}")
            print()
            
    except Exception as e:
        print(f"Error querying silos for granary {granary_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Get silos for a specific granary')
    CLIUtils.add_common_args(parser)
    parser.add_argument('--granary', required=True, help='Name of the granary to get silos for')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not ValidationUtils.validate_required_args(args, ['granary']):
        print("Error: --granary argument is required")
        sys.exit(1)
    
    if not ValidationUtils.validate_config_file(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Load configuration
        config = DatabaseManager.load_config(args.config)
        
        # Connect to database
        connection = DatabaseManager.get_connection(config)
        
        try:
            # Get silos for the specified granary
            get_silos_for_granary(connection, args.granary)
        finally:
            connection.close()
            
    except ValidationError as e:
        print(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 