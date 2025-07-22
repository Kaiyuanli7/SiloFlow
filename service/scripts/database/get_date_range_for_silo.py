#!/usr/bin/env python3
"""
Get Date Range for Silo
=======================

This script retrieves the date range for a specific silo from the database.
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

def get_date_range_for_silo(connection, granary_name, silo_name):
    """Get date range for a specific silo."""
    try:
        # First, get the silo information and sub_table_id
        # Handle both granary IDs and names by checking both columns
        silo_query = """
        SELECT 
            store.store_id as silo_id,
            store.store_name as silo_name,
            store.storepoint_id as granary_id,
            locs.store_name as granary_name,
            loc.sub_table_id
        FROM cloud_server.v_store_list store
        INNER JOIN cloud_server.v_store_list locs 
            ON locs.store_id = store.storepoint_id AND locs.level = '1'
        INNER JOIN cloud_server.base_store_location_other loc
            ON loc.storepoint_id = store.storepoint_id
        WHERE (locs.store_name = %s OR locs.store_id = %s) AND store.store_name = %s AND store.level = '4'
        """
        
        silo_df = pd.read_sql(silo_query, connection, params=[granary_name, granary_name, silo_name])
        
        if silo_df.empty:
            print(f"No silo found: {silo_name} in granary: {granary_name}")
            return
        
        silo_row = silo_df.iloc[0]
        silo_id = silo_row['silo_id']
        sub_table_id = silo_row['sub_table_id']
        
        # Now get the date range from the temperature data table
        date_query = f"""
        SELECT 
            MIN(batch) as earliest_date,
            MAX(batch) as latest_date,
            COUNT(*) as total_records
        FROM cloud_lq.lq_point_history_{sub_table_id}
        WHERE goods_allocation_id = %s
        """
        
        date_df = pd.read_sql(date_query, connection, params=[silo_id])
        
        if date_df.empty or date_df['earliest_date'].iloc[0] is None:
            print(f"No temperature data found for silo: {silo_name} in granary: {silo_row['granary_name']}")
            return
        
        date_row = date_df.iloc[0]
        
        print(f"Date Range for Silo: {silo_name}")
        print(f"Granary: {silo_row['granary_name']} (ID: {silo_row['granary_id']})")
        print("=" * 60)
        print(f"Silo ID: {silo_id}")
        print(f"Granary ID: {silo_row['granary_id']}")
        print(f"Sub Table ID: {sub_table_id}")
        print(f"Total Records: {date_row['total_records']}")
        print()
        
        print(f"Earliest Date: {date_row['earliest_date']}")
        print(f"Latest Date: {date_row['latest_date']}")
        print(f"Date Range: {date_row['earliest_date']} to {date_row['latest_date']}")
            
    except Exception as e:
        print(f"Error querying date range for silo {silo_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Get date range for a specific silo')
    CLIUtils.add_common_args(parser)
    parser.add_argument('--granary', required=True, help='Name of the granary')
    parser.add_argument('--silo', required=True, help='Name of the silo to get date range for')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not ValidationUtils.validate_required_args(args, ['granary', 'silo']):
        print("Error: --granary and --silo arguments are required")
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
            # Get date range for the specified silo
            get_date_range_for_silo(connection, args.granary, args.silo)
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