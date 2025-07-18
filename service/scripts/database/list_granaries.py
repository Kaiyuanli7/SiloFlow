#!/usr/bin/env python3
"""
List Available Granaries
========================

This script lists all available granaries in the database with their IDs and silo counts.
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

def list_granaries(config_path: str = "streaming_config.json"):
    """List all available granaries in the database."""
    
    try:
        # Load configuration
        config = DatabaseManager.load_config(config_path)
        
        # Create database connection
        db_config = DatabaseManager.get_db_config(config)
        password_encoded = urllib.parse.quote_plus(db_config['password'])
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{password_encoded}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        # Query to get all granaries and their silo counts
        query = """
        SELECT 
            loc.storepoint_id as granary_id,
            locs.store_name as granary_name,
            loc.sub_table_id,
            COUNT(store.store_id) as silo_count
        FROM cloud_server.base_store_location_other loc
        INNER JOIN cloud_server.v_store_list locs 
            ON locs.store_id = loc.storepoint_id AND locs.level = '1'
        INNER JOIN cloud_server.v_store_list store 
            ON store.storepoint_id = loc.storepoint_id AND store.level = '4'
        GROUP BY loc.storepoint_id, locs.store_name, loc.sub_table_id
        ORDER BY loc.sub_table_id, locs.store_name
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("No granaries found in database.")
            return
        
        print("Available Granaries:")
        print("=" * 80)
        print(f"{'ID':<40} {'Name':<25} {'Table':<8} {'Silos':<6}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['granary_id']:<40} {row['granary_name']:<25} {row['sub_table_id']:<8} {row['silo_count']:<6}")
        
        print("-" * 80)
        print(f"Total granaries: {len(df)}")
        print(f"Total silos: {df['silo_count'].sum()}")
        
        print("\nUsage Examples:")
        print("=" * 50)
        print("To retrieve data for a specific granary:")
        print()
        
        # Show examples for first few granaries
        for i, row in df.head(3).iterrows():
            granary_name = row['granary_name']
            print(f"# {granary_name}:")
            print(f"python automated_data_retrieval.py --full-retrieval --granary '{granary_name}'")
            print(f"python automated_data_retrieval.py --incremental --days 7 --granary '{granary_name}'")
            print(f"python automated_data_retrieval.py --date-range --start 2024-01-01 --end 2024-12-31 --granary '{granary_name}'")
            print()
        
        if len(df) > 3:
            print(f"... and {len(df) - 3} more granaries")
        
        print("\nYou can also use granary IDs instead of names:")
        print(f"python automated_data_retrieval.py --full-retrieval --granary '{df.iloc[0]['granary_id']}'")
        
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found!")
        print("Please run: python setup_data_retrieval.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error listing granaries: {e}")
        sys.exit(1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="List available granaries in the database")
    CLIUtils.add_common_args(parser)
    
    args = parser.parse_args()
    
    if not ValidationUtils.validate_config_file(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        list_granaries(args.config)
    except ValidationError as e:
        print(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    main() 