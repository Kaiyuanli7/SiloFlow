#!/usr/bin/env python3
"""
Get Date Ranges for All Silos
=============================

This script retrieves date ranges for all silos across all granaries in the database.
"""

import argparse
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
import urllib.parse
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(service_dir))

from utils.database_utils import DatabaseManager, CLIUtils, ValidationUtils
from utils.database_utils import ValidationError

def get_date_ranges(config_path: str = "streaming_config.json"):
    """Get date ranges for all silos in the database."""
    
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
        
        print("Getting Date Ranges for All Silos")
        print("=" * 60)
        
        # Query to get all granaries and their silos with date ranges
        query = """
        SELECT 
            loc.storepoint_id as granary_id,
            locs.store_name as granary_name,
            loc.sub_table_id,
            store.store_id as silo_id,
            store.store_name as silo_name,
            store.level
        FROM cloud_server.base_store_location_other loc
        INNER JOIN cloud_server.v_store_list locs 
            ON locs.store_id = loc.storepoint_id AND locs.level = '1'
        INNER JOIN cloud_server.v_store_list store 
            ON store.storepoint_id = loc.storepoint_id AND store.level = '4'
        ORDER BY loc.sub_table_id, locs.store_name, store.store_name
        """
        
        df_granaries = pd.read_sql(query, engine)
        
        if df_granaries.empty:
            print("No granaries found in database.")
            return
        
        print(f"Found {len(df_granaries)} silos across {df_granaries['granary_name'].nunique()} granaries")
        print()
        
        # Group by granary and get date ranges for each silo
        results = []
        
        # Get unique granaries
        unique_granaries = df_granaries[['granary_id', 'granary_name', 'sub_table_id']].drop_duplicates()
        
        for _, granary_row in unique_granaries.iterrows():
            granary_id = str(granary_row['granary_id'])
            granary_name = str(granary_row['granary_name'])
            sub_table_id = int(granary_row['sub_table_id'])
            
            print(f"Granary: {granary_name} (ID: {granary_id}, Table: {sub_table_id})")
            print("-" * 50)
            
            # Get silos for this granary
            silos = df_granaries[df_granaries['granary_id'] == granary_id]
            
            for _, silo_row in silos.iterrows():
                silo_id = str(silo_row['silo_id'])
                silo_name = str(silo_row['silo_name'])
                
                # Get date range for this silo
                start_date, end_date = get_silo_date_range(engine, silo_id, sub_table_id)
                
                if start_date and end_date:
                    print(f"  Silo: {silo_name} (ID: {silo_id}):")
                    print(f"     Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"     End:   {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"     Days:  {(end_date - start_date).days}")
                    
                    results.append({
                        'granary_id': granary_id,
                        'granary_name': granary_name,
                        'sub_table_id': sub_table_id,
                        'silo_id': silo_id,
                        'silo_name': silo_name,
                        'start_date': start_date,
                        'end_date': end_date,
                        'days_span': (end_date - start_date).days
                    })
                else:
                    print(f"  Silo: {silo_name} (ID: {silo_id}): No data found")
            
            print()
        
        # Summary
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Granaries: {len(unique_granaries)}")
        print(f"Total Silos: {len(df_granaries)}")
        print(f"Silos with Data: {len(results)}")
        
        if results:
            # Overall date range
            overall_start = min(r['start_date'] for r in results)
            overall_end = max(r['end_date'] for r in results)
            print(f"Overall Date Range: {overall_start.strftime('%Y-%m-%d')} to {overall_end.strftime('%Y-%m-%d')}")
            print(f"Total Span: {(overall_end - overall_start).days} days")
            
            # Statistics
            days_spans = [r['days_span'] for r in results]
            print(f"Average Silo Span: {sum(days_spans) / len(days_spans):.1f} days")
            print(f"Min Silo Span: {min(days_spans)} days")
            print(f"Max Silo Span: {max(days_spans)} days")
        
        print()
        print("Usage Examples:")
        print("=" * 30)
        print("For incremental retrieval (last 7 days):")
        print("python automated_data_retrieval.py --incremental --days 7")
        print()
        print("For full retrieval of specific granary:")
        print(f"python automated_data_retrieval.py --full-retrieval --granary '{results[0]['granary_name'] if results else 'GranaryName'}'")
        print()
        print("For specific date range:")
        if results:
            start_date = overall_start.strftime('%Y-%m-%d')
            end_date = overall_end.strftime('%Y-%m-%d')
            print(f"python automated_data_retrieval.py --date-range --start {start_date} --end {end_date}")
        
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found!")
        print("Please run: python setup_data_retrieval.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error getting date ranges: {e}")
        sys.exit(1)

def get_silo_date_range(engine, heap_id: str, sub_table_id: int):
    """Get the date range for a specific silo."""
    try:
        # Query to get min and max dates for this silo
        query = f"""
        SELECT 
            MIN(batch) as earliest_date,
            MAX(batch) as latest_date
        FROM cloud_lq.lq_point_history_{sub_table_id}
        WHERE goods_allocation_id = '{heap_id}'
        """
        
        df = pd.read_sql(query, engine)
        
        if not df.empty and df['earliest_date'].iloc[0] is not None:
            earliest_date = pd.to_datetime(df['earliest_date'].iloc[0])
            latest_date = pd.to_datetime(df['latest_date'].iloc[0])
            return earliest_date, latest_date
        
        return None, None
        
    except Exception as e:
        print(f"    Error getting dates for silo {heap_id}: {e}")
        return None, None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Get date ranges for all silos in the database")
    CLIUtils.add_common_args(parser)
    
    args = parser.parse_args()
    
    if not ValidationUtils.validate_config_file(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        get_date_ranges(args.config)
    except ValidationError as e:
        print(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 