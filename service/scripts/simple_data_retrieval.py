#!/usr/bin/env python3
"""
Simple Data Retrieval Script for SiloFlow
==========================================

Retrieves all data from a selected silo within a date range and saves to parquet file.
Uses the exact queries provided by the user for maximum compatibility.

Usage:
    python simple_data_retrieval.py --granary-name "蚬冈库" --silo-id "41f2257ce3d64083b1b5f8e59e80bc4d" --start-date "2024-07-17" --end-date "2025-07-18"
"""

import argparse
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_data_retrieval.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleDataRetriever:
    """Simple data retriever using exact user queries."""
    
    def __init__(self, db_config: dict):
        """Initialize with database configuration."""
        self.db_config = db_config
        self.engine = self._create_engine()
        
    def _create_engine(self):
        """Create SQLAlchemy engine with proper encoding."""
        password_encoded = urllib.parse.quote_plus(self.db_config['password'])
        connection_string = (
            f"mysql+pymysql://{self.db_config['user']}:{password_encoded}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            f"?charset=utf8mb4"
        )
        return create_engine(connection_string, pool_pre_ping=True, 
                           connect_args={"charset": "utf8mb4"})
    
    def get_all_granaries_and_silos(self) -> pd.DataFrame:
        """Get all granaries and silos with their sub_table_id."""
        query = """
            SELECT 
                loc.storepoint_id, 
                loc.sub_table_id,
                store.store_id, 
                store.store_name 
            FROM cloud_server.base_store_location_other loc
            INNER JOIN cloud_server.v_store_list store 
                ON store.storepoint_id = loc.storepoint_id 
                AND level = '4'
            ORDER BY loc.sub_table_id, store.store_name
        """
        try:
            # Explicitly handle encoding for database queries
            return pd.read_sql(query, self.engine)
        except Exception as e:
            # Log error with safe encoding
            error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
            logger.error(f"Failed to retrieve granaries and silos: {error_msg}")
            raise
            
    def get_granaries_with_details(self) -> pd.DataFrame:
        """Get all granaries and their corresponding silos with detailed information."""
        query = """
            SELECT 
                loc.storepoint_id,
                locs.store_name as granary_name, 
                loc.sub_table_id,
                store.store_id, 
                store.store_name as silo_name 
            FROM cloud_server.base_store_location_other loc
            INNER JOIN cloud_server.v_store_list locs 
                ON locs.store_id = loc.storepoint_id 
                AND locs.level = '1'
            INNER JOIN cloud_server.v_store_list store 
                ON store.storepoint_id = loc.storepoint_id 
                AND store.level = '4'
            ORDER BY loc.sub_table_id, store.store_name
        """
        try:
            # Explicitly handle encoding for database queries  
            return pd.read_sql(query, self.engine)
        except Exception as e:
            # Log error with safe encoding
            error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
            logger.error(f"Failed to retrieve granaries with details: {error_msg}")
            raise
    
    def get_silo_date_range(self, silo_id: str, sub_table_id: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the min and max date range for a specific silo."""
        query = f"""
            SELECT goods.name, goods.fid, minBatch.min, maxBatch.max 
            FROM cloud_server.base_store_location loc
            INNER JOIN cloud_server.base_storehouse house ON house.storepointuuid = loc.fid
            INNER JOIN cloud_server.base_goods_allocation goods ON house.fid = goods.storuuid
            LEFT JOIN (
                SELECT MAX(batch) max, goods_allocation_id 
                FROM cloud_lq.lq_point_history_{sub_table_id} 
                WHERE goods_allocation_id = '{silo_id}' 
                GROUP BY goods_allocation_id
            ) maxBatch ON maxBatch.goods_allocation_id = goods.fid
            LEFT JOIN (
                SELECT MIN(batch) min, goods_allocation_id 
                FROM cloud_lq.lq_point_history_{sub_table_id} 
                WHERE goods_allocation_id = '{silo_id}' 
                GROUP BY goods_allocation_id
            ) minBatch ON minBatch.goods_allocation_id = goods.fid
            WHERE goods.fid = '{silo_id}'
        """
        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty and df['min'].iloc[0] is not None and df['max'].iloc[0] is not None:
                min_date = pd.to_datetime(df['min'].iloc[0])
                max_date = pd.to_datetime(df['max'].iloc[0])
                return min_date, max_date
            return None, None
        except Exception as e:
            # Log error with safe encoding
            error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
            logger.error(f"Failed to get date range for silo {silo_id}: {error_msg}")
            return None, None
    
    def get_silo_data(self, granary_id: str, silo_id: str, sub_table_id: int, 
                     start_date: str, end_date: str) -> pd.DataFrame:
        """Get all data for a specific silo within the date range."""
        logger.info(f"Querying data for silo {silo_id} from {start_date} to {end_date}...")
        
        query = f"""
            SELECT
                loc.fid as storepointId,
                loc.kdmc storepointName,
                loc.kdjd,
                loc.kdwd,
                loc.kqdz,
                point.goods_allocation_id AS storeId,
                allocation.name storeName,
                line.line_no,
                b.layer_no,
                point.batch,
                SUBSTR(SUBSTRING_INDEX(point.temp, ',', b.layer_no), -5) temp,
                b.x_coordinate x,
                b.y_coordinate y,
                b.z_coordinate z,
                ROUND(a3.avg_in_temp, 2) avg_in_temp,
                ROUND(a3.max_temp, 2) max_temp,
                ROUND(a3.min_temp, 2) min_temp,
                ROUND(a3.indoor_temp, 2) indoor_temp,
                ROUND(a3.indoor_humidity, 2) indoor_humidity,
                ROUND(a3.outdoor_temp, 2) outdoor_temp,
                ROUND(a3.outdoor_humidity, 2) outdoor_humidity
            FROM cloud_server.base_store_location loc
            INNER JOIN cloud_server.base_store_line line
            INNER JOIN cloud_server.base_store_point b ON line.fid = b.line_id
            LEFT JOIN cloud_lq.lq_point_history_{sub_table_id} point 
                ON point.line_id = line.fid 
                AND point.goods_allocation_id = '{silo_id}'
            INNER JOIN cloud_server.base_goods_allocation allocation 
                ON allocation.fid = point.goods_allocation_id
            LEFT JOIN cloud_lq.lq_store_history_{sub_table_id} a3 
                ON a3.store_id = line.goods_allocation_id 
                AND a3.batch = point.batch   
                AND a3.store_id = '{silo_id}'
            WHERE
                b.is_ignore = '0'
                AND loc.fid = '{granary_id}'
                AND line.is_active = '1'
                AND point.batch BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'
            ORDER BY storeId, batch, line.line_no, b.layer_no
        """
        try:
            logger.info("Executing SQL query...")
            result = pd.read_sql(query, self.engine)
            logger.info(f"Query completed. Retrieved {len(result)} records.")
            return result
        except Exception as e:
            # Log error with safe encoding
            error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
            logger.error(f"Failed to retrieve silo data: {error_msg}")
            raise
    
    def retrieve_and_save(self, granary_name: str, silo_id: str, 
                         start_date: str, end_date: str, output_dir: str = "data/simple_retrieval"):
        """Retrieve data and save to parquet file."""
        logger.info(f"=== Starting Data Retrieval ===")
        logger.info(f"Target: {granary_name}")
        logger.info(f"Silo ID: {silo_id}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info(f"Output Directory: {output_dir}")
        
        # Step 1: Find silo information
        logger.info("Step 1: Looking up silo information...")
        all_data = self.get_all_granaries_and_silos()
        # Find all rows matching the granary_name
        # First, get possible granary_ids for the given granary_name
        # We need to join with get_granaries_with_details to get granary_name -> storepoint_id mapping
        details_df = self.get_granaries_with_details()
        matching_granary = details_df[details_df['granary_name'] == granary_name]
        if matching_granary.empty:
            logger.error(f"❌ Granary '{granary_name}' not found in database")
            return False
        granary_id = matching_granary['storepoint_id'].iloc[0]
        # Now filter all_data for both storepoint_id and store_id
        silo_info = all_data[(all_data['storepoint_id'] == granary_id) & (all_data['store_id'] == silo_id)]
        if silo_info.empty:
            logger.error(f"❌ Silo '{silo_id}' not found in granary '{granary_name}' (ID: {granary_id})")
            return False
        sub_table_id = silo_info['sub_table_id'].iloc[0]
        silo_name = str(silo_info['store_name'].iloc[0]).encode('utf-8', errors='ignore').decode('utf-8')
        logger.info(f"Found silo: {silo_name}")
        logger.info(f"   Granary ID: {granary_id}")
        logger.info(f"   Sub-table ID: {sub_table_id}")
        
        # Step 2: Get date range validation
        logger.info("Step 2: Validating date range...")
        min_date, max_date = self.get_silo_date_range(silo_id, sub_table_id)
        if min_date and max_date:
            logger.info(f"Silo data available from {min_date} to {max_date}")
        else:
            logger.warning("⚠️ Could not determine data availability")
        
        # Step 3: Retrieve the data
        logger.info("Step 3: Retrieving data from database...")
        data = self.get_silo_data(granary_id, silo_id, sub_table_id, start_date, end_date)
        
        if data.empty:
            logger.warning(f"⚠️ No data found for the specified criteria")
            logger.info("This could mean:")
            logger.info("- No data exists for this date range")
            logger.info("- The silo was not active during this period")
            logger.info("- There might be a configuration issue")
            return False
        
        logger.info(f"Retrieved {len(data)} records")
        logger.info(f"   Columns: {list(data.columns)}")
        logger.info(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Step 4: Create output directory and save
        logger.info("Step 4: Saving data to file...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Clean filename of Chinese characters and invalid characters to prevent encoding issues
        safe_granary_name = granary_name.encode('ascii', errors='ignore').decode('ascii') or 'granary'
        safe_silo_name = silo_name.encode('ascii', errors='ignore').decode('ascii') or 'silo'
        filename = f"{safe_granary_name}_{safe_silo_name}_{start_date}_to_{end_date}.parquet"
        # Clean filename of invalid characters
        filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        file_path = output_path / filename
        
        data.to_parquet(file_path, index=False)
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        
        logger.info(f"Data saved successfully!")
        logger.info(f"   File: {file_path}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        logger.info(f"   Records: {len(data)}")
        
        # Step 5: Show sample of data
        logger.info("Step 5: Data preview (first 3 rows):")
        for i, (_, row) in enumerate(data.head(3).iterrows()):
            logger.info(f"   Row {i+1}: batch={row.get('batch', 'N/A')}, temp={row.get('temp', 'N/A')}, line_no={row.get('line_no', 'N/A')}")
        
        logger.info("=== Data Retrieval Complete ===")
        return True

def load_config(config_path: str = "config/production_config.json") -> dict:
    """Load database configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        # Default configuration
        return {
            "database": {
                "host": "rm-wz9805aymm47k9z3qxo.mysql.rds.aliyuncs.com",
                "port": 3306,
                "database": "cloud_lq",
                "user": "userQuey",
                "password": "UserQ@20240807soft"
            }
        }
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simple SiloFlow Data Retrieval')
    parser.add_argument('--granary-name', help='Name of the granary')
    parser.add_argument('--silo-id', help='ID of the silo (goods_allocation_id)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', default='config/production_config.json', help='Configuration file path')
    parser.add_argument('--output-dir', default='data/simple_retrieval', help='Output directory')
    parser.add_argument('--list-granaries', action='store_true', help='List all granaries and their silos')
    parser.add_argument('--list-silos', action='store_true', help='List all silos with basic info')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize retriever
        retriever = SimpleDataRetriever(config['database'])
        
        # Handle listing operations
        if args.list_granaries:
            logger.info("Listing all granaries and their silos with date ranges...")
            df = retriever.get_granaries_with_details()
            
            print("\n=== Granaries and Their Silos with Date Ranges ===")
            print(f"Found {len(df)} silos across {df['storepoint_id'].nunique()} granaries")
            print("\nRetrieving date ranges for each silo...")
            print()
            
            # Prepare data for CSV with date ranges
            csv_data = []
            
            # Group by granary for better display
            grouped = df.groupby(['storepoint_id', 'granary_name', 'sub_table_id'])
            granary_count = 0
            for (granary_id, granary_name, sub_table_id), group in grouped:
                granary_count += 1
                print(f"[GRANARY {granary_count}/{len(grouped)}] {granary_name} (ID: {granary_id}, Sub-table: {sub_table_id})")
                
                for _, silo in group.iterrows():
                    try:
                        # Get date range for this silo
                        min_date, max_date = retriever.get_silo_date_range(silo['store_id'], sub_table_id)
                        
                        if min_date and max_date:
                            date_info = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                            days_span = (max_date - min_date).days
                            print(f"   [SILO] {silo['silo_name']} ({silo['store_id']}) - {date_info} ({days_span} days)")
                        else:
                            date_info = "No data available"
                            min_date = max_date = None
                            days_span = 0
                            print(f"   [SILO] {silo['silo_name']} ({silo['store_id']}) - {date_info}")
                        
                        # Add to CSV data
                        csv_data.append({
                            'granary_id': granary_id,
                            'granary_name': granary_name,
                            'sub_table_id': sub_table_id,
                            'silo_id': silo['store_id'],
                            'silo_name': silo['silo_name'],
                            'start_date': min_date.strftime('%Y-%m-%d') if min_date else None,
                            'end_date': max_date.strftime('%Y-%m-%d') if max_date else None,
                            'days_span': days_span,
                            'data_available': 'Yes' if min_date and max_date else 'No'
                        })
                        
                    except Exception as e:
                        print(f"   [SILO] {silo['silo_name']} ({silo['store_id']}) - Error getting dates: {str(e)}")
                        csv_data.append({
                            'granary_id': granary_id,
                            'granary_name': granary_name,
                            'sub_table_id': sub_table_id,
                            'silo_id': silo['store_id'],
                            'silo_name': silo['silo_name'],
                            'start_date': None,
                            'end_date': None,
                            'days_span': 0,
                            'data_available': 'Error'
                        })
                print()
            
            # Save detailed CSV with date ranges
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "granaries_silos_with_dates.csv"
            
            csv_df = pd.DataFrame(csv_data)
            csv_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"Detailed data with date ranges saved to: {output_file}")
            
            # Show summary
            silos_with_data = csv_df[csv_df['data_available'] == 'Yes']
            print("[SUMMARY]")
            print(f"   Total granaries: {csv_df['granary_id'].nunique()}")
            print(f"   Total silos: {len(csv_df)}")
            print(f"   Silos with data: {len(silos_with_data)}")
            print(f"   Silos without data: {len(csv_df) - len(silos_with_data)}")
            print(f"   Sub-tables used: {sorted(csv_df['sub_table_id'].unique())}")
            
            if len(silos_with_data) > 0:
                print(f"   Average data span: {silos_with_data['days_span'].mean():.0f} days")
                print(f"   Longest data span: {silos_with_data['days_span'].max()} days")
                print(f"   Shortest data span: {silos_with_data['days_span'].min()} days")
            
            # Show granaries with most silos
            silo_counts = csv_df.groupby(['granary_name']).size().sort_values(ascending=False)
            print("\n[TOP GRANARIES] Granaries with most silos:")
            for granary, count in silo_counts.head(10).items():
                data_count = len(silos_with_data[silos_with_data['granary_name'] == granary])
                print(f"   {granary}: {count} silos ({data_count} with data)")
            
            return
            
        if args.list_silos:
            logger.info("Listing all silos...")
            df = retriever.get_all_granaries_and_silos()
            print("\n=== All Available Silos ===")
            for _, row in df.iterrows():
                print(f"Granary: {row['storepoint_id']} | Silo: {row['store_name']} ({row['store_id']}) | Sub-table: {row['sub_table_id']}")
            return
            
        # Regular data retrieval mode
        if not all([args.granary_name, args.silo_id, args.start_date, args.end_date]):
            parser.error("For data retrieval, --granary-name, --silo-id, --start-date, and --end-date are required")
        
        # Retrieve and save data
        success = retriever.retrieve_and_save(
            args.granary_name,
            args.silo_id,
            args.start_date,
            args.end_date,
            args.output_dir
        )
        
        if success:
            logger.info("Data retrieval completed successfully!")
        else:
            logger.error("Data retrieval failed!")
            sys.exit(1)
            
    except Exception as e:
        # Log error with safe encoding
        error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
        logger.error(f"Error: {error_msg}")
        sys.exit(1)

if __name__ == "__main__":
    import json
    main()
