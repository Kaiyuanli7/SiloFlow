#!/usr/bin/env python3
"""
SQL Data Streamer for SiloFlow
==============================

Automatically retrieves raw CSV data from MySQL database for all granaries and silos.
Connects to the pipeline for processing without training or forecasting.

Usage:
    python sql_data_streamer.py --start-date 2024-01-01 --end-date 2024-12-31
    python sql_data_streamer.py --config streaming_config.json --no-pipeline
    python sql_data_streamer.py --create-config config.json
"""

import json
import logging
import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Union
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import pymysql
from tqdm import tqdm
import psutil
import time
import urllib.parse
import tempfile
import gc
import atexit
import signal
from functools import lru_cache

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(service_dir))

from utils.database_utils import CLIUtils, ValidationUtils, DatabaseManager
from utils.database_utils import ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_data_streamer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SQLDataStreamer:
    """Main class for streaming data from MySQL database."""
    
    def __init__(self, config_path: str = "streaming_config.json"):
        """Initialize the data streamer with configuration."""
        # Load configuration
        config = DatabaseManager.load_config(config_path)
        
        # Create database connection
        db_config = DatabaseManager.get_db_config(config)
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        
        # Use centralized data paths if available, otherwise use config
        try:
            from utils.data_paths import data_paths
            self.data_paths = data_paths
            self.output_dir = self.data_paths.get_granaries_dir()
            self.granaries_dir = self.output_dir
        except ImportError:
            # Fallback to config-based paths
            self.output_dir = Path(self.config['processing']['output_dir'])
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create granaries subdirectory for the ingestion pipeline
            self.granaries_dir = self.output_dir / "granaries"
            self.granaries_dir.mkdir(exist_ok=True)
        
        # Processing parameters - optimized for memory safety
        self.initial_chunk_size = self.config['processing']['initial_chunk_size']
        self.min_chunk_size = self.config['processing']['min_chunk_size']
        self.max_chunk_size = self.config['processing']['max_chunk_size']
        self.memory_threshold = self.config['processing']['memory_threshold_percent']
        
        # Enhanced memory safety parameters
        self.max_records_per_batch = self.config['processing'].get('max_records_per_batch', 50000)  # Reduced from 100000
        self.memory_safety_threshold = self.config['processing'].get('memory_safety_threshold_percent', 50)  # Reduced from 60
        self.batch_timeout_seconds = self.config['processing'].get('batch_timeout_seconds', 180)  # Reduced from 300
        
        # New memory management parameters
        self.force_gc_threshold = self.config['processing'].get('force_gc_threshold_percent', 70)
        self.pause_duration_seconds = self.config['processing'].get('pause_duration_seconds', 15)
        self.max_retries_per_chunk = self.config['processing'].get('max_retries_per_chunk', 3)
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        
    @lru_cache(maxsize=128)
    def is_small_data_request(self, start_date: datetime, end_date: datetime, granary_count: int) -> bool:
        """
        Determine if this is a small data request that can use the fast path.
        
        Args:
            start_date: Start date of the request
            end_date: End date of the request
            granary_count: Number of granaries in the request
            
        Returns:
            bool: True if this is a small request that can use the fast path
        """
        days_difference = (end_date - start_date).days
        return days_difference <= 1 and granary_count == 1
        
    def get_silo_data_fast(self, heap_id: str, sub_table_id: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fast path for retrieving small amounts of data without chunking.
        
        Args:
            heap_id: ID of the silo
            sub_table_id: ID of the subtable
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            pd.DataFrame: Retrieved data
        """
        query = f"""
            SELECT *
            FROM measurement_{sub_table_id}
            WHERE heap_id = %s
            AND timestamp BETWEEN %s AND %s
        """
        try:
            with self.engine.connect() as connection:
                return pd.read_sql(query, connection, params=[heap_id, start_date, end_date])
        except Exception as e:
            logger.error(f"Error in fast data retrieval for silo {heap_id}: {e}")
            return pd.DataFrame()
            
    def stream_granary_data(self, start_date: datetime, end_date: datetime,
                          granary_filter: Optional[List[int]] = None,
                          missing_ranges: Optional[Dict] = None,
                          batch_size: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Stream granary data with optimized paths for both small and large requests.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            granary_filter: Optional list of granary IDs to filter by
            missing_ranges: Optional dict of missing date ranges by silo
            batch_size: Optional batch size override
            
        Yields:
            pd.DataFrame: Chunks of granary data
        """
        # Get granaries and silos information
        granaries_df = self.get_all_granaries_and_silos()
        if granary_filter:
            granaries_df = granaries_df[
                granaries_df['storepoint_id'].isin(granary_filter) if isinstance(granary_filter[0], int)
                else granaries_df['granary_name'].isin(granary_filter)
            ]
        
        if granaries_df.empty:
            logger.warning("No granaries found matching the filter")
            return

        # Check if this is a small data request that can use the fast path
        is_small_request = self.is_small_data_request(
            start_date, 
            end_date,
            len(granaries_df['sub_table_id'].unique())
        )

        if is_small_request:
            logger.info("Using fast path for small data request")
            for _, silo_info in granaries_df.iterrows():
                heap_id = str(silo_info['store_id'])
                sub_table_id = int(silo_info['sub_table_id'])
                
                data = self.get_silo_data_fast(heap_id, sub_table_id, start_date, end_date)
                if not data.empty:
                    data['granary_id'] = silo_info['storepoint_id']
                    data['granary_name'] = silo_info['granary_name']
                    yield data
            return

        # For larger requests, use the memory-managed approach
        total_silos = len(granaries_df)
        batch_size = batch_size or self.initial_chunk_size
        current_batch = []
        current_batch_size = 0
        
        with tqdm(total=total_silos, desc="Processing silos") as pbar:
            for silo_idx, (_, silo_info) in enumerate(granaries_df.iterrows()):
                try:
                    heap_id = str(silo_info['store_id'])
                    heap_name = silo_info['store_name']
                    sub_table_id = int(silo_info['sub_table_id'])
                    
                    logger.info(f"Processing silo {silo_idx + 1}/{total_silos}: {heap_name} (ID: {heap_id})")
                    
                    # Only check memory if we have accumulated data
                    if current_batch_size > batch_size:
                        # Simple memory check and cleanup
                        if psutil.virtual_memory().percent > self.memory_threshold:
                            gc.collect()
                            time.sleep(1)
                        
                        # If memory is still high, yield current batch
                        if psutil.virtual_memory().percent > self.memory_threshold and current_batch:
                            logger.info("Memory threshold reached, yielding current batch")
                            yield pd.concat(current_batch, ignore_index=True)
                            current_batch = []
                            current_batch_size = 0
                            gc.collect()

                    # Handle missing ranges if provided
                    if missing_ranges and heap_id in missing_ranges:
                        for range_info in missing_ranges[heap_id]['missing_ranges']:
                            data = self.get_silo_data_fast(
                                heap_id, 
                                sub_table_id,
                                range_info['start'],
                                range_info['end']
                            )
                            if not data.empty:
                                data['granary_id'] = silo_info['storepoint_id']
                                data['granary_name'] = silo_info['granary_name']
                                current_batch.append(data)
                                current_batch_size += len(data)
                    else:
                        # Get the full date range for this silo
                        silo_min_date, silo_max_date = self.get_silo_date_range(heap_id, sub_table_id)
                        if silo_min_date is None or silo_max_date is None:
                            logger.warning(f"No data available for silo {heap_id}")
                            pbar.update(1)
                            continue

                        # Use the actual date range
                        data = self.get_silo_data_fast(
                            heap_id,
                            sub_table_id,
                            max(start_date, silo_min_date),
                            min(end_date, silo_max_date)
                        )
                        if not data.empty:
                            data['granary_id'] = silo_info['storepoint_id']
                            data['granary_name'] = silo_info['granary_name']
                            current_batch.append(data)
                            current_batch_size += len(data)

                    # Yield batch if it's full
                    if current_batch_size >= batch_size:
                        logger.info(f"Batch full ({current_batch_size} records), yielding...")
                        yield pd.concat(current_batch, ignore_index=True)
                        current_batch = []
                        current_batch_size = 0
                        gc.collect()

                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error processing silo {heap_id}: {e}")
                    pbar.update(1)
                    continue

        # Yield any remaining data
        if current_batch:
            yield pd.concat(current_batch, ignore_index=True)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Track resources for cleanup
        self._temp_files = []
        self._active_connections = []
        
    def _cleanup(self):
        """Cleanup resources on exit."""
        try:
            # Close database connections
            if hasattr(self, 'engine'):
                self.engine.dispose()
            
            # Remove temporary files
            for temp_file in self._temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.info(f"Received signal {signum}, cleaning up...")
        self._cleanup()
        sys.exit(0)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            config = DatabaseManager.load_config(config_path)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _create_engine(self):
        """Create SQLAlchemy engine for database connection."""
        db_config = self.config['database']
        password_encoded = urllib.parse.quote_plus(db_config['password'])
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{password_encoded}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        try:
            engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=1800,  # Reduced from 3600
                pool_size=5,  # Limit pool size
                max_overflow=10,  # Limit overflow
                connect_args={
                    'charset': 'utf8mb4',
                    'connect_timeout': 60,
                    'read_timeout': 300,
                    'write_timeout': 300
                }
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return engine
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    def get_all_granaries_and_silos(self) -> pd.DataFrame:
        """
        Get all granaries and their associated silos using the provided query.
        
        Returns DataFrame with columns:
        - storepoint_id (granary_id)
        - granary_name (granary_name) 
        - sub_table_id (table suffix like 77)
        - store_id (heap_id)
        - silo_name (heap_name)
        """
        query = """
        SELECT 
            loc.storepoint_id,
            locs.store_name as granary_name,
            loc.sub_table_id,
            store.store_id,
            store.store_name as silo_name
        FROM cloud_server.base_store_location_other loc
        INNER JOIN cloud_server.v_store_list locs 
            ON locs.store_id = loc.storepoint_id AND locs.level = '1'
        INNER JOIN cloud_server.v_store_list store 
            ON store.storepoint_id = loc.storepoint_id AND store.level = '4'
        ORDER BY loc.sub_table_id, store.store_name
        """
        
        try:
            logger.info("Retrieving all granaries and silos...")
            df = pd.read_sql(query, self.engine)
            logger.info(f"Found {len(df)} silos across {df['sub_table_id'].nunique()} granaries")
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve granaries and silos: {e}")
            raise
    
    def estimate_data_size(self, granaries_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> Dict:
        """
        Estimate the total data size and memory requirements before processing.
        
        Returns:
            Dictionary with size estimates and recommendations
        """
        logger.info("Estimating data size and memory requirements...")
        
        total_silos = len(granaries_df)
        estimated_records = 0
        silos_with_data = 0
        
        # Sample a few silos to estimate average records per silo
        sample_size = min(5, total_silos)
        sample_silos = granaries_df.sample(n=sample_size, random_state=42)
        
        for _, silo_info in sample_silos.iterrows():
            try:
                heap_id = str(silo_info['store_id'])
                sub_table_id = int(silo_info['sub_table_id'])
                
                # Get date range for this silo
                silo_min_date, silo_max_date = self.get_silo_date_range(heap_id, sub_table_id)
                
                if silo_min_date is None or silo_max_date is None:
                    continue
                
                # Adjust date range to requested range
                actual_start = max(start_date, silo_min_date)
                actual_end = min(end_date, silo_max_date)
                
                if actual_start >= actual_end:
                    continue
                
                # Estimate records for this silo
                days_in_range = (actual_end - actual_start).days
                # Assume average 24 records per day per sensor point
                estimated_silo_records = days_in_range * 24 * 10  # 10 sensor points per silo
                estimated_records += estimated_silo_records
                silos_with_data += 1
                
            except Exception as e:
                logger.warning(f"Could not estimate size for silo {heap_id}: {e}")
                continue
        
        if silos_with_data == 0:
            return {
                'total_estimated_records': 0,
                'estimated_memory_mb': 0,
                'recommended_batch_size': self.min_chunk_size,
                'recommended_processing_mode': 'single_batch',
                'warning': 'No data found in date range'
            }
        
        # Extrapolate to all silos
        avg_records_per_silo = estimated_records / silos_with_data
        total_estimated_records = int(avg_records_per_silo * total_silos)
        
        # Estimate memory usage (rough estimate: 1KB per record)
        estimated_memory_mb = total_estimated_records * 1 / 1024
        
        # Get available system memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Calculate safe batch size
        safe_memory_mb = available_memory_mb * (self.memory_safety_threshold / 100)
        recommended_batch_size = min(
            self.max_records_per_batch,
            int(safe_memory_mb * 1024)  # Convert MB to records
        )
        
        # Ensure batch size is at least the minimum
        recommended_batch_size = max(recommended_batch_size, self.min_chunk_size)
        
        # Determine processing mode
        # Ensure both are ints
        try:
            recommended_batch_size = int(recommended_batch_size) if recommended_batch_size is not None else 10000
        except Exception:
            recommended_batch_size = 10000
        try:
            total_estimated_records = int(total_estimated_records) if total_estimated_records is not None else 0
        except Exception:
            total_estimated_records = 0
        if total_estimated_records <= recommended_batch_size:
            processing_mode = 'single_batch'
        elif total_estimated_records <= recommended_batch_size * 10:
            processing_mode = 'chunked'
        else:
            processing_mode = 'streaming'
        
        logger.info(f"Data size estimation completed:")
        logger.info(f"  - Total estimated records: {total_estimated_records:,}")
        logger.info(f"  - Estimated memory usage: {estimated_memory_mb:.1f} MB")
        logger.info(f"  - Available memory: {available_memory_mb:.1f} MB")
        logger.info(f"  - Recommended batch size: {recommended_batch_size:,}")
        logger.info(f"  - Recommended processing mode: {processing_mode}")
        
        return {
            'total_estimated_records': total_estimated_records,
            'estimated_memory_mb': estimated_memory_mb,
            'recommended_batch_size': recommended_batch_size,
            'recommended_processing_mode': processing_mode,
            'available_memory_mb': available_memory_mb,
            'safe_memory_mb': safe_memory_mb
        }
    
    def get_silo_date_range(self, heap_id: str, sub_table_id: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the date range for a specific silo using the provided query.
        
        Returns tuple of (min_date, max_date) or (None, None) if no data
        """
        query = f"""
        SELECT 
            goods.name,
            goods.fid,
            minBatch.min,
            maxBatch.max
        FROM cloud_server.base_store_location loc
        INNER JOIN cloud_server.base_storehouse house ON house.storepointuuid = loc.fid
        INNER JOIN cloud_server.base_goods_allocation goods ON house.fid = goods.storuuid
        LEFT JOIN (
            SELECT max(batch) max, goods_allocation_id 
            FROM cloud_lq.lq_point_history_{sub_table_id} 
            WHERE goods_allocation_id = '{heap_id}' 
            GROUP BY goods_allocation_id
        ) maxBatch ON maxBatch.goods_allocation_id = goods.fid
        LEFT JOIN (
            SELECT min(batch) min, goods_allocation_id 
            FROM cloud_lq.lq_point_history_{sub_table_id} 
            WHERE goods_allocation_id = '{heap_id}' 
            GROUP BY goods_allocation_id
        ) minBatch ON minBatch.goods_allocation_id = goods.fid
        WHERE goods.fid = '{heap_id}'
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty and df['min'].iloc[0] is not None and df['max'].iloc[0] is not None:
                min_date = pd.to_datetime(df['min'].iloc[0])
                max_date = pd.to_datetime(df['max'].iloc[0])
                return min_date, max_date
            return None, None
        except Exception as e:
            logger.warning(f"Failed to get date range for silo {heap_id}: {e}")
            return None, None
    
    def get_silo_data_chunked(self, heap_id: str, sub_table_id: int, start_date: datetime, end_date: datetime) -> Generator[pd.DataFrame, None, None]:
        """
        Get data for a specific silo in chunks to prevent memory issues.
        
        Yields DataFrame chunks for the specified date range.
        """
        # Calculate total date range
        total_days = (end_date - start_date).days
        
        if total_days <= 0:
            return
        
        # Process in smaller time chunks to avoid memory issues
        days_per_chunk = max(1, min(15, total_days // 20))  # Reduced from 30 to 15 days
        
        current_start = start_date
        chunk_count = 0
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=days_per_chunk), end_date)
            
            # Robust memory management - never skip chunks
            memory_recovered = True
            while not self.check_memory_usage():
                memory_percent = psutil.virtual_memory().percent
                logger.warning(f"Memory usage high ({memory_percent:.1f}%) before processing chunk {chunk_count + 1}, initiating recovery...")
                
                # Try progressive memory recovery strategies
                memory_recovered = self.wait_for_memory_recovery(max_wait_time=300)  # 5 minutes max
                
                if not memory_recovered:
                    logger.error(f"Memory recovery failed for chunk {chunk_count + 1}, but continuing anyway...")
                    # Force one final cleanup and continue
                    self.force_memory_cleanup(aggressive=True)
                    time.sleep(5)
                    break
                else:
                    logger.info(f"Memory recovered successfully for chunk {chunk_count + 1}")
            
            # Memory is now acceptable (or we're proceeding anyway), proceed with data retrieval
            
            try:
                query = f"""
                SELECT
                    loc.fid as storepointId,
                    loc.kdmc as storepointName,
                    loc.kdjd,
                    loc.kdwd,
                    loc.kqdz,
                    point.goods_allocation_id AS storeId,
                    allocation.name as storeName,
                    line.line_no,
                    b.layer_no,
                    point.batch,
                    SUBSTR(SUBSTRING_INDEX(point.temp, ',', b.layer_no), -5) as temp,
                    b.x_coordinate as x,
                    b.y_coordinate as y,
                    b.z_coordinate as z,
                    ROUND(a3.avg_in_temp, 2) as avg_in_temp,
                    ROUND(a3.max_temp, 2) as max_temp,
                    ROUND(a3.min_temp, 2) as min_temp,
                    ROUND(a3.indoor_temp, 2) as indoor_temp,
                    ROUND(a3.indoor_humidity, 2) as indoor_humidity,
                    ROUND(a3.outdoor_temp, 2) as outdoor_temp,
                    ROUND(a3.outdoor_humidity, 2) as outdoor_humidity
                FROM cloud_server.base_store_location loc
                INNER JOIN cloud_server.base_store_line line
                INNER JOIN cloud_server.base_store_point b ON line.fid = b.line_id
                LEFT JOIN cloud_lq.lq_point_history_{sub_table_id} point 
                    ON point.line_id = line.fid AND point.goods_allocation_id = '{heap_id}'
                INNER JOIN cloud_server.base_goods_allocation allocation 
                    ON allocation.fid = point.goods_allocation_id
                LEFT JOIN cloud_lq.lq_store_history_{sub_table_id} a3 
                    ON a3.store_id = line.goods_allocation_id 
                    AND a3.batch = point.batch 
                    AND a3.store_id = '{heap_id}'
                WHERE
                    b.is_ignore = '0'
                    AND line.is_active = '1'
                    AND point.batch BETWEEN '{current_start.strftime('%Y-%m-%d %H:%M:%S')}' 
                        AND '{current_end.strftime('%Y-%m-%d %H:%M:%S')}'
                ORDER BY storeId, batch, line.line_no, b.layer_no
                """
                
                df = pd.read_sql(query, self.engine)
                
                if not df.empty:
                    # Convert temperature string to numeric
                    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
                    # Convert batch to datetime
                    df['batch'] = pd.to_datetime(df['batch'])
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'storepointId': 'granary_id',
                        'storepointName': 'granary_name',
                        'storeId': 'heap_id',
                        'storeName': 'heap_name',
                        'batch': 'detection_time',
                        'temp': 'temperature_grain',
                        'x': 'grid_x',
                        'y': 'grid_y',
                        'z': 'grid_z'
                    })
                    
                    chunk_count += 1
                    logger.debug(f"Retrieved chunk {chunk_count} for silo {heap_id}: {len(df)} records")
                    yield df
                else:
                    logger.debug(f"No data in chunk {chunk_count + 1} for silo {heap_id}")
                
            except Exception as e:
                logger.error(f"Failed to retrieve chunk {chunk_count + 1} for silo {heap_id}: {e}")
                continue
            
            current_start = current_end
            
            # Small delay between chunks to prevent overwhelming the database
            time.sleep(0.1)
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Log memory status for debugging
            if memory_percent > self.force_gc_threshold:
                logger.warning(f"Memory usage critical: {memory_percent:.1f}% (threshold: {self.force_gc_threshold}%)")
                return False
            elif memory_percent > self.memory_safety_threshold:
                logger.info(f"Memory usage elevated: {memory_percent:.1f}% (threshold: {self.memory_safety_threshold}%)")
                return False
            else:
                return True
                
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return False  # Assume memory is low if we can't check
    
    def force_memory_cleanup(self, aggressive: bool = False) -> bool:
        """
        Force memory cleanup using multiple strategies.
        Never fails - always tries to free memory.
        
        Args:
            aggressive: If True, use more aggressive cleanup strategies
            
        Returns:
            True if cleanup was successful, False if memory is still critical
        """
        try:
            logger.info("Starting memory cleanup...")
            
            # Get initial memory state
            initial_memory = psutil.virtual_memory()
            initial_percent = initial_memory.percent
            
            # Strategy 1: Standard garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Strategy 2: Clear any cached data
            if hasattr(self, '_cached_data'):
                del self._cached_data
                logger.info("Cleared cached data")
            
            # Strategy 3: Clear pandas cache
            try:
                import pandas as pd
                pd.util.hash_pandas_object.cache_clear()
                logger.info("Cleared pandas cache")
            except:
                pass
            
            # Strategy 4: Clear any other caches
            try:
                import functools
                functools.lru_cache.cache_clear()
                logger.info("Cleared function caches")
            except:
                pass
            
            # Strategy 5: Force Python to release memory to OS
            if aggressive:
                try:
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                    logger.info("Forced memory release to OS")
                except:
                    pass
            
            # Strategy 6: Wait for memory to stabilize
            time.sleep(2)
            
            # Check final memory state
            final_memory = psutil.virtual_memory()
            final_percent = final_memory.percent
            freed_mb = (initial_memory.available - final_memory.available) / (1024 * 1024)
            
            logger.info(f"Memory cleanup completed: {initial_percent:.1f}% -> {final_percent:.1f}% (freed {freed_mb:.1f} MB)")
            
            # Return True if memory is now acceptable, False if still critical
            return final_percent < self.memory_threshold
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return False
    
    def wait_for_memory_recovery(self, max_wait_time: int = 300) -> bool:
        """
        Wait for memory to recover naturally, with progressive strategies.
        Never gives up - always waits for memory to become available.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if memory recovered, False if still critical after max time
        """
        start_time = time.time()
        wait_strategies = [
            (5, "Short pause"),
            (15, "Medium pause with cleanup"),
            (30, "Long pause with aggressive cleanup"),
            (60, "Extended pause with system cleanup")
        ]
        
        for pause_time, strategy_name in wait_strategies:
            if time.time() - start_time > max_wait_time:
                logger.error(f"Memory recovery timeout after {max_wait_time}s")
                return False
            
            logger.info(f"Memory recovery strategy: {strategy_name} ({pause_time}s)")
            
            # Apply appropriate cleanup based on strategy
            if pause_time > 10:
                self.force_memory_cleanup(aggressive=(pause_time > 30))
            
            time.sleep(pause_time)
            
            # Check if memory has recovered
            if self.check_memory_usage():
                elapsed = time.time() - start_time
                logger.info(f"Memory recovered after {elapsed:.1f}s using {strategy_name}")
                return True
        
        # If we get here, try one final aggressive cleanup
        logger.warning("Final aggressive memory cleanup attempt...")
        self.force_memory_cleanup(aggressive=True)
        time.sleep(10)
        
        return self.check_memory_usage()
    
    def collect_data_in_batches(self, granaries_df: pd.DataFrame, start_date: datetime, end_date: datetime, 
                               granary_filter: Optional[str] = None, batch_size: Optional[int] = None,
                               missing_ranges: Optional[Dict] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Collect data from the database in manageable batches to prevent memory crashes.
        Only retrieves data for missing date ranges if missing_ranges is provided.
        
        Yields DataFrame batches that can be processed safely.
        """
        logger.info("Starting batched data collection...")
        
        if batch_size is None:
            batch_size = 10000
        batch_size = int(batch_size)
        
        # Filter granaries_df to only include silos with missing data
        if missing_ranges:
            silos_to_process = list(missing_ranges.keys())
            granaries_df = granaries_df[granaries_df['store_id'].isin(silos_to_process)].copy()
            logger.info(f"Filtered to {len(granaries_df)} silos with missing data (out of {len(silos_to_process)} total)")
        
        total_silos = len(granaries_df)
        processed_silos = 0
        skipped_silos = 0
        current_batch = []
        current_batch_size = 0
        
        # Progress tracking
        pbar = tqdm(total=total_silos, desc="Processing silos", unit="silo")
        
        for silo_idx, (_, silo_info) in enumerate(granaries_df.iterrows()):
            try:
                heap_id = str(silo_info['store_id'])
                heap_name = str(silo_info['silo_name'])
                sub_table_id = int(silo_info['sub_table_id'])
                
                logger.info(f"Processing silo {silo_idx + 1}/{total_silos}: {heap_name} (ID: {heap_id})")
                
                # Robust memory management before processing each silo
                while not self.check_memory_usage():
                    logger.warning("Memory usage high before processing silo, initiating recovery...")
                    
                    # Try progressive memory recovery
                    memory_recovered = self.wait_for_memory_recovery(max_wait_time=180)  # 3 minutes max
                    
                    if not memory_recovered:
                        logger.error("Memory recovery failed, but continuing anyway...")
                        # Force one final cleanup and continue
                        self.force_memory_cleanup(aggressive=True)
                        time.sleep(5)
                        break
                    else:
                        logger.info("Memory recovered successfully for silo processing")
                
                # If we have a current batch and memory is still high, yield it early
                if current_batch and not self.check_memory_usage():
                    logger.warning("Memory still high, yielding current batch early...")
                    combined_batch = pd.concat(current_batch, ignore_index=True)
                    current_batch = []
                    current_batch_size = 0
                    yield combined_batch
                
                # Get missing date ranges for this silo
                if missing_ranges and heap_id in missing_ranges:
                    silo_missing_ranges = missing_ranges[heap_id]['missing_ranges']
                    logger.info(f"Silo {heap_id} has {len(silo_missing_ranges)} missing date ranges")
                    
                    # Get silo data for each missing range
                    silo_data_chunks = []
                    for range_info in silo_missing_ranges:
                        range_start = range_info['start']
                        range_end = range_info['end']
                        reason = range_info['reason']
                        
                        logger.info(f"Retrieving data for {heap_id}: {range_start} to {range_end} (reason: {reason})")
                        
                        for chunk in self.get_silo_data_chunked(heap_id, sub_table_id, range_start, range_end):
                            if not chunk.empty:
                                # Add granary information to the chunk
                                chunk['granary_id'] = silo_info['storepoint_id']
                                chunk['granary_name'] = silo_info['granary_name']
                                silo_data_chunks.append(chunk)
                else:
                    # Fallback to original behavior if no missing_ranges provided
                    silo_min_date, silo_max_date = self.get_silo_date_range(heap_id, sub_table_id)
                    
                    if silo_min_date is None or silo_max_date is None:
                        logger.warning(f"No data available for silo {heap_id}")
                        pbar.update(1)
                        continue
                    
                    # Use the silo's actual data range instead of constraining to requested range
                    actual_start = silo_min_date
                    actual_end = silo_max_date
                    
                    logger.info(f"Silo {heap_id} data range: {actual_start} to {actual_end}")
                    
                    # Get silo data in chunks
                    silo_data_chunks = []
                    for chunk in self.get_silo_data_chunked(heap_id, sub_table_id, actual_start, actual_end):
                        if not chunk.empty:
                            # Add granary information to the chunk
                            chunk['granary_id'] = silo_info['storepoint_id']
                            chunk['granary_name'] = silo_info['granary_name']
                            silo_data_chunks.append(chunk)
                
                if silo_data_chunks:
                    # Combine all chunks for this silo
                    silo_data = pd.concat(silo_data_chunks, ignore_index=True)
                    logger.info(f"Successfully collected {len(silo_data)} records for silo {heap_id}")
                    
                    # Add to current batch
                    current_batch.append(silo_data)
                    current_batch_size += len(silo_data)
                    
                    # Clear silo data from memory immediately
                    del silo_data
                    del silo_data_chunks
                    gc.collect()
                    
                    # Check if batch is full
                    if int(current_batch_size) >= int(batch_size):
                        logger.info(f"Batch full ({current_batch_size} records), yielding...")
                        combined_batch = pd.concat(current_batch, ignore_index=True)
                        current_batch = []
                        current_batch_size = 0
                        yield combined_batch
                        
                        # Force garbage collection after yielding
                        gc.collect()
                else:
                    logger.warning(f"No data retrieved for silo {heap_id}")
                    skipped_silos += 1
                    
                processed_silos += 1
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error processing silo {silo_idx + 1}: {e}")
                pbar.update(1)
                continue
        
        pbar.close()
        
        # Yield any remaining data in the final batch
        if current_batch:
            logger.info(f"Yielding final batch with {current_batch_size} records...")
            combined_batch = pd.concat(current_batch, ignore_index=True)
            yield combined_batch
        
        logger.info(f"Batched data collection completed. Processed {processed_silos}/{total_silos} silos.")
    
    def _stream_large_dataset(self, granaries_df: pd.DataFrame, start_date: datetime, end_date: datetime, granary_filter: Optional[str] = None, run_pipeline: bool = True, size_estimate: Optional[Dict] = None, missing_ranges: Optional[Dict] = None) -> Dict:
        """
        Stream large datasets by processing batches directly through the ingestion pipeline.
        This prevents loading all data into memory at once.
        """
        logger.info("Starting streaming mode for large dataset...")
        
        if size_estimate is None:
            size_estimate = self.estimate_data_size(granaries_df, start_date, end_date)
        
        batch_size = size_estimate['recommended_batch_size']
        total_estimated_records = size_estimate['total_estimated_records']
        
        logger.info(f"Streaming mode: Processing {total_estimated_records:,} estimated records in batches of {batch_size:,}")
        
        # Import the ingestion module
        import sys
        script_dir = Path(__file__).resolve().parent
        root_dir = script_dir.parent.parent.parent
        
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        
        from granarypredict import ingestion
        
        # Process batches one by one
        batch_count = 0
        total_records_processed = 0
        granary_status = {}
        silo_changes = {}
        
        try:
            for batch_df in self.collect_data_in_batches(granaries_df, start_date, end_date, granary_filter, batch_size, missing_ranges):
                batch_count += 1
                total_records_processed += len(batch_df)
                
                logger.info(f"Processing batch {batch_count}: {len(batch_df)} records (Total: {total_records_processed:,})")
                
                # Process this batch through ingestion
                if hasattr(ingestion, 'ingest_and_sort_dataframe'):
                    batch_result = ingestion.ingest_and_sort_dataframe(
                        batch_df, 
                        return_new_data_status=True, 
                        output_dir=self.granaries_dir
                    )
                else:
                    # Fallback to temp CSV method
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
                        temp_csv_path = temp_file.name
                        batch_df.to_csv(temp_csv_path, index=False, encoding='utf-8')
                    
                    batch_result = ingestion.ingest_and_sort(temp_csv_path, return_new_data_status=True)
                    os.unlink(temp_csv_path)
                
                # Merge results
                if isinstance(batch_result, dict) and 'granary_status' in batch_result:
                    for granary, is_new in batch_result['granary_status'].items():
                        if granary not in granary_status:
                            granary_status[granary] = False
                        granary_status[granary] = granary_status[granary] or is_new
                    
                    for granary, changes in batch_result['silo_changes'].items():
                        if granary not in silo_changes:
                            silo_changes[granary] = []
                        silo_changes[granary].extend(changes)
                        # Remove duplicates
                        silo_changes[granary] = list(set(silo_changes[granary]))
                
                # Clear batch from memory
                del batch_df
                gc.collect()
                
                # Robust memory management during streaming
                if not self.check_memory_usage():
                    logger.warning("Memory usage high during streaming, initiating recovery...")
                    memory_recovered = self.wait_for_memory_recovery(max_wait_time=120)  # 2 minutes max
                    
                    if not memory_recovered:
                        logger.error("Memory recovery failed during streaming, but continuing...")
                        self.force_memory_cleanup(aggressive=True)
                        time.sleep(5)
                    else:
                        logger.info("Memory recovered successfully during streaming")
            
            # Run pipeline processing if requested
            pipeline_results = {}
            if run_pipeline:
                new_granaries = [g for g, is_new in granary_status.items() if is_new]
                if new_granaries:
                    logger.info(f"Running pipeline processing for {len(new_granaries)} granaries...")
                    
                    from service.granary_pipeline import run_complete_pipeline
                    
                    for granary_name in new_granaries:
                        try:
                            granary_csv_path = self.granaries_dir / f"{granary_name}.parquet"
                            if not granary_csv_path.exists():
                                granary_csv_path = self.granaries_dir / f"{granary_name}.csv"
                            
                            if granary_csv_path.exists():
                                changed_silos = silo_changes.get(granary_name, [])
                                result = run_complete_pipeline(
                                    granary_csv=str(granary_csv_path),
                                    granary_name=granary_name,
                                    skip_train=False,
                                    force_retrain=False,
                                    changed_silos=changed_silos
                                )
                                pipeline_results[granary_name] = result
                            else:
                                pipeline_results[granary_name] = {'success': False, 'error': 'Granary file not found'}
                        except Exception as e:
                            pipeline_results[granary_name] = {'success': False, 'error': str(e)}
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'granary_filter': granary_filter,
                'total_records_processed': total_records_processed,
                'batches_processed': batch_count,
                'granaries_with_new_data': [g for g, is_new in granary_status.items() if is_new],
                'silo_changes': silo_changes,
                'pipeline_results': pipeline_results,
                'processing_mode': 'streaming',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Streaming mode failed: {e}")
            # Cleanup on error
            try:
                gc.collect()
                if hasattr(self, '_temp_files'):
                    for temp_file in self._temp_files:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
            
            return {
                'success': False, 
                'error': str(e), 
                'processing_mode': 'streaming',
                'total_records_processed': total_records_processed,
                'batches_processed': batch_count
            }
    
    def stream_all_data(self, start_date: datetime, end_date: datetime, run_pipeline: bool = True, granary_filter: Optional[str] = None) -> Dict:
        """
        Stream data for all granaries and silos (or filtered by granary).
        
        Note: This method now collects ALL available data for each silo using their individual
        data ranges, rather than constraining to the specified date range. The start_date and
        end_date parameters are kept for backward compatibility but are not used for data filtering.
        
        Args:
            start_date: Start date parameter (kept for compatibility, not used for filtering)
            end_date: End date parameter (kept for compatibility, not used for filtering)
            run_pipeline: Whether to run the processing pipeline after data collection
            granary_filter: Optional granary name or ID to filter by (e.g., "蚬冈库" or granary ID)
        
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting data streaming for ALL available data from each silo")
        if granary_filter:
            logger.info(f"Filtering by granary: {granary_filter}")
        logger.info("Note: Each silo will use its own actual data range (earliest to latest available data)")
        
        try:
            # Get all granaries and silos
            logger.info("Step 1: Retrieving all granaries and silos...")
            granaries_df = self.get_all_granaries_and_silos()
            logger.info(f"Step 1 completed: Found {len(granaries_df)} silos across {granaries_df['sub_table_id'].nunique()} granaries")
            
            if granaries_df.empty:
                logger.error("No granaries found in database")
                return {'success': False, 'error': 'No granaries found'}
            
            # Filter by granary if specified
            if granary_filter:
                logger.info(f"Step 2: Filtering by granary: {granary_filter}")
                try:
                    # Ensure columns are strings and handle potential NaN values
                    granary_name_col = granaries_df['granary_name'].fillna('').astype(str)
                    store_id_col = granaries_df['storepoint_id'].fillna('').astype(str)
                    
                    logger.info(f"Step 2a: Filtering {len(granaries_df)} silos by granary filter")
                    logger.info(f"Step 2b: Looking for exact ID match or name contains match")
                    
                    # Check if the filter looks like a granary ID (32 character hex string)
                    import re
                    # Strip quotes from the filter if present
                    clean_filter = granary_filter.strip("'\"")
                    logger.info(f"Step 2b1: Original filter: '{granary_filter}', Clean filter: '{clean_filter}'")
                    
                    is_id_filter = bool(re.match(r'^[a-fA-F0-9]{32}$', clean_filter))
                    logger.info(f"Step 2b2: Is ID filter: {is_id_filter}")
                    
                    if is_id_filter:
                        # For ID filters, use exact match
                        logger.info(f"Step 2c: Using exact ID match for: {clean_filter}")
                        filtered_df = granaries_df[store_id_col == clean_filter]
                    else:
                        # For name filters, use contains match
                        logger.info(f"Step 2c: Using name contains match for: {clean_filter}")
                        filtered_df = granaries_df[granary_name_col.str.contains(clean_filter, case=False, na=False)]
                    
                    logger.info(f"Step 2a completed: Found {len(filtered_df)} matching silos")
                    
                    if filtered_df.empty:
                        logger.error(f"No granaries found matching filter: {granary_filter}")
                        logger.info("Available granaries:")
                        # Ensure granaries_df is flat before grouping
                        granaries_df = granaries_df.reset_index(drop=True)
                        unique_granaries = granaries_df.groupby(['storepoint_id', 'granary_name'], as_index=False).first().reset_index(drop=True)
                        for _, row in unique_granaries.iterrows():
                            logger.info(f"  - {row['granary_name']} (ID: {row['storepoint_id']})")
                        return {'success': False, 'error': f'No granaries found matching: {granary_filter}'}
                    
                    granaries_df = filtered_df.reset_index(drop=True)
                    logger.info(f"Step 2 completed: Found {len(granaries_df)} silos in filtered granary(ies)")
                except Exception as e:
                    logger.error(f"Step 2 failed: Error filtering granaries: {e}")
                    logger.error(f"Granaries DataFrame shape: {granaries_df.shape}")
                    logger.error(f"Granaries DataFrame columns: {list(granaries_df.columns)}")
                    logger.error(f"Granaries DataFrame head:\n{granaries_df.head()}")
                    raise
            
            # Step 3: Check existing data and determine what needs to be retrieved
            logger.info("Step 3: Checking existing data and determining missing ranges...")
            # Ensure granaries_df is properly typed as DataFrame
            granaries_df = pd.DataFrame(granaries_df).reset_index(drop=True)
            
            # Check what data already exists
            existing_data = self.check_existing_data(granaries_df)
            
            # Determine what date ranges are missing
            missing_ranges = self.get_missing_date_ranges(granaries_df, existing_data, start_date, end_date)
            
            # If no missing data, return early
            if not missing_ranges:
                logger.info("No missing data found - all requested data is already available")
                return {
                    'success': True,
                    'message': 'All data already available',
                    'total_granaries': len(granaries_df['sub_table_id'].unique()),
                    'processed_granaries': 0,
                    'failed_granaries': 0,
                    'total_records': 0,
                    'pipeline_success': True,
                    'granary_results': [],
                    'existing_data_summary': existing_data,
                    'missing_ranges': missing_ranges
                }
            
            # Step 4: Collect missing data from database
            logger.info("Step 4: Collecting missing data from database...")
            
            # Estimate data size for missing ranges only
            size_estimate = self.estimate_data_size(granaries_df, start_date, end_date)
            logger.info(f"Estimated data size for missing ranges: {size_estimate['total_estimated_records']:,}, Recommended batch size: {size_estimate['recommended_batch_size']:,}")
            
            # Check if we should use streaming mode for very large datasets
            if size_estimate['recommended_processing_mode'] == 'streaming':
                logger.info("Using streaming mode for large dataset...")
                return self._stream_large_dataset(granaries_df, start_date, end_date, granary_filter, run_pipeline, size_estimate, missing_ranges)
            
            # Collect data in batches for smaller datasets (only missing ranges)
            raw_data_batches = list(self.collect_data_in_batches(granaries_df, start_date, end_date, granary_filter, size_estimate['recommended_batch_size'], missing_ranges))
            
            if not raw_data_batches:
                logger.error("No raw data collected from database")
                return {'success': False, 'error': 'No data found in specified date range'}
            
            # Combine all batches into a single DataFrame
            logger.info(f"Combining {len(raw_data_batches)} batches...")
            raw_data = pd.concat(raw_data_batches, ignore_index=True)
            logger.info(f"Combined data shape: {raw_data.shape}")
            
            # Clear batch data from memory
            del raw_data_batches
            gc.collect()
            
            if raw_data.empty:
                logger.error("No raw data collected from database")
                return {'success': False, 'error': 'No data found in specified date range'}
            
            logger.info(f"Step 4 completed: Collected {len(raw_data)} total records")
            
            # Step 5: Process raw data directly through ingestion pipeline (optimized - no temp CSV)
            logger.info("Step 5: Processing raw data directly through ingestion pipeline...")
            
            try:
                # Import the ingestion module
                import sys
                # Add the root directory to sys.path so we can import granarypredict
                # The script is in service/scripts/data_retrieval/, so we need to go up 3 levels
                script_dir = Path(__file__).resolve().parent
                root_dir = script_dir.parent.parent.parent
                logger.info(f"Script directory: {script_dir}")
                logger.info(f"Root directory: {root_dir}")
                logger.info(f"Current working directory: {Path.cwd()}")
                
                if str(root_dir) not in sys.path:
                    sys.path.insert(0, str(root_dir))
                    logger.info(f"Added {root_dir} to sys.path")
                
                logger.info(f"sys.path: {sys.path}")
                
                from granarypredict import ingestion
                logger.info(f"Successfully imported granarypredict.ingestion from: {ingestion.__file__}")
                logger.info(f"Available attributes in ingestion: {[attr for attr in dir(ingestion) if not attr.startswith('_')]}")
                
                # Use the optimized ingest_and_sort_dataframe function (no temp CSV needed)
                logger.info("Step 4a: Running ingest_and_sort_dataframe on raw data...")
                if hasattr(ingestion, 'ingest_and_sort_dataframe'):
                    logger.info("SUCCESS: ingest_and_sort_dataframe function found!")
                    change_info = ingestion.ingest_and_sort_dataframe(raw_data, return_new_data_status=True, output_dir=self.granaries_dir)
                elif hasattr(ingestion, 'ingest_and_sort'):
                    # Fallback to original method if new function not available
                    logger.warning("ingest_and_sort_dataframe not found, falling back to temp CSV method")
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
                        temp_csv_path = temp_file.name
                        raw_data.to_csv(temp_csv_path, index=False, encoding='utf-8')
                        logger.info(f"Saved raw data to temporary CSV: {temp_csv_path}")
                    
                    change_info = ingestion.ingest_and_sort(temp_csv_path, return_new_data_status=True)
                    
                    # Clean up temporary file
                    os.unlink(temp_csv_path)
                    logger.info(f"Cleaned up temporary file: {temp_csv_path}")
                else:
                    raise AttributeError(f"Neither ingest_and_sort_dataframe nor ingest_and_sort function found in ingestion module. Available functions: {[attr for attr in dir(ingestion) if callable(getattr(ingestion, attr)) and not attr.startswith('_')]}")
                
                # Handle both old and new return formats for backward compatibility
                if isinstance(change_info, dict) and 'granary_status' in change_info:
                    # New format with silo tracking
                    granary_status = change_info['granary_status']
                    silo_changes = change_info['silo_changes']
                else:
                    # Old format - fallback
                    granary_status = change_info if isinstance(change_info, dict) else {}
                    silo_changes = {}
                
                new_granaries = [g for g, is_new in granary_status.items() if is_new]
                
                logger.info(f"Step 4a completed: Ingestion pipeline processed {len(new_granaries)} granaries with new data")
                for granary in new_granaries:
                    logger.info(f"  {granary}: Changed silos: {silo_changes.get(granary, [])}")
                
                # Step 5: Run pipeline processing if requested
                if run_pipeline and new_granaries:
                    logger.info("Step 5: Running pipeline processing on granaries with new data...")
                    
                    # Import the pipeline module
                    from service.granary_pipeline import run_complete_pipeline
                    
                    pipeline_results = {}
                    for granary_name in new_granaries:
                        try:
                            logger.info(f"Step 5a: Running pipeline for granary: {granary_name}")
                            
                            # Find the granary CSV file created by ingest_and_sort
                            granary_csv_path = self.granaries_dir / f"{granary_name}.parquet"
                            if not granary_csv_path.exists():
                                # Try CSV format as fallback
                                granary_csv_path = self.granaries_dir / f"{granary_name}.csv"
                            
                            if granary_csv_path.exists():
                                # Run the complete pipeline for this granary
                                changed_silos = silo_changes.get(granary_name, [])
                                result = run_complete_pipeline(
                                    granary_csv=str(granary_csv_path),
                                    granary_name=granary_name,
                                    skip_train=False,  # Always train to ensure we have models
                                    force_retrain=False,
                                    changed_silos=changed_silos
                                )
                                pipeline_results[granary_name] = result
                                logger.info(f"Step 5a completed: Pipeline result for {granary_name}: {result['success']}")
                            else:
                                logger.error(f"Step 5a failed: Granary file not found for {granary_name}")
                                pipeline_results[granary_name] = {'success': False, 'error': 'Granary file not found'}
                                
                        except Exception as e:
                            logger.error(f"Step 5a failed: Error running pipeline for {granary_name}: {e}")
                            pipeline_results[granary_name] = {'success': False, 'error': str(e)}
                    
                    logger.info("Step 5 completed: Pipeline processing finished")
                
                # Prepare results
                results = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'granary_filter': granary_filter,
                    'total_records_collected': len(raw_data),
                    'granaries_with_new_data': new_granaries,
                    'silo_changes': silo_changes,
                    'pipeline_results': pipeline_results if run_pipeline else {},
                    'existing_data_summary': existing_data,
                    'missing_ranges': missing_ranges,
                    'success': True
                }
                
                logger.info("Data streaming completed successfully:")
                logger.info(f"  - Total records collected: {len(raw_data)}")
                logger.info(f"  - Granaries with new data: {len(new_granaries)}")
                logger.info(f"  - Pipeline processing: {'Completed' if run_pipeline else 'Skipped'}")
                
                return results
                
            except Exception as e:
                logger.error(f"Step 4 failed: Error in ingestion pipeline: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Data streaming failed with exception: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {'success': False, 'error': str(e), 'exception_type': type(e).__name__}
    
    def check_existing_data(self, granaries_df: pd.DataFrame) -> Dict:
        """
        Check what data already exists for each granary and silo.
        
        Args:
            granaries_df: DataFrame with granary and silo information (should be pre-filtered)
            
        Returns:
            Dictionary with existing data information for each granary
        """
        logger.info(f"Checking existing data for {len(granaries_df)} silos across {granaries_df['sub_table_id'].nunique()} granaries...")
        
        existing_data = {}
        
        for _, silo_info in granaries_df.iterrows():
            granary_name = silo_info['granary_name']
            heap_id = silo_info['store_id']  # Use store_id instead of heap_id
            
            if granary_name not in existing_data:
                existing_data[granary_name] = {
                    'granary_file': None,
                    'existing_silos': {},
                    'total_records': 0,
                    'date_range': {'min': None, 'max': None}
                }
            
            # Check if granary file exists
            granary_file = self.granaries_dir / f"{granary_name}.parquet"
            if granary_file.exists():
                existing_data[granary_name]['granary_file'] = str(granary_file)
                
                try:
                    # Read existing granary data to check silo information
                    df_existing = pd.read_parquet(granary_file)
                    
                    # Get data for this specific silo
                    silo_data = df_existing[df_existing['heap_id'] == heap_id]
                    
                    if not silo_data.empty:
                        # Convert detection_time to datetime if it's not already
                        if 'detection_time' in silo_data.columns:
                            silo_data['detection_time'] = pd.to_datetime(silo_data['detection_time'])
                            
                            silo_min_date = silo_data['detection_time'].min()
                            silo_max_date = silo_data['detection_time'].max()
                            silo_records = len(silo_data)
                            
                            existing_data[granary_name]['existing_silos'][heap_id] = {
                                'min_date': silo_min_date,
                                'max_date': silo_max_date,
                                'records': silo_records,
                                'date_range_days': (silo_max_date - silo_min_date).days
                            }
                            
                            # Update granary-level statistics
                            existing_data[granary_name]['total_records'] += silo_records
                            
                            # Update granary date range
                            if existing_data[granary_name]['date_range']['min'] is None:
                                existing_data[granary_name]['date_range']['min'] = silo_min_date
                            else:
                                existing_data[granary_name]['date_range']['min'] = min(
                                    existing_data[granary_name]['date_range']['min'], 
                                    silo_min_date
                                )
                            
                            if existing_data[granary_name]['date_range']['max'] is None:
                                existing_data[granary_name]['date_range']['max'] = silo_max_date
                            else:
                                existing_data[granary_name]['date_range']['max'] = max(
                                    existing_data[granary_name]['date_range']['max'], 
                                    silo_max_date
                                )
                            
                            logger.debug(f"Found existing data for {granary_name}/{heap_id}: "
                                        f"{silo_records:,} records from {silo_min_date} to {silo_max_date}")
                        else:
                            logger.warning(f"No detection_time column found in existing data for {granary_name}/{heap_id}")
                    else:
                        logger.debug(f"No existing data found for silo {heap_id} in granary {granary_name}")
                        
                except Exception as e:
                    logger.warning(f"Could not read existing data for {granary_name}: {e}")
                    existing_data[granary_name]['existing_silos'][heap_id] = {
                        'min_date': None,
                        'max_date': None,
                        'records': 0,
                        'date_range_days': 0,
                        'error': str(e)
                    }
            else:
                logger.debug(f"No existing file found for granary {granary_name}")
        
        # Log summary
        total_existing_granaries = sum(1 for g in existing_data.values() if g['granary_file'])
        total_existing_silos = sum(len(g['existing_silos']) for g in existing_data.values())
        total_existing_records = sum(g['total_records'] for g in existing_data.values())
        
        logger.info(f"Existing data summary:")
        logger.info(f"  Granaries with data: {total_existing_granaries}")
        logger.info(f"  Silos with data: {total_existing_silos}")
        logger.info(f"  Total existing records: {total_existing_records:,}")
        
        return existing_data
    
    def get_missing_date_ranges(self, granaries_df: pd.DataFrame, existing_data: Dict, 
                               start_date: datetime, end_date: datetime) -> Dict:
        """
        Determine what date ranges are missing for each silo.
        
        Args:
            granaries_df: DataFrame with granary and silo information (should be pre-filtered)
            existing_data: Dictionary with existing data information
            start_date: Requested start date
            end_date: Requested end date
            
        Returns:
            Dictionary with missing date ranges for each silo
        """
        logger.info(f"Determining missing date ranges for {len(granaries_df)} silos...")
        
        missing_ranges = {}
        skipped_silos = 0
        total_silos = len(granaries_df)
        
        for _, silo_info in granaries_df.iterrows():
            granary_name = silo_info['granary_name']
            heap_id = silo_info['store_id']  # Use store_id instead of heap_id
            
            # Get the silo's actual data range from database
            sub_table_id = silo_info['sub_table_id']
            silo_min_date, silo_max_date = self.get_silo_date_range(heap_id, sub_table_id)
            
            if silo_min_date is None or silo_max_date is None:
                logger.warning(f"No data available in database for silo {heap_id}")
                continue
            
            # Check what we already have for this silo
            existing_silo_data = existing_data.get(granary_name, {}).get('existing_silos', {}).get(heap_id, {})
            
            if existing_silo_data and existing_silo_data.get('records', 0) > 0:
                # We have existing data - check what's missing
                existing_min = existing_silo_data['min_date']
                existing_max = existing_silo_data['max_date']
                
                missing_ranges_list = []
                
                # Check for missing data before existing range
                if existing_min > silo_min_date:
                    missing_ranges_list.append({
                        'start': silo_min_date,
                        'end': existing_min - timedelta(days=1),
                        'reason': 'data_before_existing'
                    })
                
                # Check for missing data after existing range
                if existing_max < silo_max_date:
                    missing_ranges_list.append({
                        'start': existing_max + timedelta(days=1),
                        'end': silo_max_date,
                        'reason': 'data_after_existing'
                    })
                
                if missing_ranges_list:
                    missing_ranges[heap_id] = {
                        'granary_name': granary_name,
                        'silo_min_date': silo_min_date,
                        'silo_max_date': silo_max_date,
                        'existing_min_date': existing_min,
                        'existing_max_date': existing_max,
                        'existing_records': existing_silo_data['records'],
                        'missing_ranges': missing_ranges_list,
                        'total_missing_days': sum((r['end'] - r['start']).days + 1 for r in missing_ranges_list)
                    }
                    logger.info(f"Silo {heap_id} ({granary_name}): Missing {len(missing_ranges_list)} date ranges")
                else:
                    # All data is already available
                    skipped_silos += 1
                    logger.info(f"Silo {heap_id} ({granary_name}): All data already available, skipping")
            else:
                # No existing data - need to retrieve everything
                missing_ranges[heap_id] = {
                    'granary_name': granary_name,
                    'silo_min_date': silo_min_date,
                    'silo_max_date': silo_max_date,
                    'existing_min_date': None,
                    'existing_max_date': None,
                    'existing_records': 0,
                    'missing_ranges': [{
                        'start': silo_min_date,
                        'end': silo_max_date,
                        'reason': 'no_existing_data'
                    }],
                    'total_missing_days': (silo_max_date - silo_min_date).days + 1
                }
                logger.info(f"Silo {heap_id} ({granary_name}): No existing data, will retrieve all")
        
        # Log summary
        total_missing_silos = len(missing_ranges)
        total_missing_days = sum(info['total_missing_days'] for info in missing_ranges.values())
        
        logger.info(f"Missing data summary:")
        logger.info(f"  Silos to retrieve: {total_missing_silos}")
        logger.info(f"  Silos to skip: {skipped_silos}")
        logger.info(f"  Total missing days: {total_missing_days:,}")
        
        if skipped_silos > 0:
            logger.info(f"  Skipping {skipped_silos}/{total_silos} silos (all data already available)")
        
        return missing_ranges
    
    def _get_earliest_available_date(self, granary_filter: Optional[str] = None) -> Optional[datetime]:
        """Get the earliest available date from the database, optionally filtered by granary."""
        try:
            # Get granaries info first
            granaries_df = self.get_all_granaries_and_silos()
            
            if granary_filter:
                # Filter by granary if specified
                # Ensure columns are strings and handle potential NaN values
                granary_name_col = granaries_df['granary_name'].fillna('').astype(str)
                store_id_col = granaries_df['storepoint_id'].fillna('').astype(str)
                
                filtered_df = granaries_df[
                    (granary_name_col.str.contains(granary_filter, case=False, na=False)) |
                    (store_id_col.str.contains(granary_filter, case=False, na=False))
                ]
                if not filtered_df.empty:
                    granaries_df = filtered_df
            
            if granaries_df.empty:
                return None
            
            # Get unique table IDs from filtered granaries
            table_ids = list(pd.Series(granaries_df['sub_table_id']).unique())
            
            # Build query for filtered tables only
            union_queries = []
            for table_id in table_ids:
                union_queries.append(f"SELECT MIN(batch) as earliest_date FROM cloud_lq.lq_point_history_{table_id}")
            
            if not union_queries:
                return None
            
            query = " UNION ALL ".join(union_queries)
            query = f"""
            SELECT MIN(earliest_date) as earliest
            FROM ({query}) as all_dates
            WHERE earliest_date IS NOT NULL
            """
            
            df = pd.read_sql(query, self.engine)
            if not df.empty and df['earliest'].iloc[0] is not None:
                earliest_date = pd.to_datetime(df['earliest'].iloc[0])
                logger.info(f"Earliest available date: {earliest_date}")
                return earliest_date
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not determine earliest available date: {e}")
            return None
    
    def _get_latest_available_date(self, granary_filter: Optional[str] = None) -> Optional[datetime]:
        """Get the latest available date from the database, optionally filtered by granary."""
        try:
            # Get granaries info first
            granaries_df = self.get_all_granaries_and_silos()
            
            if granary_filter:
                # Filter by granary if specified
                # Ensure columns are strings and handle potential NaN values
                granary_name_col = granaries_df['granary_name'].fillna('').astype(str)
                store_id_col = granaries_df['storepoint_id'].fillna('').astype(str)
                
                filtered_df = granaries_df[
                    (granary_name_col.str.contains(granary_filter, case=False, na=False)) |
                    (store_id_col.str.contains(granary_filter, case=False, na=False))
                ]
                if not filtered_df.empty:
                    granaries_df = filtered_df
            
            if granaries_df.empty:
                return None
            
            # Get unique table IDs from filtered granaries
            table_ids = list(pd.Series(granaries_df['sub_table_id']).unique())
            
            # Build query for filtered tables only
            union_queries = []
            for table_id in table_ids:
                union_queries.append(f"SELECT MAX(batch) as latest_date FROM cloud_lq.lq_point_history_{table_id}")
            
            if not union_queries:
                return None
            
            query = " UNION ALL ".join(union_queries)
            query = f"""
            SELECT MAX(latest_date) as latest
            FROM ({query}) as all_dates
            WHERE latest_date IS NOT NULL
            """
            
            df = pd.read_sql(query, self.engine)
            if not df.empty and df['latest'].iloc[0] is not None:
                latest_date = pd.to_datetime(df['latest'].iloc[0])
                logger.info(f"Latest available date: {latest_date}")
                return latest_date
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not determine latest available date: {e}")
            return None
    
    def parallel_granary_data_retrieval(self, start_date, end_date, granary_filter=None, chunk_days=7, max_workers=4):
        """
        Efficiently retrieve and store data for all granaries in parallel, chunked by date, writing directly to Parquet per granary.
        """
        import concurrent.futures
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm
        granaries_df = self.get_all_granaries_and_silos()
        if granary_filter:
            granaries_df = granaries_df[granaries_df['granary_name'].str.contains(granary_filter, case=False, na=False) |
                                        granaries_df['storepoint_id'].astype(str) == str(granary_filter)]
        granary_groups = granaries_df.groupby(['storepoint_id', 'granary_name'])
        def process_granary(granary_tuple):
            granary_id, granary_name = granary_tuple
            silos = granaries_df[granaries_df['storepoint_id'] == granary_id]['silo_name'].unique()
            all_chunks = []
            for silo in silos:
                # Get min/max date for this silo
                date_query = f"SELECT MIN(detection_time), MAX(detection_time) FROM cloud_lq.sensor_data WHERE storepoint_id='{granary_id}' AND store_name='{silo}'"
                min_date, max_date = self.engine.execute(date_query).fetchone()
                if not min_date or not max_date:
                    continue
                current = max(start_date, min_date)
                last = min(end_date, max_date)
                while current <= last:
                    chunk_end = min(current + timedelta(days=chunk_days-1), last)
                    query = f"SELECT * FROM cloud_lq.sensor_data WHERE storepoint_id='{granary_id}' AND store_name='{silo}' AND detection_time BETWEEN '{current}' AND '{chunk_end}'"
                    chunk_df = pd.read_sql(query, self.engine)
                    if not chunk_df.empty:
                        all_chunks.append(chunk_df)
                    current = chunk_end + timedelta(days=1)
            if all_chunks:
                full_df = pd.concat(all_chunks, ignore_index=True)
                table = pa.Table.from_pandas(full_df)
                out_path = Path(f"data/processed/{granary_name}_processed.parquet")
                pq.write_table(table, out_path)
            return granary_name, len(all_chunks)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_granary, granary_groups.groups.keys()), total=len(granary_groups)))
        return results

def create_default_config(output_path: str = "streaming_config.json"):
    """Create a default configuration file."""
    config = {
        "database": {
            "host": "rm-wz9805aymm47k9z3qxo.mysql.rds.aliyuncs.com",
            "port": 3306,
            "database": "cloud_lq",
            "user": "userQuey",
            "password": "UserQ@20240807soft"
        },
        "processing": {
            "initial_chunk_size": 50000,
            "min_chunk_size": 10000,
            "max_chunk_size": 150000,
            "memory_threshold_percent": 75,
            "output_dir": "data/streaming",
            "log_level": "INFO",
            "max_records_per_batch": 100000,
            "memory_safety_threshold_percent": 60,
            "batch_timeout_seconds": 300,
            "force_gc_threshold_percent": 70,
            "pause_duration_seconds": 15,
            "max_retries_per_chunk": 3
        },
        "advanced": {
            "max_retries": 3,
            "retry_delay": 5,
            "enable_progress_bar": True
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Default configuration created: {output_path}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="SQL Data Streamer for SiloFlow")
    CLIUtils.add_common_args(parser)
    CLIUtils.add_date_range_args(parser)
    CLIUtils.add_granary_args(parser)
    parser.add_argument("--no-pipeline", action="store_true", help="Skip pipeline processing")
    parser.add_argument("--create-config", help="Create default configuration file")
    args = parser.parse_args()
    if args.create_config:
        create_default_config(args.create_config)
        return
    if not ValidationUtils.validate_required_args(args, ['start', 'end']):
        logger.error("Both --start and --end are required")
        sys.exit(1)
    if not ValidationUtils.validate_config_file(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}. Use YYYY-MM-DD format.")
        sys.exit(1)
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)
    try:
        streamer = SQLDataStreamer(args.config)
        # Use new parallel optimized retrieval
        results = streamer.parallel_granary_data_retrieval(
            start_date=start_date,
            end_date=end_date,
            granary_filter=getattr(args, 'granary', None),
            chunk_days=7,
            max_workers=4
        )
        print("Parallel retrieval results:", results)
    except Exception as e:
        logger.error(f"Error in parallel retrieval: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()