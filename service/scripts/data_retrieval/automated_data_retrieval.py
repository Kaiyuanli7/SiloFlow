#!/usr/bin/env python3
"""
Automated Data Retrieval for SiloFlow
=====================================

This script automates the complete workflow of:
1. Retrieving data from all granaries and silos in the MySQL database
2. Processing the data through the pipeline (preprocessing only, no training)
3. Organizing the data for further analysis

Usage:
    python automated_data_retrieval.py --full-retrieval
    python automated_data_retrieval.py --incremental --days 7
    python automated_data_retrieval.py --date-range --start 2024-01-01 --end 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(service_dir))

from utils.database_utils import CLIUtils, ValidationUtils, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_data_retrieval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutomatedDataRetrieval:
    """Main class for automated data retrieval and processing."""
    
    def __init__(self, config_path: str = "streaming_config.json"):
        """Initialize the automated data retrieval system."""
        self.config_path = config_path
        # Use centralized data paths
        try:
            from utils.data_paths import data_paths
            self.data_paths = data_paths
            self.output_dir = self.data_paths.get_granaries_dir()
            self.processed_dir = self.data_paths.get_processed_dir()
            self.models_dir = self.data_paths.get_models_dir()
        except ImportError:
            # Fallback to local paths
            self.output_dir = Path("data/granaries")
            self.processed_dir = Path("data/processed")
            self.models_dir = Path("models")
            
            # Ensure directories exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(exist_ok=True)
        
        # Import the SQL data streamer
        try:
            from sql_data_streamer import SQLDataStreamer
            self.streamer = SQLDataStreamer(config_path)
        except ImportError as e:
            logger.error(f"Failed to import SQLDataStreamer: {e}")
            raise
    
    def get_last_processed_date(self) -> Optional[datetime]:
        """Get the last processed date from existing data files."""
        try:
            # Check for existing processed files
            processed_files = list(self.processed_dir.glob("*_processed.parquet"))
            
            if not processed_files:
                return None
            
            # Find the latest date across all processed files
            latest_date = None
            
            for file_path in processed_files:
                try:
                    # Read the file to get the latest date
                    df = pd.read_parquet(file_path)
                    if 'detection_time' in df.columns:
                        file_latest = pd.to_datetime(df['detection_time']).max()
                        if latest_date is None or file_latest > latest_date:
                            latest_date = file_latest
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
                    continue
            
            return latest_date
            
        except Exception as e:
            logger.warning(f"Could not determine last processed date: {e}")
            return None
    
    def check_data_availability(self, granary_filter: Optional[str] = None) -> Dict:
        """Check what data is already available and what needs to be retrieved."""
        try:
            # Get all granaries and silos
            granaries_df = self.streamer.get_all_granaries_and_silos()
            
            if granaries_df.empty:
                return {
                    'success': False,
                    'error': 'No granaries found in database',
                    'existing_data': {},
                    'missing_ranges': {}
                }
            
            # Apply granary filter to the DataFrame if specified
            if granary_filter:
                logger.info(f"Filtering granaries DataFrame by: {granary_filter}")
                try:
                    # Ensure columns are strings and handle potential NaN values
                    granary_name_col = granaries_df['granary_name'].fillna('').astype(str)
                    store_id_col = granaries_df['storepoint_id'].fillna('').astype(str)
                    
                    # Check if the filter looks like a granary ID (32 character hex string)
                    import re
                    clean_filter = granary_filter.strip("'\"")
                    is_id_filter = bool(re.match(r'^[a-fA-F0-9]{32}$', clean_filter))
                    
                    if is_id_filter:
                        # For ID filters, use exact match
                        filtered_df = granaries_df[store_id_col == clean_filter]
                    else:
                        # For name filters, use contains match
                        filtered_df = granaries_df[granary_name_col.str.contains(clean_filter, case=False, na=False)]
                    
                    if filtered_df.empty:
                        logger.warning(f"No granaries found matching filter: {granary_filter}")
                        return {
                            'success': False,
                            'error': f'No granaries found matching: {granary_filter}',
                            'existing_data': {},
                            'missing_ranges': {}
                        }
                    
                    granaries_df = filtered_df.reset_index(drop=True)
                    logger.info(f"Filtered to {len(granaries_df)} silos for granary: {granary_filter}")
                    
                except Exception as e:
                    logger.error(f"Error filtering granaries: {e}")
                    return {
                        'success': False,
                        'error': f'Error filtering granaries: {e}',
                        'existing_data': {},
                        'missing_ranges': {}
                    }
            
            # Check existing data (no need to pass granary_filter since DataFrame is already filtered)
            existing_data = self.streamer.check_existing_data(granaries_df)
            
            # Determine missing ranges (use a wide date range to check all available data)
            start_date = datetime(2020, 1, 1)  # Very early date
            end_date = datetime.now()
            missing_ranges = self.streamer.get_missing_date_ranges(granaries_df, existing_data, start_date, end_date)
            
            return {
                'success': True,
                'existing_data': existing_data,
                'missing_ranges': missing_ranges,
                'total_granaries': len(granaries_df['sub_table_id'].unique()),
                'total_silos': len(granaries_df),
                'silos_with_data': sum(len(g['existing_silos']) for g in existing_data.values()),
                'silos_needing_data': len(missing_ranges),
                'granary_filter': granary_filter
            }
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return {
                'success': False,
                'error': str(e),
                'existing_data': {},
                'missing_ranges': {}
            }
    
    def full_retrieval(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, granary_filter: Optional[str] = None) -> Dict:
        """
        Perform full data retrieval for all available data from each silo (or filtered by granary).
        
        This method checks existing data first and only retrieves missing data ranges,
        ensuring efficient data collection without duplicating existing data.
        
        Args:
            start_date: Optional start date (kept for compatibility, not used for filtering)
            end_date: Optional end date (kept for compatibility, not used for filtering)
            granary_filter: Optional granary name or ID to filter by
        
        Returns:
            Dictionary with retrieval results
        """
        logger.info("Starting full data retrieval with existing data checking...")
        if granary_filter:
            logger.info(f"Filtering by granary: {granary_filter}")
        logger.info("Note: Will check existing data and only retrieve missing ranges")
        
        # Set default dates for compatibility (these are not used for data filtering)
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Use a very early date for compatibility (not used for filtering)
            start_date = datetime(2020, 1, 1)
        
        # Check data availability first
        logger.info("Checking existing data availability...")
        availability = self.check_data_availability(granary_filter)
        
        if not availability['success']:
            logger.error(f"Failed to check data availability: {availability['error']}")
            return availability
        
        logger.info(f"Data availability summary:")
        logger.info(f"  Total granaries: {availability['total_granaries']}")
        logger.info(f"  Total silos: {availability['total_silos']}")
        logger.info(f"  Silos with existing data: {availability['silos_with_data']}")
        logger.info(f"  Silos needing data: {availability['silos_needing_data']}")
        
        if availability['silos_needing_data'] == 0:
            logger.info("All data is already available - no retrieval needed")
            return {
                'success': True,
                'message': 'All data already available',
                'total_granaries': availability['total_granaries'],
                'processed_granaries': 0,
                'failed_granaries': 0,
                'total_records': 0,
                'pipeline_success': True,
                'granary_results': [],
                'existing_data_summary': availability['existing_data'],
                'missing_ranges': availability['missing_ranges']
            }
        
        logger.info(f"Retrieving missing data for {availability['silos_needing_data']} silos")
        
        # Run the data streamer (it will automatically use the missing ranges)
        results = self.streamer.stream_all_data(
            start_date=start_date,
            end_date=end_date,
            run_pipeline=False,  # Skip pipeline processing for data retrieval
            granary_filter=granary_filter
        )
        
        # Add availability information to results
        if results.get('success'):
            results['data_availability'] = availability
        
        logger.info("Full data retrieval completed")
        return results
    
    def incremental_retrieval(self, days: int = 7, granary_filter: Optional[str] = None) -> Dict:
        """
        Perform incremental data retrieval for recent data (or filtered by granary).
        
        This method checks existing data and only retrieves data for the specified time period
        that is not already available.
        
        Args:
            days: Number of days to look back from current date
            granary_filter: Optional granary name or ID to filter by
        
        Returns:
            Dictionary with retrieval results
        """
        logger.info(f"Starting incremental data retrieval for last {days} days...")
        if granary_filter:
            logger.info(f"Filtering by granary: {granary_filter}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Requested date range: {start_date} to {end_date}")
        
        # Check data availability for this specific range
        logger.info("Checking existing data for the requested date range...")
        availability = self.check_data_availability(granary_filter)
        
        if not availability['success']:
            logger.error(f"Failed to check data availability: {availability['error']}")
            return availability
        
        # Filter missing ranges to only include those in our requested date range
        filtered_missing_ranges = {}
        for silo_id, range_info in availability['missing_ranges'].items():
            silo_missing_ranges = []
            for missing_range in range_info['missing_ranges']:
                range_start = missing_range['start']
                range_end = missing_range['end']
                
                # Check if this range overlaps with our requested range
                if range_start <= end_date and range_end >= start_date:
                    # Adjust range to fit within requested bounds
                    adjusted_start = max(range_start, start_date)
                    adjusted_end = min(range_end, end_date)
                    
                    silo_missing_ranges.append({
                        'start': adjusted_start,
                        'end': adjusted_end,
                        'reason': f"{missing_range['reason']}_incremental"
                    })
            
            if silo_missing_ranges:
                filtered_missing_ranges[silo_id] = {
                    **range_info,
                    'missing_ranges': silo_missing_ranges,
                    'total_missing_days': sum((r['end'] - r['start']).days + 1 for r in silo_missing_ranges)
                }
        
        if not filtered_missing_ranges:
            logger.info("No missing data found for the requested date range")
            return {
                'success': True,
                'message': 'No missing data for requested date range',
                'total_granaries': availability['total_granaries'],
                'processed_granaries': 0,
                'failed_granaries': 0,
                'total_records': 0,
                'pipeline_success': True,
                'granary_results': [],
                'existing_data_summary': availability['existing_data'],
                'missing_ranges': filtered_missing_ranges
            }
        
        logger.info(f"Found missing data for {len(filtered_missing_ranges)} silos in requested range")
        
        # Run the data streamer with filtered missing ranges
        results = self.streamer.stream_all_data(
            start_date=start_date,
            end_date=end_date,
            run_pipeline=False,  # Skip pipeline processing for data retrieval
            granary_filter=granary_filter
        )
        
        # Add availability information to results
        if results.get('success'):
            results['data_availability'] = availability
            results['filtered_missing_ranges'] = filtered_missing_ranges
        
        logger.info("Incremental data retrieval completed")
        return results
    
    def date_range_retrieval(self, start_date: datetime, end_date: datetime, granary_filter: Optional[str] = None) -> Dict:
        """
        Perform data retrieval for a specific date range (or filtered by granary).
        
        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            granary_filter: Optional granary name or ID to filter by
        
        Returns:
            Dictionary with retrieval results
        """
        logger.info(f"Starting date range retrieval from {start_date} to {end_date}")
        if granary_filter:
            logger.info(f"Filtering by granary: {granary_filter}")
        
        # Run the data streamer
        results = self.streamer.stream_all_data(
            start_date=start_date,
            end_date=end_date,
            run_pipeline=False,  # Skip pipeline processing for data retrieval
            granary_filter=granary_filter
        )
        
        logger.info("Date range retrieval completed")
        return results
    
    def _get_earliest_available_date(self) -> Optional[datetime]:
        """Get the earliest available date from the database."""
        try:
            # Query to find the earliest date across all tables
            query = """
            SELECT MIN(earliest_date) as earliest
            FROM (
                SELECT MIN(batch) as earliest_date FROM cloud_lq.lq_point_history_77
                UNION ALL
                SELECT MIN(batch) as earliest_date FROM cloud_lq.lq_point_history_78
                UNION ALL
                SELECT MIN(batch) as earliest_date FROM cloud_lq.lq_point_history_79
                UNION ALL
                SELECT MIN(batch) as earliest_date FROM cloud_lq.lq_point_history_80
                -- Add more tables as needed
            ) as all_dates
            WHERE earliest_date IS NOT NULL
            """
            
            df = pd.read_sql(query, self.streamer.engine)
            if not df.empty and df['earliest'].iloc[0] is not None:
                earliest_date = pd.to_datetime(df['earliest'].iloc[0])
                logger.info(f"Earliest available date: {earliest_date}")
                return earliest_date
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not determine earliest available date: {e}")
            return None
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate a summary report of the retrieval process."""
        report = []
        report.append("=" * 60)
        report.append("AUTOMATED DATA RETRIEVAL SUMMARY")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Granaries: {results.get('total_granaries', 0)}")
        report.append(f"Processed Granaries: {results.get('processed_granaries', 0)}")
        report.append(f"Failed Granaries: {results.get('failed_granaries', 0)}")
        report.append(f"Total Records: {results.get('total_records', 0):,}")
        report.append(f"Pipeline Success: {results.get('pipeline_success', False)}")
        
        # Add existing data information
        if results.get('existing_data_summary'):
            existing_data = results['existing_data_summary']
            total_existing_granaries = sum(1 for g in existing_data.values() if g['granary_file'])
            total_existing_silos = sum(len(g['existing_silos']) for g in existing_data.values())
            total_existing_records = sum(g['total_records'] for g in existing_data.values())
            
            report.append(f"\nEXISTING DATA SUMMARY:")
            report.append("-" * 40)
            report.append(f"Granaries with existing data: {total_existing_granaries}")
            report.append(f"Silos with existing data: {total_existing_silos}")
            report.append(f"Total existing records: {total_existing_records:,}")
        
        # Add missing ranges information
        if results.get('missing_ranges'):
            missing_ranges = results['missing_ranges']
            total_missing_silos = len(missing_ranges)
            total_missing_days = sum(info['total_missing_days'] for info in missing_ranges.values())
            
            report.append(f"\nMISSING DATA SUMMARY:")
            report.append("-" * 40)
            report.append(f"Silos needing data: {total_missing_silos}")
            report.append(f"Total missing days: {total_missing_days:,}")
            
            if total_missing_silos > 0:
                report.append("\nMISSING DATA DETAILS:")
                report.append("-" * 40)
                for silo_id, info in missing_ranges.items():
                    granary_name = info['granary_name']
                    missing_ranges_list = info['missing_ranges']
                    total_days = info['total_missing_days']
                    report.append(f"{granary_name}/{silo_id}: {len(missing_ranges_list)} ranges, {total_days} days")
        
        if results.get('granary_results'):
            report.append("\nGRANARY DETAILS:")
            report.append("-" * 40)
            for granary_result in results['granary_results']:
                if granary_result.get('success'):
                    report.append(f"✓ {granary_result['granary_name']}: {granary_result.get('total_records', 0):,} records")
                else:
                    report.append(f"✗ {granary_result['granary_name']}: Failed")
        
        # File locations
        report.append("\nOUTPUT FILES:")
        report.append("-" * 40)
        report.append(f"Raw Data: {self.output_dir}")
        report.append(f"Processed Data: {self.processed_dir}")
        report.append(f"Models: {self.models_dir}")
        
        return "\n".join(report)
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """Clean up old temporary files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old log files
            log_files = list(Path(".").glob("*.log"))
            for log_file in log_files:
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Cleaned up old log file: {log_file}")
            
            # Clean up old temporary files
            temp_dir = Path("temp_uploads")
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*"):
                    if temp_file.stat().st_mtime < cutoff_date.timestamp():
                        temp_file.unlink()
                        logger.info(f"Cleaned up old temp file: {temp_file}")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Automated Data Retrieval for SiloFlow")
    CLIUtils.add_common_args(parser)
    CLIUtils.add_retrieval_mode_args(parser)
    CLIUtils.add_granary_args(parser)
    CLIUtils.add_date_range_args(parser)
    parser.add_argument("--cleanup", action="store_true", help="Clean up old files after retrieval")
    parser.add_argument("--days-to-keep", type=int, default=30, help="Days to keep files during cleanup")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full_retrieval, args.incremental, args.date_range]):
        logger.error("Must specify one of: --full-retrieval, --incremental, or --date-range")
        sys.exit(1)
    
    if args.date_range and (not args.start or not args.end):
        logger.error("--date-range requires both --start and --end dates")
        sys.exit(1)
    
    try:
        # Initialize the automated retrieval system
        retrieval = AutomatedDataRetrieval(args.config)
        
        # Perform the requested retrieval
        if args.full_retrieval:
            results = retrieval.full_retrieval(granary_filter=args.granary)
        elif args.incremental:
            results = retrieval.incremental_retrieval(args.days, granary_filter=args.granary)
        elif args.date_range:
            start_date = datetime.strptime(args.start, "%Y-%m-%d")
            end_date = datetime.strptime(args.end, "%Y-%m-%d")
            results = retrieval.date_range_retrieval(start_date, end_date, granary_filter=args.granary)
        
        # Generate and display summary report
        report = retrieval.generate_summary_report(results)
        print("\n" + report)
        
        # Save report to file
        report_file = Path("data_retrieval_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")
        
        # Cleanup if requested
        if args.cleanup:
            retrieval.cleanup_old_files(args.days_to_keep)
        
        if results.get('success'):
            logger.info("Automated data retrieval completed successfully!")
        else:
            logger.error("Automated data retrieval failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Automated data retrieval failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 