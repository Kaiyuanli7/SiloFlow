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
    
    def full_retrieval(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, granary_filter: Optional[str] = None) -> Dict:
        """
        Perform full data retrieval for all available data from each silo (or filtered by granary).
        
        This method collects ALL available data for each silo using their individual data ranges
        (from earliest to latest available data), ensuring complete data collection across all silos
        regardless of when they started collecting data.
        
        Args:
            start_date: Optional start date (kept for compatibility, not used for filtering)
            end_date: Optional end date (kept for compatibility, not used for filtering)
            granary_filter: Optional granary name or ID to filter by
        
        Returns:
            Dictionary with retrieval results
        """
        logger.info("Starting full data retrieval for ALL available data from each silo...")
        if granary_filter:
            logger.info(f"Filtering by granary: {granary_filter}")
        logger.info("Note: Each silo will use its own actual data range (earliest to latest available data)")
        
        # Set default dates for compatibility (these are not used for data filtering)
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Use a very early date for compatibility (not used for filtering)
            start_date = datetime(2020, 1, 1)
        
        logger.info("Retrieving ALL available data from each silo (using individual silo date ranges)")
        
        # Run the data streamer
        results = self.streamer.stream_all_data(
            start_date=start_date,
            end_date=end_date,
            run_pipeline=False,  # Skip pipeline processing for data retrieval
            granary_filter=granary_filter
        )
        
        logger.info("Full data retrieval completed")
        return results
    
    def incremental_retrieval(self, days: int = 7, granary_filter: Optional[str] = None) -> Dict:
        """
        Perform incremental data retrieval for recent data (or filtered by granary).
        
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
        
        # Check if we have existing data and adjust start date
        last_processed = self.get_last_processed_date()
        if last_processed:
            # Start from the day after the last processed date
            start_date = max(start_date, last_processed + timedelta(days=1))
            logger.info(f"Adjusting start date to {start_date} based on existing data")
        
        if start_date >= end_date:
            logger.info("No new data to retrieve")
            return {
                'success': True,
                'message': 'No new data to retrieve',
                'processed_granaries': 0,
                'total_records': 0
            }
        
        logger.info(f"Retrieving data from {start_date} to {end_date}")
        
        # Run the data streamer
        results = self.streamer.stream_all_data(
            start_date=start_date,
            end_date=end_date,
            run_pipeline=False,  # Skip pipeline processing for data retrieval
            granary_filter=granary_filter
        )
        
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