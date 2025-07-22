#!/usr/bin/env python3
"""
Fix Parquet Standardization Script

This script re-converts existing Parquet files to apply proper column standardization
so they work with the Dashboard. The original Parquet files were converted without
standardization, causing KeyError: 'detection_time' issues.

Usage:
    python scripts/fix_parquet_standardization.py --directory data/preloaded/
"""

import argparse
import pathlib
import sys
from typing import List
import pandas as pd

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from granarypredict.ingestion import read_granary_csv, standardize_granary_csv, save_granary_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_parquet_standardization(
    input_path: pathlib.Path,
    compression: str = 'snappy',
    backup_original: bool = True
) -> pathlib.Path:
    """Re-convert a Parquet file with proper column standardization."""
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Fixing standardization for: {input_path}")
    
    # Read the existing Parquet file
    df = read_granary_csv(input_path)
    
    # Check if standardization is needed
    if 'detection_time' in df.columns:
        logger.info(f"File {input_path.name} already has standardized columns, skipping...")
        return input_path
    
    logger.info(f"Original columns: {list(df.columns)}")
    
    # Create backup if requested (before modifying the file)
    if backup_original:
        backup_path = input_path.with_suffix('.parquet.backup')
        if not backup_path.exists():
            logger.info(f"Creating backup: {backup_path}")
            save_granary_data(df, backup_path, format='parquet', compression=compression)
    
    # Apply column standardization
    logger.info("Applying column standardization...")
    df = standardize_granary_csv(df)
    
    logger.info(f"Standardized columns: {list(df.columns)}")
    
    # Save the standardized version
    result_path = save_granary_data(
        df=df,
        filepath=input_path,
        format='parquet',
        compression=compression
    )
    
    # Get file size comparison
    original_size = input_path.stat().st_size
    logger.info(f"✅ Fixed standardization: {input_path.name}")
    logger.info(f"   File size: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
    
    return result_path

def fix_batch_directory(
    directory: pathlib.Path,
    compression: str = 'snappy',
    backup_original: bool = True,
    pattern: str = "*.parquet"
) -> List[pathlib.Path]:
    """Fix standardization for all Parquet files in a directory."""
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    logger.info(f"Scanning directory: {directory}")
    logger.info(f"Pattern: {pattern}")
    
    # Find all Parquet files
    parquet_files = list(directory.glob(pattern))
    
    if not parquet_files:
        logger.warning(f"No Parquet files found in {directory} matching pattern '{pattern}'")
        return []
    
    logger.info(f"Found {len(parquet_files)} Parquet files to fix")
    
    fixed_files = []
    for parquet_file in parquet_files:
        try:
            # Skip backup files
            if parquet_file.name.endswith('.backup'):
                continue
                
            result_path = fix_parquet_standardization(
                input_path=parquet_file,
                compression=compression,
                backup_original=backup_original
            )
            fixed_files.append(result_path)
        except Exception as e:
            logger.error(f"Failed to fix {parquet_file}: {e}")
    
    logger.info(f"✅ Successfully fixed {len(fixed_files)} files")
    return fixed_files

def main():
    parser = argparse.ArgumentParser(
        description="Fix Parquet files to apply proper column standardization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all Parquet files in a directory
  python scripts/fix_parquet_standardization.py --directory data/preloaded/
  
  # Fix single file
  python scripts/fix_parquet_standardization.py --input data/preloaded/file.parquet
  
  # Fix without creating backups
  python scripts/fix_parquet_standardization.py --directory data/preloaded/ --no-backup
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=pathlib.Path,
        help='Input Parquet file path'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=pathlib.Path,
        help='Directory containing Parquet files to fix'
    )
    
    # Conversion options
    parser.add_argument(
        '--compression', '-c',
        choices=['snappy', 'gzip', 'brotli'],
        default='snappy',
        help='Parquet compression method (default: snappy)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    
    parser.add_argument(
        '--pattern',
        default='*.parquet',
        help='File pattern for batch processing (default: *.parquet)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.input:
            # Single file fix
            fix_parquet_standardization(
                input_path=args.input,
                compression=args.compression,
                backup_original=not args.no_backup
            )
        else:
            # Batch fix
            fix_batch_directory(
                directory=args.directory,
                compression=args.compression,
                backup_original=not args.no_backup,
                pattern=args.pattern
            )
        
        logger.info("🎉 Standardization fix completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Standardization fix failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 