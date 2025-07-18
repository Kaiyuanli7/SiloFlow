#!/usr/bin/env python3
"""
CSV to Parquet Conversion Utility

This script converts existing CSV files to Parquet format for better performance
with large datasets (multiple GBs to tens of GBs).

Usage:
    python scripts/convert_csv_to_parquet.py --input data.csv --output data.parquet
    python scripts/convert_csv_to_parquet.py --batch data/processed/
"""

import argparse
import pathlib
import sys
from typing import List, Optional

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from granarypredict.ingestion import convert_csv_to_parquet, read_granary_csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_single_file(
    input_path: pathlib.Path,
    output_path: Optional[pathlib.Path] = None,
    compression: str = 'snappy',
    delete_original: bool = False
) -> pathlib.Path:
    """Convert a single CSV file to Parquet."""
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Converting: {input_path}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.parquet')
    
    # Convert file
    result_path = convert_csv_to_parquet(
        csv_filepath=input_path,
        parquet_filepath=output_path,
        compression=compression,
        delete_original=delete_original
    )
    
    # Get file size comparison
    original_size = input_path.stat().st_size if input_path.exists() else 0
    parquet_size = result_path.stat().st_size
    compression_ratio = (1 - parquet_size / original_size) * 100 if original_size > 0 else 0
    
    logger.info(f"✅ Converted: {input_path.name} → {result_path.name}")
    logger.info(f"   Original: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
    logger.info(f"   Parquet:  {parquet_size:,} bytes ({parquet_size/1024/1024:.1f} MB)")
    logger.info(f"   Compression: {compression_ratio:.1f}% smaller")
    
    return result_path

def convert_batch_directory(
    directory: pathlib.Path,
    compression: str = 'snappy',
    delete_original: bool = False,
    pattern: str = "*.csv*"
) -> List[pathlib.Path]:
    """Convert all CSV files in a directory to Parquet."""
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    logger.info(f"Scanning directory: {directory}")
    logger.info(f"Pattern: {pattern}")
    
    # Find all CSV files (including compressed)
    csv_files = list(directory.glob(pattern))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {directory} matching pattern '{pattern}'")
        return []
    
    logger.info(f"Found {len(csv_files)} CSV files to convert")
    
    converted_files = []
    for csv_file in csv_files:
        try:
            result_path = convert_single_file(
                input_path=csv_file,
                compression=compression,
                delete_original=delete_original
            )
            converted_files.append(result_path)
        except Exception as e:
            logger.error(f"Failed to convert {csv_file}: {e}")
    
    logger.info(f"✅ Successfully converted {len(converted_files)} files")
    return converted_files

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format for better performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python scripts/convert_csv_to_parquet.py --input data.csv --output data.parquet
  
  # Convert with gzip compression
  python scripts/convert_csv_to_parquet.py --input data.csv --compression gzip
  
  # Convert all CSV files in directory
  python scripts/convert_csv_to_parquet.py --batch data/processed/
  
  # Convert and delete original files
  python scripts/convert_csv_to_parquet.py --batch data/processed/ --delete-original
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=pathlib.Path,
        help='Input CSV file path'
    )
    input_group.add_argument(
        '--batch', '-b',
        type=pathlib.Path,
        help='Directory containing CSV files to convert'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=pathlib.Path,
        help='Output Parquet file path (for single file conversion)'
    )
    
    # Conversion options
    parser.add_argument(
        '--compression', '-c',
        choices=['snappy', 'gzip', 'brotli'],
        default='snappy',
        help='Parquet compression method (default: snappy)'
    )
    
    parser.add_argument(
        '--delete-original', '-d',
        action='store_true',
        help='Delete original CSV files after conversion'
    )
    
    parser.add_argument(
        '--pattern',
        default='*.csv*',
        help='File pattern for batch conversion (default: *.csv*)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.input:
            # Single file conversion
            convert_single_file(
                input_path=args.input,
                output_path=args.output,
                compression=args.compression,
                delete_original=args.delete_original
            )
        else:
            # Batch conversion
            convert_batch_directory(
                directory=args.batch,
                compression=args.compression,
                delete_original=args.delete_original,
                pattern=args.pattern
            )
        
        logger.info("🎉 Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 