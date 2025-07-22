#!/usr/bin/env python3
"""
Script to compress existing model files to save storage space.

This script will:
1. Find all .joblib model files in the models directory
2. Compress them using gzip compression
3. Optionally remove the original files
4. Show compression statistics

Usage:
    python scripts/compress_models.py [--remove-originals] [--compression-level 6]
"""

import argparse
import logging
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from granarypredict.compression_utils import batch_compress_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compress existing model files to save storage space")
    parser.add_argument(
        "--remove-originals", 
        action="store_true",
        help="Remove original uncompressed files after compression"
    )
    parser.add_argument(
        "--compression-level", 
        type=int, 
        default=6,
        choices=range(1, 10),
        help="Compression level (1-9, higher = more compression but slower)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing model files (default: models)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.joblib",
        help="File pattern to match (default: *.joblib)"
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return 1
    
    logger.info(f"Starting model compression...")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"File pattern: {args.pattern}")
    logger.info(f"Compression level: {args.compression_level}")
    logger.info(f"Remove originals: {args.remove_originals}")
    
    try:
        compressed_files = batch_compress_models(
            models_dir=models_dir,
            pattern=args.pattern,
            compression_level=args.compression_level,
            use_gzip=True,
            backup_original=not args.remove_originals
        )
        
        if compressed_files:
            logger.info(f"✅ Successfully compressed {len(compressed_files)} model files")
            
            # Calculate total space saved
            total_original_size = 0
            total_compressed_size = 0
            
            for compressed_file in compressed_files:
                if compressed_file.exists():
                    compressed_size = compressed_file.stat().st_size
                    total_compressed_size += compressed_size
                    
                    # Try to find original file size
                    original_file = compressed_file.with_suffix('').with_suffix('.joblib')
                    if original_file.exists():
                        total_original_size += original_file.stat().st_size
            
            if total_original_size > 0:
                space_saved = total_original_size - total_compressed_size
                space_saved_mb = space_saved / (1024 * 1024)
                compression_ratio = (1 - total_compressed_size / total_original_size) * 100
                
                logger.info(f"Compression Summary:")
                logger.info(f"   Original size: {total_original_size / (1024*1024):.2f} MB")
                logger.info(f"   Compressed size: {total_compressed_size / (1024*1024):.2f} MB")
                logger.info(f"   Space saved: {space_saved_mb:.2f} MB ({compression_ratio:.1f}%)")
        else:
            logger.info("No model files found to compress")
            
    except Exception as e:
        logger.error(f"Error during compression: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 