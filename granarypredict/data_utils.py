from __future__ import annotations

"""Utility helpers for data ordering & grouping (May-2025)."""

from typing import List

import pandas as pd

__all__ = [
    "comprehensive_sort",
    "assign_group_id",
]

# Default sort hierarchy – columns will be used only if present.
_SORT_HIERARCHY: List[str] = [
    "granary_id",  # warehouse identifier
    "heap_id",     # silo / heap identifier
    "grid_x",
    "grid_y",
    "grid_z",
    "detection_time",  # timestamp last to keep chronological order within sensor
]


def comprehensive_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* sorted according to the canonical hierarchy.

    1. granary_id ➜ 2. heap_id ➜ 3. grid_x ➜ 4. grid_y ➜ 5. grid_z ➜ 6. detection_time

    Columns missing in *df* are simply ignored.
    """
    sort_cols = [c for c in _SORT_HIERARCHY if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def assign_group_id(df: pd.DataFrame, *, col_name: str = "_group_id") -> pd.DataFrame:
    """Add a column (*col_name*) that uniquely identifies a physical silo.

    Priority: granary_id+heap_id → granary_id → heap_id → constant "all".
    """
    if {"granary_id", "heap_id"}.issubset(df.columns):
        df[col_name] = df["granary_id"].astype(str) + "_" + df["heap_id"].astype(str)
    elif "granary_id" in df.columns:
        df[col_name] = df["granary_id"].astype(str)
    elif "heap_id" in df.columns:
        df[col_name] = df["heap_id"].astype(str)
    else:
        df[col_name] = "all"
    return df 

"""
Data compression utilities for granary data processing.
Provides efficient compression options for CSV files and other data formats.
"""

import pandas as pd
import pathlib
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

def save_compressed_csv(
    df: pd.DataFrame, 
    filepath: Union[str, pathlib.Path], 
    compression: Optional[str] = 'gzip',
    index: bool = False,
    **kwargs
) -> pathlib.Path:
    """
    Save DataFrame as compressed CSV with automatic file extension.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or pathlib.Path
        Base filepath (extension will be added automatically)
    compression : str or None, default 'gzip'
        Compression method: 'gzip', 'bz2', 'zip', 'xz', or None
    index : bool, default False
        Whether to save DataFrame index
    **kwargs : 
        Additional arguments passed to pd.to_csv()
    
    Returns:
    --------
    pathlib.Path
        Path to the saved compressed file
    """
    filepath = pathlib.Path(filepath)
    
    # Add compression extension if not present
    if compression:
        if not filepath.suffix.startswith(f'.{compression}'):
            filepath = filepath.with_suffix(f'{filepath.suffix}.{compression}')
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with compression
    df.to_csv(filepath, compression=compression, index=index, **kwargs)
    
    # Get file size info
    original_size = len(df.to_csv(index=index, **kwargs))
    compressed_size = filepath.stat().st_size
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    logger.info(f"Saved compressed CSV: {filepath}")
    logger.info(f"  Original size: {original_size:,} bytes")
    logger.info(f"  Compressed size: {compressed_size:,} bytes")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}%")
    
    return filepath

def read_compressed_csv(
    filepath: Union[str, pathlib.Path],
    **kwargs
) -> pd.DataFrame:
    """
    Read compressed CSV file with automatic compression detection.
    
    Parameters:
    -----------
    filepath : str or pathlib.Path
        Path to compressed CSV file
    **kwargs :
        Additional arguments passed to pd.read_csv()
    
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    filepath = pathlib.Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect compression from file extension
    compression = None
    if filepath.suffix.endswith('.gz') or filepath.suffix.endswith('.gzip'):
        compression = 'gzip'
    elif filepath.suffix.endswith('.bz2'):
        compression = 'bz2'
    elif filepath.suffix.endswith('.zip'):
        compression = 'zip'
    elif filepath.suffix.endswith('.xz'):
        compression = 'xz'
    
    logger.info(f"Reading compressed CSV: {filepath} (compression: {compression})")
    
    return pd.read_csv(filepath, compression=compression, **kwargs)

def save_parquet(
    df: pd.DataFrame,
    filepath: Union[str, pathlib.Path],
    compression: str = 'snappy',
    index: bool = False,
    **kwargs
) -> pathlib.Path:
    """
    Save DataFrame as Parquet file (much more efficient than CSV).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or pathlib.Path
        Filepath (will add .parquet extension if not present)
    compression : str, default 'snappy'
        Compression method: 'snappy', 'gzip', 'brotli', or None
    index : bool, default False
        Whether to save DataFrame index
    **kwargs :
        Additional arguments passed to df.to_parquet()
    
    Returns:
    --------
    pathlib.Path
        Path to the saved Parquet file
    """
    filepath = pathlib.Path(filepath)
    
    # Add .parquet extension if not present
    if filepath.suffix != '.parquet':
        filepath = filepath.with_suffix('.parquet')
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet
    df.to_parquet(filepath, compression=compression, index=index, **kwargs)
    
    # Get file size info
    original_size = len(df.to_csv(index=index))
    parquet_size = filepath.stat().st_size
    compression_ratio = (1 - parquet_size / original_size) * 100
    
    logger.info(f"Saved Parquet file: {filepath}")
    logger.info(f"  Original CSV size: {original_size:,} bytes")
    logger.info(f"  Parquet size: {parquet_size:,} bytes")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}%")
    
    return filepath

def read_parquet(
    filepath: Union[str, pathlib.Path],
    **kwargs
) -> pd.DataFrame:
    """
    Read Parquet file.
    
    Parameters:
    -----------
    filepath : str or pathlib.Path
        Path to Parquet file
    **kwargs :
        Additional arguments passed to pd.read_parquet()
    
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    filepath = pathlib.Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Reading Parquet file: {filepath}")
    
    return pd.read_parquet(filepath, **kwargs)

def get_compression_recommendation(
    df: pd.DataFrame,
    file_size_threshold_mb: float = 10.0
) -> dict:
    """
    Get compression recommendations based on DataFrame characteristics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    file_size_threshold_mb : float, default 10.0
        Size threshold in MB to recommend Parquet over CSV
    
    Returns:
    --------
    dict
        Compression recommendations
    """
    # Estimate CSV size
    csv_size_mb = len(df.to_csv()) / (1024 * 1024)
    
    recommendations = {
        'csv_size_mb': csv_size_mb,
        'recommended_format': 'parquet' if csv_size_mb > file_size_threshold_mb else 'csv',
        'compression_options': {}
    }
    
    # CSV compression options
    recommendations['compression_options']['csv'] = {
        'gzip': {
            'description': 'Best balance of compression and speed',
            'compression_ratio': '60-80%',
            'speed': 'Fast'
        },
        'bz2': {
            'description': 'Best compression ratio',
            'compression_ratio': '70-90%',
            'speed': 'Slow'
        },
        'zip': {
            'description': 'Good compression, widely compatible',
            'compression_ratio': '50-70%',
            'speed': 'Medium'
        }
    }
    
    # Parquet compression options
    recommendations['compression_options']['parquet'] = {
        'snappy': {
            'description': 'Fastest compression and decompression',
            'compression_ratio': '40-60%',
            'speed': 'Very Fast'
        },
        'gzip': {
            'description': 'Good compression ratio',
            'compression_ratio': '60-80%',
            'speed': 'Medium'
        },
        'brotli': {
            'description': 'Best compression ratio for Parquet',
            'compression_ratio': '70-90%',
            'speed': 'Slow'
        }
    }
    
    return recommendations 