"""
Polars-optimized versions of data utilities for large dataset operations.
Provides significant performance improvements for sorting and grouping operations.
"""

from __future__ import annotations
from typing import List, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import Polars
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

# Default sort hierarchy – same as data_utils.py
_SORT_HIERARCHY: List[str] = [
    "granary_id",  # warehouse identifier
    "heap_id",     # silo / heap identifier
    "grid_x",
    "grid_y", 
    "grid_z",
    "detection_time",  # timestamp last to keep chronological order within sensor
]

def comprehensive_sort_optimized(df: pd.DataFrame, row_threshold: int = 50_000) -> pd.DataFrame:
    """
    Optimized sorting with automatic Polars backend selection for large datasets.
    
    Uses Polars for datasets >50K rows (5-15x faster), pandas for smaller datasets.
    Always returns pandas DataFrame for compatibility.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort
    row_threshold : int, default 50_000
        Row count threshold above which to use Polars optimization
        
    Returns
    -------
    pd.DataFrame
        Sorted DataFrame (always pandas format)
    """
    dataset_size = len(df)
    
    # Use Polars optimization for large datasets
    if HAS_POLARS and dataset_size > row_threshold:
        logger.info(f"*** POLARS SORTING: {dataset_size:,} rows detected, using Polars-optimized sorting")
        try:
            return _comprehensive_sort_polars(df)
        except Exception as e:
            logger.warning(f"Polars sorting failed, falling back to pandas: {e}")
            return _comprehensive_sort_pandas(df)
    else:
        logger.debug(f"Using pandas sorting for {dataset_size:,} rows")
        return _comprehensive_sort_pandas(df)

def _comprehensive_sort_polars(df: pd.DataFrame) -> pd.DataFrame:
    """Polars-optimized sorting - 5-15x faster for large datasets."""
    # Convert to Polars
    df_pl = pl.from_pandas(df)
    
    # Find available sort columns
    sort_cols = [c for c in _SORT_HIERARCHY if c in df_pl.columns]
    
    if sort_cols:
        # Polars sorting is much faster than pandas for large datasets
        df_pl = df_pl.sort(sort_cols)
    
    # Convert back to pandas
    return df_pl.to_pandas()

def _comprehensive_sort_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Standard pandas sorting (original implementation)."""
    sort_cols = [c for c in _SORT_HIERARCHY if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df

def assign_group_id_optimized(df: pd.DataFrame, 
                             col_name: str = "_group_id",
                             row_threshold: int = 50_000) -> pd.DataFrame:
    """
    Optimized group ID assignment with Polars acceleration for large datasets.
    
    Uses Polars for datasets >50K rows, pandas for smaller datasets.
    Always returns pandas DataFrame for compatibility.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process
    col_name : str, default "_group_id"
        Name of the group ID column to create
    row_threshold : int, default 50_000
        Row count threshold above which to use Polars optimization
        
    Returns
    -------
    pd.DataFrame
        DataFrame with group ID column added
    """
    dataset_size = len(df)
    
    # Use Polars optimization for large datasets
    if HAS_POLARS and dataset_size > row_threshold:
        logger.info(f"*** POLARS GROUPING: {dataset_size:,} rows detected, using Polars-optimized grouping")
        try:
            return _assign_group_id_polars(df, col_name)
        except Exception as e:
            logger.warning(f"Polars grouping failed, falling back to pandas: {e}")
            return _assign_group_id_pandas(df, col_name)
    else:
        logger.debug(f"Using pandas grouping for {dataset_size:,} rows")
        return _assign_group_id_pandas(df, col_name)

def _assign_group_id_polars(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Polars-optimized group ID assignment."""
    # Convert to Polars
    df_pl = pl.from_pandas(df)
    
    # Polars string operations are faster for large datasets
    if {"granary_id", "heap_id"}.issubset(df_pl.columns):
        df_pl = df_pl.with_columns(
            (pl.col("granary_id").cast(pl.Utf8) + "_" + pl.col("heap_id").cast(pl.Utf8))
            .alias(col_name)
        )
    elif "granary_id" in df_pl.columns:
        df_pl = df_pl.with_columns(pl.col("granary_id").cast(pl.Utf8).alias(col_name))
    elif "heap_id" in df_pl.columns:
        df_pl = df_pl.with_columns(pl.col("heap_id").cast(pl.Utf8).alias(col_name))
    else:
        df_pl = df_pl.with_columns(pl.lit("all").alias(col_name))
    
    # Convert back to pandas
    return df_pl.to_pandas()

def _assign_group_id_pandas(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Standard pandas group ID assignment (original implementation)."""
    df = df.copy()
    if {"granary_id", "heap_id"}.issubset(df.columns):
        df[col_name] = df["granary_id"].astype(str) + "_" + df["heap_id"].astype(str)
    elif "granary_id" in df.columns:
        df[col_name] = df["granary_id"].astype(str)
    elif "heap_id" in df.columns:
        df[col_name] = df["heap_id"].astype(str)
    else:
        df[col_name] = "all"
    return df

def optimized_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data processing pipeline with Polars optimizations.
    
    Combines sorting and group assignment with automatic backend selection.
    """
    dataset_size = len(df)
    logger.info(f"Processing dataset with {dataset_size:,} rows")
    
    # Apply optimized sorting
    df = comprehensive_sort_optimized(df)
    
    # Apply optimized group assignment  
    df = assign_group_id_optimized(df)
    
    return df

# Convenience function for backward compatibility
def comprehensive_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop-in replacement for the original comprehensive_sort function.
    Automatically uses Polars optimization for large datasets.
    """
    return comprehensive_sort_optimized(df)

def assign_group_id(df: pd.DataFrame, *, col_name: str = "_group_id") -> pd.DataFrame:
    """
    Drop-in replacement for the original assign_group_id function.  
    Automatically uses Polars optimization for large datasets.
    """
    return assign_group_id_optimized(df, col_name)
