"""
GPU-Accelerated Data Utilities for SiloFlow

This module provides GPU-accelerated data processing using NVIDIA RAPIDS cuDF
for massive performance improvements in data manipulation operations.

Key Features:
- Drop-in replacement for pandas with up to 150x speedup
- GPU-accelerated sorting, grouping, and aggregation
- Automatic fallback to pandas/Polars if GPU not available
- Optimized for large datasets (100K+ rows)
- Compatible with existing SiloFlow pipeline

Performance Comparisons:
- 100M records aggregation: cuDF ~69ms vs pandas ~1.37s (20x speedup)
- Large dataset sorting: 10-50x faster than pandas
- Memory-efficient GPU processing for multi-GB datasets

Usage:
    from granarypredict.gpu_data_utils import (
        gpu_comprehensive_sort_optimized,
        gpu_assign_group_id_optimized,
        gpu_data_pipeline
    )
"""

import warnings
import logging
from typing import Optional, List, Union, Dict, Any
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def clean_log_message(message: str) -> str:
    """Remove emojis and unicode characters that cause encoding issues on Windows."""
    # Remove common emojis and replace with text equivalents
    replacements = {
        '🚀': '[GPU]',
        '⚡': '[Polars]', 
        '🐼': '[Pandas]',
        '✅': '[OK]',
        '⚠️': '[WARNING]',
        '🔍': '[INFO]',
        '🔄': '[PROCESSING]'
    }
    clean_msg = message
    for emoji, replacement in replacements.items():
        clean_msg = clean_msg.replace(emoji, replacement)
    return clean_msg

# GPU Detection and Import Management
try:
    import cudf
    import cupy as cp
    HAS_CUDF = True
    logger.info("RAPIDS cuDF available - GPU-accelerated data processing enabled")
except ImportError:
    HAS_CUDF = False
    logger.info("RAPIDS cuDF not available - falling back to pandas/Polars")

# Fallback imports
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

def detect_optimal_backend(df_size: int, gpu_memory_mb: Optional[float] = None) -> str:
    """
    Automatically detect the optimal data processing backend based on data size and available resources.
    
    Args:
        df_size: Number of rows in the dataset
        gpu_memory_mb: Available GPU memory in MB (if known)
    
    Returns:
        str: Recommended backend ('cudf', 'polars', or 'pandas')
    """
    # Estimate memory requirements (rough calculation)
    # Assume average 50 bytes per row for typical sensor data
    estimated_memory_mb = (df_size * 50) / (1024 * 1024)
    
    if HAS_CUDF and df_size >= 50_000:  # cuDF sweet spot for large datasets
        if gpu_memory_mb is None or estimated_memory_mb < (gpu_memory_mb * 0.7):  # Use 70% of GPU memory
            return 'cudf'
    
    if HAS_POLARS and df_size >= 10_000:  # Polars good for medium-large datasets
        return 'polars'
    
    return 'pandas'  # Default fallback

def gpu_comprehensive_sort_optimized(
    df: pd.DataFrame, 
    sort_columns: List[str], 
    ascending: Union[bool, List[bool]] = True,
    auto_backend: bool = True
) -> pd.DataFrame:
    """
    GPU-accelerated comprehensive sorting with automatic backend selection.
    
    Provides massive performance improvements for large datasets using RAPIDS cuDF.
    Falls back gracefully to Polars or pandas if GPU not available.
    
    Args:
        df: Input DataFrame
        sort_columns: Columns to sort by
        ascending: Sort order (True for ascending, False for descending)
        auto_backend: If True, automatically selects optimal backend
    
    Returns:
        Sorted DataFrame
    
    Performance:
        - GPU (cuDF): 10-50x faster for large datasets
        - Polars: 5-15x faster than pandas
        - pandas: Baseline performance
    """
    if df.empty:
        return df
    
    backend = detect_optimal_backend(len(df)) if auto_backend else 'cudf'
    
    # GPU-accelerated sorting with cuDF
    if backend == 'cudf' and HAS_CUDF:
        try:
            logger.info(clean_log_message(f"🚀 Using GPU (cuDF) for sorting {len(df):,} rows by {sort_columns}"))
            
            # Convert pandas DataFrame to cuDF
            gpu_df = cudf.from_pandas(df)
            
            # Perform GPU-accelerated sorting
            sorted_gpu_df = gpu_df.sort_values(
                by=sort_columns, 
                ascending=ascending, 
                ignore_index=True
            )
            
            # Convert back to pandas for compatibility
            result = sorted_gpu_df.to_pandas()
            
            logger.info(clean_log_message(f"✅ GPU sorting completed for {len(result):,} rows"))
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  GPU sorting failed, falling back to Polars: {e}")
            backend = 'polars' if HAS_POLARS else 'pandas'
    
    # Fallback to Polars if available
    if backend == 'polars' and HAS_POLARS:
        try:
            logger.info(clean_log_message(f"⚡ Using Polars for sorting {len(df):,} rows by {sort_columns}"))
            
            # Convert to Polars LazyFrame for optimization
            pl_df = pl.from_pandas(df).lazy()
            
            # Apply sorting
            sorted_pl_df = pl_df.sort(by=sort_columns, descending=(not ascending if isinstance(ascending, bool) else [not asc for asc in ascending]))
            
            # Collect and convert back to pandas
            result = sorted_pl_df.collect().to_pandas()
            
            logger.info(clean_log_message(f"✅ Polars sorting completed for {len(result):,} rows"))
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  Polars sorting failed, falling back to pandas: {e}")
    
    # Final fallback to pandas
    logger.info(clean_log_message(f"🐼 Using pandas for sorting {len(df):,} rows by {sort_columns}"))
    result = df.sort_values(sort_columns).reset_index(drop=True)
    logger.info(clean_log_message(f"✅ Pandas sorting completed for {len(result):,} rows"))
    return result

def gpu_assign_group_id_optimized(
    df: pd.DataFrame, 
    group_columns: List[str],
    group_id_col: str = 'group_id',
    auto_backend: bool = True
) -> pd.DataFrame:
    """
    GPU-accelerated group ID assignment with automatic backend selection.
    
    Efficiently assigns unique group IDs based on combinations of specified columns.
    Uses GPU acceleration for massive performance improvements on large datasets.
    
    Args:
        df: Input DataFrame
        group_columns: Columns to group by
        group_id_col: Name of the group ID column to create
        auto_backend: If True, automatically selects optimal backend
    
    Returns:
        DataFrame with group_id column added
    
    Performance:
        - GPU (cuDF): 15-60x faster for grouping operations
        - Polars: 3-8x faster than pandas
        - pandas: Baseline performance
    """
    if df.empty:
        return df
    
    backend = detect_optimal_backend(len(df)) if auto_backend else 'cudf'
    
    # GPU-accelerated grouping with cuDF
    if backend == 'cudf' and HAS_CUDF:
        try:
            logger.info(clean_log_message(f"🚀 Using GPU (cuDF) for group assignment on {len(df):,} rows by {group_columns}"))
            
            # Convert to cuDF
            gpu_df = cudf.from_pandas(df)
            
            # Create unique group combinations using GPU
            group_combinations = gpu_df[group_columns].drop_duplicates().reset_index(drop=True)
            group_combinations[group_id_col] = cp.arange(len(group_combinations))
            
            # Merge back to assign group IDs
            result_gpu = gpu_df.merge(group_combinations, on=group_columns, how='left')
            
            # Convert back to pandas
            result = result_gpu.to_pandas()
            
            logger.info(clean_log_message(f"✅ GPU group assignment completed: {len(group_combinations):,} unique groups"))
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  GPU group assignment failed, falling back to Polars: {e}")
            backend = 'polars' if HAS_POLARS else 'pandas'
    
    # Fallback to Polars if available
    if backend == 'polars' and HAS_POLARS:
        try:
            logger.info(clean_log_message(f"⚡ Using Polars for group assignment on {len(df):,} rows by {group_columns}"))
            
            # Convert to Polars
            pl_df = pl.from_pandas(df)
            
            # Create group ID mapping
            group_mapping = (
                pl_df
                .select(group_columns)
                .unique()
                .with_row_count(group_id_col)
            )
            
            # Join back to original data
            result_pl = pl_df.join(group_mapping, on=group_columns, how='left')
            result = result_pl.to_pandas()
            
            logger.info(clean_log_message(f"✅ Polars group assignment completed: {len(group_mapping):,} unique groups"))
            return result
            
        except Exception as e:
            logger.warning(clean_log_message(f"⚠️  Polars group assignment failed, falling back to pandas: {e}"))
    
    # Final fallback to pandas
    logger.info(clean_log_message(f"🐼 Using pandas for group assignment on {len(df):,} rows by {group_columns}"))
    
    # Create unique combinations and assign IDs
    unique_groups = df[group_columns].drop_duplicates().reset_index(drop=True)
    unique_groups[group_id_col] = range(len(unique_groups))
    
    # Merge back to original dataframe
    result = df.merge(unique_groups, on=group_columns, how='left')
    
    logger.info(clean_log_message(f"✅ Pandas group assignment completed: {len(unique_groups):,} unique groups"))
    return result

def gpu_aggregation_optimized(
    df: pd.DataFrame,
    group_columns: List[str],
    agg_config: Dict[str, Union[str, List[str]]],
    auto_backend: bool = True
) -> pd.DataFrame:
    """
    GPU-accelerated data aggregation with automatic backend selection.
    
    Performs high-performance aggregations using GPU acceleration when available.
    This is where cuDF shows the most dramatic performance improvements.
    
    Args:
        df: Input DataFrame
        group_columns: Columns to group by
        agg_config: Aggregation configuration (e.g., {'temperature': ['mean', 'std'], 'count': 'count'})
        auto_backend: If True, automatically selects optimal backend
    
    Returns:
        Aggregated DataFrame
    
    Performance:
        - GPU (cuDF): 20-100x faster for complex aggregations
        - Polars: 5-15x faster than pandas
        - pandas: Baseline performance
    """
    if df.empty:
        return df
    
    backend = detect_optimal_backend(len(df)) if auto_backend else 'cudf'
    
    # GPU-accelerated aggregation with cuDF (best performance gains here)
    if backend == 'cudf' and HAS_CUDF:
        try:
            logger.info(f"🚀 Using GPU (cuDF) for aggregation on {len(df):,} rows by {group_columns}")
            
            # Convert to cuDF
            gpu_df = cudf.from_pandas(df)
            
            # Perform GPU-accelerated aggregation
            result_gpu = gpu_df.groupby(group_columns).agg(agg_config).reset_index()
            
            # Flatten column names if multi-level
            if isinstance(result_gpu.columns, pd.MultiIndex):
                result_gpu.columns = ['_'.join(col).strip() if col[1] else col[0] for col in result_gpu.columns.values]
            
            # Convert back to pandas
            result = result_gpu.to_pandas()
            
            logger.info(f"✅ GPU aggregation completed: {len(result):,} groups")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  GPU aggregation failed, falling back to Polars: {e}")
            backend = 'polars' if HAS_POLARS else 'pandas'
    
    # Fallback to Polars if available
    if backend == 'polars' and HAS_POLARS:
        try:
            logger.info(f"⚡ Using Polars for aggregation on {len(df):,} rows by {group_columns}")
            
            # Convert to Polars and build aggregation expressions
            pl_df = pl.from_pandas(df)
            
            # Build Polars aggregation expressions
            agg_exprs = []
            for col, funcs in agg_config.items():
                if isinstance(funcs, str):
                    funcs = [funcs]
                for func in funcs:
                    if func == 'mean':
                        agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                    elif func == 'std':
                        agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
                    elif func == 'count':
                        agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
                    elif func == 'sum':
                        agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                    elif func == 'min':
                        agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
                    elif func == 'max':
                        agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            
            # Perform aggregation
            result_pl = pl_df.group_by(group_columns).agg(agg_exprs)
            result = result_pl.to_pandas()
            
            logger.info(f"✅ Polars aggregation completed: {len(result):,} groups")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  Polars aggregation failed, falling back to pandas: {e}")
    
    # Final fallback to pandas
    logger.info(f"🐼 Using pandas for aggregation on {len(df):,} rows by {group_columns}")
    result = df.groupby(group_columns).agg(agg_config).reset_index()
    
    # Flatten column names if multi-level
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = ['_'.join(col).strip() if col[1] else col[0] for col in result.columns.values]
    
    logger.info(f"✅ Pandas aggregation completed: {len(result):,} groups")
    return result

def gpu_data_pipeline(
    df: pd.DataFrame,
    sort_columns: Optional[List[str]] = None,
    group_columns: Optional[List[str]] = None,
    agg_config: Optional[Dict[str, Union[str, List[str]]]] = None,
    auto_backend: bool = True,
    return_backend_info: bool = False
) -> Union[pd.DataFrame, tuple]:
    """
    Comprehensive GPU-accelerated data processing pipeline.
    
    Combines sorting, grouping, and aggregation operations with automatic
    backend selection for optimal performance across different data sizes.
    
    Args:
        df: Input DataFrame
        sort_columns: Columns to sort by (optional)
        group_columns: Columns to group by (optional)
        agg_config: Aggregation configuration (optional)
        auto_backend: If True, automatically selects optimal backend
        return_backend_info: If True, returns tuple with (result, backend_used)
    
    Returns:
        Processed DataFrame or tuple with (DataFrame, backend_info)
    
    Example:
        # Basic sorting and grouping
        result = gpu_data_pipeline(
            df=data,
            sort_columns=['timestamp', 'silo_id'],
            group_columns=['silo_id', 'date'],
            agg_config={'temperature': ['mean', 'std'], 'humidity': 'mean'}
        )
        
        # With backend information
        result, backend = gpu_data_pipeline(
            df=data,
            sort_columns=['timestamp'],
            return_backend_info=True
        )
        print(f"Processing completed using: {backend}")
    """
    if df.empty:
        result = df
        backend_used = 'none'
    else:
        backend_used = detect_optimal_backend(len(df)) if auto_backend else 'cudf'
        
        result = df.copy()
        
        # Apply sorting if requested
        if sort_columns:
            result = gpu_comprehensive_sort_optimized(
                result, sort_columns, auto_backend=auto_backend
            )
        
        # Apply grouping and aggregation if requested
        if group_columns and agg_config:
            result = gpu_aggregation_optimized(
                result, group_columns, agg_config, auto_backend=auto_backend
            )
        elif group_columns:
            # Just assign group IDs without aggregation
            result = gpu_assign_group_id_optimized(
                result, group_columns, auto_backend=auto_backend
            )
    
    if return_backend_info:
        return result, backend_used
    return result

def get_gpu_data_backend_info() -> Dict[str, Any]:
    """
    Get information about available GPU data processing backends.
    
    Returns:
        Dict with backend availability and recommendations
    """
    info = {
        'cudf_available': HAS_CUDF,
        'polars_available': HAS_POLARS,
        'recommended_thresholds': {
            'cudf_min_rows': 50_000,
            'polars_min_rows': 10_000,
            'pandas_fallback': True
        },
        'performance_estimates': {
            'cudf_speedup': '10-150x vs pandas',
            'polars_speedup': '3-15x vs pandas',
            'optimal_use_cases': {
                'cudf': 'Large datasets (50K+ rows), complex aggregations',
                'polars': 'Medium datasets (10K+ rows), general processing',
                'pandas': 'Small datasets (<10K rows), compatibility'
            }
        }
    }
    
    if HAS_CUDF:
        try:
            # Get GPU memory info if available
            gpu_memory = cudf.get_gpu_memory_info()
            info['gpu_memory_mb'] = gpu_memory.total / (1024 * 1024)
            info['gpu_memory_free_mb'] = gpu_memory.free / (1024 * 1024)
        except:
            info['gpu_memory_mb'] = 'unknown'
    
    return info

# Export main functions
__all__ = [
    'gpu_comprehensive_sort_optimized',
    'gpu_assign_group_id_optimized', 
    'gpu_aggregation_optimized',
    'gpu_data_pipeline',
    'detect_optimal_backend',
    'get_gpu_data_backend_info',
    'HAS_CUDF',
    'HAS_POLARS'
]
