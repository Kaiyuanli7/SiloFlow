#!/usr/bin/env python3
"""
Streaming Data Processor for Massive Datasets
==============================================

Handles datasets with hundreds of millions of rows through:
- Chunked processing with automatic memory management
- Streaming feature engineering 
- Incremental model training
- Out-of-core operations using Dask/Vaex
"""

import logging
import gc
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from contextlib import contextmanager

# Optional dependencies for massive scale
try:
    import dask.dataframe as dd
    import dask
    from dask.distributed import Client, LocalCluster
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    dd = None

try:
    import vaex
    HAS_VAEX = True
except ImportError:
    HAS_VAEX = False
    vaex = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

logger = logging.getLogger(__name__)

class MassiveDatasetProcessor:
    """
    Process datasets with hundreds of millions of rows efficiently.
    
    Features:
    - Automatic chunking with memory management
    - Streaming feature engineering
    - Multiple backend support (Pandas, Dask, Vaex, Polars)
    - Out-of-core processing
    - Incremental model training
    """
    
    def __init__(self, 
                 chunk_size: int = 100_000,
                 memory_threshold_percent: float = 75.0,
                 backend: str = "auto",
                 enable_dask: bool = True,
                 n_workers: int = None):
        """
        Initialize the massive dataset processor.
        
        Parameters
        ----------
        chunk_size : int
            Initial chunk size for processing
        memory_threshold_percent : float
            Memory threshold to trigger chunk size reduction
        backend : str
            Processing backend: 'auto', 'pandas', 'dask', 'vaex', 'polars'
        enable_dask : bool
            Whether to use Dask for parallel processing
        n_workers : int
            Number of workers for parallel processing
        """
        self.chunk_size = chunk_size
        self.memory_threshold = memory_threshold_percent
        self.backend = self._select_backend(backend)
        self.enable_dask = enable_dask and HAS_DASK
        self.n_workers = n_workers or min(8, max(2, self._get_cpu_count() // 2))
        
        # Dynamic memory management
        self.min_chunk_size = 10_000
        self.max_chunk_size = 1_000_000
        self.current_chunk_size = chunk_size
        
        # Dask client for distributed processing
        self.dask_client = None
        
        # Performance tracking
        self.processed_rows = 0
        self.processed_chunks = 0
        
        logger.info(f"Initialized MassiveDatasetProcessor:")
        logger.info(f"  Backend: {self.backend}")
        logger.info(f"  Initial chunk size: {self.chunk_size:,}")
        logger.info(f"  Memory threshold: {self.memory_threshold}%")
        logger.info(f"  Workers: {self.n_workers}")
        
    def _select_backend(self, backend: str) -> str:
        """Select the best available backend for processing."""
        if backend == "auto":
            if HAS_POLARS:
                return "polars"  # Fastest for large datasets
            elif HAS_VAEX:
                return "vaex"    # Best for billion+ row datasets
            elif HAS_DASK:
                return "dask"    # Good parallel processing
            else:
                return "pandas"  # Fallback
        return backend
    
    def _get_cpu_count(self) -> int:
        """Get CPU count safely."""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except:
            return 4
    
    @contextmanager
    def dask_cluster(self):
        """Context manager for Dask distributed processing."""
        if not self.enable_dask:
            yield None
            return
            
        try:
            # Create local cluster for massive processing
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit='4GB',  # Per worker memory limit
                dashboard_address=None,  # Disable dashboard for performance
            )
            self.dask_client = Client(cluster)
            logger.info(f"Started Dask cluster with {self.n_workers} workers")
            yield self.dask_client
        except Exception as e:
            logger.warning(f"Failed to start Dask cluster: {e}")
            yield None
        finally:
            if self.dask_client:
                self.dask_client.close()
                self.dask_client = None
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is below threshold."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < self.memory_threshold
        except ImportError:
            return True  # Assume OK if psutil not available
    
    def _adjust_chunk_size(self):
        """Dynamically adjust chunk size based on memory usage."""
        if not self._check_memory_usage():
            # Reduce chunk size if memory is high
            self.current_chunk_size = max(
                self.min_chunk_size,
                int(self.current_chunk_size * 0.75)
            )
            logger.info(f"Reduced chunk size to {self.current_chunk_size:,} due to memory pressure")
        elif self.current_chunk_size < self.chunk_size:
            # Increase chunk size if memory is OK
            self.current_chunk_size = min(
                self.max_chunk_size,
                int(self.current_chunk_size * 1.25)
            )
            logger.info(f"Increased chunk size to {self.current_chunk_size:,}")
    
    def read_massive_dataset(self, file_path: Union[str, Path]) -> Iterator[pd.DataFrame]:
        """
        Read a massive dataset in chunks with automatic memory management.
        
        Supports:
        - Parquet files (recommended for massive datasets)
        - CSV files (with chunking)
        - Multiple file formats
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Reading massive dataset: {file_path}")
        logger.info(f"Backend: {self.backend}, Initial chunk size: {self.current_chunk_size:,}")
        
        if self.backend == "polars" and HAS_POLARS:
            yield from self._read_with_polars(file_path)
        elif self.backend == "vaex" and HAS_VAEX:
            yield from self._read_with_vaex(file_path)
        elif self.backend == "dask" and HAS_DASK:
            yield from self._read_with_dask(file_path)
        else:
            yield from self._read_with_pandas(file_path)
    
    def _read_with_pandas(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read massive dataset with Pandas chunking."""
        if file_path.suffix.lower() == '.parquet':
            # For Parquet, use PyArrow for memory-efficient reading
            if HAS_ARROW:
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(batch_size=self.current_chunk_size):
                    df = batch.to_pandas()
                    self._track_progress(len(df))
                    yield df
                    self._adjust_chunk_size()
                    gc.collect()
            else:
                # Fallback: read entire Parquet file (not ideal for massive data)
                df = pd.read_parquet(file_path)
                for chunk in self._chunk_dataframe(df):
                    yield chunk
        else:
            # CSV chunking
            for chunk in pd.read_csv(file_path, chunksize=self.current_chunk_size):
                self._track_progress(len(chunk))
                yield chunk
                self._adjust_chunk_size()
                gc.collect()
    
    def _read_with_polars(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read massive dataset with Polars (fastest option)."""
        logger.info("Using Polars backend for massive dataset processing")
        
        if file_path.suffix.lower() == '.parquet':
            # Polars lazy reading for memory efficiency
            lazy_df = pl.scan_parquet(file_path)
        else:
            lazy_df = pl.scan_csv(file_path)
        
        # Process in chunks
        total_rows = lazy_df.select(pl.count()).collect().item()
        logger.info(f"Total rows in dataset: {total_rows:,}")
        
        for offset in range(0, total_rows, self.current_chunk_size):
            chunk_pl = lazy_df.slice(offset, self.current_chunk_size).collect()
            chunk_pd = chunk_pl.to_pandas()  # Convert to Pandas for compatibility
            
            self._track_progress(len(chunk_pd))
            yield chunk_pd
            self._adjust_chunk_size()
            
            del chunk_pl, chunk_pd
            gc.collect()
    
    def _read_with_vaex(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read massive dataset with Vaex (best for billion+ rows)."""
        logger.info("Using Vaex backend for massive dataset processing")
        
        if file_path.suffix.lower() == '.parquet':
            df_vaex = vaex.open(str(file_path))
        else:
            df_vaex = vaex.from_csv(str(file_path))
        
        logger.info(f"Total rows in dataset: {len(df_vaex):,}")
        
        # Process in chunks
        for i in range(0, len(df_vaex), self.current_chunk_size):
            end_idx = min(i + self.current_chunk_size, len(df_vaex))
            chunk_vaex = df_vaex[i:end_idx]
            chunk_pd = chunk_vaex.to_pandas_df()
            
            self._track_progress(len(chunk_pd))
            yield chunk_pd
            self._adjust_chunk_size()
            
            del chunk_vaex, chunk_pd
            gc.collect()
    
    def _read_with_dask(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read massive dataset with Dask (good parallel processing)."""
        logger.info("Using Dask backend for massive dataset processing")
        
        if file_path.suffix.lower() == '.parquet':
            ddf = dd.read_parquet(file_path)
        else:
            ddf = dd.read_csv(file_path)
        
        logger.info(f"Dask partitions: {ddf.npartitions}")
        
        # Process each partition
        for partition in ddf.to_delayed():
            chunk_pd = partition.compute()
            self._track_progress(len(chunk_pd))
            yield chunk_pd
            self._adjust_chunk_size()
            gc.collect()
    
    def _chunk_dataframe(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Split a DataFrame into chunks."""
        for i in range(0, len(df), self.current_chunk_size):
            end_idx = min(i + self.current_chunk_size, len(df))
            chunk = df.iloc[i:end_idx].copy()
            self._track_progress(len(chunk))
            yield chunk
            self._adjust_chunk_size()
    
    def _track_progress(self, chunk_rows: int):
        """Track processing progress."""
        self.processed_rows += chunk_rows
        self.processed_chunks += 1
        
        if self.processed_chunks % 10 == 0:
            logger.info(f"Processed {self.processed_chunks} chunks, {self.processed_rows:,} total rows")
    
    def process_massive_features(self, 
                                file_path: Union[str, Path],
                                output_path: Union[str, Path],
                                feature_functions: list = None) -> bool:
        """
        Process massive datasets with streaming feature engineering.
        
        Parameters
        ----------
        file_path : str or Path
            Input file path
        output_path : str or Path  
            Output file path
        feature_functions : list
            List of feature engineering functions to apply
        
        Returns
        -------
        bool
            True if processing was successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Default feature functions for massive datasets
            if feature_functions is None:
                feature_functions = [
                    self._add_basic_time_features,
                    self._add_essential_lags,
                    self._add_rolling_features,
                ]
            
            # Process in chunks and write incrementally
            first_chunk = True
            total_processed = 0
            
            logger.info(f"Starting massive feature processing: {file_path} -> {output_path}")
            
            for chunk in self.read_massive_dataset(file_path):
                # Apply feature engineering to chunk
                processed_chunk = chunk.copy()
                
                for feature_func in feature_functions:
                    try:
                        processed_chunk = feature_func(processed_chunk)
                    except Exception as e:
                        logger.warning(f"Feature function {feature_func.__name__} failed: {e}")
                        continue
                
                # Write chunk to output
                if output_path.suffix.lower() == '.parquet':
                    if first_chunk:
                        processed_chunk.to_parquet(output_path, index=False)
                        first_chunk = False
                    else:
                        # Append to existing Parquet file
                        existing_table = pq.read_table(output_path)
                        new_table = pa.Table.from_pandas(processed_chunk)
                        combined_table = pa.concat_tables([existing_table, new_table])
                        pq.write_table(combined_table, output_path)
                else:
                    # CSV append mode
                    mode = 'w' if first_chunk else 'a'
                    header = first_chunk
                    processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                    first_chunk = False
                
                total_processed += len(processed_chunk)
                
                # Memory cleanup
                del processed_chunk
                gc.collect()
                
                if total_processed % 100_000 == 0:
                    logger.info(f"Processed {total_processed:,} rows so far...")
            
            logger.info(f"Massive feature processing completed! Total rows: {total_processed:,}")
            return True
            
        except Exception as e:
            logger.error(f"Massive feature processing failed: {e}")
            return False
    
    def _add_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic time features optimized for streaming."""
        if 'detection_time' not in df.columns:
            return df
        
        df = df.copy()
        df['detection_time'] = pd.to_datetime(df['detection_time'], errors='coerce')
        
        # Basic time features
        df['year'] = df['detection_time'].dt.year
        df['month'] = df['detection_time'].dt.month
        df['day'] = df['detection_time'].dt.day
        df['hour'] = df['detection_time'].dt.hour
        df['dayofweek'] = df['detection_time'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        return df
    
    def _add_essential_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add essential lag features optimized for streaming."""
        if 'temperature_grain' not in df.columns:
            return df
        
        # Simple 1-day lag (more complex lags require state management)
        group_cols = [c for c in ['granary_id', 'heap_id', 'grid_x', 'grid_y', 'grid_z'] if c in df.columns]
        
        if group_cols:
            df['lag_temp_1d'] = df.groupby(group_cols)['temperature_grain'].shift(1)
            df['delta_temp_1d'] = df['temperature_grain'] - df['lag_temp_1d']
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features optimized for streaming."""
        if 'temperature_grain' not in df.columns:
            return df
        
        group_cols = [c for c in ['granary_id', 'heap_id', 'grid_x', 'grid_y', 'grid_z'] if c in df.columns]
        
        if group_cols:
            # Simple rolling features (3-day window for efficiency)
            grouped = df.groupby(group_cols)['temperature_grain']
            df['roll_mean_3d'] = grouped.transform(lambda x: x.rolling(3, min_periods=1).mean())
            df['roll_std_3d'] = grouped.transform(lambda x: x.rolling(3, min_periods=1).std())
        
        return df


def create_massive_processing_pipeline(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    chunk_size: int = 100_000,
    backend: str = "auto"
) -> bool:
    """
    Create a complete pipeline for processing massive datasets.
    
    This is the main entry point for handling datasets with hundreds of millions of rows.
    """
    processor = MassiveDatasetProcessor(
        chunk_size=chunk_size,
        backend=backend,
        enable_dask=True
    )
    
    return processor.process_massive_features(input_path, output_path)


# Utility functions for massive dataset handling
def estimate_memory_requirements(file_path: Union[str, Path], 
                                sample_rows: int = 10000) -> Dict[str, Any]:
    """
    Estimate memory requirements for processing a massive dataset.
    
    Returns recommendations for chunk size and processing strategy.
    """
    file_path = Path(file_path)
    
    try:
        # Sample the dataset to estimate memory usage
        if file_path.suffix.lower() == '.parquet':
            if HAS_ARROW:
                parquet_file = pq.ParquetFile(file_path)
                sample_table = parquet_file.read_row_group(0, columns=None)
                sample_df = sample_table.to_pandas()[:sample_rows]
                total_rows = parquet_file.metadata.num_rows
            else:
                sample_df = pd.read_parquet(file_path, nrows=sample_rows)
                total_rows = len(pd.read_parquet(file_path, columns=['detection_time']))
        else:
            sample_df = pd.read_csv(file_path, nrows=sample_rows)
            # Estimate total rows (this is approximate)
            total_rows = sum(1 for _ in open(file_path)) - 1
        
        # Calculate memory usage per row
        memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        estimated_total_memory_gb = (total_rows * memory_per_row) / (1024**3)
        
        # Get available memory
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_memory_gb = 8  # Assume 8GB if psutil not available
        
        # Calculate recommended chunk size (use 10% of available memory)
        target_chunk_memory_gb = available_memory_gb * 0.1
        recommended_chunk_size = int(target_chunk_memory_gb * (1024**3) / memory_per_row)
        
        # Ensure reasonable bounds
        recommended_chunk_size = max(10_000, min(1_000_000, recommended_chunk_size))
        
        return {
            'total_rows': total_rows,
            'memory_per_row_bytes': memory_per_row,
            'estimated_total_memory_gb': estimated_total_memory_gb,
            'available_memory_gb': available_memory_gb,
            'recommended_chunk_size': recommended_chunk_size,
            'fits_in_memory': estimated_total_memory_gb < available_memory_gb * 0.7,
            'processing_strategy': 'in_memory' if estimated_total_memory_gb < available_memory_gb * 0.7 else 'streaming'
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate memory requirements: {e}")
        return {
            'error': str(e),
            'recommended_chunk_size': 50_000,
            'processing_strategy': 'streaming'
        }


if __name__ == "__main__":
    # Example usage for massive datasets
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample file
    input_file = "massive_dataset.parquet"  # Replace with actual path
    output_file = "processed_massive_dataset.parquet"
    
    # Estimate memory requirements
    memory_info = estimate_memory_requirements(input_file)
    print(f"Memory analysis: {memory_info}")
    
    # Process the massive dataset
    success = create_massive_processing_pipeline(
        input_path=input_file,
        output_path=output_file,
        chunk_size=memory_info.get('recommended_chunk_size', 100_000)
    )
    
    if success:
        print("Massive dataset processing completed successfully!")
    else:
        print("Massive dataset processing failed!")
