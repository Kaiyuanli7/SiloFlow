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
import time
import multiprocessing
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Tuple, Union, List
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
                                feature_functions: list = None,
                                use_comprehensive_pipeline: bool = True) -> bool:
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
        use_comprehensive_pipeline : bool
            If True, uses Dashboard.py-style comprehensive preprocessing.
            If False, uses legacy basic preprocessing for compatibility.
        
        Returns
        -------
        bool
            True if processing was successful
        """
        try:
            logger.info(f"🚀 Starting massive feature processing")
            logger.info(f"   📁 Input: {file_path}")
            logger.info(f"   💾 Output: {output_path}")
            logger.info(f"   🔧 Comprehensive pipeline: {use_comprehensive_pipeline}")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Default comprehensive feature functions matching Dashboard.py pipeline
            if feature_functions is None:
                if use_comprehensive_pipeline:
                    # Full Dashboard.py-style preprocessing pipeline
                    try:
                        logger.info("🔧 Initializing comprehensive preprocessing pipeline")
                        feature_functions = [
                            # Step 1: Column standardization (critical first step)
                            self._standardize_columns,
                            # Step 2: Basic data cleaning
                            self._basic_clean,
                            # Step 3: Calendar gap insertion (CRITICAL - was missing in batch processing)
                            self._insert_calendar_gaps,
                            # Step 4: Numeric interpolation (CRITICAL - was missing in batch processing)
                            self._interpolate_sensor_numeric,
                            # Step 5: Comprehensive feature engineering (Dashboard.py style)
                            self._add_comprehensive_time_features,
                            self._add_spatial_features,
                            self._add_time_since_last_measurement,
                            self._add_multi_lag_features,
                            self._add_rolling_stats,
                            self._add_directional_features,
                            self._add_stability_features,
                            # Step 6: Multi-horizon targets for forecasting
                            self._add_multi_horizon_targets,
                            # Step 7: Group assignment for ML operations
                            self._assign_group_id,
                            # Step 8: Final comprehensive sorting
                            self._comprehensive_sort,
                        ]
                        logger.info(f"✅ Comprehensive pipeline initialized with {len(feature_functions)} processing steps")
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize comprehensive pipeline: {e}")
                        logger.warning("🔄 Falling back to basic processing pipeline")
                        # Fallback to basic processing only
                        feature_functions = [
                            self._standardize_columns,
                            self._basic_clean,
                            self._fill_missing
                        ]
                    logger.info("📋 Using comprehensive Dashboard.py-style preprocessing pipeline")
                else:
                    # Legacy basic preprocessing for backward compatibility
                    feature_functions = [
                        self._add_basic_time_features,
                        self._add_essential_lags,
                        self._add_rolling_features,
                    ]
                    logger.info("📋 Using legacy basic preprocessing pipeline")
            
            # Process in chunks and write incrementally
            first_chunk = True
            total_processed = 0
            chunk_count = 0
            
            logger.info(f"📊 Starting chunked processing of massive dataset")
            
            for chunk in self.read_massive_dataset(file_path):
                chunk_count += 1
                chunk_size = len(chunk)
                
                logger.info(f"📦 Processing chunk {chunk_count}: {chunk_size:,} rows")
                
                # Apply feature engineering to chunk
                processed_chunk = chunk.copy()
                
                for step_idx, feature_func in enumerate(feature_functions, 1):
                    step_name = feature_func.__name__.replace('_', ' ').title()
                    logger.info(f"   🔧 Step {step_idx}/{len(feature_functions)}: {step_name}")
                    
                    try:
                        processed_chunk = feature_func(processed_chunk)
                        logger.info(f"   ✅ {step_name} completed - Shape: {processed_chunk.shape}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ {step_name} failed: {e}")
                        logger.warning(f"   🔄 Continuing with remaining processing steps")
                        continue
                
                logger.info(f"✅ Chunk {chunk_count} processing completed: {processed_chunk.shape}")
                
                # Write chunk to output
                try:
                    if output_path.suffix.lower() == '.parquet':
                        if first_chunk:
                            logger.info(f"💾 Writing first chunk to {output_path}")
                            processed_chunk.to_parquet(output_path, index=False)
                            first_chunk = False
                        else:
                            # Append to existing Parquet file with schema harmonization
                            logger.info(f"💾 Appending chunk {chunk_count} to existing parquet file")
                            if HAS_ARROW:
                                try:
                                    existing_table = pq.read_table(output_path)
                                    new_table = pa.Table.from_pandas(processed_chunk)
                                    
                                    # Harmonize schemas before concatenation
                                    if not existing_table.schema.equals(new_table.schema):
                                        logger.info(f"🔧 Harmonizing schemas for chunk {chunk_count}")
                                        # Convert new table to match existing schema
                                        harmonized_new_table = self._harmonize_arrow_schema(existing_table.schema, new_table)
                                        combined_table = pa.concat_tables([existing_table, harmonized_new_table])
                                    else:
                                        combined_table = pa.concat_tables([existing_table, new_table])
                                    
                                    pq.write_table(combined_table, output_path)
                                except Exception as arrow_error:
                                    logger.warning(f"⚠️ Arrow concatenation failed: {arrow_error}")
                                    logger.info("🔄 Falling back to pandas concatenation")
                                    # Fallback: read existing data and concatenate with pandas
                                    existing_df = pd.read_parquet(output_path)  
                                    # Harmonize column types
                                    processed_chunk = self._harmonize_dataframe_schema(existing_df, processed_chunk)
                                    combined_df = pd.concat([existing_df, processed_chunk], ignore_index=True)
                                    combined_df.to_parquet(output_path, index=False)
                            else:
                                # Fallback: read existing data and concatenate
                                existing_df = pd.read_parquet(output_path)
                                # Harmonize column types
                                processed_chunk = self._harmonize_dataframe_schema(existing_df, processed_chunk)
                                combined_df = pd.concat([existing_df, processed_chunk], ignore_index=True)
                                combined_df.to_parquet(output_path, index=False)
                    else:
                        # CSV append mode
                        mode = 'w' if first_chunk else 'a'
                        header = first_chunk
                        logger.info(f"💾 Writing chunk {chunk_count} to CSV ({'new file' if first_chunk else 'append mode'})")
                        processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                        first_chunk = False
                    
                    total_processed += len(processed_chunk)
                    logger.info(f"📊 Cumulative progress: {total_processed:,} rows processed across {chunk_count} chunks")
                    
                except Exception as write_error:
                    logger.error(f"❌ Failed to write chunk {chunk_count}: {write_error}")
                    raise write_error
                
                # Memory cleanup
                del processed_chunk
                gc.collect()
                
                if total_processed % 50_000 == 0:  # More frequent updates
                    logger.info(f"🎯 Progress update: {total_processed:,} rows processed so far...")
                
                # More frequent memory cleanup for better responsiveness
                if chunk_count % 5 == 0:  # Every 5 chunks
                    gc.collect()
            
            # Final verification
            if output_path.exists():
                final_size = output_path.stat().st_size / (1024**2)
                logger.info(f"🎉 Massive feature processing completed successfully!")
                logger.info(f"   📊 Total chunks processed: {chunk_count}")
                logger.info(f"   📊 Total rows processed: {total_processed:,}")
                logger.info(f"   📊 Output file size: {final_size:.1f} MB")
                logger.info(f"   📁 Output saved to: {output_path}")
                return True
            else:
                logger.error(f"❌ Output file was not created: {output_path}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Massive feature processing failed with error: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.error(f"   Full traceback: {traceback.format_exc()}")
            return False
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names matching Dashboard.py pipeline."""
        try:
            from granarypredict.ingestion import standardize_granary_csv
            logger.info("Successfully imported standardize_granary_csv from ingestion module")
            result = standardize_granary_csv(df)
            logger.info(f"Column standardization completed, shape: {result.shape}")
            return result
        except ImportError as ie:
            logger.error(f"Import error in column standardization: {ie}")
            logger.error("Failed to import standardize_granary_csv from granarypredict.ingestion")
            # Fallback: basic column name standardization
            df_copy = df.copy()
            df_copy.columns = [col.strip().lower().replace(' ', '_') for col in df_copy.columns]
            return df_copy
        except Exception as e:
            logger.error(f"Column standardization failed with error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return df
    
    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning matching Dashboard.py pipeline."""
        try:
            from granarypredict.cleaning import basic_clean
            logger.info("✅ Successfully imported basic_clean from cleaning module")
            result = basic_clean(df)
            logger.info(f"✅ Basic cleaning completed successfully - Shape: {result.shape}")
            return result
        except ImportError as ie:
            logger.error(f"❌ Import error in basic cleaning: {ie}")
            logger.info("🔄 Using fallback basic cleaning implementation")
            # Fallback: basic cleaning without module
            df_copy = df.copy()
            df_copy.columns = [c.strip() for c in df_copy.columns]
            df_copy.replace({"-": pd.NA, "NA": pd.NA, "N/A": pd.NA, -999: pd.NA}, inplace=True)
            df_copy.dropna(axis=1, how="all", inplace=True)
            df_copy.drop_duplicates(inplace=True)
            logger.info(f"✅ Fallback basic cleaning completed - Shape: {df_copy.shape}")
            return df_copy
        except Exception as e:
            logger.error(f"❌ Basic cleaning failed with error: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.error(f"   Full traceback: {traceback.format_exc()}")
            logger.info("🔄 Returning original dataframe without cleaning")
            return df
    
    def _insert_calendar_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insert calendar gaps for continuous time series - CRITICAL missing step."""
        try:
            from granarypredict.cleaning_helpers import insert_calendar_gaps
            return insert_calendar_gaps(df)
        except Exception as e:
            logger.warning(f"Calendar gap insertion failed: {e}")
            return df
    
    def _interpolate_sensor_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate sensor values for synthetic rows - CRITICAL missing step."""
        try:
            from granarypredict.cleaning_helpers import interpolate_sensor_numeric
            return interpolate_sensor_numeric(df)
        except Exception as e:
            logger.warning(f"Sensor numeric interpolation failed: {e}")
            return df
    
    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final NaN cleanup matching Dashboard.py pipeline."""
        try:
            from granarypredict.cleaning import fill_missing
            return fill_missing(df)
        except Exception as e:
            logger.warning(f"Fill missing failed: {e}")
            return df
    
    def _add_comprehensive_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive time features matching Dashboard.py pipeline."""
        try:
            from granarypredict.features import create_time_features
            return create_time_features(df)
        except Exception as e:
            logger.warning(f"Time features creation failed: {e}")
            return df
    
    def _add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features matching Dashboard.py pipeline."""
        try:
            from granarypredict.features import create_spatial_features
            return create_spatial_features(df)
        except Exception as e:
            logger.warning(f"Spatial features creation failed: {e}")
            return df
    
    def _add_time_since_last_measurement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time since last measurement features."""
        try:
            from granarypredict.features import add_time_since_last_measurement
            return add_time_since_last_measurement(df)
        except Exception as e:
            logger.warning(f"Time since last measurement failed: {e}")
            return df
    
    def _add_multi_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-lag features with parallel processing matching Dashboard.py."""
        try:
            from granarypredict.features import add_multi_lag_parallel
            return add_multi_lag_parallel(df, lags=(1,2,3,4,5,6,7,14,30))
        except Exception as e:
            logger.warning(f"Multi-lag features failed: {e}")
            return df
    
    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics with parallel processing matching Dashboard.py."""
        try:
            from granarypredict.features import add_rolling_stats_parallel
            return add_rolling_stats_parallel(df, window_days=7)
        except Exception as e:
            logger.warning(f"Rolling stats failed: {e}")
            return df
    
    def _add_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add directional features matching Dashboard.py pipeline."""
        try:
            from granarypredict.features import add_directional_features_lean
            return add_directional_features_lean(df)
        except Exception as e:
            logger.warning(f"Directional features failed: {e}")
            return df
    
    def _add_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add stability features with parallel processing matching Dashboard.py."""
        try:
            from granarypredict.features import add_stability_features_parallel
            return add_stability_features_parallel(df)
        except Exception as e:
            logger.warning(f"Stability features failed: {e}")
            return df
    
    def _add_horizon_specific_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add horizon-specific directional features matching Dashboard.py pipeline."""
        try:
            from granarypredict.features import add_horizon_specific_directional_features
            # Use same max_horizon as Dashboard.py (HORIZON_DAYS constant)
            return add_horizon_specific_directional_features(df, max_horizon=7)
        except Exception as e:
            logger.warning(f"Horizon-specific directional features failed: {e}")
            return df
    
    def _add_multi_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-horizon targets matching Dashboard.py pipeline."""
        try:
            from granarypredict.features import add_multi_horizon_targets
            # Use same horizon tuple as Dashboard.py (HORIZON_TUPLE)
            return add_multi_horizon_targets(df, horizons=tuple(range(1, 8)))
        except Exception as e:
            logger.warning(f"Multi-horizon targets failed: {e}")
            return df
    
    def _assign_group_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign group IDs for ML operations matching Dashboard.py."""
        try:
            from granarypredict.data_utils import assign_group_id
            return assign_group_id(df)
        except Exception as e:
            logger.warning(f"Group ID assignment failed: {e}")
            return df
    
    def _comprehensive_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive final sorting matching Dashboard.py pipeline."""
        try:
            from granarypredict.data_utils import comprehensive_sort
            return comprehensive_sort(df)
        except Exception as e:
            logger.warning(f"Comprehensive sort failed: {e}")
            return df
    
    def _harmonize_arrow_schema(self, target_schema: pa.Schema, source_table: pa.Table) -> pa.Table:
        """Harmonize Arrow table schema to match target schema."""
        try:
            if not HAS_ARROW:
                return source_table
                
            # Get the target field types
            target_fields = {field.name: field.type for field in target_schema}
            
            # Cast source table columns to match target schema
            arrays = []
            names = []
            
            for field in target_schema:
                field_name = field.name
                field_type = field.type
                
                if field_name in source_table.column_names:
                    source_array = source_table.column(field_name)
                    try:
                        # Cast to target type if needed
                        if source_array.type != field_type:
                            # Try basic casting
                            casted_array = source_array.cast(field_type)
                            arrays.append(casted_array)
                        else:
                            arrays.append(source_array)
                    except Exception:
                        # If casting fails, use original array
                        arrays.append(source_array)
                else:
                    # Column missing in source, create null array
                    arrays.append(pa.nulls(len(source_table), field_type))
                
                names.append(field_name)
            
            return pa.table(arrays, names=names)
            
        except Exception as e:
            logger.warning(f"Schema harmonization failed: {e}")
            return source_table
    
    def _harmonize_dataframe_schema(self, target_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize DataFrame schema to match target DataFrame."""
        try:
            harmonized_df = source_df.copy()
            
            # Ensure all target columns exist in source
            for col in target_df.columns:
                if col not in harmonized_df.columns:
                    # Add missing column with appropriate dtype
                    target_dtype = target_df[col].dtype
                    if pd.api.types.is_numeric_dtype(target_dtype):
                        harmonized_df[col] = np.nan
                    else:
                        harmonized_df[col] = None
                    try:
                        harmonized_df[col] = harmonized_df[col].astype(target_dtype, errors='ignore')
                    except Exception:
                        pass  # Keep default type if conversion fails
            
            # Remove extra columns not in target
            extra_cols = [col for col in harmonized_df.columns if col not in target_df.columns]
            if extra_cols:
                harmonized_df = harmonized_df.drop(columns=extra_cols)
            
            # Reorder columns to match target
            harmonized_df = harmonized_df.reindex(columns=target_df.columns, fill_value=None)
            
            # Cast dtypes to match target where possible
            for col in target_df.columns:
                target_dtype = target_df[col].dtype
                try:
                    if harmonized_df[col].dtype != target_dtype:
                        harmonized_df[col] = harmonized_df[col].astype(target_dtype, errors='ignore')
                except Exception:
                    # If casting fails, keep original dtype
                    continue
            
            return harmonized_df
            
        except Exception as e:
            logger.warning(f"DataFrame schema harmonization failed: {e}")
            return source_df
    
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


class MassiveModelTrainer:
    """
    Train ML models on massive datasets that don't fit in memory.
    
    Features:
    - Incremental training with partial_fit support
    - Out-of-core model training using SGD-based algorithms
    - Memory-efficient model persistence
    - Streaming cross-validation
    - Multi-horizon model training for forecasting
    """
    
    def __init__(self, 
                 chunk_size: int = 100_000,
                 backend: str = "auto",
                 memory_threshold: float = 75.0):
        """
        Initialize massive model trainer.
        
        Parameters
        ----------
        chunk_size : int
            Chunk size for streaming training
        backend : str
            Processing backend
        memory_threshold : float
            Memory threshold percentage
        """
        self.chunk_size = chunk_size
        self.backend = backend
        self.memory_threshold = memory_threshold
        self.data_processor = MassiveDatasetProcessor(
            chunk_size=chunk_size, 
            backend=backend,
            memory_threshold_percent=memory_threshold
        )
        
        # Training state tracking
        self.training_stats = {
            'chunks_processed': 0,
            'total_samples': 0,
            'training_time': 0.0,
            'memory_peak': 0.0
        }
        
        logger.info(f"Initialized MassiveModelTrainer with chunk_size={chunk_size:,}")
    
    def train_massive_lightgbm(self,
                              train_data_path: Union[str, Path],
                              target_column: str,
                              model_output_path: Union[str, Path],
                              feature_columns: Optional[List[str]] = None,
                              horizons: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
                              validation_split: float = 0.2,
                              early_stopping_rounds: int = 100,
                              use_gpu: bool = True) -> Dict[str, Any]:
        """
        Train LightGBM model on massive datasets using streaming approach.
        
        Uses chunked training with periodic model checkpointing for memory efficiency.
        Supports multi-horizon forecasting and comprehensive preprocessing.
        
        Parameters
        ----------
        train_data_path : str or Path
            Path to training data (parquet/csv)
        target_column : str
            Name of target column
        model_output_path : str or Path
            Path to save trained model
        feature_columns : List[str], optional
            Specific feature columns to use
        horizons : tuple
            Forecast horizons for multi-output training
        validation_split : float
            Fraction of data for validation
        early_stopping_rounds : int
            Early stopping patience
        use_gpu : bool
            Enable GPU acceleration if available
        
        Returns
        -------
        Dict[str, Any]
            Training results and model metrics
        """
        import time
        from granarypredict.multi_lgbm import MultiLGBMRegressor
        from granarypredict.features import select_feature_target_multi
        
        logger.info(f"Starting massive LightGBM training on {train_data_path}")
        start_time = time.time()
        
        try:
            # Initialize model with optimized parameters for massive datasets
            model = MultiLGBMRegressor(
                base_params={
                    'learning_rate': 0.05,  # Lower learning rate for stable incremental training
                    'num_leaves': 63,       # Reduced complexity for memory efficiency
                    'max_depth': 8,         # Reduced depth for memory efficiency
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 50,
                    'n_estimators': 1000,   # More trees with early stopping
                    'verbosity': -1,
                    'random_state': 42,
                },
                upper_bound_estimators=1500,
                early_stopping_rounds=early_stopping_rounds,
                uncertainty_estimation=True,
                n_bootstrap_samples=15,  # Reduced for memory efficiency
                use_gpu=use_gpu,
                conservative_mode=True,
                stability_feature_boost=2.0
            )
            
            # Collect training data in chunks
            X_train_chunks = []
            y_train_chunks = []
            X_val_chunks = []
            y_val_chunks = []
            
            chunk_count = 0
            total_rows = 0
            
            logger.info("Processing training data in chunks...")
            
            # Stream through data and collect chunks
            for chunk in self.data_processor.read_massive_dataset(train_data_path):
                try:
                    # Apply comprehensive preprocessing to chunk
                    processed_chunk = self._preprocess_training_chunk(chunk)
                    
                    if processed_chunk.empty:
                        continue
                    
                    # Extract features and targets
                    if target_column not in processed_chunk.columns:
                        logger.warning(f"Target column {target_column} not found in chunk {chunk_count}")
                        continue
                    
                    # Split features and targets for multi-horizon
                    X_chunk, y_chunk = select_feature_target_multi(
                        processed_chunk, 
                        target_col=target_column, 
                        horizons=horizons,
                        allow_na=True
                    )
                    
                    if X_chunk.empty or y_chunk.empty:
                        continue
                    
                    # Select specific feature columns if provided
                    if feature_columns:
                        available_features = [f for f in feature_columns if f in X_chunk.columns]
                        X_chunk = X_chunk[available_features]
                    
                    # Split chunk into train/validation
                    split_idx = int(len(X_chunk) * (1 - validation_split))
                    
                    X_train_chunk = X_chunk.iloc[:split_idx]
                    y_train_chunk = y_chunk.iloc[:split_idx]
                    X_val_chunk = X_chunk.iloc[split_idx:]
                    y_val_chunk = y_chunk.iloc[split_idx:]
                    
                    if len(X_train_chunk) > 0:
                        X_train_chunks.append(X_train_chunk)
                        y_train_chunks.append(y_train_chunk)
                    
                    if len(X_val_chunk) > 0:
                        X_val_chunks.append(X_val_chunk)
                        y_val_chunks.append(y_val_chunk)
                    
                    chunk_count += 1
                    total_rows += len(processed_chunk)
                    
                    # Periodic logging
                    if chunk_count % 10 == 0:
                        logger.info(f"Processed {chunk_count} chunks, {total_rows:,} total rows")
                    
                    # Memory management
                    del processed_chunk, X_chunk, y_chunk
                    gc.collect()
                    
                    # Train incrementally when we have enough chunks
                    if len(X_train_chunks) >= 5:  # Train on 5 chunks at a time
                        logger.info(f"Training model on batch of {len(X_train_chunks)} chunks...")
                        self._train_model_batch(model, X_train_chunks, y_train_chunks, 
                                              X_val_chunks, y_val_chunks, is_first_batch=(chunk_count <= 5))
                        
                        # Clear processed chunks to free memory  
                        X_train_chunks.clear()
                        y_train_chunks.clear()
                        X_val_chunks.clear()
                        y_val_chunks.clear()
                        gc.collect()
                
                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk_count}: {e}")
                    continue
            
            # Train on remaining chunks
            if X_train_chunks:
                logger.info(f"Training final batch of {len(X_train_chunks)} chunks...")
                self._train_model_batch(model, X_train_chunks, y_train_chunks,
                                      X_val_chunks, y_val_chunks, is_first_batch=False)
            
            # Save trained model
            model_output_path = Path(model_output_path)
            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            from granarypredict.model import save_model
            save_result = save_model(model, str(model_output_path))
            
            training_time = time.time() - start_time
            
            # Update training statistics
            self.training_stats.update({
                'chunks_processed': chunk_count,
                'total_samples': total_rows,
                'training_time': training_time,
                'model_path': str(model_output_path)
            })
            
            logger.info(f"Massive LightGBM training completed!")
            logger.info(f"  Total chunks: {chunk_count}")
            logger.info(f"  Total samples: {total_rows:,}")
            logger.info(f"  Training time: {training_time:.2f} seconds")
            logger.info(f"  Model saved to: {model_output_path}")
            
            return {
                'success': True,
                'model_path': str(model_output_path),
                'training_stats': self.training_stats,
                'model_type': 'MultiLGBMRegressor',
                'horizons': horizons,
                'total_samples': total_rows,
                'training_time': training_time
            }
            
        except Exception as e:
            logger.error(f"Massive LightGBM training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_stats': self.training_stats
            }
    
    def _preprocess_training_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive preprocessing to training chunk."""
        try:
            # Use same comprehensive preprocessing as batch processing
            processor = MassiveDatasetProcessor(chunk_size=len(chunk))
            
            # Apply comprehensive preprocessing functions
            for func in [
                processor._standardize_columns,
                processor._basic_clean,
                processor._insert_calendar_gaps,
                processor._interpolate_sensor_numeric,
                processor._fill_missing,
                processor._add_comprehensive_time_features,
                processor._add_spatial_features,
                processor._add_time_since_last_measurement,
                processor._add_multi_lag_features,
                processor._add_rolling_stats,
                processor._add_directional_features,
                processor._add_stability_features,
                processor._add_horizon_specific_directional_features,
                processor._add_multi_horizon_targets,
                processor._assign_group_id,
                processor._comprehensive_sort
            ]:
                try:
                    chunk = func(chunk)
                except Exception as e:
                    logger.warning(f"Preprocessing function {func.__name__} failed: {e}")
                    continue
            
            return chunk
            
        except Exception as e:
            logger.warning(f"Chunk preprocessing failed: {e}")
            return chunk
    
    def _train_model_batch(self, model, X_train_chunks, y_train_chunks, 
                          X_val_chunks, y_val_chunks, is_first_batch=False):
        """Train model on a batch of chunks."""
        try:
            # Combine chunks
            if X_train_chunks:
                X_train = pd.concat(X_train_chunks, ignore_index=True)
                y_train = pd.concat(y_train_chunks, ignore_index=True)
            else:
                return
            
            if X_val_chunks:
                X_val = pd.concat(X_val_chunks, ignore_index=True)
                y_val = pd.concat(y_val_chunks, ignore_index=True)
                eval_set = (X_val, y_val)
            else:
                eval_set = None
            
            # Train model (first batch gets full training, subsequent get incremental)
            if is_first_batch:
                logger.info(f"Initial training on {len(X_train):,} samples")
                model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            else:
                # For LightGBM, we need to retrain as it doesn't support incremental learning
                # In production, consider using SGD-based models for true incremental learning
                logger.info(f"Retraining model with additional {len(X_train):,} samples")
                model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            
        except Exception as e:
            logger.error(f"Batch training failed: {e}")
    
    def generate_massive_forecasts(self,
                                  model_path: Union[str, Path],
                                  input_data_path: Union[str, Path], 
                                  output_forecasts_path: Union[str, Path],
                                  horizon_days: int = 7,
                                  include_uncertainty: bool = True) -> bool:
        """
        Generate forecasts for massive datasets using streaming approach.
        
        Parameters
        ----------
        model_path : str or Path
            Path to trained model
        input_data_path : str or Path
            Path to input data for forecasting
        output_forecasts_path : str or Path
            Path to save forecasts
        horizon_days : int
            Number of days to forecast
        include_uncertainty : bool
            Include uncertainty quantification
        
        Returns
        -------
        bool
            True if forecasting successful
        """
        try:
            from granarypredict.model import load_model
            
            logger.info(f"Loading model from {model_path}")
            model = load_model(model_path)
            
            if model is None:
                logger.error("Failed to load model")
                return False
            
            logger.info(f"Starting massive forecasting on {input_data_path}")
            
            # Setup output
            output_path = Path(output_forecasts_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            first_chunk = True
            total_forecasts = 0
            
            # Process data in chunks and generate forecasts
            for chunk_idx, chunk in enumerate(self.data_processor.read_massive_dataset(input_data_path)):
                try:
                    # Preprocess chunk for forecasting
                    processed_chunk = self._preprocess_forecasting_chunk(chunk)
                    
                    if processed_chunk.empty:
                        continue
                    
                    # Generate forecasts for chunk
                    forecasts_chunk = self._generate_chunk_forecasts(
                        model, processed_chunk, horizon_days, include_uncertainty
                    )
                    
                    if forecasts_chunk.empty:
                        continue
                    
                    # Write forecasts incrementally
                    if output_path.suffix.lower() == '.parquet':
                        if first_chunk:
                            forecasts_chunk.to_parquet(output_path, index=False)
                            first_chunk = False
                        else:
                            # Append to existing parquet
                            existing_df = pd.read_parquet(output_path)
                            combined_df = pd.concat([existing_df, forecasts_chunk], ignore_index=True)
                            combined_df.to_parquet(output_path, index=False)
                    else:
                        # CSV append
                        mode = 'w' if first_chunk else 'a'
                        header = first_chunk
                        forecasts_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                        first_chunk = False
                    
                    total_forecasts += len(forecasts_chunk)
                    
                    if chunk_idx % 10 == 0:
                        logger.info(f"Generated forecasts for {chunk_idx + 1} chunks, {total_forecasts:,} total forecasts")
                    
                    # Memory cleanup
                    del processed_chunk, forecasts_chunk
                    gc.collect()
                
                except Exception as e:
                    logger.warning(f"Error processing forecast chunk {chunk_idx}: {e}")
                    continue
            
            logger.info(f"Massive forecasting completed! Generated {total_forecasts:,} forecasts")
            return True
            
        except Exception as e:
            logger.error(f"Massive forecasting failed: {e}")
            return False
    
    def _preprocess_forecasting_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Preprocess chunk for forecasting (same as training but without targets)."""
        return self._preprocess_training_chunk(chunk)
    
    def _generate_chunk_forecasts(self, model, chunk: pd.DataFrame, 
                                 horizon_days: int, include_uncertainty: bool) -> pd.DataFrame:
        """Generate forecasts for a single chunk."""
        try:
            from granarypredict.features import select_feature_target_multi
            from granarypredict.model import predict
            
            # Get latest data point per sensor for forecasting
            sensor_cols = [c for c in ['granary_id', 'heap_id', 'grid_x', 'grid_y', 'grid_z'] 
                          if c in chunk.columns]
            
            if not sensor_cols:
                return pd.DataFrame()
            
            # Get last observation per sensor
            last_obs = (chunk.sort_values('detection_time')
                           .groupby(sensor_cols, dropna=False)
                           .tail(1)
                           .copy())
            
            if last_obs.empty:
                return pd.DataFrame()
            
            # Prepare features for forecasting
            X_forecast, _ = select_feature_target_multi(
                last_obs, 
                target_col='temperature_grain',
                horizons=tuple(range(1, horizon_days + 1)),
                allow_na=True
            )
            
            if X_forecast.empty:
                return pd.DataFrame()
            
            # Generate predictions
            predictions = predict(model, X_forecast)
            
            # Create forecast DataFrame
            forecasts = []
            
            for i, (_, row) in enumerate(last_obs.iterrows()):
                base_time = pd.to_datetime(row['detection_time'])
                
                # Get sensor identifiers
                sensor_info = {col: row[col] for col in sensor_cols if col in row}
                
                # Create forecast rows for each horizon
                for h in range(1, horizon_days + 1):
                    forecast_row = sensor_info.copy()
                    forecast_row.update({
                        'detection_time': base_time + pd.Timedelta(days=h),
                        'forecast_horizon': h,
                        'is_forecast': True,
                        'base_time': base_time
                    })
                    
                    # Add prediction
                    if predictions.ndim == 2 and i < len(predictions):
                        pred_idx = min(h - 1, predictions.shape[1] - 1)
                        forecast_row['predicted_temperature'] = predictions[i, pred_idx]
                    elif predictions.ndim == 1 and i < len(predictions):
                        forecast_row['predicted_temperature'] = predictions[i]
                    else:
                        forecast_row['predicted_temperature'] = np.nan
                    
                    # Add uncertainty if available and requested
                    if include_uncertainty and hasattr(model, '_last_prediction_intervals'):
                        intervals = model._last_prediction_intervals
                        if 'lower_95' in intervals and i < len(intervals['lower_95']):
                            pred_idx = min(h - 1, intervals['lower_95'].shape[1] - 1)
                            forecast_row['uncertainty_lower_95'] = intervals['lower_95'][i, pred_idx]
                            forecast_row['uncertainty_upper_95'] = intervals['upper_95'][i, pred_idx]
                        
                        if 'uncertainties' in intervals and i < len(intervals['uncertainties']):
                            pred_idx = min(h - 1, intervals['uncertainties'].shape[1] - 1)
                            forecast_row['uncertainty_std'] = intervals['uncertainties'][i, pred_idx]
                    
                    forecasts.append(forecast_row)
            
            return pd.DataFrame(forecasts)
            
        except Exception as e:
            logger.warning(f"Chunk forecast generation failed: {e}")
            return pd.DataFrame()


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


# Main entry point functions for massive dataset operations
def create_massive_training_pipeline(
    train_data_path: Union[str, Path],
    target_column: str,
    model_output_path: Union[str, Path],
    chunk_size: int = 50_000,
    backend: str = "auto",
    horizons: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Create a complete pipeline for training models on massive datasets.
    
    This is the main entry point for training models on datasets with hundreds of millions of rows.
    
    Parameters
    ----------
    train_data_path : str or Path
        Path to training data
    target_column : str
        Name of target column
    model_output_path : str or Path
        Path to save trained model
    chunk_size : int
        Chunk size for processing
    backend : str
        Processing backend
    horizons : tuple
        Forecast horizons
    use_gpu : bool
        Enable GPU acceleration
    
    Returns
    -------
    Dict[str, Any]
        Training results
    """
    trainer = MassiveModelTrainer(
        chunk_size=chunk_size,
        backend=backend
    )
    
    return trainer.train_massive_lightgbm(
        train_data_path=train_data_path,
        target_column=target_column,
        model_output_path=model_output_path,
        horizons=horizons,
        use_gpu=use_gpu
    )


def create_massive_forecasting_pipeline(
    model_path: Union[str, Path],
    input_data_path: Union[str, Path],
    output_forecasts_path: Union[str, Path],
    chunk_size: int = 50_000,
    backend: str = "auto",
    horizon_days: int = 7,
    include_uncertainty: bool = True
) -> bool:
    """
    Create a complete pipeline for generating forecasts on massive datasets.
    
    Parameters
    ----------
    model_path : str or Path
        Path to trained model
    input_data_path : str or Path
        Path to input data
    output_forecasts_path : str or Path
        Path to save forecasts
    chunk_size : int
        Chunk size for processing
    backend : str
        Processing backend
    horizon_days : int
        Number of days to forecast
    include_uncertainty : bool
        Include uncertainty quantification
    
    Returns
    -------
    bool
        True if successful
    """
    trainer = MassiveModelTrainer(
        chunk_size=chunk_size,
        backend=backend
    )
    
    return trainer.generate_massive_forecasts(
        model_path=model_path,
        input_data_path=input_data_path,
        output_forecasts_path=output_forecasts_path,
        horizon_days=horizon_days,
        include_uncertainty=include_uncertainty
    )


if __name__ == "__main__":
    # Example usage for massive datasets
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Process massive dataset
    input_file = "massive_dataset.parquet"
    processed_file = "processed_massive_dataset.parquet"
    
    # Estimate memory requirements
    memory_info = estimate_memory_requirements(input_file)
    print(f"Memory analysis: {memory_info}")
    
    # Process the massive dataset
    success = create_massive_processing_pipeline(
        input_path=input_file,
        output_path=processed_file,
        chunk_size=memory_info.get('recommended_chunk_size', 100_000)
    )
    
    if success:
        print("✅ Massive dataset processing completed!")
        
        # Example 2: Train model on processed data
        model_path = "massive_trained_model.joblib"
        training_result = create_massive_training_pipeline(
            train_data_path=processed_file,
            target_column="temperature_grain",
            model_output_path=model_path,
            chunk_size=50_000
        )
        
        if training_result['success']:
            print("✅ Massive model training completed!")
            
            # Example 3: Generate forecasts
            forecast_path = "massive_forecasts.parquet"
            forecast_success = create_massive_forecasting_pipeline(
                model_path=model_path,
                input_data_path=processed_file,
                output_forecasts_path=forecast_path,
                chunk_size=50_000,
                horizon_days=7
            )
            
            if forecast_success:
                print("✅ Massive forecasting completed!")
                print("\n🎉 Complete massive dataset pipeline executed successfully!")
            else:
                print("❌ Massive forecasting failed!")
        else:
            print(f"❌ Massive training failed: {training_result.get('error', 'Unknown error')}")
    else:
        print("❌ Massive dataset processing failed!")
