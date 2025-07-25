"""
Optimized Automated Granary Processor with Advanced Performance Features
========================================================================

This module provides enhanced processing capabilities with:
- Memory pool management
- Concurrent processing with resource limits
- Smart caching strategies
- Performance monitoring
- Graceful degradation under load
"""

import asyncio
import gc
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union
import weakref

import pandas as pd

# Optional high-performance imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryPool:
    """Memory pool for efficient DataFrame management."""
    
    def __init__(self, max_size: int = 100):
        self.pool = []
        self.max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get_dataframe(self, shape: tuple) -> pd.DataFrame:
        """Get a pre-allocated DataFrame from pool."""
        async with self._lock:
            # Try to find a suitable DataFrame in pool
            for i, df in enumerate(self.pool):
                if df.shape[0] >= shape[0] and df.shape[1] >= shape[1]:
                    return self.pool.pop(i).iloc[:shape[0], :shape[1]].copy()
            
            # Create new DataFrame if none suitable found
            return pd.DataFrame()
    
    async def return_dataframe(self, df: pd.DataFrame):
        """Return a DataFrame to the pool."""
        async with self._lock:
            if len(self.pool) < self.max_size:
                # Clear the DataFrame but keep its structure
                df.iloc[:, :] = None
                self.pool.append(df)


class ResourceMonitor:
    """Monitor system resources and adapt processing accordingly."""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # Percent
        self.memory_threshold = 75.0  # Percent
        self.last_check = 0
        self.check_interval = 5  # seconds
    
    def should_throttle(self) -> bool:
        """Check if processing should be throttled due to resource constraints."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return False
        
        self.last_check = current_time
        
        if not PSUTIL_AVAILABLE:
            return False
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
                return True
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.cpu_threshold:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
            return False
    
    def get_optimal_workers(self) -> int:
        """Get optimal number of workers based on current system state."""
        if not PSUTIL_AVAILABLE:
            return min(4, multiprocessing.cpu_count())
        
        try:
            memory = psutil.virtual_memory()
            cpu_count = multiprocessing.cpu_count()
            
            # Reduce workers if memory is high
            if memory.percent > 70:
                return max(1, cpu_count // 4)
            elif memory.percent > 50:
                return max(2, cpu_count // 2)
            else:
                return min(8, cpu_count)
                
        except Exception:
            return 4


class OptimizedGranaryProcessor:
    """
    High-performance granary processor with advanced optimizations.
    
    Features:
    - Memory pooling for DataFrame reuse
    - Intelligent resource monitoring
    - Concurrent processing with backpressure
    - Smart caching with LRU eviction
    - Performance metrics collection
    - Graceful degradation under load
    """
    
    def __init__(self, 
                 max_concurrent_granaries: int = 4,
                 enable_memory_pool: bool = True,
                 enable_model_cache: bool = True):
        
        self.max_concurrent = max_concurrent_granaries
        self.memory_pool = MemoryPool() if enable_memory_pool else None
        self.resource_monitor = ResourceMonitor()
        self.enable_model_cache = enable_model_cache
        
        # Performance tracking
        self.metrics = {
            'total_processed': 0,
            'total_processing_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_optimizations': 0
        }
        
        # Semaphore for controlling concurrent granary processing
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_granaries)
        
        # Model cache with weak references to allow garbage collection
        self._model_cache = weakref.WeakValueDictionary() if enable_model_cache else None
        
        # Thread pools for different types of operations
        self._io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="IO")
        self._cpu_executor = ProcessPoolExecutor(max_workers=min(4, multiprocessing.cpu_count()))
        
        logger.info(f"OptimizedGranaryProcessor initialized:")
        logger.info(f"  Max concurrent granaries: {max_concurrent_granaries}")
        logger.info(f"  Memory pool: {'enabled' if enable_memory_pool else 'disabled'}")
        logger.info(f"  Model cache: {'enabled' if enable_model_cache else 'disabled'}")
    
    async def process_all_granaries_optimized(self, csv_path: str) -> Dict:
        """
        Process all granaries with advanced optimizations.
        
        PROPER WORKFLOW:
        1. Ingest raw file and sort into granary-specific files 
        2. Process each granary with existing models
        3. Generate forecasts
        
        This method provides significant performance improvements over the standard processor:
        - Concurrent granary processing
        - Smart resource management
        - Memory pooling
        - Adaptive worker scaling
        """
        start_time = time.time()
        
        try:
            logger.info(f"[OPT-PREFLIGHT] Starting optimized processor preflight checks...")
            
            # Pre-flight checks
            if self.resource_monitor.should_throttle():
                logger.warning("[OPT-PREFLIGHT] System under load - reducing processing intensity")
                self.max_concurrent = max(1, self.max_concurrent // 2)
            else:
                logger.info(f"[OPT-PREFLIGHT] System resources OK, using {self.max_concurrent} concurrent workers")
            
            # STEP 1: Process raw CSV to identify and sort granaries
            logger.info("[OPT-STEP1] Starting raw CSV processing to identify granaries...")
            logger.info(f"[OPT-STEP1] Input file: {csv_path}")
            
            from automated_processor import AutomatedGranaryProcessor
            temp_processor = AutomatedGranaryProcessor()
            
            logger.info("[OPT-STEP1] Calling automated processor for raw CSV ingestion...")
            change_info = await temp_processor.process_raw_csv(csv_path)
            new_granaries = change_info.get('granaries', [])
            
            logger.info(f"[OPT-STEP1-RESULT] Raw CSV processing complete")
            logger.info(f"[OPT-STEP1-RESULT] Granaries identified: {len(new_granaries)}")
            logger.info(f"[OPT-STEP1-RESULT] Granary list: {new_granaries}")
            
            if not new_granaries:
                logger.warning("[OPT-STEP1-WARN] No granaries identified from uploaded file")
                return {
                    'success': True,
                    'message': 'No granaries found in uploaded file',
                    'granaries_processed': 0,
                    'successful_granaries': 0,
                    'forecasts': {}
                }
            
            logger.info(f"[OPT-STEP2] Starting optimized processing of {len(new_granaries)} granaries")
            logger.info(f"[OPT-STEP2] Granary names: {new_granaries}")
            
            # STEP 2: Process granaries in optimized batches
            results = await self._process_granaries_concurrent_with_ingestion(new_granaries)
            
            # Update metrics
            processing_time = time.time() - start_time
            total_granaries = len(new_granaries)
            self.metrics['total_processed'] += total_granaries
            self.metrics['total_processing_time'] += int(processing_time)
            
            logger.info(f"[OPT-COMPLETE] Processing completed successfully")
            logger.info(f"[OPT-METRICS] Total time: {processing_time:.2f}s")
            logger.info(f"[OPT-METRICS] Granaries processed: {total_granaries}")
            logger.info(f"[OPT-METRICS] Successful granaries: {results.get('successful_granaries', 0)}")
            logger.info(f"[OPT-METRICS] Errors: {len(results.get('errors', []))}")
            
            if results.get('errors'):
                logger.warning(f"[OPT-ERRORS] Processing errors encountered:")
                for error in results.get('errors', [])[:5]:  # Log first 5 errors
                    logger.warning(f"[OPT-ERROR-DETAIL] {error}")
            
            # Add performance information to results
            results['performance'] = {
                'processing_time_seconds': round(processing_time, 2),
                'granaries_per_second': round(total_granaries / processing_time, 2) if processing_time > 0 else 0,
                'optimization_version': '2.1.0',
                'metrics': self.metrics.copy()
            }
            
            logger.info(f"[OPT-SUCCESS] Optimized processing completed in {processing_time:.2f}s")
            if processing_time > 0:
                logger.info(f"[OPT-PERFORMANCE] Performance: {total_granaries / processing_time:.2f} granaries/second")
            else:
                logger.info(f"[OPT-PERFORMANCE] Performance: instant processing")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Optimized processing failed: {e}")
            raise

    async def _analyze_dataset(self, csv_path: str) -> Dict:
        """Quickly analyze dataset to plan optimal processing strategy."""
        try:
            # Use efficient sampling for large files
            if Path(csv_path).suffix.lower() == '.parquet':
                sample_df = pd.read_parquet(csv_path, columns=['granary_id'])
            else:
                sample_df = pd.read_csv(csv_path, usecols=['granary_id'], nrows=10000)
                
            granaries = sample_df['granary_id'].unique().tolist()
            file_size = Path(csv_path).stat().st_size
            
            return {
                'granaries': granaries,
                'file_size_mb': file_size / (1024 * 1024),
                'estimated_complexity': len(granaries) * (file_size / (1024 * 1024))
            }
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            return {'granaries': ['default'], 'file_size_mb': 0, 'estimated_complexity': 1}

    async def _process_granaries_concurrent_with_ingestion(self, granary_list: List[str]) -> Dict:
        """Process granaries concurrently after raw CSV ingestion."""
        logger.info(f"[BATCH-SETUP] Setting up concurrent processing for {len(granary_list)} granaries")
        
        # Adaptive batch sizing based on system resources
        optimal_workers = self.resource_monitor.get_optimal_workers()
        batch_size = min(optimal_workers, self.max_concurrent, len(granary_list))
        
        logger.info(f"[BATCH-SETUP] Optimal workers: {optimal_workers}, Max concurrent: {self.max_concurrent}")
        logger.info(f"[BATCH-SETUP] Using batch size: {batch_size}")
        
        # Process granaries in batches
        all_results = {
            "success": True,
            "forecasts": {},
            "granaries_processed": 0,
            "successful_granaries": 0,
            "errors": []
        }
        
        total_batches = (len(granary_list) + batch_size - 1) // batch_size
        logger.info(f"[BATCH-PLAN] Will process {len(granary_list)} granaries in {total_batches} batches")
        
        for i in range(0, len(granary_list), batch_size):
            batch = granary_list[i:i + batch_size]
            batch_num = i//batch_size + 1
            logger.info(f"[BATCH-{batch_num}] Starting batch {batch_num}/{total_batches}: {batch}")
            
            # Check resources before each batch
            if self.resource_monitor.should_throttle():
                logger.warning(f"[BATCH-{batch_num}] System overloaded - pausing 5 seconds before processing")
                await asyncio.sleep(5)
            
            # Process batch concurrently
            logger.info(f"[BATCH-{batch_num}] Processing batch concurrently...")
            batch_results = await self._process_batch_with_ingestion(batch)
            
            logger.info(f"[BATCH-{batch_num}] Batch completed - Success: {batch_results.get('successful_granaries', 0)}/{len(batch)}")
            if batch_results.get('errors'):
                logger.warning(f"[BATCH-{batch_num}] Batch errors: {len(batch_results.get('errors', []))}")
            
            # Merge results
            all_results["forecasts"].update(batch_results.get("forecasts", {}))
            all_results["granaries_processed"] += len(batch)
            all_results["successful_granaries"] += batch_results.get("successful_granaries", 0)
            all_results["errors"].extend(batch_results.get("errors", []))
            
            # Force garbage collection between batches
            if self.memory_pool:
                self.metrics['memory_optimizations'] += 1
            gc.collect()
        
        logger.info(f"[BATCH-COMPLETE] All batches completed")
        logger.info(f"[BATCH-SUMMARY] Total processed: {all_results['granaries_processed']}")
        logger.info(f"[BATCH-SUMMARY] Total successful: {all_results['successful_granaries']}")
        logger.info(f"[BATCH-SUMMARY] Total errors: {len(all_results['errors'])}")
        
        return all_results

    async def _process_batch_with_ingestion(self, granary_batch: List[str]) -> Dict:
        """Process a batch of granaries after ingestion."""
        tasks = []
        
        for granary_name in granary_batch:
            # Use semaphore to limit concurrent processing
            task = asyncio.create_task(
                self._process_single_granary_after_ingestion(granary_name)
            )
            tasks.append(task)
        
        # Wait for all tasks in the batch to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        forecasts = {}
        successful_granaries = 0
        errors = []
        
        for i, result in enumerate(batch_results):
            granary_name = granary_batch[i]
            if isinstance(result, Exception):
                errors.append(f"Granary {granary_name}: {str(result)}")
            elif isinstance(result, dict) and result.get("success", False):
                successful_granaries += 1
                forecasts[granary_name] = result
            else:
                errors.append(f"Granary {granary_name}: Processing failed")
        
        return {
            "forecasts": forecasts,
            "successful_granaries": successful_granaries,
            "errors": errors
        }

    async def _process_single_granary_after_ingestion(self, granary_name: str) -> Dict:
        """Process a single granary after ingestion using silo-level forecasting."""
        start_time = time.time()
        
        try:
            logger.info(f"[GRANARY-{granary_name}] Starting silo-level granary processing")
            
            from utils.data_paths import data_paths
            
            # Get the granary-specific parquet file (should exist after ingestion)
            granaries_dir = data_paths.get_granaries_dir()
            granary_csv = granaries_dir / f"{granary_name}.parquet"
            
            logger.info(f"[GRANARY-{granary_name}] Looking for granary file: {granary_csv}")
            
            if not granary_csv.exists():
                logger.error(f"[GRANARY-{granary_name}] Granary file not found: {granary_csv}")
                return {
                    "success": False,
                    "granary_name": granary_name,
                    "error": f"Granary file not found: {granary_csv}"
                }
            
            logger.info(f"[GRANARY-{granary_name}] Granary file found, size: {granary_csv.stat().st_size} bytes")
            
            # Check if model exists for this granary
            models_dir = data_paths.get_models_dir()
            model_path = models_dir / f"{granary_name}_forecast_model.joblib"
            has_model = model_path.exists()
            
            logger.info(f"[GRANARY-{granary_name}] Model check - Path: {model_path}")
            logger.info(f"[GRANARY-{granary_name}] Model exists: {has_model}")
            
            if not has_model:
                logger.warning(f"[GRANARY-{granary_name}] No model found - skipping forecasting for this granary")
                return {
                    "success": True,
                    "granary_name": granary_name,
                    "processing_time": time.time() - start_time,
                    "forecasts": {
                        "csv_content": "",
                        "total_records": 0,
                        "csv_filename": f"{granary_name}_forecast.csv",
                        "summary": {"method": "skipped_no_model", "reason": "No trained model available"}
                    },
                    "processing": {
                        "optimization_applied": True,
                        "method": "skipped_no_model",
                        "skip_train": True,
                        "model_available": False
                    }
                }
            
            # Load granary data to identify silos
            logger.info(f"[GRANARY-{granary_name}] Loading granary data to identify silos...")
            import pandas as pd
            df = pd.read_parquet(granary_csv)
            logger.info(f"[GRANARY-{granary_name}] Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Identify unique silos in this granary
            silo_columns = ['storepointId', 'silo_id', 'heap_id']  # Possible silo identifiers
            silo_col = None
            for col in silo_columns:
                if col in df.columns:
                    silo_col = col
                    break
            
            if silo_col is None:
                logger.warning(f"[GRANARY-{granary_name}] No silo identifier found, using default")
                unique_silos = ['default']
            else:
                unique_silos = df[silo_col].unique().tolist()
                logger.info(f"[GRANARY-{granary_name}] Found {len(unique_silos)} silos using column '{silo_col}': {unique_silos[:10]}...")
            
            # Process each silo for forecasting
            logger.info(f"[GRANARY-{granary_name}] Starting silo-level forecasting...")
            
            # Import the actual pipeline processor
            from granary_pipeline import run_complete_pipeline
            
            logger.info(f"[GRANARY-{granary_name}] Calling run_complete_pipeline with silo information...")
            
            # Run the actual complete pipeline (it will handle preprocessing and forecasting)
            results = run_complete_pipeline(
                granary_csv=str(granary_csv),
                granary_name=granary_name,
                skip_train=True,  # Model already exists
                force_retrain=False,
                changed_silos=unique_silos  # Pass all silos for processing
            )
            
            processing_time = time.time() - start_time
            logger.info(f"[GRANARY-{granary_name}] Pipeline completed in {processing_time:.2f}s")
            logger.info(f"[GRANARY-{granary_name}] Pipeline success: {results.get('success', False)}")
            
            # Convert the pipeline results to the expected format
            if results.get("success", False):
                logger.info(f"[GRANARY-{granary_name}] Processing successful, extracting forecast data...")
                
                # Extract forecast data if available
                forecasts_dir = data_paths.get_forecasts_dir()
                forecast_file = forecasts_dir / f"{granary_name}_forecast.csv"
                
                logger.info(f"[GRANARY-{granary_name}] Looking for forecast file: {forecast_file}")
                
                csv_content = ""
                total_records = 0
                
                if forecast_file.exists():
                    logger.info(f"[GRANARY-{granary_name}] Forecast file found, reading content...")
                    with open(forecast_file, 'r', encoding='utf-8') as f:
                        csv_content = f.read()
                    # Count records (excluding header)
                    total_records = len(csv_content.strip().split('\n')) - 1 if csv_content else 0
                    logger.info(f"[GRANARY-{granary_name}] Forecast records: {total_records}")
                    logger.info(f"[GRANARY-{granary_name}] Silos processed: {len(unique_silos)}")
                else:
                    logger.warning(f"[GRANARY-{granary_name}] Forecast file not found at: {forecast_file}")
                
                return {
                    "success": True,
                    "granary_name": granary_name,
                    "processing_time": processing_time,
                    "forecasts": {
                        "csv_content": csv_content,
                        "total_records": total_records,
                        "csv_filename": f"{granary_name}_forecast.csv",
                        "summary": {
                            "method": "silo_level_forecasting", 
                            "silos_processed": len(unique_silos),
                            "silo_identifier": silo_col or "default"
                        }
                    },
                    "processing": {
                        "optimization_applied": True,
                        "method": "silo_level_processing",
                        "skip_train": True,
                        "model_available": True,
                        "silos_identified": len(unique_silos)
                    }
                }
            else:
                logger.error(f"[GRANARY-{granary_name}] Pipeline processing failed")
                logger.error(f"[GRANARY-{granary_name}] Pipeline errors: {results.get('errors', [])}")
                return {
                    "success": False,
                    "granary_name": granary_name,
                    "error": results.get("errors", ["Pipeline processing failed"])
                }
                
        except Exception as e:
            logger.error(f"[GRANARY-{granary_name}] Exception during processing: {str(e)}")
            logger.exception(f"[GRANARY-{granary_name}] Full exception traceback:")
            return {
                "success": False,
                "granary_name": granary_name,
                "error": str(e)
            }

    @lru_cache(maxsize=128)
    def _get_cached_model(self, model_path: str):
        """Get model from cache with LRU eviction."""
        if not JOBLIB_AVAILABLE:
            return None
            
        try:
            if self.enable_model_cache and self._model_cache is not None and model_path in self._model_cache:
                self.metrics['cache_hits'] += 1
                return self._model_cache[model_path]
            
            # Load model and cache it
            model = joblib.load(model_path)
            if self.enable_model_cache and self._model_cache is not None:
                self._model_cache[model_path] = model
                self.metrics['cache_misses'] += 1
            
            return model
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return None

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        return {
            **self.metrics,
            'cache_hit_ratio': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'avg_processing_time': self.metrics['total_processing_time'] / max(1, self.metrics['total_processed'])
        }

    async def cleanup(self):
        """Clean up resources."""
        if self._io_executor:
            self._io_executor.shutdown(wait=True)
        if self._cpu_executor:
            self._cpu_executor.shutdown(wait=True)
        
        logger.info("OptimizedGranaryProcessor cleaned up")


# Factory function to create optimized processor instance
def create_optimized_processor(**kwargs) -> OptimizedGranaryProcessor:
    """Create an optimized processor with intelligent defaults."""
    
    # Auto-detect optimal settings based on system resources
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        cpu_count = multiprocessing.cpu_count()
        
        # Scale concurrent granaries based on available memory
        if memory.total > 16 * (1024**3):  # 16GB+
            max_concurrent = min(8, cpu_count)
        elif memory.total > 8 * (1024**3):  # 8GB+
            max_concurrent = min(4, cpu_count // 2)
        else:  # < 8GB
            max_concurrent = min(2, cpu_count // 4)
        
        kwargs.setdefault('max_concurrent_granaries', max_concurrent)
    
    return OptimizedGranaryProcessor(**kwargs)
