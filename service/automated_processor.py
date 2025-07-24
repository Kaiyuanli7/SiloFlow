import asyncio
import atexit
import gc
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd

# Optional import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

# Optimized import handling - add granarypredict to path only once
GRANARYPREDICT_DIR = Path(__file__).parent.parent / "granarypredict"
if str(GRANARYPREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(GRANARYPREDICT_DIR))

# Configure logging early
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class AutomatedGranaryProcessor:
    def __init__(self):
        # Use centralized data path manager
        from utils.data_paths import data_paths
        from utils.silo_filtering import get_existing_silo_files
        self.data_paths = data_paths
        
        # Get standardized directory paths
        self.models_dir = self.data_paths.get_models_dir()
        self.processed_dir = self.data_paths.get_processed_dir()
        self.granaries_dir = self.data_paths.get_granaries_dir()
        self.forecasts_dir = self.data_paths.get_forecasts_dir()
        self.temp_dir = self.data_paths.get_temp_dir()
        self.uploads_dir = self.data_paths.get_uploads_dir()
        
        # New optimized paths
        self.simple_retrieval_dir = self.data_paths.get_simple_retrieval_dir()
        self.streaming_output_dir = self.data_paths.get_streaming_output_dir()
        self.batch_output_dir = self.data_paths.get_batch_output_dir()
        
        # Memory management parameters (optimized)
        self.memory_threshold = 75  # Reduced from 80% for better performance
        self.max_retries = 3
        self.retry_delay = 2  # Reduced from 5 seconds for faster recovery
        
        # Track resources for cleanup
        self._temp_files = []
        self._active_processes = []
        self._cached_data = {}  # Add caching for frequently accessed data
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Initialized with optimized data paths:")
        logger.info(f"  Models: {self.models_dir}")
        logger.info(f"  Processed: {self.processed_dir}")
        logger.info(f"  Granaries: {self.granaries_dir}")
        logger.info(f"  Forecasts: {self.forecasts_dir}")
        logger.info(f"  Simple Retrieval: {self.simple_retrieval_dir}")
        logger.info(f"  Streaming: {self.streaming_output_dir}")
        logger.info(f"  Batch: {self.batch_output_dir}")
    
    def _get_existing_silo_files(self, granary_name: str) -> set:
        """Get set of existing silo files for a granary using centralized utility"""
        from utils.silo_filtering import get_existing_silo_files
        return get_existing_silo_files(granary_name)
        
    def _cleanup(self):
        """Cleanup resources on exit."""
        try:
            # Remove temporary files
            for temp_file in self._temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.info(f"Received signal {signum}, cleaning up...")
        self._cleanup()
        sys.exit(0)
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        if not PSUTIL_AVAILABLE:
            return True  # Skip memory check if psutil not available
            
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > self.memory_threshold:
                logger.warning(f"Memory usage high: {memory_percent:.1f}% (threshold: {self.memory_threshold}%)")
                return False
            return True
                
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return False
    
    def _force_memory_cleanup(self):
        """Force memory cleanup when usage is high."""
        logger.info("Forcing memory cleanup...")
        
        # Get initial memory state (if available)
        if PSUTIL_AVAILABLE:
            initial_memory = psutil.virtual_memory()
            initial_percent = initial_memory.percent
        else:
            initial_percent = 0
        
        # Strategy 1: Standard garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Strategy 2: Clear any cached data
        if hasattr(self, '_cached_data') and self._cached_data:
            cache_size = len(self._cached_data)
            self._cached_data.clear()
            logger.info(f"Cleared cached data ({cache_size} items)")
        
        # Strategy 3: Clear pandas cache
        try:
            import pandas as pd
            if hasattr(pd.util, 'hash_pandas_object') and hasattr(pd.util.hash_pandas_object, 'cache_clear'):
                pd.util.hash_pandas_object.cache_clear()
                logger.info("Cleared pandas cache")
        except Exception:
            pass
        
        # Strategy 4: Clear any other caches
        try:
            # Clear garbage collection unreachable objects
            gc.collect(2)  # Full collection including generation 2
            logger.info("Performed full garbage collection")
        except Exception:
            pass
        
        # Strategy 5: Wait for memory to stabilize
        time.sleep(3)
        
        # Check final memory state
        final_memory = psutil.virtual_memory()
        final_percent = final_memory.percent
        freed_mb = (initial_memory.available - final_memory.available) / (1024 * 1024)
        
        logger.info(f"Memory cleanup completed: {initial_percent:.1f}% -> {final_percent:.1f}% (freed {freed_mb:.1f} MB)")
        
        # If still high, wait longer
        if not self._check_memory_usage():
            logger.warning("Memory still high after cleanup, waiting for recovery...")
            time.sleep(15)
            
            # Try one more aggressive cleanup
            gc.collect()
            time.sleep(5)
    
    def _safe_file_operation(self, operation, *args, **kwargs):
        """Safely perform file operations with retries and memory management."""
        for attempt in range(self.max_retries):
            try:
                # Check memory before operation
                if not self._check_memory_usage():
                    self._force_memory_cleanup()
                
                result = operation(*args, **kwargs)
                return result
                
            except Exception as e:
                logger.error(f"File operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    self._force_memory_cleanup()
                else:
                    raise
    
    async def process_raw_csv(self, csv_path: str) -> Dict:
        """Simplified ingestion for basic data processing"""
        try:
            from granarypredict import ingestion
            logger.info(f"Ingesting raw CSV: {csv_path}")
            
            # Simple ingestion without complex filtering
            def ingestion_operation():
                return ingestion.ingest_and_sort(csv_path)
            
            granary_status = self._safe_file_operation(ingestion_operation)
            
            # Convert to simple format
            if isinstance(granary_status, dict):
                granaries_with_new_data = [name for name, is_new in granary_status.items() if is_new]
            else:
                granaries_with_new_data = []
            
            logger.info(f"Granaries processed: {granaries_with_new_data}")
            
            return {
                'granaries': granaries_with_new_data,
                'silo_changes': {}  # Simplified - no silo tracking
            }
            
        except Exception as e:
            logger.error(f"Error ingesting CSV: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def process_granary(self, granary_name: str, changed_silos: Optional[List[str]] = None) -> Dict:
        """Simplified granary processing"""
        try:
            from granary_pipeline import run_complete_pipeline
            
            granary_csv = self.granaries_dir / f"{granary_name}.parquet"
            
            if not granary_csv.exists():
                return {
                    'success': False,
                    'errors': [f"Granary Parquet file not found: {granary_csv}"],
                    'steps_completed': []
                }
            
            # Check if model exists
            model_path = self.models_dir / f"{granary_name}_forecast_model.joblib"
            skip_train = model_path.exists()
            
            logger.info(f"Processing granary: {granary_name}")
            
            # Check memory before processing
            if not self._check_memory_usage():
                self._force_memory_cleanup()
            
            # Run simplified pipeline
            results = run_complete_pipeline(
                granary_csv=str(granary_csv),
                granary_name=granary_name,
                skip_train=skip_train,
                force_retrain=False,
                changed_silos=None  # Simplified - no silo filtering
            )
            
            # Force cleanup after processing
            gc.collect()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing granary {granary_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'errors': [str(e)],
                'steps_completed': []
            }
    
    # Removed generate_forecasts method - simplified pipeline doesn't need forecasting
    
    async def process_all_granaries(self, csv_path: str) -> Dict:
        """Simplified pipeline processing"""
        try:
            logger.info(f"Starting simplified pipeline for: {csv_path}")
            
            # 1. Simple ingestion
            change_info = await self.process_raw_csv(csv_path)
            new_granaries = change_info['granaries']
            
            if not new_granaries:
                logger.info("No new granaries to process")
                return {
                    'success': True,
                    'message': 'No new granaries found',
                    'granaries_processed': 0,
                    'forecasts': {}
                }
            
            # 2. Process each granary
            results = {}
            successful_granaries = 0
            
            for granary in new_granaries:
                logger.info(f"Processing granary: {granary}")
                
                # Check memory before processing each granary
                if not self._check_memory_usage():
                    self._force_memory_cleanup()
                
                # Process granary
                process_result = await self.process_granary(granary)
                
                if process_result.get('success', False):
                    successful_granaries += 1
                    
                results[granary] = {
                    "processing": process_result,
                    "forecasts": None  # Simplified - no forecasting
                }
                
                # Force cleanup after each granary
                gc.collect()
            
            return {
                'success': True,
                'granaries_processed': len(new_granaries),
                'successful_granaries': successful_granaries,
                'forecasts': results
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'errors': [str(e)],
                'granaries_processed': 0,
                'forecasts': {}
            }
    
    def cleanup_temp_files(self, temp_path: str):
        """Clean up temporary files with enhanced error handling"""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file {temp_path}: {e}") 