from __future__ import annotations

"""Shared core objects for the service layer (singleton instances, common utilities).
OPTIMIZED VERSION with performance enhancements and resource pooling.
"""

import sys
from pathlib import Path
import logging

# Ensure granarypredict package is importable before importing processor
root_dir = Path(__file__).resolve().parent.parent
granarypredict_dir = root_dir / "granarypredict"
if str(granarypredict_dir) not in sys.path:
    sys.path.insert(0, str(granarypredict_dir))

# Local imports to avoid circular dependencies
from automated_processor import AutomatedGranaryProcessor  # noqa: E402
from optimized_processor import create_optimized_processor  # noqa: E402
from utils.data_paths import data_paths  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure all data directories exist
data_paths.ensure_directories()

# Create both standard and optimized processor instances
processor = AutomatedGranaryProcessor()
optimized_processor = create_optimized_processor()

# Hybrid processor that delegates to optimized version when possible
class HybridProcessor:
    """Hybrid processor that uses optimized version when available, falls back to standard."""
    
    def __init__(self):
        self.standard = processor
        self.optimized = optimized_processor
        self.use_optimized = True  # Default to optimized processing
        # Add granaries_dir attribute from the underlying processors
        self.granaries_dir = getattr(self.standard, 'granaries_dir', None) or data_paths.get_granaries_dir()
        
    async def process_all_granaries(self, csv_path: str, use_optimization: bool = True) -> dict:
        """Process all granaries with optional optimization."""
        if use_optimization and self.use_optimized:
            try:
                logger.info("ROCKET Using optimized processing pipeline")
                return await self.optimized.process_all_granaries_optimized(csv_path)
            except Exception as e:
                logger.warning(f"Optimized processing failed, falling back to standard: {e}")
                self.use_optimized = False  # Disable for future requests
        
        # Fall back to standard processing
        logger.info("MEMO Using standard processing pipeline")
        return await self.standard.process_all_granaries(csv_path)
    
    async def process_granary(self, granary_name: str, changed_silos=None):
        """Process single granary - always use standard processor for now."""
        return await self.standard.process_granary(granary_name, changed_silos)
    
    def cleanup_temp_files(self, *args, **kwargs):
        """Cleanup temp files."""
        return self.standard.cleanup_temp_files(*args, **kwargs)
    
    def get_performance_metrics(self):
        """Get performance metrics from optimized processor."""
        if self.use_optimized:
            return self.optimized.get_performance_metrics()
        return {"status": "standard_processor", "metrics": "not_available"}
    
    def __getattr__(self, name):
        """Delegate missing attributes to the standard processor."""
        if hasattr(self.standard, name):
            return getattr(self.standard, name)
        elif hasattr(self.optimized, name):
            return getattr(self.optimized, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# Create hybrid processor instance
hybrid_processor = HybridProcessor()

logger.info("Core processors initialized:")
logger.info(f"  Standard processor: {processor}")
logger.info(f"  Optimized processor: {optimized_processor}")
logger.info(f"  Hybrid processor: {hybrid_processor}")
logger.info(f"  Data paths: {data_paths.get_data_summary()}")

# Export the hybrid processor as the default
processor = hybrid_processor 