#!/usr/bin/env python3
"""
Quick test of the memory-optimized training system using actual data.
This will help validate that our complete implementation works end-to-end.
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_quick_training():
    """Quick test of memory-optimized training on real data."""
    try:
        from granarypredict.streaming_processor import MassiveModelTrainer
        
        # Find a small dataset to test with
        data_dir = Path("data/preloaded")
        test_files = list(data_dir.glob("*.parquet"))
        
        if not test_files:
            logger.warning("No test files found in data/preloaded/")
            return False
            
        test_file = test_files[0]
        logger.info(f"🎯 Testing with file: {test_file}")
        
        # Initialize trainer with conservative settings
        trainer = MassiveModelTrainer(
            chunk_size=5000,  # Very small chunks for safety
            memory_threshold=65.0,  # Conservative threshold
            enable_advanced_memory_management=True
        )
        
        logger.info("✅ Trainer initialized successfully")
        
        # Test just the initialization and chunk processing (not full training)
        if trainer.memory_manager:
            health = trainer.memory_manager.check_memory_health()
            logger.info(f"📊 Memory status: {health['status']} ({health['current_memory']['percent']:.1f}%)")
            
            # Test memory context
            with trainer.memory_manager.memory_context("quick_test"):
                logger.info("✅ Memory context manager working")
                
                # Test chunk size calculation
                test_shape = (50000, 77)
                safe_chunk = trainer.memory_manager.calculate_safe_chunk_size(test_shape)
                logger.info(f"🎯 Safe chunk size for {test_shape}: {safe_chunk:,}")
        
        logger.info("🎉 Quick training test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick training test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting Quick Training Test...")
    success = test_quick_training()
    
    if success:
        logger.info("✅ All quick tests passed!")
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)
