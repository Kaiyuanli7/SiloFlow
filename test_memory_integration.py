#!/usr/bin/env python3
"""
Test script to verify memory management integration
"""

import logging
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_manager_integration():
    """Test that memory manager integrates properly with streaming processor"""
    
    try:
        # Test memory manager import
        logger.info("Testing memory manager import...")
        from granarypredict.memory_manager import AdvancedMemoryManager
        
        # Create memory manager instance
        memory_manager = AdvancedMemoryManager()
        health = memory_manager.check_memory_health()
        logger.info(f"✅ Memory manager working - Current memory: {health['current_memory']['percent']:.1f}%")
        
        # Test streaming processor import
        logger.info("Testing streaming processor import...")
        from granarypredict.streaming_processor import MassiveModelTrainer
        
        # Create trainer with memory management
        trainer = MassiveModelTrainer(enable_advanced_memory_management=True)
        logger.info(f"✅ Trainer created - Memory manager available: {trainer.memory_manager is not None}")
        
        if trainer.memory_manager:
            # Test memory context manager
            with trainer.memory_manager.memory_context("test_context"):
                logger.info("✅ Memory context manager working")
            
            # Test memory health check
            health = trainer.memory_manager.check_memory_health()
            logger.info(f"✅ Memory health check: {health['current_memory']['percent']:.1f}% usage")
            
            # Test memory cleanup
            trainer.memory_manager.cleanup()
            logger.info("✅ Memory cleanup executed")
            
        logger.info("🎉 All memory management integration tests passed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_gpu_memory_management():
    """Test GPU memory management if available"""
    try:
        from granarypredict.memory_manager import AdvancedMemoryManager
        
        memory_manager = AdvancedMemoryManager()
        
        if memory_manager.has_gpu:
            gpu_info = memory_manager.get_gpu_memory_info()
            logger.info(f"✅ GPU detected: {gpu_info}")
        else:
            logger.info("ℹ️  No GPU detected - using CPU-only memory management")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU memory test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting memory management integration tests...")
    
    # Test basic integration
    integration_ok = test_memory_manager_integration()
    
    # Test GPU management
    gpu_ok = test_gpu_memory_management()
    
    if integration_ok and gpu_ok:
        logger.info("✅ All tests passed! Memory management is ready for use.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        sys.exit(1)
