#!/usr/bin/env python3
"""
Advanced Memory Management System for SiloFlow
==============================================

Provides intelligent memory management with:
- Proactive memory monitoring and cleanup
- Dynamic chunk size adjustment based on available memory
- GPU memory management for CUDA operations
- Emergency fallback strategies
- Memory pressure detection and mitigation
"""

import gc
import logging
import time
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
import psutil
import pandas as pd

logger = logging.getLogger(__name__)

class AdvancedMemoryManager:
    """
    Intelligent memory manager that prevents OOM errors and optimizes performance.
    """
    
    def __init__(self, 
                 target_memory_usage: float = 70.0,  # Target memory usage percentage
                 emergency_threshold: float = 85.0,  # Emergency cleanup threshold
                 critical_threshold: float = 95.0,   # Critical memory threshold
                 min_free_gb: float = 1.0):          # Minimum free memory in GB
        
        self.target_memory_usage = target_memory_usage
        self.emergency_threshold = emergency_threshold
        self.critical_threshold = critical_threshold
        self.min_free_gb = min_free_gb
        
        # Memory tracking
        self.baseline_memory = self._get_memory_info()
        self.peak_memory = self.baseline_memory.copy()
        self.cleanup_count = 0
        self.emergency_cleanups = 0
        
        # GPU memory tracking
        self.gpu_available = self._check_gpu_availability()
        self.gpu_memory_info = self._get_gpu_memory_info() if self.gpu_available else None
        
        logger.info(f"🧠 Memory Manager initialized:")
        logger.info(f"   Target usage: {target_memory_usage}%")
        logger.info(f"   Emergency threshold: {emergency_threshold}%")
        logger.info(f"   Critical threshold: {critical_threshold}%")
        logger.info(f"   Min free memory: {min_free_gb:.1f} GB")
        if self.gpu_available:
            logger.info(f"   GPU memory detected: {self.gpu_memory_info['total_gb']:.1f} GB")
    
    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available (compatibility property)."""
        return self.gpu_available
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information (compatibility method)."""
        if self.gpu_available and self.gpu_memory_info:
            return self.gpu_memory_info.copy()
        else:
            return {'available': False, 'total_gb': 0, 'used_gb': 0, 'free_gb': 0}
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache (compatibility method)."""
        if self.gpu_available:
            self._clear_gpu_memory()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get comprehensive memory information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for memory management."""
        try:
            import cupy
            return cupy.cuda.runtime.getDeviceCount() > 0
        except:
            try:
                import torch
                return torch.cuda.is_available()
            except:
                return False
    
    def _get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """Get GPU memory information."""
        try:
            import cupy
            mempool = cupy.get_default_memory_pool()
            total_bytes = cupy.cuda.runtime.memGetInfo()[1]
            used_bytes = mempool.used_bytes()
            
            return {
                'total_gb': total_bytes / (1024**3),
                'used_gb': used_bytes / (1024**3),
                'free_gb': (total_bytes - used_bytes) / (1024**3),
                'percent': (used_bytes / total_bytes) * 100
            }
        except:
            try:
                import torch
                if torch.cuda.is_available():
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    allocated_bytes = torch.cuda.memory_allocated(0)
                    
                    return {
                        'total_gb': total_bytes / (1024**3),
                        'used_gb': allocated_bytes / (1024**3),
                        'free_gb': (total_bytes - allocated_bytes) / (1024**3),
                        'percent': (allocated_bytes / total_bytes) * 100
                    }
            except:
                pass
        return None
    
    def check_memory_health(self) -> Dict[str, Any]:
        """Comprehensive memory health check."""
        current_memory = self._get_memory_info()
        gpu_memory = self._get_gpu_memory_info() if self.gpu_available else None
        
        # Update peak memory tracking
        if current_memory['percent'] > self.peak_memory['percent']:
            self.peak_memory = current_memory.copy()
        
        health_status = {
            'status': 'healthy',
            'current_memory': current_memory,
            'gpu_memory': gpu_memory,
            'recommendations': [],
            'actions_needed': []
        }
        
        # Analyze memory status
        if current_memory['percent'] >= self.critical_threshold:
            health_status['status'] = 'critical'
            health_status['actions_needed'].append('immediate_cleanup')
            health_status['recommendations'].append('Emergency memory cleanup required')
            
        elif current_memory['percent'] >= self.emergency_threshold:
            health_status['status'] = 'warning'
            health_status['actions_needed'].append('proactive_cleanup')
            health_status['recommendations'].append('Proactive memory cleanup recommended')
            
        elif current_memory['available_gb'] < self.min_free_gb:
            health_status['status'] = 'warning'
            health_status['actions_needed'].append('increase_available')
            health_status['recommendations'].append(f'Free memory below {self.min_free_gb} GB threshold')
        
        # GPU memory analysis
        if gpu_memory and gpu_memory['percent'] > 80:
            health_status['actions_needed'].append('gpu_cleanup')
            health_status['recommendations'].append('GPU memory usage high')
        
        return health_status
    
    def calculate_safe_chunk_size(self, 
                                 data_shape: tuple, 
                                 dtype_size: int = 8,
                                 safety_factor: float = 0.3) -> int:
        """
        Calculate safe chunk size based on available memory and data characteristics.
        
        Args:
            data_shape: Shape of the data (rows, cols)
            dtype_size: Size of data type in bytes (8 for float64)
            safety_factor: Safety margin (0.3 = use only 30% of available memory)
        """
        memory_info = self._get_memory_info()
        available_bytes = memory_info['available_gb'] * (1024**3) * safety_factor
        
        # Calculate memory per row
        cols = data_shape[1] if len(data_shape) > 1 else 1
        memory_per_row = cols * dtype_size
        
        # Calculate safe chunk size
        safe_chunk_size = int(available_bytes / memory_per_row)
        
        # Apply reasonable bounds
        min_chunk = 1_000
        max_chunk = 500_000
        safe_chunk_size = max(min_chunk, min(max_chunk, safe_chunk_size))
        
        logger.info(f"💾 Calculated safe chunk size: {safe_chunk_size:,} rows")
        logger.info(f"   Available memory: {memory_info['available_gb']:.1f} GB")
        logger.info(f"   Memory per row: {memory_per_row} bytes")
        logger.info(f"   Safety factor: {safety_factor}")
        
        return safe_chunk_size
    
    def proactive_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Perform proactive memory cleanup.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        logger.info(f"🧹 Starting {'aggressive' if aggressive else 'standard'} memory cleanup...")
        
        initial_memory = self._get_memory_info()
        cleanup_results = {
            'initial_memory': initial_memory,
            'actions_performed': [],
            'memory_freed_gb': 0,
            'success': False
        }
        
        # Standard cleanup
        try:
            # 1. Garbage Collection
            collected = gc.collect()
            cleanup_results['actions_performed'].append(f'gc_collect: {collected} objects')
            
            # 2. Clear pandas caches
            try:
                import pandas as pd
                if hasattr(pd.core.common, '_decons_group_index'):
                    pd.core.common._decons_group_index.cache_clear()
                cleanup_results['actions_performed'].append('pandas_cache_cleared')
            except:
                pass
            
            # 3. GPU memory cleanup
            if self.gpu_available:
                try:
                    import cupy
                    mempool = cupy.get_default_memory_pool()
                    pinned_mempool = cupy.get_default_pinned_memory_pool()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    cleanup_results['actions_performed'].append('gpu_memory_freed')
                except:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            cleanup_results['actions_performed'].append('torch_cache_cleared')
                    except:
                        pass
            
            # Aggressive cleanup
            if aggressive:
                # 4. Force full garbage collection
                for i in range(3):
                    gc.collect()
                cleanup_results['actions_performed'].append('aggressive_gc_x3')
                
                # 5. Clear all possible caches
                try:
                    import sys
                    if hasattr(sys, '_clear_type_cache'):
                        sys._clear_type_cache()
                    cleanup_results['actions_performed'].append('type_cache_cleared')
                except:
                    pass
                
                # 6. Wait for memory to stabilize
                time.sleep(2)
                cleanup_results['actions_performed'].append('memory_stabilization_wait')
            
            # Calculate results
            final_memory = self._get_memory_info()
            cleanup_results['final_memory'] = final_memory
            cleanup_results['memory_freed_gb'] = initial_memory['used_gb'] - final_memory['used_gb']
            cleanup_results['success'] = True
            
            self.cleanup_count += 1
            if aggressive:
                self.emergency_cleanups += 1
            
            logger.info(f"✅ Memory cleanup completed:")
            logger.info(f"   Memory usage: {initial_memory['percent']:.1f}% → {final_memory['percent']:.1f}%")
            logger.info(f"   Memory freed: {cleanup_results['memory_freed_gb']:.2f} GB")
            logger.info(f"   Actions: {', '.join(cleanup_results['actions_performed'])}")
            
        except Exception as e:
            logger.error(f"❌ Memory cleanup failed: {e}")
            cleanup_results['error'] = str(e)
        
        return cleanup_results
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if not self.gpu_available:
            return
        
        try:
            import cupy
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            logger.debug("✅ CuPy GPU memory cleared")
        except:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("✅ PyTorch GPU cache cleared")
            except:
                logger.debug("⚠️ No GPU memory clearing available")
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """
        Context manager for memory-safe operations.
        
        Usage:
            with memory_manager.memory_context("model_training"):
                # Memory-monitored operation
                model.fit(X, y)
        """
        logger.info(f"🎯 Starting memory-monitored operation: {operation_name}")
        
        # Pre-operation memory check
        initial_health = self.check_memory_health()
        
        # Proactive cleanup if needed
        if 'proactive_cleanup' in initial_health['actions_needed']:
            self.proactive_cleanup(aggressive=False)
        elif 'immediate_cleanup' in initial_health['actions_needed']:
            self.proactive_cleanup(aggressive=True)
        
        try:
            yield self
            
        except MemoryError as e:
            logger.error(f"💥 Memory error in {operation_name}: {e}")
            # Emergency cleanup
            logger.info("🚨 Performing emergency memory cleanup...")
            self.proactive_cleanup(aggressive=True)
            raise
            
        except Exception as e:
            logger.error(f"❌ Error in {operation_name}: {e}")
            raise
            
        finally:
            # Post-operation cleanup and reporting
            final_health = self.check_memory_health()
            logger.info(f"✅ Completed memory-monitored operation: {operation_name}")
            logger.info(f"   Final memory usage: {final_health['current_memory']['percent']:.1f}%")
            
            # Cleanup if memory is still high
            if final_health['current_memory']['percent'] > self.target_memory_usage:
                logger.info("🧹 Post-operation cleanup...")
                self.proactive_cleanup(aggressive=False)
    
    def get_memory_recommendations(self, data_size_gb: float) -> Dict[str, Any]:
        """
        Get memory optimization recommendations for processing data of given size.
        """
        current_memory = self._get_memory_info()
        
        recommendations = {
            'processing_strategy': 'unknown',
            'recommended_chunk_size': 50_000,
            'use_gpu': False,
            'parallel_workers': 1,
            'memory_actions': [],
            'warnings': []
        }
        
        # Determine processing strategy
        if data_size_gb < current_memory['available_gb'] * 0.5:
            recommendations['processing_strategy'] = 'in_memory'
            recommendations['parallel_workers'] = min(4, psutil.cpu_count())
        elif data_size_gb < current_memory['available_gb'] * 2:
            recommendations['processing_strategy'] = 'streaming'
            recommendations['parallel_workers'] = 2
        else:
            recommendations['processing_strategy'] = 'chunked_streaming'
            recommendations['parallel_workers'] = 1
            recommendations['warnings'].append('Data size significantly exceeds available memory')
        
        # GPU recommendations
        if self.gpu_available and self.gpu_memory_info:
            if data_size_gb < self.gpu_memory_info['free_gb'] * 0.5:
                recommendations['use_gpu'] = True
            else:
                recommendations['warnings'].append('Data may not fit in GPU memory')
        
        # Memory actions
        if current_memory['percent'] > self.emergency_threshold:
            recommendations['memory_actions'].append('immediate_cleanup')
        elif current_memory['available_gb'] < self.min_free_gb:
            recommendations['memory_actions'].append('free_more_memory')
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        current_memory = self._get_memory_info()
        
        return {
            'baseline_memory': self.baseline_memory,
            'current_memory': current_memory,
            'peak_memory': self.peak_memory,
            'cleanup_count': self.cleanup_count,
            'emergency_cleanups': self.emergency_cleanups,
            'memory_growth_gb': current_memory['used_gb'] - self.baseline_memory['used_gb'],
            'gpu_available': self.gpu_available,
            'gpu_memory': self.gpu_memory_info
        }


# Factory function for easy integration
def create_memory_manager(conservative: bool = True) -> AdvancedMemoryManager:
    """
    Create a memory manager with appropriate settings.
    
    Args:
        conservative: Whether to use conservative memory settings
    """
    if conservative:
        return AdvancedMemoryManager(
            target_memory_usage=60.0,
            emergency_threshold=75.0,
            critical_threshold=85.0,
            min_free_gb=2.0
        )
    else:
        return AdvancedMemoryManager(
            target_memory_usage=70.0,
            emergency_threshold=85.0,
            critical_threshold=95.0,
            min_free_gb=1.0
        )
