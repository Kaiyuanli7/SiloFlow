#!/usr/bin/env python3
"""Memory monitoring utilities for SiloFlow"""

import resource
import logging
import os

logger = logging.getLogger(__name__)

def log_memory_usage(prefix: str):
    """Log current memory usage with a descriptive prefix"""
    try:
        # Get memory usage in MB
        if os.name == 'posix':  # Linux and macOS
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        else:  # Windows
            import psutil
            process = psutil.Process(os.getpid())
            usage = process.memory_info().rss / (1024 * 1024)
            
        logger.debug(f"{prefix} - Memory usage: {usage:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not log memory usage: {e}") 