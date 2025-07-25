#!/usr/bin/env python3
"""
SiloFlow Automated Pipeline Service Startup Script - OPTIMIZED VERSION
=====================================================================

This script starts the automated HTTP service for grain temperature forecasting
with performance optimizations including:
- Process pool management
- Memory monitoring
- Resource optimization
- Performance tuning
"""

import os
import sys
import logging
import multiprocessing
from pathlib import Path

# Add the service directory to Python path so it's recognized as a package
sys.path.insert(0, str(Path(__file__).parent))

# Configure enhanced logging with performance monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available with performance libraries."""
    missing_deps = []
    performance_deps = []
    
    # Core dependencies
    core_deps = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'lightgbm': 'Gradient boosting',
        'joblib': 'Model serialization'
    }
    
    # Performance dependencies (optional but recommended)
    perf_deps = {
        'psutil': 'System monitoring',
        'uvloop': 'High-performance event loop',
        'orjson': 'Fast JSON serialization'
    }
    
    # Check core dependencies
    for dep, desc in core_deps.items():
        try:
            __import__(dep)
            logger.info(f"[OK] {desc}: {dep}")
        except ImportError:
            missing_deps.append(f"{dep} ({desc})")
    
    # Check performance dependencies
    for dep, desc in perf_deps.items():
        try:
            __import__(dep)
            logger.info(f"[PERF] {desc}: {dep}")
            performance_deps.append(dep)
        except ImportError:
            logger.warning(f"[WARNING] Optional performance dependency missing: {dep} ({desc})")
    
    if missing_deps:
        logger.error(f"[ERROR] Missing required dependencies: {', '.join(missing_deps)}")
        return False
    
    logger.info(f"[INFO] Performance optimizations available: {len(performance_deps)}/{len(perf_deps)}")
    return True

def check_system_resources():
    """Check and log system resources for optimization planning."""
    try:
        import psutil
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"[MEMORY] System Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        
        # CPU info
        cpu_count = multiprocessing.cpu_count()
        logger.info(f"[CPU] CPU Cores: {cpu_count} logical cores")
        
        # Disk info
        disk = psutil.disk_usage('/')
        logger.info(f"[DISK] Disk Space: {disk.free / (1024**3):.1f} GB free of {disk.total / (1024**3):.1f} GB")
        
        # Optimization recommendations
        if memory.total < 4 * (1024**3):  # Less than 4GB
            logger.warning("[WARNING] Low memory detected - enabling conservative processing mode")
            os.environ['SILOFLOW_CONSERVATIVE_MODE'] = '1'
        
        if cpu_count < 4:
            logger.warning("[WARNING] Limited CPU cores - reducing parallel processing")
            os.environ['SILOFLOW_MAX_WORKERS'] = '2'
        else:
            os.environ['SILOFLOW_MAX_WORKERS'] = str(min(8, cpu_count))
        
        return True
        
    except ImportError:
        logger.warning("[WARNING] psutil not available - cannot optimize based on system resources")
        return True

def setup_performance_environment():
    """Set up environment variables for optimal performance."""
    
    # Set optimized environment variables
    env_vars = {
        'PYTHONUNBUFFERED': '1',  # Ensure immediate stdout/stderr output
        'PYTHONDONTWRITEBYTECODE': '1',  # Don't create .pyc files
        'OMP_NUM_THREADS': '1',  # Prevent numpy/scipy thread conflicts
        'NUMEXPR_MAX_THREADS': str(min(4, multiprocessing.cpu_count())),  # Limit numexpr threads
        'OPENBLAS_NUM_THREADS': '1',  # Prevent OpenBLAS thread conflicts
        'MKL_NUM_THREADS': '1',  # Prevent MKL thread conflicts
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"[CONFIG] Set {key}={value}")
    
    # Configure uvicorn for better performance
    try:
        import uvloop
        logger.info("[PERF] uvloop available - will use high-performance event loop")
    except ImportError:
        logger.info("[INFO] Using standard asyncio event loop")

def check_directories():
    """Check and create required directories with optimized structure."""
    required_dirs = [
        "models",
        "data/processed", 
        "data/granaries",
        "data/batch",
        "data/forecasts",
        "temp_uploads",
        "logs/performance",  # New: Performance logs
        "cache/models",      # New: Model cache
        "cache/data"         # New: Data cache
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"[DIR] Directory ready: {dir_path}")

def main():
    """Main startup function with performance optimizations."""
    logger.info("[STARTUP] Starting SiloFlow Automated Pipeline Service (Optimized)")
    
    # Performance setup
    setup_performance_environment()
    
    # System resource check
    check_system_resources()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("[ERROR] Service startup failed due to missing dependencies")
        sys.exit(1)
    
    # Check and create directories
    check_directories()
    
    # Import and start the service
    try:
        # Change to the service directory
        service_dir = Path(__file__).parent
        os.chdir(str(service_dir))
        
        # Add current directory to path for absolute imports
        if str(service_dir) not in sys.path:
            sys.path.insert(0, str(service_dir))
        
        # Import main module directly (not as relative import)
        import main
        import uvicorn
        
        logger.info("[OK] Service initialized successfully")
        logger.info("[INFO] Starting HTTP server on http://0.0.0.0:8000")
        logger.info("[INFO] API documentation available at http://localhost:8000/docs")
        logger.info("[INFO] Health check available at http://localhost:8000/health")
        
        # Determine optimal server configuration
        workers = 1  # Use single worker with internal process pooling
        
        # Configure uvicorn with performance optimizations
        uvicorn_config = {
            "app": main.app,
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info",
            "access_log": True,
            "workers": workers,
            "loop": "auto",  # Auto-select best event loop
            "http": "auto",  # Auto-select best HTTP implementation
            "ws_ping_interval": 20,
            "ws_ping_timeout": 20,
            "timeout_keep_alive": 5,
        }
        
        # Use uvloop if available for better performance
        try:
            import uvloop
            uvicorn_config["loop"] = "uvloop"
            logger.info("[PERF] Using uvloop for enhanced performance")
        except ImportError:
            logger.info("[INFO] Using standard asyncio event loop")
        
        # Start the server with optimized configuration
        uvicorn.run(**uvicorn_config)
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to start service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set multiprocessing start method for better performance on Windows
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    
    main() 