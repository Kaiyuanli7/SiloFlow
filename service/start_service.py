#!/usr/bin/env python3
"""
SiloFlow Automated Pipeline Service Startup Script
==================================================

This script starts the automated HTTP service for grain temperature forecasting.
"""

import os
import sys
import logging
from pathlib import Path

# Add the service directory to Python path so it's recognized as a package
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
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
    """Check if all required dependencies are available."""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        import lightgbm
        import joblib
        logger.info("All dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def check_directories():
    """Check and create required directories."""
    required_dirs = [
        "models",
        "data/processed", 
        "data/granaries",
        "temp_uploads"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {dir_path}")

def main():
    """Main startup function."""
    logger.info("Starting SiloFlow Automated Pipeline Service")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Service startup failed due to missing dependencies")
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
        
        logger.info("Service initialized successfully")
        logger.info("Starting HTTP server on http://0.0.0.0:8000")
        logger.info("API documentation available at http://localhost:8000/docs")
        
        # Start the server
        uvicorn.run(
            main.app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 