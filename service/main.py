"""Central FastAPI application file – now a slim orchestrator that delegates
all route implementations to the modules in *service.routes*.

This replaces the previous monolithic version to improve maintainability.
OPTIMIZED VERSION with async support, connection pooling, and performance enhancements.
"""

from __future__ import annotations


import asyncio
import logging
import multiprocessing
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
# Configure logging early ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global resources for optimization
_worker_pool = None
_model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with resource pooling."""
    global _worker_pool
    
    # Startup: Initialize worker pool and model cache
    logger.info("[STARTUP] Starting SiloFlow Pipeline Service with optimizations...")
    
    # Create process pool for CPU-intensive tasks
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(cpu_count, 8)  # Cap at 8 to avoid resource exhaustion
    _worker_pool = multiprocessing.Pool(processes=max_workers)
    logger.info(f"[OK] Initialized process pool with {max_workers} workers")
    
    # Pre-warm model cache (if models exist)
    await _prewarm_models()
    
    # Start performance monitoring
    from performance_monitor import get_performance_monitor
    performance_monitor = get_performance_monitor()
    monitoring_task = asyncio.create_task(performance_monitor.start_monitoring())
    logger.info("[PERF] Performance monitoring started")

    # --- Automated daily data retrieval, sorting, and preprocessing ---
    from datetime import datetime
    from core import processor
    import sys
    sys.path.insert(0, str((Path(__file__).parent.parent / "scripts")))
    from scripts.simple_data_retrieval import SimpleDataRetriever, load_config

    async def automated_daily_job():
        while True:
            try:
                logger.info("[AUTO] Starting daily data retrieval and preprocessing...")
                # Load the CSV of silos with data
                csv_path = Path("data/simple_retrieval/granaries_silos_with_dates.csv")
                if not csv_path.exists():
                    logger.error(f"[AUTO] CSV file not found: {csv_path}")
                    await asyncio.sleep(86400)
                    continue
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                today = datetime.now().strftime("%Y-%m-%d")
                config = load_config()
                retriever = SimpleDataRetriever(config['database'])

                # Only process silos with data_available == 'Yes'
                silos = df[df['data_available'] == 'Yes']
                # Track which granaries were updated
                updated_granaries = set()
                from granarypredict.streaming_processor import MassiveDatasetProcessor
                for idx, row in silos.iterrows():
                    granary_name = row['granary_name']
                    silo_id = row['silo_id']
                    start_date = today
                    end_date = today
                    try:
                        logger.info(f"[AUTO] Retrieving data for {granary_name} / {silo_id} for {today}")
                        success = retriever.retrieve_and_save(granary_name, silo_id, start_date, end_date, output_dir="data/simple_retrieval")
                        if not success:
                            logger.warning(f"[AUTO] No data for {granary_name} / {silo_id} on {today}")
                            continue
                        # Find the output file
                        safe_granary_name = granary_name.encode('ascii', errors='ignore').decode('ascii') or 'granary'
                        safe_silo_id = silo_id.encode('ascii', errors='ignore').decode('ascii') or 'silo'
                        filename = f"{safe_granary_name}_{safe_silo_id}_{today}_to_{today}.parquet"
                        filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                        file_path = Path("data/simple_retrieval") / filename
                        if not file_path.exists():
                            logger.warning(f"[AUTO] Output file not found: {file_path}")
                            continue
                        # Log record count in raw file
                        try:
                            df_raw = pd.read_parquet(file_path)
                            logger.info(f"[AUTO] Raw file {file_path} contains {len(df_raw)} records.")
                        except Exception as read_e:
                            logger.warning(f"[AUTO] Could not read raw file {file_path}: {read_e}")
                        # Sort raw data using streaming processor
                        try:
                            sorted_file = Path("service/data/granaries") / f"{safe_granary_name}_sorted.parquet"
                            MassiveDatasetProcessor.sort_file(str(file_path), str(sorted_file), sort_by=['granary_name', 'detection_time'])
                            updated_granaries.add(granary_name)
                            logger.info(f"[AUTO] Sorted {silo_id} data into granary file for {granary_name}: {sorted_file}")
                        except Exception as sort_e:
                            logger.error(f"[AUTO] Sorting failed for {granary_name} / {silo_id}: {sort_e}")
                        # Check sorted granary file existence and log record count
                        if sorted_file.exists():
                            try:
                                df_granary = pd.read_parquet(sorted_file)
                                logger.info(f"[AUTO] Sorted granary file {sorted_file} now contains {len(df_granary)} records.")
                            except Exception as g_e:
                                logger.warning(f"[AUTO] Could not read sorted granary file {sorted_file}: {g_e}")
                        else:
                            logger.warning(f"[AUTO] Sorted granary file not found after sorting: {sorted_file}")
                        # Delete raw data file after sorting
                        try:
                            file_path.unlink()
                            logger.info(f"[AUTO] Deleted raw data file: {file_path}")
                        except Exception as del_e:
                            logger.warning(f"[AUTO] Could not delete raw data file {file_path}: {del_e}")
                    except Exception as e:
                        logger.error(f"[AUTO] Error processing {granary_name} / {silo_id}: {e}")
                # Preprocess each updated granary file
                for granary_name in updated_granaries:
                    try:
                        safe_granary_name = granary_name.encode('ascii', errors='ignore').decode('ascii') or 'granary'
                        sorted_file = Path("service/data/granaries") / f"{safe_granary_name}_sorted.parquet"
                        processed_file = Path("service/data/processed") / f"{safe_granary_name}_processed.parquet"
                        # Preprocess sorted granary file using streaming processor
                        from granarypredict.streaming_processor import MassiveDatasetProcessor
                        MassiveDatasetProcessor.process_file(str(sorted_file), str(processed_file))
                        logger.info(f"[AUTO] Preprocessed granary file for {granary_name}: {processed_file}")
                        # Check processed file existence and log record count
                        if processed_file.exists():
                            try:
                                df_processed = pd.read_parquet(processed_file)
                                logger.info(f"[AUTO] Processed file {processed_file} contains {len(df_processed)} records.")
                            except Exception as p_e:
                                logger.warning(f"[AUTO] Could not read processed file {processed_file}: {p_e}")
                        else:
                            logger.warning(f"[AUTO] Processed file not found after preprocessing: {processed_file}")
                    except Exception as e:
                        logger.error(f"[AUTO] Error preprocessing granary {granary_name}: {e}")
                logger.info("[AUTO] Daily data retrieval and preprocessing completed.")
            except Exception as e:
                logger.error(f"[AUTO] Automated job failed: {e}")
            # Sleep for 24 hours
            await asyncio.sleep(86400)
    asyncio.create_task(automated_daily_job())
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("[SHUTDOWN] Shutting down SiloFlow Pipeline Service...")
    
    # Stop performance monitoring
    performance_monitor.stop_monitoring()
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    logger.info("[OK] Performance monitoring stopped")
    
    if _worker_pool:
        _worker_pool.close()
        _worker_pool.join()
        logger.info("[OK] Process pool shut down gracefully")

async def _prewarm_models():
    """Pre-warm model cache for faster inference."""
    try:
        from pathlib import Path
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.joblib"))
            if model_files:
                logger.info(f"[PRELOAD] Pre-warming {len(model_files)} models...")
                # Pre-load models in background
                asyncio.create_task(_load_models_async(model_files))
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")

async def _load_models_async(model_files):
    """Asynchronously load models into cache."""
    import joblib
    for model_file in model_files[:5]:  # Limit to first 5 models
        try:
            model = joblib.load(model_file)
            _model_cache[str(model_file)] = model
            logger.info(f"[CACHE] Pre-loaded model: {model_file.name}")
        except Exception as e:
            logger.warning(f"Failed to pre-load {model_file}: {e}")

# ---------------------------------------------------------------------------
# FastAPI application setup with optimizations
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SiloFlow Automated Pipeline",
    description="Automated grain temperature forecasting service with performance optimizations",
    version="2.1.0-optimized",
    lifespan=lifespan
)

# Add compression middleware for response optimization
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS – optimized for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only needed methods
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# ---------------------------------------------------------------------------
# Include modular routers with optimizations
# ---------------------------------------------------------------------------
from routes import router as all_routes  # noqa: E402
from performance_monitor import get_performance_monitor, track_performance  # noqa: E402

app.include_router(all_routes)

# Initialize performance monitoring
performance_monitor = get_performance_monitor()

# Export worker pool and model cache for use in routes
def get_worker_pool():
    """Get the global worker pool for CPU-intensive tasks."""
    return _worker_pool

def get_model_cache():
    """Get the global model cache."""
    return _model_cache

# ---------------------------------------------------------------------------
# Enhanced health and monitoring endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health_check():
    """Enhanced health check with system information."""
    import psutil
    import sys
    
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "status": "healthy",
            "version": "2.1.0-optimized",
            "system": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "available_memory_gb": round(memory.available / (1024**3), 2),
                "worker_pool_active": _worker_pool is not None,
                "cached_models": len(_model_cache),
                "python_version": sys.version.split()[0],
            },
            "optimizations": {
                "compression": "gzip",
                "async_processing": True,
                "model_preloading": True,
                "process_pool": True,
                "performance_monitoring": True,
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "version": "2.1.0-optimized"
        }


@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Get current performance metrics."""
    return performance_monitor.get_current_metrics()


@app.get("/metrics/summary", tags=["monitoring"])
async def get_metrics_summary(window_minutes: int = 10):
    """Get performance summary for specified time window."""
    return performance_monitor.get_performance_summary(window_minutes)


@app.get("/metrics/export", tags=["monitoring"])
async def export_metrics():
    """Export performance metrics to file."""
    from pathlib import Path
    import time
    
    export_path = Path("logs/performance") / f"metrics_export_{int(time.time())}.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    performance_monitor.export_metrics(str(export_path))
    
    return {
        "status": "exported",
        "filepath": str(export_path),
        "timestamp": time.time()
    }


# Add middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Middleware to track all HTTP requests for performance monitoring."""
    import uuid
    import time
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    performance_monitor.record_request_start(request_id)
    
    try:
        response = await call_next(request)
        success = 200 <= response.status_code < 400
        performance_monitor.record_request_end(request_id, success)
        
        # Add performance headers
        processing_time = time.time() - start_time
        response.headers["X-Processing-Time"] = str(round(processing_time, 3))
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        performance_monitor.record_request_end(request_id, success=False)
        raise


# ---------------------------------------------------------------------------
# Entrypoint – ``python -m uvicorn service.main:app --reload``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        workers=1,  # Use 1 worker with internal process pool
        access_log=True,
        loop="asyncio"  # Use asyncio event loop
    ) 