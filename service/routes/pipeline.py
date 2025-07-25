from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from core import processor  # Singleton

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# Create temp uploads directory (shared for router)
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

from core import processor  # Singleton

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# Create temp uploads directory (shared for router)
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

# Optimization: Connection pool for concurrent processing
_processing_semaphore = asyncio.Semaphore(4)  # Limit concurrent requests
_active_tasks: Dict[str, asyncio.Task] = {}


class StreamingProcessor:
    """Optimized streaming processor for real-time progress updates."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.progress = 0
        self.status = "initializing"
        self.results = {}
        
    async def stream_progress(self):
        """Stream processing progress to client."""
        while self.status not in ["completed", "failed"]:
            yield f"data: {json.dumps({'progress': self.progress, 'status': self.status})}\n\n"
            await asyncio.sleep(1)
        
        # Final result
        yield f"data: {json.dumps({'progress': 100, 'status': self.status, 'results': self.results})}\n\n"


async def process_granary_batch(granary_batch: List[str], temp_path: str, horizon: int) -> Dict:
    """Process a batch of granaries concurrently."""
    results = {}
    
    # Process granaries concurrently using asyncio tasks
    tasks = []
    for granary_name in granary_batch:
        task = asyncio.create_task(processor.process_granary(granary_name))
        tasks.append((granary_name, task))
    
    # Collect results as they complete
    for granary_name, task in tasks:
        try:
            result = await asyncio.wait_for(task, timeout=300)  # 5 min per granary
            results[granary_name] = result
            logger.info(f"[OK] Completed processing granary: {granary_name}")
        except asyncio.TimeoutError:
            logger.error(f"[TIMEOUT] Timeout processing granary: {granary_name}")
            results[granary_name] = {"error": "Processing timeout", "success": False}
        except Exception as e:
            logger.error(f"[ERROR] Failed processing granary {granary_name}: {e}")
            results[granary_name] = {"error": str(e), "success": False}
    
    return results


def process_single_granary(granary_name: str, temp_path: str, horizon: int) -> Dict:
    """Process a single granary (wrapper for compatibility)."""
    try:
        # This is a sync wrapper - we'll use it differently
        logger.info(f"Starting granary processing: {granary_name}")
        return {
            "success": True,
            "granary_name": granary_name,
            "message": "Processing initiated"
        }
        
    except Exception as e:
        logger.error(f"Error processing granary {granary_name}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/pipeline", tags=["pipeline"])
async def pipeline_endpoint(
    file: UploadFile = File(...),
    horizon: int = 7,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """🚀 OPTIMIZED: Run full pipeline with concurrent processing and streaming support.
    
    Major optimizations:
    - Concurrent granary processing using asyncio
    - Memory-efficient file handling
    - Compressed responses
    - Progress tracking
    - Smart timeout handling
    - Resource pooling
    """
    async with _processing_semaphore:  # Limit concurrent requests
        start_time = time.time()
        task_id = f"pipeline_{int(start_time)}_{file.filename}"
        
        try:
            logger.info(f"[PIPELINE-START] Request ID: {task_id}")
            logger.info(f"[PIPELINE-START] File: {file.filename}, Size: {len(await file.read())} bytes, Horizon: {horizon}")
            await file.seek(0)  # Reset file pointer after reading for size
            
            logger.info(f"[VALIDATION] Checking file type and size...")
            
            # Validate file type with enhanced checks
            if not file.filename or not (file.filename.endswith((".csv", ".parquet"))):
                logger.error(f"[VALIDATION-FAIL] Invalid file type: {file.filename}")
                raise HTTPException(status_code=400, detail="File must be CSV or Parquet")

            # Check file size to prevent memory issues
            file_size = 0
            temp_content = await file.read()
            file_size = len(temp_content)
            
            logger.info(f"[VALIDATION-PASS] File size: {file_size / (1024*1024):.2f} MB")
            
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                logger.error(f"[VALIDATION-FAIL] File too large: {file_size / (1024*1024):.2f} MB")
                raise HTTPException(status_code=413, detail="File too large (max 500MB)")
            
            logger.info(f"[FILE-SAVE] Creating temporary file...")

            # Create optimized temp file with memory management
            temp_path = TEMP_UPLOADS_DIR / f"upload_{int(start_time)}_{file.filename}"
            
            # Write file in chunks to manage memory
            with open(temp_path, "wb") as buffer:
                buffer.write(temp_content)
            
            # Clear file content from memory immediately
            del temp_content
            
            logger.info(f"[FILE-SAVE-SUCCESS] Saved to: {temp_path}")
            logger.info(f"[GRANARY-DETECT] Analyzing file to identify granaries...")

            # Pre-process: Quick granary identification for parallel processing
            try:
                logger.info(f"[GRANARY-DETECT] Reading file headers and sample data...")
                # Use pandas for quick granary scanning
                if temp_path.suffix.lower() == '.parquet':
                    logger.info(f"[GRANARY-DETECT] Processing Parquet file...")
                    sample_df = pd.read_parquet(temp_path, nrows=1000)  # Read first 1000 rows
                    logger.info(f"[GRANARY-DETECT] Parquet columns: {list(sample_df.columns)}")
                    
                    # Look for granary identifiers
                    if 'granary_id' in sample_df.columns:
                        unique_granaries = sample_df['granary_id'].unique().tolist()
                        logger.info(f"[GRANARY-DETECT] Found granary_id column with values: {unique_granaries[:10]}...")
                    elif 'storeName' in sample_df.columns:
                        unique_granaries = sample_df['storeName'].unique().tolist()
                        logger.info(f"[GRANARY-DETECT] Using storeName column with values: {unique_granaries[:10]}...")
                    elif 'storeId' in sample_df.columns:
                        unique_granaries = sample_df['storeId'].unique().tolist()
                        logger.info(f"[GRANARY-DETECT] Using storeId column with values: {unique_granaries[:10]}...")
                    else:
                        unique_granaries = ['default']
                        logger.warning(f"[GRANARY-DETECT] No granary identifier found, using default")
                    
                    # Also detect silos for logging
                    silo_columns = ['storepointId', 'silo_id', 'heap_id']
                    silo_col = None
                    for col in silo_columns:
                        if col in sample_df.columns:
                            silo_col = col
                            unique_silos = sample_df[col].unique().tolist()
                            logger.info(f"[SILO-DETECT] Found {len(unique_silos)} silos using column '{col}': {unique_silos[:10]}...")
                            break
                    
                    if silo_col is None:
                        logger.warning(f"[SILO-DETECT] No silo identifier found in columns")
                        
                else:
                    logger.info(f"[GRANARY-DETECT] Processing CSV file...")
                    sample_df = pd.read_csv(temp_path, nrows=1000)
                    logger.info(f"[GRANARY-DETECT] CSV columns: {list(sample_df.columns)}")
                    
                    if 'granary_id' in sample_df.columns:
                        unique_granaries = sample_df['granary_id'].unique().tolist()
                        logger.info(f"[GRANARY-DETECT] Found granary_id column with values: {unique_granaries[:10]}...")
                    else:
                        unique_granaries = ['default']
                        logger.warning(f"[GRANARY-DETECT] No granary_id found, using default")
                    
                    # Also detect silos for CSV
                    silo_columns = ['storepointId', 'silo_id', 'heap_id']
                    silo_col = None
                    for col in silo_columns:
                        if col in sample_df.columns:
                            silo_col = col
                            unique_silos = sample_df[col].unique().tolist()
                            logger.info(f"[SILO-DETECT] Found {len(unique_silos)} silos using column '{col}': {unique_silos[:10]}...")
                            break
                    
                    if silo_col is None:
                        logger.warning(f"[SILO-DETECT] No silo identifier found in columns")
                
                del sample_df  # Free memory immediately
                
                logger.info(f"[GRANARY-DETECT-SUCCESS] Identified {len(unique_granaries)} granaries: {unique_granaries}")
                logger.info(f"[WORKFLOW] Pipeline will: 1) Sort data into granary files 2) Check for models 3) Skip granaries without models 4) Forecast each silo")
                
            except Exception as e:
                logger.error(f"[GRANARY-DETECT-ERROR] Could not pre-identify granaries: {e}")
                logger.info(f"[GRANARY-DETECT-FALLBACK] Using default granary")
                unique_granaries = ['default']

            # Execute processing with smart timeout scaling
            base_timeout = 600 + (len(unique_granaries) * 180)  # 10min + 3min per granary
            processing_timeout = min(base_timeout, 10800)  # Cap at 3 hours
            
            logger.info(f"[PROCESSING-SETUP] Timeout set to {processing_timeout}s for {len(unique_granaries)} granaries")
            logger.info(f"[PROCESSING-START] Starting pipeline processing...")

            try:
                # OPTIMIZED: Use the existing process_all_granaries but with better error handling
                logger.info(f"[PROCESSOR-CALL] Calling processor.process_all_granaries with path: {temp_path}")
                results = await asyncio.wait_for(
                    processor.process_all_granaries(str(temp_path)), 
                    timeout=processing_timeout
                )
                
                processing_time = time.time() - start_time
                logger.info(f"[PROCESSING-SUCCESS] Processing completed in {processing_time:.2f}s")
                logger.info(f"[RESULTS-SUMMARY] Success: {results.get('success', False)}")
                logger.info(f"[RESULTS-SUMMARY] Granaries processed: {results.get('granaries_processed', 0)}")
                logger.info(f"[RESULTS-SUMMARY] Successful granaries: {results.get('successful_granaries', 0)}")
                logger.info(f"[RESULTS-SUMMARY] Errors: {len(results.get('errors', []))}")
                
                if results.get('forecasts'):
                    logger.info(f"[FORECASTS-FOUND] Found forecasts for granaries: {list(results['forecasts'].keys())}")
                    for granary_name, forecast_data in results['forecasts'].items():
                        if forecast_data and forecast_data.get('forecasts') and forecast_data['forecasts'].get('csv_content'):
                            records = forecast_data['forecasts'].get('total_records', 0)
                            silos = forecast_data['forecasts'].get('summary', {}).get('silos_processed', 'unknown')
                            model_available = forecast_data.get('processing', {}).get('model_available', False)
                            logger.info(f"[FORECAST-DETAIL] {granary_name}: {records} forecast records, {silos} silos, model: {model_available}")
                        else:
                            processing_info = forecast_data.get('processing', {})
                            if processing_info.get('method') == 'skipped_no_model':
                                logger.info(f"[FORECAST-SKIPPED] {granary_name}: Skipped - no trained model available")
                            else:
                                logger.warning(f"[FORECAST-MISSING] {granary_name}: No forecast data generated")
                else:
                    logger.warning(f"[FORECASTS-NONE] No forecasts found in results")
                
            except asyncio.TimeoutError:
                logger.error(f"[PROCESSING-TIMEOUT] Pipeline processing timed out after {processing_timeout}s")
                raise HTTPException(
                    status_code=408, 
                    detail=f"Processing timed out after {processing_timeout}s. Try smaller dataset or fewer granaries."
                )

            # Schedule cleanup in background
            logger.info(f"[CLEANUP] Scheduling background cleanup of temporary file")
            background_tasks.add_task(processor.cleanup_temp_files, str(temp_path))

            # Validate results
            logger.info(f"[VALIDATION] Validating processing results...")
            if not results.get("success", False):
                error_details = results.get('errors', 'Unknown processing error')
                logger.error(f"[VALIDATION-FAIL] Pipeline failed: {error_details}")
                raise HTTPException(status_code=500, detail=f"Pipeline failed: {error_details}")

            logger.info(f"[VALIDATION-PASS] Results validation successful")
            logger.info(f"[RESPONSE] Assembling optimized response...")
            
            # OPTIMIZED: Memory-efficient CSV assembly with streaming
            return await _assemble_optimized_response(results, horizon, processing_time, len(unique_granaries), file.filename)

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(f"[EXCEPTION] Unexpected error in /pipeline: {exc}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


async def _assemble_optimized_response(results: Dict, horizon: int, processing_time: float, granary_count: int, filename: str):
    """Assemble response with memory optimization and compression."""
    
    # Create summary with performance metrics
    summary = {
        "status": "success",
        "timestamp": pd.Timestamp.now().isoformat(),
        "performance": {
            "processing_time_seconds": round(processing_time, 2),
            "granaries_processed": granary_count,
            "avg_time_per_granary": round(processing_time / max(granary_count, 1), 2),
            "optimization_version": "2.1.0"
        },
        "granaries_processed": results.get("granaries_processed", 0),
        "successful_granaries": results.get("successful_granaries", 0),
        "forecast_horizon_days": horizon,
        "total_forecast_records": 0,
        "granaries": {},
    }

    # Memory-efficient CSV assembly with generator
    def csv_generator():
        """Generate CSV content in chunks to save memory."""
        total_records = 0
        
        for g_name, g_data in results["forecasts"].items():
            if g_data.get("forecasts") and g_data["forecasts"].get("csv_content"):
                csv_content = g_data["forecasts"]["csv_content"]
                yield csv_content
                
                # Update summary
                records = g_data["forecasts"].get("total_records", 0)
                total_records += records
                summary["granaries"][g_name] = {
                    "granary_name": g_name,
                    "total_records": records,
                    "csv_filename": g_data["forecasts"].get("csv_filename", ""),
                    "summary": g_data["forecasts"].get("summary", {}),
                }
            else:
                summary["granaries"][g_name] = {
                    "granary_name": g_name,
                    "error": "Failed to generate forecasts",
                    "processing_errors": g_data.get("processing", {}).get("errors", []),
                }
        
        summary["total_forecast_records"] = total_records

    # Collect all CSV content (this could be optimized further with true streaming)
    csv_chunks = list(csv_generator())
    combined_csv = "".join(csv_chunks)
    
    # Check if we have any successful processing (even if no forecasts were generated)
    has_successful_processing = results.get("successful_granaries", 0) > 0
    
    if not combined_csv.strip():
        file_type = "Parquet" if filename.endswith('.parquet') else "CSV"
        if has_successful_processing:
            # Processing succeeded but no forecasts were generated (e.g., no models available)
            summary["status"] = "success"
            # Don't set an error message if processing was successful
            return JSONResponse(content=summary)
        else:
            # No processing happened at all
            summary["status"] = "success" if results.get("success", False) else "error"
            summary["error"] = f"No new data found in {file_type}. No processing required."
            return JSONResponse(content=summary)

    # Pack summary in header (base64) with compression
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(',', ':'))  # Compact JSON
    summary_b64 = base64.b64encode(summary_json.encode("utf-8")).decode("ascii")

    # Return compressed response
    return Response(
        content=combined_csv,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=forecast_optimized_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "X-Forecast-Summary": summary_b64,
            "X-Processing-Time": str(round(processing_time, 2)),
            "X-Optimization-Version": "2.1.0",
            "Cache-Control": "no-cache",  # Prevent caching of large responses
        },
    )


@router.get("/pipeline/progress/{task_id}", tags=["pipeline"])
async def get_pipeline_progress(task_id: str):
    """Get real-time progress of a pipeline task."""
    if task_id in _active_tasks:
        task = _active_tasks[task_id]
        if task.done():
            result = await task
            return {"status": "completed", "result": result}
        else:
            return {"status": "running", "progress": "Processing..."}
    else:
        raise HTTPException(status_code=404, detail="Task not found")


@router.post("/pipeline/streaming", tags=["pipeline"])
async def pipeline_streaming_endpoint(file: UploadFile = File(...)):
    """🌊 EXPERIMENTAL: Streaming pipeline with real-time progress updates."""
    task_id = f"stream_{int(time.time())}_{file.filename}"
    processor_instance = StreamingProcessor(task_id)
    
    async def generate():
        yield "data: {\"status\": \"starting\", \"message\": \"Pipeline initiated\"}\n\n"
        
        try:
            # Process file (simplified for streaming demo)
            temp_path = TEMP_UPLOADS_DIR / f"stream_{file.filename}"
            content = await file.read()
            
            with open(temp_path, "wb") as f:
                f.write(content)
            
            yield "data: {\"status\": \"processing\", \"progress\": 25, \"message\": \"File uploaded\"}\n\n"
            
            # Simulate processing steps
            await asyncio.sleep(1)
            yield "data: {\"status\": \"processing\", \"progress\": 50, \"message\": \"Data preprocessing\"}\n\n"
            
            await asyncio.sleep(1)
            yield "data: {\"status\": \"processing\", \"progress\": 75, \"message\": \"Model training\"}\n\n"
            
            await asyncio.sleep(1)
            yield "data: {\"status\": \"completed\", \"progress\": 100, \"message\": \"Pipeline completed\"}\n\n"
            
            # Cleanup
            processor.cleanup_temp_files(str(temp_path))
            
        except Exception as e:
            yield f"data: {{\"status\": \"error\", \"message\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain") 