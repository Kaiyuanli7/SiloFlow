from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core import processor  # Singleton

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# Create temp uploads directory (shared for router)
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(exist_ok=True, parents=True)


@router.post("/process", tags=["process"])
async def process_endpoint(
    file: UploadFile = File(...),
):
    """Process-only endpoint - Ingests and processes granaries without training or forecasting.
    
    This endpoint:
    1. Ingests the uploaded CSV/Parquet file
    2. Splits data by granary into separate Parquet files
    3. Preprocesses each granary's data (cleaning, feature engineering)
    4. Saves processed files to data/processed/
    
    Returns processing status for each granary.
    """
    try:
        logger.info("Received process request: %s", file.filename)

        # Validate file type -------------------------------------------------
        if not file.filename or not (file.filename.endswith(".csv") or file.filename.endswith(".parquet")):
            raise HTTPException(status_code=400, detail="File must be CSV or Parquet")

        # Persist upload to a unique temp file --------------------------------
        temp_path = TEMP_UPLOADS_DIR / f"upload_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info("Saved uploaded file to %s", temp_path)

        # Execute ingestion and preprocessing only ----------------------------
        try:
            # Step 1: Ingest and sort the data (to get granary list)
            ingestion_results = await asyncio.wait_for(
                processor.process_raw_csv(str(temp_path)), timeout=1800  # 30 min for ingestion
            )
            
            # Get all granaries from the uploaded file (not just new ones)
            granaries_from_file = ingestion_results.get("granaries", [])
            
            # If no new granaries from ingestion, extract granary names from the uploaded file
            if not granaries_from_file:
                logger.info("No new granaries from ingestion, checking file directly...")
                
                # Method 1: Use the filename as granary name (most common case)
                filename_without_ext = temp_path.stem
                # Remove timestamp prefix if present (upload_20250724_093600_)
                if filename_without_ext.startswith('upload_'):
                    parts = filename_without_ext.split('_')
                    if len(parts) >= 4:  # upload_YYYYMMDD_HHMMSS_actualname
                        granary_from_filename = '_'.join(parts[3:])
                    else:
                        granary_from_filename = filename_without_ext
                else:
                    granary_from_filename = filename_without_ext
                
                logger.info(f"Extracted granary name from filename: {granary_from_filename}")
                
                # Method 2: Also try reading the file to extract granary names from data
                try:
                    if temp_path.suffix.lower() == '.csv':
                        df = pd.read_csv(temp_path)
                    else:
                        df = pd.read_parquet(temp_path)
                    
                    granaries_from_data = []
                    if 'granary_name' in df.columns:
                        granaries_from_data = df['granary_name'].unique().tolist()
                    elif 'granary_id' in df.columns:
                        granaries_from_data = df['granary_id'].unique().tolist()
                    
                    if granaries_from_data:
                        logger.info(f"Found granaries in data columns: {granaries_from_data}")
                        # Use data-based granaries if available
                        all_granaries = granaries_from_data
                    else:
                        # Fall back to filename-based granary
                        all_granaries = [granary_from_filename]
                        logger.info(f"Using filename-based granary: {all_granaries}")
                        
                except Exception as e:
                    logger.error(f"Error reading uploaded file: {e}")
                    # Fall back to filename-based granary
                    all_granaries = [granary_from_filename]
                    logger.info(f"Fallback to filename-based granary: {all_granaries}")
                
                # Filter to granaries that don't have processed files yet
                granaries_to_process = []
                for granary in all_granaries:
                    processed_file = processor.processed_dir / f"{granary}_processed.parquet"
                    if not processed_file.exists():
                        granaries_to_process.append(granary)
                        logger.info(f"Granary {granary} needs preprocessing")
                    else:
                        logger.info(f"Granary {granary} already has processed file, skipping")
                
                granaries_from_file = granaries_to_process
            
            if not granaries_from_file:
                # Clean-up temp file --------------------------------------------------
                processor.cleanup_temp_files(str(temp_path))
                
                return JSONResponse(content={
                    "status": "success",
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "granaries_processed": 0,
                    "successful_granaries": 0,
                    "message": "No granaries need preprocessing - all processed files already exist",
                    "results": {}
                })
            
            # Step 2: Process each granary (preprocessing only)
            processing_results = {}
            successful_granaries = 0
            
            for granary_name in granaries_from_file:
                logger.info(f"Processing granary: {granary_name}")
                
                try:
                    # Force preprocessing by calling process_granary directly
                    granary_result = await asyncio.wait_for(
                        processor.process_granary(granary_name, changed_silos=None), timeout=1800  # 30 min per granary
                    )
                    
                    if granary_result.get("success", False):
                        successful_granaries += 1
                        processing_results[granary_name] = {
                            "success": True,
                            "steps_completed": ["ingest", "preprocess"],
                            "processing_details": granary_result
                        }
                        
                        # Check if processed file exists
                        processed_file = processor.processed_dir / f"{granary_name}_processed.parquet"
                        if processed_file.exists():
                            processing_results[granary_name]["processed_file"] = str(processed_file)
                            processing_results[granary_name]["file_size_mb"] = round(processed_file.stat().st_size / (1024 * 1024), 2)
                        else:
                            processing_results[granary_name]["warning"] = "Processing succeeded but no output file found"
                    else:
                        processing_results[granary_name] = {
                            "success": False,
                            "error": f"Processing failed: {granary_result.get('errors', 'Unknown error')}"
                        }
                        
                except asyncio.TimeoutError:
                    logger.error(f"Processing timed out for granary: {granary_name}")
                    processing_results[granary_name] = {
                        "success": False,
                        "error": "Processing timed out"
                    }
                except Exception as e:
                    logger.error(f"Error processing granary {granary_name}: {e}")
                    processing_results[granary_name] = {
                        "success": False,
                        "error": f"Processing error: {str(e)}"
                    }
            
        except asyncio.TimeoutError:
            logger.error("Processing timed out")
            raise HTTPException(status_code=408, detail="Processing timed out – try smaller dataset")

        # Clean-up temp file --------------------------------------------------
        processor.cleanup_temp_files(str(temp_path))

        # Prepare response with processing results ----------------------------
        response_data = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "granaries_processed": len(granaries_from_file),
            "successful_granaries": successful_granaries,
            "results": processing_results
        }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /process: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
