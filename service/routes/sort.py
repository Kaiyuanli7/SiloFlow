from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# Create temp uploads directory (shared for router)
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(exist_ok=True, parents=True)


@router.post("/sort", tags=["sort"])
async def sort_endpoint(
    file: UploadFile = File(...),
):
    """Sort-only endpoint - Ingests and sorts raw data without preprocessing.
    
    This endpoint:
    1. Ingests the uploaded CSV/Parquet file
    2. Sorts and deduplicates the data
    3. Splits data by granary into separate Parquet files
    4. Saves sorted files to data/granaries/
    
    Returns sorting status for each granary.
    """
    try:
        logger.info("Received sort request: %s", file.filename)

        # Validate file type -------------------------------------------------
        if not file.filename or not (file.filename.endswith(".csv") or file.filename.endswith(".parquet")):
            raise HTTPException(status_code=400, detail="File must be CSV or Parquet")

        # Persist upload to a unique temp file --------------------------------
        temp_path = TEMP_UPLOADS_DIR / f"upload_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info("Saved uploaded file to %s", temp_path)

        # Execute ingestion and sorting only ----------------------------------
        try:
            from granarypredict import ingestion
            
            def sort_operation():
                return ingestion.ingest_and_sort(str(temp_path), return_new_data_status=True)
            
            results = await asyncio.to_thread(sort_operation)
            
        except Exception as e:
            logger.error("Sorting failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Sorting failed: {e}")

        # Clean-up temp file --------------------------------------------------
        try:
            temp_path.unlink()
            logger.info("Cleaned up temp file: %s", temp_path)
        except Exception as e:
            logger.warning("Could not clean up temp file %s: %s", temp_path, e)

        # Handle both old and new return formats for backward compatibility --
        if isinstance(results, dict) and 'granary_status' in results:
            # New format with detailed status
            granary_status = results['granary_status']
            silo_changes = results.get('silo_changes', {})
        else:
            # Old format - fallback
            granary_status = results if isinstance(results, dict) else {}
            silo_changes = {}

        # Prepare response with sorting results ------------------------------
        response_data = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "granaries_sorted": len([g for g, status in granary_status.items() if status]),
            "total_granaries": len(granary_status),
            "results": {}
        }

        # Add details for each granary
        for granary_name, is_new in granary_status.items():
            if is_new:
                # Check if granary file exists
                from core import processor
                granary_file = processor.granaries_dir / f"{granary_name}.parquet"
                
                if granary_file.exists():
                    response_data["results"][granary_name] = {
                        "success": True,
                        "steps_completed": ["sort"],
                        "sorted_file": str(granary_file),
                        "file_size_mb": round(granary_file.stat().st_size / (1024 * 1024), 2),
                        "silos_changed": silo_changes.get(granary_name, [])
                    }
                else:
                    response_data["results"][granary_name] = {
                        "success": False,
                        "error": "Sorted file not found"
                    }
            else:
                response_data["results"][granary_name] = {
                    "success": True,
                    "status": "No new data - skipped",
                    "steps_completed": []
                }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /sort: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
