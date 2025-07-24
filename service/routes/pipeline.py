from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from core import processor  # Singleton

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

# Create temp uploads directory (shared for router)
TEMP_UPLOADS_DIR = Path("temp_uploads")
TEMP_UPLOADS_DIR.mkdir(exist_ok=True, parents=True)


@router.post("/process", tags=["pipeline"])
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
            results = await asyncio.wait_for(
                processor.process_raw_csv(str(temp_path)), timeout=3600  # 1 hour timeout
            )
        except asyncio.TimeoutError:
            logger.error("Processing timed out")
            raise HTTPException(status_code=408, detail="Processing timed out – try smaller dataset")

        # Clean-up temp file --------------------------------------------------
        processor.cleanup_temp_files(str(temp_path))

        # Prepare response with processing results ----------------------------
        response_data = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "granaries_processed": len(results.get("granaries", [])),
            "successful_granaries": len(results.get("granaries", [])),
            "results": {}
        }

        # Add details for each processed granary
        for granary_name in results.get("granaries", []):
            processed_file = processor.processed_dir / f"{granary_name}_processed.parquet"
            if processed_file.exists():
                response_data["results"][granary_name] = {
                    "success": True,
                    "steps_completed": ["preprocess"],
                    "processed_file": str(processed_file),
                    "file_size_mb": round(processed_file.stat().st_size / (1024 * 1024), 2)
                }
            else:
                response_data["results"][granary_name] = {
                    "success": False,
                    "error": "Processed file not found"
                }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /process: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@router.post("/pipeline", tags=["pipeline"])
async def pipeline_endpoint(
    file: UploadFile = File(...),
    horizon: int | None = 7,
):
    """Run full pipeline (ingestion ➜ preprocess ➜ train ➜ forecast) for all granaries in the uploaded file.

    Compared with `/train` and `/forecast` this endpoint is the end-to-end one-stop shop.
    """
    try:
        logger.info("Received pipeline request: %s, horizon=%s", file.filename, horizon)

        # Validate file type -------------------------------------------------
        if not file.filename or not (file.filename.endswith(".csv") or file.filename.endswith(".parquet")):
            raise HTTPException(status_code=400, detail="File must be CSV or Parquet")

        # Persist upload to a unique temp file --------------------------------
        temp_path = TEMP_UPLOADS_DIR / f"upload_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info("Saved uploaded file to %s", temp_path)

        # Execute processor with generous timeout ----------------------------
        try:
            results = await asyncio.wait_for(
                processor.process_all_granaries(str(temp_path)), timeout=10800  # 3 h
            )
        except asyncio.TimeoutError:
            logger.error("Pipeline processing timed out")
            raise HTTPException(status_code=408, detail="Processing timed out – try smaller dataset")

        # Clean-up temp file --------------------------------------------------
        processor.cleanup_temp_files(str(temp_path))

        if not results.get("success", False):
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {results.get('errors')}")

        # Assemble CSV forecasts from all granaries --------------------------
        combined_rows: list[str] = []
        summary = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "granaries_processed": results.get("granaries_processed", 0),
            "successful_granaries": results.get("successful_granaries", 0),
            "forecast_horizon_days": horizon,
            "total_forecast_records": 0,
            "granaries": {},
        }

        for g_name, g_data in results["forecasts"].items():
            if g_data["forecasts"] and g_data["forecasts"].get("csv_content"):
                csv_content = g_data["forecasts"]["csv_content"]
                combined_rows.append(csv_content)
                summary["granaries"][g_name] = {
                    "granary_name": g_name,
                    "total_records": g_data["forecasts"]["total_records"],
                    "csv_filename": g_data["forecasts"]["csv_filename"],
                    "summary": g_data["forecasts"]["summary"],
                }
                summary["total_forecast_records"] += g_data["forecasts"]["total_records"]
            else:
                summary["granaries"][g_name] = {
                    "granary_name": g_name,
                    "error": "Failed to generate forecasts",
                    "processing_errors": g_data["processing"].get("errors", []),
                }

        combined_csv = "".join(combined_rows)
        if not combined_csv.strip():
            summary["status"] = "success" if results.get("success", False) else "error"
            summary["error"] = "No new data found in CSV. No processing required."
            return JSONResponse(content=summary)

        # Pack summary in header (base64) ------------------------------------
        summary_b64 = base64.b64encode(json.dumps(summary, ensure_ascii=False).encode("utf-8")).decode("ascii")

        return Response(
            content=combined_csv,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "X-Forecast-Summary": summary_b64,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /pipeline: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") 