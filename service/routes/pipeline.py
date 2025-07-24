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


@router.post("/pipeline", tags=["pipeline"])
async def pipeline_endpoint(
    file: UploadFile = File(...),
    horizon: int | None = 7,
):
    """Run full pipeline (ingestion ➜ preprocess ➜ train ➜ forecast) for all granaries in the uploaded file.

    This is the end-to-end one-stop shop that performs all steps:
    1. Ingests and sorts the uploaded CSV/Parquet file
    2. Preprocesses each granary's data (cleaning, feature engineering)
    3. Trains models for each granary (if not already trained)
    4. Generates forecasts for the specified horizon
    
    Returns combined forecast CSV with processing summary.
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