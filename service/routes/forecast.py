from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from core import processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


@router.get("/forecast", tags=["forecast"])
async def forecast_all_endpoint():
    """Generate a *single-day* forecast for every processed granary that has a trained model."""
    try:
        processed_files = list(processor.processed_dir.glob("*_processed.parquet"))
        if not processed_files:
            raise HTTPException(status_code=400, detail="No processed data available. Run /pipeline first.")

        horizon = 1  # Only latest day
        forecasts: dict[str, dict] = {}
        skipped: list[str] = []

        for parquet_path in processed_files:
            granary_name = parquet_path.stem.replace("_processed", "")
            model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib"
            compressed_model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib.gz"

            if not (model_file.exists() or compressed_model_file.exists()):
                skipped.append(granary_name)
                continue

            forecast_result = await processor.generate_forecasts(granary_name, horizon=horizon)
            if forecast_result:
                forecasts[granary_name] = forecast_result
            else:
                skipped.append(granary_name)

        payload = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "horizon_days": horizon,
            "forecasts_count": len(forecasts),
            "skipped": skipped,
            "forecasts": forecasts,
        }
        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /forecast: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") 