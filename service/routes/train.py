from __future__ import annotations

import gc
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from core import processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


@router.post("/train", tags=["train"])
async def train_endpoint():
    """Train models for every processed granary Parquet that lacks a model.

    Runs sequentially to keep memory/GPU usage in check. Models are saved in
    `models/`. Granaries already owning a model (compressed or not) are skipped.
    """
    try:
        processed_files = list(processor.processed_dir.glob("*_processed.parquet"))
        if not processed_files:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "No processed Parquet files found. Run /pipeline or /process first.",
                },
                status_code=400,
            )

        trained: list[str] = []
        skipped: list[str] = []
        errors: dict[str, list[str]] = {}

        for parquet_path in processed_files:
            granary_name = parquet_path.stem.replace("_processed", "")
            model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib"
            compressed_model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib.gz"

            if model_file.exists() or compressed_model_file.exists():
                skipped.append(granary_name)
                continue

            logger.info("[TRAIN] Training model for granary: %s", granary_name)
            train_result = await processor.process_granary(granary_name)

            if train_result.get("success"):
                trained.append(granary_name)
            else:
                errors[granary_name] = train_result.get("errors", ["Unknown error"])

            # Free resources -------------------------------------------------
            gc.collect()
            try:
                import torch  # type: ignore

                torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception:
                pass

        return JSONResponse(
            content={
                "status": "success",
                "trained_granaries": trained,
                "skipped_granaries": skipped,
                "errors": errors,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )
    except Exception as exc:
        logger.exception("Unexpected error in /train: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") 