from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..core import processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


@router.get("/models", tags=["models"])
async def list_models():
    """Return metadata on available trained models."""
    try:
        models: list[dict] = []
        for pattern in ["*_forecast_model.joblib", "*_forecast_model.joblib.gz"]:
            for model_file in processor.models_dir.glob(pattern):
                granary_name = model_file.stem.replace("_forecast_model", "")
                if granary_name.endswith(".joblib"):
                    granary_name = granary_name[:-7]
                models.append(
                    {
                        "granary": granary_name,
                        "model_path": str(model_file),
                        "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                        "modified": pd.Timestamp.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                        "compressed": model_file.suffix == ".gz",
                    }
                )
        return JSONResponse(content={"status": "success", "models_count": len(models), "models": models})
    except Exception as exc:
        logger.exception("Error listing models: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error listing models: {exc}")


@router.delete("/models/{granary_name}", tags=["models"])
async def delete_model(granary_name: str):
    """Delete a model for the specified granary."""
    try:
        model_path = processor.models_dir / f"{granary_name}_forecast_model.joblib"
        compressed_path = processor.models_dir / f"{granary_name}_forecast_model.joblib.gz"
        if model_path.exists():
            model_path.unlink()
        elif compressed_path.exists():
            compressed_path.unlink()
        else:
            raise HTTPException(status_code=404, detail=f"Model not found for granary: {granary_name}")
        return JSONResponse(content={"status": "success", "message": f"Model deleted: {granary_name}"})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error deleting model: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error deleting model: {exc}") 