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
async def list_models(page: int = 1, per_page: int = 10):
    """List all available models with pagination"""
    try:
        from ..utils.data_paths import data_paths
        models_dir = data_paths.get_models_dir()
        model_files = list(models_dir.glob("*_forecast_model.joblib*"))
        
        # Pagination logic
        total_models = len(model_files)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_files = model_files[start_idx:end_idx]
        
        models = []
        for file_path in paginated_files:
            granary_name = file_path.stem.replace("_forecast_model", "")
            if granary_name.endswith(".joblib"):
                granary_name = granary_name[:-7]
                
            models.append({
                "granary": granary_name,
                "model_path": str(file_path),
                "size_mb": file_path.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat()
            })
        
        return {
            'status': 'success',
            'page': page,
            'per_page': per_page,
            'total_models': total_models,
            'total_pages': (total_models + per_page - 1) // per_page,
            'models': models
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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