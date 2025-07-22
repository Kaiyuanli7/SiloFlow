from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@router.get("/health", tags=["health"])
async def health_check():
    """Simple health-check showing directory presence."""
    try:
        required_dirs = ["models", "data/processed", "data/granaries"]
        dir_status = {d: Path(d).exists() for d in required_dirs}
        return {
            "status": "healthy",
            "service": "SiloFlow Automated Pipeline",
            "timestamp": pd.Timestamp.now().isoformat(),
            "directories": dir_status,
        }
    except Exception as exc:
        logger.exception("Health check failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {exc}") 