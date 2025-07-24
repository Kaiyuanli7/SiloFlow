"""Central FastAPI application file – now a slim orchestrator that delegates
all route implementations to the modules in *service.routes*.

This replaces the previous monolithic version to improve maintainability.
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging early ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SiloFlow Automated Pipeline",
    description="Automated grain temperature forecasting service",
    version="2.0.0",
)

# CORS – open for now, lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Include modular routers
# ---------------------------------------------------------------------------
from routes import router as all_routes  # noqa: E402

app.include_router(all_routes)


# ---------------------------------------------------------------------------
# Entrypoint – ``python -m uvicorn service.main:app --reload``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 