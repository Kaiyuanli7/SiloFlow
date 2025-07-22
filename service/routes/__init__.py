"""FastAPI router aggregation for SiloFlow service."""

from fastapi import APIRouter

# Individual routers are imported purely for side-effects so they register variables
from .pipeline import router as pipeline_router  # noqa: F401
from .train import router as train_router  # noqa: F401
from .forecast import router as forecast_router  # noqa: F401
from .models import router as models_router  # noqa: F401
from .health import router as health_router  # noqa: F401


# Expose a combined router if the main app wants to include everything in one call
router = APIRouter()
for r in [pipeline_router, train_router, forecast_router, models_router, health_router]:
    router.include_router(r) 