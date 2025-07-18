from __future__ import annotations

"""Shared core objects for the service layer (singleton instances, common utilities)."""

import sys
from pathlib import Path
import logging

# Ensure granarypredict package is importable before importing processor
root_dir = Path(__file__).resolve().parent.parent
granarypredict_dir = root_dir / "granarypredict"
if str(granarypredict_dir) not in sys.path:
    sys.path.insert(0, str(granarypredict_dir))

# Local imports to avoid circular dependencies
from .automated_processor import AutomatedGranaryProcessor  # noqa: E402
from .utils.data_paths import data_paths  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure all data directories exist
data_paths.ensure_directories()

# Singleton processor instance used by all API routers
processor = AutomatedGranaryProcessor()
logger.info("AutomatedGranaryProcessor singleton initialised: %s", processor)
logger.info("Data paths initialized: %s", data_paths.get_data_summary()) 