from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from .config import MODELS_DIR
from .evaluate import time_series_cv
from .compression_utils import save_compressed_model, load_compressed_model, get_lightgbm_compression_params

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train RandomForestRegressor and return model & metrics."""
    logger.info("Training RandomForest (n=%d) on shape X=%s", n_estimators, X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    logger.info("Validation MAE: %.3f, RMSE: %.3f", metrics["mae"], metrics["rmse"])
    return model, metrics


def train_gb_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_type: str = "hist",  # 'hist' or 'gb'
    n_estimators: int = 300,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
) -> Tuple[Any, Dict[str, float]]:
    """Train a gradient-boosting regressor with time-series CV."""
    logger.info("Training %s GradientBoosting (n=%d) on X shape=%s", model_type, n_estimators, X.shape)
    if model_type == "hist":
        mdl = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            random_state=42,
        )
    elif model_type == "gb":
        mdl = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
    else:
        raise ValueError("model_type must be 'hist' or 'gb'")

    mdl, metrics = time_series_cv(mdl, X, y)
    return mdl, metrics


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    # Tuned hyper-parameters (May-2025)
    n_estimators: int = 1185,
    learning_rate: float = 0.03347500352712116,
    max_depth: int | None = 7,
    num_leaves: int = 24,
    subsample: float = 0.8832753633141975,
    colsample_bytree: float = 0.6292206613991069,
    min_child_samples: int = 44,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, float]]:
    """Train a LightGBM regressor using the project-wide tuned defaults.

    The default values reflect a hyper-parameter search run on multiple
    silo datasets (see tuning.md).  Callers can still override any of the
    parameters as needed.
    """
    logger.info(
        "Training LightGBM (n_estimators=%d, lr=%.4f, max_depth=%s, num_leaves=%d) on X shape=%s",
        n_estimators,
        learning_rate,
        str(max_depth),
        num_leaves,
        X.shape,
    )

    # Get LightGBM compression parameters
    compression_params = get_lightgbm_compression_params(compression_level=6, enable_compression=True)
    
    # Handle max_depth parameter properly
    max_depth_param = int(max_depth) if max_depth is not None else -1

    model = LGBMRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=max_depth_param,
        num_leaves=int(num_leaves),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=int(min_child_samples),
        random_state=random_state,
        n_jobs=-1,
        **compression_params,  # Add compression parameters
    )

    # Fit on full data
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
    }
    logger.info("LightGBM MAE: %.3f RMSE: %.3f", metrics["mae"], metrics["rmse"])
    return model, metrics


def save_model(model: Any, name: str = "rf_model.joblib", 
               use_compression: bool = True, 
               compression_config: Optional[Dict] = None) -> Dict:
    """
    Save a model with adaptive compression based on model type and size.
    
    Parameters:
    -----------
    model : Any
        The model to save
    name : str
        Name of the model file
    use_compression : bool
        Whether to use compression
    compression_config : Optional[Dict]
        Custom compression configuration
        
    Returns:
    --------
    Dict
        Compression statistics and metadata
    """
    path = MODELS_DIR / name
    
    if use_compression:
        stats = save_compressed_model(model, path, compression_config=compression_config)
        logger.info("Saved compressed model to %s", stats['path'])
        return stats
    else:
        joblib.dump(model, path)
        file_size = path.stat().st_size
        stats = {
            'path': str(path),
            'compression_algorithm': 'none',
            'compression_level': 0,
            'size_category': 'uncompressed',
            'uncompressed_size_mb': file_size / (1024 * 1024),
            'compressed_size_mb': file_size / (1024 * 1024),
            'compression_ratio': 1.0,
            'space_saved_mb': 0,
            'save_time_seconds': 0
        }
        logger.info("Saved uncompressed model to %s (%.2f MB)", path, stats['compressed_size_mb'])
        return stats


def load_model(path: str | Path, use_compression: bool | None = None) -> Any:
    """
    Load a model with automatic compression detection and comprehensive fallbacks.
    
    This function uses the new adaptive loading strategy that handles multiple
    compression formats and provides detailed error reporting.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Use the new adaptive loading from compression_utils
    try:
        model = load_compressed_model(path)
        return model
    except Exception as e:
        # If compressed loading fails, provide helpful debugging info
        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.error(f"Failed to load model from {path} ({file_size_mb:.2f} MB): {e}")
        
        # Last resort: try standard joblib without any compression
        try:
            logger.warning("Attempting final fallback: standard joblib without compression...")
            model = joblib.load(path, mmap_mode=None)
            logger.info(f"Successfully loaded model using final fallback from {path}")
            return model
        except Exception as final_error:
            error_msg = (f"All loading strategies failed for {path}. "
                        f"Compressed loading error: {str(e)}. "
                        f"Final fallback error: {str(final_error)}")
            logger.error(error_msg)
            raise RuntimeError(error_msg)


def predict(model: Any, X_future: pd.DataFrame) -> np.ndarray:
    return model.predict(X_future)


__all__ = [
    "train_random_forest",
    "train_gb_models",
    "train_lightgbm",
    "save_model",
    "load_model",
    "predict",
] 