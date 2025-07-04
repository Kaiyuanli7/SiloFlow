from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .multi_lgbm import MultiLGBMRegressor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Models directory inside deploy package
MODELS_DIR = (Path(__file__).resolve().parent / "models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_estimators: int = 1000,
    learning_rate: float = 0.03,
) -> Tuple[Any, dict]:
    mdl = MultiLGBMRegressor(
        base_params={
            "learning_rate": learning_rate,
            "max_depth": 7,
            "num_leaves": 24,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "min_child_samples": 20,
        },
        upper_bound_estimators=n_estimators,
        early_stopping_rounds=100,
    )
    # Internal 95/5 split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.05, shuffle=False)
    mdl.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)  # type: ignore[arg-type]
    y_pred = mdl.predict(X)
    metrics = {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
    }
    logger.info("LightGBM MAE %.3f RMSE %.3f", metrics["mae"], metrics["rmse"])
    return mdl, metrics


def save_model(model: Any, name: str) -> Path:
    p = MODELS_DIR / name
    joblib.dump(model, p)
    return p


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


# Unified entry point – always LightGBM multi-output

def train_model(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[Any, dict]:
    return train_lightgbm(X, y) 