from __future__ import annotations

import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import preprocessing, features, model_utils, forecasting

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HORIZONS: Tuple[int, ...] = tuple(range(1, 8))
TRAIN_DIR = Path("data/training")
FORECAST_DIR = Path("data/forecasting")
PRED_DIR = Path("data/forecasting")  # save preds alongside forecasting CSVs

__all__ = ["train_granary", "forecast_heap"]


def _latest_model_path(granary_id: str) -> Path | None:
    pattern = f"deploy/models/g{granary_id}_lgbm_*.joblib"
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    return Path(paths[-1])


def train_granary(granary_id: str):
    """Train or retrain model for *granary_id*. Returns metrics & model path."""
    csv_path = TRAIN_DIR / f"g{granary_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, encoding="utf-8")
    df_proc = preprocessing.preprocess(df)
    X, y = features.select_feature_target_multi(df_proc, horizons=HORIZONS)
    mdl, metrics = model_utils.train_model(X, y)

    model_name = f"g{granary_id}_lgbm_{datetime.now():%Y%m%d_%H%M%S}.joblib"
    model_path = model_utils.save_model(mdl, model_name)
    logger.info("Model saved: %s", model_path)
    return metrics, model_path


def forecast_heap(granary_id: str, heap_id: str) -> Path:
    """Generate 7-day prediction CSV for given heap; returns path."""
    csv_path = FORECAST_DIR / f"g{granary_id}_h{heap_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    model_path = _latest_model_path(granary_id)
    if model_path is None:
        raise FileNotFoundError(f"No model found for granary {granary_id}. Train first.")

    mdl = model_utils.load_model(model_path)

    df = pd.read_csv(csv_path, encoding="utf-8")
    df_proc = preprocessing.preprocess(df)
    future_df = forecasting.make_future(df_proc, horizon_days=7)
    Xf, _ = features.select_feature_target_multi(future_df, horizons=HORIZONS, allow_na=True)
    Xf = Xf.reindex(columns=mdl.feature_names_in_, fill_value=0)

    preds = model_utils.predict(mdl, Xf)
    if preds.ndim == 2:
        for idx, h in enumerate(HORIZONS):
            future_df[f"pred_h{h}d"] = preds[:, idx]
        future_df["predicted_temp"] = preds[:, 0]
    else:
        future_df["predicted_temp"] = preds

    out_cols = [
        c for c in [
            "granary_id",
            "heap_id",
            "grid_x",
            "grid_y",
            "grid_z",
            "detection_time",
            "forecast_day",
            "predicted_temp",
        ] + [f"pred_h{h}d" for h in HORIZONS]
        if c in future_df.columns
    ]

    out_path = PRED_DIR / f"g{granary_id}_h{heap_id}_pred.csv"
    future_df[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote forecast CSV %s", out_path)
    return out_path 