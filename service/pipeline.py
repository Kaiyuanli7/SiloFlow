from __future__ import annotations

"""Shared training & forecasting helpers for production service (per-granary models)."""

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, Tuple, List

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

from granarypredict import (
    ingestion,
    cleaning,
    features,
    model as model_utils,
)
from granarypredict.config import DATA_DIR, MODELS_DIR
from granarypredict.data_utils import comprehensive_sort, assign_group_id

logger = logging.getLogger("service.pipeline")

RAW_DIR = DATA_DIR / "raw" / "by_granary"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_DIR = DATA_DIR / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS: Tuple[int, int, int] = (1, 2, 3)  # days ahead

ENV_COLUMNS = [
    "temperature_inside",
    "temperature_outside",
    "humidity_warehouse",
    "humidity_outside",
]

FUTURE_SAFE_EXTRA = [
    "max_temp",
    "min_temp",
    "line_no",
    "layer_no",
]


def append_daily(df: pd.DataFrame) -> Path:
    """Append today's rows (all heaps) into the granary CSV; return path."""
    if "granary_id" not in df.columns:
        raise ValueError("granary_id missing after standardisation")
    granary = df["granary_id"].iat[0]
    out_path = RAW_DIR / f"{granary}.csv"

    if out_path.exists():
        base = pd.read_csv(out_path, encoding="utf-8")
        combo = pd.concat([base, df], ignore_index=True, sort=False)
        combo.drop_duplicates(
            subset=[
                "detection_time",
                "heap_id",
                "grid_x",
                "grid_y",
                "grid_z",
            ],
            inplace=True,
        )
    else:
        combo = df

    combo.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Granary %s → CSV rows=%d", granary, len(combo))
    return out_path


# --------------------------- Pre-processing ----------------------------------

def preprocess(df: pd.DataFrame, *, future_safe: bool = True) -> pd.DataFrame:
    df = cleaning.basic_clean(df)
    df = cleaning.fill_missing(df)
    df = features.create_time_features(df)
    df = features.create_spatial_features(df)
    df = features.add_sensor_lag(df)

    df = comprehensive_sort(df)
    df = assign_group_id(df)

    if future_safe:
        df = df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

    df = features.add_multi_horizon_targets(df, horizons=HORIZONS)
    return df


# --------------------------- Training ----------------------------------------

def train_model(full_df: pd.DataFrame, *, granary: str) -> Path:
    X, y = features.select_feature_target_multi(
        full_df, target_col="temperature_grain", horizons=HORIZONS
    )
    base = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.033,
        max_depth=7,
        num_leaves=24,
        n_jobs=-1,
    )
    mdl = MultiOutputRegressor(base)
    mdl.fit(X, y)

    model_file = f"{granary}_fs_lgbm.joblib"
    model_utils.save_model(mdl, name=model_file)
    return MODELS_DIR / model_file


# --------------------------- Forecasting -------------------------------------

def build_future_template(df: pd.DataFrame, *, days: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    last_ts = pd.to_datetime(df["detection_time"]).max()
    last_date = last_ts.normalize()

    frames: List[pd.DataFrame] = []
    max_cycle = df.get("detection_cycle").max() if "detection_cycle" in df.columns else None

    same_day_rows = df[df["detection_time"].dt.normalize() == last_date]
    for d in range(1, days + 1):
        tmp = same_day_rows.copy()
        tmp["detection_time"] = tmp["detection_time"] + pd.Timedelta(days=d)
        tmp["forecast_day"] = d
        if max_cycle is not None:
            tmp["detection_cycle"] = max_cycle + d
        frames.append(tmp)

    fut = pd.concat(frames, ignore_index=True)
    fut = features.create_time_features(fut)
    fut = features.create_spatial_features(fut)
    fut = features.add_sensor_lag(fut)
    return fut


def forecast(mdl, future_df: pd.DataFrame) -> pd.DataFrame:
    X_fut, _ = features.select_feature_target_multi(
        future_df, target_col="temperature_grain", horizons=HORIZONS, allow_na=True
    )
    preds = mdl.predict(X_fut)
    for idx, h in enumerate(HORIZONS):
        future_df[f"pred_h{h}d"] = preds[:, idx]
    return future_df


def save_forecast(granary: str, df: pd.DataFrame) -> Path:
    out_path = FORECAST_DIR / f"{granary}_{dt.date.today().isoformat()}_fc.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path 