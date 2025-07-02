from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------
# TIME & SPATIAL FEATURES
# ------------------------------------------------------------

def create_time_features(df: pd.DataFrame, timestamp_col: str = "detection_time") -> pd.DataFrame:
    df = df.copy()
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' missing")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["year"] = df[timestamp_col].dt.year
    df["month"] = df[timestamp_col].dt.month
    df["day"] = df[timestamp_col].dt.day
    df["hour"] = df[timestamp_col].dt.hour
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    if "grid_index" in df.columns:
        return df.drop(columns=["grid_index"])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df

# ------------------------------------------------------------
# BASIC FEATURE / TARGET SELECTION
# ------------------------------------------------------------

def select_feature_target(
    df: pd.DataFrame,
    target_col: str = "temperature_grain",
    drop_cols: Tuple[str, ...] = (
        "avg_grain_temp",
        "avg_in_temp",
        "temperature_grain",
        "detection_time",
        "granary_id",
        "heap_id",
        "forecast_day",
    ),
) -> Tuple[pd.DataFrame, pd.Series]:
    if df[target_col].notna().any():
        mask = df[target_col].notna()
        X = df.loc[mask].drop(columns=list(drop_cols), errors="ignore")
        y = df.loc[mask, target_col]
    else:
        X = df.drop(columns=list(drop_cols), errors="ignore")
        y = pd.Series(dtype="float64")
    X = encode_categoricals(X)
    return X, y

# ------------------------------------------------------------
# SENSOR LAG & ROLLING FEATURES (1-day lag minimum)
# ------------------------------------------------------------

def add_sensor_lag(
    df: pd.DataFrame,
    *,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    lag_days: int = 1,
) -> pd.DataFrame:
    if {"grid_x", "grid_y", "grid_z", timestamp_col, temp_col}.issubset(df.columns):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
        df["_date"] = df[timestamp_col].dt.floor("D")
        lag_df = df[group_cols + ["_date", temp_col]].copy()
        lag_df["_date"] = lag_df["_date"] + pd.Timedelta(days=lag_days)
        lag_df.rename(columns={temp_col: "lag_temp_1d"}, inplace=True)
        df = df.merge(lag_df, on=group_cols + ["_date"], how="left")
        df.drop(columns=["_date"], inplace=True)
    return df


def add_multi_lag(
    df: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 3, 7, 14, 30),
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    df = add_sensor_lag(df, temp_col=temp_col, timestamp_col=timestamp_col)
    for d in lags:
        if d == 1:
            continue
        lag_col = f"lag_temp_{d}d"
        if {"grid_x", "grid_y", "grid_z", temp_col, timestamp_col}.issubset(df.columns):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
            df["_date"] = df[timestamp_col].dt.floor("D")
            lag_df = df[group_cols + ["_date", temp_col]].copy()
            lag_df["_date"] = lag_df["_date"] + pd.Timedelta(days=d)
            lag_df.rename(columns={temp_col: lag_col}, inplace=True)
            df = df.merge(lag_df, on=group_cols + ["_date"], how="left")
            df.drop(columns=["_date"], inplace=True)
            df[f"delta_temp_{d}d"] = df[temp_col] - df[lag_col]
    return df


def add_rolling_stats(
    df: pd.DataFrame,
    *,
    window_days: int = 7,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    if not {timestamp_col, temp_col}.issubset(df.columns):
        return df
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.set_index(timestamp_col, inplace=True)
    df[f"roll_mean_{window_days}d"] = (
        df.groupby([c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns])[temp_col]
        .transform(lambda s: s.rolling(f"{window_days}D", min_periods=1).mean())
    )
    df.reset_index(inplace=True)
    return df

# ------------------------------------------------------------
# MULTI-HORIZON TARGETS
# ------------------------------------------------------------

def add_multi_horizon_targets(
    df: pd.DataFrame,
    *,
    target_col: str = "temperature_grain",
    horizons: tuple[int, ...] = (1, 2, 3),
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    if timestamp_col not in df.columns or target_col not in df.columns:
        return df
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    for h in horizons:
        tgt_col = f"{target_col}_h{h}d"
        df[tgt_col] = df.groupby([c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns])[target_col].shift(-h)
    return df


def select_feature_target_multi(
    df: pd.DataFrame,
    *,
    target_col: str = "temperature_grain",
    horizons: tuple[int, ...] = (1, 2, 3),
    allow_na: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=["detection_time"], errors="ignore")
    y_cols = [f"{target_col}_h{h}d" for h in horizons if f"{target_col}_h{h}d" in df.columns]
    y = df[y_cols].copy()
    if not allow_na:
        mask = y.notna().all(axis=1)
        X = X.loc[mask]
        y = y.loc[mask]
    X = encode_categoricals(X)
    return X, y

__all__ = [
    "create_time_features",
    "create_spatial_features",
    "encode_categoricals",
    "select_feature_target",
    "add_sensor_lag",
    "add_multi_lag",
    "add_rolling_stats",
    "add_multi_horizon_targets",
    "select_feature_target_multi",
] 