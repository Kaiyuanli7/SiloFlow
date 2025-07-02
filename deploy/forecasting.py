from __future__ import annotations

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from . import features

TARGET_TEMP_COL = "temperature_grain"


def make_future(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    if df.empty or "detection_time" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    latest_ts = df["detection_time"].max()

    keep_cols = [c for c in df.columns if c in {"grid_x", "grid_y", "grid_z", "granary_id", "heap_id"}]
    sensors = df[keep_cols].drop_duplicates().reset_index(drop=True)

    frames: List[pd.DataFrame] = []
    for d in range(1, horizon_days + 1):
        tmp = sensors.copy()
        tmp["detection_time"] = latest_ts + timedelta(days=d)
        tmp["forecast_day"] = d
        frames.append(tmp)

    future_df = pd.concat(frames, ignore_index=True)

    future_df = features.create_time_features(future_df)
    future_df = features.create_spatial_features(future_df)
    future_df = features.add_sensor_lag(future_df)

    if TARGET_TEMP_COL not in future_df.columns:
        future_df[TARGET_TEMP_COL] = np.nan
    return future_df 