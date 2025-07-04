from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from . import cleaning, features, data_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ENV_COLUMNS = [
    "temperature_inside",
    "temperature_outside",
    "humidity_warehouse",
    "humidity_outside",
]

# Additional columns unavailable for real future dates
FUTURE_SAFE_EXTRA = [
    "max_temp",
    "min_temp",
    "line_no",
    "layer_no",
]


# ---------------------------------------------------------------------
# Calendar-gap insertion & interpolation (simplified)
# ---------------------------------------------------------------------

def _insert_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
    if "detection_time" not in df.columns:
        return df
    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    frames: List[pd.DataFrame] = [df]
    for _, sub in df.groupby(group_cols) if group_cols else [(None, df)]:
        sub = sub.sort_values("detection_time")
        date_floor = sub["detection_time"].dt.floor("D")
        full_range = pd.date_range(date_floor.min(), date_floor.max(), freq="D")
        missing_dates = sorted(set(full_range.date) - set(date_floor.dt.date.unique()))
        if not missing_dates:
            continue
        template = sub.iloc[-1].copy()
        new_rows = []
        for md in missing_dates:
            row = template.copy()
            row["detection_time"] = pd.Timestamp(md)
            # Null numeric cols (measurement-like)
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in group_cols:
                    row[col] = np.nan
            new_rows.append(row)
        if new_rows:
            frames.append(pd.DataFrame(new_rows))
    return pd.concat(frames, ignore_index=True)


def _interpolate_sensor_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if "detection_time" not in df.columns:
        return df
    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    df.sort_values("detection_time", inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    if group_cols:
        df[num_cols] = (
            df.groupby(group_cols)[num_cols]
            .apply(lambda g: g.interpolate(method="linear", limit_direction="forward").ffill())
            .reset_index(level=group_cols, drop=True)
        )
    else:
        df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="forward").ffill()
    return df


# ---------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    logger.info("Starting preprocessing on shape=%s", df.shape)
    df = cleaning.basic_clean(df)
    df = _insert_calendar_gaps(df)
    df = _interpolate_sensor_numeric(df)
    # Always future-safe: remove env & extra columns
    df = df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")
    df = cleaning.fill_missing(df)
    df = features.create_time_features(df)
    df = features.create_spatial_features(df)
    df = features.add_multi_lag(df)
    df = features.add_rolling_stats(df, window_days=7)
    df = features.add_rolling_stats(df, window_days=30)
    df = features.add_multi_horizon_targets(df)
    df = data_utils.assign_group_id(df)
    df = data_utils.comprehensive_sort(df)
    logger.info("Preprocessing done – final shape=%s", df.shape)
    return df 