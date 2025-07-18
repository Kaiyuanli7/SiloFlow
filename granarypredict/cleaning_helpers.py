from __future__ import annotations
import numpy as np
import pandas as pd

def insert_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* where any missing *calendar days* for each sensor are back-filled with synthetic rows so models see a continuous timeline."""
    if "detection_time" not in df.columns:
        return df
    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    if df["detection_time"].isna().all():
        return df
    df_valid = df[df["detection_time"].notna()].copy()
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df_valid.columns]
    if not group_cols:
        group_cols = []
    frames = [df_valid]
    static_like = set(group_cols + ["granary_id", "heap_id", "grain_type", "warehouse_type"])
    for key, sub in (df_valid.groupby(group_cols) if group_cols else [(None, df_valid)]):
        sub = sub.sort_values("detection_time")
        date_series = sub["detection_time"].dropna().dt.floor("D")
        if date_series.empty:
            continue
        start_date, end_date = date_series.min(), date_series.max()
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        if start_date == end_date:
            continue
        full_range = pd.date_range(start_date, end_date, freq="D")
        missing_dates = sorted(set(full_range.date) - set(date_series.dt.date.unique()))
        if not missing_dates:
            continue
        template = sub.iloc[-1].copy()
        new_rows = []
        for md in missing_dates:
            row = template.copy()
            row["detection_time"] = pd.Timestamp(md)
            for col in df_valid.select_dtypes(include=[np.number]).columns:
                if col not in static_like:
                    row[col] = np.nan
            new_rows.append(row)
        if new_rows:
            frames.append(pd.DataFrame(new_rows))
    df_full = pd.concat(frames, ignore_index=True)
    if df["detection_time"].isna().any():
        df_full = pd.concat([df_full, df[df["detection_time"].isna()]], ignore_index=True)
    return df_full

def interpolate_sensor_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """For each sensor group, linearly interpolate numeric columns along chronological order so values for synthetic gap rows equal the average of previous and next real measurements."""
    if "detection_time" not in df.columns:
        return df
    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    df.sort_values("detection_time", inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df
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

__all__ = [
    "insert_calendar_gaps",
    "interpolate_sensor_numeric",
]
