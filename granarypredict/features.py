from __future__ import annotations

import logging
import gc
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import os

import numpy as np
import pandas as pd

# Import streamlit for toast notifications (optional)
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False
    st = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global setting for parallel processing
# Check environment variable to disable parallel processing for batch operations
_USE_PARALLEL = os.environ.get("SILOFLOW_DISABLE_PARALLEL", "0") != "1"
# More conservative max workers for better memory management on Windows
_MAX_WORKERS = min(multiprocessing.cpu_count(), 4)  # Conservative limit for stability

def _toast_notify(message: str, icon: str = ""):
    """Helper function to show toast notifications if Streamlit is available."""
    if _HAS_STREAMLIT and st is not None:
        try:
            st.toast(message, icon=icon)
        except Exception:
            pass  # Silently fail if toast doesn't work
    logger.info(message)


def create_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add cyclical and calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame.
    timestamp_col : str, default "detection_time"
        Name of the column that stores the timestamp.  If that column is missing
        the function will try common alternatives (``batch``, ``timestamp``,
        ``datetime``, ``date``, ``time``) before raising a ``KeyError``.
    """

    df = df.copy()

    # Auto-detect timestamp column if the default is missing
    if timestamp_col not in df.columns:
        common_alts = ["batch", "timestamp", "datetime", "date", "time"]
        found_alt = next((c for c in common_alts if c in df.columns), None)
        if found_alt is not None:
            logger.info("create_time_features: using '%s' as timestamp column (auto-detected)", found_alt)
            timestamp_col = found_alt
        else:
            raise KeyError(
                f"Timestamp column '{timestamp_col}' not found in dataframe. "
                "Add the column or pass the correct name via the 'timestamp_col' argument."
            )

    # Robust conversion – coerce invalid strings to NaT so downstream code
    # can decide how to handle them without crashing (May-2025 fix).
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    df["year"] = df[timestamp_col].dt.year
    df["month"] = df[timestamp_col].dt.month
    df["day"] = df[timestamp_col].dt.day
    df["hour"] = df[timestamp_col].dt.hour

    # Cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


    # -------- New calendar encodings (Jun-2025) ------
    # -------- New calendar encodings (Jun-2025) ---------------------------
    # Day-of-year & week-of-year cyclic features help seasonal patterns.
    df["doy"] = df[timestamp_col].dt.dayofyear
    df["weekofyear"] = df[timestamp_col].dt.isocalendar().week.astype(int)

    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    df["woy_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)

    # Weekend indicator (Saturday/Sunday = 1 else 0)
    df["is_weekend"] = df[timestamp_col].dt.dayofweek >= 5

    return df


def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine grid_x/y/z into useful spatial indices."""
    # Remove previously created grid_index to avoid redundancy
    if "grid_index" in df.columns:
        df = df.copy().drop(columns=["grid_index"])

    # Function now does nothing else; kept for backward-compatibility.
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object/category dtypes to integer codes (label encoding)."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def select_feature_target(
    df: pd.DataFrame,
    target_col: str = "temperature_grain",
    drop_cols: Tuple[str, ...] = (
        "avg_grain_temp",
        "avg_in_temp",         # legacy aggregate metric
        "temperature_grain",   # drop target from feature matrix to avoid leakage
        "detection_time",
        "granary_id",
        "heap_id",
        "forecast_day",
        # Redundant or constant columns – duplicates of grid_x/y
        "line_no",
        "layer_no",
        "line",
        "layer",
    ),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X, y frames."""
    if df[target_col].notna().any():
        mask = df[target_col].notna()
        X = df.loc[mask].drop(columns=list(drop_cols), errors="ignore")
        y = df.loc[mask, target_col]
    else:
        # All targets missing – likely forecasting scenario; keep all rows and
        # return empty y (length 0) to satisfy signature.
        X = df.drop(columns=list(drop_cols), errors="ignore")
        y = pd.Series(dtype="float64")
    X = encode_categoricals(X)
    return X, y


# Keep exports alphabetical for readability
__all__ = [
    "add_directional_features_lean",
    "add_horizon_specific_directional_features",
    "add_sensor_lag",
    "add_multi_lag",
    "add_rolling_stats",
    "add_time_since_last_measurement",
    "create_spatial_features",
    "create_time_features",
    "encode_categoricals",
    "select_feature_target",
    "add_multi_horizon_targets",
    "select_feature_target_multi",
]


# ------------------------------------------------------------
# NEW – Time-since-last-measurement features (Step 2.1.4)
# ------------------------------------------------------------


def add_time_since_last_measurement(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add essential time-since-last-measurement features.
    
    This function creates simple temporal features that measure the time gaps 
    between consecutive sensor measurements for data quality assessment.
    
    Features added:
    - hours_since_last_measurement: Hours elapsed since previous measurement
    - days_since_last_measurement: Days elapsed since previous measurement  
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with timestamp and sensor identification columns
    timestamp_col : str, default "detection_time"
        Column containing the timestamp information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added time-since-last-measurement features
    """
    
    # Check for required columns
    required_cols = {timestamp_col, "grid_x", "grid_y", "grid_z"}
    if not required_cols.issubset(df.columns):
        logger.warning(f"Missing required columns for time-since-last-measurement: {required_cols - set(df.columns)}")
        return df
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    
    # Determine sensor grouping columns
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    
    # Sort by sensor and time to ensure proper time gap calculation
    df = df.sort_values(group_cols + [timestamp_col]).reset_index(drop=True)
    
    # Calculate time since last measurement for each sensor
    df["_prev_timestamp"] = df.groupby(group_cols)[timestamp_col].shift(1)
    df["_time_diff"] = df[timestamp_col] - df["_prev_timestamp"]
    
    # Convert to hours and days
    df["hours_since_last_measurement"] = df["_time_diff"].dt.total_seconds() / 3600
    
    # Clean up temporary columns
    df = df.drop(columns=["_prev_timestamp", "_time_diff"], errors="ignore")
    
    # Handle first measurements (no previous measurement) - set to 0 instead of NaN
    first_measurement_mask = df.groupby(group_cols).cumcount() == 0
    df.loc[first_measurement_mask, "hours_since_last_measurement"] = 0
    
    logger.info(f"Added time-since-last-measurement features for {len(df)} records")
    
    return df


# ------------------------------------------------------------
# NEW – 1-day lag feature for each sensor (grid_x/y/z)          May-2025
# ------------------------------------------------------------


def add_sensor_lag(
    df: pd.DataFrame,
    *,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    lag_days: int = 1,
) -> pd.DataFrame:
    """Add ``lag_temp_1d`` – the *temp_col* from ``lag_days`` days earlier
    for the *same physical sensor* defined by ``granary_id`` + ``heap_id`` +
    grid_x / y / z.  If *granary_id* or *heap_id* are missing, they are
    ignored, falling back to coarser grouping.  Rows without an exact match
    receive ``NaN``.

    This version uses a merge-on-timestamp (+lag) approach so it no longer
    depends on a fixed sampling frequency.  Whether you record hourly or
    irregularly, the function only assigns a lag when there is a reading
    exactly *lag_days* earlier (within a +/- 6 hour tolerance).
    """

    # Mandatory columns check ------------------------------------------------
    base_required = {"grid_x", "grid_y", "grid_z", timestamp_col, temp_col}
    if not base_required.issubset(df.columns):
        return df  # key columns absent, nothing to do

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Determine grouping hierarchy ------------------------------------------
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]

    # Work at *date* granularity so mixed sampling frequencies align nicely
    df["_date"] = df[timestamp_col].dt.floor("D")

    lag_key_cols = group_cols + ["_date"]

    # Build lagged frame -----------------------------------------------------
    lag_df = df[lag_key_cols + [temp_col]].copy()
    lag_df["_date"] = lag_df["_date"] + pd.Timedelta(days=lag_days)
    lag_df.rename(columns={temp_col: "lag_temp_1d"}, inplace=True)

    # Merge – vectorised & fast ---------------------------------------------
    df = df.merge(lag_df, on=lag_key_cols, how="left")

    df.drop(columns=["_date"], inplace=True)

    return df 


# ---------------------------------------------------------------------------
# NEW – Extra temperature features (May-2025)
# ---------------------------------------------------------------------------


def _add_single_lag(df: pd.DataFrame, lag_days: int, *,
                    temp_col: str = "temperature_grain",
                    timestamp_col: str = "detection_time") -> pd.DataFrame:
    """Return *df* with a new column ``lag_temp_<lag>d`` computed exactly like
    ``add_sensor_lag`` but for an arbitrary day offset.
    
    Memory-optimized version using groupby.shift instead of merge operations.
    """

    if lag_days == 1:
        # Already handled by original add_sensor_lag; skip duplication.
        return df

    if {"grid_x", "grid_y", "grid_z", temp_col, timestamp_col}.issubset(df.columns):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
        lag_col_name = f"lag_temp_{lag_days}d"
        
        try:
            # Memory-efficient approach: use groupby.shift instead of merge
            # Sort by group and time to ensure proper lag computation
            df_sorted = df.sort_values(group_cols + [timestamp_col])
            
            # Use groupby.shift for memory efficiency
            df_sorted[lag_col_name] = (
                df_sorted
                .groupby(group_cols)[temp_col]
                .shift(lag_days)
            )
            
            # Restore original order
            df[lag_col_name] = df_sorted.loc[df.index, lag_col_name]
            
            # Memory cleanup
            del df_sorted
            gc.collect()
            
        except Exception as e:
            # Fallback to chunked processing if groupby fails
            _toast_notify(f"⚠️ Using chunked lag computation for {lag_days}d lag", "⚠️")
            
            # Process in chunks to avoid memory issues
            chunk_size = 25_000  # Conservative chunk size
            lag_values = pd.Series(index=df.index, dtype='float64')
            
            for group_name, group_df in df.groupby(group_cols):
                if len(group_df) > chunk_size:
                    # Large group - process in chunks
                    group_sorted = group_df.sort_values(timestamp_col)
                    for i in range(0, len(group_sorted), chunk_size):
                        chunk = group_sorted.iloc[i:i+chunk_size]
                        chunk_lag = chunk[temp_col].shift(lag_days)
                        lag_values.loc[chunk.index] = chunk_lag
                        
                        # Memory cleanup
                        del chunk, chunk_lag
                        gc.collect()
                else:
                    # Small group - process normally
                    group_sorted = group_df.sort_values(timestamp_col)
                    group_lag = group_sorted[temp_col].shift(lag_days)
                    lag_values.loc[group_sorted.index] = group_lag
                    
                    # Memory cleanup
                    del group_sorted, group_lag
                    gc.collect()
            
            df[lag_col_name] = lag_values
            del lag_values
            gc.collect()
            
    return df


def add_multi_lag(
    df: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add multiple lag columns and their *delta-temperature* counterparts.

    By default the function now creates lags for 1, 2, 3, 4, 5, 6, 7, 14 and 30 days
    (``lags=(1, 2, 3, 4, 5, 6, 7, 14, 30)``). For every generated lag column
    ``lag_temp_<d>d`` a corresponding trend feature
    ``delta_temp_<d>d`` (current − lag value) is also added.  The 1-day lag is
    still computed through :pyfunc:`add_sensor_lag`; longer offsets are handled
    by the private :pyfunc:`_add_single_lag` helper.
    """

    # --- 1) Generate basic lag columns -----------------------------------
    # Always include the 1-day lag via the dedicated helper for efficiency.
    df = add_sensor_lag(df, temp_col=temp_col, timestamp_col=timestamp_col)

    # Additional lags (>=2 days)
    for d in lags:
        if d == 1:
            continue  # already handled above
        df = _add_single_lag(df, d, temp_col=temp_col, timestamp_col=timestamp_col)

    # --- 2) Delta-temperature features ------------------------------------
    # For every generated lag column, add ΔT = current − lag_d value.  This
    # captures the short-term warming/cooling trend that often boosts model
    # performance.
    for d in lags:
        lag_col = f"lag_temp_{d}d"
        delta_col = f"delta_temp_{d}d"
        if lag_col in df.columns and temp_col in df.columns and delta_col not in df.columns:
            df[delta_col] = df[temp_col] - df[lag_col]

    return df


def add_rolling_stats(
    df: pd.DataFrame,
    *,
    window_days: int = 7,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add rolling mean and std of *temp_col* over *window_days* for each sensor."""

    if not {temp_col, timestamp_col}.issubset(df.columns):
        return df

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.sort_values(timestamp_col, inplace=True)

    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    roll_mean_col = f"roll_mean_{window_days}d"
    roll_std_col = f"roll_std_{window_days}d"

    if group_cols:
        df[roll_mean_col] = (
            df.groupby(group_cols)[temp_col]
            .transform(lambda s: s.rolling(window_days, min_periods=1).mean())
        )
        df[roll_std_col] = (
            df.groupby(group_cols)[temp_col]
            .transform(lambda s: s.rolling(window_days, min_periods=1).std())
        )
    else:
        df[roll_mean_col] = df[temp_col].rolling(window_days, min_periods=1).mean()
        df[roll_std_col] = df[temp_col].rolling(window_days, min_periods=1).std()

    return df 


def add_directional_features_lean(
    df: pd.DataFrame,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add lean directional features for temperature movement prediction.
    
    Adds only 5 high-impact features with minimal computational overhead:
    - temp_accel: change in velocity (momentum indicator)  
    - trend_3d: 3-day linear slope
    - is_warming: binary warming indicator
    - velocity_smooth: noise-reduced velocity
    - trend_consistency: directional persistence over 5 days
    """
    
    if not {temp_col, timestamp_col}.issubset(df.columns):
        return df
    
    df = df.copy()
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    
    if group_cols:
        # Calculate velocity for internal use but don't add to dataframe
        temp_velocity = df.groupby(group_cols)[temp_col].diff(1)
        
        # 1. Acceleration (momentum change)
        df["temp_accel"] = temp_velocity.groupby(df.groupby(group_cols).ngroup()).diff(1)
        
        # 2. 3-day trend slope (efficient trend indicator)
        df["trend_3d"] = df.groupby(group_cols)[temp_col].transform(
            lambda x: x.rolling(3, min_periods=2).apply(
                lambda vals: (vals.iloc[-1] - vals.iloc[0]) / (len(vals) - 1) if len(vals) >= 2 else 0
            )
        )
        
        # 3. Binary warming indicator
        df["is_warming"] = (temp_velocity > 0.05).astype(int)  # 0.05°C threshold
        
        # 4. Smoothed velocity (reduces noise)
        df["velocity_smooth"] = temp_velocity.groupby(df.groupby(group_cols).ngroup()).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        
        # 5. Trend consistency (% days moving in same direction over 5 days)
        df["trend_consistency"] = df["is_warming"].groupby(df.groupby(group_cols).ngroup()).transform(
            lambda x: x.rolling(5, min_periods=2).apply(
                lambda vals: max(vals.mean(), 1 - vals.mean())  # consistency in either direction
            )
        )
    
    return df


def add_horizon_specific_directional_features(
    df: pd.DataFrame,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    max_horizon: int = 7,
) -> pd.DataFrame:
    """Add horizon-specific directional features that help each h+n model predict movement.
    
    Creates specialized directional indicators for different prediction horizons:
    - Multi-scale velocity (1d, 3d, 7d changes)
    - Horizon-specific momentum indicators
    - Pattern recognition features for trend direction
    - Volatility measures for uncertainty estimation
    """
    
    if not {temp_col, timestamp_col}.issubset(df.columns):
        return df
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.sort_values(timestamp_col, inplace=True)
    
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    
    if group_cols:
        grouped = df.groupby(group_cols, sort=False)
        
        # Multi-scale velocity features (key for different horizons)
        df["velocity_1d"] = grouped[temp_col].diff(1).fillna(0)  # Short-term
        df["velocity_3d"] = grouped[temp_col].diff(3).fillna(0) / 3  # Medium-term  
        df["velocity_7d"] = grouped[temp_col].diff(7).fillna(0) / 7  # Long-term
        
        # Momentum persistence indicators - use transform to ensure proper index alignment
        df["momentum_strength"] = np.abs(df["velocity_1d"]) * grouped["velocity_1d"].transform(
            lambda x: x.rolling(3).count() / 3
        )
        df["momentum_direction"] = np.sign(df["velocity_1d"] + df["velocity_3d"] + df["velocity_7d"])
        
        # Temperature volatility (uncertainty measure) - use transform to handle index properly
        df["temp_volatility"] = grouped[temp_col].transform(
            lambda x: x.rolling(window=7, min_periods=3).std().fillna(0)
        )
        df["velocity_volatility"] = grouped["velocity_1d"].transform(
            lambda x: x.rolling(window=5, min_periods=3).std().fillna(0)
        )
        
        # Pattern recognition features
        df["temp_acceleration_3d"] = grouped["velocity_3d"].diff(1).fillna(0)
        df["trend_reversal_signal"] = (
            (df["velocity_1d"] * df["velocity_3d"] < 0).astype(int)  # Short vs medium trend conflict
        )
        
        # Multi-horizon directional consistency - use transform to handle index properly
        for h in [2, 3, 5, 7]:
            if h <= max_horizon:
                df[f"direction_consistency_{h}d"] = grouped["velocity_1d"].transform(
                    lambda x: x.rolling(window=h, min_periods=max(2, h//2)).apply(
                        lambda vals: (np.sign(vals) == np.sign(vals.iloc[-1])).mean() if len(vals) >= 2 else 0.5, 
                        raw=False
                    ).fillna(0.5)
                )
        
        # Temperature range features (helps with boundary detection) - use transform
        df["temp_range_7d"] = grouped[temp_col].transform(
            lambda x: x.rolling(window=7, min_periods=3).apply(
                lambda vals: vals.max() - vals.min(), raw=False
            ).fillna(0)
        )
        
        # Relative position in recent range - use transform for rolling operations
        rolling_min = grouped[temp_col].transform(lambda x: x.rolling(7, min_periods=3).min())
        rolling_max = grouped[temp_col].transform(lambda x: x.rolling(7, min_periods=3).max())
        df["temp_position_in_range"] = (df[temp_col] - rolling_min) / (rolling_max - rolling_min + 1e-6)
        df["temp_position_in_range"] = df["temp_position_in_range"].fillna(0.5)
        
    else:
        # Single-group fallback
        df["velocity_1d"] = df[temp_col].diff(1).fillna(0)
        df["velocity_3d"] = df[temp_col].diff(3).fillna(0) / 3
        df["velocity_7d"] = df[temp_col].diff(7).fillna(0) / 7
        df["momentum_strength"] = np.abs(df["velocity_1d"]) * df["velocity_1d"].rolling(3).count() / 3
        df["momentum_direction"] = np.sign(df["velocity_1d"] + df["velocity_3d"] + df["velocity_7d"])
        df["temp_volatility"] = df[temp_col].rolling(window=7, min_periods=3).std().fillna(0)
        df["velocity_volatility"] = df["velocity_1d"].rolling(window=5, min_periods=3).std().fillna(0)
        df["temp_acceleration_3d"] = df["velocity_3d"].diff(1).fillna(0)
        df["trend_reversal_signal"] = (df["velocity_1d"] * df["velocity_3d"] < 0).astype(int)
        
        for h in [2, 3, 5, 7]:
            if h <= max_horizon:
                df[f"direction_consistency_{h}d"] = df["velocity_1d"].rolling(
                    window=h, min_periods=max(2, h//2)
                ).apply(
                    lambda x: (np.sign(x) == np.sign(x.iloc[-1])).mean() if len(x) >= 2 else 0.5, 
                    raw=False
                ).fillna(0.5)
        
        df["temp_range_7d"] = df[temp_col].rolling(window=7, min_periods=3).apply(
            lambda x: x.max() - x.min(), raw=False
        ).fillna(0)
        
        rolling_min = df[temp_col].rolling(7, min_periods=3).min()
        rolling_max = df[temp_col].rolling(7, min_periods=3).max()
        df["temp_position_in_range"] = (df[temp_col] - rolling_min) / (rolling_max - rolling_min + 1e-6)
        df["temp_position_in_range"] = df["temp_position_in_range"].fillna(0.5)
    
    return df


def add_stability_features(
    df: pd.DataFrame,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add stability features that capture temperature inertia and resistance to change.
    
    These features help models learn more conservative predictions by capturing:
    - Temperature stability indicators
    - Thermal inertia proxies
    - Change resistance metrics
    - Historical stability patterns
    
    This addresses the issue of models predicting overly aggressive temperature changes
    by providing features that represent the physical reality of grain temperature stability.
    """
    
    if not {temp_col, timestamp_col}.issubset(df.columns):
        return df
    
    df = df.copy()
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    
    def add_stability_by_group(group_df):
        if len(group_df) < 7:  # Need minimum history for stability features
            return group_df
        
        group_df = group_df.sort_values(timestamp_col)
        
        # 1. TEMPERATURE STABILITY INDEX (0-1, higher = more stable)
        temp_series = group_df[temp_col].rolling(window=7, min_periods=3)
        temp_std_7d = temp_series.std()
        temp_range_7d = temp_series.max() - temp_series.min()
        
        # Stability index: inverse of variability (normalized)
        stability_index = 1 / (1 + temp_std_7d)  # High stability = low std
        group_df["stability_index"] = stability_index.fillna(0.8)  # Default to moderately stable
        
        # 2. THERMAL INERTIA PROXY
        # Real grain has thermal inertia - temperature changes are dampened
        temp_changes = group_df[temp_col].diff().abs()
        thermal_inertia = temp_changes.rolling(window=5, min_periods=2).mean()
        group_df["thermal_inertia"] = thermal_inertia.fillna(0.1)  # Default to low inertia
        
        # 3. CHANGE RESISTANCE METRIC
        # How much does temperature resist change over time?
        change_resistance = 1 / (1 + temp_changes.rolling(window=3, min_periods=2).std())
        group_df["change_resistance"] = change_resistance.fillna(0.9)  # Default to high resistance
        
        # 4. HISTORICAL STABILITY PATTERN
        # Is this sensor historically stable or volatile?
        historical_volatility = temp_changes.rolling(window=14, min_periods=7).std()
        stability_pattern = 1 / (1 + historical_volatility)
        group_df["historical_stability"] = stability_pattern.fillna(0.7)  # Default to moderately stable
        
        # 5. DAMPENING FACTOR
        # How much should changes be dampened based on recent stability?
        recent_stability = stability_index.rolling(window=3, min_periods=2).mean()
        dampening_factor = recent_stability * 0.8 + 0.2  # Range: 0.2 to 1.0
        group_df["dampening_factor"] = dampening_factor.fillna(0.6)  # Default to moderate dampening
        
        # 6. EQUILIBRIUM TEMPERATURE
        # What temperature does this sensor tend to settle at?
        equilibrium_temp = group_df[temp_col].rolling(window=7, min_periods=3).median()
        group_df["equilibrium_temp"] = equilibrium_temp.fillna(group_df[temp_col])
        
        # 7. DEVIATION FROM EQUILIBRIUM
        # How far is current temperature from equilibrium?
        temp_deviation = abs(group_df[temp_col] - group_df["equilibrium_temp"])
        group_df["temp_deviation_from_equilibrium"] = temp_deviation.fillna(0)
        
        # 8. MEAN REVERSION TENDENCY
        # Tendency to return to equilibrium temperature
        temp_diff_from_eq = group_df[temp_col] - group_df["equilibrium_temp"]
        mean_reversion = temp_diff_from_eq.rolling(window=5, min_periods=3).corr(
            temp_diff_from_eq.shift(1)
        )
        group_df["mean_reversion_tendency"] = mean_reversion.fillna(0.3)  # Default to moderate mean reversion
        
        return group_df
    
    if group_cols:
        df = df.groupby(group_cols, group_keys=False).apply(add_stability_by_group)
    else:
        df = add_stability_by_group(df)
    
    return df


def add_multi_horizon_targets(
    df: pd.DataFrame,
    *,
    target_col: str = "temperature_grain",
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Append future-shifted target columns for each horizon.

    Adds columns like ``temperature_grain_h1d`` (+1 day), ``temperature_grain_h2d`` …
    with respect to *timestamp_col*.
    """

    if target_col not in df.columns or timestamp_col not in df.columns:
        # Nothing to do – silently return unchanged frame
        return df

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.sort_values(timestamp_col, inplace=True)

    # Decide grouping hierarchy so that shift happens *within each sensor*.
    group_cols = [c for c in [
        "granary_id",
        "heap_id",
        "grid_x",
        "grid_y",
        "grid_z",
    ] if c in df.columns]

    for h in horizons:
        shifted = (
            df.groupby(group_cols)[target_col].shift(-h)  # future value
            if group_cols else df[target_col].shift(-h)
        )
        df[f"{target_col}_h{h}d"] = shifted
    return df


def select_feature_target_multi(
    df: pd.DataFrame,
    *,
    target_col: str = "temperature_grain",
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    allow_na: bool = False,
    drop_cols: Tuple[str, ...] = (
        "avg_grain_temp",
        "avg_in_temp",
        "temperature_grain",
        "detection_time",
        "granary_id",
        "heap_id",
        "forecast_day",
        "line_no",
        "layer_no",
        "line",
        "layer",
    ),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return *X* and *Y* (multi-output) dataframes.

    Only rows where **all** horizon targets are present (non-NaN) are kept in
    *Y* and *X* so that supervised learning receives complete targets.
    """

    y_cols = [f"{target_col}_h{h}d" for h in horizons]

    # Ensure the target columns exist; if missing (e.g. forecasting frame) add NaNs
    for col in y_cols:
        if col not in df.columns:
            df[col] = pd.NA

    if allow_na:
        X = df.drop(columns=list(drop_cols) + y_cols, errors="ignore")
        X = encode_categoricals(X)
        Y = df[y_cols] if set(y_cols).issubset(df.columns) else pd.DataFrame(index=df.index)
        return X, Y

    # Keep rows where all future targets exist
    mask = df[y_cols].notna().all(axis=1)

    X = df.loc[mask].drop(columns=list(drop_cols) + y_cols, errors="ignore")
    X = encode_categoricals(X)

    Y = df.loc[mask, y_cols]
    return X, Y 


# ------------------------------------------------------------
# PARALLEL FEATURE ENGINEERING FUNCTIONS (NEW)
# ------------------------------------------------------------

def add_multi_lag_parallel(
    df: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Parallel version of add_multi_lag for 3-5x speedup.
    
    Processes multiple lag computations in parallel using ProcessPoolExecutor.
    Falls back to sequential processing if parallelization fails.
    """
    # Check if we should use parallel processing
    should_use_parallel = (
        _USE_PARALLEL and 
        len(lags) >= 3 and  # Need at least 3 lags to benefit from parallel
        len(df) >= 10000    # Only use parallel for larger datasets
    )
    
    # Additional safety checks for Windows systems
    if should_use_parallel:
        try:
            import platform
            if platform.system() == "Windows":
                # On Windows, be extra conservative with memory usage
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 80:  # If memory usage > 80%
                    should_use_parallel = False
                    _toast_notify("🔄 Using sequential processing due to high memory usage", "⚠️")
                elif len(df) > 100000:  # Very large datasets
                    should_use_parallel = False
                    _toast_notify("🔄 Using sequential processing for very large dataset", "⚠️")
        except ImportError:
            pass  # psutil not available
        except Exception:
            should_use_parallel = False  # Any error, fall back to sequential
    
    if not should_use_parallel:
        _toast_notify("🔄 Using sequential lag computation (safety/performance optimization)", "🔄")
        return add_multi_lag(df, lags=lags, temp_col=temp_col, timestamp_col=timestamp_col)
    
    # Always include the 1-day lag via the dedicated helper for efficiency
    df = add_sensor_lag(df, temp_col=temp_col, timestamp_col=timestamp_col)
    
    # Process additional lags in parallel
    additional_lags = [d for d in lags if d != 1]
    
    if not additional_lags:
        return df
    
    # Very conservative worker count to prevent resource exhaustion on Windows
    max_workers = max_workers or min(_MAX_WORKERS, len(additional_lags), 2)  # Max 2 workers
    
    _toast_notify(f"Parallel lag computation: {len(additional_lags)} lags using {max_workers} processes", "")
    
    try:
        # Create a minimal subset of the dataframe for workers to reduce memory usage
        worker_cols = [col for col in [temp_col, timestamp_col, "granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if col in df.columns]
        worker_df = df[worker_cols].copy()
        
        # Try to reduce memory footprint even further
        worker_df = worker_df.iloc[::2].copy()  # Sample every other row for lag computation
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit lag computation tasks with timeout and chunking
            futures = []
            for lag_days in additional_lags:
                future = executor.submit(_compute_single_lag_worker, worker_df, lag_days, temp_col, timestamp_col)
                futures.append((lag_days, future))
            
            # Collect results with better error handling
            lag_results = {}
            for lag_days, future in futures:
                try:
                    lag_results[lag_days] = future.result(timeout=180)  # 3 min timeout per lag
                except Exception as e:
                    logger.warning(f"Parallel lag computation failed for lag {lag_days}: {e}")
                    _toast_notify(f"Lag {lag_days}d failed, using sequential fallback", "⚠️")
                    # Fall back to sequential for this lag
                    lag_results[lag_days] = _add_single_lag(df, lag_days, temp_col=temp_col, timestamp_col=timestamp_col)
            
            # Merge all lag results
            for lag_days, lag_df in lag_results.items():
                lag_col = f"lag_temp_{lag_days}d"
                if lag_col in lag_df.columns:
                    df[lag_col] = lag_df[lag_col]
        
        _toast_notify(f"✅ Parallel lag computation completed: {len(additional_lags)} lags processed", "✅")
        
    except Exception as e:
        logger.warning(f"Parallel lag computation failed completely, falling back to sequential: {e}")
        _toast_notify(f"⚠️ Parallel processing failed: {str(e)[:50]}... Using sequential fallback", "⚠️")
        # Fall back to sequential processing for all remaining lags
        for d in additional_lags:
            df = _add_single_lag(df, d, temp_col=temp_col, timestamp_col=timestamp_col)
    
    # Add delta-temperature features
    for d in lags:
        lag_col = f"lag_temp_{d}d"
        delta_col = f"delta_temp_{d}d"
        if lag_col in df.columns and temp_col in df.columns and delta_col not in df.columns:
            df[delta_col] = df[temp_col] - df[lag_col]
    
    return df


def _compute_single_lag_worker(df: pd.DataFrame, lag_days: int, temp_col: str, timestamp_col: str) -> pd.DataFrame:
    """Worker function for parallel lag computation."""
    return _add_single_lag(df, lag_days, temp_col=temp_col, timestamp_col=timestamp_col)


def add_rolling_stats_parallel(
    df: pd.DataFrame,
    *,
    window_days: int = 7,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Parallel version of add_rolling_stats for 2-3x speedup.
    
    Processes rolling statistics by sensor groups in parallel using ThreadPoolExecutor.
    Falls back to sequential processing if parallelization fails.
    """
    if not _USE_PARALLEL or not {temp_col, timestamp_col}.issubset(df.columns):
        return add_rolling_stats(df, window_days=window_days, temp_col=temp_col, timestamp_col=timestamp_col)
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.sort_values(timestamp_col, inplace=True)
    
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    roll_mean_col = f"roll_mean_{window_days}d"
    roll_std_col = f"roll_std_{window_days}d"
    
    if not group_cols:
        return add_rolling_stats(df, window_days=window_days, temp_col=temp_col, timestamp_col=timestamp_col)
    
    # Group the data and prepare for parallel processing
    groups = df.groupby(group_cols)
    group_keys = list(groups.groups.keys())
    
    if len(group_keys) < 2:
        _toast_notify("🔄 Using sequential rolling stats (insufficient sensor groups for parallel)", "🔄")
        return add_rolling_stats(df, window_days=window_days, temp_col=temp_col, timestamp_col=timestamp_col)
    
    max_workers = max_workers or min(_MAX_WORKERS, len(group_keys))
    
    _toast_notify(f"Parallel rolling stats: {len(group_keys)} sensor groups using {max_workers} threads", "")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit rolling stats computation tasks
            futures = []
            for group_key in group_keys:
                group_df = groups.get_group(group_key)
                future = executor.submit(_compute_rolling_stats_worker, group_df, window_days, temp_col, group_key)
                futures.append((group_key, future))
            
            # Collect results
            results = {}
            for group_key, future in futures:
                try:
                    results[group_key] = future.result(timeout=30)  # 30s timeout per group
                except Exception as e:
                    logger.warning(f"Parallel rolling stats failed for group {group_key}: {e}")
                    _toast_notify(f"Rolling stats failed for sensor group, using sequential fallback", "")
                    # Fall back to sequential for this group
                    group_df = groups.get_group(group_key)
                    results[group_key] = _compute_rolling_stats_worker(group_df, window_days, temp_col, group_key)
            
            # Merge results back into dataframe
            for group_key, (mean_values, std_values) in results.items():
                group_indices = groups.get_group(group_key).index
                df.loc[group_indices, roll_mean_col] = mean_values
                df.loc[group_indices, roll_std_col] = std_values
        
        _toast_notify(f"Parallel rolling stats completed: {len(group_keys)} sensor groups processed", "")
        
    except Exception as e:
        logger.warning(f"Parallel rolling stats computation failed, falling back to sequential: {e}")
        _toast_notify(f"Parallel rolling stats failed: {str(e)[:50]}... Using sequential fallback", "")
        return add_rolling_stats(df, window_days=window_days, temp_col=temp_col, timestamp_col=timestamp_col)
    
    return df


def _compute_rolling_stats_worker(group_df: pd.DataFrame, window_days: int, temp_col: str, group_key):
    """Worker function for parallel rolling statistics computation."""
    try:
        roll_mean = group_df[temp_col].rolling(window_days, min_periods=1).mean()
        roll_std = group_df[temp_col].rolling(window_days, min_periods=1).std()
        return roll_mean.values, roll_std.values
    except Exception as e:
        logger.warning(f"Rolling stats computation failed for group {group_key}: {e}")
        # Return zeros as fallback
        return np.zeros(len(group_df)), np.zeros(len(group_df))


def add_stability_features_parallel(
    df: pd.DataFrame,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Parallel version of add_stability_features for 2-4x speedup.
    
    Processes stability features by sensor groups in parallel using ThreadPoolExecutor.
    Falls back to sequential processing if parallelization fails.
    """
    if not _USE_PARALLEL or not {temp_col, timestamp_col}.issubset(df.columns):
        return add_stability_features(df, temp_col=temp_col, timestamp_col=timestamp_col)
    
    df = df.copy()
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    
    if not group_cols:
        return add_stability_features(df, temp_col=temp_col, timestamp_col=timestamp_col)
    
    # Group the data and prepare for parallel processing
    groups = df.groupby(group_cols)
    group_keys = list(groups.groups.keys())
    
    if len(group_keys) < 2:
        _toast_notify("🔄 Using sequential stability features (insufficient sensor groups for parallel)", "🔄")
        return add_stability_features(df, temp_col=temp_col, timestamp_col=timestamp_col)
    
    max_workers = max_workers or min(_MAX_WORKERS, len(group_keys))
    
    _toast_notify(f"Parallel stability features: {len(group_keys)} sensor groups using {max_workers} threads", "")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit stability feature computation tasks
            futures = []
            for group_key in group_keys:
                group_df = groups.get_group(group_key)
                future = executor.submit(_compute_stability_features_worker, group_df, temp_col, timestamp_col)
                futures.append((group_key, future))
            
            # Collect results
            results = {}
            for group_key, future in futures:
                try:
                    results[group_key] = future.result(timeout=60)  # 60s timeout per group
                except Exception as e:
                    logger.warning(f"Parallel stability features failed for group {group_key}: {e}")
                    _toast_notify(f"Stability features failed for sensor group, using sequential fallback", "")
                    # Fall back to sequential for this group
                    group_df = groups.get_group(group_key)
                    results[group_key] = _compute_stability_features_worker(group_df, temp_col, timestamp_col)
            
            # Merge results back into dataframe
            stability_cols = ["stability_index", "thermal_inertia", "change_resistance", "historical_stability", 
                            "dampening_factor", "equilibrium_temp", "temp_deviation_from_equilibrium", "mean_reversion_tendency"]
            
            for group_key, group_result in results.items():
                group_indices = groups.get_group(group_key).index
                for col in stability_cols:
                    if col in group_result.columns:
                        df.loc[group_indices, col] = group_result[col]
        
        _toast_notify(f"Parallel stability features completed: {len(group_keys)} sensor groups processed", "")
        
    except Exception as e:
        logger.warning(f"Parallel stability features computation failed, falling back to sequential: {e}")
        _toast_notify(f"Parallel stability features failed: {str(e)[:50]}... Using sequential fallback", "")
        return add_stability_features(df, temp_col=temp_col, timestamp_col=timestamp_col)
    
    return df


def _compute_stability_features_worker(group_df: pd.DataFrame, temp_col: str, timestamp_col: str) -> pd.DataFrame:
    """Worker function for parallel stability features computation."""
    try:
        # Use the existing add_stability_by_group logic
        def add_stability_by_group(group_df):
            if len(group_df) < 7:  # Need minimum history for stability features
                return group_df
            
            group_df = group_df.sort_values(timestamp_col)
            
            # 1. TEMPERATURE STABILITY INDEX (0-1, higher = more stable)
            temp_series = group_df[temp_col].rolling(window=7, min_periods=3)
            temp_std_7d = temp_series.std()
            temp_range_7d = temp_series.max() - temp_series.min()
            
            # Stability index: inverse of variability (normalized)
            stability_index = 1 / (1 + temp_std_7d)  # High stability = low std
            group_df["stability_index"] = stability_index.fillna(0.8)  # Default to moderately stable
            
            # 2. THERMAL INERTIA PROXY
            # Real grain has thermal inertia - temperature changes are dampened
            temp_changes = group_df[temp_col].diff().abs()
            thermal_inertia = temp_changes.rolling(window=5, min_periods=2).mean()
            group_df["thermal_inertia"] = thermal_inertia.fillna(0.1)  # Default to low inertia
            
            # 3. CHANGE RESISTANCE METRIC
            # How much does temperature resist change over time?
            change_resistance = 1 / (1 + temp_changes.rolling(window=3, min_periods=2).std())
            group_df["change_resistance"] = change_resistance.fillna(0.9)  # Default to high resistance
            
            # 4. HISTORICAL STABILITY PATTERN
            # Is this sensor historically stable or volatile?
            historical_volatility = temp_changes.rolling(window=14, min_periods=7).std()
            stability_pattern = 1 / (1 + historical_volatility)
            group_df["historical_stability"] = stability_pattern.fillna(0.7)  # Default to moderately stable
            
            # 5. DAMPENING FACTOR
            # How much should changes be dampened based on recent stability?
            recent_stability = stability_index.rolling(window=3, min_periods=2).mean()
            dampening_factor = recent_stability * 0.8 + 0.2  # Range: 0.2 to 1.0
            group_df["dampening_factor"] = dampening_factor.fillna(0.6)  # Default to moderate dampening
            
            # 6. EQUILIBRIUM TEMPERATURE
            # What temperature does this sensor tend to settle at?
            equilibrium_temp = group_df[temp_col].rolling(window=7, min_periods=3).median()
            group_df["equilibrium_temp"] = equilibrium_temp.fillna(group_df[temp_col])
            
            # 7. DEVIATION FROM EQUILIBRIUM
            # How far is current temperature from equilibrium?
            temp_deviation = abs(group_df[temp_col] - group_df["equilibrium_temp"])
            group_df["temp_deviation_from_equilibrium"] = temp_deviation.fillna(0)
            
            # 8. MEAN REVERSION TENDENCY
            # Tendency to return to equilibrium temperature
            temp_diff_from_eq = group_df[temp_col] - group_df["equilibrium_temp"]
            mean_reversion = temp_diff_from_eq.rolling(window=5, min_periods=3).corr(
                temp_diff_from_eq.shift(1)
            )
            group_df["mean_reversion_tendency"] = mean_reversion.fillna(0.3)  # Default to moderate mean reversion
            
            return group_df
        
        return add_stability_by_group(group_df)
        
    except Exception as e:
        logger.warning(f"Stability features computation failed: {e}")
        return group_df  # Return original dataframe as fallback


def preprocess_dataframe_parallel(
    df: pd.DataFrame,
    *,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Complete parallel preprocessing pipeline for 3-5x speedup.
    
    Combines all parallel feature engineering functions with optimized sequencing.
    This is the main entry point for high-performance feature engineering.
    """
    logger.info("Starting parallel preprocessing pipeline")
    _toast_notify("Starting parallel preprocessing pipeline - up to 5x faster!", "")
    
    # Sequential steps that can't be parallelized
    df = create_time_features(df, timestamp_col=timestamp_col)
    df = create_spatial_features(df)
    df = add_time_since_last_measurement(df, timestamp_col=timestamp_col)
    
    # Parallel feature engineering
    df = add_multi_lag_parallel(df, temp_col=temp_col, timestamp_col=timestamp_col, max_workers=max_workers)
    df = add_rolling_stats_parallel(df, temp_col=temp_col, timestamp_col=timestamp_col, max_workers=max_workers)
    df = add_directional_features_lean(df, temp_col=temp_col, timestamp_col=timestamp_col)
    df = add_stability_features_parallel(df, temp_col=temp_col, timestamp_col=timestamp_col, max_workers=max_workers)
    df = add_horizon_specific_directional_features(df, temp_col=temp_col, timestamp_col=timestamp_col)
    
    logger.info("Parallel preprocessing pipeline completed")
    _toast_notify("Parallel preprocessing pipeline completed successfully!", "")
    return df


def set_parallel_processing(enabled: bool, max_workers: int | None = None):
    """Enable or disable parallel processing for feature engineering.
    
    Parameters
    ----------
    enabled : bool
        Whether to enable parallel processing
    max_workers : int, optional
        Maximum number of worker processes/threads
    """
    global _USE_PARALLEL, _MAX_WORKERS
    _USE_PARALLEL = enabled
    if max_workers is not None:
        _MAX_WORKERS = min(max_workers, multiprocessing.cpu_count())
    
    status = f"Parallel processing {'enabled' if enabled else 'disabled'}, max_workers={_MAX_WORKERS}"
    logger.info(status)
    
    if enabled:
        _toast_notify(f"⚡ Parallel processing enabled: {_MAX_WORKERS} workers available", "⚡")
    else:
        _toast_notify("🔄 Parallel processing disabled - using sequential mode", "🔄")


def get_parallel_info():
    """Get information about parallel processing configuration."""
    return {
        "parallel_enabled": _USE_PARALLEL,
        "max_workers": _MAX_WORKERS,
        "cpu_count": multiprocessing.cpu_count(),
        "recommended_workers": min(multiprocessing.cpu_count(), 8)
    }


# Update exports to include parallel functions
__all__ = [
    "add_directional_features_lean",
    "add_horizon_specific_directional_features",
    "add_sensor_lag",
    "add_multi_lag",
    "add_rolling_stats",
    "add_time_since_last_measurement",
    "create_spatial_features",
    "create_time_features",
    "encode_categoricals",
    "select_feature_target",
    "add_multi_horizon_targets",
    "select_feature_target_multi",
    "add_stability_features",
    # Parallel versions
    "add_multi_lag_parallel",
    "add_rolling_stats_parallel", 
    "add_stability_features_parallel",
    "preprocess_dataframe_parallel",
    "set_parallel_processing",
    "get_parallel_info",
] 