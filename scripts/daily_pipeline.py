import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests

from granarypredict import ingestion, cleaning, features, model as model_utils
from granarypredict.config import DATA_DIR, MODELS_DIR
from granarypredict.data_utils import comprehensive_sort, assign_group_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("daily_pipeline")

# -----------------------------------------------------------------------------
# Constants & helper paths
# -----------------------------------------------------------------------------
RAW_BY_GRANARY = DATA_DIR / "raw" / "by_granary"
RAW_BY_GRANARY.mkdir(parents=True, exist_ok=True)

FORECAST_DIR = DATA_DIR / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 2, 3)  # days ahead to predict

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

# -----------------------------------------------------------------------------
# 1. API data fetch (placeholder – company should implement auth/endpoint here)
# -----------------------------------------------------------------------------

def fetch_new_data(api_url: str, api_key: str, date_str: str) -> pd.DataFrame:
    """Fetch raw sensor CSV/JSON for *date_str* (YYYY-MM-DD) via company API.

    The function expects the API to return either CSV text or JSON rows.
    """
    logger.info("Fetching %s from %s", date_str, api_url)

    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(api_url, params={"date": date_str}, headers=headers, timeout=60)
    resp.raise_for_status()

    # Heuristic: try CSV then JSON
    try:
        df = pd.read_csv(pd.compat.StringIO(resp.text), encoding="utf-8")
    except Exception:
        payload = resp.json()
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError("Unsupported API response format")

    # Map to canonical schema
    df = ingestion.standardize_granary_csv(df)
    return df

# -----------------------------------------------------------------------------
# 2. Persist incrementally to per-granary CSVs
# -----------------------------------------------------------------------------

def append_to_granary_csv(df: pd.DataFrame) -> List[Path]:
    """Append *df* rows to <granary>.csv files. Returns list of updated CSV paths."""
    if {"granary_id"}.issubset(df.columns):
        granaries = df["granary_id"].unique()
    else:
        raise ValueError("Column 'granary_id' missing after standardisation")

    written: List[Path] = []
    for g in granaries:
        g_df = df[df["granary_id"] == g].copy()
        out_path = RAW_BY_GRANARY / f"{g}.csv"
        if out_path.exists():
            base = pd.read_csv(out_path, encoding="utf-8")
            combo = pd.concat([base, g_df], ignore_index=True, sort=False)
            # Drop duplicate sensor-time records
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
            combo = g_df
        combo.to_csv(out_path, index=False, encoding="utf-8")
        written.append(out_path)
        logger.info("Written %s (rows=%d)", out_path.name, len(combo))
    return written

# -----------------------------------------------------------------------------
# 3. Model training & forecasting per granary
# -----------------------------------------------------------------------------

def preprocess_full(df: pd.DataFrame, *, future_safe: bool = True) -> pd.DataFrame:
    """Run cleaning + feature engineering pipeline used during Streamlit app."""
    df = cleaning.basic_clean(df)
    df = cleaning.fill_missing(df)
    df = features.create_time_features(df)
    df = features.create_spatial_features(df)
    df = features.add_sensor_lag(df)
    df = comprehensive_sort(df)
    df = assign_group_id(df)

    if future_safe:
        df = df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

    # Add multi-horizon training targets
    df = features.add_multi_horizon_targets(df, horizons=HORIZONS)
    return df


def make_future(df: pd.DataFrame, *, horizon_days: int = 3) -> pd.DataFrame:
    """Create future template rows for *horizon_days* based on last date in *df*."""
    if df.empty:
        return pd.DataFrame()
    last_ts = pd.to_datetime(df["detection_time"]).max()
    last_date = last_ts.normalize()

    frames = []
    max_cycle = df.get("detection_cycle").max() if "detection_cycle" in df.columns else None
    for d in range(1, horizon_days + 1):
        tmp = df[df["detection_time"].dt.normalize() == last_date].copy()  # last day snapshot
        tmp["detection_time"] = tmp["detection_time"] + pd.Timedelta(days=d)
        tmp["forecast_day"] = d
        if max_cycle is not None:
            tmp["detection_cycle"] = max_cycle + d
        frames.append(tmp)

    future_df = pd.concat(frames, ignore_index=True)
    future_df = features.create_time_features(future_df)
    future_df = features.create_spatial_features(future_df)
    future_df = features.add_sensor_lag(future_df)
    return future_df


def train_and_forecast(csv_path: Path):
    granary = csv_path.stem
    logger.info("=== Processing granary: %s ===", granary)

    full_df = pd.read_csv(csv_path, encoding="utf-8", parse_dates=["detection_time"])
    full_df = preprocess_full(full_df, future_safe=True)

    # Train on 100% of data
    X_full, y_full = features.select_feature_target_multi(
        full_df, target_col="temperature_grain", horizons=HORIZONS
    )
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    base = LGBMRegressor(n_estimators=1200, learning_rate=0.033, max_depth=7, num_leaves=24, n_jobs=-1)
    mdl = MultiOutputRegressor(base)
    mdl.fit(X_full, y_full)

    model_name = f"{granary}_fs_lgbm.joblib"
    model_utils.save_model(mdl, name=model_name)
    logger.info("Saved model → %s", model_name)

    # ---------------- Forecast ----------------
    future_df = make_future(full_df, horizon_days=3)
    if future_df.empty:
        logger.warning("No future template generated for %s (empty df)", granary)
        return

    X_future, _ = features.select_feature_target_multi(
        future_df, target_col="temperature_grain", horizons=HORIZONS, allow_na=True
    )
    preds = mdl.predict(X_future)
    # preds shape: (n_samples, len(HORIZONS))
    for idx, h in enumerate(HORIZONS):
        future_df[f"pred_h{h}d"] = preds[:, idx]

    out_forecast = FORECAST_DIR / f"{granary}_{dt.date.today().isoformat()}_fc.csv"
    future_df.to_csv(out_forecast, index=False, encoding="utf-8")
    logger.info("Forecast saved → %s (rows=%d)", out_forecast.name, len(future_df))

# -----------------------------------------------------------------------------
# Entry-point CLI
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Daily pipeline: fetch new data, update CSVs, retrain models, forecast 3-day horizon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", required=True, help="Base URL of the sensor data API endpoint")
    parser.add_argument("--api-key", required=True, help="Bearer token or API key for auth")
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Date to fetch (YYYY-MM-DD). Defaults to today.",
    )

    args = parser.parse_args(argv)

    new_df = fetch_new_data(args.api_url, args.api_key, args.date)
    updated_csvs = append_to_granary_csv(new_df)

    for csv_path in updated_csvs:
        train_and_forecast(csv_path)

    logger.info("Pipeline completed ✅")


if __name__ == "__main__":
    main() 