from __future__ import annotations

"""Append incoming rows to per-granary training CSVs and per-heap forecast CSVs."""

import pathlib
from pathlib import Path
from typing import Tuple

import pandas as pd

TRAIN_DIR = Path("data/training")
FORECAST_DIR = Path("data/forecasting")

# Ensure dirs exist
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["append_rows"]


def _write_append(df: pd.DataFrame, path: Path) -> int:
    """Append *df* to *path* (CSV). Returns number of rows written."""
    if df.empty:
        return 0
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")
    return len(df)


def append_rows(df: pd.DataFrame) -> Tuple[int, int]:
    """Append rows in *df* to the correct training / forecasting CSVs.

    Returns (n_training_rows_appended, n_forecast_rows_appended).
    """
    if df.empty:
        return 0, 0

    required_cols = {"granary_id", "heap_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in uploaded CSV: {missing}")

    n_train = n_forec = 0

    # Per-heap forecasting CSVs -----------------------------------------
    for (g_id, h_id), sub in df.groupby(["granary_id", "heap_id"], dropna=False):
        f_path = FORECAST_DIR / f"g{g_id}_h{h_id}.csv"
        n_forec += _write_append(sub, f_path)

    # Per-granary training CSVs -----------------------------------------
    for g_id, g_sub in df.groupby("granary_id"):
        t_path = TRAIN_DIR / f"g{g_id}.csv"
        n_train += _write_append(g_sub, t_path)

    return n_train, n_forec 