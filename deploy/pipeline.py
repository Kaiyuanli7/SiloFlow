from __future__ import annotations

import argparse
import pathlib
from datetime import datetime

import pandas as pd
import numpy as np

from . import preprocessing, features, model_utils, forecasting

HORIZON_DAYS = 7


def run_pipeline(csv_path: str | pathlib.Path, eval_path: str | pathlib.Path | None = None) -> None:
    csv_path = pathlib.Path(csv_path)
    print(f"[INFO] Reading training CSV {csv_path}…")
    df_raw = pd.read_csv(csv_path, encoding="utf-8")

    # 1. Preprocess
    df_proc = preprocessing.preprocess(df_raw)
    print(f"[INFO] Preprocessed shape: {df_proc.shape}")

    # 2. Add multi-horizon targets (1–7 days) ensured by preprocess
    horizons = tuple(range(1, HORIZON_DAYS + 1))
    X, y = features.select_feature_target_multi(df_proc, horizons=horizons)

    # 3. Train multi-output LightGBM
    mdl, metrics = model_utils.train_model(X, y)
    print("[INFO] Training metrics (train data):", metrics)

    feature_cols = list(X.columns)

    # 5. Optional evaluation on a separate CSV ---------------------
    if eval_path is not None:
        eval_path = pathlib.Path(eval_path)
        print(f"[INFO] Reading evaluation CSV {eval_path}…")
        df_eval_raw = pd.read_csv(eval_path, encoding="utf-8")
        df_eval_proc = preprocessing.preprocess(df_eval_raw)
        X_eval, y_eval = features.select_feature_target_multi(
            df_eval_proc, horizons=horizons, allow_na=True
        )
        # Align columns to training model
        X_eval = X_eval.reindex(columns=feature_cols, fill_value=0)
        preds_eval = model_utils.predict(mdl, X_eval)

        # Compute MAE/RMSE per horizon where ground-truth exists
        def _metric(col_idx: int):
            tgt_col = f"temperature_grain_h{col_idx+1}d"
            if tgt_col not in y_eval.columns:
                return None, None
            mask = y_eval[tgt_col].notna()
            if not mask.any():
                return None, None
            mae = np.abs(y_eval.loc[mask, tgt_col] - preds_eval[mask.index][mask, col_idx]).mean()  # type: ignore
            rmse = np.sqrt(((y_eval.loc[mask, tgt_col] - preds_eval[mask.index][mask, col_idx]) ** 2).mean())  # type: ignore
            return mae, rmse

        for h_idx, h in enumerate(horizons):
            mae_h, rmse_h = _metric(h_idx)
            if mae_h is not None:
                print(f"[INFO] Eval horizon h+{h}: MAE={mae_h:.3f} RMSE={rmse_h:.3f}")

    # 6. Generate forecast rows ------------------------------
    future_df = forecasting.make_future(df_proc, horizon_days=HORIZON_DAYS)
    X_future, _ = features.select_feature_target_multi(future_df, horizons=horizons, allow_na=True)
    preds = model_utils.predict(mdl, X_future)

    # Assign horizon-specific prediction cols
    if preds.ndim == 2:
        for idx, h in enumerate(horizons):
            if idx < preds.shape[1]:
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
        ]
        if c in future_df.columns
    ]
    out_df = future_df[out_cols]

    out_dir = pathlib.Path("deploy_outputs")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"predictions_{csv_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[INFO] Forecast CSV written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM multi-output model and forecast 7 days")
    parser.add_argument("csv", help="Path to training CSV")
    parser.add_argument("--eval", help="Path to evaluation CSV (optional)")
    args = parser.parse_args()
    run_pipeline(args.csv, eval_path=args.eval) 