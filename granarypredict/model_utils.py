"""
Unified model training utility for SiloFlow.
This module provides a single function to train a model on a processed DataFrame,
using the exact same logic as Dashboard.py, for use in both dashboard and batch processing.
"""
import numpy as np
import pandas as pd
from granarypredict.features import select_feature_target_multi
from granarypredict.multi_lgbm import MultiLGBMRegressor

def train_model_on_processed_df(
    df,
    base_params,
    target_col="temperature_grain",
    horizons=(1,2,3,4,5,6,7),
    train_pct=0.8,
    anchor_early_stop=True,
    balance_horizons=True,
    horizon_strategy="increasing",
    use_gpu=True,
    conservative_mode=True,
    stability_feature_boost=3.0,
    directional_feature_boost=2.0,
    uncertainty_estimation=True,
    n_bootstrap_samples=25,
    verbose=False,
):
    """
    Unified training function for processed DataFrames.
    Returns trained model and train/val metrics.
    """
    # Feature selection
    X_all, y_all = select_feature_target_multi(
        df, target_col=target_col, horizons=horizons
    )
    # Chronological train/val split
    split_idx = int(len(X_all) * train_pct)
    X_tr, y_tr = X_all.iloc[:split_idx], y_all.iloc[:split_idx]
    X_val, y_val = X_all.iloc[split_idx:], y_all.iloc[split_idx:]
    # Model setup
    mdl = MultiLGBMRegressor(
        base_params=base_params,
        uncertainty_estimation=uncertainty_estimation,
        n_bootstrap_samples=n_bootstrap_samples,
        directional_feature_boost=directional_feature_boost,
        conservative_mode=conservative_mode,
        stability_feature_boost=stability_feature_boost,
        use_gpu=use_gpu,
        gpu_optimization=True,
    )
    # Fit model
    mdl.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        verbose=verbose,
        anchor_df=df.iloc[split_idx:],
        horizon_tuple=horizons,
        use_anchor_early_stopping=anchor_early_stop,
        balance_horizons=balance_horizons,
        horizon_strategy=horizon_strategy,
    )
    # Metrics
    preds = mdl.predict(X_val)
    mae = np.mean(np.abs(y_val.values - preds))
    rmse = np.sqrt(np.mean((y_val.values - preds) ** 2))
    return mdl, {"mae": mae, "rmse": rmse}
