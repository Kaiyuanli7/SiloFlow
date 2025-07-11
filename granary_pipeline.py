#!/usr/bin/env python3
"""
Granary Data Pipeline - Production Script
==========================================

This script handles:
1. Ingesting new unprocessed CSV data from sensors
2. Organizing data by granary (all heaps from same granary in same file)
3. Preprocessing newly ingested data with all feature engineering
4. Running forecasting with existing trained models
5. Optional model training when needed

Usage:
    python granary_pipeline.py --input new_data.csv --mode forecast
    python granary_pipeline.py --input new_data.csv --mode train --granary "蚬冈库"
"""

import argparse
import pathlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging

# Core imports from granarypredict package
from granarypredict import cleaning, features, model as model_utils, ingestion
from granarypredict.config import ALERT_TEMP_THRESHOLD, MODELS_DIR
from granarypredict.data_utils import comprehensive_sort, assign_group_id
from granarypredict.multi_lgbm import MultiLGBMRegressor
from granarypredict.optuna_cache import load_optimal_params, save_optimal_params

# Constants from Dashboard.py
HORIZON_DAYS = 7
HORIZON_TUPLE = tuple(range(1, HORIZON_DAYS + 1))
TARGET_TEMP_COL = "temperature_grain"

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GranaryDataPipeline:
    """Production pipeline for granary temperature data processing and forecasting."""
    
    def __init__(self, base_data_dir: str = "data/granaries"):
        """Initialize pipeline with base directory for granary-specific data files.
        
        Args:
            base_data_dir: Directory where granary-specific CSV files are stored
        """
        self.base_data_dir = pathlib.Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = pathlib.Path("models")
        self.forecasts_dir = pathlib.Path("data/forecasts")
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized - Data dir: {self.base_data_dir}")
        
    def identify_granary(self, df: pd.DataFrame) -> str:
        """Identify which granary this data belongs to.
        
        Args:
            df: Raw sensor data
            
        Returns:
            granary_name: Name of the granary
        """
        # Standardize the dataframe first
        df_std = ingestion.standardize_granary_csv(df)
        
        # Look for granary identifier columns
        granary_cols = [c for c in ["granary_id", "storepointName", "warehouse_name"] if c in df_std.columns]
        
        if granary_cols:
            granary_col = granary_cols[0]
            granary_values = df_std[granary_col].dropna().unique()
            if len(granary_values) > 0:
                granary_name = str(granary_values[0])
                logger.info(f"Identified granary: {granary_name}")
                return granary_name
        
        # Fallback: try to infer from filename or use default
        logger.warning("Could not identify granary from data, using 'unknown_granary'")
        return "unknown_granary"
    
    def get_granary_file_path(self, granary_name: str) -> pathlib.Path:
        """Get the file path for a specific granary's data.
        
        Args:
            granary_name: Name of the granary
            
        Returns:
            Path to the granary's CSV file
        """
        safe_name = granary_name.replace(" ", "_").replace("/", "_")
        return self.base_data_dir / f"{safe_name}.csv"
    
    def ingest_new_data(self, input_csv: str) -> List[str]:
        """Ingest new CSV data and organize by granary.
        
        Args:
            input_csv: Path to new unprocessed CSV file
            
        Returns:
            List of granary names that were updated
        """
        logger.info(f"Ingesting new data from: {input_csv}")
        
        # Load and standardize new data
        df_new = pd.read_csv(input_csv, encoding="utf-8")
        df_new = ingestion.standardize_granary_csv(df_new)
        
        updated_granaries = []
        
        # Group new data by granary
        granary_col = None
        for col in ["granary_id", "storepointName", "warehouse_name"]:
            if col in df_new.columns:
                granary_col = col
                break
        
        if granary_col:
            # Group by granary and process each group
            for granary_name, group_df in df_new.groupby(granary_col):
                granary_name = str(granary_name)
                self._update_granary_file(granary_name, group_df)
                updated_granaries.append(granary_name)
        else:
            # Single granary file
            granary_name = self.identify_granary(df_new)
            self._update_granary_file(granary_name, df_new)
            updated_granaries.append(granary_name)
        
        logger.info(f"Updated granaries: {updated_granaries}")
        return updated_granaries
    
    def _update_granary_file(self, granary_name: str, new_data: pd.DataFrame):
        """Update a specific granary's data file with new data.
        
        Args:
            granary_name: Name of the granary
            new_data: New sensor data for this granary
        """
        granary_file = self.get_granary_file_path(granary_name)
        
        if granary_file.exists():
            # Load existing data and append new data
            df_existing = pd.read_csv(granary_file, encoding="utf-8")
            logger.info(f"Loaded existing data for {granary_name}: {len(df_existing)} rows")
            
            # Combine and remove duplicates based on timestamp and sensor location
            df_combined = pd.concat([df_existing, new_data], ignore_index=True)
            
            # Remove duplicates based on key columns
            key_cols = [c for c in ["detection_time", "granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] 
                       if c in df_combined.columns]
            if key_cols:
                df_combined = df_combined.drop_duplicates(subset=key_cols, keep='last')
            
            logger.info(f"Combined data for {granary_name}: {len(df_combined)} rows")
        else:
            # New granary file
            df_combined = new_data.copy()
            logger.info(f"Creating new granary file for {granary_name}: {len(df_combined)} rows")
        
        # Sort by timestamp
        if "detection_time" in df_combined.columns:
            df_combined["detection_time"] = pd.to_datetime(df_combined["detection_time"], errors="coerce")
            df_combined = df_combined.sort_values("detection_time")
        
        # Save updated file
        df_combined.to_csv(granary_file, index=False, encoding="utf-8")
        logger.info(f"Saved updated data to: {granary_file}")
    
    def preprocess_granary_data(self, granary_name: str, process_all: bool = False) -> pd.DataFrame:
        """Preprocess data for a specific granary.
        
        Args:
            granary_name: Name of the granary
            process_all: If True, process all data. If False, only process recent data.
            
        Returns:
            Preprocessed dataframe
        """
        granary_file = self.get_granary_file_path(granary_name)
        
        if not granary_file.exists():
            raise FileNotFoundError(f"No data file found for granary: {granary_name}")
        
        logger.info(f"Preprocessing data for granary: {granary_name}")
        df = pd.read_csv(granary_file, encoding="utf-8")
        
        if process_all:
            # Full preprocessing pipeline
            df_processed = self._preprocess_df(df)
        else:
            # Quick preprocessing for latest data only
            df_processed = self._preprocess_latest_data(df)
        
        # Save processed version
        processed_file = granary_file.parent / f"{granary_file.stem}_processed.csv"
        df_processed.to_csv(processed_file, index=False, encoding="utf-8")
        logger.info(f"Saved processed data to: {processed_file}")
        
        return df_processed
    
    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing pipeline (copied from Dashboard.py)."""
        if df.empty:
            return df
            
        logger.info("Starting full preprocessing pipeline...")
        
        # Basic cleaning
        df = cleaning.basic_clean(df)
        
        # Insert calendar gaps and interpolate
        df = self._insert_calendar_gaps(df)
        df = self._interpolate_sensor_numeric(df)
        
        # Subsample to reduce density
        df = self._subsample_per_sensor(df, hours=4)
        
        # Final cleaning
        df = cleaning.fill_missing(df)
        
        # Feature engineering
        df = features.create_time_features(df)
        df = features.create_spatial_features(df)
        df = features.add_time_since_last_measurement(df)
        
        # Parallel feature engineering
        df = features.add_multi_lag_parallel(df, lags=(1,2,3,4,5,6,7,14,30))
        df = features.add_rolling_stats_parallel(df, window_days=7)
        df = features.add_directional_features_lean(df)
        df = features.add_stability_features_parallel(df)
        df = features.add_horizon_specific_directional_features(df, max_horizon=HORIZON_DAYS)
        
        # Multi-horizon targets
        df = features.add_multi_horizon_targets(df, horizons=HORIZON_TUPLE)
        
        # Final sorting and grouping
        df = assign_group_id(df)
        df = comprehensive_sort(df)
        
        logger.info("Full preprocessing complete")
        return df
    
    def _preprocess_latest_data(self, df: pd.DataFrame, days_back: int = 30) -> pd.DataFrame:
        """Quick preprocessing for only the latest data."""
        if df.empty:
            return df
            
        logger.info(f"Preprocessing latest {days_back} days of data...")
        
        # Get latest data only
        df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
        latest_date = df["detection_time"].max()
        cutoff_date = latest_date - timedelta(days=days_back)
        df_recent = df[df["detection_time"] >= cutoff_date].copy()
        
        # Run full preprocessing on recent data
        return self._preprocess_df(df_recent)
    
    def get_latest_model(self, granary_name: str) -> Optional[pathlib.Path]:
        """Find the most recent trained model for a granary.
        
        Args:
            granary_name: Name of the granary
            
        Returns:
            Path to the latest model file, or None if not found
        """
        safe_name = granary_name.replace(" ", "_").replace("/", "_")
        pattern = f"{safe_name}*.joblib"
        
        model_files = list(self.models_dir.glob(pattern))
        if not model_files:
            # Try more general patterns
            model_files = list(self.models_dir.glob("*.joblib"))
        
        if model_files:
            # Sort by modification time and return newest
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found latest model for {granary_name}: {latest_model}")
            return latest_model
        
        logger.warning(f"No model found for granary: {granary_name}")
        return None
    
    def forecast_granary(self, granary_name: str, horizon_days: int = 7) -> Optional[pd.DataFrame]:
        """Generate forecast for a specific granary using its latest model.
        
        Args:
            granary_name: Name of the granary
            horizon_days: Number of days to forecast ahead
            
        Returns:
            Forecast dataframe or None if failed
        """
        logger.info(f"Generating {horizon_days}-day forecast for granary: {granary_name}")
        
        # Load latest model
        model_path = self.get_latest_model(granary_name)
        if not model_path:
            logger.error(f"No model available for forecasting granary: {granary_name}")
            return None
        
        try:
            model = model_utils.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
        
        # Get preprocessed data
        try:
            df_processed = self.preprocess_granary_data(granary_name, process_all=False)
        except Exception as e:
            logger.error(f"Failed to preprocess data for {granary_name}: {e}")
            return None
        
        # Generate forecast
        try:
            forecast_df = self._generate_forecast(model, df_processed, horizon_days)
            
            # Save forecast
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            forecast_file = self.forecasts_dir / f"{granary_name}_forecast_{timestamp}.csv"
            forecast_df.to_csv(forecast_file, index=False, encoding="utf-8")
            logger.info(f"Forecast saved to: {forecast_file}")
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Failed to generate forecast for {granary_name}: {e}")
            return None
    
    def _generate_forecast(self, model, df_processed: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        """Generate forecast using the model (adapted from Dashboard.py)."""
        
        # Get last known data for each sensor
        sensors_key = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] 
                      if c in df_processed.columns]
        
        last_rows = (
            df_processed.sort_values("detection_time")
            .groupby(sensors_key, dropna=False)
            .tail(1)
            .copy()
        )
        
        # Prepare feature matrix
        X_snap, _ = features.select_feature_target_multi(
            last_rows, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE, allow_na=True
        )
        
        # Get model features and align
        model_feats = self._get_feature_cols(model, X_snap)
        X_snap_aligned = X_snap.reindex(columns=model_feats, fill_value=0)
        
        # Generate predictions
        preds_mat = model_utils.predict(model, X_snap_aligned)
        
        # Build future dataframes
        all_future_frames = []
        last_dt = pd.to_datetime(last_rows["detection_time"]).max()
        
        for h in range(1, horizon_days + 1):
            day_frame = last_rows.copy()
            day_frame["detection_time"] = last_dt + timedelta(days=h)
            day_frame["forecast_day"] = h
            
            # Extract prediction for this horizon
            if preds_mat.ndim == 2:
                idx = min(h - 1, preds_mat.shape[1] - 1)
                pred_val = preds_mat[:, idx]
            else:
                pred_val = preds_mat
            
            day_frame["predicted_temp"] = pred_val
            day_frame["is_forecast"] = True
            all_future_frames.append(day_frame)
        
        forecast_df = pd.concat(all_future_frames, ignore_index=True)
        
        # Keep only essential columns
        core_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z", 
                                "detection_time", "forecast_day", "predicted_temp"] 
                    if c in forecast_df.columns]
        
        return forecast_df[core_cols]
    
    def train_granary_model(self, granary_name: str) -> Optional[pathlib.Path]:
        """Train a new model for a specific granary with optimized configuration.
        
        Configuration:
        - 100% training data (no validation split)
        - No future-safe mode (all variables included)
        - Quantile regression enabled
        - Increasing horizon balance strategy
        - Anchor-day early stopping
        - Auto Optuna caching (20 trials if no cache exists)
        
        Args:
            granary_name: Name of the granary
            
        Returns:
            Path to saved model or None if failed
        """
        logger.info(f"Training new model for granary: {granary_name}")
        
        # Get preprocessed data
        try:
            df_processed = self.preprocess_granary_data(granary_name, process_all=True)
        except Exception as e:
            logger.error(f"Failed to preprocess data for training: {e}")
            return None
        
        # Use ALL data for training (no future-safe mode)
        X_all, y_all = features.select_feature_target_multi(
            df_processed, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
        )
        
        # Initialize base parameters with quantile regression
        base_params = {
            "learning_rate": 0.033,
            "max_depth": 7,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "min_child_samples": 20,
            "objective": "quantile",
            "alpha": 0.5,
        }
        
        # Auto Optuna optimization with caching
        base_params = self._get_optuna_params(granary_name, df_processed, base_params)
        
        # Create internal 95/5 split for early stopping estimation
        df_train_tmp, df_eval_tmp = self._split_train_eval_frac(df_processed, test_frac=0.05)
        X_train_tmp, y_train_tmp = features.select_feature_target_multi(
            df_train_tmp, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
        )
        X_eval_tmp, y_eval_tmp = features.select_feature_target_multi(
            df_eval_tmp, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
        )
        
        try:
            # Create finder model to determine optimal iterations
            finder = MultiLGBMRegressor(
                base_params=base_params,
                upper_bound_estimators=2000,
                early_stopping_rounds=50,
                uncertainty_estimation=True,
                n_bootstrap_samples=50,
                conservative_mode=True,
                directional_feature_boost=2.0,
                stability_feature_boost=3.0,
            )
            
            logger.info("Finding optimal iterations with internal 95/5 split...")
            finder.fit(
                X_train_tmp, y_train_tmp,
                eval_set=(X_eval_tmp, y_eval_tmp),
                verbose=False,
                anchor_df=df_eval_tmp,
                horizon_tuple=HORIZON_TUPLE,
                use_anchor_early_stopping=True,
                balance_horizons=True,
                horizon_strategy="increasing"  # Increasing balance configuration
            )
            
            best_iterations = finder.best_iteration_
            logger.info(f"Optimal iterations found: {best_iterations}")
            
            # Train final model on 100% of data with fixed iterations
            final_params = base_params.copy()
            final_params["n_estimators"] = best_iterations
            
            final_model = MultiLGBMRegressor(
                base_params=final_params,
                upper_bound_estimators=best_iterations,
                early_stopping_rounds=0,  # No early stopping for final model
                uncertainty_estimation=True,
                n_bootstrap_samples=50,
                conservative_mode=True,
                directional_feature_boost=2.0,
                stability_feature_boost=3.0,
            )
            
            logger.info("Training final model on 100% of data...")
            final_model.fit(
                X_all, y_all,
                balance_horizons=True,
                horizon_strategy="increasing"  # Increasing balance configuration
            )
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = granary_name.replace(" ", "_").replace("/", "_")
            model_name = f"{safe_name}_lgbm_{timestamp}.joblib"
            model_path = self.models_dir / model_name
            
            model_utils.save_model(final_model, str(model_path))
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Training configuration: 100% data, quantile regression, increasing balance, {best_iterations} iterations")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to train model for {granary_name}: {e}")
            return None
    
    def _get_optuna_params(self, granary_name: str, df_processed: pd.DataFrame, base_params: dict) -> dict:
        """Get Optuna-optimized parameters with auto caching.
        
        If no cached parameters exist for this granary, runs a quick 20-trial Optuna
        optimization and caches the results for future use.
        
        Args:
            granary_name: Name of the granary
            df_processed: Preprocessed dataframe
            base_params: Base parameter dictionary
            
        Returns:
            Optimized parameter dictionary
        """
        try:
            from granarypredict.optuna_cache import load_optimal_params, save_optimal_params
            import optuna
        except ImportError:
            logger.warning("Optuna or optuna_cache not available, using default parameters")
            return base_params
        
        # Create model configuration for cache key
        model_config = {
            "model_type": "LightGBM",
            "future_safe": False,
            "use_quantile": True,
            "balance_horizons": True,
            "horizon_strategy": "increasing",
            "anchor_early_stop": True,
            "train_split": "100pct"
        }
        
        # Try to load cached parameters
        cached_result = load_optimal_params(f"{granary_name}.csv", df_processed, model_config)
        
        if cached_result:
            best_params, best_value, timestamp = cached_result
            logger.info(f"Using cached Optuna parameters for {granary_name} (MAE: {best_value:.4f}, cached: {timestamp})")
            
            # Update base params with cached optimal params
            optimized_params = base_params.copy()
            optimized_params.update(best_params)
            
            # Ensure quantile objective is preserved
            optimized_params.update({
                "objective": "quantile",
                "alpha": 0.5,
            })
            
            return optimized_params
        
        # No cached parameters - run quick Optuna optimization
        logger.info(f"No cached parameters found for {granary_name}, running 20-trial Optuna optimization...")
        
        # Create 95/5 split for optimization
        df_opt_train, df_opt_val = self._split_train_eval_frac(df_processed, test_frac=0.05)
        X_opt_train, y_opt_train = features.select_feature_target_multi(
            df_opt_train, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
        )
        X_opt_val, y_opt_val = features.select_feature_target_multi(
            df_opt_val, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
        )
        
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
                "objective": "quantile",
                "alpha": 0.5,
            }
            
            try:
                # Quick model with reduced trees for optimization speed
                model_tmp = MultiLGBMRegressor(
                    base_params=params,
                    upper_bound_estimators=500,  # Reduced for speed
                    early_stopping_rounds=30,   # Aggressive early stopping
                    uncertainty_estimation=False,  # Disabled for speed
                    conservative_mode=False,    # Simplified for speed
                )
                
                model_tmp.fit(
                    X_opt_train, y_opt_train,
                    eval_set=(X_opt_val, y_opt_val),
                    verbose=False,
                    anchor_df=df_opt_val,
                    horizon_tuple=HORIZON_TUPLE,
                    use_anchor_early_stopping=True,
                    balance_horizons=True,
                    horizon_strategy="increasing"
                )
                
                # Compute anchor-day MAE for optimization target
                preds = model_tmp.predict(X_opt_val)
                if preds.ndim == 2:
                    # Use average MAE across all horizons
                    maes = []
                    for h_idx in range(min(preds.shape[1], len(HORIZON_TUPLE))):
                        target_col = f"temperature_grain_h{HORIZON_TUPLE[h_idx]}d"
                        if target_col in df_opt_val.columns:
                            mask = ~np.isnan(preds[:, h_idx]) & df_opt_val[target_col].notna()
                            if mask.sum() > 0:
                                mae_h = np.abs(preds[mask, h_idx] - df_opt_val.loc[mask, target_col]).mean()
                                maes.append(mae_h)
                    
                    return np.mean(maes) if maes else 999.0
                else:
                    # Single output - use primary target
                    mask = ~np.isnan(preds) & df_opt_val[TARGET_TEMP_COL].notna()
                    if mask.sum() > 0:
                        return np.abs(preds[mask] - df_opt_val.loc[mask, TARGET_TEMP_COL]).mean()
                    else:
                        return 999.0
                        
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 999.0
        
        try:
            # Run Optuna optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20, n_jobs=1)  # Quick 20-trial optimization
            
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Optuna optimization completed for {granary_name}")
            logger.info(f"Best MAE: {best_value:.4f}")
            logger.info(f"Best params: {best_params}")
            
            # Cache the results
            try:
                cache_key = save_optimal_params(
                    csv_filename=f"{granary_name}.csv",
                    df=df_processed,
                    model_config=model_config,
                    optimal_params=best_params,
                    best_value=best_value,
                    n_trials=20
                )
                logger.info(f"Cached optimal parameters for future use: {cache_key}")
            except Exception as cache_exc:
                logger.warning(f"Failed to cache parameters: {cache_exc}")
            
            # Update base params with optimized params
            optimized_params = base_params.copy()
            optimized_params.update(best_params)
            
            # Ensure quantile objective is preserved
            optimized_params.update({
                "objective": "quantile",
                "alpha": 0.5,
            })
            
            return optimized_params
            
        except Exception as e:
            logger.warning(f"Optuna optimization failed for {granary_name}: {e}")
            logger.info("Using default parameters")
            return base_params
    
    # Helper functions (copied from Dashboard.py)
    def _get_feature_cols(self, model, X_fallback: pd.DataFrame) -> List[str]:
        """Get feature columns expected by the model."""
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "feature_name_"):
            return list(model.feature_name_)
        return list(X_fallback.columns)
    
    def _split_train_eval_frac(self, df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe chronologically."""
        df = df.copy()
        df["_date"] = pd.to_datetime(df["detection_time"], errors="coerce").dt.date
        unique_dates = sorted(df["_date"].dropna().unique())
        
        if not unique_dates:
            return df, pd.DataFrame()
        
        n_test_days = max(1, int(len(unique_dates) * test_frac))
        cutoff_dates = unique_dates[-n_test_days:]
        
        df_eval = df[df["_date"].isin(cutoff_dates)].copy()
        df_train = df[~df["_date"].isin(cutoff_dates)].copy()
        
        df_train.drop(columns=["_date"], inplace=True)
        df_eval.drop(columns=["_date"], inplace=True)
        
        return df_train, df_eval
    
    def _subsample_per_sensor(self, df: pd.DataFrame, hours: int = 4) -> pd.DataFrame:
        """Subsample data to one record every N hours per sensor."""
        if df.empty or "detection_time" not in df.columns:
            return df
            
        df = df.copy()
        df["_dt_floor"] = pd.to_datetime(df["detection_time"], errors="coerce").dt.floor(f"{hours}H")
        key_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z", "_dt_floor"] 
                   if c in df.columns]
        
        return (
            df.sort_values("detection_time")
            .groupby(key_cols, dropna=False)
            .head(1)
            .drop(columns=["_dt_floor"])
            .reset_index(drop=True)
        )
    
    def _insert_calendar_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insert missing calendar days per sensor."""
        if "detection_time" not in df.columns:
            return df
        
        df = df.copy()
        df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
        
        if df["detection_time"].isna().all():
            return df
        
        df_valid = df[df["detection_time"].notna()].copy()
        group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] 
                     if c in df_valid.columns]
        
        frames = [df_valid]
        static_like = set(group_cols + ["granary_id", "heap_id", "grain_type", "warehouse_type"])
        
        for key, sub in (df_valid.groupby(group_cols) if group_cols else [(None, df_valid)]):
            sub = sub.sort_values("detection_time")
            date_series = sub["detection_time"].dropna().dt.floor("D")
            
            if date_series.empty:
                continue
                
            start_date, end_date = date_series.min(), date_series.max()
            if pd.isna(start_date) or pd.isna(end_date) or start_date == end_date:
                continue
            
            full_range = pd.date_range(start_date, end_date, freq="D")
            missing_dates = sorted(set(full_range.date) - set(date_series.dt.date.unique()))
            
            if missing_dates:
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
    
    def _interpolate_sensor_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate numeric columns per sensor."""
        if "detection_time" not in df.columns:
            return df
        
        df = df.copy()
        df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
        df.sort_values("detection_time", inplace=True)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return df
        
        group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] 
                     if c in df.columns]
        
        if group_cols:
            df[num_cols] = (
                df.groupby(group_cols)[num_cols]
                .apply(lambda g: g.interpolate(method="linear", limit_direction="forward").ffill())
                .reset_index(level=group_cols, drop=True)
            )
        else:
            df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="forward").ffill()
        
        return df


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Granary Data Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file with new sensor data")
    parser.add_argument("--mode", choices=["ingest", "forecast", "train"], default="forecast", 
                       help="Operation mode")
    parser.add_argument("--granary", help="Specific granary name (for train mode)")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days")
    parser.add_argument("--data-dir", default="data/granaries", help="Base directory for granary data")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GranaryDataPipeline(base_data_dir=args.data_dir)
    
    if args.mode == "ingest":
        # Just ingest and organize data
        updated_granaries = pipeline.ingest_new_data(args.input)
        print(f"Ingested data for granaries: {updated_granaries}")
        
    elif args.mode == "forecast":
        # Ingest data and generate forecasts
        updated_granaries = pipeline.ingest_new_data(args.input)
        
        for granary_name in updated_granaries:
            print(f"Generating forecast for {granary_name}...")
            forecast_df = pipeline.forecast_granary(granary_name, args.horizon)
            if forecast_df is not None:
                print(f"✓ Forecast generated for {granary_name}")
                print(f"  Predicted temperature range: {forecast_df['predicted_temp'].min():.1f} - {forecast_df['predicted_temp'].max():.1f}°C")
            else:
                print(f"✗ Failed to generate forecast for {granary_name}")
                
    elif args.mode == "train":
        # Train model for specific granary
        if not args.granary:
            # Ingest data first and train for all updated granaries
            updated_granaries = pipeline.ingest_new_data(args.input)
            granaries_to_train = updated_granaries
        else:
            granaries_to_train = [args.granary]
        
        for granary_name in granaries_to_train:
            print(f"Training model for {granary_name}...")
            model_path = pipeline.train_granary_model(granary_name)
            if model_path:
                print(f"✓ Model trained and saved: {model_path}")
            else:
                print(f"✗ Failed to train model for {granary_name}")


if __name__ == "__main__":
    main() 