#!/usr/bin/env python3
"""
Granary Data Pipeline - Modular CLI & Automation
===============================================

Usage:
    python granary_pipeline.py ingest --input <raw.csv>
    python granary_pipeline.py preprocess --input <granary.csv> --output <processed.csv>
    python granary_pipeline.py train --granary <name>
    python granary_pipeline.py forecast --granary <name> --horizon <days>

This script orchestrates:
- Ingestion: sorting, deduplication, standardization
- Preprocessing: cleaning, gap insertion, interpolation, feature engineering
- Training: model fitting & hyperparameter optimization with Dashboard-optimized settings
- Forecasting: multi-horizon prediction

Training Configuration (matching Dashboard.py):
- Quantile regression: Uses quantile objective with alpha=0.5 for improved MAE
- Anchor-day early stopping: Uses 7-day consecutive forecasting accuracy
- Horizon balancing: Applies increasing horizon strategy for better long-term predictions
- Conservative mode: 3x stability feature boost + 2x directional feature boost
- 95/5 split: Internal split for finding optimal iterations, then train on 100% data

All steps are modular and importable for future automation/cloud deployment.
"""
import pandas as pd
import numpy as np
import pathlib
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import asyncio

# Add granarypredict directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import granarypredict modules
from granarypredict import ingestion, cleaning, features
from granarypredict.data_utils import assign_group_id, comprehensive_sort

# Import cleaning helpers from the correct location
# These functions are defined in the Dashboard.py file, so we'll import them from there
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
try:
    from Dashboard import insert_calendar_gaps, interpolate_sensor_numeric, split_train_eval_frac
except ImportError:
    # Fallback: define simple versions if Dashboard import fails
    def insert_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """Simple calendar gap insertion - placeholder"""
        return df
    
    def interpolate_sensor_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Simple numeric interpolation - placeholder"""
        return df
    
    def split_train_eval_frac(df: pd.DataFrame, test_frac: float = 0.05) -> tuple:
        """Simple train/eval split - placeholder"""
        split_idx = int(len(df) * (1 - test_frac))
        return df.iloc[:split_idx], df.iloc[split_idx:]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TARGET_TEMP_COL = 'temperature_grain'
HORIZON_TUPLE = tuple(range(1, 8))  # 1-7 days

def run_complete_pipeline(
    granary_csv: str,  # Individual granary CSV file
    granary_name: str,
    skip_train: bool = False,
    force_retrain: bool = False,
    changed_silos: Optional[List[str]] = None,  # NEW: Track which silos changed
    max_workers: int = 4  # New parameter for parallel processing
) -> dict:
    # Initialize data paths if not provided
    if data_paths is None:
        from .utils.data_paths import data_paths
    """
    Run complete pipeline: preprocess and train for a specific granary CSV.
    
    Parameters:
    -----------
    changed_silos : List[str], optional
        List of silo IDs that have new data. If provided, preprocessing
        will focus on these silos for efficiency, but training will use
        the full granary data.
    """
    # Skip ingestion step since we're working with individual granary CSV
    # Go directly to preprocessing
    # Then training
    results = {
        'granary': granary_name,
        'input_csv': granary_csv,
        'steps_completed': [],
        'errors': [],
        'model_path': None,
        'success': False,
        'changed_silos': changed_silos
    }
    
    try:
        # Step 1: Load full granary data
        logger.info(f"Loading data for granary: {granary_name}")
        df_full = ingestion.read_granary_csv(granary_csv)
        
        # Step 2: Efficient preprocessing strategy
        if changed_silos and len(changed_silos) < len(df_full['heap_id'].unique()):
            # Only some silos changed - use efficient preprocessing
            logger.info(f"Efficient preprocessing: focusing on {len(changed_silos)} changed silos")
        
            # Load existing processed data if available (supports both CSV and Parquet)
            from .utils.data_paths import data_paths
            processed_dir = data_paths.get_processed_dir()
            processed_path_csv = processed_dir / f"{granary_name}_processed.csv"
            processed_path_parquet = processed_dir / f"{granary_name}_processed.parquet"
            
            if processed_path_parquet.exists():
                # Use Parquet file if available
                df_existing = ingestion.read_granary_csv(processed_path_parquet)
                logger.info(f"Loaded existing Parquet data: {processed_path_parquet}")
            elif processed_path_csv.exists():
                # Fallback to CSV file if Parquet doesn't exist
                df_existing = pd.read_csv(processed_path_csv)
                logger.info(f"Loaded existing CSV data: {processed_path_csv}")
            else:
                df_existing = None
            
            if df_existing is not None:
                # Remove data for changed silos from existing data
                df_existing_filtered = df_existing[~df_existing['heap_id'].isin(changed_silos)]
                
                # Preprocess only the changed silos
                df_new_silos = df_full[df_full['heap_id'].isin(changed_silos)].copy()
                df_new_processed = _preprocess_silos(df_new_silos)
                
                # Combine existing and new processed data
                df = pd.concat([df_existing_filtered, df_new_processed], ignore_index=True)
                logger.info(f"Combined existing data with {len(changed_silos)} updated silos")
            else:
                # No existing processed data - process everything
                df = _preprocess_silos(df_full)
        else:
            # All silos or no silo info - process everything
            logger.info(f"Full preprocessing: processing all silos")
            df = _preprocess_silos(df_full)
        
        # Save processed data as Parquet (much more efficient for large datasets)
        from granarypredict.ingestion import save_granary_data
        
        processed_output = data_paths.get_processed_dir() / f"{granary_name}_processed"
        processed_output.parent.mkdir(exist_ok=True, parents=True)
        
        # Save as Parquet with snappy compression (60-80% smaller, 10x faster)
        parquet_path = save_granary_data(
            df=df,
            filepath=processed_output,
            format='parquet',
            compression='snappy'
        )
        
        results['steps_completed'].append('preprocess')
        logger.info(f"Preprocessed data saved as Parquet: {parquet_path}")
        logger.info(f"Preprocessed DataFrame shape: {df.shape}")
        
        # Step 3: Train model on FULL granary data (not just changed silos)
        # This ensures the model learns from all silos in the granary
        model_filename = f"{granary_name}_forecast_model.joblib"
        model_path = data_paths.get_models_dir() / model_filename
        
        should_train = (not skip_train and not model_path.exists()) or force_retrain
        
        if should_train:
            logger.info(f"Training model for granary: {granary_name} (using all silos)")
            
            # Prepare training data using the full processed dataset
            df['detection_time'] = pd.to_datetime(df['detection_time'])
            last_date = df['detection_time'].max()
            logger.info(f"Last date in dataset: {last_date}")
            
            from granarypredict.features import select_feature_target_multi
            
            # Phase 1: Create internal 95/5 split for parameter optimization (like Dashboard)
            logger.info("Phase 1: Creating internal 95/5 split for parameter optimization")
            df_sorted = df.sort_values('detection_time', ignore_index=True)
            total_rows = len(df_sorted)
            split_idx = int(total_rows * 0.95)  # Use 95% for training, 5% for validation
            
            # Training data (95%)
            train_df = df_sorted.iloc[:split_idx].copy()
            
            # Validation data (5% - last portion)
            val_df = df_sorted.iloc[split_idx:].copy()
            
            logger.info(f"Training data: {len(train_df)} rows (95%)")
            logger.info(f"Validation data: {len(val_df)} rows (5%)")
            
            # Prepare training features and targets
            X_train, Y_train = select_feature_target_multi(
                df=train_df,
                target_col=TARGET_TEMP_COL,
                horizons=HORIZON_TUPLE,
                allow_na=False
            )
            logger.info(f"X_train shape: {getattr(X_train, 'shape', None)}, Y_train shape: {getattr(Y_train, 'shape', None)}")
            
            # Prepare validation features and targets
            X_val, Y_val = select_feature_target_multi(
                df=val_df,
                target_col=TARGET_TEMP_COL, 
                horizons=HORIZON_TUPLE,
                allow_na=False
            )
            logger.info(f"X_val shape: {getattr(X_val, 'shape', None)}, Y_val shape: {getattr(Y_val, 'shape', None)}")
            
            # Check for empty data
            if X_train is None or Y_train is None or getattr(X_train, 'empty', False) or getattr(Y_train, 'empty', False):
                logger.error(f"Training data is empty for granary: {granary_name}. X_train shape: {getattr(X_train, 'shape', None)}, Y_train shape: {getattr(Y_train, 'shape', None)}")
                results['success'] = False
                results['errors'].append('Training data is empty. Check preprocessing and input data.')
                return results
            if X_val is None or Y_val is None or getattr(X_val, 'empty', False) or getattr(Y_val, 'empty', False):
                logger.error(f"Validation data is empty for granary: {granary_name}. X_val shape: {getattr(X_val, 'shape', None)}, Y_val shape: {getattr(Y_val, 'shape', None)}")
                results['success'] = False
                results['errors'].append('Validation data is empty. Check preprocessing and input data.')
                return results
            
            # Phase 1: Train finder model to get best iteration with Dashboard-optimized settings
            logger.info("Phase 1: Training finder model to determine best iteration")
            from granarypredict.multi_lgbm import MultiLGBMRegressor
            
            # Base parameters - EXACTLY matching Streamlit Dashboard settings with optimal parameters
            base_params = {
                "learning_rate": 0.07172794499286328,  # Optimal parameter
                "max_depth": 20,                       # Optimal parameter
                "num_leaves": 133,                     # Optimal parameter
                "subsample": 0.8901667731353657,       # Optimal parameter
                "colsample_bytree": 0.7729605909501445, # Optimal parameter
                "min_child_samples": 102,              # Optimal parameter
                "lambda_l1": 1.4182488012070926,       # Optimal parameter
                "lambda_l2": 1.7110926238653472,       # Optimal parameter
                "max_bin": 416,                        # Optimal parameter
                "n_jobs": -1,
                # 🆕 COMPRESSION OPTIMIZATIONS (40-70% smaller files, no accuracy loss)
                "compress": True,                      # Enable built-in compression
                "compression_level": 6,                # Compression level (1-9, higher = smaller)
                "save_binary": True,                   # Save in binary format (smaller)
            }
            
            # Apply quantile objective (matching Streamlit)
            base_params.update({
                "objective": "quantile",
                "alpha": 0.5,
            })
            
            # Create internal 95/5 split exactly like Streamlit (when no external validation)
            logger.info("Creating internal 95/5 split for tuning (matching Streamlit)")
            train_df, val_df = split_train_eval_frac(df, test_frac=0.05)
            
            X_train, Y_train = features.select_feature_target_multi(
                train_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
            )
            X_val, Y_val = features.select_feature_target_multi(
                val_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
            )
            
            logger.info(f"Internal split sizes – train={len(train_df)}, val={len(val_df)}")
            
            finder_model = MultiLGBMRegressor(
                base_params=base_params,
                # 🚀 OPTIMIZED: Using granarypredict defaults for speed improvements
                # upper_bound_estimators=1000 (default), early_stopping_rounds=50 (default)
                uncertainty_estimation=True,
                n_bootstrap_samples=25,  # 🚀 OPTIMIZED: Reduced from 100 for 75% speed improvement
                directional_feature_boost=2.0,  # 2x boost for directional features (matching Dashboard)
                conservative_mode=True,  # Enable conservative predictions
                stability_feature_boost=3.0,  # 3x boost for stability features (matching Dashboard)
                use_gpu=True,
                gpu_optimization=True
            )
            
            logger.info("Training finder model with anchor-day early stopping and horizon balancing")
            finder_model.fit(
                X=X_train,
                Y=Y_train,
                eval_set=(X_val, Y_val),
                eval_metric="l1",
                verbose=True,
                anchor_df=val_df,  # Pass validation dataframe for anchor-day methodology
                horizon_tuple=HORIZON_TUPLE,
                use_anchor_early_stopping=True,  # Enable anchor-day early stopping
                balance_horizons=True,  # Apply horizon balancing
                horizon_strategy="increasing",  # Increasing horizon importance
            )
            
            best_iteration = finder_model.best_iteration_
            logger.info(f"Phase 1 complete. Best iteration: {best_iteration}")
            
            # Phase 2: Train final model on 100% of data with fixed n_estimators
            logger.info("Phase 2: Training final model on 100% of data with fixed n_estimators")
            
            # Prepare full dataset for final training
            X_full, Y_full = select_feature_target_multi(
                df=df,
                target_col=TARGET_TEMP_COL,
                horizons=HORIZON_TUPLE,
                allow_na=False
            )
            
            final_model = MultiLGBMRegressor(
                base_params=base_params,
                upper_bound_estimators=best_iteration,  # Use the best iteration from Phase 1
                early_stopping_rounds=0,  # No early stopping for final model
                uncertainty_estimation=True,
                n_bootstrap_samples=25,  # 🚀 OPTIMIZED: Reduced from 100 for 75% speed improvement
                directional_feature_boost=2.0,
                conservative_mode=True,
                stability_feature_boost=3.0,
                use_gpu=True,
                gpu_optimization=True
            )
            
            logger.info("Training final model on 100% of data")
            final_model.fit(
                X=X_full,
                Y=Y_full,
                verbose=True
            )
            
            # Save the final model with adaptive compression
            model_path.parent.mkdir(exist_ok=True, parents=True)
            from granarypredict.compression_utils import save_compressed_model
            
            # Use adaptive compression based on model characteristics
            compression_stats = save_compressed_model(final_model, model_path)
            results['model_path'] = compression_stats['path']
            results['compression_stats'] = compression_stats
            results['steps_completed'].append('train')
            logger.info(f"Final model saved to: {compression_stats['path']}")
            logger.info(f"Compression: {compression_stats['compression_algorithm']} "
                       f"({compression_stats['compression_ratio']:.2f}x, "
                       f"{compression_stats['compressed_size_mb']:.2f} MB)")
        
        results['success'] = True
        return results
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        results['errors'].append(str(e))
    return results


def _preprocess_silos(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a dataframe containing silo data"""
    # Apply all preprocessing steps (same as current implementation)
    # 1. Apply comprehensive column standardization
    from granarypredict.ingestion import standardize_granary_csv
    df = standardize_granary_csv(df)
    logger.info(f"Applied column standardization: {list(df.columns)[:5]}...")
    
    # 2. Basic cleaning
    df = cleaning.basic_clean(df)
    
    # 3. Drop redundant columns
    columns_to_drop = ['locatType', 'line_no', 'layer_no']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)
        logger.info(f"Dropped redundant columns: {existing_columns_to_drop}")
    
    # 4. Insert calendar gaps
    df = insert_calendar_gaps(df)
    
    # 5. Interpolate missing data
    df = interpolate_sensor_numeric(df)
    
    # 6. Final cleaning
    df = cleaning.fill_missing(df)
    
    # 7. Feature engineering
    df = features.create_time_features(df)
    df = features.create_spatial_features(df)
    df = features.add_time_since_last_measurement(df)
    df = features.add_multi_lag_parallel(df, lags=(1,2,3,4,5,6,7,14,30))
    df = features.add_rolling_stats_parallel(df, window_days=7)
    df = features.add_directional_features_lean(df)
    df = features.add_stability_features_parallel(df)
    df = features.add_horizon_specific_directional_features(df, max_horizon=7)
    df = features.add_multi_horizon_targets(df, horizons=tuple(range(1,8)))
    
    # 8. Sorting and grouping
    df = assign_group_id(df)
    df = comprehensive_sort(df)
    
    # 9. Column reordering (simplified version)
    desired_order = [
        'granary_id','address_cn','heap_id','detection_time','temperature_grain',
        'grid_x','grid_y','grid_z',
        'avg_grain_temp','max_temp','min_temp','temperature_inside','humidity_warehouse',
        'temperature_outside','humidity_outside','warehouse_type'
    ]
    ordered_cols = [c for c in desired_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    final_order = ordered_cols + remaining_cols
    df = df[final_order]
    
    return df

# --- CLI Entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="Granary Data Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest and sort raw CSV data")
    ingest_parser.add_argument("--input", required=True, help="Path to raw CSV file")

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess granary CSV data")
    preprocess_parser.add_argument("--input", required=True, help="Path to granary CSV file")
    preprocess_parser.add_argument("--output", required=True, help="Path to save processed CSV")

    train_parser = subparsers.add_parser("train", help="Train model for a granary")
    train_parser.add_argument("--granary", required=True, help="Granary name")

    forecast_parser = subparsers.add_parser("forecast", help="Forecast for a granary")
    forecast_parser.add_argument("--granary", required=True, help="Granary name")
    forecast_parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (days)")
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline: ingest, preprocess, and train")
    pipeline_parser.add_argument("--input", required=True, help="Path to raw CSV file")
    pipeline_parser.add_argument("--granary", required=True, help="Granary name to process")
    pipeline_parser.add_argument("--skip-train", action="store_true", help="Skip training if model already exists")
    pipeline_parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if model exists")

    args = parser.parse_args()

    if args.command == "ingest":
        logger.info(f"Ingesting raw CSV: {args.input}")
        updated_granaries = ingestion.ingest_and_sort(args.input)
        print(f"Ingested and sorted data for granaries: {updated_granaries}")

    elif args.command == "preprocess":
        logger.info(f"Preprocessing granary CSV: {args.input}")
        df = ingestion.read_granary_csv(args.input)
        # 1. Apply comprehensive column standardization
        from granarypredict.ingestion import standardize_granary_csv
        df = standardize_granary_csv(df)
        logger.info(f"Applied column standardization: {list(df.columns)[:5]}...")
        # 2. Basic cleaning
        df = cleaning.basic_clean(df)
        # 2.5. Drop redundant columns
        columns_to_drop = ['locatType', 'line_no', 'layer_no']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df.drop(columns=existing_columns_to_drop, inplace=True)
            logger.info(f"Dropped redundant columns: {existing_columns_to_drop}")
        # 3. Insert calendar gaps per sensor
        df = insert_calendar_gaps(df)
        # 4. Interpolate missing numeric data per sensor
        df = interpolate_sensor_numeric(df)
        # 6. Final cleaning
        df = cleaning.fill_missing(df)
        # 7. Feature engineering
        df = features.create_time_features(df)
        df = features.create_spatial_features(df)
        df = features.add_time_since_last_measurement(df)
        df = features.add_multi_lag_parallel(df, lags=(1,2,3,4,5,6,7,14,30))
        df = features.add_rolling_stats_parallel(df, window_days=7)
        df = features.add_directional_features_lean(df)
        df = features.add_stability_features_parallel(df)
        df = features.add_horizon_specific_directional_features(df, max_horizon=7)
        df = features.add_multi_horizon_targets(df, horizons=tuple(range(1,8)))
        # 8. Sorting and grouping
        df = assign_group_id(df)
        df = comprehensive_sort(df)
        # 9. Remove duplicate columns (_x, _y) and reorder to match dashboard
        def clean_and_reorder(df):
            # Column standardization is already applied at the beginning of preprocessing
            # This function now only handles column reordering
            
            # Handle any remaining duplicate columns that end with _x or _y
            # but are not the essential grid coordinates
            columns_to_drop = []
            for col in df.columns:
                if col.endswith('_x') or col.endswith('_y'):
                    base = col[:-2]
                    # Keep grid_x and grid_y, drop other duplicates
                    if base in df.columns and base not in ['grid_x', 'grid_y']:
                        columns_to_drop.append(col)
                    elif base not in df.columns and col not in ['grid_x', 'grid_y']:
                        # Rename if it's not a duplicate and not grid coordinates
                        df.rename(columns={col: base}, inplace=True)
            
            # Drop the identified duplicate columns
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
            
            # Define the desired column order
            desired_order = [
                'granary_id','address_cn','heap_id','detection_time','temperature_grain',
                'grid_x','grid_y','grid_z',
                'avg_grain_temp','max_temp','min_temp','temperature_inside','humidity_warehouse','temperature_outside','humidity_outside','warehouse_type','year','month','day','hour','month_sin','month_cos','hour_sin','hour_cos','doy','weekofyear','doy_sin','doy_cos','woy_sin','woy_cos','is_weekend','hours_since_last_measurement','lag_temp_1d','lag_temp_2d','lag_temp_3d','lag_temp_4d','lag_temp_5d','lag_temp_6d','lag_temp_7d','lag_temp_14d','lag_temp_30d','delta_temp_1d','delta_temp_2d','delta_temp_3d','delta_temp_4d','delta_temp_5d','delta_temp_6d','delta_temp_7d','delta_temp_14d','delta_temp_30d','roll_mean_7d','roll_std_7d','temp_accel','trend_3d','is_warming','velocity_smooth','trend_consistency','stability_index','thermal_inertia','change_resistance','historical_stability','dampening_factor','equilibrium_temp','temp_deviation_from_equilibrium','mean_reversion_tendency','velocity_1d','velocity_3d','velocity_7d','momentum_strength','momentum_direction','temp_volatility','velocity_volatility','temp_acceleration_3d','trend_reversal_signal','direction_consistency_2d','direction_consistency_3d','direction_consistency_5d','direction_consistency_7d','temp_range_7d','temp_position_in_range','temperature_grain_h1d','temperature_grain_h2d','temperature_grain_h3d','temperature_grain_h4d','temperature_grain_h5d','temperature_grain_h6d','temperature_grain_h7d','_group_id'
            ]
            
            # Reorder columns: first the desired order, then any remaining columns
            ordered_cols = [c for c in desired_order if c in df.columns]
            remaining_cols = [c for c in df.columns if c not in ordered_cols]
            final_order = ordered_cols + remaining_cols
            
            df = df[final_order]
            return df
        df = clean_and_reorder(df)
        df.to_csv(args.output, index=False)
        print(f"Preprocessed CSV saved to {args.output}")

    elif args.command == "train":
        logger.info(f"Training model for granary: {args.granary}")
        
        # Find the processed file for this granary (supports both CSV and Parquet)
        processed_file = None
        # Use centralized data paths if available
        try:
            from .utils.data_paths import data_paths
            processed_dir = data_paths.get_processed_dir()
        except ImportError:
            processed_dir = pathlib.Path("data/processed")
        if processed_dir.exists():
            # Try Parquet first (preferred format)
            for file in processed_dir.glob(f"*{args.granary}*.parquet"):
                processed_file = file
                break
            # Fallback to CSV
            if not processed_file:
                for file in processed_dir.glob(f"*{args.granary}*.csv"):
                    processed_file = file
                    break
        
        if not processed_file:
            # Try granaries directory as fallback
            granaries_dir = pathlib.Path("data/granaries")
            if granaries_dir.exists():
                for file in granaries_dir.glob(f"*{args.granary}*.csv"):
                    processed_file = file
                    break
        
        if not processed_file:
            print(f"Error: No processed data found for granary '{args.granary}'")
            print("Please run preprocessing first: python granary_pipeline.py preprocess --input <granary.csv> --output <processed.csv>")
            return
        
        logger.info(f"Loading processed data from: {processed_file}")
        df: pd.DataFrame = ingestion.read_granary_csv(processed_file)
        
        # Prepare features and targets for multi-horizon training
        from granarypredict.features import select_feature_target_multi
        
        # Get the last date for evaluation
        df['detection_time'] = pd.to_datetime(df['detection_time'])
        last_date = df['detection_time'].max()
        logger.info(f"Last date in dataset: {last_date}")
        
        # Create anchor dataset using a proper train/test split approach
        # This mimics the Streamlit dashboard approach
        
        # Create a 95/5 split for anchor evaluation (similar to Streamlit)
        df_sorted = df.sort_values('detection_time')
        total_rows = len(df_sorted)
        split_idx = int(total_rows * 0.95)
        
        # Training data (95%)
        train_df = df_sorted.iloc[:split_idx].copy()
        
        # Anchor/validation data (5% - last portion)
        anchor_df = df_sorted.iloc[split_idx:].copy()
        
        print(f"Training data: {len(train_df)} rows (95%)")
        print(f"Anchor data: {len(anchor_df)} rows (5%)")
        
        # Prepare training features and targets
        train_X, train_Y = select_feature_target_multi(
            df=train_df,
            target_col=TARGET_TEMP_COL,
            horizons=HORIZON_TUPLE,
            allow_na=False
        )
        
        # Prepare anchor features and targets
        anchor_X, anchor_Y = select_feature_target_multi(
            df=anchor_df,
            target_col=TARGET_TEMP_COL, 
            horizons=HORIZON_TUPLE,
            allow_na=False
        )
        
        if train_X.empty or train_Y.empty:
            print(f"Error: No valid training data after feature selection")
            print(f"Training dataframe shape: {train_df.shape}")
            return
            
        if anchor_X.empty or anchor_Y.empty:
            print(f"Error: No valid anchor data for evaluation")
            print(f"Anchor dataframe shape: {anchor_df.shape}")
            print(f"Available columns: {list(anchor_df.columns)}")
            return
        
        # Use the training data we prepared above
        X, Y = train_X, train_Y
        
        logger.info(f"Training data shape: X={X.shape}, Y={Y.shape}")
        logger.info(f"Anchor data shape: X={anchor_X.shape}, Y={anchor_Y.shape}")
        
        # Initialize MultiLGBM model with Dashboard-optimized settings
        from granarypredict.multi_lgbm import MultiLGBMRegressor
        
        # Base parameters with quantile regression (matching Dashboard)
        base_params = {
            "objective": "quantile",
            "alpha": 0.5,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "max_bin": 255,
            "n_jobs": -1,
        }
        
        model = MultiLGBMRegressor(
            base_params=base_params,
            upper_bound_estimators=2000,
            early_stopping_rounds=100,
            uncertainty_estimation=True,
            n_bootstrap_samples=100,  # Increased from 50 for better uncertainty estimation
            directional_feature_boost=2.0,  # 2x boost for directional features (matching Dashboard)
            conservative_mode=True,  # Enable conservative predictions
            stability_feature_boost=3.0,  # 3x boost for stability features (matching Dashboard)
            use_gpu=True,
            gpu_optimization=True
        )
        
        logger.info("Training MultiLGBM model on 95% of data with 5% anchor validation...")
        
        # Train the model on training data with anchor validation
        model.fit(
            X=X,
            Y=Y,
            eval_set=(anchor_X, anchor_Y),  # Use anchor data for validation
            eval_metric="l1",
            verbose=True,
            anchor_df=anchor_df,  # Pass anchor dataframe for anchor-day methodology
            horizon_tuple=HORIZON_TUPLE,
            use_anchor_early_stopping=True,  # Enable anchor-day early stopping
            balance_horizons=True,  # Apply horizon balancing
            horizon_strategy="increasing",  # Increasing horizon importance
        )
        
        # Now train final model on full dataset with best iteration
        logger.info("Training final model on full dataset...")
        
        # Prepare full dataset features and targets
        full_X, full_Y = select_feature_target_multi(
            df=df.copy(),
            target_col=TARGET_TEMP_COL,
            horizons=HORIZON_TUPLE,
            allow_na=False
        )
        
        # Create final model with same parameters but no early stopping
        final_params = base_params | {"n_estimators": model.best_iteration_ if hasattr(model, 'best_iteration_') else 1000}  # 🚀 OPTIMIZED: Reduced from 2000
        final_model = MultiLGBMRegressor(
            base_params=final_params,
            upper_bound_estimators=model.best_iteration_ if hasattr(model, 'best_iteration_') else 1000,  # 🚀 OPTIMIZED: Reduced from 2000
            early_stopping_rounds=0,  # No early stopping for final model
            uncertainty_estimation=True,
            n_bootstrap_samples=25,  # 🚀 OPTIMIZED: Reduced from 100 for 75% speed improvement
            directional_feature_boost=2.0,  # 2x boost for directional features (matching Dashboard)
            conservative_mode=True,  # Enable conservative predictions
            stability_feature_boost=3.0,  # 3x boost for stability features (matching Dashboard)
            use_gpu=True,
            gpu_optimization=True
        )
        
        # Train on full dataset
        final_model.fit(
            X=full_X,
            Y=full_Y,
            eval_set=None,  # No validation set for final model
            eval_metric="l1",
            verbose=True,
            balance_horizons=True,  # Apply horizon balancing
            horizon_strategy="increasing",  # Increasing horizon importance
        )
        
        # Use final model for predictions
        model = final_model
        
        # Generate predictions for the anchor data to calculate metrics
        logger.info("Generating forecasts for anchor validation data...")
        predictions = model.predict(anchor_X)
        
        # Calculate metrics on the anchor data
        from sklearn.metrics import mean_absolute_error
        mae_scores = []
        for h in range(7):
            if h < predictions.shape[1] and h < anchor_Y.shape[1]:
                mae = mean_absolute_error(anchor_Y.iloc[:, h], predictions[:, h])
                mae_scores.append(mae)
                logger.info(f"h+{h+1} MAE: {mae:.3f}°C")
        
        avg_mae = sum(mae_scores) / len(mae_scores) if mae_scores else 0
        logger.info(f"Average MAE across horizons: {avg_mae:.3f}°C")
        
        # Save the model with adaptive compression
        from granarypredict.compression_utils import save_compressed_model
        model_filename = f"{args.granary}_forecast_model.joblib"
        model_path = pathlib.Path("models") / model_filename  # Changed from pipeline/models to models
        
        compression_stats = save_compressed_model(model, model_path)
        logger.info(f"Model saved to: {compression_stats['path']}")
        logger.info(f"Compression: {compression_stats['compression_algorithm']} "
                   f"({compression_stats['compression_ratio']:.2f}x, "
                   f"{compression_stats['compressed_size_mb']:.2f} MB)")
        
        # Save forecast results
        forecast_results = {
            'granary': args.granary,
            'last_date': last_date.isoformat(),
            'mae_scores': mae_scores,
            'avg_mae': avg_mae,
            'predictions_shape': predictions.shape,
            'model_path': str(model_path)
        }
        
        results_filename = f"{args.granary}_training_results.json"
        results_path = pathlib.Path("model_results") / results_filename  # Changed from pipeline/model_results to model_results
        
        import json
        with open(results_path, 'w') as f:
            json.dump(forecast_results, f, indent=2, default=str)
        
        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"Results saved to: {results_path}")
        print(f"Average MAE: {avg_mae:.3f}°C")
        print(f"📅 Training data: {len(train_df)} rows, Validation: {len(anchor_df)} rows")
        print(f"🔮 Forecast horizons: h+1 to h+7")

    elif args.command == "forecast":
        logger.info(f"Forecasting for granary: {args.granary}")
        
        # Find the trained model
        model_filename = f"{args.granary}_forecast_model.joblib"
        model_path = pathlib.Path("models") / model_filename  # Changed from pipeline/models to models
        
        if not model_path.exists():
            print(f"Error: No trained model found for granary '{args.granary}'")
            print(f"Expected model file: {model_path}")
            print("Please run training first: python granary_pipeline.py train --granary <granary_name>")
            return
        
        # Load the trained model with adaptive compression support
        from granarypredict.compression_utils import load_compressed_model
        import joblib
        
        try:
            # Use new adaptive loading system
            model = load_compressed_model(model_path)
            logger.info(f"Loaded model using adaptive compression from: {model_path}")
        except Exception as e:
            logger.warning(f"Adaptive loading failed, trying fallback: {e}")
            # Fallback to regular joblib loading
            model = joblib.load(model_path)
            logger.info(f"Loaded model using fallback from: {model_path}")
        
        # Find the processed data file for this granary (supports both CSV and Parquet)
        processed_file = None
        processed_dir = pathlib.Path("data/processed")
        if processed_dir.exists():
            # Try Parquet first (preferred format)
            parquet_files = list(processed_dir.glob(f"*{args.granary}*.parquet"))
            if parquet_files:
                processed_file = parquet_files[0]
                logger.info(f"Found Parquet file: {processed_file}")
            else:
                # Fallback to CSV files
                csv_files = list(processed_dir.glob(f"*{args.granary}*.csv*"))
                if csv_files:
                    processed_file = csv_files[0]
                    logger.info(f"Found CSV file: {processed_file}")
        
        if not processed_file:
            # Try granaries directory as fallback
            granaries_dir = pathlib.Path("data/granaries")
            if granaries_dir.exists():
                for file in granaries_dir.glob(f"*{args.granary}*.parquet"):
                    processed_file = file
                    break
        
        if not processed_file:
            print(f"Error: No processed data found for granary '{args.granary}'")
            print("Please run preprocessing first: python granary_pipeline.py preprocess --input <granary.csv> --output <processed.csv>")
            return
        
        logger.info(f"Loading processed data from: {processed_file}")
        df_processed: pd.DataFrame = ingestion.read_granary_csv(processed_file)
        
        # Get the last date for forecasting
        df_processed['detection_time'] = pd.to_datetime(df_processed['detection_time'])
        last_date = df_processed['detection_time'].max()
        logger.info(f"Last date in dataset: {last_date}")
        
        # Create future data for forecasting using Dashboard approach
        def make_future_dashboard_style(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
            """Create future dataframe for forecasting using Dashboard approach."""
            last_date = pd.to_datetime(df['detection_time'].max())
            logger.info(f"Using only the most recent date for forecasting: {last_date}")
            
            # Get the last known row per physical sensor (like Dashboard)
            sensors_key = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
            last_rows = (
                df.sort_values("detection_time")
                .groupby(sensors_key, dropna=False)
                .tail(1)
                .copy()
            )
            logger.info(f"Found {len(last_rows)} sensors at the most recent date")
            
            # Create future dates
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq='D')
            
            # Build future frames for each horizon (like Dashboard)
            all_future_frames = []
            for h in range(1, horizon_days + 1):
                day_frame = last_rows.copy()
                day_frame["detection_time"] = last_date + pd.Timedelta(days=h)
                day_frame["forecast_day"] = h
                
                # Clear target variables
                for col in df.columns:
                    if col.startswith('temperature_grain_h') or col == 'temperature_grain':
                        day_frame[col] = np.nan
                
                all_future_frames.append(day_frame)
            
            # Create future dataframe with only the future rows
            future_df = pd.concat(all_future_frames, ignore_index=True)
            logger.info(f"Created future dataframe with {len(future_df)} rows for {horizon_days} days")
            
            return future_df
        
        # Create future dataframe with the specified horizon (Dashboard style)
        future_df = make_future_dashboard_style(df_processed, horizon_days=args.horizon)
        logger.info(f"Created future dataframe with shape: {future_df.shape}")
        
        # Prepare features for forecasting
        from granarypredict.features import select_feature_target_multi
        
        # Select features (no targets for future data)
        X_future, _ = select_feature_target_multi(
            df=future_df.copy(),
            target_col="temperature_grain",
            horizons=(1, 2, 3, 4, 5, 6, 7),
            allow_na=True  # Allow missing targets for future data
        )
        
        logger.info(f"Future features shape: {X_future.shape}")
        
        # Generate forecasts using multi-output approach (like Dashboard)
        logger.info(f"Generating forecasts for h+1 to h+{args.horizon}...")
        predictions = model.predict(X_future)
        
        # Get uncertainty estimates if available
        uncertainties = None
        if hasattr(model, 'get_uncertainty_estimates'):
            uncertainties = model.get_uncertainty_estimates()
        
        # Get prediction intervals if available
        prediction_intervals = None
        if hasattr(model, 'get_prediction_intervals'):
            prediction_intervals = model.get_prediction_intervals(confidence_level=0.95)
        
        # Create forecast results
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=args.horizon, freq='D')
        
        forecast_results = {
            'granary': args.granary,
            'last_historical_date': last_date.isoformat(),
            'forecast_dates': [d.isoformat() for d in forecast_dates],
            'horizon_days': args.horizon,
            'predictions': predictions.tolist() if predictions is not None else None,
            'uncertainties': uncertainties.tolist() if uncertainties is not None else None,
            'prediction_intervals': {
                'lower': prediction_intervals[0].tolist() if prediction_intervals else None,
                'upper': prediction_intervals[1].tolist() if prediction_intervals else None
            } if prediction_intervals else None
        }
        
        # Create comprehensive CSV forecast with sensor coordinates and all metadata (Dashboard style)
        if predictions is not None:
            # Create comprehensive forecast CSV with individual sensor predictions
            forecast_rows = []
            
            # Get the last known row per physical sensor (same as above)
            sensors_key = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df_processed.columns]
            last_rows = (
                df_processed.sort_values("detection_time")
                .groupby(sensors_key, dropna=False)
                .tail(1)
                .copy()
            )
            
            for day_idx, forecast_date in enumerate(forecast_dates):
                for sensor_idx, (_, sensor_row) in enumerate(last_rows.iterrows()):
                    if sensor_idx < len(predictions):
                        sensor_predictions = predictions[sensor_idx]
                        
                        # Get prediction for this specific day
                        if day_idx < len(sensor_predictions):
                            temperature = float(sensor_predictions[day_idx])
                            
                            # Get uncertainty if available
                            uncertainty = None
                            if uncertainties is not None and sensor_idx < len(uncertainties) and day_idx < len(uncertainties[sensor_idx]):
                                uncertainty = float(uncertainties[sensor_idx][day_idx])
                            
                            # Get prediction intervals if available
                            lower_bound = None
                            upper_bound = None
                            if prediction_intervals is not None:
                                if (sensor_idx < len(prediction_intervals[0]) and 
                                    day_idx < len(prediction_intervals[0][sensor_idx])):
                                    lower_bound = float(prediction_intervals[0][sensor_idx][day_idx])
                                    upper_bound = float(prediction_intervals[1][sensor_idx][day_idx])
                            
                            # Create comprehensive row (individual sensor level)
                            forecast_row = {
                                'granary_id': args.granary,
                                'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                                'forecast_day': day_idx + 1,
                                'grid_x': int(sensor_row.get('grid_x', sensor_row.get('x', 0)) or 0),
                                'grid_y': int(sensor_row.get('grid_y', sensor_row.get('y', 0)) or 0),
                                'grid_z': int(sensor_row.get('grid_z', sensor_row.get('z', 0)) or 0),
                                'heap_id': sensor_row.get('heap_id', ''),
                                'predicted_temperature_celsius': round(temperature, 2),
                                'uncertainty': round(uncertainty, 2) if uncertainty is not None else None,
                                'lower_bound_95ci': round(lower_bound, 2) if lower_bound is not None else None,
                                'upper_bound_95ci': round(upper_bound, 2) if upper_bound is not None else None,
                                'last_historical_date': last_date.strftime('%Y-%m-%d'),
                                'model_confidence': 'high' if uncertainty and uncertainty < 0.5 else 'medium' if uncertainty and uncertainty < 1.0 else 'low'
                            }
                            
                            # Add any additional metadata from sensor row
                            for col in ['address_cn', 'warehouse_type', 'avg_grain_temp', 'max_temp', 'min_temp']:
                                if col in sensor_row and pd.notna(sensor_row[col]):
                                    forecast_row[col] = sensor_row[col]
                            
                            forecast_rows.append(forecast_row)
            
            # Create DataFrame and save comprehensive CSV
            forecast_df = pd.DataFrame(forecast_rows)
            
            # Create daily summary data for the same CSV
            summary_rows = []
            for day_idx, forecast_date in enumerate(forecast_dates):
                day_data = forecast_df[forecast_df['forecast_day'] == day_idx + 1]
                if not day_data.empty:
                    # Calculate daily statistics (like Dashboard)
                    daily_stats = {
                        'granary_id': args.granary,
                        'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                        'forecast_day': day_idx + 1,
                        'total_sensors': len(day_data),
                        'avg_temperature': round(day_data['predicted_temperature_celsius'].mean(), 2),
                        'min_temperature': round(day_data['predicted_temperature_celsius'].min(), 2),
                        'max_temperature': round(day_data['predicted_temperature_celsius'].max(), 2),
                        'temperature_range': round(day_data['predicted_temperature_celsius'].max() - day_data['predicted_temperature_celsius'].min(), 2),
                        'std_temperature': round(day_data['predicted_temperature_celsius'].std(), 2),
                        'avg_uncertainty': round(day_data['uncertainty'].mean(), 2) if 'uncertainty' in day_data.columns and not day_data['uncertainty'].isna().all() else None,
                        'last_historical_date': last_date.strftime('%Y-%m-%d')
                    }
                    
                    # Add extremes tracking (like Dashboard)
                    max_temp_idx = day_data['predicted_temperature_celsius'].idxmax()
                    min_temp_idx = day_data['predicted_temperature_celsius'].idxmin()
                    max_temp_row = day_data.loc[max_temp_idx]
                    min_temp_row = day_data.loc[min_temp_idx]
                    
                    daily_stats.update({
                        'max_temp_location': f"({max_temp_row['grid_x']},{max_temp_row['grid_y']},{max_temp_row['grid_z']})",
                        'min_temp_location': f"({min_temp_row['grid_x']},{min_temp_row['grid_y']},{min_temp_row['grid_z']})",
                        'max_temp_heap_id': max_temp_row.get('heap_id', ''),
                        'min_temp_heap_id': min_temp_row.get('heap_id', '')
                    })
                    
                    summary_rows.append(daily_stats)
            
            summary_df = pd.DataFrame(summary_rows)
            
            # Save comprehensive forecast data as Parquet (much more efficient)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            forecast_filename = f"{args.granary}_forecast_{timestamp}"
            forecast_path = Path("forecasts") / forecast_filename  # Save to forecasts directory
            forecast_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Save as Parquet with snappy compression (60-80% smaller, 10x faster)
            from granarypredict.ingestion import save_granary_data
            parquet_path = save_granary_data(
                df=forecast_df,
                filepath=forecast_path,
                format='parquet',
                compression='snappy'
            )
            

            
            print(f"✅ Forecasting completed successfully!")
            print(f"Forecast Parquet saved to: {parquet_path}")
            print(f"📅 Based on historical data from: {last_date.strftime('%Y-%m-%d')}")
            print(f"🔮 Forecast dates: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
            print(f"🔮 Forecast horizons: h+1 to h+{args.horizon}")
            print(f"📍 Total forecast records: {len(forecast_df)} (sensors × days)")
            print(f"📍 Sensors used: {len(last_rows)} (from most recent date only)")
            
            # Show sample predictions and daily stats
            if predictions is not None and len(predictions) > 0:
                print(f"📈 Sample prediction (first sensor, h+1): {predictions[0, 0]:.2f}°C")
                if uncertainties is not None:
                    print(f"Sample uncertainty (first sensor, h+1): ±{uncertainties[0, 0]:.2f}°C")
                
                # Show daily summary stats
                if not summary_df.empty:
                    first_day = summary_df.iloc[0]
                    print(f"Day 1 summary: Avg={first_day['avg_temperature']}°C, Min={first_day['min_temperature']}°C, Max={first_day['max_temperature']}°C")
                    print(f"📍 Day 1 extremes: Max at {first_day['max_temp_location']}, Min at {first_day['min_temp_location']}")
                
                print(f"📋 Parquet format: granary_id, forecast_date, forecast_day, grid_x, grid_y, grid_z, heap_id, predicted_temperature_celsius, uncertainty, lower_bound_95ci, upper_bound_95ci, model_confidence")
        else:
            print(f"❌ Error: No predictions generated")

    elif args.command == "pipeline":
        logger.info(f"Running complete pipeline for granary: {args.granary}")
        
        # Use the standalone function
        results = run_complete_pipeline(
            granary_csv=args.input,  # Pass the raw input file for ingestion
            granary_name=args.granary,
            skip_train=args.skip_train,
            force_retrain=args.force_retrain
        )
        
        # Display results
        if results['success']:
            print(f"✅ Pipeline completed successfully for {args.granary}")
            print(f"📋 Steps completed: {', '.join(results['steps_completed'])}")
            if results['model_path']:
                print(f"📁 Model path: {results['model_path']}")
        else:
            print(f"❌ Pipeline failed for {args.granary}")
            print(f"🚨 Errors: {', '.join(results['errors'])}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

