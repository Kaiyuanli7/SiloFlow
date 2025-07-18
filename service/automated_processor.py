import asyncio
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import logging
import tempfile
import os
import sys

# Add granarypredict directory to Python path
granarypredict_dir = Path(__file__).parent.parent / "granarypredict"
if str(granarypredict_dir) not in sys.path:
    sys.path.insert(0, str(granarypredict_dir))

logger = logging.getLogger(__name__)

class AutomatedGranaryProcessor:
    def __init__(self):
        # Use centralized data path manager
        from .utils.data_paths import data_paths
        self.data_paths = data_paths
        
        # Get standardized directory paths
        self.models_dir = self.data_paths.get_models_dir()
        self.processed_dir = self.data_paths.get_processed_dir()
        self.granaries_dir = self.data_paths.get_granaries_dir()
        self.forecasts_dir = self.data_paths.get_forecasts_dir()
        self.temp_dir = self.data_paths.get_temp_dir()
        
        logger.info(f"Initialized with standardized data paths:")
        logger.info(f"  Models: {self.models_dir}")
        logger.info(f"  Processed: {self.processed_dir}")
        logger.info(f"  Granaries: {self.granaries_dir}")
        logger.info(f"  Forecasts: {self.forecasts_dir}")
        logger.info(f"  Temp: {self.temp_dir}")
        
    async def process_raw_csv(self, csv_path: str) -> Dict:
        """Enhanced ingestion that returns both granary and silo change information"""
        try:
            from granarypredict import ingestion
            logger.info(f"Ingesting raw CSV: {csv_path}")
            
            # Get detailed change information
            change_info = ingestion.ingest_and_sort(csv_path, return_new_data_status=True)
            
            # Handle both old and new return formats for backward compatibility
            if isinstance(change_info, dict) and 'granary_status' in change_info:
                # New format with silo tracking
                granary_status = change_info['granary_status']
                silo_changes = change_info['silo_changes']
            else:
                # Old format - fallback
                granary_status = change_info if isinstance(change_info, dict) else {}
                silo_changes = {}
            
            new_granaries = [g for g, is_new in granary_status.items() if is_new]
            
            logger.info(f"Granaries with new data: {new_granaries}")
            for granary in new_granaries:
                logger.info(f"  {granary}: Changed silos: {silo_changes.get(granary, [])}")
            
            return {
                'granaries': new_granaries,
                'silo_changes': silo_changes
            }
            
        except Exception as e:
            logger.error(f"Error ingesting CSV: {e}")
            raise
    
    async def process_granary(self, granary_name: str, changed_silos: Optional[List[str]] = None) -> Dict:
        """Process granary with optional silo filtering for preprocessing"""
        try:
            from granary_pipeline import run_complete_pipeline
            
            granary_csv = self.granaries_dir / f"{granary_name}.parquet"
            
            if not granary_csv.exists():
                return {
                    'success': False,
                    'errors': [f"Granary Parquet file not found: {granary_csv}"],
                    'steps_completed': []
                }
            
            # Check if model exists
            model_path = self.models_dir / f"{granary_name}_forecast_model.joblib"
            skip_train = model_path.exists()
            
            logger.info(f"Processing granary: {granary_name}")
            if changed_silos:
                logger.info(f"  Focus on changed silos: {changed_silos}")
            
            # Run pipeline with silo filtering for preprocessing
            results = run_complete_pipeline(
                granary_csv=str(granary_csv),
                granary_name=granary_name,
                skip_train=skip_train,
                force_retrain=False,
                changed_silos=changed_silos  # ✅ Pass changed silos info
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing granary {granary_name}: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'steps_completed': []
            }
    
    async def generate_forecasts(self, granary_name: str, horizon: int = 7, changed_silos: Optional[List[str]] = None) -> Optional[Dict]:
        """Generate h+1 to h+7 forecasts for a granary using the same method as Streamlit Dashboard
        
        Parameters:
        -----------
        granary_name : str
            Name of the granary to forecast
        horizon : int
            Forecast horizon in days (default: 7)
        changed_silos : Optional[List[str]]
            List of silo IDs that were updated. If provided, only forecast for these silos.
        """
        try:
            logger.info(f"Generating forecasts for {granary_name} (horizon: {horizon})")
            if changed_silos:
                logger.info(f"  Forecasting only for changed silos: {changed_silos}")
            
            # Load the trained model (same as Streamlit)
            model_filename = f"{granary_name}_forecast_model.joblib"
            model_path = self.models_dir / model_filename
            
            if not model_path.exists():
                logger.error(f"No trained model found for granary '{granary_name}'")
                return None
            
            import joblib
            from granarypredict.compression_utils import load_compressed_model
            
            # Try loading as compressed model first
            try:
                model = load_compressed_model(model_path, use_gzip=True)
            except Exception:
                # Fallback to regular joblib loading
                model = joblib.load(model_path)
            logger.info(f"Loaded model from: {model_path}")
            
            # Load processed data (same as Streamlit) - supports both CSV and Parquet
            processed_path = self.processed_dir / f"{granary_name}_processed"
            
            # Try Parquet first (preferred format)
            parquet_file = processed_path.with_suffix('.parquet')
            csv_file = processed_path.with_suffix('.csv')
            
            if parquet_file.exists():
                logger.info(f"Loading processed data from Parquet: {parquet_file}")
                from granarypredict.ingestion import read_granary_csv
                df_processed = read_granary_csv(parquet_file)
            elif csv_file.exists():
                logger.info(f"Loading processed data from CSV: {csv_file}")
                from granarypredict.ingestion import read_granary_csv
                df_processed = read_granary_csv(csv_file)
            else:
                logger.error(f"No processed data found for granary '{granary_name}'")
                logger.error(f"Checked for: {parquet_file} and {csv_file}")
                return None
            
            logger.info(f"Loaded processed data with {len(df_processed)} rows")
            
            # Filter to changed silos if specified (for preprocessing consistency)
            if changed_silos and 'heap_id' in df_processed.columns:
                original_count = len(df_processed)
                df_processed = df_processed[df_processed['heap_id'].isin(changed_silos)]
                logger.info(f"Filtered to {len(df_processed)} rows for {len(changed_silos)} changed silos")
            
            # Use EXACTLY the same forecast generation method as Streamlit Dashboard
            from granarypredict import features
            from granarypredict.multi_lgbm import MultiLGBMRegressor
            from sklearn.multioutput import MultiOutputRegressor
            from datetime import timedelta
            import numpy as np
            
            # Constants matching Streamlit
            TARGET_TEMP_COL = 'temperature_grain'
            HORIZON_TUPLE = tuple(range(1, 8))  # 1-7 days
            HORIZON_DAYS = 7
            
            # Get feature columns function (same as Streamlit)
            def get_feature_cols(model, X_fallback):
                if hasattr(model, "feature_names_in_"):
                    return list(getattr(model, "feature_names_in_"))
                if hasattr(model, "feature_name_"):
                    return list(getattr(model, "feature_name_"))
                return list(X_fallback.columns)
            
            # Use Streamlit's exact forecast generation logic
            if isinstance(model, (MultiOutputRegressor, MultiLGBMRegressor)) and horizon <= HORIZON_DAYS:
                # 1. Take last known row per physical sensor as input snapshot (EXACTLY like Streamlit)
                sensors_key = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df_processed.columns]
                
                last_rows = (
                    df_processed.sort_values("detection_time")
                    .groupby(sensors_key, dropna=False)
                    .tail(1)
                    .copy()
                )
                
                # 2. Prepare design matrix (EXACTLY like Streamlit)
                X_snap, _ = features.select_feature_target_multi(
                    last_rows, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE, allow_na=True
                )
                model_feats = get_feature_cols(model, X_snap)
                X_snap_aligned = X_snap.reindex(columns=model_feats, fill_value=0)
                
                # 3. Direct multi-output prediction (EXACTLY like Streamlit)
                preds_mat = model.predict(X_snap_aligned)  # shape (n, 7)
                
                # 4. Build future frames for each horizon (EXACTLY like Streamlit)
                all_future_frames = []
                last_dt = pd.to_datetime(last_rows["detection_time"]).max()
                
                for h in range(1, horizon + 1):
                    day_frame = last_rows.copy()
                    day_frame["detection_time"] = last_dt + timedelta(days=h)
                    day_frame["forecast_day"] = h
                    
                    idx = min(h - 1, preds_mat.shape[1] - 1)  # fallback to last available output
                    if preds_mat.ndim == 2:
                        pred_val = preds_mat[:, idx]
                    else:
                        pred_val = preds_mat  # 1-D: same value for all horizons
                    
                    day_frame["predicted_temp"] = pred_val
                    day_frame["temperature_grain"] = pred_val
                    day_frame[TARGET_TEMP_COL] = pred_val
                    day_frame["is_forecast"] = True
                    all_future_frames.append(day_frame)
                
                future_df = pd.concat(all_future_frames, ignore_index=True)
                
                # Clear actual temperature values for forecast rows (EXACTLY like Streamlit)
                future_df.loc[future_df["is_forecast"], TARGET_TEMP_COL] = np.nan
                
            else:
                # Fallback for other model types (same as Streamlit)
                logger.warning(f"Using fallback forecast method for model type: {type(model)}")
                # ... implement fallback logic if needed
                return None
            
            # Create comprehensive CSV forecast (same format as CLI but with Streamlit logic)
            forecast_rows = []
            for _, row in future_df.iterrows():
                forecast_row = {
                    'granary_id': granary_name,
                    'forecast_date': pd.to_datetime(row['detection_time']).strftime('%Y-%m-%d'),
                    'forecast_day': int(row['forecast_day']),
                    'grid_x': int(row.get('grid_x', row.get('x', 0)) or 0),
                    'grid_y': int(row.get('grid_y', row.get('y', 0)) or 0),
                    'grid_z': int(row.get('grid_z', row.get('z', 0)) or 0),
                    'heap_id': row.get('heap_id', ''),
                    'predicted_temperature_celsius': round(float(row['predicted_temp']), 2),
                    'last_historical_date': last_dt.strftime('%Y-%m-%d'),
                    'model_confidence': 'high'  # Default confidence
                }
                
                # Add metadata
                for col in ['address_cn', 'warehouse_type', 'avg_grain_temp', 'max_temp', 'min_temp']:
                    if col in row and not pd.isna(row[col]):
                        forecast_row[col] = row[col]
                
                forecast_rows.append(forecast_row)
                
            # Create DataFrame and convert to CSV for API response
            forecast_df = pd.DataFrame(forecast_rows)
            csv_content = forecast_df.to_csv(index=False)
            
            # Save forecast as Parquet for efficiency
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            forecast_filename = f"{granary_name}_forecast_{timestamp}"
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
            logger.info(f"Saved forecast Parquet to: {parquet_path}")
                
            return {
                "granary_name": granary_name,
                "forecast_horizon_days": horizon,
                "total_records": len(forecast_df),
                "csv_content": csv_content,
                "csv_filename": forecast_filename,
                "columns": list(forecast_df.columns),
                "changed_silos": [int(silo) for silo in changed_silos] if changed_silos else None,
                "summary": {
                    "min_temperature": float(forecast_df['predicted_temperature_celsius'].min()),
                    "max_temperature": float(forecast_df['predicted_temperature_celsius'].max()),
                    "average_temperature": float(forecast_df['predicted_temperature_celsius'].mean()),
                    "total_sensors": len(forecast_df[['grid_x', 'grid_y', 'grid_z']].drop_duplicates()),
                    "forecast_dates": sorted(forecast_df['forecast_date'].unique().tolist())
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating forecasts for {granary_name}: {e}")
            return None
    
    async def process_all_granaries(self, csv_path: str) -> Dict:
        """Complete automated pipeline with silo-level tracking"""
        try:
            logger.info(f"Starting automated pipeline for: {csv_path}")
            
            # 1. Enhanced ingestion with silo tracking
            change_info = await self.process_raw_csv(csv_path)
            new_granaries = change_info['granaries']
            silo_changes = change_info['silo_changes']
            
            if not new_granaries:
                return {
                    'success': True,
                    'errors': ['No new data found in CSV'],
                    'granaries_processed': 0,
                    'forecasts': {}
                }
            
            # 2. Process each granary with silo information
            results = {}
            successful_granaries = 0
            
            for granary in new_granaries:
                logger.info(f"Processing granary: {granary}")
                changed_silos = silo_changes.get(granary, [])
                
                # Process granary (preprocess with silo focus, train on full granary)
                process_result = await self.process_granary(granary, changed_silos)
                
                if process_result['success']:
                    # Generate forecasts for changed silos only
                    forecasts = await self.generate_forecasts(granary, changed_silos=changed_silos)
                    successful_granaries += 1
                    
                    results[granary] = {
                        "processing": process_result,
                        "forecasts": forecasts,
                        "changed_silos": [int(silo) for silo in changed_silos] if changed_silos else None  # Track which silos were updated
                    }
                else:
                    results[granary] = {
                        "processing": process_result,
                        "forecasts": None,
                        "changed_silos": [int(silo) for silo in changed_silos] if changed_silos else None
                    }
            
            return {
                'success': True,
                'granaries_processed': len(new_granaries),
                'successful_granaries': successful_granaries,
                'forecasts': results
            }
            
        except Exception as e:
            logger.error(f"Error in automated pipeline: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'granaries_processed': 0,
                'forecasts': {}
            }
    
    def cleanup_temp_files(self, temp_path: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file {temp_path}: {e}") 