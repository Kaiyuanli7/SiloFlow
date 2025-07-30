from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from core import processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


@router.get("/forecast", tags=["forecast"])  
async def forecast_silo_endpoint(
    granary_name: str = Query(..., description="Name or ID of the granary"),
    silo_id: str = Query(..., description="ID of the silo within the granary"),
    horizon_days: int = Query(7, description="Number of days to forecast", ge=1, le=30)
):
    """Generate forecasts for a specific silo within a granary using the latest available data."""
    try:
        # Validate inputs
        if not granary_name or not granary_name.strip():
            raise HTTPException(status_code=400, detail="granary_name is required")
        if not silo_id or not silo_id.strip():
            raise HTTPException(status_code=400, detail="silo_id is required")
        
        granary_name = granary_name.strip()
        silo_id = silo_id.strip()
        
        logger.info(f"Starting forecast for granary '{granary_name}', silo '{silo_id}', horizon {horizon_days} days")
        
        # Check if processed data exists for this granary
        processed_file = processor.processed_dir / f"{granary_name}_processed.parquet"
        if not processed_file.exists():
            error_msg = f"Processed file not found: {processed_file}"
            print(error_msg)
            logger.error(error_msg)
            # Check for similar files
            similar_files = list(processor.processed_dir.glob(f"*{granary_name}*_processed.parquet"))
            similar_files_msg = f"Similar processed files found: {[str(f) for f in similar_files]}"
            print(similar_files_msg)
            logger.error(similar_files_msg)
            raise HTTPException(
                status_code=404,
                detail=f"No processed data found for granary '{granary_name}'. Run /pipeline first. Checked path: {processed_file}. Similar files: {[str(f) for f in similar_files]}"
            )

        # Check if model exists for this granary
        model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib"
        compressed_model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib.gz"

        if not (model_file.exists() or compressed_model_file.exists()):
            error_msg = f"Model file not found for granary '{granary_name}'. Checked: {model_file}, {compressed_model_file}"
            print(error_msg)
            logger.error(error_msg)
            similar_models = list(processor.models_dir.glob(f"*{granary_name}*_forecast_model.joblib*"))
            similar_models_msg = f"Similar model files found: {[str(f) for f in similar_models]}"
            print(similar_models_msg)
            logger.error(similar_models_msg)
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for granary '{granary_name}'. Train the model first. Checked: {model_file}, {compressed_model_file}. Similar files: {[str(f) for f in similar_models]}"
            )

        # Load processed data and filter for the specific silo
        try:
            logger.info(f"Loading processed data from: {processed_file}")
            df = pd.read_parquet(processed_file)
        except Exception as e:
            error_msg = f"Failed to load processed data from {processed_file}: {e}"
            print(error_msg)
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

        # Identify the silo column (try different possible names)
        silo_columns = ['heap_id', 'silo_id', 'storepointId']
        silo_col = None
        for col in silo_columns:
            if col in df.columns:
                silo_col = col
                break

        if silo_col is None:
            error_msg = f"No silo identifier column found in data. Columns present: {list(df.columns)}"
            print(error_msg)
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=f"No silo identifier column found in data. Expected one of: {silo_columns}. Columns present: {list(df.columns)}"
            )

        # Filter data for the specific silo
        silo_data = df[df[silo_col] == silo_id].copy()
        if silo_data.empty:
            available_silos = df[silo_col].unique().tolist()
            error_msg = f"No data found for silo '{silo_id}' in granary '{granary_name}'. Available silos: {available_silos}"
            print(error_msg)
            logger.warning(error_msg)
            raise HTTPException(
                status_code=404,
                detail=f"Silo '{silo_id}' not found in granary '{granary_name}'. Available silos: {available_silos[:10]}"
            )

        logger.info(f"Found {len(silo_data)} records for silo '{silo_id}' in granary '{granary_name}'")

        # Get the latest data point for this silo
        try:
            silo_data['detection_time'] = pd.to_datetime(silo_data['detection_time'])
            silo_data = silo_data.sort_values('detection_time')
            latest_date = silo_data['detection_time'].max()
        except Exception as e:
            error_msg = f"Failed to process detection_time for silo '{silo_id}' in granary '{granary_name}': {e}"
            print(error_msg)
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

        # Get the most recent data for forecasting (use sensors from the latest date)
        latest_data = silo_data[silo_data['detection_time'] == latest_date].copy()

        logger.info(f"Using {len(latest_data)} sensors from latest date: {latest_date}")

        if latest_data.empty:
            error_msg = f"No recent data found for silo '{silo_id}' in granary '{granary_name}' on latest date {latest_date}"
            print(error_msg)
            logger.warning(error_msg)
            raise HTTPException(
                status_code=404,
                detail=error_msg
            )
        
        # Generate forecasts using the granary pipeline forecasting logic
        forecast_result = await generate_silo_forecast(
            granary_name=granary_name,
            silo_data=latest_data,
            model_file=model_file if model_file.exists() else compressed_model_file,
            horizon_days=horizon_days,
            silo_id=silo_id,
            latest_date=latest_date
        )
        
        if not forecast_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Forecast generation failed: {forecast_result.get('error', 'Unknown error')}"
            )
        
        # Return forecast results
        payload = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "granary_name": granary_name,
            "silo_id": silo_id,
            "horizon_days": horizon_days,
            "latest_data_date": latest_date.isoformat(),
            "sensors_used": len(latest_data),
            "forecasts": forecast_result["forecasts"],
            "summary": forecast_result.get("summary", {})
        }
        
        logger.info(f"Successfully generated {len(forecast_result['forecasts'])} forecast records for silo '{silo_id}'")
        return JSONResponse(content=payload)
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unexpected error in silo forecast for granary '{granary_name}', silo '{silo_id}': %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


async def generate_silo_forecast(
    granary_name: str,
    silo_data: pd.DataFrame,
    model_file: Path,
    horizon_days: int,
    silo_id: str,
    latest_date: pd.Timestamp
) -> dict:
    """Generate forecasts for a specific silo using the trained model."""
    try:
        # Import necessary modules
        from granarypredict.model import load_model, predict
        from granarypredict.features import select_feature_target_multi
        import numpy as np
        
        logger.info(f"Loading model from: {model_file}")
        model = load_model(str(model_file))
        
        if model is None:
            return {"success": False, "error": "Failed to load model"}
        
        # Prepare features for forecasting using the same logic as granary_pipeline.py
        horizons = tuple(range(1, horizon_days + 1))
        
        # Create future dataframe for forecasting (similar to Dashboard approach)
        future_rows = []
        
        # Get the last known row per physical sensor 
        sensor_cols = [c for c in ['granary_id', 'heap_id', 'grid_x', 'grid_y', 'grid_z'] 
                      if c in silo_data.columns]
        
        if sensor_cols:
            # Group by sensor location to get latest reading per sensor
            last_sensors = (silo_data.sort_values('detection_time')
                           .groupby(sensor_cols, dropna=False)
                           .tail(1)
                           .copy())
        else:
            # Fallback: use all data from latest date
            last_sensors = silo_data.copy()
        
        logger.info(f"Using {len(last_sensors)} sensors for forecasting")
        
        # Generate future dates for each sensor
        for horizon in horizons:
            future_date = latest_date + pd.Timedelta(days=horizon)
            
            for _, sensor_row in last_sensors.iterrows():
                future_row = sensor_row.copy()
                future_row['detection_time'] = future_date
                future_row['forecast_horizon'] = horizon
                future_row['is_forecast'] = True
                future_rows.append(future_row)
        
        future_df = pd.DataFrame(future_rows)
        
        # Select features for forecasting
        X_future, _ = select_feature_target_multi(
            df=future_df.copy(),
            target_col="temperature_grain",
            horizons=horizons,
            allow_na=True
        )

        if X_future.empty:
            return {"success": False, "error": "No valid features generated for forecasting"}

        # Align prediction features to trained model's feature set
        trained_features = None
        # Try LightGBM sklearn API
        if hasattr(model, 'feature_name_'):
            trained_features = list(model.feature_name_)
        # Try native Booster API
        elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
            trained_features = list(model.booster_.feature_name())
        # Try custom attribute
        elif hasattr(model, 'feature_names'):
            trained_features = list(model.feature_names)
        if trained_features:
            missing = [f for f in trained_features if f not in X_future.columns]
            extra = [f for f in X_future.columns if f not in trained_features]
            logger.info(f"Aligning prediction features. Model expects {len(trained_features)} features. Missing: {missing}. Extra: {extra}")
            print(f"Aligning prediction features. Model expects {len(trained_features)} features. Missing: {missing}. Extra: {extra}")
            X_future = X_future.reindex(columns=trained_features)

        logger.info(f"Generated feature matrix with shape: {X_future.shape}")
        print(f"Generated feature matrix with shape: {X_future.shape}")

        # Generate predictions
        predictions = predict(model, X_future)

        if predictions is None:
            return {"success": False, "error": "Model prediction failed"}

        logger.info(f"Generated predictions with shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        print(f"Generated predictions with shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        
        # Create forecast results
        forecast_records = []
        prediction_idx = 0
        
        for horizon in range(1, horizon_days + 1):
            horizon_data = future_df[future_df['forecast_horizon'] == horizon]
            
            for sensor_idx, (_, sensor_row) in enumerate(horizon_data.iterrows()):
                if prediction_idx < len(predictions):
                    # Get prediction for this sensor and horizon
                    if predictions.ndim == 2:
                        # Multi-output model: use appropriate horizon column
                        pred_col = min(horizon - 1, predictions.shape[1] - 1)
                        predicted_temp = float(predictions[prediction_idx, pred_col])
                    else:
                        # Single output: use as-is
                        predicted_temp = float(predictions[prediction_idx])
                    
                    forecast_record = {
                        'granary_id': granary_name,
                        'silo_id': silo_id,
                        'forecast_date': (latest_date + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d'),
                        'forecast_horizon': horizon,
                        'grid_x': int(sensor_row.get('grid_x', sensor_row.get('x', 0)) or 0),
                        'grid_y': int(sensor_row.get('grid_y', sensor_row.get('y', 0)) or 0),
                        'grid_z': int(sensor_row.get('grid_z', sensor_row.get('z', 0)) or 0),
                        'predicted_temperature_celsius': round(predicted_temp, 2),
                        'base_date': latest_date.strftime('%Y-%m-%d'),
                        'sensor_location': f"({sensor_row.get('grid_x', 0)},{sensor_row.get('grid_y', 0)},{sensor_row.get('grid_z', 0)})"
                    }
                    
                    # Add uncertainty information if available
                    if hasattr(model, '_last_prediction_intervals'):
                        intervals = model._last_prediction_intervals
                        if 'uncertainties' in intervals and prediction_idx < len(intervals['uncertainties']):
                            pred_col = min(horizon - 1, intervals['uncertainties'].shape[1] - 1)
                            forecast_record['uncertainty_std'] = round(float(intervals['uncertainties'][prediction_idx, pred_col]), 2)
                        
                        if 'lower_95' in intervals and prediction_idx < len(intervals['lower_95']):
                            pred_col = min(horizon - 1, intervals['lower_95'].shape[1] - 1)
                            forecast_record['confidence_lower_95'] = round(float(intervals['lower_95'][prediction_idx, pred_col]), 2)
                            
                        if 'upper_95' in intervals and prediction_idx < len(intervals['upper_95']):
                            pred_col = min(horizon - 1, intervals['upper_95'].shape[1] - 1)
                            forecast_record['confidence_upper_95'] = round(float(intervals['upper_95'][prediction_idx, pred_col]), 2)
                    
                    forecast_records.append(forecast_record)
                    prediction_idx += 1
        
        # Calculate summary statistics
        if forecast_records:
            temps = [r['predicted_temperature_celsius'] for r in forecast_records]
            summary = {
                'total_predictions': len(forecast_records),
                'sensors_forecasted': len(last_sensors),
                'avg_predicted_temperature': round(np.mean(temps), 2),
                'min_predicted_temperature': round(np.min(temps), 2),
                'max_predicted_temperature': round(np.max(temps), 2),
                'temperature_range': round(np.max(temps) - np.min(temps), 2),
                'forecast_horizons': list(range(1, horizon_days + 1)),
                'model_type': type(model).__name__ if model else 'unknown'
            }
        else:
            summary = {'total_predictions': 0, 'error': 'No forecasts generated'}
        
        return {
            "success": True,
            "forecasts": forecast_records,
            "summary": summary
        }
        
    except Exception as e:
        logger.exception(f"Error generating silo forecast: {e}")
        return {"success": False, "error": str(e)}


@router.get("/forecast/all", tags=["forecast"])
async def forecast_all_endpoint():
    """Generate a *single-day* forecast for every processed granary that has a trained model."""
    try:
        processed_files = list(processor.processed_dir.glob("*_processed.parquet"))
        if not processed_files:
            raise HTTPException(status_code=400, detail="No processed data available. Run /pipeline first.")

        horizon = 1  # Only latest day
        forecasts: dict[str, dict] = {}
        skipped: list[str] = []

        for parquet_path in processed_files:
            granary_name = parquet_path.stem.replace("_processed", "")
            model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib"
            compressed_model_file = processor.models_dir / f"{granary_name}_forecast_model.joblib.gz"

            if not (model_file.exists() or compressed_model_file.exists()):
                skipped.append(granary_name)
                continue

            # This would need to be implemented if the generate_forecasts method exists
            # forecast_result = await processor.generate_forecasts(granary_name, horizon=horizon)
            # For now, skip since the method was removed
            skipped.append(granary_name)
            continue

        payload = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "horizon_days": horizon,
            "forecasts_count": len(forecasts),
            "skipped": skipped,
            "forecasts": forecasts,
            "note": "Legacy endpoint - use /forecast with granary_name and silo_id parameters instead"
        }
        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /forecast/all: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") 