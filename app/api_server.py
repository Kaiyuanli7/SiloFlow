# api_server.py
# FastAPI server for SiloFlow forecasting API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pathlib
import uvicorn
from granarypredict.model import load_model, predict
from granarypredict.features import select_feature_target_multi
from service.scripts.simple_data_retrieval import SimpleDataRetriever, load_config
import Dashboard


app = FastAPI()

# Pydantic model for request body
class ForecastRequest(BaseModel):
    granary_name: str
    silo_id: str

# Pydantic model for request body
@app.post("/api/forecast")
def forecast_endpoint(request: ForecastRequest):
    from datetime import datetime, timedelta
    import logging

    # Setup terminal logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("api_server")

    try:
        granary_name = str(request.granary_name).strip()
        silo_id = str(request.silo_id).strip()
        logger.info(f"Received forecast request: granary_name={granary_name}, silo_id={silo_id}")
        # Validate input
        if not all([granary_name, silo_id]):
            logger.error("Missing required parameters.")
            raise HTTPException(status_code=400, detail="Missing required parameters: granary_name, silo_id are required.")

        # Use last 30 days by default
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Use granary_name as the model name
        import glob
        model_dir = os.path.join("models")
        pattern = f"{granary_name}_processed_lgbm_*.joblib"
        all_files = os.listdir(model_dir) if os.path.exists(model_dir) else []
        logger.info(f"All files in models dir: {all_files}")
        logger.info(f"Glob pattern used: {pattern}")
        model_files = glob.glob(os.path.join(model_dir, pattern))
        logger.info(f"Files found by glob: {model_files}")
        if not model_files:
            logger.error(f"Model file not found for granary_name {granary_name}")
            raise HTTPException(status_code=404, detail=f"Model file not found for granary_name {granary_name}")
        model_path = model_files[0]
        logger.info(f"Using model file: {model_path}")

        # Query DB for silo data using granary_name and silo_id
        config = load_config("service/config/streaming_config.json")
        retriever = SimpleDataRetriever(config['database'])
        # Step 1: Find silo info
        all_data = retriever.get_all_granaries_and_silos()
        details_df = retriever.get_granaries_with_details()
        granary_names = details_df['granary_name'].unique().tolist()
        logger.info(f"All granaries found in database: {granary_names}")
        matching_granary = details_df[details_df['granary_name'] == granary_name]
        if matching_granary.empty:
            logger.error(f"Granary '{granary_name}' not found in database. Available granaries: {granary_names}")
            raise HTTPException(status_code=404, detail=f"Granary '{granary_name}' not found in database. Available granaries: {granary_names}")
        granary_id = matching_granary['storepoint_id'].iloc[0]
        silo_info = all_data[(all_data['storepoint_id'] == granary_id) & (all_data['store_id'] == silo_id)]
        if silo_info.empty:
            logger.error(f"Silo '{silo_id}' not found in granary '{granary_name}' (ID: {granary_id})")
            raise HTTPException(status_code=404, detail=f"Silo '{silo_id}' not found in granary '{granary_name}' (ID: {granary_id})")
        sub_table_id = silo_info['sub_table_id'].iloc[0]
        logger.info(f"Found silo: {silo_info['store_name'].iloc[0]}")
        logger.info(f"   Granary ID: {granary_id}")
        logger.info(f"   Sub-table ID: {sub_table_id}")

        # Step 2: Retrieve the data
        df = retriever.get_silo_data(granary_id, silo_id, sub_table_id, start_date, end_date)
        if df.empty:
            logger.warning("No data found for the specified criteria.")
            raise HTTPException(status_code=404, detail="No data found for the specified criteria")
        logger.info(f"Retrieved {len(df)} records from database.")
        # Output raw pulled data as CSV for debugging
        try:
            raw_csv_path = f"debug_raw_{granary_name}_{silo_id}_{start_date}_{end_date}.csv"
            df.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"Raw pulled data written to {raw_csv_path}")
        except Exception as e:
            logger.error(f"Failed to write raw CSV: {e}")

        # Apply all preprocessing steps from Dashboard.py
        try:
            # 1. Standardize columns
            from granarypredict.ingestion import standardize_granary_csv
            df_std = standardize_granary_csv(df)
            logger.info(f"Standardization complete. Columns: {list(df_std.columns)}")
            # 2. Basic cleaning
            from granarypredict import cleaning
            df_clean = cleaning.basic_clean(df_std)
            logger.info(f"Basic cleaning complete. Columns: {list(df_clean.columns)}")
            # 3. Fill missing values
            df_filled = cleaning.fill_missing(df_clean)
            logger.info(f"Fill missing complete. Columns: {list(df_filled.columns)}")
            # 4. Time features
            from granarypredict import features
            df_time = features.create_time_features(df_filled)
            logger.info(f"Time features added. Columns: {list(df_time.columns)}")
            # 5. Spatial features
            df_spatial = features.create_spatial_features(df_time)
            logger.info(f"Spatial features added. Columns: {list(df_spatial.columns)}")
            # 6. Time since last measurement
            df_quality = features.add_time_since_last_measurement(df_spatial)
            logger.info(f"Time since last measurement added. Columns: {list(df_quality.columns)}")
            # 7. Lag features
            df_lag = features.add_multi_lag_parallel(df_quality, lags=(1,2,3,4,5,6,7,14,30))
            logger.info(f"Lag features added. Columns: {list(df_lag.columns)}")
            # 8. Rolling stats
            df_roll = features.add_rolling_stats_parallel(df_lag, window_days=7)
            logger.info(f"Rolling stats added. Columns: {list(df_roll.columns)}")
            # 9. Directional features
            df_dir = features.add_directional_features_lean(df_roll)
            logger.info(f"Directional features added. Columns: {list(df_dir.columns)}")
            # 10. Stability features
            df_stab = features.add_stability_features_parallel(df_dir)
            logger.info(f"Stability features added. Columns: {list(df_stab.columns)}")
            # 11. Horizon-specific directional features
            HORIZON_DAYS = 7
            df_horizon = features.add_horizon_specific_directional_features(df_stab, max_horizon=HORIZON_DAYS)
            logger.info(f"Horizon-specific directional features added. Columns: {list(df_horizon.columns)}")
            # 12. Multi-horizon targets
            HORIZON_TUPLE = (1, 2, 3, 4, 5, 6, 7)
            df_targets = features.add_multi_horizon_targets(df_horizon, horizons=HORIZON_TUPLE)
            logger.info(f"Multi-horizon targets added. Columns: {list(df_targets.columns)}")
            # 13. Assign group id
            from granarypredict.data_utils import assign_group_id, comprehensive_sort
            df_grouped = assign_group_id(df_targets)
            logger.info(f"Group id assigned. Columns: {list(df_grouped.columns)}")
            # 14. Comprehensive sort
            processed_df = comprehensive_sort(df_grouped)
            logger.info(f"Comprehensive sort complete. Columns: {list(processed_df.columns)}")
            # Output processed data as CSV for debugging
            processed_csv_path = f"debug_processed_{granary_name}_{silo_id}_{start_date}_{end_date}.csv"
            processed_df.to_csv(processed_csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"Processed data written to {processed_csv_path}")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")


        # Load the trained model (no training allowed)
        try:
            model = Dashboard.load_trained_model(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

        # Ensure no training is performed in forecast path
        if hasattr(model, "fit") and getattr(model, "_is_fitted", True) is False:
            logger.error("Model is not trained. Forecasting should not trigger training.")
            raise HTTPException(status_code=500, detail="Model is not trained. Forecasting should not trigger training.")

        try:
            features = Dashboard.get_feature_cols(model, processed_df)
            if not features:
                logger.error("No feature columns found for prediction.")
                raise HTTPException(status_code=500, detail="No feature columns found for prediction.")
            logger.info(f"Selected features: {features}")
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Feature selection failed: {e}")

        # Use the FastAPI-compatible forecast function (guaranteed no training)
        evaluations = {
            model_path: {
                "df_base": processed_df,
                "categories_map": {},
            }
        }
        forecast_result = Dashboard.generate_and_store_forecast_api(model_path, horizon=7, evaluations=evaluations)
        if not forecast_result.get("success"):
            logger.error(f"Forecasting failed: {forecast_result.get('error')}")
            raise HTTPException(status_code=500, detail=f"Forecasting failed: {forecast_result.get('error')}")
        logger.info("Forecast request completed successfully.")
        # Return all forecast rows as JSON
        future_df = forecast_result["future_df"]
        core_cols = [
            "granary_id",
            "heap_id",
            "grid_x",
            "grid_y",
            "grid_z",
            "detection_time",
            "forecast_day",
            "predicted_temp",
        ]
        if future_df is not None:
            forecast_json = future_df[core_cols].to_dict(orient="records") if all(c in future_df.columns for c in core_cols) else future_df.to_dict(orient="records")
        else:
            forecast_json = []
        return {
            "success": True,
            "parquet_path": forecast_result.get("parquet_path"),
            "rows": len(forecast_json),
            "forecasts": forecast_json
        }

    except HTTPException as he:
        # Already logged and raised
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    except HTTPException as he:
        # Already logged and raised
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("api_server")
    logger.info("Starting SiloFlow FastAPI service on 0.0.0.0:8502...")
    uvicorn.run(app, host="0.0.0.0", port=8502)
