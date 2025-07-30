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

        # Error prevention: check if last 7 days (including today) are present for each grid BEFORE filling
        try:
            required_cols = {'grid_x', 'grid_y', 'grid_z', 'detection_time'}
            if required_cols.issubset(df.columns):
                df['detection_time'] = pd.to_datetime(df['detection_time'])
                last_7_days = pd.date_range(end=pd.to_datetime(end_date), periods=7, freq='D')
                missing_grids = []
                for grid, group in df.groupby(['grid_x', 'grid_y', 'grid_z']):
                    group_dates = set(group['detection_time'])
                    missing = [d for d in last_7_days if d not in group_dates]
                    if missing:
                        missing_grids.append({'grid': grid, 'missing_dates': [d.strftime('%Y-%m-%d') for d in missing]})
                if missing_grids:
                    logger.error(f"Insufficient data for forecasting. Missing last 7 days for grids: {missing_grids}")
                    raise HTTPException(status_code=400, detail=f"Insufficient data for forecasting. Missing last 7 days for grids: {missing_grids}")
            else:
                logger.warning("DataFrame missing required columns for grid/date filling. Skipping last 7 day check.")
        except Exception as e:
            logger.error(f"Error during last 7 day missing check: {e}")
            # Continue with original df if check fails
        # Fill missing dates for each grid (xyz) by replicating nearest available row
        try:
            import numpy as np
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            required_cols = {'grid_x', 'grid_y', 'grid_z', 'detection_time'}
            if required_cols.issubset(df.columns):
                df['detection_time'] = pd.to_datetime(df['detection_time'])
                filled_rows = []
                for grid, group in df.groupby(['grid_x', 'grid_y', 'grid_z']):
                    group_dates = set(group['detection_time'])
                    for date in all_dates:
                        if date not in group_dates:
                            # Find nearest available date in group
                            nearest = min(group_dates, key=lambda d: abs((d - date).days)) if group_dates else None
                            if nearest is not None:
                                nearest_row = group[group['detection_time'] == nearest].iloc[0].copy()
                                nearest_row['detection_time'] = date
                                filled_rows.append(nearest_row)
                if filled_rows:
                    df = pd.concat([df, pd.DataFrame(filled_rows)], ignore_index=True)
                    logger.info(f"Filled {len(filled_rows)} missing date rows by nearest grid replication.")
            else:
                logger.warning("DataFrame missing required columns for grid/date filling. Skipping fill.")
            # Error prevention: check if last 7 days (including today) are present for each grid
            last_7_days = pd.date_range(end=pd.to_datetime(end_date), periods=7, freq='D')
            missing_grids = []
            for grid, group in df.groupby(['grid_x', 'grid_y', 'grid_z']):
                group_dates = set(group['detection_time'])
                missing = [d for d in last_7_days if d not in group_dates]
                if missing:
                    missing_grids.append({'grid': grid, 'missing_dates': [d.strftime('%Y-%m-%d') for d in missing]})
            if missing_grids:
                logger.error(f"Insufficient data for forecasting. Missing last 7 days for grids: {missing_grids}")
                raise HTTPException(status_code=400, detail=f"Insufficient data for forecasting. Missing last 7 days for grids: {missing_grids}")
        except Exception as e:
            logger.error(f"Error during missing date filling: {e}")
            # Continue with original df if filling fails
        # Preprocess and forecast
        try:
            processed_df = Dashboard._preprocess_df(df)
            logger.info(f"Preprocessing complete. Processed shape: {processed_df.shape}")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

        try:
            model = Dashboard.load_trained_model(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

        try:
            features = Dashboard.get_feature_cols(model, processed_df)
            if not features:
                logger.error("No feature columns found for prediction.")
                raise HTTPException(status_code=500, detail="No feature columns found for prediction.")
            logger.info(f"Selected features: {features}")
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Feature selection failed: {e}")

        # Use the new FastAPI-compatible forecast function
        # evaluations dict for API context (simulate session_state)
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
        # Only include columns that would be stored in the Parquet file
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
