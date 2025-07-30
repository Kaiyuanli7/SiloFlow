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
        # Return the 7-day forecast data directly
        future_df = forecast_result["future_df"]
        if future_df is None or future_df.empty:
            return {"success": True, "forecasts": []}
        # Convert DataFrame to list of dicts (records)
        forecasts = future_df.to_dict(orient="records")
        return {
            "success": True,
            "forecasts": forecasts,
            "rows": len(forecasts)
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
