from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from granarypredict import model as model_utils
from service import pipeline

app = FastAPI(title="GranaryPredict Service", version="1.0.0")
logger = logging.getLogger("service.server")


class ForecastRequest(BaseModel):
    granary_id: str
    # base64-encoded CSV or raw rows could also be used; we keep simple JSON list
    rows: List[Dict]


@app.post("/ingest", summary="Append daily sensor data")
async def ingest_daily(file: UploadFile = File(...)):
    """Accept a CSV upload (one granary) and append it to history CSV."""
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")

    df = pipeline.ingestion.standardize_granary_csv(df)  # type: ignore
    csv_path = pipeline.append_daily(df)
    return {"status": "ok", "csv": str(csv_path)}


@app.post("/forecast", summary="Return 3-day forecast for provided granary")
async def forecast(req: ForecastRequest):
    df_new = pd.DataFrame(req.rows)
    if df_new.empty:
        raise HTTPException(status_code=400, detail="No rows supplied")

    df_new = pipeline.ingestion.standardize_granary_csv(df_new)  # type: ignore
    csv_path = pipeline.append_daily(df_new)

    # load model or retrain quickly if missing
    model_path = pipeline.MODELS_DIR / f"{req.granary_id}_fs_lgbm.joblib"
    if not model_path.exists():
        full_df = pd.read_csv(csv_path, parse_dates=["detection_time"])
        full_df = pipeline.preprocess(full_df)
        model_path = pipeline.train_model(full_df, granary=req.granary_id)

    mdl = model_utils.load_model(model_path)

    # Build future template using *all history* (faster than computing again)
    full_df = pd.read_csv(csv_path, parse_dates=["detection_time"])
    full_df = pipeline.preprocess(full_df, future_safe=True)
    fut_df = pipeline.build_future_template(full_df)
    fut_df = pipeline.forecast(mdl, fut_df)

    out_path = pipeline.save_forecast(req.granary_id, fut_df)

    # Return only necessary forecast columns as JSON
    cols = [
        "granary_id",
        "heap_id",
        "grid_x",
        "grid_y",
        "grid_z",
        "detection_time",
        "forecast_day",
        "pred_h1d",
        "pred_h2d",
        "pred_h3d",
    ]
    return JSONResponse(
        content={"forecast": fut_df[cols].to_dict(orient="records"), "file": str(out_path)}
    )


# ---------------- Scheduled Training ----------------
from apscheduler.schedulers.background import BackgroundScheduler
import datetime as dt
import json

scheduler = BackgroundScheduler()


def weekly_training_job():
    logger.info("Starting weekly retraining for all granaries …")
    for csv_path in pipeline.RAW_DIR.glob("*.csv"):
        granary = csv_path.stem
        full_df = pd.read_csv(csv_path, parse_dates=["detection_time"])
        full_df = pipeline.preprocess(full_df, future_safe=True)
        pipeline.train_model(full_df, granary=granary)
    logger.info("Weekly retraining completed.")


# ---------------- Load schedule config ----------------
CONFIG_PATH = Path(__file__).with_name("schedule_config.json")
DEFAULT_SCHEDULE = {"day_of_week": "sun", "hour": 2, "minute": 0}

try:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            schedule_cfg = json.load(fh)
            logger.info("Loaded training schedule from %s: %s", CONFIG_PATH.name, schedule_cfg)
    else:
        schedule_cfg = DEFAULT_SCHEDULE
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(schedule_cfg, fh, indent=2)
except Exception as exc:
    logger.warning("Failed to parse schedule config – falling back to default: %s", exc)
    schedule_cfg = DEFAULT_SCHEDULE

# Register job with dynamic cron parameters
scheduler.add_job(weekly_training_job, "cron", id="weekly_train", **schedule_cfg)
scheduler.start()


@app.get("/healthz", include_in_schema=False)
async def health() -> Dict[str, str]:
    return {"status": "ok", "time": dt.datetime.utcnow().isoformat()} 