from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import tempfile
from pathlib import Path

from . import ingestion, tasks

app = FastAPI(title="Granary ML Service")

MAX_UPLOAD_MB = 10

# ------------------------------------------------------------------
# Ingest endpoint
# ------------------------------------------------------------------

@app.post("/ingest")
def ingest_csv(file: UploadFile = File(...)):
    if file.content_type not in {"text/csv", "application/vnd.ms-excel"}:
        raise HTTPException(status_code=400, detail="File must be CSV")
    size_mb = file.size / (1024 * 1024) if file.size else 0
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail="File too large")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file.file.read())
        tmp_path = Path(tmp.name)

    df = pd.read_csv(tmp_path, encoding="utf-8")
    n_train, n_forec = ingestion.append_rows(df)
    tmp_path.unlink(missing_ok=True)
    return {"status": "ok", "train_rows": n_train, "forecast_rows": n_forec}

# ------------------------------------------------------------------
# Train model for a granary
# ------------------------------------------------------------------

@app.post("/train/{granary_id}")
def train_model(granary_id: str):
    try:
        metrics, model_path = tasks.train_granary(granary_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "ok", "metrics": metrics, "model_path": str(model_path)}

# ------------------------------------------------------------------
# Forecast for a heap
# ------------------------------------------------------------------

@app.post("/forecast/{granary_id}/{heap_id}")
def forecast_heap(granary_id: str, heap_id: str):
    try:
        csv_path = tasks.forecast_heap(granary_id, heap_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(csv_path, media_type="text/csv", filename=Path(csv_path).name) 