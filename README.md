# 🛢️ SiloFlow – Grain-Temperature Forecasting Pipeline

SiloFlow is an end-to-end toolkit that ingests raw sensor CSVs from grain warehouses, cleans & enriches the data, trains a machine-learning model, and visualises both historical and forecast temperatures through a Streamlit dashboard.

The project started as *GranaryPredict* and was renamed in May-2025 – imports remain backward-compatible via the `siloflow` ↔︎ `granarypredict` alias.

---

## Key Capabilities

1. **Data ingestion** – parse the StorePoint CSV export or any file mapped to the canonical schema.
2. **Cleaning & imputation** – handle duplicates, obvious sentinels (‐999/“NA”), forward/back-fill categorical gaps, statistical fill for numerics.
3. **Feature engineering**
   • cyclic calendar features (month/hour sin & cos)  
   • 1-, 7-, 30-day sensor lag + ∆T  
   • rolling mean/std windows  
   • auto-label-encoding for categoricals
4. **Model zoo** – Random Forest, HistGradientBoost, tuned LightGBM (+ optional Multi-Output wrapper).  
   Hyper-parameters can be overridden at run-time or via `granarypredict.model.train_*` helpers.
5. **Evaluation suite** – per-horizon MAE/RMSE, overall confidence & accuracy, plus a brand-new *Extremes* tab that spot-lights biggest over/under predictions and average daily error.
6. **Forecasting** – one-click generation of t+1…t+3-day predictions; optionally *future-safe* (excludes environment-only columns).
7. **Production REST API** – `/ingest` CSV-upload endpoint and `/forecast` JSON endpoint built with FastAPI & APScheduler; supports automatic weekly retraining and multi-horizon output.
8. **Cascaded selectors** – Warehouse → Silo filters propagate across all plots & tables.
9. **Synthetic data generator** – create reproducible demo datasets for quick experimentation.

---

## Directory Overview
```
├─ app/                  # Streamlit dashboard (run this!)
│   ├─ Dashboard.py
│   └─ debug_future.py   # optional CLI reproduction of the dashboard logic
│
├─ data/
│   ├─ preloaded/        # sample CSVs shipped with the repo
│   ├─ raw/              # untouched dumps (and by_silo/ organiser output)
│   └─ processed/        # cleaned feature tables (optional)
│
├─ granarypredict/       # Core Python package
│   ├─ ingestion.py      # file / API loaders + schema mapping
│   ├─ cleaning.py       # data cleansing utilities
│   ├─ features.py       # feature engineering helpers
│   ├─ model.py          # training / persistence / inference
│   ├─ evaluate.py       # cross-validation helpers
│   └─ ...
│
├─ service/              # FastAPI micro-service (ingest + forecast endpoints)
├─ models/               # saved .joblib models (auto-created)
├─ scripts/              # CLI helpers (trainer, synthetic data) 
└─ README.md             # ← you are here
```

---

## Quick-Start (Windows / macOS / Linux)
```bash
# 1. Clone & enter repo
$ git clone https://github.com/kaiyaunli7/siloflow.git
$ cd siloflow

# 2. Create & activate a Python 3.11 virtual environment
$ python -m venv .venv
$ source .venv/bin/activate      # Windows: .venv\Scripts\activate.bat

# 3. Install requirements (use a mirror if behind a firewall)
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install -e .               # editable install for granarypredict/

# 5. Launch dashboard
$ streamlit run app/Dashboard.py

# 6 (optional) Run REST API server (production use)
$ uvicorn service.server:app --reload  # visit http://localhost:8000/docs
```

Open http://localhost:8501 and explore:
1. 📂 **Data** – upload a CSV or pick a bundled sample.  Multi-silo files are auto-organised.
2. 🏗️ **Train / Retrain** – choose algorithm, iterations, and **split mode**:
   • *Percentage* (e.g. 80 / 20)  
   • *Last 30 days* (train on history, validate on the most recent month)
3. 🔍 **Evaluate Model** – get per-horizon metrics, 3-D grid, time-series, and the **Extremes** analysis:
   • average daily |error|  
   • worst over-prediction row per day  
   • worst under-prediction row per day  
   • plots of the above
4. 🔮 **Forecast** – extend predictions into the future and compare max/min hot-spots.

---

## Canonical CSV Schema
Column | Type | Notes
-------|------|------
`detection_time` | datetime | timestamp of sensor reading
`granary_id` | str | warehouse name (中文 allowed)
`heap_id` | str | silo / heap identifier
`grid_x, grid_y, grid_z` | int | 3-D probe location
`temperature_grain` | float | °C at probe
`temperature_inside/outside` | float | env temps (optional for future-safe models)
`humidity_warehouse/outside` | float | %RH (optional)
`avg_grain_temp` | float | daily average (optional)

The `granarypredict.ingestion.standardize_granary_csv()` helper converts StorePoint/Result-147 headers automatically.

---

## Command-Line Utilities

• **Synthetic data**  
```bash
python scripts/generate_fake_sensor_data.py --days 30 --grid 4 5 3 \
       --output data/raw/synthetic_sensor.csv
```

• **Global model trainer** (multi-file)  
```bash
python scripts/train_global_model.py data/raw/*.csv --algo lgbm --n-estimators 1200 --future-safe
```

• **Data organiser** – split a mixed CSV into `data/raw/by_silo/<granary>/<heap>/<date>.csv`
```bash
python -m granarypredict.data_organizer mixed.csv
```

---

## Extending / Customising

| What you want | Where to look |
|---------------|--------------|
Tweak hyper-parameters | `granarypredict/model.py` / dashboard training sidebar |
Add new features | `granarypredict/features.py` |
Change alert thresholds | `granarypredict/config.py` |
Integrate REST weather API | `granarypredict/ingestion.py` |

---

## Contributing
Pull requests are welcome!  Please run `flake8` & `black`, and test the dashboard locally before submitting.  For significant changes, update this README accordingly.

---

© 2025 Kaiyuan Li – MIT License

---

## REST API Endpoints

Once the service is running (see step 6 above) interactive Swagger docs are available at `/docs`.  The primary endpoints are:

| Method | Path | Purpose |
|--------|------|---------|
| POST   | `/ingest`   | Append a daily CSV dump to the historical store. |
| POST   | `/forecast` | Return 1-, 2-, 3-day grain-temperature predictions for the given granary as JSON (and persists a CSV under `data/forecast/`). |
| GET    | `/healthz`  | Simple liveness probe used by Docker/K8s. |

The service schedules a weekly retraining job (cron style) which can be adjusted in `service/schedule_config.json`.

---