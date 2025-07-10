Below is a “walk-through” of the full life-cycle in today’s codebase, plus a checklist of the **minimum files** you’d need if you wanted a stripped-down, headless pipeline 

1. reads a CSV
2. preprocesses / feature-engineers it
3. trains (with early-stopping)
4. produces a multi-day forecast

---

A. End-to-end flow in the Streamlit app

---

1. Upload

- Dashboard.py → load_uploaded_file()
- reads the CSV with pandas
- normalises the header via granarypredict.ingestion.standardize_granary_csv()
- drops it into the “Raw Data” expander.
1. Full preprocessing (once per rerun)
- _get_preprocessed_df() → _preprocess_df()
- cleaning.basic_clean() (duplicates, sentinel values)
- calendar gap injection → interpolation → cleaning.fill_missing()
- features.create_time_features() and create_spatial_features()
- lags / rolling stats later added just before training or forecasting.
- Result is cached in st.session_state["processed_df"].
1. Train / split logic
- Sidebar form captures algorithm, % split, etc.
- When “Train” is pressed:
1. features.add_multi_horizon_targets() ensures h+1/2/3 labels.
2. comprehensive_sort() + assign_group_id() stable ordering.
3. Split mode
- 80/20 or user % → split_train_eval_frac()
- “Last 30 days” → split_train_last_n_days()
- 100 % → ➜ internal 90 / 10 LightGBM probe split.
1. Model selection
- RF → RandomForestRegressor, wrapped in MultiOutputRegressor
- HGB → HistGradientBoostingRegressor, wrapped
- **LGBM → MultiLGBMRegressor** (upper-bound = 2000 trees)
- external val-set ⇒ callback-based early-stopping
- 100 % path ⇒ probe-split, find best_iter, refit on full data.
1. Audit toasts (_d()) show every step.
2. Model persisted:

models/<stem>_[fs_]lgbm_<best_iter>.joblib

1. Evaluation
- “Evaluate” button reloads selected model(s).
- Aligns feature matrices via get_feature_cols() (uses .feature_names_in_).
- Computes MAE/RMSE per horizon, confidence %, plots, feature-importance.
- Stores base data + categories in st.session_state["evaluations"].
1. Forecast generation
- generate_and_store_forecast(model_name, horizon)
- For wrappers (MultiOutputRegressor *or* MultiLGBMRegressor) and horizon ≤ 3, uses **direct** design-matrix snapshot → one predict call.
- Otherwise falls back to recursive day-by-day make_future() loop.
- Adds lags / rolling stats each day, applies category codes.
- Persists future_df and X_future for debug.
- Tab “Predictions” shows 3-D map; “Time Series” merges history + future.

---

B. Minimal headless pipeline

---

Keep only the pieces that touch the four stages; nothing UI-specific.

1. **Data ingestion & schema**
- granarypredict/ingestion.py
- granarypredict/cleaning.py
1. **Feature engineering**
- granarypredict/features.py
1. **Model helpers**
- granarypredict/multi_lgbm.py (early-stopping wrapper)
- granarypredict/model.py (save / load helpers)
1. **Utility**
- granarypredict/data_utils.py (comprehensive_sort, etc.)

1. **Driver script (example)**

1. 1. **Driver script (example)**

python

Apply to model.py

import pandas as pd

from pathlib import Path

from granarypredict import ingestion, cleaning, features, model as m

from granarypredict.multi_lgbm import MultiLGBMRegressor

CSV = "data/raw/my_history.csv"

HORIZONS = (1, 2, 3)

*# 1. Read & standardise*

df = pd.read_csv(CSV, encoding="utf-8")

df = ingestion.standardize_granary_csv(df)

*# 2. Pre-clean*

df = cleaning.basic_clean(df)

df = features.create_time_features(df)

df = features.create_spatial_features(df)

df = features.add_multi_horizon_targets(df, horizons=HORIZONS)

*# 3. Chronological 90/10 probe split*

df_train, df_val = m.split_train_eval_frac(df, test_frac=0.1)

X_tr, y_tr = features.select_feature_target_multi(df_train, "temperature_grain", HORIZONS)

X_val, y_val = features.select_feature_target_multi(df_val, "temperature_grain", HORIZONS)

probe = MultiLGBMRegressor(upper_bound_estimators=2000, early_stopping_rounds=100)

probe.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

best_n = probe.best_iteration_

*# 4. Re-fit on full history*

X_all, y_all = features.select_feature_target_multi(df, "temperature_grain", HORIZONS)

final = MultiLGBMRegressor(base_params={"n_estimators": best_n}, upper_bound_estimators=best_n, early_stopping_rounds=0)

final.fit(X_all, y_all)

m.save_model(final, name=f"lgbm_{best_n}.joblib")

*# 5. Forecast next 3 days (one call)*

last_rows = df.sort_values("detection_time").groupby(["grid_x","grid_y","grid_z"]).tail(1)

X_snap, _ = features.select_feature_target_multi(last_rows, "temperature_grain", HORIZONS)

X_snap = X_snap.reindex(columns=final.feature_names_in_, fill_value=0)

preds = final.predict(X_snap)           *# shape (n_sensors, 3)*

With just those files (+ requirements.txt) you can perform the

CSV → train → forecast pipeline on the CLI or in another application.

---

C. Where the audit logs fire

---

- [DATA] – after load & after preprocess
- [TRAIN] – split summary, LightGBM init, wrapper fit complete
- [SPLIT] – last-30, fraction, or 100 % path details
- [MODEL] – algorithm-specific instantiation info
- [TRAIN] – external early-stopping best_iter, internal probe best_n, refit done
- [SAVE] – before/after persisting model
- [EVAL] – prediction array shape during evaluation loop
- [FORECAST] – start + stored rows count

Turn on **Verbose debug mode** in the sidebar to see every breadcrumb live, or

inspect st.session_state["debug_msgs"] after a run.