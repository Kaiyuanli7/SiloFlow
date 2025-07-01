import pathlib
from datetime import timedelta, datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from itertools import islice

from granarypredict import cleaning, features, model as model_utils
from granarypredict.config import ALERT_TEMP_THRESHOLD, MODELS_DIR
from granarypredict import ingestion
from granarypredict.data_utils import comprehensive_sort, assign_group_id
# from granarypredict.data_organizer import organize_mixed_csv  # deprecated
from granarypredict.multi_lgbm import MultiLGBMRegressor  # NEW

# Streamlit reload may have stale module; fetch grain thresholds safely
try:
    from granarypredict.config import GRAIN_ALERT_THRESHOLDS  # type: ignore
except ImportError:
    GRAIN_ALERT_THRESHOLDS = {}

from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor  # NEW

# ---------------------------------------------------------------------
# 🈯️  Simple i18n helper  (EN / 中文)  -----------------------------------
# ---------------------------------------------------------------------


_TRANSLATIONS_ZH: dict[str, str] = {
    # Sidebar & section titles
    "Data": "数据",
    "Train / Retrain Model": "训练 / 重新训练模型",
    "Training split mode": "训练拆分模式",
    "Percentage": "百分比",
    "Last 30 days": "最近 30 天",
    "Train split (%)": "训练集比例 (%)",
    "Algorithm": "算法",
    "Iterations / Trees": "迭代 / 树数",
    "Future-safe (exclude env vars)": "未来安全（排除环境变量）",
    "Evaluate Model": "评估模型",
    "Select evaluated model": "选择已评估模型",
    "Generate Forecast": "生成预测",
    # Tab labels
    "Summary": "摘要",
    "Predictions": "预测明细",
    "3D Grid": "三维网格",
    "Time Series": "时间序列",
    "Extremes": "极值",
    "Debug": "调试",
    "Evaluation": "评估",
    "Forecast": "预测",
    # Extremes plots
    "Average Daily Absolute Error (h+1)": "每日平均绝对误差 (h+1)",
    "Over-Prediction (h+1)": "过预测 (h+1)",
    "Under-Prediction (h+1)": "欠预测 (h+1)",
    # Misc
    "Select date": "选择日期",
    # 🔽 NEW translations
    "Verbose debug mode": "详细调试模式",
    "Upload your own CSV": "上传您的 CSV",
    "Or pick a bundled sample dataset:": "或选择一个捆绑示例数据集：",
    "Sample dataset": "示例数据集",
    "Raw Data": "原始数据",
    "Sorted Data": "已排序数据",
    "Location Filter": "位置筛选器",
    "Warehouse": "仓库",
    "Silo": "筒仓",
    "Train on uploaded CSV": "使用上传的 CSV 进行训练",
    "Model file": "模型文件",
    "Apply to all models": "应用到所有模型",
    "Evaluate": "评估",
    "Eval & Forecast": "评估并预测",
    "No forecast generated yet for this model.": "该模型尚未生成预测。",
    "Uploaded file appears empty or unreadable. Please verify the CSV.": "上传的文件为空或无法读取。请检查 CSV。",
    "Model not found – please train or select another.": "未找到模型 – 请训练或选择其他模型。",
    "No spatial temperature data present.": "没有空间温度数据。",
    "Detected mixed dataset – organising into per-silo files…": "检测到混合数据集 – 正在按筒仓整理文件…",
    "Training model – please wait...": "正在训练模型 – 请稍候...",
    "No saved models yet.": "尚未保存任何模型。",
    "Please upload a CSV first to evaluate.": "请先上传 CSV 以进行评估。",
    "Evaluating model(s) – please wait...": "正在评估模型 – 请稍候...",
    "Generating forecast…": "正在生成预测…",
    "Forecast generated – switch tabs to view.": "预测已生成 – 切换选项卡查看。",
    "Model Leaderboard": "模型排行榜",
    "No evaluations yet.": "尚无评估结果。",
    "Debug Log (full)": "调试日志（完整）",
    "Forecast Summary (per day)": "预测摘要（每日）",
    "Top Predictive Features": "最具预测力的特征",
    "Daily Extremes (h+1)": "每日极值 (h+1)",
    "No horizon-1 predictions available to compute extremes.": "没有可用于计算极值的 h+1 预测。",
    "Feature Matrices (first 100 rows)": "特征矩阵（前 100 行）",
    "Training – X_train": "训练 – X_train",
    "Evaluation – X_eval": "评估 – X_eval",
    "Model Feature Columns (order)": "模型特征列（顺序）",
    "No forecast generated for this model yet.": "尚未为该模型生成预测。",
    "Forecast Summary (predicted)": "预测摘要（预测）",
    "Daily Predicted Extremes": "每日预测极值",
    "No predictions found to compute extremes.": "未找到用于计算极值的预测。",
    "Future Feature Matrix (first 100 rows)": "未来特征矩阵（前 100 行）",
    "|Mean(X_eval) − Mean(X_future)| (Top 20)": "|Mean(X_eval) − Mean(X_future)|（前 20）",
    "X_future matrix not available yet.": "X_future 矩阵尚不可用。",
    "Please evaluate the model first.": "请先评估模型。",
    "Unable to access base data or model for forecasting.": "无法访问基础数据或模型进行预测。",
    "High temperature forecast detected for at least one grain type – monitor closely!": "检测到某些粮食类型的高温预测 – 请密切监控！",
    "All predicted temperatures within safe limits for their grain types": "所有预测温度均在其粮食类型的安全范围内",
    "LightGBM uses early stopping; optimal number of trees will be selected automatically.": "LightGBM 使用提前停止；将自动选择最佳树数量。",
    "Conf (%)": "置信度 (%)",
    "Acc (%)": "准确率 (%)",
    "MAE h+1": "MAE h+1",
    "MAE h+2": "MAE h+2",
    "MAE h+3": "MAE h+3",
    "RMSE h+1": "RMSE h+1",
    "Confidence (%)": "置信度 (%)",
    "Accuracy (%)": "准确率 (%)",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "MAPE (%)": "MAPE (%)",
}


def _t(msg: str) -> str:
    """Translate *msg* to Chinese if the user selected that language."""
    lang = st.session_state.get("lang", "en")
    if lang == "zh":
        return _TRANSLATIONS_ZH.get(msg, msg)
    return msg

# ---------------------------------------------------------------------
# Debug helper – collects messages in session state
# ---------------------------------------------------------------------

# Defined early so it's available everywhere

def _d(msg):
    if not st.session_state.get("debug_mode"):
        return
    import streamlit as _st
    _st.toast(str(msg))
    log = _st.session_state.setdefault("debug_msgs", [])
    log.append(str(msg))

# Add constant after imports
ENV_COLUMNS = [
    "temperature_inside",
    "temperature_outside",
    "humidity_warehouse",
    "humidity_outside",
]

# Additional columns unavailable for real future dates that should be
# removed when training a "future-safe" model.
FUTURE_SAFE_EXTRA = [
    "max_temp",      # historic max air temp inside silo
    "min_temp",      # historic min inside temp
    "line_no",       # production line identifier (constant, but 0-filled in future)
    "layer_no",      # vertical layer identifier (constant, but 0-filled in future)
]

# Preset horizons (days) for quick selector controls
PRESET_HORIZONS = [7, 14, 30, 90, 180, 365]

# -----------------------------------------------------------------------------
# 🔧 GLOBAL FORECAST HORIZON (days)
# -----------------------------------------------------------------------------
# Change **one** number here to adjust how many days ahead the model should
# learn and predict throughout the entire dashboard.  All downstream helper
# functions reference HORIZON_TUPLE instead of hard-coding (1, 2, 3).

HORIZON_DAYS: int = 7
# Tuple (1, 2, …, HORIZON_DAYS)
HORIZON_TUPLE: tuple[int, ...] = tuple(range(1, HORIZON_DAYS + 1))

# Target column representing daily average grain temperature for evaluation/forecast
TARGET_TEMP_COL = "temperature_grain"  # per-sensor target for model & metrics

# Utility to detect if a model is "future-safe" by filename convention (contains 'fs_')
def is_future_safe_model(model_name: str) -> bool:
    return "fs_" in model_name.lower()

st.set_page_config(page_title="SiloFlow", layout="wide")



# Directory that holds bundled sample CSVs shipped with the repo
PRELOADED_DATA_DIR = pathlib.Path("data/preloaded")

# Directory that holds bundled pre-trained models
PRELOADED_MODEL_DIR = MODELS_DIR / "preloaded"

# Ensure directory exists so globbing is safe
PRELOADED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV safely, rewinding pointer and handling encoding."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        # Convert to internal schema regardless of source CSV variant.
        df = ingestion.standardize_granary_csv(df)
        return df
    except pd.errors.EmptyDataError:
        st.error(_t("Uploaded file appears empty or unreadable. Please verify the CSV."))
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_trained_model(path: Optional[str | pathlib.Path] = None):
    """Attempt to load a model from user-saved or preloaded directories."""
    # Default fallback
    if path is None:
        path = MODELS_DIR / "rf_model.joblib"

    path = pathlib.Path(path)

    # If given just a filename, search in both dirs
    if not path.is_absolute() and not path.exists():
        user_path = MODELS_DIR / path
        preload_path = PRELOADED_MODEL_DIR / path
        if user_path.exists():
            path = user_path
        elif preload_path.exists():
            path = preload_path

    if path.exists():
        return model_utils.load_model(path)

    st.warning(_t("Model not found – please train or select another."))
    return None


def plot_3d_grid(df: pd.DataFrame, *, key: str, color_by_delta: bool = False):
    required_cols = {"grid_x", "grid_y", "grid_z", "temperature_grain"}
    if not required_cols.issubset(df.columns):
        st.info(_t("No spatial temperature data present."))
        return

    # Build point hover/label text
    texts = []
    has_pred = "predicted_temp" in df.columns
    for _, row in df.iterrows():
        if color_by_delta and has_pred:
            diff = row["predicted_temp"] - row["temperature_grain"]
            texts.append(
                f"Pred: {row['predicted_temp']:.1f}°C<br>Actual: {row['temperature_grain']:.1f}°C<br>Δ: {diff:+.1f}°C"
            )
        elif has_pred:
            diff = row["predicted_temp"] - row["temperature_grain"]
            texts.append(
                f"Pred: {row['predicted_temp']:.1f}°C<br>Actual: {row['temperature_grain']:.1f}°C<br>Δ: {diff:+.1f}°C"
            )
        else:
            texts.append(f"Actual: {row['temperature_grain']:.1f}°C")

    if color_by_delta and has_pred:
        color_vals = df["predicted_temp"] - df["temperature_grain"]
        c_scale = "RdBu_r"
        cbar_title = "Δ (°C)"
    else:
        color_vals = df["predicted_temp"] if has_pred else df["temperature_grain"]
        c_scale = "RdBu_r"  # red = hot, blue = cold
        cbar_title = "Temp (°C)" if not color_by_delta else "Pred (°C)"

    fig = go.Figure(data=go.Scatter3d(
        x=df["grid_x"],
        y=df["grid_z"],
        z=df["grid_y"],
        mode="markers",
        marker=dict(
            size=6,
            color=color_vals,
            colorscale=c_scale,
            colorbar=dict(title=cbar_title),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    ))
    # Ensure integer ticks on axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(dtick=1, title="grid_x"),
            yaxis=dict(dtick=1, title="grid_z"),
            zaxis=dict(dtick=1, title="grid_y", autorange="reversed"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_time_series(df: pd.DataFrame, *, key: str):
    if "predicted_temp" not in df.columns or "detection_time" not in df.columns:
        return
    tmp = df.copy()
    # Use floor("D") to keep datetime64 dtype; avoids Plotly treating axis as categorical
    tmp["date"] = pd.to_datetime(tmp["detection_time"]).dt.floor("D")

    fig = go.Figure()

    # -------- Actual line --------
    grp_actual = tmp.groupby("date").agg(actual=(TARGET_TEMP_COL, "mean")).reset_index()
    fig.add_trace(
        go.Scatter(
            x=grp_actual["date"],
            y=grp_actual["actual"],
            mode="lines+markers",
            name="Actual Avg",
            line=dict(color="#1f77b4"),
        )
    )

    # ------- Predicted line (continuous across eval & filled) -------
    if "predicted_temp" in tmp.columns:
        grp_pred = (
            tmp.groupby("date").agg(predicted=("predicted_temp", "mean")).reset_index().sort_values("date")
        )

        # Determine cutoff between evaluation (has actual data) and future-only predictions
        actual_mask = tmp[TARGET_TEMP_COL].notna()
        last_actual_date = tmp.loc[actual_mask, "date"].max()

        if pd.isna(last_actual_date):
            pred_eval = pd.DataFrame()
            pred_future = grp_pred
        else:
            pred_eval = grp_pred[grp_pred["date"] <= last_actual_date]
            pred_future = grp_pred[grp_pred["date"] > last_actual_date]

        if not pred_eval.empty:
            fig.add_trace(
                go.Scatter(
                    x=pred_eval["date"],
                    y=pred_eval["predicted"],
                    mode="lines+markers",
                    name="Predicted (eval)",
                    line=dict(color="#ff7f0e"),
                    connectgaps=True,
                )
            )

        if not pred_future.empty:
            fig.add_trace(
                go.Scatter(
                    x=pred_future["date"],
                    y=pred_future["predicted"],
                    mode="lines+markers",
                    name="Predicted (future)",
                    line=dict(color="#9467bd"),
                    connectgaps=True,
                )
            )

    fig.update_layout(
        title="Average Grain Temperature Over Time",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def list_available_models() -> list[str]:
    """Return unique model filenames from user-saved and preloaded dirs."""
    names = {p.name for p in MODELS_DIR.glob("*.joblib")}
    names.update({p.name for p in PRELOADED_MODEL_DIR.glob("*.joblib")})
    return sorted(names)


def split_train_eval(df: pd.DataFrame, horizon: int = 5):
    """Split by unique date; last 'horizon' dates form evaluation set."""
    df = df.copy()
    df["_date"] = pd.to_datetime(df["detection_time"]).dt.date
    unique_dates = sorted(df["_date"].unique())
    if len(unique_dates) <= horizon + 1:
        return df, pd.DataFrame()  # not enough data
    cutoff_dates = unique_dates[-horizon:]
    df_eval = df[df["_date"].isin(cutoff_dates)].copy()
    df_train = df[~df["_date"].isin(cutoff_dates)].copy()
    # map forecast_day index 1..horizon
    date_to_idx = {date: idx for idx, date in enumerate(cutoff_dates, start=1)}
    df_eval["forecast_day"] = df_eval["_date"].map(date_to_idx)
    df_train.drop(columns=["_date"], inplace=True)
    df_eval.drop(columns=["_date"], inplace=True)
    return df_train, df_eval


# -------------------------------------------------------------------
# NEW – fraction‐based chronological split (May-2025)
# -------------------------------------------------------------------


def split_train_eval_frac(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronologically split *df* by unique date where the **last** fraction
    (``test_frac``) of dates becomes the evaluation set.

    Returns (df_train, df_eval) similar to ``split_train_eval`` but sized by
    proportion instead of fixed horizon.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["detection_time"]).dt.date
    unique_dates = sorted(df["_date"].unique())
    if not unique_dates:
        return df, pd.DataFrame()

    n_test_days = max(1, int(len(unique_dates) * test_frac))
    cutoff_dates = unique_dates[-n_test_days:]

    df_eval = df[df["_date"].isin(cutoff_dates)].copy()
    df_train = df[~df["_date"].isin(cutoff_dates)].copy()

    # map forecast_day index 1..n_test_days
    date_to_idx = {date: idx for idx, date in enumerate(cutoff_dates, start=1)}
    df_eval["forecast_day"] = df_eval["_date"].map(date_to_idx)

    df_train.drop(columns=["_date"], inplace=True)
    df_eval.drop(columns=["_date"], inplace=True)
    return df_train, df_eval


def forecast_summary(df_eval: pd.DataFrame) -> pd.DataFrame:
    if "forecast_day" not in df_eval.columns:
        return pd.DataFrame()

    grp = (
        df_eval.groupby("forecast_day")
        .agg(
            actual_mean=(TARGET_TEMP_COL, "mean"),
            pred_mean=("predicted_temp", "mean"),
        )
        .reset_index()
    )

    # Map numeric forecast_day → actual calendar date (first occurrence)
    if "detection_time" in df_eval.columns:
        day_to_date = (
            pd.to_datetime(df_eval["detection_time"]).dt.floor("D").groupby(df_eval["forecast_day"]).first()
        )
        grp["date"] = grp["forecast_day"].map(day_to_date)
        # Re-order so date is the first column and drop forecast_day numeric index
        grp = grp[["date", "actual_mean", "pred_mean"]]
        grp = grp.rename(columns={"date": "calendar_date"})
    
    # Percent absolute error (only where actual_mean is finite & non-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        grp["pct_error"] = (grp["pred_mean"] - grp["actual_mean"]).abs() / grp["actual_mean"].replace(0, np.nan) * 100

    # Confidence via R² per day (skip NaNs)
    conf_vals: list[float] = []
    for day in df_eval["forecast_day"].unique():
        subset = df_eval[df_eval["forecast_day"] == day][[TARGET_TEMP_COL, "predicted_temp"]].dropna()
        if len(subset) > 1:
            r2 = r2_score(subset[TARGET_TEMP_COL], subset["predicted_temp"])
            conf_vals.append(max(0, min(100, r2 * 100)))
        else:
            conf_vals.append(np.nan)
    # Align list length with grp after possibly dropping forecast_day col
    grp["confidence_%"] = conf_vals[: len(grp)]

    return grp


def compute_overall_metrics(df_eval: pd.DataFrame) -> tuple[float, float]:
    """Return (confidence %, accuracy %) or (nan, nan) if not computable.
    This helper now drops rows containing NaNs before computing metrics to avoid
    ValueError from scikit-learn when all/any NaNs are present.
    """
    required = {TARGET_TEMP_COL, "predicted_temp"}
    if not required.issubset(df_eval.columns) or df_eval.empty:
        return float("nan"), float("nan")

    valid = df_eval[list(required)].dropna()
    if valid.empty:
        return float("nan"), float("nan")

    r2 = r2_score(valid[TARGET_TEMP_COL], valid["predicted_temp"])
    conf = max(0, min(100, r2 * 100))

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_err = (valid[TARGET_TEMP_COL] - valid["predicted_temp"]).abs() / valid[TARGET_TEMP_COL].replace(0, np.nan)
    avg_pct_err = pct_err.mean(skipna=True) * 100 if not pct_err.empty else float("nan")
    acc = max(0, 100 - avg_pct_err)
    return conf, acc


# --------------------------------------------------
# Helper to build future rows for forecasting
def make_future(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Generate a future dataframe for the next ``horizon_days`` days.

    For each unique spatial location (grid_x/y/z) present in *df*, this function
    creates ``horizon_days`` duplicated rows with the *detection_time* advanced
    by 1..horizon_days. It also appends a *forecast_day* column (1-indexed).

    The resulting frame is passed through the same feature generators so it can
    be fed directly into the model for prediction.
    """
    if df.empty or "detection_time" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"])
    latest_ts = df["detection_time"].max()

    # Keep spatial coords plus any constant categorical IDs (grain_type / warehouse_type)
    keep_cols = [c for c in df.columns if c in {"grid_x", "grid_y", "grid_z", "granary_id", "heap_id"}]
    sensors = df[keep_cols].drop_duplicates().reset_index(drop=True)

    # Add constant metadata if available (assuming single value across file)
    for const_col in ["grain_type", "warehouse_type"]:
        if const_col in df.columns:
            sensors[const_col] = df[const_col].dropna().iloc[0]

    # Prepare base detection_cycle if present
    max_cycle = df["detection_cycle"].max() if "detection_cycle" in df.columns else None

    frames: List[pd.DataFrame] = []
    for d in range(1, horizon_days + 1):
        tmp = sensors.copy()
        tmp["detection_time"] = latest_ts + timedelta(days=d)
        tmp["forecast_day"] = d
        if max_cycle is not None:
            tmp["detection_cycle"] = max_cycle + d
        frames.append(tmp)

    future_df = pd.concat(frames, ignore_index=True)
    # Feature engineering to match training pipeline
    future_df = features.create_time_features(future_df)
    future_df = features.create_spatial_features(future_df)
    future_df = features.add_sensor_lag(future_df)
    # Ensure both legacy and new target columns exist so downstream feature
    # selection works regardless of current configuration.
    if "temperature_grain" not in future_df.columns:
        future_df["temperature_grain"] = np.nan
    if TARGET_TEMP_COL not in future_df.columns:
        future_df[TARGET_TEMP_COL] = np.nan
    return future_df
# --------------------------------------------------


def main():
    st.session_state.setdefault("evaluations", {})
    st.session_state.setdefault("forecasts", {})  # NEW: container for forecast results

    # Debug toggle – placed at very top so early messages are captured
    st.sidebar.checkbox(_t("Verbose debug mode"), key="debug_mode", help="Show detailed internal processing messages", value=True)

    # ---------------- Language selector (appears very top) -------------
    st.sidebar.selectbox(
        "Language / 语言",
        options=["English", "中文"],
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
        key="lang_selector",
        on_change=lambda: st.session_state.update({"lang": "en" if st.session_state.get("lang_selector") == "English" else "zh"}),
    )
    # Ensure lang key present
    st.session_state.setdefault("lang", "en")

    with st.sidebar.expander(_t('Data'), expanded=("uploaded_file" not in st.session_state)):
        uploaded_file = st.file_uploader(_t("Upload your own CSV"), type=["csv"], key="uploader")

        # ------------------------------------------------------------------
        # Offer bundled sample datasets so users can start instantly
        if PRELOADED_DATA_DIR.exists():
            sample_files = sorted(PRELOADED_DATA_DIR.glob("*.csv"))
            if sample_files:
                st.caption(_t("Or pick a bundled sample dataset:"))
                sample_names = ["-- Select sample --"] + [p.name for p in sample_files]
                sample_choice = st.selectbox(
                    _t("Sample dataset"),  # non-empty label for accessibility
                    options=sample_names,
                    key="sample_selector",
                    label_visibility="collapsed",  # hide visually but keep for screen readers
                )
                if sample_choice and sample_choice != "-- Select sample --":
                    uploaded_file = PRELOADED_DATA_DIR / sample_choice  # path object -> pd.read_csv works
                    st.info(f"Sample dataset '{sample_choice}' selected.")

    if uploaded_file:
        df = load_uploaded_file(uploaded_file)
        _d(f"[DATA] Uploaded file loaded – shape={df.shape} cols={list(df.columns)[:10]}…")
        with st.expander(_t("Raw Data"), expanded=False):
            st.dataframe(df, use_container_width=True)

        # ------------------------------------------------------------------
        # Auto-organise if the upload mixes multiple silos  (removed in v1.1)
        # ------------------------------------------------------------------
        # (functionality removed)
        
        # Full preprocessing once
        _d("Running full preprocessing on uploaded dataframe (cached)…")
        df = _get_preprocessed_df(uploaded_file)
        _d(f"[DATA] Preprocessing complete – shape={df.shape} cols={list(df.columns)[:10]}…")

        # Display sorted table directly below Raw Data
        df_sorted_display = df
        with st.expander(_t("Sorted Data"), expanded=False):
            _st_dataframe_safe(df_sorted_display, key="sorted")

        # ------------------------------
        # Global Warehouse → Silo filter
        # ------------------------------

        st.markdown(f"### {_t('Location Filter')}")
        with st.container():
            # Detect possible column names coming from different CSV formats
            wh_col_candidates = [c for c in ["granary_id", "storepointName"] if c in df.columns]
            silo_col_candidates = [c for c in ["heap_id", "storeName"] if c in df.columns]

            wh_col = wh_col_candidates[0] if wh_col_candidates else None
            silo_col = silo_col_candidates[0] if silo_col_candidates else None

            # 1️⃣ Warehouse selector – always shown if the column exists
            if wh_col:
                warehouses = sorted(df[wh_col].dropna().unique())
                warehouses_opt = ["All"] + warehouses
                sel_wh_global = st.selectbox(
                    _t("Warehouse"),
                    options=warehouses_opt,
                    key="global_wh",
                )
            else:
                sel_wh_global = "All"

            # 2️⃣ Silo selector – rendered only after a specific warehouse is chosen
            if wh_col and sel_wh_global != "All" and silo_col:
                silos = sorted(df[df[wh_col] == sel_wh_global][silo_col].dropna().unique())
                silos_opt = ["All"] + silos
                sel_silo_global = st.selectbox(
                    _t("Silo"),
                    options=silos_opt,
                    key="global_silo",
                )
            else:
                sel_silo_global = "All"

            # Persist selection in session state for downstream use
            st.session_state["global_filters"] = {
                "wh": sel_wh_global,
                "silo": sel_silo_global,
            }

        with st.sidebar.expander(_t('Train / Retrain Model'), expanded=False):
            model_choice = st.selectbox(
                _t("Algorithm"),
                ["RandomForest", "HistGradientBoosting", "LightGBM"],
                index=0,
            )
            if model_choice == "LightGBM":
                st.caption(_t("LightGBM uses early stopping; optimal number of trees will be selected automatically."))
                n_trees = 2000  # upper bound (not shown to user)
            else:
                n_trees = st.slider(_t("Iterations / Trees"), 100, 1000, 300, step=100)
            future_safe = st.checkbox(_t("Future-safe (exclude env vars)"), value=True)

            # ---------------- Training split mode -----------------
            split_mode = st.radio(
                _t("Training split mode"),
                [_t("Percentage"), _t("Last 30 days")],
                index=0,
                horizontal=True,
                help="Choose how to divide data into training vs validation sets.",
            )

            if split_mode == _t("Percentage"):
                train_pct = st.slider(
                    _t("Train split (%)"),
                    50,
                    100,
                    80,
                    step=5,
                    help="Percentage of data used for training; set to 100% to train on the whole dataset without a validation split.",
                )
                use_last_30 = False
            else:
                # Fixed 30-day window selected – ignore percentage slider.
                train_pct = None
                use_last_30 = True

            train_pressed = st.button(_t("Train on uploaded CSV"))

        if train_pressed and uploaded_file:
            with st.spinner(_t("Training model – please wait...")):
                # -------- Data preparation --------
                df = _get_preprocessed_df(uploaded_file)

                if "temperature_grain_h1d" not in df.columns:
                    df = features.add_multi_horizon_targets(df, horizons=HORIZON_TUPLE)

                if future_safe:
                    df = df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

                # Ensure multi-horizon target columns present (fast-path CSVs may lack them)
                if "temperature_grain_h1d" not in df.columns:
                    df = features.add_multi_horizon_targets(df, horizons=HORIZON_TUPLE)

                # Consistent sorting & grouping
                df = comprehensive_sort(df)
                df = assign_group_id(df)

                # Feature matrix / target (MULTI-OUTPUT)
                X_all, y_all = features.select_feature_target_multi(
                    df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                )  # NEW

                # -------- Group-aware hold-out with user-defined split --------
                _d(f"[TRAIN] Preparing train/test split – total rows={len(df)} sensors={df[['grid_x','grid_y','grid_z']].drop_duplicates().shape[0] if {'grid_x','grid_y','grid_z'}.issubset(df.columns) else 'N/A'}")

                if use_last_30:
                    df_train_tmp, df_eval_tmp = split_train_last_n_days(df, n_days=30)
                    X_tr, y_tr = features.select_feature_target_multi(
                        df_train_tmp, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                    )
                    X_te, y_te = features.select_feature_target_multi(
                        df_eval_tmp, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                    )
                    perform_validation = not X_te.empty
                    _d(f"[SPLIT] Last-30days mode – train rows={len(df_train_tmp)}, val rows={len(df_eval_tmp)}")
                elif train_pct == 100:
                    X_tr, y_tr = X_all, y_all
                    X_te = y_te = pd.DataFrame()
                    perform_validation = False
                    _d("[SPLIT] 100% training – no explicit validation set")
                else:
                    test_frac_chrono = max(0.05, 1 - train_pct / 100)
                    df_train_tmp, df_eval_tmp = split_train_eval_frac(df, test_frac=test_frac_chrono)
                    X_tr, y_tr = features.select_feature_target_multi(
                        df_train_tmp, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                    )
                    X_te, y_te = features.select_feature_target_multi(
                        df_eval_tmp, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                    )
                    perform_validation = not X_te.empty
                    _d(f"[SPLIT] Fraction mode ({train_pct}% train) – train rows={len(df_train_tmp)}, val rows={len(df_eval_tmp)}")

                # -------- Model selection & training --------
                if model_choice == "RandomForest":
                    base_mdl = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42)
                    suffix = "rf"
                    use_wrapper = True
                    _d(f"[MODEL] RandomForest initialised n_estimators={n_trees}")
                elif model_choice == "HistGradientBoosting":
                    base_mdl = HistGradientBoostingRegressor(max_depth=None, learning_rate=0.1, max_iter=n_trees, random_state=42)
                    suffix = "hgb"
                    use_wrapper = True
                    _d(f"[MODEL] HistGradientBoosting initialised max_iter={n_trees}")
                else:  # LightGBM with early stopping
                    base_params = dict(
                        learning_rate=0.03347500352712116,
                        max_depth=7,
                        num_leaves=24,
                        subsample=0.8832753633141975,
                        colsample_bytree=0.6292206613991069,
                        min_child_samples=44,
                    )
                    base_mdl = MultiLGBMRegressor(
                        base_params=base_params,
                        upper_bound_estimators=n_trees,
                        early_stopping_rounds=100,
                    )
                    suffix = "lgbm"
                    use_wrapper = False
                    _d(f"[TRAIN] LightGBM initialised – upper_bound={n_trees}, early_stop=100, base_params={base_params}")

                # ---------------- Fit -----------------------
                if use_wrapper:
                    mdl = MultiOutputRegressor(base_mdl)
                    mdl.fit(X_tr, y_tr)
                    _d("[TRAIN] Wrapper model fit complete")
                else:
                    if perform_validation and not X_te.empty:
                        # Standard early-stopping using external validation split
                        base_mdl.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
                        _d(f"[TRAIN] External early-stopping complete – best_iter={base_mdl.best_iteration_}")
                        mdl = base_mdl
                    else:
                        # ----------------------------------------------------------
                        # No validation split (user selected 100 % train) –> create
                        # an internal 90/10 chronological split to pick the best
                        # iteration, then refit on the full dataset with that
                        # fixed n_estimators so behaviour matches the legacy flow.
                        # ----------------------------------------------------------
                        int_train_df, int_val_df = split_train_eval_frac(df, test_frac=0.1)

                        X_int_tr, y_int_tr = features.select_feature_target_multi(
                            int_train_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                        )
                        X_int_val, y_int_val = features.select_feature_target_multi(
                            int_val_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                        )

                        finder = MultiLGBMRegressor(
                            base_params=base_params,
                            upper_bound_estimators=n_trees,
                            early_stopping_rounds=100,
                        )
                        finder.fit(X_int_tr, y_int_tr, eval_set=(X_int_val, y_int_val), verbose=False)

                        best_n = finder.best_iteration_ or n_trees

                        # Refit on **all** data with the chosen tree count
                        final_params = base_params | {"n_estimators": best_n}
                        final_lgbm = MultiLGBMRegressor(
                            base_params=final_params,
                            upper_bound_estimators=best_n,
                            early_stopping_rounds=0,
                        )
                        final_lgbm.fit(X_all, y_all)
                        mdl = final_lgbm
                        _d(f"[TRAIN] Refit on full data finished with best_n trees")
                        _d(
                            f"[TRAIN] Internal split sizes – train={len(int_train_df)}, val={len(int_val_df)}; "
                            f"best_n={best_n}"
                        )

                _d(f"{model_choice} model trained (wrapper={use_wrapper})")

                # Validation on unseen groups (if possible)
                if perform_validation and not X_te.empty:
                    preds = mdl.predict(X_te)
                    _d("Predictions generated on validation split")
                    mae_val = mean_absolute_error(y_te, preds)
                    rmse_val = mean_squared_error(y_te, preds) ** 0.5
                    _d(f"Validation metrics → MAE: {mae_val:.3f}, RMSE: {rmse_val:.3f}")
                else:
                    mae_val = rmse_val = float("nan")

                # -------- Persist model --------
                csv_stem = pathlib.Path(uploaded_file.name).stem.replace(" ", "_").lower()
                best_iter_val = int(getattr(mdl, 'best_iteration_', n_trees))
                _d(f"[SAVE] Persisting model with best_iter={best_iter_val}")
                model_name = f"{csv_stem}_{'fs_' if future_safe else ''}{suffix}_{best_iter_val}.joblib"
                _d(f"[SAVE] Model written to {model_name}")
                model_utils.save_model(mdl, name=model_name)

            if np.isnan(mae_val):
                st.sidebar.success(f"{model_choice} trained (validation split contained no ground-truth targets).")
            else:
                st.sidebar.success(f"{model_choice} trained! MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")
            # Persist split settings for later evaluation
            st.session_state["last_train_split_mode"] = "last30" if use_last_30 else "percentage"
            if not use_last_30:
                st.session_state["last_train_pct"] = 100 if train_pct == 100 else train_pct

        # Existing model evaluation
        with st.sidebar.expander(_t('Evaluate Model'), expanded=False):
            model_files = list_available_models()
            if not model_files:
                st.write(_t("No saved models yet."))
                eval_pressed = False
                selected_model = None
                eval_fc_pressed = False
                all_models_chk = False
            else:
                selected_model = st.selectbox(_t("Model file"), model_files)
                # Checkbox to act on all models
                all_models_chk = st.checkbox(_t("Apply to all models"), key="chk_eval_all")

                col_eval, col_evalfc = st.columns([1,1])
                with col_eval:
                    eval_pressed = st.button(_t("Evaluate"), key="btn_eval_single", use_container_width=True)
                with col_evalfc:
                    eval_fc_pressed = st.button(_t("Eval & Forecast"), key="btn_eval_fc", use_container_width=True)

        if (eval_pressed or eval_fc_pressed):
            if uploaded_file is None:
                st.warning(_t("Please upload a CSV first to evaluate."))
            else:
                # Determine which models to evaluate
                target_models = list_available_models() if all_models_chk else [selected_model]
                with st.spinner(_t("Evaluating model(s) – please wait...")):
                    df = _get_preprocessed_df(uploaded_file)
                    # Use same train/test split fraction recorded during training (default 20%)
                    test_frac = max(0.01, 1 - st.session_state.get("last_train_pct", 80)/100)

                    # ---------- Split matching the training configuration ----------
                    split_mode_prev = st.session_state.get("last_train_split_mode", "percentage")

                    if split_mode_prev == "last30":
                        df_train_base, df_eval_base = split_train_last_n_days(df, n_days=30)
                    else:
                        df_train_base, df_eval_base = split_train_eval_frac(df, test_frac=test_frac)

                    _d(f"Evaluation split – train rows: {len(df_train_base)}, test rows: {len(df_eval_base)}")
                    use_gap_fill = False  # skip calendar gap generation – evaluate only real rows

                    X_train_base, _ = features.select_feature_target_multi(
                        df_train_base, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE, allow_na=True
                    )  # NEW

                    for mdl_name in target_models:
                        df_train = df_train_base.copy()
                        df_eval = df_eval_base.copy()

                        # Evaluation now uses **all** rows from the split so that overlapping
                        # multi-horizon predictions are preserved; removal of sub-sampling.

                        mdl = load_trained_model(mdl_name)
                        if not mdl:
                            st.error(f"Could not load {mdl_name}")
                            continue

                        # If model is future-safe, drop env/extra cols from evaluation/training sets to mimic determinate-only input
                        if is_future_safe_model(mdl_name):
                            df_train = df_train.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")
                            df_eval = df_eval.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

                        # ---------------- Ensure category codes align with training ----------------
                        cat_cols_train_loop = df_train.select_dtypes(include=["object", "category"]).columns
                        categories_map = {c: pd.Categorical(df_train[c]).categories.tolist() for c in cat_cols_train_loop}

                        X_eval, _ = features.select_feature_target_multi(
                            df_eval, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE, allow_na=True
                        )  # NEW
                        # Align features to the model's expected input
                        feature_cols_mdl = get_feature_cols(mdl, X_eval)
                        X_eval_aligned = X_eval.reindex(columns=feature_cols_mdl, fill_value=0)
                        # NEW – generate aligned training design matrix for debugging visualisation
                        X_train, _ = features.select_feature_target_multi(
                            df_train, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE, allow_na=True
                        )  # NEW
                        X_train_aligned = X_train.reindex(columns=feature_cols_mdl, fill_value=0)
                        preds = model_utils.predict(mdl, X_eval_aligned)
                        _d(f"[EVAL] Predictions generated – shape={preds.shape} for model={mdl_name}")

                        # -------- Attach predictions to df_eval --------
                        if getattr(preds, "ndim", 1) == 2:
                            # Multi-output – assign for each configured horizon available
                            for idx, h in enumerate(HORIZON_TUPLE):
                                if idx < preds.shape[1]:
                                    df_eval.loc[X_eval_aligned.index, f"pred_h{h}d"] = preds[:, idx]
                            # For backward-compatibility plots keep original col name (use horizon-1)
                            df_eval.loc[X_eval_aligned.index, "predicted_temp"] = preds[:, 0]
                        else:
                            # Single-output – treat as horizon 1 only
                            df_eval.loc[X_eval_aligned.index, "predicted_temp"] = preds
                            df_eval.loc[X_eval_aligned.index, "pred_h1d"] = preds

                        df_eval["is_forecast"] = False

                        # Combine training (actual only) and evaluation rows for full context time-series
                        df_train_plot = df_train.copy()
                        df_train_plot["is_forecast"] = False
                        df_predplot_all = pd.concat([df_eval, df_train_plot], ignore_index=True)

                        # metrics
                        df_eval_actual = df_eval[df_eval[TARGET_TEMP_COL].notna()].copy()
                        # ----- Metrics per horizon -----
                        def _metric(col):
                            mask = df_eval_actual[[TARGET_TEMP_COL, col]].notna().all(axis=1)
                            if not mask.any():
                                return float("nan"), float("nan"), float("nan")
                            err = df_eval_actual.loc[mask, TARGET_TEMP_COL] - df_eval_actual.loc[mask, col]
                            mae_c = err.abs().mean()
                            rmse_c = (err ** 2).mean() ** 0.5
                            mape_c = (err.abs() / df_eval_actual.loc[mask, TARGET_TEMP_COL]).mean() * 100
                            return mae_c, rmse_c, mape_c

                        mae_h1, rmse_h1, mape_h1 = _metric("pred_h1d")
                        mae_h2, rmse_h2, mape_h2 = _metric("pred_h2d") if "pred_h2d" in df_eval_actual.columns else (float("nan"),)*3
                        mae_h3, rmse_h3, mape_h3 = _metric("pred_h3d") if "pred_h3d" in df_eval_actual.columns else (float("nan"),)*3

                        # Retain legacy overall MAE/RMSE as horizon 1
                        mae = mae_h1
                        rmse = rmse_h1
                        mape = mape_h1
                        conf, acc = compute_overall_metrics(df_eval_actual)

                        # -------------- Feature Importance ----------------
                        def _compute_importance(model, feature_cols):
                            if isinstance(model, (MultiOutputRegressor, MultiLGBMRegressor)):
                                # average over outputs
                                imps = np.mean([
                                    getattr(est, "feature_importances_", np.zeros(len(feature_cols)))
                                    for est in model.estimators_
                                ], axis=0)
                            elif hasattr(model, "feature_importances_"):
                                imps = getattr(model, "feature_importances_")
                            elif hasattr(model, "coef_"):
                                imps = np.abs(getattr(model, "coef_"))
                            else:
                                imps = np.zeros(len(feature_cols))
                            return imps

                        feat_importances = _compute_importance(mdl, feature_cols_mdl)
                        fi_df = (
                            pd.DataFrame({"feature": feature_cols_mdl, "importance": feat_importances})
                            .sort_values("importance", ascending=False)
                            .reset_index(drop=True)
                        )

                        st.session_state["evaluations"][mdl_name] = {
                            "df_eval": df_eval,
                            "df_predplot_all": df_predplot_all,
                            "confidence": conf,
                            "accuracy": acc,
                            "rmse": rmse,
                            "mae": mae,
                            "mape": mape,
                            "mae_h1": mae_h1,
                            "mae_h2": mae_h2,
                            "mae_h3": mae_h3,
                            "rmse_h1": rmse_h1,
                            "rmse_h2": rmse_h2,
                            "rmse_h3": rmse_h3,
                            "feature_cols": feature_cols_mdl,
                            "feature_importance": fi_df,
                            "categories_map": categories_map,
                            "horizon": len(df_eval["forecast_day"].unique()) if "forecast_day" in df_eval.columns else 0,
                            "df_base": df,
                            "model_name": mdl_name,
                            "future_safe": is_future_safe_model(mdl_name),
                            # NEW debug matrices
                            "X_train": X_train_aligned,
                            "X_eval": X_eval_aligned,
                        }

                    # last evaluated model as active
                    if target_models:
                        st.session_state["active_model"] = target_models[0]

                    st.sidebar.success("Evaluation(s) completed.")

                    # If user requested Eval & Forecast, automatically create forecast for selected model
                    if eval_fc_pressed:
                        st.sidebar.write(_t("Generating forecast…"))
                        models_to_fc = target_models  # already respects all_models_chk
                        for mdl in models_to_fc:
                            generate_and_store_forecast(mdl, horizon=HORIZON_DAYS)
                        st.sidebar.success("Forecast(s) created.")

    # Render evaluation view (chosen via dropdown instead of tabs)
    if st.session_state["evaluations"]:
        eval_keys = list(st.session_state["evaluations"].keys())
        active_model = st.session_state.get("active_model", eval_keys[0])

        chosen_model = st.selectbox(
            _t("Select evaluated model"),
            options=eval_keys,
            index=eval_keys.index(active_model) if active_model in eval_keys else 0,
        )
        # Persist selection so next rerun keeps the same model
        st.session_state["active_model"] = chosen_model

        # Inner tabs for the chosen model
        inner_tabs = st.tabs([_t("Evaluation"), _t("Forecast")])

        # --- Evaluation Tab ---
        with inner_tabs[0]:
            render_evaluation(chosen_model)

        # --- Forecast Tab ---
        with inner_tabs[1]:
            if chosen_model in st.session_state.get("forecasts", {}):
                render_forecast(chosen_model)
            else:
                st.info(_t("No forecast generated yet for this model."))
                if st.button(_t("Generate Forecast"), key=f"btn_gen_fc_main_{chosen_model}"):
                    with st.spinner(_t("Generating forecast…")):
                        if generate_and_store_forecast(chosen_model, horizon=HORIZON_DAYS):
                            st.success(_t("Forecast generated – switch tabs to view."))

    # --------------------------------------------------
    # Leaderboard (full-width collapsible panel) -----------------------------------
    evals = st.session_state["evaluations"]

    st.markdown("---")
    with st.expander(_t('Model Leaderboard'), expanded=False):
        if not evals:
            st.write(_t("No evaluations yet."))
        else:
            data = []
            for name, d in evals.items():
                data.append(
                    {
                        "model": name,
                        "confidence": d.get("confidence", float("nan")),
                        "accuracy": d.get("accuracy", float("nan")),
                        "rmse": d.get("rmse", float("nan")),
                        "mae": d.get("mae", float("nan")),
                    }
                )
            df_leader = (
                pd.DataFrame(data)
                .sort_values(["confidence", "accuracy"], ascending=False)
                .reset_index(drop=True)
            )
            df_leader.insert(0, "rank", df_leader.index + 1)
            st.dataframe(df_leader, use_container_width=True)

    # Optionally still expose full log
    if st.session_state.get("debug_mode"):
        dbg_log = st.session_state.get("debug_msgs", [])
        if dbg_log:
            with st.expander(_t('Debug Log (full)'), expanded=False):
                st.code("\n".join(dbg_log), language="text")



# ================== NEW HELPER RENDER FUNCTIONS ==================

def render_evaluation(model_name: str):
    """Render the evaluation view for a given *model_name* inside its tab."""
    res = st.session_state["evaluations"][model_name]
    # Categories captured during initial evaluation (may be empty)
    categories_map = res.get("categories_map", {})
    df_eval = res["df_eval"]
    df_predplot_all = res["df_predplot_all"]

    # Ensure 'forecast_day' exists for downstream UI widgets
    if "forecast_day" not in df_eval.columns and "detection_time" in df_eval.columns:
        date_series = pd.to_datetime(df_eval["detection_time"]).dt.floor("D")
        unique_dates_sorted = sorted(date_series.unique())
        date2idx = {d: idx for idx, d in enumerate(unique_dates_sorted, start=1)}
        df_eval["forecast_day"] = date_series.map(date2idx)
        # Apply same mapping to the combined prediction frame if present
        if "detection_time" in df_predplot_all.columns:
            df_predplot_all["forecast_day"] = pd.to_datetime(df_predplot_all["detection_time"]).dt.floor("D").map(date2idx)

    # ---------------- Warehouse → Silo cascading filters -----------------
    wh_col_candidates = [c for c in ["granary_id", "storepointName"] if c in df_eval.columns]
    silo_col_candidates = [c for c in ["heap_id", "storeName"] if c in df_eval.columns]

    wh_col = wh_col_candidates[0] if wh_col_candidates else None
    silo_col = silo_col_candidates[0] if silo_col_candidates else None

    # Apply global location filter (chosen in the sidebar)
    global_filters = st.session_state.get("global_filters", {})
    sel_wh = global_filters.get("wh", "All")
    sel_silo = global_filters.get("silo", "All")

    if wh_col:
        if sel_wh != "All":
            df_eval = df_eval[df_eval[wh_col] == sel_wh]
            df_predplot_all = df_predplot_all[df_predplot_all[wh_col] == sel_wh]

        if silo_col and sel_silo != "All":
            df_eval = df_eval[df_eval[silo_col] == sel_silo]
            df_predplot_all = df_predplot_all[df_predplot_all[silo_col] == sel_silo]

    # -------- Metrics (re-computed on filtered subset) ---------
    conf_val, acc_val = compute_overall_metrics(df_eval)
    if {TARGET_TEMP_COL, "predicted_temp"}.issubset(df_eval.columns) and not df_eval.empty:
        mae_val = (df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]).abs().mean()
        rmse_val = ((df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]) ** 2).mean() ** 0.5
        mape_val = ((df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]).abs() / df_eval[TARGET_TEMP_COL]).mean() * 100
    else:
        rmse_val = mae_val = mape_val = float("nan")

    # Per-horizon metrics table
    mae_h1 = res.get("mae_h1", float("nan"))
    mae_h2 = res.get("mae_h2", float("nan"))
    mae_h3 = res.get("mae_h3", float("nan"))
    rmse_h1 = res.get("rmse_h1", float("nan"))
    rmse_h2 = res.get("rmse_h2", float("nan"))
    rmse_h3 = res.get("rmse_h3", float("nan"))

    metric_cols = st.columns(6)
    with metric_cols[0]:
        st.metric(_t("Conf (%)"), "--" if pd.isna(conf_val) else f"{conf_val:.2f}")
    with metric_cols[1]:
        st.metric(_t("Acc (%)"), "--" if pd.isna(acc_val) else f"{acc_val:.2f}")
    with metric_cols[2]:
        st.metric(_t("MAE h+1"), "--" if pd.isna(mae_h1) else f"{mae_h1:.2f}")
    with metric_cols[3]:
        st.metric(_t("MAE h+2"), "--" if pd.isna(mae_h2) else f"{mae_h2:.2f}")
    with metric_cols[4]:
        st.metric(_t("MAE h+3"), "--" if pd.isna(mae_h3) else f"{mae_h3:.2f}")
    with metric_cols[5]:
        st.metric(_t("RMSE h+1"), "--" if pd.isna(rmse_h1) else f"{rmse_h1:.2f}")

    st.markdown("---")

    tab_labels = [_t("Summary"), _t("Predictions"), _t("3D Grid"), _t("Time Series"), _t("Anchor 7-day"), _t("Extremes"), _t("Debug")]
    summary_tab, pred_tab, grid_tab, ts_tab, anchor_tab, extremes_tab, debug_tab = st.tabs(tab_labels)

    with summary_tab:
        if "predicted_temp" in df_eval.columns:
            st.subheader(_t("Forecast Summary (per day)"))
            st.dataframe(
                forecast_summary(df_eval),
                use_container_width=True,
                key=f"summary_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}",
            )

            def exceeds(row):
                thresh = GRAIN_ALERT_THRESHOLDS.get(row.get("grain_type"), ALERT_TEMP_THRESHOLD)
                return row["predicted_temp"] >= thresh

            if df_eval.apply(exceeds, axis=1).any():
                st.error(_t("High temperature forecast detected for at least one grain type – monitor closely!"))
            else:
                st.success(_t("All predicted temperatures within safe limits for their grain types"))

            # ---------------- Top Features ----------------
            fi_df = res.get("feature_importance")
            if fi_df is not None and not fi_df.empty:
                st.markdown(f"### {_t('Top Predictive Features')}")
                st.dataframe(fi_df.head(15), use_container_width=True, key=f"feat_imp_{model_name}")

    with pred_tab:
        _st_dataframe_safe(df_predplot_all, key=f"pred_df_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")

    with grid_tab:
        # Build list of unique dates present in evaluation subset only
        unique_dates = sorted(pd.to_datetime(df_eval["detection_time"]).dt.floor("D").unique())
        date_choice = st.selectbox(
            _t("Select date"),
            options=[d.strftime("%Y-%m-%d") for d in unique_dates],
            key=f"day_{model_name}_grid_{len(unique_dates)}",
        )
        sel_date = pd.to_datetime(date_choice)
        df_predplot = df_predplot_all[pd.to_datetime(df_predplot_all["detection_time"]).dt.floor("D") == sel_date]
        plot_3d_grid(
            df_predplot,
            key=f"grid_{model_name}_{date_choice}",
            color_by_delta=True,
        )

    with ts_tab:
        # Merge past (training+eval) with future predictions so the trend is continuous
        hist_df = res.get("df_predplot_all")
        if hist_df is None:
            hist_df = pd.DataFrame()
        combined_df = pd.concat([hist_df, df_eval], ignore_index=True, sort=False)
        plot_time_series(
            combined_df,
            key=f"time_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}",
        )

    # -------------- ANCHOR 7-DAY TAB -------------------
    with anchor_tab:
        st.subheader("7-Day Forecast from Anchor Day (forecast_day=1)")

        if "forecast_day" not in df_eval.columns:
            st.info("forecast_day column missing – cannot compute anchor forecast.")
        else:
            # Build mapping from forecast_day -> calendar date (min date for that day)
            day_date_pairs = []
            for d_val in sorted(df_eval["forecast_day"].unique()):
                d_date = (
                    pd.to_datetime(df_eval[df_eval["forecast_day"] == d_val]["detection_time"])
                    .dt.floor("D")
                    .min()
                )
                if pd.notna(d_date):
                    day_date_pairs.append((d_date.strftime("%Y-%m-%d"), d_val))

            if not day_date_pairs:
                st.info("No anchor dates available.")
                return

            date_options = [p[0] for p in day_date_pairs]
            mapping_day = {p[0]: p[1] for p in day_date_pairs}

            sel_date_str = st.selectbox("Select anchor date", options=date_options, index=0, key=f"anchor_sel_{model_name}")
            sel_anchor = mapping_day[sel_date_str]

            anchor_rows = df_eval[df_eval["forecast_day"] == sel_anchor].copy()
            if anchor_rows.empty:
                st.info(f"No rows for selected anchor date {sel_date_str}.")
            else:
                # Determine anchor calendar date from selection
                anchor_date = pd.to_datetime(sel_date_str)

                records = []
                for h in HORIZON_TUPLE:
                    pred_col = f"pred_h{h}d"
                    target_date = anchor_date + pd.Timedelta(days=h)

                    # ---- Collect per-sensor prediction & actual ----
                    pred_subset = anchor_rows.copy()
                    if pred_col not in pred_subset.columns:
                        continue  # skip if horizon not available
                    pred_subset = pred_subset.assign(pred_val=pred_subset[pred_col])

                    act_subset = df_eval[pd.to_datetime(df_eval["detection_time"]).dt.floor("D") == target_date].copy()
                    act_subset = act_subset.assign(actual_val=act_subset[TARGET_TEMP_COL])

                    key_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in pred_subset.columns and c in act_subset.columns]

                    merged = pred_subset[key_cols + ["pred_val"]].merge(
                        act_subset[key_cols + ["actual_val"]], on=key_cols, how="inner"
                    )

                    if merged.empty:
                        mae = max_err = float("nan")
                    else:
                        diffs = (merged["pred_val"] - merged["actual_val"]).abs()
                        mae = diffs.mean()
                        max_err = diffs.max()

                    pred_mean = merged["pred_val"].mean() if not merged.empty else float("nan")
                    actual_mean = merged["actual_val"].mean() if not merged.empty else float("nan")

                    records.append({
                        "horizon_day": h,
                        "forecast_date": target_date.strftime("%Y-%m-%d"),
                        "predicted_mean": round(pred_mean, 2) if pd.notna(pred_mean) else "--",
                        "actual_mean": round(actual_mean, 2) if pd.notna(actual_mean) else "--",
                        "MAE": round(mae, 2) if pd.notna(mae) else "--",
                        "max_abs_err": round(max_err, 2) if pd.notna(max_err) else "--",
                    })

                anchor_tbl = pd.DataFrame(records)
                st.dataframe(anchor_tbl, use_container_width=True, key=f"anchor_{model_name}")

                # ---- Plot predicted vs actual over 7-day horizon ----
                try:
                    plot_df = anchor_tbl.replace("--", np.nan).dropna(subset=["predicted_mean", "actual_mean"]).copy()
                    plot_df["predicted_mean"] = pd.to_numeric(plot_df["predicted_mean"])
                    plot_df["actual_mean"] = pd.to_numeric(plot_df["actual_mean"])
                    if not plot_df.empty:
                        fig_anchor = go.Figure()
                        fig_anchor.add_trace(
                            go.Scatter(
                                x=plot_df["forecast_date"],
                                y=plot_df["predicted_mean"],
                                mode="lines+markers",
                                name="Predicted",
                            )
                        )
                        fig_anchor.add_trace(
                            go.Scatter(
                                x=plot_df["forecast_date"],
                                y=plot_df["actual_mean"],
                                mode="lines+markers",
                                name="Actual",
                            )
                        )
                        fig_anchor.update_layout(
                            title="Anchor-day 7-Day Forecast vs Actual",
                            xaxis_title="Date",
                            yaxis_title="Temperature (°C)",
                            xaxis=dict(tickformat="%Y-%m-%d"),
                        )
                        st.plotly_chart(fig_anchor, use_container_width=True, key=f"anchor_plot_{model_name}")
                except Exception as exc:
                    _d(f"Anchor plot error: {exc}")

    # ------------------ EXTREMES TAB ------------------
    with extremes_tab:
        st.subheader(_t("Daily Extremes (h+1)"))

        df_eval_actual = df_eval[df_eval[TARGET_TEMP_COL].notna()].copy()
        if df_eval_actual.empty or "pred_h1d" not in df_eval_actual.columns and "predicted_temp" not in df_eval_actual.columns:
            st.info(_t("No horizon-1 predictions available to compute extremes."))
        else:
            # Use column alias – pred_h1d preferred but fall back to predicted_temp
            pred_col = "pred_h1d" if "pred_h1d" in df_eval_actual.columns else "predicted_temp"
            df_eval_actual["date"] = pd.to_datetime(df_eval_actual["detection_time"]).dt.date
            df_eval_actual["error"] = df_eval_actual[pred_col] - df_eval_actual[TARGET_TEMP_COL]

            rows = []
            for d, grp in df_eval_actual.groupby("date"):
                daily_avg_dev = grp["error"].abs().mean()
                # Over-prediction (max positive error)
                over_row = grp.loc[grp["error"].idxmax()]
                # Under-prediction (most negative error)
                under_row = grp.loc[grp["error"].idxmin()]

                for typ, r in [("Over", over_row), ("Under", under_row)]:
                    rows.append(
                        {
                            "date": d,
                            "type": typ,
                            "predicted": r[pred_col],
                            "actual": r[TARGET_TEMP_COL],
                            "error": r["error"],
                            "avg_daily_abs_error": daily_avg_dev,
                            "grid_x": r.get("grid_x"),
                            "grid_y": r.get("grid_y"),
                            "grid_z": r.get("grid_z"),
                        }
                    )

            extremes_df = pd.DataFrame(rows)
            # Sort by date then over/under
            extremes_df.sort_values(["date", "type"], inplace=True)
            _st_dataframe_safe(extremes_df, key=f"extremes_{model_name}_{len(rows)}")

            # -------------- Time-series plots ----------------
            if not extremes_df.empty:
                # Ensure date as datetime for proper plotting
                extremes_df["date"] = pd.to_datetime(extremes_df["date"])

                # ---- 1. Average daily absolute error plot ----
                daily_avg = (
                    extremes_df[["date", "avg_daily_abs_error"]]
                    .drop_duplicates(subset="date")
                    .sort_values("date")
                )
                fig_avg = go.Figure(
                    data=[
                        go.Scatter(
                            x=daily_avg["date"],
                            y=daily_avg["avg_daily_abs_error"],
                            mode="lines+markers",
                            name="Avg |Error|",
                        )
                    ]
                )
                fig_avg.update_layout(
                    title="Average Daily Absolute Error (h+1)",
                    xaxis_title="Date",
                    yaxis_title="Avg |Error| (°C)",
                    xaxis=dict(tickformat="%Y-%m-%d"),
                )
                st.plotly_chart(fig_avg, use_container_width=True, key=f"avg_err_plot_{model_name}")

                # Helper to plot over/under lines
                def _plot_pred_vs_actual(sub_df: pd.DataFrame, title: str, key: str):
                    sub_df = sub_df.sort_values("date")
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=sub_df["date"],
                            y=sub_df["actual"],
                            mode="lines+markers",
                            name="Actual",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=sub_df["date"],
                            y=sub_df["predicted"],
                            mode="lines+markers",
                            name="Predicted",
                        )
                    )
                    fig.update_layout(
                        title=title,
                        xaxis_title="Date",
                        yaxis_title="Temperature (°C)",
                        xaxis=dict(tickformat="%Y-%m-%d"),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=key)

                # ---- 2. Over-prediction line plot ----
                over_df = extremes_df[extremes_df["type"] == "Over"]
                if not over_df.empty:
                    _plot_pred_vs_actual(over_df, "Over-Prediction (h+1)", f"over_plot_{model_name}")

                # ---- 3. Under-prediction line plot ----
                under_df = extremes_df[extremes_df["type"] == "Under"]
                if not under_df.empty:
                    _plot_pred_vs_actual(under_df, "Under-Prediction (h+1)", f"under_plot_{model_name}")

    # ------------------ DEBUG TAB ------------------
    with debug_tab:
        st.subheader(f"{_t('Feature Matrices (first 100 rows)')} (Training – X_train)")
        x_train_dbg = res.get("X_train")
        if x_train_dbg is not None:
            st.write(_t("Training – X_train"))
            _st_dataframe_safe(x_train_dbg, key=f"xtrain_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")
        x_eval_dbg = res.get("X_eval")
        if x_eval_dbg is not None:
            st.write(_t("Evaluation – X_eval"))
            _st_dataframe_safe(x_eval_dbg, key=f"xeval_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")
        st.write(_t("Model Feature Columns (order)"))
        st.code(", ".join(res.get("feature_cols", [])))


def render_forecast(model_name: str):
    """Render the forecast view (if available) for *model_name*."""
    forecast_data = st.session_state.get("forecasts", {}).get(model_name)
    if not forecast_data:
        st.info(_t("No forecast generated for this model yet."))
        return

    # -------- Initial dataframe & warehouse/silo filters --------
    future_df = forecast_data["future_df"]

    # Apply global location filter (chosen in the sidebar)
    global_filters = st.session_state.get("global_filters", {})
    sel_wh_fc = global_filters.get("wh", "All")
    sel_silo_fc = global_filters.get("silo", "All")

    wh_col_candidates = [c for c in ["granary_id", "storepointName"] if c in future_df.columns]
    silo_col_candidates = [c for c in ["heap_id", "storeName"] if c in future_df.columns]

    wh_col = wh_col_candidates[0] if wh_col_candidates else None
    silo_col = silo_col_candidates[0] if silo_col_candidates else None

    if wh_col:
        if sel_wh_fc != "All":
            future_df = future_df[future_df[wh_col] == sel_wh_fc]

        if silo_col and sel_silo_fc != "All":
            future_df = future_df[future_df[silo_col] == sel_silo_fc]

    # Update forecast_data copy with filtered df for downstream plots
    df_plot_base = future_df.copy()

    # -------- Metrics forwarded from last evaluation (confidence etc.) ---------
    res_eval = st.session_state.get("evaluations", {}).get(model_name, {})
    conf_val = res_eval.get("confidence", float("nan"))
    acc_val = res_eval.get("accuracy", float("nan"))
    rmse_val = res_eval.get("rmse", float("nan"))
    mae_val = res_eval.get("mae", float("nan"))
    mape_val = res_eval.get("mape", float("nan"))

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric(_t("Confidence (%)"), "--" if pd.isna(conf_val) else f"{conf_val:.3f}")
    with metric_cols[1]:
        st.metric(_t("Accuracy (%)"), "--" if pd.isna(acc_val) else f"{acc_val:.3f}")
    with metric_cols[2]:
        st.metric(_t("RMSE"), "--" if pd.isna(rmse_val) else f"{rmse_val:.3f}")
    with metric_cols[3]:
        st.metric(_t("MAE"), "--" if pd.isna(mae_val) else f"{mae_val:.3f}")
    with metric_cols[4]:
        st.metric(_t("MAPE (%)"), "--" if pd.isna(mape_val) else f"{mape_val:.3f}")

    st.markdown("---")

    # Tabs similar to evaluation (+Debug)
    summary_tab, pred_tab, grid_tab, ts_tab, extremes_tab, debug_tab = st.tabs([_t("Summary"), _t("Predictions"), _t("3D Grid"), _t("Time Series"), _t("Extremes"), _t("Debug")])

    with summary_tab:
        # Only predicted statistics available
        grp = (
            future_df.groupby("forecast_day")
            .agg(pred_mean=("predicted_temp", "mean"), pred_max=("predicted_temp", "max"), pred_min=("predicted_temp", "min"))
            .reset_index()
        )
        st.subheader(_t("Forecast Summary (predicted)"))
        _st_dataframe_safe(grp, key=f"forecast_summary_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}")

    with pred_tab:
        _st_dataframe_safe(future_df, key=f"future_pred_df_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}")

    with grid_tab:
        day_choice = st.selectbox(
            "Select day",
            options=list(range(1, len(future_df['forecast_day'].unique()) + 1)),
            key=f"future_day_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}",
        )
        day_df = future_df[future_df.get("forecast_day", 1) == day_choice]
        plot_3d_grid(day_df, key=f"future_grid_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}")

    with ts_tab:
        # Merge past (training+eval) with future predictions so the trend is continuous
        hist_df = res_eval.get("df_predplot_all")
        if hist_df is None:
            hist_df = pd.DataFrame()
        combined_df = pd.concat([hist_df, future_df], ignore_index=True, sort=False)
        plot_time_series(
            combined_df,
            key=f"future_ts_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}",
        )

    # ------------------ EXTREMES TAB ------------------
    with extremes_tab:
        st.subheader(_t("Daily Predicted Extremes"))

        if future_df.empty or "predicted_temp" not in future_df.columns:
            st.info(_t("No predictions found to compute extremes."))
        else:
            tmp = future_df.copy()
            tmp["date"] = pd.to_datetime(tmp["detection_time"]).dt.date

            rows = []
            for d, grp in tmp.groupby("date"):
                # Highest and lowest predicted temperature for the day
                max_row = grp.loc[grp["predicted_temp"].idxmax()]
                min_row = grp.loc[grp["predicted_temp"].idxmin()]

                for typ, r in [("Max", max_row), ("Min", min_row)]:
                    rows.append(
                        {
                            "date": d,
                            "type": typ,
                            "predicted": r["predicted_temp"],
                            "grid_x": r.get("grid_x"),
                            "grid_y": r.get("grid_y"),
                            "grid_z": r.get("grid_z"),
                        }
                    )

            extreme_pred_df = pd.DataFrame(rows)
            if extreme_pred_df.empty:
                st.info(_t("No predictions found to compute extremes."))
            else:
                extreme_pred_df.sort_values(["date", "type"], inplace=True)
                _st_dataframe_safe(extreme_pred_df, key=f"forecast_extremes_{model_name}_{len(rows)}")

    # ------------------ DEBUG TAB ------------------
    with debug_tab:
        st.subheader(f"{_t('Future Feature Matrix (first 100 rows)')} (Training – X_train)")
        x_future_dbg = st.session_state.get("forecasts", {}).get(model_name, {}).get("X_future")
        if x_future_dbg is not None:
            st.dataframe(x_future_dbg.head(100), use_container_width=True)
            # Compare to evaluation matrix if available
            eval_res = st.session_state["evaluations"].get(model_name, {})
            x_eval_dbg = eval_res.get("X_eval")
            if x_eval_dbg is not None:
                delta = (x_eval_dbg.mean() - x_future_dbg.mean()).abs().sort_values(ascending=False)
                st.subheader(_t("|Mean(X_eval) − Mean(X_future)| (Top 20)"))
                st.dataframe(delta.head(20).to_frame(name="abs_diff"), use_container_width=True)
        else:
            st.info(_t("X_future matrix not available yet."))

# --------------------------------------------------
# Helper to create & store forecast
def generate_and_store_forecast(model_name: str, horizon: int) -> bool:
    """Generate future_df for *model_name* and store in session_state['forecasts'].
    Returns True if successful, False otherwise."""
    res_eval = st.session_state.get("evaluations", {}).get(model_name)
    if res_eval is None:
        st.error(_t("Please evaluate the model first."))
        return False

    base_df = res_eval.get("df_base")
    categories_map = res_eval.get("categories_map", {})
    mdl = load_trained_model(model_name)

    if not isinstance(base_df, pd.DataFrame) or mdl is None:
        st.error(_t("Unable to access base data or model for forecasting."))
        return False

    # Special handling if the model is *direct* multi-output and the requested
    # horizon fits within the model's native multi-output dimensions.
    if isinstance(mdl, (MultiOutputRegressor, MultiLGBMRegressor)) and horizon <= HORIZON_DAYS:
        # 1. Take **last known row** per physical sensor as input snapshot
        sensors_key = [c for c in [
            "granary_id", "heap_id", "grid_x", "grid_y", "grid_z"
        ] if c in base_df.columns]

        last_rows = (
            base_df.sort_values("detection_time")
            .groupby(sensors_key, dropna=False)
            .tail(1)
            .copy()
        )

        # Prepare design matrix
        X_snap, _ = features.select_feature_target_multi(
            last_rows, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE, allow_na=True
        )
        model_feats = get_feature_cols(mdl, X_snap)
        X_snap_aligned = X_snap.reindex(columns=model_feats, fill_value=0)

        preds_mat = model_utils.predict(mdl, X_snap_aligned)  # shape (n, 3)

        n_out = preds_mat.shape[1] if getattr(preds_mat, "ndim", 1) == 2 else 1

        # Build future frames for 1, 2, 3-day horizons ------------------
        all_future_frames: list[pd.DataFrame] = []
        last_dt = pd.to_datetime(last_rows["detection_time"]).max()

        for h in range(1, horizon + 1):
            day_frame = last_rows.copy()
            day_frame["detection_time"] = last_dt + timedelta(days=h)
            day_frame["forecast_day"] = h

            idx = min(h - 1, n_out - 1)  # fallback to last available output
            if getattr(preds_mat, "ndim", 1) == 2:
                pred_val = preds_mat[:, idx]
            else:
                pred_val = preds_mat  # 1-D: same value for all horizons

            day_frame["predicted_temp"] = pred_val
            day_frame["temperature_grain"] = pred_val
            day_frame[TARGET_TEMP_COL] = pred_val
            day_frame["is_forecast"] = True
            all_future_frames.append(day_frame)

        future_df = pd.concat(all_future_frames, ignore_index=True)

        # Clear actual temperature values for forecast rows to avoid confusion in plots
        future_df.loc[future_df["is_forecast"], TARGET_TEMP_COL] = np.nan

        # Assign debug matrix for consistency with fallback path
        X_day_aligned = X_snap_aligned.copy()

    else:
        # Fallback – original recursive loop (supports arbitrary horizons)
        hist_df = base_df.copy()
        all_future_frames: list[pd.DataFrame] = []

        for d in range(1, horizon + 1):
            # Generate placeholder rows for ONE day ahead
            day_df = make_future(hist_df, horizon_days=1)
            day_df = _inject_future_lag(day_df, hist_df)
            day_df["forecast_day"] = d

            # Extra features (lags, rolling) before encoding
            day_df = features.add_multi_lag(day_df, lags=(1,3,7,14,30))
            day_df = features.add_rolling_stats(day_df, window_days=7)

            # Apply categories levels
            for col, cats in categories_map.items():
                if col in day_df.columns:
                    day_df[col] = pd.Categorical(day_df[col], categories=cats)

            X_day, _ = features.select_feature_target_multi(
                day_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE, allow_na=True
            )
            model_feats = get_feature_cols(mdl, X_day)
            X_day_aligned = X_day.reindex(columns=model_feats, fill_value=0)
            preds = model_utils.predict(mdl, X_day_aligned)

            if preds.ndim == 2:
                preds_step = preds[:, 0]
            else:
                preds_step = preds

            day_df["predicted_temp"] = preds_step
            day_df["temperature_grain"] = preds_step  # feed back as history for next lag
            day_df[TARGET_TEMP_COL] = preds_step
            day_df["is_forecast"] = True

            hist_df = pd.concat([hist_df, day_df], ignore_index=True, sort=False)
            all_future_frames.append(day_df)

        future_df = pd.concat(all_future_frames, ignore_index=True)

        # Clear actual temperature values for forecast rows to avoid confusion in plots
        future_df.loc[future_df["is_forecast"], TARGET_TEMP_COL] = np.nan

    st.session_state.setdefault("forecasts", {})[model_name] = {
        "future_df": future_df,
        "future_horizon": horizon,
        "X_future": X_day_aligned,  # last horizon step matrix for debug
    }
    _d(f"[FORECAST] Stored forecast – rows={len(future_df)}")
    return True


# ---------------- Utility to extract feature column order from a model -----------------
def get_feature_cols(model, X_fallback: pd.DataFrame) -> list[str]:
    """Return the exact feature columns the *model* expects.

    1. scikit-learn 1.0+ estimators expose ``feature_names_in_``.
    2. LightGBM exposes ``feature_name_``.
    3. Fallback: use the columns of *X_fallback* (already aligned for current dataset).
    """
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    if hasattr(model, "feature_name_"):
        return list(getattr(model, "feature_name_"))
    return list(X_fallback.columns)


# ----------------- Helper for future lag injection -----------------
def _inject_future_lag(future_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Populate lag_temp_1d in *future_df* using the last known
    temperature_grain for each sensor from *history_df*.  Assumes both frames
    contain grid_x/y/z columns.
    """
    if {"grid_x", "grid_y", "grid_z", "temperature_grain"}.issubset(history_df.columns):
        last_vals = (
            history_df.sort_values("detection_time")
            .dropna(subset=["temperature_grain"])
            .groupby(["grid_x", "grid_y", "grid_z"])["temperature_grain"]
            .last()
        )
        idx = future_df.set_index(["grid_x", "grid_y", "grid_z"]).index
        future_df["lag_temp_1d"] = [last_vals.get(key, np.nan) for key in idx]
    return future_df


# ---------------------------------------------------------------------
# Common dataframe preprocessing (clean → fill → features → lag → sort)
# ---------------------------------------------------------------------

def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full data-prep pipeline exactly once."""
    if df.empty:
        return df
    _d("Starting basic_clean…")
    before_cols = list(df.columns)
    df = cleaning.basic_clean(df)
    _d(f"basic_clean: cols before={len(before_cols)} after={len(df.columns)} rows={len(df)}")

    # -------------------------------------------------------------
    # 1️⃣ Insert missing calendar-day rows first
    # -------------------------------------------------------------
    df = _insert_calendar_gaps(df)
    _d("insert_calendar_gaps: added rows for missing dates")

    # -------------------------------------------------------------
    # 2️⃣ Interpolate numeric columns per sensor across the now-complete
    #    timeline so gap rows take the average of surrounding real values.
    # -------------------------------------------------------------
    df = _interpolate_sensor_numeric(df)
    _d("_interpolate_sensor_numeric: linear interpolation applied per sensor")

    # -------------------------------------------------------------
    # 3️⃣ Final fill_missing to tidy up any residual NaNs (categoricals etc.)
    # -------------------------------------------------------------
    na_before = df.isna().sum().sum()
    df = cleaning.fill_missing(df)
    na_after = df.isna().sum().sum()
    _d(f"fill_missing (final): total NaNs before={na_before} after={na_after}")

    df = features.create_time_features(df)
    _d("create_time_features: added year/month/day/hour cols")
    df = features.create_spatial_features(df)
    _d("create_spatial_features: removed grid_index if present")
    before_lag_na = df["temperature_grain"].isna().sum() if "temperature_grain" in df.columns else 0
    df = features.add_sensor_lag(df)
    after_lag_na = df["lag_temp_1d"].isna().sum() if "lag_temp_1d" in df.columns else 0
    _d(f"add_sensor_lag: lag NaNs={after_lag_na} (target NaNs before={before_lag_na})")
    # Ensure group identifiers available for downstream splitting/evaluation
    df = assign_group_id(df)
    _d("assign_group_id: _group_id column added to dataframe")
    df = comprehensive_sort(df)
    _d("comprehensive_sort: dataframe sorted by granary/heap/grid/date")

    # -------------------------------------------------------------
    # 4️⃣ Extra temperature features (multi-lag, rolling stats, delta)
    # -------------------------------------------------------------
    df = features.add_multi_lag(df, lags=(1,3,7,14,30))
    df = features.add_rolling_stats(df, window_days=7)
    _d("add_multi_lag & add_rolling_stats: extra features added")

    # -------------------------------------------------------------
    # 5️⃣ Multi-horizon targets (1–3 days ahead)
    # -------------------------------------------------------------
    df = features.add_multi_horizon_targets(df, horizons=HORIZON_TUPLE)  # NEW
    _d("add_multi_horizon_targets: future target columns added")

    return df


# ---------------------------------------------------------------------
# Helper: insert rows for missing calendar dates per sensor
# ---------------------------------------------------------------------

def _insert_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* where any missing *calendar days* for each sensor are
    back-filled with synthetic rows so models see a continuous timeline.

    • Sensor grouping columns: granary_id, heap_id, grid_x/y/z (subset present).
    • For each missing date, copies the most recent known row for that sensor
      and nulls out numeric, non-static measurement columns so they can be
      filled later (mean, ffill etc.).
    """
    if "detection_time" not in df.columns:
        return df

    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")

    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    if not group_cols:
        group_cols = []  # treat whole frame as one group

    frames = [df]

    # Helper to decide which numeric cols are *measurements* (varying) vs static
    static_like = set(group_cols + ["granary_id", "heap_id", "grain_type", "warehouse_type"])  # do not null

    for key, sub in df.groupby(group_cols) if group_cols else [(None, df)]:
        sub = sub.sort_values("detection_time")
        date_floor = sub["detection_time"].dt.floor("D")
        full_range = pd.date_range(date_floor.min(), date_floor.max(), freq="D")
        missing_dates = sorted(set(full_range.date) - set(date_floor.dt.date.unique()))
        if not missing_dates:
            continue

        # Use last known row as template (static cols correct)
        template = sub.iloc[-1].copy()

        new_rows = []
        for md in missing_dates:
            row = template.copy()
            row["detection_time"] = pd.Timestamp(md)
            # Null out non-static numeric columns to be filled later
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in static_like:
                    row[col] = np.nan
            new_rows.append(row)

        if new_rows:
            frames.append(pd.DataFrame(new_rows))

    df_full = pd.concat(frames, ignore_index=True)
    return df_full


# ---------------------------------------------------------------------
# Helper to fetch raw (possibly organised) dataframe
# ---------------------------------------------------------------------

def _get_active_df(uploaded_file):
    """Return the raw dataframe – organised slice concat if available."""
    if st.session_state.get("organized_df") is not None:
        return st.session_state["organized_df"].copy()
    return load_uploaded_file(uploaded_file)


# ---------------------------------------------------------------------
# Helper to fetch whichever DataFrame (raw/organised) is active & processed
# ---------------------------------------------------------------------

def _preprocess_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper around the heavy preprocessing pipeline (no Streamlit caching – we persist processed CSVs instead)."""
    _d("_preprocess_cached: running full pipeline (no Streamlit cache)")
    return _preprocess_df(df)

def _get_preprocessed_df(uploaded_file):
    """Return a fully-preprocessed dataframe (cached in session)."""
    # --------------------------------------------------------
    # 0️⃣ Fast-path: if the uploaded file is already a processed
    #    CSV (name ends with _processed.csv or resides in data/processed),
    #    simply load and return it.
    # --------------------------------------------------------
    if _looks_processed(uploaded_file):
        try:
            _d("✅ Detected preprocessed CSV – loading directly, skipping heavy pipeline")
            df_fast = pd.read_csv(uploaded_file, encoding="utf-8") if isinstance(uploaded_file, (str, pathlib.Path)) else pd.read_csv(uploaded_file.name, encoding="utf-8")
            st.session_state["processed_df"] = df_fast.copy()
            return df_fast
        except Exception as exc:
            _d(f"Could not load preprocessed CSV fast-path: {exc}; falling back to pipeline")

    raw_df = _get_active_df(uploaded_file)

    # Use cached preprocessing to avoid repeating heavy work across reruns
    proc = _preprocess_cached(raw_df)
    _d("🔄 Received dataframe from _preprocess_cached (may be cache hit or miss)")

    # --------------------------------------------------------
    # Persist a processed CSV alongside others for future fast-path
    # --------------------------------------------------------
    try:
        if hasattr(uploaded_file, "name"):
            orig_name = pathlib.Path(uploaded_file.name).stem
        else:
            orig_name = pathlib.Path(uploaded_file).stem if isinstance(uploaded_file, (str, pathlib.Path)) else "uploaded"

        processed_dir = pathlib.Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_csv = processed_dir / f"{orig_name}_processed.csv"
        if not out_csv.exists():
            proc.to_csv(out_csv, index=False, encoding="utf-8")
            _d(f"💾 Saved processed CSV to {out_csv}")
    except Exception as exc:
        _d(f"Could not persist processed CSV: {exc}")

    st.session_state["processed_df"] = proc.copy()
    return proc


# ---------------------------------------------------------------------
# Helper to safely display DataFrames in Streamlit (Category→str)
# ---------------------------------------------------------------------

def _st_dataframe_safe(df: pd.DataFrame, key: str | None = None):
    """Wrapper around st.dataframe that converts category columns to string
    to avoid pyarrow ArrowInvalid errors when categories mix numeric & text.
    """
    df_disp = df.copy()
    for col in df_disp.select_dtypes(include=["category"]).columns:
        df_disp[col] = df_disp[col].astype(str)
    st.dataframe(df_disp, use_container_width=True, key=key)


# ---------------------------------------------------------------------
# Helper to guess if an uploaded file is already processed
# ---------------------------------------------------------------------

def _looks_processed(upload):
    """Return True if *upload* path or name suggests preprocessed dataset."""
    if isinstance(upload, (str, pathlib.Path)):
        p = pathlib.Path(upload)
        if "data/processed" in p.as_posix() or p.name.endswith("_processed.csv"):
            return True
    elif hasattr(upload, "name"):
        name = upload.name
        if name.endswith("_processed.csv"):
            return True
    return False


# ---------------------------------------------------------------------
# Helper: numeric interpolation per sensor across calendar-completed frame
# ---------------------------------------------------------------------

def _interpolate_sensor_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """For each sensor group, linearly interpolate numeric columns along
    chronological order so values for synthetic gap rows equal the average of
    previous and next real measurements."""

    if "detection_time" not in df.columns:
        return df

    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    df.sort_values("detection_time", inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df

    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    if group_cols:
        df[num_cols] = (
            df.groupby(group_cols)[num_cols]
            .apply(lambda g: g.interpolate(method="linear").ffill().bfill())
            .reset_index(level=group_cols, drop=True)
        )
    else:
        df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()

    return df


# -------------------------------------------------------------------
# NEW – fixed‐window split (last *n_days* for training)  Jun-2025
# -------------------------------------------------------------------


def split_train_last_n_days(df: pd.DataFrame, n_days: int = 30):
    """Return (df_train, df_eval) where training data is restricted to the
    most recent *n_days* of records (by unique date).  All earlier data
    becomes the evaluation set.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset ordered arbitrarily.
    n_days : int, default 30
        Number of unique dates to keep for training.
    """
    if df.empty or "detection_time" not in df.columns:
        return df, pd.DataFrame()

    df = df.copy()
    df["_date"] = pd.to_datetime(df["detection_time"]).dt.date

    unique_dates = sorted(df["_date"].unique())
    if not unique_dates:
        return df, pd.DataFrame()

    # Latest *n_days* are reserved for evaluation; all earlier data for training
    cutoff_dates = unique_dates[-n_days:]

    df_eval = df[df["_date"].isin(cutoff_dates)].copy()
    df_train = df[~df["_date"].isin(cutoff_dates)].copy()

    # Assign forecast_day within evaluation set: 1 = oldest day in eval, …
    date_to_idx = {date: idx for idx, date in enumerate(sorted(cutoff_dates), start=1)}
    df_eval["forecast_day"] = df_eval["_date"].map(date_to_idx).astype(int)

    df_train.drop(columns=["_date"], inplace=True)
    df_eval.drop(columns=["_date"], inplace=True)
    return df_train, df_eval


# ---------------- NEW – evaluation sub-sampler --------------------

def _subsample_every_k_days(df: pd.DataFrame, k: int = 3, *, date_col: str = "detection_time") -> pd.DataFrame:
    """Return *df* where only one row every *k* days (by *date_col*) is kept.

    This reduces overlap between successive multi-horizon predictions in the
    evaluation split.  Rows belonging to dates that are not selected are
    dropped entirely.  If *date_col* is missing or the frame is empty the
    input is returned unchanged.
    """
    if df.empty or date_col not in df.columns:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_date_only"] = df[date_col].dt.date

    unique_dates = sorted(df["_date_only"].unique())
    if not unique_dates:
        return df.drop(columns=["_date_only"], errors="ignore")

    keep_dates = set(unique_dates[::k])  # every k-th day starting from first
    df = df[df["_date_only"].isin(keep_dates)].copy()
    df.drop(columns=["_date_only"], inplace=True)
    return df


if __name__ == "__main__":
    main() 