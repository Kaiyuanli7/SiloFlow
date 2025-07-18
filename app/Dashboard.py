import pathlib
from datetime import timedelta, datetime
from typing import Optional, List
import multiprocessing

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
from granarypredict.optuna_cache import load_optimal_params, save_optimal_params, clear_cache, list_cached_params  # NEW

# Streamlit reload may have stale module; fetch grain thresholds safely
try:
    from granarypredict.config import GRAIN_ALERT_THRESHOLDS  # type: ignore
except ImportError:
    GRAIN_ALERT_THRESHOLDS = {}

from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor  # NEW
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------
# 🈯️  Simple i18n helper  (EN / 中文)  -----------------------------------
# ---------------------------------------------------------------------


_TRANSLATIONS_ZH: dict[str, str] = {
    # Core UI elements
    "Language / 语言": "语言 / Language",
    "English": "英文",
    "中文": "中文",
    
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
    "Performance Optimization": "性能优化",
    "Parameter Cache": "参数缓存",
    
    # Training options
    "Optuna hyperparameter optimization": "Optuna 超参数优化",
    "Quantile regression objective": "分位数回归目标",
    "Anchor-day early stopping": "锚定日提前停止",
    "Horizon Balancing Configuration": "预测期平衡配置",
    "Balance horizon training": "平衡预测期训练",
    "Horizon weighting strategy": "预测期权重策略",
    "equal": "等权重",
    "increasing": "递增权重",
    "decreasing": "递减权重",
    "Optuna trials": "Optuna 试验次数",
    "Fast Optuna mode": "快速 Optuna 模式",
    "Enable parallel trials": "启用并行试验",
    "Parallel jobs": "并行作业数",
    "Use parameter cache": "使用参数缓存",
    "Force re-optimization": "强制重新优化",
    "Clear cache": "清除缓存",
    "Clear all cache": "清除所有缓存",
    "Refresh": "刷新",
    
    # Help text
    "Enable Optuna to automatically tune LightGBM parameters for optimal performance": "启用 Optuna 自动调整 LightGBM 参数以获得最佳性能",
    "Use LightGBM quantile regression (alpha 0.5) for improved mean absolute error performance": "使用 LightGBM 分位数回归（alpha 0.5）以提高平均绝对误差性能",
    "Use 7-day consecutive forecasting accuracy for early stopping with optimized interval checking": "使用 7 天连续预测准确性进行提前停止，优化间隔检查",
    "Ensures equal priority for all forecast horizons (H+1 through H+7) during model training": "确保在模型训练期间所有预测期（H+1 到 H+7）具有相同优先级",
    "equal: All horizons get equal priority (recommended) | increasing: Later horizons get more weight | decreasing: Earlier horizons get more weight": "等权重：所有预测期具有相同优先级（推荐）| 递增权重：后期预测期权重更大 | 递减权重：前期预测期权重更大",
    "Choose how to divide data into training vs validation sets.": "选择如何将数据分为训练集和验证集。",
    "Percentage of data used for training; set to 100% to train on the whole dataset without a validation split.": "用于训练的数据百分比；设置为 100% 可在整个数据集上训练而不进行验证分割。",
    "Use performance optimizations: 2-fold CV, lower tree limits, aggressive early stopping": "使用性能优化：2 折交叉验证、较低的树限制、积极的提前停止",
    "Run multiple Optuna trials in parallel for 2-4x faster optimization": "并行运行多个 Optuna 试验，优化速度提高 2-4 倍",
    "Number of parallel processes (max: {} CPU cores)": "并行进程数（最大：{} 个 CPU 核心）",
    "Automatically save/load optimal parameters from previous Optuna runs. Works even when Optuna is disabled!": "自动保存/加载先前 Optuna 运行的最优参数。即使禁用 Optuna 也能工作！",
    "Run Optuna even if cached parameters exist": "即使存在缓存参数也运行 Optuna",
    "Clear all cached parameters": "清除所有缓存的参数",
    "Show detailed internal processing messages": "显示详细的内部处理消息",
    
    # Status messages and notifications
    "Optuna hyperparameter optimization enabled": "Optuna 超参数优化已启用",
    "Using default LightGBM parameters": "使用默认 LightGBM 参数",
    "Quantile regression objective enabled": "分位数回归目标已启用",
    "Using standard regression objective": "使用标准回归目标",
    "Anchor-day early stopping enabled for enhanced 7-day accuracy": "锚定日提前停止已启用，提高 7 天准确性",
    "Using standard early stopping method": "使用标准提前停止方法",
    "Horizon balancing enabled - correcting forecast horizon bias": "预测期平衡已启用 - 纠正预测期偏差",
    "Using standard horizon weighting approach": "使用标准预测期权重方法",
    "Future-safe mode enabled - environmental variables excluded": "未来安全模式已启用 - 排除环境变量",
    "All variables included, including environmental data": "包含所有变量，包括环境数据",
    
    # Processing messages
    "Processing uploaded file: {}": "处理上传的文件：{}",
    "Processing uploaded file...": "处理上传的文件...",
    "Applied filters: {}": "应用的过滤器：{}",
    "Displaying all locations": "显示所有位置",
    "Data preprocessing complete. Dataset shape: {}": "数据预处理完成。数据集形状：{}",
    "Selected model: {}": "选择的模型：{}",
    "Initiating model training process...": "启动模型训练过程...",
    "Starting model evaluation...": "开始模型评估...",
    "Starting evaluation and forecast generation...": "开始评估和预测生成...",
    "Evaluation completed for {} model(s)": "已完成 {} 个模型的评估",
    "Forecast generated for {} model(s)": "已为 {} 个模型生成预测",
    "Generating forecast for selected model...": "为选定模型生成预测...",
    
    # Conservative system messages
    "Conservative Temperature System": "保守温度系统",
    "Conservative mode": "保守模式",
    "Stability boost": "稳定性增强",
    "Directional boost": "方向性增强",
    "Bootstrap samples": "自助采样数",
    "Conservative System Features (Click to expand)": "保守系统特性（点击展开）",
    "Thermal Physics Features:": "热物理特性：",
    "Thermal Inertia": "热惯性",
    "Stability Index": "稳定指数",
    "Change Resistance": "变化阻力",
    "Equilibrium Temperature": "平衡温度",
    "Mean Reversion": "均值回归",
    "Historical Stability": "历史稳定性",
    "Dampening Factor": "阻尼因子",
    "Progressive Penalties": "累进惩罚",
    "How This Improves Predictions:": "这如何改善预测：",
    
    # Parallel processing messages
    "Parallel processing available: {}, max_workers: {}, CPU cores: {}": "并行处理可用：{}，最大工作进程：{}，CPU 核心：{}",
    "Parallel processing: {} workers on {} cores": "并行处理：{} 个工作进程在 {} 个核心上",
    "Parallel Processing": "并行处理",
    "Workers": "工作进程",
    "Expected Speedup": "预期加速",
    "3-5x faster feature engineering": "特征工程速度提高 3-5 倍",
    "2-4x faster hyperparameter optimization": "超参数优化速度提高 2-4 倍",
    "Active": "激活",
    "Disabled": "禁用",
    "Enable All Cores": "启用所有核心",
    "Disable Parallel": "禁用并行",
    "Enable Parallel Processing": "启用并行处理",
    "Could not retrieve parallel processing information: {}": "无法检索并行处理信息：{}",
    
    # Parameter cache messages
    "Cached parameter sets": "缓存的参数集",
    "Best MAE": "最佳 MAE",
    "Trials": "试验",
    "Data shape": "数据形状",
    "Cached": "缓存时间",
    "Clear this": "清除此项",
    "No cached parameters available.": "没有可用的缓存参数。",
    "Train a model with Optuna optimization to create cache entries.": "使用 Optuna 优化训练模型以创建缓存条目。",
    "Clearing parameter cache...": "正在清除参数缓存...",
    "Parameter cache cleared!": "参数缓存已清除！",
    "Clearing cache for {}...": "正在清除 {} 的缓存...",
    "Refreshing cache view...": "正在刷新缓存视图...",
    "Using default parameters - enable Optuna or use parameter cache for optimized settings": "使用默认参数 - 启用 Optuna 或使用参数缓存以获得优化设置",
    "Force re-optimization enabled but Optuna is disabled - using default parameters": "强制重新优化已启用但 Optuna 已禁用 - 使用默认参数",
    
    # Tab labels
    "Summary": "摘要",
    "Predictions": "预测明细",
    "3D Grid": "三维网格",
    "Time Series": "时间序列",
    "Anchor 7-day": "锚定 7 天",
    "Uncertainty": "不确定性",
    "Extremes": "极值",
    "Debug": "调试",
    "Evaluation": "评估",
    "Forecast": "预测",
    
    # Data section
    "Verbose debug mode": "详细调试模式",
    "Upload your own data file": "上传您的数据文件",
    "Or pick a bundled sample dataset:": "或选择一个捆绑示例数据集：",
    "Sample dataset": "示例数据集",
    "-- Select sample --": "-- 选择示例 --",
    "Loading sample dataset: {}": "正在加载示例数据集：{}",
    "Sample dataset '{}' selected.": "已选择示例数据集'{}'。",
    "Raw Data": "原始数据",
    "Sorted Data": "已排序数据",
    
    # Location filter
    "Location Filter": "位置筛选器",
    "Warehouse": "仓库",
    "Silo": "筒仓",
    "All": "全部",
    
    # Training section
    "Train on uploaded CSV": "使用上传的 CSV 进行训练",
    "Model file": "模型文件",
    "Apply to all models": "应用到所有模型",
    "Evaluate": "评估",
    "Eval & Forecast": "评估并预测",
    
    # Status indicators
    "ACTIVE": "激活",
    "DISABLED": "禁用",
    "High": "高",
    "Medium": "中等",
    "Low": "低",
    "Sequential mode: Recommended for small trial counts or debugging": "顺序模式：推荐用于小试验次数或调试",
    
    # Messages and notifications
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
    "Model evaluation completed successfully.": "模型评估成功完成。",
    "Forecast generation completed successfully.": "预测生成成功完成。",
    "training completed (validation split contained no ground-truth targets).": "训练完成（验证分割不包含真实目标）。",
    "training completed. MAE: {}, RMSE: {}": "训练完成。MAE：{}，RMSE：{}",
    "Model training completed successfully": "模型训练成功完成",
    "Training completed successfully. MAE: {}, RMSE: {}": "训练成功完成。MAE：{}，RMSE：{}",
    
    # Optuna optimization
    "Starting Optuna hyperparameter optimization: {} trials": "开始 Optuna 超参数优化：{} 次试验",
    "Using {} parallel processes for enhanced optimization speed": "使用 {} 个并行进程以提高优化速度",
    "Live trial results will be displayed below as optimization progresses...": "优化过程中的实时试验结果将显示在下方...",
    "Live Optuna Trial Results": "实时 Optuna 试验结果",
    "Best Trial": "最佳试验",
    "Trial": "试验",
    "Learning Rate": "学习率",
    "Max Depth": "最大深度",
    "Num Leaves": "叶节点数",
    "Optuna Optimization Complete": "Optuna 优化完成",
    "Total Trials": "总试验数",
    "Improvement": "改进",
    "View All Trial Results": "查看所有试验结果",
    "Subsample": "子采样",
    "Colsample": "列采样",
    "Best Parameters Found": "找到的最佳参数",
    "Optuna optimization completed. Best MAE: {}{}": "Optuna 优化完成。最佳 MAE：{}{}",
    "Parallel optimization completed using {} processes": "使用 {} 个进程的并行优化完成",
    "Optuna optimization failed: {}": "Optuna 优化失败：{}",
    "Using cached optimal parameters!": "使用缓存的最优参数！",
    "Best Parameters Found: Learning Rate: {}, Max Depth: {}, Num Leaves: {}": "找到的最佳参数：学习率：{}，最大深度：{}，叶节点数：{}",
    "Optimal parameters saved to cache!": "最优参数已保存到缓存！",
    "Failed to save parameters to cache": "保存参数到缓存失败",
    
    # Model leaderboard
    "Model Leaderboard": "模型排行榜",
    "No evaluations yet.": "尚无评估结果。",
    "model": "模型",
    "confidence": "置信度",
    "accuracy": "准确率",
    "rmse": "RMSE",
    "mae": "MAE",
    "rank": "排名",
    
    # Debug
    "Debug Log (full)": "调试日志（完整）",
    
    # Evaluation metrics
    "Forecast Summary (per day)": "预测摘要（每日）",
    "Top Predictive Features": "最具预测力的特征",
    "Daily Extremes (h+1)": "每日极值 (h+1)",
    "No horizon-1 predictions available to compute extremes.": "没有可用于计算极值的 h+1 预测。",
    "Feature Matrices (first 100 rows)": "特征矩阵（前 100 行）",
    "Training – X_train": "训练 – X_train",
    "Evaluation – X_eval": "评估 – X_eval",
    "Model Feature Columns (order)": "模型特征列（顺序）",
    "High temperature forecast detected for at least one grain type – monitor closely!": "检测到某些粮食类型的高温预测 – 请密切监控！",
    "All predicted temperatures within safe limits for their grain types": "所有预测温度均在其粮食类型的安全范围内",
    "LightGBM uses early stopping; optimal number of trees will be selected automatically.": "LightGBM 使用提前停止；将自动选择最佳树数量。",
    
    # Metrics labels
    "Conf (%)": "置信度 (%)",
    "Acc (%)": "准确率 (%)",
    "MAE h+1": "MAE h+1",
    "MAE h+2": "MAE h+2",
    "MAE h+3": "MAE h+3",
    "RMSE h+1": "RMSE h+1",
    "RMSE h+2": "RMSE h+2",
    "RMSE h+3": "RMSE h+3",
    "MAPE h+1": "MAPE h+1",
    "Confidence (%)": "置信度 (%)",
    "Accuracy (%)": "准确率 (%)",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "MAPE (%)": "MAPE (%)",
    "Forecast horizon (h + ? days)": "预测范围 (h + ? 天)",
    "MAE per horizon": "各预测期 MAE",
    "RMSE per horizon": "各预测期 RMSE",
    "MAPE per horizon": "各预测期 MAPE",
    
    # Explanations
    "Row-wise horizon metrics (above) average the error of each h-day-ahead prediction across all evaluation rows.\nFor a real-world, bulletin-style view of performance, switch to the 'Anchor 7-day' tab, where metrics are computed by freezing predictions on an anchor day and comparing them with observations that occur h days later.": "上方按预测期汇总的指标是对评估集每一行的 h 日预测误差取平均。\n若想查看更贴近实际业务的表现，请切换到 \"Anchor 7-day\" 页签：在那里，指标在锚定日冻结预测，再与 h 天后的真实温度比较",
    "Anchor metrics emulate operational use: predictions are frozen on the selected anchor day (forecast_day = 1) and each horizon h is scored against the real temperature measured h days later.\nThis gives the most realistic estimate of future-forecast performance.": "Anchor 指标模拟实际操作流程：在锚定日（forecast_day = 1）冻结预测，并在 h 天后用真实观测温度打分。\n这能提供对未来预测性能的最真实估计。",
    
    # Uncertainty analysis
    "Uncertainty Analysis & Prediction Confidence": "不确定性分析和预测置信度",
    "Uncertainty Estimation": "不确定性估计",
    "Prediction Confidence Intervals": "预测置信区间",
    "Avg Uncertainty": "平均不确定性",
    "Max Uncertainty": "最大不确定性",
    "Min Uncertainty": "最小不确定性",
    "Reliability": "可靠性",
    "Uncertainty by Forecast Horizon": "按预测期的不确定性",
    "Horizon": "预测期",
    "Days Ahead": "预测天数",
    "Avg Uncertainty (°C)": "平均不确定性 (°C)",
    "Confidence Level": "置信水平",
    "Available Confidence Intervals": "可用置信区间",
    "Type": "类型",
    "Available": "可用",
    "Lower Bound": "下界",
    "Upper Bound": "上界",
    "How to Interpret Uncertainty": "如何解释不确定性",
    "Uncertainty intervals will be available after making predictions": "进行预测后将提供不确定性区间",
    "Could not load model for uncertainty analysis: {}": "无法加载模型进行不确定性分析：{}",
    "Please ensure the model is properly trained and saved.": "请确保模型已正确训练和保存。",
    
    # Forecasting
    "No forecast generated for this model yet.": "尚未为该模型生成预测。",
    "Forecast Summary (predicted)": "预测摘要（预测）",
    "Daily Predicted Extremes": "每日预测极值",
    "No predictions found to compute extremes.": "未找到用于计算极值的预测。",
    "Future Feature Matrix (first 100 rows)": "未来特征矩阵（前 100 行）",
    "|Mean(X_eval) − Mean(X_future)| (Top 20)": "|Mean(X_eval) − Mean(X_future)|（前 20）",
    "X_future matrix not available yet.": "X_future 矩阵尚不可用。",
    "Please evaluate the model first.": "请先评估模型。",
    "Unable to access base data or model for forecasting.": "无法访问基础数据或模型进行预测。",
    "Download predictions CSV": "下载预测 CSV",
    "Select day": "选择天数",
    
    # Anchor 7-day tab
    "7-Day Forecast from Anchor Day (forecast_day=1)": "锚定日的 7 天预测（forecast_day=1）",
    "forecast_day column missing – cannot compute anchor forecast.": "forecast_day 列缺失 – 无法计算锚定预测。",
    "No anchor dates available.": "没有可用的锚定日期。",
    "Select anchor date": "选择锚定日期",
    "No rows for selected anchor date {}.": "选定锚定日期 {} 没有行。",
    "horizon_day": "预测期天数",
    "forecast_date": "预测日期",
    "predicted_mean": "预测平均值",
    "actual_mean": "实际平均值",
    "max_abs_err": "最大绝对误差",
    "Anchor-day 7-Day Forecast vs Actual": "锚定日 7 天预测与实际对比",
    "Date": "日期",
    "Temperature (°C)": "温度 (°C)",
    "Predicted": "预测",
    "Actual": "实际",
    "Sensor-level discrepancies (> 0.5 °C)": "传感器级别差异（> 0.5 °C）",
    "h+{}: No matching sensor readings available.": "h+{}：没有匹配的传感器读数。",
    "h+{}: No sensor differences > {} °C.": "h+{}：没有传感器差异 > {} °C。",
    "Aggregate MAE Metrics Across All Anchors": "所有锚定点的聚合 MAE 指标",
    "Avg MAE (all anchors × 7 days)": "平均 MAE（所有锚定点 × 7 天）",
    "Max MAE (all anchors × 7 days)": "最大 MAE（所有锚定点 × 7 天）",
    "MAE by Forecast Horizon (All Anchors)": "按预测期的 MAE（所有锚定点）",
    "h+{} MAE": "h+{} MAE",
    "h+{} Max |Error|": "h+{} 最大 |误差|",
    
    # Extremes
    "Average Daily Absolute Error (h+1)": "每日平均绝对误差 (h+1)",
    "Over-Prediction (h+1)": "过预测 (h+1)",
    "Under-Prediction (h+1)": "欠预测 (h+1)",
    "Select date": "选择日期",
    "date": "日期",
    "type": "类型",
    "predicted": "预测",
    "actual": "实际",
    "error": "误差",
    "avg_daily_abs_error": "每日平均绝对误差",
    "Over": "过度",
    "Under": "不足",
    "Max": "最大",
    "Min": "最小",
    "Avg |Error|": "平均 |误差|",
    "Prediction column '{}' not found – unable to draw time-series for h+{}.": "未找到预测列 '{}' – 无法为 h+{} 绘制时间序列。",
    "Actual Avg": "实际平均值",
    "Predicted h+{} (eval)": "预测 h+{}（评估）",
    "Predicted h+{} (future)": "预测 h+{}（未来）",
    "Average Grain Temperature – h+{}": "平均粮食温度 – h+{}",
    "Xaxis": "X 轴",
    "Yaxis": "Y 轴",
    
    # Plot labels
    "Temp (°C)": "温度 (°C)",
    "Pred (°C)": "预测 (°C)",
    "Δ (°C)": "Δ (°C)",
    "grid_x": "网格 X",
    "grid_z": "网格 Z",
    "grid_y": "网格 Y",
    "Prediction Uncertainty by Forecast Horizon": "按预测期的预测不确定性",
    "Forecast Horizon": "预测期",
    "Average Uncertainty (°C)": "平均不确定性 (°C)",
    
    # Conservative system detailed messages
    "Conservative Temperature System": "保守温度系统",
    "Models temperature resistance to change": "模型温度变化阻力",
    "Measures temperature consistency": "测量温度一致性",
    "Quantifies fluctuation dampening": "量化波动阻尼",
    "Natural settling point per sensor": "每个传感器的自然平衡点",
    "Tendency to return to equilibrium": "回归平衡的趋势",
    "Long-term stability patterns": "长期稳定模式",
    "Dynamic change dampening": "动态变化阻尼",
    "Increasing constraints for longer horizons": "对较长预测期的递增约束",
    "Reduces aggressive changes": "减少激进变化",
    "Improves 7-day accuracy": "提高 7 天准确性",
    "Sensor-specific learning": "传感器特定学习",
    "Physical realism": "物理现实性",
    "Predictions respect thermal inertia": "预测尊重热惯性",
    "Less cumulative error buildup": "更少的累积误差堆积",
    "Each probe learns its stability characteristics": "每个探针学习其稳定特性",
    "Temperature evolution follows grain physics": "温度演变遵循粮食物理学",
    "Conservative mode disabled - predictions may be aggressive": "保守模式禁用 - 预测可能激进",
    "Enable conservative mode in model training to get more stable predictions": "在模型训练中启用保守模式以获得更稳定的预测",
    
    # More uncertainty messages
    "Low uncertainty": "低不确定性",
    "High confidence predictions": "高置信度预测",
    "Medium uncertainty": "中等不确定性",
    "Moderate confidence": "中等置信度",
    "High uncertainty": "高不确定性",
    "Lower confidence, use with caution": "低置信度，谨慎使用",
    "~2/3 of actual values should fall within this range": "约 2/3 的实际值应在此范围内",
    "~19/20 of actual values should fall within this range": "约 19/20 的实际值应在此范围内",
    
    # Additional status messages
    "System Configuration:": "系统配置：",
    "Stability Features": "稳定性特征",
    "8 thermal physics features": "8 个热物理特征",
    "Stability Boost": "稳定性增强",
    "3.0x feature importance": "3.0 倍特征重要性",
    "Uncertainty Samples": "不确定性样本",
    "50 bootstrap samples": "50 个自助采样",
    "Directional Features": "方向性特征",
    "2.0x importance boost": "2.0 倍重要性增强",
    "Horizon Balancing": "预测期平衡",
    "Enabled": "已启用",
    "Conservative Loss": "保守损失",
    "Thermal inertia penalties": "热惯性惩罚",
    "Expected Benefits:": "预期收益：",
    "More stable predictions": "更稳定的预测",
    "Respects grain thermal inertia": "尊重粮食热惯性",
    "Reduced aggressive changes": "减少激进变化",
    "Conservative temperature evolution": "保守的温度演变",
    "Better 7-day accuracy": "更好的 7 天准确性",
    "Less cumulative error buildup": "更少的累积误差堆积",
    "Uncertainty quantification": "不确定性量化",
    "Confidence intervals for all predictions": "所有预测的置信区间",
    "Conservative system initialized - thermal stability features active": "保守系统已初始化 - 热稳定性特征激活",
    
    # System status messages
    "System": "系统",
    "ACTIVE": "激活",
    "DISABLED": "禁用",
    "Conservative System Status": "保守系统状态",
    "Conservative Temperature Prediction": "保守温度预测",
    "Quantile Regression Active": "分位数回归激活",
    "Using quantile objective with uncertainty quantification for enhanced 7-day forecasting accuracy.": "使用分位数目标和不确定性量化以增强 7 天预测准确性。",
    
    # Missing translations for recent updates
    "Optuna configured for {} parallel processes": "Optuna 配置为 {} 个并行进程",
    "Approximately {}x faster optimization": "优化速度约提高 {} 倍",
    "Recommended for": "推荐用于",
    "trials with 4+ CPU cores available": "需要 4+ CPU 核心的试验",
    "sets available": "套可用",
    "Later horizons (H+7) will receive 4x more weight than earlier horizons (H+1)": "后期预测期（H+7）将比前期预测期（H+1）获得 4 倍权重",
    "Earlier horizons (H+1) will receive 4x more weight than later horizons (H+7)": "前期预测期（H+1）将比后期预测期（H+7）获得 4 倍权重",
    "All horizons will receive equal weight (1.0x each)": "所有预测期将获得相等权重（各 1.0 倍）",
    
    # Comprehensive missing translations
    "Parameter Cache": "参数缓存",
    "Use parameter cache": "使用参数缓存",
    "Automatically save/load optimal parameters to skip redundant Optuna optimization": "自动保存/加载最优参数以跳过冗余的 Optuna 优化",
    "Force re-optimization": "强制重新优化",
    "Run Optuna even if cached parameters exist": "即使存在缓存参数也运行 Optuna",
    "Clear cache": "清除缓存",
    "Clear all cached parameters": "清除所有缓存参数",
    "Clearing parameter cache...": "正在清除参数缓存...",
    "Parameter cache cleared!": "参数缓存已清除！",
    "Cached parameters": "缓存参数",
    "Future-safe mode enabled - environmental variables excluded": "未来安全模式已启用 - 排除环境变量",
    "All variables included, including environmental data": "包含所有变量，包括环境数据",
    "Anchor-day early stopping": "锚日早停法",
    "Use 7-day consecutive forecasting accuracy for early stopping with optimized interval checking": "使用7天连续预测准确性进行早停，并采用优化的间隔检查",
    "Anchor-day early stopping enabled for enhanced 7-day accuracy": "锚日早停法已启用，以增强7天准确性",
    "Using standard early stopping method": "使用标准早停方法",
    "Horizon Balancing Configuration": "预测期平衡配置",
    "Balance horizon training": "平衡预测期训练",
    "Ensures equal priority for all forecast horizons (H+1 through H+7) during model training": "确保模型训练期间所有预测期（H+1到H+7）具有相等的优先级",
    "Horizon balancing enabled - correcting forecast horizon bias": "预测期平衡已启用 - 纠正预测期偏差",
    "Using standard horizon weighting approach": "使用标准预测期权重方法",
    "Horizon weighting strategy": "预测期权重策略",
    "equal": "相等",
    "increasing": "递增",
    "decreasing": "递减",
    "equal: All horizons get equal priority (recommended) | increasing: Later horizons get more weight | decreasing: Earlier horizons get more weight": "相等：所有预测期获得相等优先级（推荐）| 递增：后期预测期获得更多权重 | 递减：前期预测期获得更多权重",
    "Performance Optimization": "性能优化",
    "Parallel Processing": "并行处理",
    "Active": "激活",
    "Workers": "工作器",
    "Expected Speedup": "预期加速",
    "3-5x faster feature engineering": "特征工程快3-5倍",
    "2-4x faster hyperparameter optimization": "超参数优化快2-4倍",
    "Enable All Cores": "启用所有核心",
    "Disable Parallel": "禁用并行",
    "Disabled": "已禁用",
    "Enable Parallel Processing": "启用并行处理",
    "Could not retrieve parallel processing information: {}": "无法获取并行处理信息：{}",
    "Train on uploaded CSV": "在上传的CSV上训练",
    "Initiating model training process...": "正在启动模型训练过程...",
    "Training model – please wait...": "正在训练模型 - 请稍候...",
    "Using cached optimal parameters!": "使用缓存的最优参数！",
    "Best MAE": "最佳 MAE",
    "Cached": "已缓存",
    "Starting Optuna hyperparameter optimization: {} trials": "开始 Optuna 超参数优化：{} 次试验",
    "Using {} parallel processes for enhanced optimization speed": "使用 {} 个并行进程以提高优化速度",
    "Live trial results will be displayed below as optimization progresses...": "优化进行时下方将显示实时试验结果...",
    
    # Conservative system translations
    "Conservative System Status": "保守系统状态",
    "Conservative Temperature Prediction": "保守温度预测",
    "System Configuration": "系统配置",
    "Stability Features": "稳定性特征",
    "8 thermal physics features": "8 个热物理特征",
    "Stability Boost": "稳定性增强",
    "3.0x feature importance": "3.0 倍特征重要性",
    "Uncertainty Samples": "不确定性样本",
    "50 bootstrap samples": "50 个自举样本",
    "Directional Features": "方向性特征",
    "2.0x importance boost": "2.0 倍重要性增强",
    "Horizon Balancing": "预测期平衡",
    "Enabled": "已启用",
    "Conservative Loss": "保守损失",
    "Thermal inertia penalties": "热惯性惩罚",
    "Expected Benefits": "预期效益",
    "More stable predictions": "更稳定的预测",
    "Respects grain thermal inertia": "尊重粮食热惯性",
    "Reduced aggressive changes": "减少激进变化",
    "Conservative temperature evolution": "保守温度演化",
    "Better 7-day accuracy": "更好的7天准确性",
    "Less cumulative error buildup": "更少的累积误差积累",
    "Uncertainty quantification": "不确定性量化",
    "Confidence intervals for all predictions": "所有预测的置信区间",
    
    # Additional cache management translations
    "Cached parameter sets": "缓存参数集",
    "Data shape": "数据形状",
    "Trials": "试验",
    "Clear this": "清除此项",
    "Clearing cache for {}...": "正在清除 {} 的缓存...",
    "Clear all cache": "清除所有缓存",
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
    """Load uploaded file with automatic format detection (CSV or Parquet)."""
    try:
        # Check file extension to determine format
        file_extension = pathlib.Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.parquet':
            # Load Parquet file
            _d(f"[LOAD] Reading Parquet file: {uploaded_file.name}")
            df = pd.read_parquet(uploaded_file)
        elif file_extension in ['.csv', '.gz', '.gzip', '.bz2', '.zip', '.xz']:
            # Load CSV file (including compressed)
            _d(f"[LOAD] Reading data file: {uploaded_file.name}")
            df = pd.read_csv(uploaded_file)
        else:
            # Try to read as CSV (fallback)
            _d(f"[LOAD] Attempting to read as CSV: {uploaded_file.name}")
            df = pd.read_csv(uploaded_file)
        
        _d(f"[LOAD] Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
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
        try:
            return model_utils.load_model(path)
        except (FileNotFoundError, ValueError) as exc:
            # Show the issue to the user but keep the app running gracefully
            st.error(str(exc))
            return None

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
        # Set midpoint to 0 for delta mode so 0 is white
        colorbar_midpoint = 0
    else:
        color_vals = df["predicted_temp"] if has_pred else df["temperature_grain"]
        c_scale = "RdBu_r"  # red = hot, blue = cold
        cbar_title = "Temp (°C)" if not color_by_delta else "Pred (°C)"
        # Calculate midpoint to make 0 white for temperature mode
        min_val = color_vals.min()
        max_val = color_vals.max()
        if min_val < 0 and max_val > 0:
            # If we have both positive and negative values, set midpoint to 0
            colorbar_midpoint = 0
        else:
            # If all values are on one side of 0, use the middle of the range
            colorbar_midpoint = (min_val + max_val) / 2

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
            cmid=colorbar_midpoint,  # Set the midpoint for the color scale
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


def plot_time_series(
    df: pd.DataFrame,
    *,
    key: str,
    horizon_day: int = 1,
):
    """Plot actual vs predicted *average* grain temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated dataframe that may contain historic rows, evaluation rows
        and/or future forecast rows.
    key : str
        Streamlit component key.
    horizon_day : int, default ``1``
        Forecast horizon to visualise (1-7).  ``1`` corresponds to the legacy
        behaviour that uses the *predicted_temp* column.  For *horizon_day > 1*
        the function looks for a column called ``pred_h{horizon_day}d``.
    """

    if "detection_time" not in df.columns:
        return  # nothing to plot

    # Determine which prediction column to use --------------------------
    pred_col = "predicted_temp" if horizon_day == 1 else f"pred_h{horizon_day}d"
    if pred_col not in df.columns:
        # Gracefully exit if the requested horizon is unavailable
        st.info(f"Prediction column '{pred_col}' not found – unable to draw time-series for h+{horizon_day}.")
        return

    tmp = df.copy()
    # Use floor("D") to keep datetime64 dtype; avoids Plotly treating axis as categorical
    tmp["date"] = pd.to_datetime(tmp["detection_time"]).dt.floor("D")

    fig = go.Figure()

    # -------- Actual line ---------------------------------------------
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

    # ------- Predicted line (continuous across eval + future) ----------
    grp_pred = (
        tmp.groupby("date").agg(predicted=(pred_col, "mean")).reset_index().sort_values("date")
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
                name=f"Predicted h+{horizon_day} (eval)",
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
                name=f"Predicted h+{horizon_day} (future)",
                line=dict(color="#9467bd"),
                connectgaps=True,
            )
        )

    fig.update_layout(
        title=f"Average Grain Temperature – h+{horizon_day}",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def list_available_models() -> list[str]:
    """Return unique model filenames from user-saved and preloaded dirs."""
    # Look for both uncompressed and compressed model files
    names = {p.name for p in MODELS_DIR.glob("*.joblib")}
    names.update({p.name for p in MODELS_DIR.glob("*.joblib.gz")})
    names.update({p.name for p in PRELOADED_MODEL_DIR.glob("*.joblib")})
    names.update({p.name for p in PRELOADED_MODEL_DIR.glob("*.joblib.gz")})
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
    df["_date"] = pd.to_datetime(df["detection_time"], errors="coerce").dt.date
    unique_dates = sorted(df["_date"].dropna().unique())
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

    # Pre-compute per-row absolute and squared errors to speed up group aggregations
    df_eval = df_eval.copy()
    if {TARGET_TEMP_COL, "predicted_temp"}.issubset(df_eval.columns):
        df_eval["abs_err"] = (df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]).abs()
        df_eval["sqr_err"] = (df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]) ** 2

    grp = (
        df_eval.groupby("forecast_day")
        .agg(
            actual_mean=(TARGET_TEMP_COL, "mean"),
            pred_mean=("predicted_temp", "mean"),
            mae=("abs_err", "mean"),
            rmse=("sqr_err", lambda s: np.sqrt(np.nanmean(s))),
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
        grp = grp[["date", "actual_mean", "pred_mean", "mae", "rmse"]]
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
    
    # Verify parallel processing is available
    try:
        parallel_info = features.get_parallel_info()
        _d(f"[PARALLEL] Parallel processing available: {parallel_info['parallel_enabled']}, max_workers: {parallel_info['max_workers']}, CPU cores: {parallel_info['cpu_count']}")
        if st.session_state.get("debug_mode"):
            st.toast(f"🔧 Parallel processing: {parallel_info['max_workers']} workers on {parallel_info['cpu_count']} cores", icon="🔧")
    except Exception as e:
        _d(f"[PARALLEL] Could not verify parallel processing: {e}")

    # ---------------- Language selector (appears very top) -------------
    def on_language_change():
        new_lang = "en" if st.session_state.get("lang_selector") == "English" else "zh"
        old_lang = st.session_state.get("lang", "en")
        if new_lang != old_lang:
            st.toast(f"🌐 Language changed to {'English' if new_lang == 'en' else '中文'}", icon="🌐")
        st.session_state.update({"lang": new_lang})
    
    st.sidebar.selectbox(
        "Language / 语言",
        options=["English", "中文"],
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
        key="lang_selector",
        on_change=on_language_change,
    )
    # Ensure lang key present
    st.session_state.setdefault("lang", "en")

    with st.sidebar.expander(_t('Data'), expanded=("uploaded_file" not in st.session_state)):
        uploaded_file = st.file_uploader(_t("Upload your own data file"), type=["csv", "parquet"], key="uploader")

        # ------------------------------------------------------------------
        # Offer bundled sample datasets so users can start instantly
        if PRELOADED_DATA_DIR.exists():
            # Include both CSV and Parquet files
            sample_files = sorted(list(PRELOADED_DATA_DIR.glob("*.csv")) + list(PRELOADED_DATA_DIR.glob("*.parquet")))
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
                    st.toast(f"📂 Loading sample dataset: {sample_choice}", icon="📂")
                    st.info(f"Sample dataset '{sample_choice}' selected.")

    if uploaded_file:
        # Show notification for file upload
        if hasattr(uploaded_file, 'name'):
            st.toast(f"Processing uploaded file: {uploaded_file.name}")
        else:
            st.toast("Processing uploaded file...")
        
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
        st.toast(f"Data preprocessing complete. Dataset shape: {df.shape}")
        _d(f"[DATA] Preprocessing complete – shape={df.shape} cols={list(df.columns)[:10]}…")

        # --- Save preprocessed data as Parquet (always) ---
        import os
        from granarypredict.ingestion import save_granary_data
        base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_path = os.path.join(processed_dir, f"{base_name}_processed.parquet")
        save_granary_data(df, processed_path, format="parquet", compression="snappy")
        _d(f"[DATA] Saved preprocessed data as Parquet: {processed_path}")

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
            old_filters = st.session_state.get("global_filters", {})
            new_filters = {
                "wh": sel_wh_global,
                "silo": sel_silo_global,
            }
            
            # Show notification if filters changed
            if old_filters != new_filters:
                if sel_wh_global != "All" or sel_silo_global != "All":
                    filter_text = f"Warehouse: {sel_wh_global}"
                    if sel_silo_global != "All":
                        filter_text += f", Silo: {sel_silo_global}"
                    st.toast(f"Applied filters: {filter_text}")
                else:
                    st.toast("Displaying all locations")
            
            st.session_state["global_filters"] = new_filters

        with st.sidebar.expander(_t('Train / Retrain Model'), expanded=False):
            model_choice = st.selectbox(
                _t("Algorithm"),
                ["LightGBM"],
                index=0,
            )
            # Initialize LightGBM-specific variables outside conditional
            tune_optuna = False
            use_quantile = False
            n_trials = 0
            optuna_speed_mode = False  # Initialize to prevent UnboundLocalError
            optuna_parallel = False  # Initialize to prevent UnboundLocalError
            optuna_n_jobs = 1  # Initialize to prevent UnboundLocalError
            
            if model_choice == "LightGBM":
                st.caption(_t("LightGBM uses early stopping; optimal number of trees will be selected automatically."))
                n_trees = 2000  # upper bound (not shown to user)
                tune_optuna = st.checkbox(_t("Optuna hyperparameter optimization"), value=False, help=_t("Enable Optuna to automatically tune LightGBM parameters for optimal performance"))
                if tune_optuna and "optuna_enabled" not in st.session_state:
                    st.toast(_t("Optuna hyperparameter optimization enabled"))
                    st.session_state["optuna_enabled"] = True
                elif not tune_optuna and st.session_state.get("optuna_enabled"):
                    st.toast(_t("Using default LightGBM parameters"))
                    st.session_state["optuna_enabled"] = False
                
                use_quantile = st.checkbox(_t("Quantile regression objective"), value=True, help=_t("Use LightGBM quantile regression (alpha 0.5) for improved mean absolute error performance"))
                if use_quantile:
                    st.info(f"**{_t('Quantile Regression Active')}**: {_t('Using quantile objective with uncertainty quantification for enhanced 7-day forecasting accuracy.')}")
                if use_quantile and "quantile_enabled" not in st.session_state:
                    st.toast(_t("Quantile regression objective enabled"))
                    st.session_state["quantile_enabled"] = True
                elif not use_quantile and st.session_state.get("quantile_enabled"):
                    st.toast(_t("Using standard regression objective"))
                    st.session_state["quantile_enabled"] = False
                
                # Parameter cache controls (always available)
                st.subheader(f"📦 {_t('Parameter Cache')}")
                use_param_cache = st.checkbox(
                    _t("Use parameter cache"), 
                    value=True, 
                    help=_t("Automatically save/load optimal parameters from previous Optuna runs. Works even when Optuna is disabled!")
                )
                    
                col_cache1, col_cache2 = st.columns(2)
                with col_cache1:
                    force_reoptimize = st.checkbox(
                        _t("Force re-optimization"), 
                        value=False, 
                        help=_t("Run Optuna even if cached parameters exist")
                    )
                with col_cache2:
                    if st.button(_t("Clear cache"), help=_t("Clear all cached parameters")):
                        st.toast(f"🧽 {_t('Clearing parameter cache...')}", icon="🧽")
                        clear_cache()
                        st.success(_t("Parameter cache cleared!"))
                
                # Show cached parameter info with debugging
                cached_params = list_cached_params()
                if cached_params:
                    st.write(f"📋 **{_t('Cached parameters')}**: {len(cached_params)} {_t('sets available')}")
                    
                    # Debug: Show cache details for current CSV
                    if st.checkbox("🔍 Show cache debugging info", value=False):
                        csv_filename = uploaded_file.name if uploaded_file else "No file"
                        st.write(f"**Debug Info for {csv_filename}:**")
                        
                        # Show all cached files
                        for key, info in cached_params.items():
                            cached_csv = info.get('csv_filename', 'Unknown')
                            cached_mae = info.get('best_value', 'N/A')
                            cached_trials = info.get('n_trials', 'N/A')
                            cached_shape = info.get('data_shape', 'N/A')
                            
                            match_indicator = "✅" if cached_csv == csv_filename else "❌"
                            st.write(f"{match_indicator} **{cached_csv}** - MAE: {cached_mae}, Trials: {cached_trials}, Shape: {cached_shape}")
                        
                        # Show current configuration
                        if uploaded_file:
                            current_config = {
                                "quantile_mode": use_quantile,
                                "speed_mode": optuna_speed_mode if tune_optuna else "N/A",
                                "n_trials": n_trials if tune_optuna else "N/A",
                                "cache_enabled": use_param_cache,
                                "force_reopt": force_reoptimize
                            }
                            st.write(f"**Current Config**: {current_config}")
                
                if tune_optuna:
                    n_trials = st.slider(_t("Optuna trials"), 20, 200, 50, step=10)
                    optuna_speed_mode = st.checkbox(
                        _t("Fast Optuna mode"), 
                        value=True, 
                        help=_t("Use performance optimizations: 2-fold CV, lower tree limits, aggressive early stopping")
                    )
                    
                    # NEW: GPU Acceleration Settings (moved before parallel settings)
                    st.markdown(f"**{_t('GPU Acceleration')}**")
                    
                    # Check GPU availability first
                    try:
                        from granarypredict.multi_lgbm import detect_gpu_availability
                        gpu_config = detect_gpu_availability()
                        gpu_available = gpu_config['available']
                        
                        if gpu_available:
                            st.success(f"✅ {_t('GPU detected and available for acceleration')}")
                        else:
                            st.warning(f"{_t('No GPU detected - will use CPU acceleration')}")
                    except Exception as e:
                        st.warning(f"{_t('Could not detect GPU availability')}: {str(e)[:50]}...")
                        gpu_available = False
                    
                    use_gpu_optuna = st.checkbox(
                        _t("Enable GPU acceleration"),
                        value=gpu_available,  # Default to True only if GPU is available
                        help=_t("Use GPU acceleration for faster LightGBM training during Optuna optimization"),
                        disabled=not gpu_available  # Disable if no GPU available
                    )
                    
                    if use_gpu_optuna and gpu_available:
                        gpu_optimization_mode = st.selectbox(
                            _t("GPU Optimization Mode"),
                            ["auto", "balanced", "speed", "accuracy"],
                            help=_t("Auto: Smart optimization based on dataset size. Balanced: Speed/accuracy trade-off. Speed: Maximum speed. Accuracy: Maximum precision.")
                        )
                        
                        gpu_device_id = st.slider(
                            _t("GPU Device ID"),
                            0, 3, 0, step=1,
                            help=_t("Select which GPU device to use (0 = first GPU)")
                        )
                        
                        gpu_use_double_precision = st.checkbox(
                            _t("Use double precision"),
                            value=True,
                            help=_t("Use double precision for better accuracy (slower but more precise)")
                        )
                    else:
                        gpu_optimization_mode = "cpu"
                        gpu_device_id = 0
                        gpu_use_double_precision = True
                        if not gpu_available:
                            st.info(f"💻 {_t('Using CPU acceleration - no GPU available')}")
                    
                    # Parallel Optuna optimization settings
                    st.markdown(f"**⚡ {_t('Parallel Optimization')}**")
                    optuna_parallel = st.checkbox(
                        _t("Enable parallel trials"),
                        value=False,
                        help=_t("Run multiple Optuna trials in parallel for 2-4x faster optimization")
                    )
                    
                    if optuna_parallel:
                        # NEW: Smart parallel process recommendation based on GPU usage
                        if use_gpu_optuna and gpu_available:
                            # With GPU: Can use more parallel processes since GPU handles model training
                            max_processes = min(8, multiprocessing.cpu_count())
                            default_processes = min(4, multiprocessing.cpu_count())
                            help_text = _t("With GPU acceleration: Can use more parallel processes (GPU handles model training)")
                        else:
                            # Without GPU: Be more conservative since CPU cores are shared with LightGBM
                            max_processes = min(4, multiprocessing.cpu_count())
                            default_processes = min(2, multiprocessing.cpu_count())
                            help_text = _t("Without GPU: Conservative setting to avoid overwhelming CPU cores")
                        
                        optuna_n_jobs = st.slider(
                            _t("Number of parallel processes"), 
                            2, max_processes, default_processes, step=1,
                            help=help_text
                        )
                        
                        # NEW: Smart LightGBM threading configuration
                        if use_gpu_optuna and gpu_available:
                            # GPU mode: Use fewer CPU threads per model to free up cores for Optuna
                            lgbm_n_jobs = st.selectbox(
                                _t("LightGBM CPU threads per model"),
                                [1, 2, 4],
                                index=1,  # Default to 2
                                help=_t("With GPU: Use fewer CPU threads to free up cores for parallel Optuna trials")
                            )
                        else:
                            # CPU mode: Use all available cores for each model
                            lgbm_n_jobs = st.selectbox(
                                _t("LightGBM CPU threads per model"),
                                [1, 2, 4, -1],
                                index=3,  # Default to -1 (all cores)
                                help=_t("Without GPU: Use all CPU cores for maximum model training speed")
                            )
                        
                        # Performance expectations
                        if use_gpu_optuna and gpu_available:
                            expected_speedup = min(optuna_n_jobs * 2, 8)  # GPU provides additional speedup
                            st.caption(f"**{_t('Expected Speedup')}**: {_t('Approximately {}x faster optimization').format(expected_speedup)}")
                            st.caption(f"**{_t('GPU + Parallel')}**: {_t('GPU acceleration + {} parallel processes').format(optuna_n_jobs)}")
                        else:
                            expected_speedup = min(optuna_n_jobs, 4)
                            st.caption(f"**{_t('Expected Speedup')}**: {_t('Approximately {}x faster optimization').format(expected_speedup)}")
                            st.caption(f"**{_t('Recommended for')}**: {n_trials}+ {_t('trials with 4+ CPU cores available')}")
                    else:
                        optuna_n_jobs = 1
                        lgbm_n_jobs = -1  # Use all cores for single-threaded Optuna
                        st.caption(_t("Sequential mode: Recommended for small trial counts or debugging"))
                    
                    
            else:
                n_trees = st.slider(_t("Iterations / Trees"), 100, 1000, 300, step=100)
                optuna_speed_mode = False  # Default when Optuna is disabled
                optuna_parallel = False  # Default when Optuna is disabled
                optuna_n_jobs = 1  # Default when Optuna is disabled
                n_trials = 50  # Default when Optuna is disabled
            future_safe = st.checkbox(_t("Future-safe (exclude env vars)"), value=False)
            if future_safe and "future_safe_enabled" not in st.session_state:
                st.toast(_t("Future-safe mode enabled - environmental variables excluded"))
                st.session_state["future_safe_enabled"] = True
            elif not future_safe and st.session_state.get("future_safe_enabled"):
                st.toast(_t("All variables included, including environmental data"))
                st.session_state["future_safe_enabled"] = False
            
            anchor_early_stop = st.checkbox(
                _t("Anchor-day early stopping"), 
                value=True, 
                help=_t("Use 7-day consecutive forecasting accuracy for early stopping with optimized interval checking")
            )
            if anchor_early_stop and "anchor_stop_enabled" not in st.session_state:
                st.toast(_t("Anchor-day early stopping enabled for enhanced 7-day accuracy"))
                st.session_state["anchor_stop_enabled"] = True
            elif not anchor_early_stop and st.session_state.get("anchor_stop_enabled"):
                st.toast(_t("Using standard early stopping method"))
                st.session_state["anchor_stop_enabled"] = False
            
            # NEW: Horizon balancing to fix H+1 > H+7 bias
            st.subheader(_t("Horizon Balancing Configuration"))
            balance_horizons = st.checkbox(
                _t("Balance horizon training"), 
                value=True, 
                help=_t("Ensures equal priority for all forecast horizons (H+1 through H+7) during model training")
            )
            if balance_horizons and "horizon_balance_enabled" not in st.session_state:
                st.toast(_t("Horizon balancing enabled - correcting forecast horizon bias"))
                st.session_state["horizon_balance_enabled"] = True
            elif not balance_horizons and st.session_state.get("horizon_balance_enabled"):
                st.toast(_t("Using standard horizon weighting approach"))
                st.session_state["horizon_balance_enabled"] = False
            
            horizon_strategy = st.selectbox(
                _t("Horizon weighting strategy"),
                [_t("equal"), _t("increasing"), _t("decreasing")],
                index=0,
                help=_t("equal: All horizons get equal priority (recommended) | increasing: Later horizons get more weight | decreasing: Earlier horizons get more weight")
            )
            
            # Show notification when strategy changes
            if horizon_strategy != st.session_state.get("last_horizon_strategy", _t("equal")):
                if horizon_strategy == _t("increasing"):
                    st.toast(_t("Later horizons (H+7) will receive 4x more weight than earlier horizons (H+1)"))
                elif horizon_strategy == _t("decreasing"):
                    st.toast(_t("Earlier horizons (H+1) will receive 4x more weight than later horizons (H+7)"))
                else:
                    st.toast(_t("All horizons will receive equal weight (1.0x each)"))
                st.session_state["last_horizon_strategy"] = horizon_strategy

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
            if train_pressed:
                st.toast(_t("Initiating model training process..."))

        if train_pressed and uploaded_file:
            with st.spinner(_t("Training model – please wait...")):
                if 'use_gpu_optuna' not in locals():
                    use_gpu_optuna = False
                if 'gpu_available' not in locals():
                    gpu_available = False
                if 'gpu_optimization_mode' not in locals():
                    gpu_optimization_mode = "cpu"
                if 'gpu_device_id' not in locals():
                    gpu_device_id = 0
                if 'gpu_use_double_precision' not in locals():
                    gpu_use_double_precision = True
                if 'lgbm_n_jobs' not in locals():
                    lgbm_n_jobs = -1
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
                    _d(f"[TARGETS] Multi-horizon targets shape: y_tr={y_tr.shape}, y_te={y_te.shape}, horizons={HORIZON_TUPLE}")
                    _d(f"[TARGETS] Target columns: {list(y_tr.columns) if hasattr(y_tr, 'columns') else 'single column'}")
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

                # Default starting parameters (moved outside conditional) - Using optimal parameters
                base_params = dict(
                    learning_rate=0.07172794499286328,  # Optimal parameter
                    max_depth=20,                       # Optimal parameter
                    num_leaves=133,                     # Optimal parameter
                    subsample=0.8901667731353657,       # Optimal parameter
                    colsample_bytree=0.7729605909501445, # Optimal parameter
                    min_child_samples=102,              # Optimal parameter
                    lambda_l1=1.4182488012070926,       # Optimal parameter
                    lambda_l2=1.7110926238653472,       # Optimal parameter
                    max_bin=416,                        # Optimal parameter
                    n_jobs=lgbm_n_jobs if 'lgbm_n_jobs' in locals() else -1,  # NEW: Use configured threading
                    # 🆕 COMPRESSION OPTIMIZATIONS (40-70% smaller files, no accuracy loss)
                    compress=True,                      # Enable built-in compression
                    compression_level=6,                # Compression level (1-9, higher = smaller)
                    save_binary=True,                   # Save in binary format (smaller)
                )

                # Apply quantile objective if selected
                if use_quantile:
                    base_params.update({
                        "objective": "quantile",
                        "alpha": 0.5,
                    })

                # Initialize variables that will be used later
                use_wrapper = False
                suffix = "lgbm"
                base_mdl = None

                if model_choice == "LightGBM":
                    # --------- Always check for cached parameters first ------------------------
                    csv_filename = uploaded_file.name
                    _d(f"[MODEL-CONFIG] LightGBM selected, tune_optuna: {tune_optuna}")
                    _d(f"[CACHE-DEBUG] Checking cache for CSV: {csv_filename}")
                    _d(f"[CACHE-DEBUG] use_param_cache: {use_param_cache}")
                    _d(f"[CACHE-DEBUG] force_reoptimize: {force_reoptimize}")
                    
                    # Create model configuration for cache lookup
                    model_config = {
                        "model_type": "LightGBM",
                        "future_safe": future_safe,
                        "use_quantile": use_quantile,
                        "balance_horizons": balance_horizons,
                        "horizon_strategy": horizon_strategy,  # Include horizon strategy
                        "anchor_early_stop": anchor_early_stop,
                        "optuna_speed_mode": optuna_speed_mode,
                        "optuna_parallel": optuna_parallel,
                        "optuna_n_jobs": optuna_n_jobs if optuna_parallel else 1,
                        "train_split": "last30" if use_last_30 else f"pct{train_pct}",
                        "n_trials": n_trials,  # Include number of trials
                        "conservative_mode": True,  # Always enabled
                        "uncertainty_estimation": True,  # Always enabled
                        "stability_feature_boost": 3.0,  # Include feature boost settings
                        "directional_feature_boost": 2.0,
                        "n_trees": n_trees,  # Include tree count
                        "data_rows": len(df),  # Include data size for better cache validation
                        "horizons": HORIZON_TUPLE  # Include forecast horizons
                    }
                    
                    # Check for cached parameters (regardless of Optuna setting)
                    cached_result = None
                    if use_param_cache and not force_reoptimize:
                        _d("[CACHE] Checking for cached optimal parameters...")
                        _d(f"[CACHE] Data filename: {csv_filename}")
                        _d(f"[CACHE] Model config keys: {list(model_config.keys())}")
                        _d(f"[CACHE] Model config: {model_config}")
                        cached_result = load_optimal_params(csv_filename, df, model_config)
                        _d(f"[CACHE] Cache loading result: {cached_result is not None}")
                        if cached_result:
                            best_params, best_value, timestamp = cached_result
                            _d(f"[CACHE] Found cached params: {best_params}")
                            _d(f"[CACHE] Best value: {best_value}, timestamp: {timestamp}")
                    else:
                        _d(f"[CACHE] Skipping cache check - use_param_cache: {use_param_cache}, force_reoptimize: {force_reoptimize}")
                    
                    # Use cached parameters if available
                    if cached_result and not force_reoptimize:
                        # Use cached parameters
                        best_params, best_value, timestamp = cached_result
                        base_params.update(best_params)
                        
                        # Keep quantile objective if it was selected
                        if use_quantile:
                            base_params.update({
                                "objective": "quantile",
                                "alpha": 0.5,
                            })
                        
                        _d(f"[CACHE-HIT] Using cached optimal parameters from {timestamp}")
                        _d(f"[CACHE-HIT] Best cached anchor-7d MAE: {best_value:.4f}")
                        _d(f"[CACHE-HIT] Cached parameters: {best_params}")
                        
                        st.success(f"📦 **{_t('Using cached optimal parameters!')}**\n{_t('Best MAE')}: {best_value:.4f} | {_t('Cached')}: {timestamp}")
                        
                        # Enhanced cached parameters display
                        with st.expander("🔍 **Cached Parameters Details**", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Optimal Parameters:**")
                                for param, value in best_params.items():
                                    if isinstance(value, float):
                                        st.write(f"- **{param}**: {value:.4f}")
                                    else:
                                        st.write(f"- **{param}**: {value}")
                                        
                            with col2:
                                st.write("**Cache Information:**")
                                st.write(f"- **Best MAE**: {best_value:.4f}")
                                st.write(f"- **Cached**: {timestamp}")
                                st.write(f"- **Data File**: {csv_filename}")
                                st.write(f"- **Data Shape**: {df.shape}")
                                st.write(f"- **Speed Mode**: {optuna_speed_mode}")
                                st.write(f"- **Trials**: {n_trials}")
                                
                        st.toast(f"📦 Loaded cached parameters: MAE {best_value:.4f}", icon="📦")
                        
                    # If no cached parameters found, decide whether to run Optuna or use defaults
                    elif tune_optuna:
                            # Run Optuna optimization
                            try:
                                import optuna
                                
                                # Provide detailed cache miss information
                                if force_reoptimize:
                                    _d(f"[CACHE-SKIP] Force re-optimization enabled for {csv_filename}")
                                    st.info("🔄 **Force re-optimization enabled** - running Optuna despite available cache")
                                else:
                                    _d(f"[CACHE-MISS] No cached parameters found for {csv_filename}")
                                    _d(f"[CACHE-MISS] Configuration: {model_config}")
                                    st.info("🔍 **No cached parameters found** - running Optuna optimization")
                                    
                                # Show current cached parameters for reference
                                cached_params = list_cached_params()
                                if cached_params:
                                    _d(f"[CACHE-INFO] Found {len(cached_params)} cached parameter sets:")
                                    for key, info in cached_params.items():
                                        _d(f"[CACHE-INFO] {key}: {info.get('csv_filename')} - MAE: {info.get('best_value', 'N/A')}")
                                else:
                                    _d("[CACHE-INFO] No cached parameters available")
                                
                                # Clear any previous trial results and show starting message
                                st.session_state["optuna_trial_results"] = []
                                with st.container():
                                    st.info(_t("Starting Optuna hyperparameter optimization: {} trials").format(n_trials))
                                    if optuna_parallel and optuna_n_jobs > 1:
                                        st.info(_t("Using {} parallel processes for enhanced optimization speed").format(optuna_n_jobs))
                                    st.info(_t("Live trial results will be displayed below as optimization progresses..."))

                                # --------------------------------------------------
                                # Determine optimisation split
                                # --------------------------------------------------
                                if X_te.empty:
                                    # 100 % training selected ➜ create internal 95/5 split
                                    _d("[OPTUNA] Creating internal 95/5 split for tuning (100% train mode)")
                                    _opt_tr_df, _opt_val_df = split_train_eval_frac(df, test_frac=0.05)
                                    X_opt_tr, y_opt_tr = features.select_feature_target_multi(
                                        _opt_tr_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                                    )
                                    X_opt_val, y_opt_val = features.select_feature_target_multi(
                                        _opt_val_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                                    )
                                else:
                                    X_opt_tr, y_opt_tr, X_opt_val, y_opt_val = X_tr, y_tr, X_te, y_te

                                                                # ------------------------------------------------------------------
                                # Optuna objective using GroupKFold by calendar-week and MAE metric
                                # ------------------------------------------------------------------

                                # Create grouping by ISO calendar week to avoid leakage within the same week
                                if X_te.empty:
                                    # We created internal split, use _opt_tr_df
                                    _grp_src_df = _opt_tr_df.copy()
                                else:
                                    # Using external validation, use original df rows
                                    _grp_src_df = df.loc[X_opt_tr.index].copy()
                                
                                _grp_src_df["_week"] = pd.to_datetime(_grp_src_df["detection_time"], errors="coerce").dt.isocalendar().week
                                groups_arr = _grp_src_df.loc[X_opt_tr.index, "_week"].to_numpy()

                                def objective(trial):
                                    params = {
                                        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                                        "max_depth": trial.suggest_int("max_depth", 3, 20),  # Increased from 10 to 20
                                        "num_leaves": trial.suggest_int("num_leaves", 16, 256),  # Increased from 128
                                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500),  # Increased from 300
                                        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
                                        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
                                        "max_bin": trial.suggest_int("max_bin", 255, 511),  # For better precision with deep trees
                                        "n_jobs": lgbm_n_jobs,  # NEW: Use configured threading
                                        # 🆕 COMPRESSION OPTIMIZATIONS (40-70% smaller files, no accuracy loss)
                                        "compress": True,                      # Enable built-in compression
                                        "compression_level": 6,                # Compression level (1-9, higher = smaller)
                                        "save_binary": True,                   # Save in binary format (smaller)
                                    }
                                    
                                    # NEW: GPU Acceleration Parameters
                                    if use_gpu_optuna and gpu_available:
                                        # Add GPU device parameters
                                        params.update({
                                            "device": "gpu",
                                            "gpu_platform_id": 0,
                                            "gpu_device_id": gpu_device_id,
                                            "gpu_use_dp": gpu_use_double_precision,
                                        })
                                        
                                        # Optimize GPU-specific parameters based on mode
                                        if gpu_optimization_mode == "auto":
                                            # Auto-optimize based on dataset size
                                            dataset_size = len(X_opt_tr)
                                            if dataset_size > 50000:
                                                params.update({
                                                    "gpu_use_dp": False,  # Use single precision for large datasets
                                                    "max_bin": 255,  # Smaller bins for GPU efficiency
                                                })
                                        elif gpu_optimization_mode == "speed":
                                            params.update({
                                                "gpu_use_dp": False,  # Single precision for speed
                                                "max_bin": 255,  # Smaller bins
                                                "feature_fraction": 0.8,  # Reduce feature sampling
                                            })
                                        elif gpu_optimization_mode == "accuracy":
                                            params.update({
                                                "gpu_use_dp": True,  # Double precision for accuracy
                                                "max_bin": 511,  # Larger bins for precision
                                            })
                                    else:
                                        # CPU fallback
                                        params["device"] = "cpu"
                                    
                                    # Apply quantile objective if selected
                                    if use_quantile:
                                        params.update({
                                            "objective": "quantile",
                                            "alpha": 0.5,
                                        })

                                    # ========================================================
                                    # ANCHOR-DAY PERFORMANCE OPTIMIZATION
                                    # ========================================================
                                    # Instead of generic MAE, optimize for continuous 7-day 
                                    # forecast performance using anchor-day methodology
                                    
                                    # OPTIMIZATION: Conditional CV folds based on speed mode
                                    n_folds = 2 if optuna_speed_mode else 3
                                    kf = GroupKFold(n_splits=n_folds)
                                    fold_anchor_maes = []
                                    
                                    for train_idx, val_idx in kf.split(X_opt_tr, y_opt_tr, groups=groups_arr):
                                        X_tr_fold = X_opt_tr.iloc[train_idx]
                                        y_tr_fold = y_opt_tr.iloc[train_idx]
                                        X_val_fold = X_opt_tr.iloc[val_idx]
                                        y_val_fold = y_opt_tr.iloc[val_idx]

                                        # OPTIMIZATION: Conditional performance settings based on speed mode
                                        trial_trees = min(500, n_trees) if optuna_speed_mode else n_trees
                                        trial_early_stop = 50 if optuna_speed_mode else 100
                                        
                                        mdl_tmp = MultiLGBMRegressor(
                                            base_params=params,
                                            upper_bound_estimators=trial_trees,
                                            early_stopping_rounds=trial_early_stop,
                                            uncertainty_estimation=True,
                                            n_bootstrap_samples=30,
                                            directional_feature_boost=2.0,  # 2x boost for directional features
                                            conservative_mode=True,  # Enable conservative predictions
                                            stability_feature_boost=3.0,  # 3x boost for stability features
                                            use_gpu=(use_gpu_optuna and gpu_available),  # NEW: GPU acceleration with availability check
                                            gpu_optimization=(gpu_optimization_mode == "auto"),  # NEW: Auto-optimization
                                        )
                                        
                                        # Generate anchor-day evaluation on this fold
                                        try:
                                            # OPTIMIZATION: Get the dataframe subset for this validation fold (minimal copy)
                                            val_df_fold = df.loc[X_val_fold.index, 
                                                               ['detection_time', 'temperature_grain'] + 
                                                               [f'temperature_grain_h{h}d' for h in HORIZON_TUPLE]
                                                              ].copy()
                                        except Exception as e:
                                            # Fallback to using the full validation set if optimization fails
                                            val_df_fold = df.loc[X_val_fold.index].copy()
                                        
                                        mdl_tmp.fit(
                                            X_tr_fold, y_tr_fold, 
                                            eval_set=(X_val_fold, y_val_fold), 
                                            verbose=False,
                                            anchor_df=val_df_fold,  # Use fold validation df for anchor-day methodology
                                            horizon_tuple=HORIZON_TUPLE,
                                            use_anchor_early_stopping=anchor_early_stop,
                                            balance_horizons=balance_horizons,  # NEW: Apply horizon balancing
                                            horizon_strategy=horizon_strategy,  # NEW: Pass strategy selection

                                        )
                                        
                                        # Continue with anchor-day evaluation
                                        try:
                                            
                                            # Generate predictions for all horizons
                                            preds_tmp = mdl_tmp.predict(X_val_fold)
                                            
                                            # Attach multi-horizon predictions to validation dataframe
                                            for idx, h in enumerate(HORIZON_TUPLE):
                                                if idx < preds_tmp.shape[1]:
                                                    val_df_fold.loc[X_val_fold.index, f"pred_h{h}d"] = preds_tmp[:, idx]
                                            
                                            # Compute anchor-day performance (simplified version)
                                            anchor_maes = []
                                            
                                            # Get unique anchor dates (days where we have 7-day continuous forecasts)
                                            val_df_fold["anchor_date"] = pd.to_datetime(val_df_fold["detection_time"]).dt.date
                                            anchor_dates = val_df_fold["anchor_date"].unique()
                                            
                                            # OPTIMIZATION: Conditional anchor date sampling based on speed mode
                                            n_anchor_dates = 5 if optuna_speed_mode else 10
                                            for anchor_date in anchor_dates[-n_anchor_dates:]:
                                                anchor_rows = val_df_fold[val_df_fold["anchor_date"] == anchor_date]
                                                if len(anchor_rows) < 5:  # Need minimum sensors
                                                    continue
                                                    
                                                # Compute MAE for each horizon on this anchor date
                                                horizon_maes = []
                                                for h in HORIZON_TUPLE:
                                                    pred_col = f"pred_h{h}d"
                                                    if pred_col in anchor_rows.columns:
                                                        actual_col = f"temperature_grain_h{h}d"
                                                        if actual_col in anchor_rows.columns:
                                                            mask = anchor_rows[[actual_col, pred_col]].notna().all(axis=1)
                                                            if mask.sum() > 0:
                                                                mae_h = np.abs(anchor_rows.loc[mask, actual_col] - anchor_rows.loc[mask, pred_col]).mean()
                                                                horizon_maes.append(mae_h)
                                            
                                            if horizon_maes:
                                                anchor_maes.append(np.mean(horizon_maes))
                                            
                                            if anchor_maes:
                                                fold_anchor_maes.append(np.mean(anchor_maes))
                                            else:
                                                fold_anchor_maes.append(999.0)  # Penalty for no valid anchor evaluation
                                                
                                        except Exception as e:
                                            fold_anchor_maes.append(999.0)  # Penalty for evaluation failure

                                    mean_anchor_mae = float(np.mean(fold_anchor_maes)) if fold_anchor_maes else 999.0

                                    # Store trial results for persistent display instead of disappearing toast
                                    import streamlit as _st
                                    trial_results = _st.session_state.setdefault("optuna_trial_results", [])
                                    trial_results.append({
                                        "trial": trial.number,
                                        "mae": mean_anchor_mae,
                                        "params": {k: v for k, v in trial.params.items()}
                                    })
                                    # Keep only last 20 trials to show more progress
                                    if len(trial_results) > 20:
                                        trial_results.pop(0)
                                    
                                    _d(f"[OPTUNA] Trial {trial.number} – Anchor-7d MAE {mean_anchor_mae:.4f}")
                                    return mean_anchor_mae
                                
                                # ------------------------------------------------------------------
                                # Create and run Optuna study with parallel optimization
                                # ------------------------------------------------------------------
                                study = optuna.create_study(direction="minimize")
                                
                                # Initialize trial results display
                                st.session_state["optuna_trial_results"] = []
                                trial_placeholder = st.empty()
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Parallel optimization with progress feedback
                                if optuna_parallel and optuna_n_jobs > 1:
                                    _d(f"[OPTUNA] Starting parallel optimization: {n_trials} trials using {optuna_n_jobs} processes")
                                    st.toast(f"Parallel Optuna optimization: {optuna_n_jobs} processes × {n_trials} trials")
                                    
                                    # Create a callback to update trial results display
                                    def trial_callback(study, trial):
                                        trial_results = st.session_state.get("optuna_trial_results", [])
                                        
                                        # Update progress bar and status using actual trial number
                                        current_trial = trial.number + 1
                                        progress = current_trial / n_trials
                                        progress_bar.progress(min(progress, 1.0))
                                        status_text.text(f"Trial {current_trial}/{n_trials} completed - Current MAE: {trial.value:.4f}")
                                        
                                        if trial_results:
                                            with trial_placeholder.container():
                                                st.markdown("### **Live Optuna Trial Results**")
                                                
                                                # Show best trial so far
                                                best_trial = min(trial_results, key=lambda x: x['mae'])
                                                st.success(f"**Best Trial**: #{best_trial['trial']} - MAE: {best_trial['mae']:.4f}")
                                                
                                                # Show recent trials table (last 10 trials)
                                                recent_trials = trial_results[-10:]  # Last 10 trials
                                                trials_df = pd.DataFrame([
                                                    {
                                                        "Trial": t["trial"],
                                                        "MAE": f"{t['mae']:.4f}",
                                                        "Learning Rate": f"{t['params'].get('learning_rate', 0):.4f}",
                                                        "Max Depth": t['params'].get('max_depth', 0),
                                                        "Num Leaves": t['params'].get('num_leaves', 0)
                                                    }
                                                    for t in recent_trials
                                                ])
                                                st.dataframe(trials_df, use_container_width=True, key=f"trials_{current_trial}")
                                    
                                    study.optimize(objective, n_trials=n_trials, n_jobs=optuna_n_jobs, callbacks=[trial_callback])
                                    st.toast(f"Parallel optimization completed using {optuna_n_jobs} processes")
                                else:
                                    _d(f"[OPTUNA] Starting sequential optimization: {n_trials} trials")
                                    
                                    # Create a callback to update trial results display
                                    def trial_callback(study, trial):
                                        trial_results = st.session_state.get("optuna_trial_results", [])
                                        
                                        # Update progress bar and status using actual trial number
                                        current_trial = trial.number + 1
                                        progress = current_trial / n_trials
                                        progress_bar.progress(min(progress, 1.0))
                                        status_text.text(f"Trial {current_trial}/{n_trials} completed - Current MAE: {trial.value:.4f}")
                                        
                                        if trial_results:
                                            with trial_placeholder.container():
                                                st.markdown("### **Live Optuna Trial Results**")
                                                
                                                # Show best trial so far
                                                best_trial = min(trial_results, key=lambda x: x['mae'])
                                                st.success(f"**Best Trial**: #{best_trial['trial']} - MAE: {best_trial['mae']:.4f}")
                                                
                                                # Show recent trials table (last 10 trials)
                                                recent_trials = trial_results[-10:]  # Last 10 trials
                                                trials_df = pd.DataFrame([
                                                    {
                                                        "Trial": t["trial"],
                                                        "MAE": f"{t['mae']:.4f}",
                                                        "Learning Rate": f"{t['params'].get('learning_rate', 0):.4f}",
                                                        "Max Depth": t['params'].get('max_depth', 0),
                                                        "Num Leaves": t['params'].get('num_leaves', 0)
                                                    }
                                                    for t in recent_trials
                                                ])
                                                st.dataframe(trials_df, use_container_width=True, key=f"trials_{current_trial}")
                                    
                                    study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])
                                
                                # Update base_params with best parameters
                                best_params = study.best_params
                                base_params.update(best_params)
                                
                                # Keep quantile objective if it was selected
                                if use_quantile:
                                    base_params.update({
                                        "objective": "quantile",
                                        "alpha": 0.5,
                                    })
                                
                                _d(f"[OPTUNA] Optimization complete. Best anchor-7d MAE: {study.best_value:.4f}")
                                _d(f"[OPTUNA] Best params: {best_params}")
                                
                                # Final summary of optimization results - hide progress indicators
                                progress_bar.empty()
                                status_text.empty()
                                
                                trial_results = st.session_state.get("optuna_trial_results", [])
                                if trial_results:
                                    with trial_placeholder.container():
                                        st.markdown("### **Optuna Optimization Complete**")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Trials", len(trial_results))
                                        with col2:
                                            best_mae = min(t['mae'] for t in trial_results)
                                            st.metric("Best MAE", f"{best_mae:.4f}")
                                        with col3:
                                            worst_mae = max(t['mae'] for t in trial_results)
                                            improvement = ((worst_mae - best_mae) / worst_mae * 100)
                                            st.metric("Improvement", f"{improvement:.1f}%")
                                        
                                        # Show all trials in an expandable section
                                        with st.expander("**View All Trial Results**", expanded=False):
                                            all_trials_df = pd.DataFrame([
                                                {
                                                    "Trial": t["trial"],
                                                    "MAE": f"{t['mae']:.4f}",
                                                    "Learning Rate": f"{t['params'].get('learning_rate', 0):.4f}",
                                                    "Max Depth": t['params'].get('max_depth', 0),
                                                    "Num Leaves": t['params'].get('num_leaves', 0),
                                                    "Subsample": f"{t['params'].get('subsample', 0):.3f}",
                                                    "Colsample": f"{t['params'].get('colsample_bytree', 0):.3f}"
                                                }
                                                for t in trial_results
                                            ])
                                            st.dataframe(all_trials_df.sort_values("MAE"), use_container_width=True, key="final_trials")
                                        
                                        st.success(f"**Best Parameters Found**: Learning Rate: {best_params.get('learning_rate', 0):.4f}, "
                                                 f"Max Depth: {best_params.get('max_depth', 0)}, Num Leaves: {best_params.get('num_leaves', 0)}")
                                
                                # Save optimal parameters to cache
                                if use_param_cache:
                                    try:
                                        cache_key = save_optimal_params(
                                            csv_filename=csv_filename,
                                            df=df,
                                            model_config=model_config,
                                            optimal_params=best_params,
                                            best_value=study.best_value,
                                            n_trials=n_trials
                                        )
                                        _d(f"[CACHE] Saved optimal parameters with key: {cache_key}")
                                        st.success(f"💾 Optimal parameters saved to cache!")
                                    except Exception as cache_exc:
                                        _d(f"[CACHE] Failed to save parameters: {cache_exc}")
                                        st.warning("Failed to save parameters to cache")
                                
                                # Performance feedback
                                n_folds = 2 if optuna_speed_mode else 3
                                trial_trees = min(500, n_trees) if optuna_speed_mode else n_trees
                                parallel_info = f", {optuna_n_jobs} parallel jobs" if optuna_parallel and optuna_n_jobs > 1 else ""
                                speed_info = f" (Fast mode: {n_folds}-fold CV, {trial_trees} trees{parallel_info})" if optuna_speed_mode else f"{parallel_info}"
                                st.toast(f"Optuna optimization completed. Best MAE: {study.best_value:.4f}{speed_info}")
                                
                            except Exception as exc:
                                _d(f"[OPTUNA] Tuning failed or Optuna not installed: {exc}")
                                st.warning(f"Optuna optimization failed: {exc}")
                    
                    # No cached parameters and Optuna disabled - use default parameters
                    else:
                        if not cached_result:
                            _d(f"[MODEL-CONFIG] No cached parameters found and Optuna disabled - using default parameters")
                            _d(f"[MODEL-CONFIG] Default base_params: {base_params}")
                            if force_reoptimize:
                                st.info("🔄 **Force re-optimization enabled** but Optuna is disabled - using default parameters")
                            else:
                                st.info("📋 **Using default parameters** - enable Optuna or use parameter cache for optimized settings")

                    # ================================================================
                    # PERFORMANCE OPTIMIZATIONS FOR 7-DAY FORECASTING
                    # ================================================================
                    # 1. Anchor-day early stopping: Check every 10 iterations (10x faster)
                    # 2. Fast Optuna mode: 2-fold CV, lower tree limits, aggressive stopping  
                    # 3. Conditional early stopping: 50 rounds with anchor mode vs 100 standard
                    # 4. Vectorized computations: Batch operations for anchor MAE calculation
                    # 5. Minimal dataframe copying: Only essential columns in Optuna loops
                    # ================================================================
                    
                    main_early_stop = 50 if anchor_early_stop else 100
                    
                    base_mdl = MultiLGBMRegressor(
                        base_params=base_params,
                        # 🚀 OPTIMIZED: Using granarypredict defaults for speed improvements
                        # upper_bound_estimators=1000 (default), early_stopping_rounds=50 (default)
                        uncertainty_estimation=True,
                        n_bootstrap_samples=25,  # 🚀 OPTIMIZED: Reduced from 100 for 75% speed improvement
                        directional_feature_boost=2.0,  # 2x boost for directional features
                        conservative_mode=True,  # Enable conservative predictions
                        stability_feature_boost=3.0,  # 3x boost for stability features
                        use_gpu=(use_gpu_optuna and gpu_available),  # NEW: GPU acceleration with availability check
                        gpu_optimization=(gpu_optimization_mode == "auto"),  # NEW: Auto-optimization
                    )
                    use_wrapper = False
                    _d(f"[TRAIN] LightGBM initialised – upper_bound={n_trees}, early_stop={main_early_stop}, base_params={base_params}")
                    _d(f"[TRAIN-UNCERTAINTY] Uncertainty estimation ENABLED: n_bootstrap_samples=100")
                    _d(f"[TRAIN-CONSERVATIVE] Conservative mode ENABLED: stability_feature_boost=3.0x")
                    _d(f"[TRAIN-DIRECTIONAL] Directional feature boost: 2.0x for movement prediction")
                    st.toast("🔬 Training with conservative mode: 3x stability boost + 2x directional boost", icon="🔬")
                    
                    # Enhanced conservative system feedback
                    with st.expander(_t("Conservative System Status"), expanded=True):
                        st.success(f"**{_t('Conservative Temperature Prediction')}**: {_t('Active')}")
                        st.write(f"**{_t('System Configuration')}:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**{_t('Stability Features')}**: {_t('8 thermal physics features')}")
                            st.info(f"**{_t('Stability Boost')}**: {_t('3.0x feature importance')}")
                            st.info(f"**{_t('Uncertainty Samples')}**: {_t('50 bootstrap samples')}")
                        with col2:
                            st.info(f"**{_t('Directional Features')}**: {_t('2.0x importance boost')}")
                            st.info(f"**{_t('Horizon Balancing')}**: {_t('Enabled')}")
                            st.info(f"**{_t('Conservative Loss')}**: {_t('Thermal inertia penalties')}")
                        
                        st.markdown(f"**{_t('Expected Benefits')}:**")
                        st.markdown(f"""
                        - **{_t('More stable predictions')}**: {_t('Respects grain thermal inertia')}
                        - **{_t('Reduced aggressive changes')}**: {_t('Conservative temperature evolution')}
                        - **{_t('Better 7-day accuracy')}**: {_t('Less cumulative error buildup')}
                        - **{_t('Uncertainty quantification')}**: {_t('Confidence intervals for all predictions')}
                        """)
                        
                        # Real-time training feedback
                        st.toast("Conservative system initialized - thermal stability features active")
                        if hasattr(st, 'rerun'):
                            pass  # Don't force rerun during training

                # ---------------- Fit -----------------------
                if use_wrapper:
                    mdl = MultiOutputRegressor(base_mdl)
                    mdl.fit(X_tr, y_tr)
                    _d("[TRAIN] Wrapper model fit complete")
                else:
                    if perform_validation and not X_te.empty:
                        # Anchor-day early stopping using external validation split for 7-day consecutive accuracy
                        base_mdl.fit(
                            X_tr, y_tr, 
                            eval_set=(X_te, y_te), 
                            verbose=False,
                            anchor_df=df_eval_tmp,  # Pass evaluation dataframe for anchor-day methodology
                            horizon_tuple=HORIZON_TUPLE,
                            use_anchor_early_stopping=anchor_early_stop,
                            balance_horizons=balance_horizons,  # NEW: Apply horizon balancing
                            horizon_strategy=horizon_strategy,  # NEW: Pass strategy selection
                        )
                        _d(f"[TRAIN] Anchor-day early-stopping complete – best_iter={base_mdl.best_iteration_}")
                        mdl = base_mdl
                    else:
                        # ----------------------------------------------------------
                        # No validation split (user selected 100 % train) –> create
                        # an internal 95/5 chronological split to pick the best
                        # iteration, then refit on the full dataset with that
                        # fixed n_estimators so behaviour matches the legacy flow.
                        # ----------------------------------------------------------
                        int_train_df, int_val_df = split_train_eval_frac(df, test_frac=0.05)

                        X_int_tr, y_int_tr = features.select_feature_target_multi(
                            int_train_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                        )
                        X_int_val, y_int_val = features.select_feature_target_multi(
                            int_val_df, target_col=TARGET_TEMP_COL, horizons=HORIZON_TUPLE
                        )

                        finder = MultiLGBMRegressor(
                            base_params=base_params,
                            # 🚀 OPTIMIZED: Using granarypredict defaults for speed improvements
                            # upper_bound_estimators=1000 (default), early_stopping_rounds=50 (default)
                            uncertainty_estimation=True,
                            n_bootstrap_samples=25,  # 🚀 OPTIMIZED: Reduced from 100 for 75% speed improvement
                            directional_feature_boost=2.0,  # 2x boost for directional features
                            conservative_mode=True,  # Enable conservative predictions
                            stability_feature_boost=3.0,  # 3x boost for stability features
                            use_gpu=(use_gpu_optuna and gpu_available),  # NEW: GPU acceleration with availability check
                            gpu_optimization=(gpu_optimization_mode == "auto"),  # NEW: Auto-optimization
                        )
                        _d(f"[TRAIN-INTERNAL] Creating finder model with uncertainty estimation enabled")
                        finder.fit(
                            X_int_tr, y_int_tr, 
                            eval_set=(X_int_val, y_int_val), 
                            verbose=False,
                            anchor_df=int_val_df,
                            horizon_tuple=HORIZON_TUPLE,
                            use_anchor_early_stopping=anchor_early_stop,
                            balance_horizons=balance_horizons,
                            horizon_strategy=horizon_strategy,
                        )
                        best_n = finder.best_iteration_
                        
                        # Refit on **all** data with the chosen tree count
                        final_params = base_params | {"n_estimators": best_n}
                        final_lgbm = MultiLGBMRegressor(
                            base_params=final_params,
                            upper_bound_estimators=best_n,
                            early_stopping_rounds=0,
                            uncertainty_estimation=True,
                            n_bootstrap_samples=100,  # Increased from 50 for better uncertainty estimation
                            directional_feature_boost=2.0,  # 2x boost for directional features
                            conservative_mode=True,  # Enable conservative predictions
                            stability_feature_boost=3.0,  # 3x boost for stability features
                            use_gpu=(use_gpu_optuna and gpu_available),  # NEW: GPU acceleration with availability check
                            gpu_optimization=(gpu_optimization_mode == "auto"),  # NEW: Auto-optimization
                        )
                        _d(f"[TRAIN-FINAL] Creating final model with uncertainty estimation enabled for full dataset")
                        final_lgbm.fit(X_all, y_all, balance_horizons=balance_horizons, horizon_strategy=horizon_strategy)
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
                st.toast("Model training completed successfully")
                st.sidebar.success(f"{model_choice} training completed (validation split contained no ground-truth targets).")
            else:
                st.toast(f"Training completed successfully. MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")
                st.sidebar.success(f"{model_choice} training completed. MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")
            # Persist split settings for later evaluation
            st.session_state["last_train_split_mode"] = "last30" if use_last_30 else "percentage"
            if not use_last_30:
                st.session_state["last_train_pct"] = 100 if train_pct == 100 else train_pct

        # Parallel processing status
        with st.sidebar.expander(_t("Performance Optimization"), expanded=False):
            try:
                parallel_info = features.get_parallel_info()
                if parallel_info['parallel_enabled']:
                    st.success(f"**{_t('Parallel Processing')}**: {_t('Active')}")
                    st.info(f"**{_t('Workers')}**: {parallel_info['max_workers']} of {parallel_info['cpu_count']} CPU cores")
                    st.info(f"**{_t('Expected Speedup')}**: {_t('3-5x faster feature engineering')}")
                    
                    # Show Optuna parallelization status if training is active
                    if 'optuna_parallel' in locals() and optuna_parallel:
                        st.success(f"**Optuna Parallel**: {optuna_n_jobs} processes")
                        st.info(f"**{_t('Expected Speedup')}**: {_t('2-4x faster hyperparameter optimization')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(_t("Enable All Cores")):
                            features.set_parallel_processing(True, parallel_info['cpu_count'])
                            st.rerun()
                    with col2:
                        if st.button(_t("Disable Parallel")):
                            features.set_parallel_processing(False)
                            st.rerun()
                else:
                    st.warning(f"**{_t('Parallel Processing')}**: {_t('Disabled')}")
                    if st.button(_t("Enable Parallel Processing")):
                        features.set_parallel_processing(True)
                        st.rerun()
            except Exception as e:
                st.error(_t("Could not retrieve parallel processing information: {}").format(e))

        # Parameter cache management
        with st.sidebar.expander(_t("Parameter Cache"), expanded=False):
            cached_params = list_cached_params()
            if cached_params:
                st.write(f"**{_t('Cached parameter sets')}**: {len(cached_params)}")
                
                # Show cache details
                for cache_key, cache_info in cached_params.items():
                    with st.expander(f"{cache_info['csv_filename']} (MAE: {cache_info['best_value']:.3f})", expanded=False):
                        st.write(f"**{_t('Best MAE')}**: {cache_info['best_value']:.4f}")
                        st.write(f"**{_t('Trials')}**: {cache_info['n_trials']}")
                        st.write(f"**{_t('Data shape')}**: {cache_info['data_shape']}")
                        st.write(f"**{_t('Cached')}**: {cache_info['timestamp']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(_t("Clear this"), key=f"clear_{cache_key}"):
                                st.toast(_t("Clearing cache for {}...").format(cache_info['csv_filename']))
                                clear_cache(cache_info['csv_filename'])
                                st.rerun()
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(_t("Clear all cache")):
                        st.toast(_t("Clearing all cached parameters..."))
                        clear_cache()
                        st.rerun()
                with col2:
                    if st.button("Refresh"):
                        st.toast("Refreshing cache view...")
                        st.rerun()
            else:
                st.write("No cached parameters available.")
                st.caption("Train a model with Optuna optimization to create cache entries.")

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
                    if eval_pressed:
                        st.toast("Starting model evaluation...")
                with col_evalfc:
                    eval_fc_pressed = st.button(_t("Eval & Forecast"), key="btn_eval_fc", use_container_width=True)
                    if eval_fc_pressed:
                        st.toast("Starting evaluation and forecast generation...")

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
                        
                        # Log uncertainty estimation if applicable
                        if isinstance(mdl, MultiLGBMRegressor) and hasattr(mdl, 'uncertainty_estimation'):
                            if mdl.uncertainty_estimation:
                                _d(f"[UNCERTAINTY] Model uses uncertainty estimation (n_bootstrap={getattr(mdl, 'n_bootstrap_samples', 50)})")
                                st.toast(f"🔬 {mdl_name}: Uncertainty estimation active with {getattr(mdl, 'n_bootstrap_samples', 50)} bootstrap samples", icon="🔬")
                                

                            else:
                                _d(f"[UNCERTAINTY] Model uncertainty estimation disabled - point predictions only")
                                st.toast(f"{mdl_name}: No uncertainty estimation - point predictions only", icon="")

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

                        # ---- Compute metrics for **all** horizons up to HORIZON_DAYS ----
                        mae_by_h, rmse_by_h, mape_by_h = {}, {}, {}
                        for h in HORIZON_TUPLE:
                            col_name = f"pred_h{h}d"
                            if col_name in df_eval_actual.columns:
                                mae_v, rmse_v, mape_v = _metric(col_name)
                            else:
                                mae_v = rmse_v = mape_v = float("nan")
                            mae_by_h[h] = mae_v
                            rmse_by_h[h] = rmse_v
                            mape_by_h[h] = mape_v

                        # For backward‐compatibility, keep first horizon values
                        mae_h1 = mae_by_h.get(1, float("nan"))
                        rmse_h1 = rmse_by_h.get(1, float("nan"))
                        mape_h1 = mape_by_h.get(1, float("nan"))

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
                            "mae_h2": mae_by_h.get(2, float("nan")),
                            "mae_h3": mae_by_h.get(3, float("nan")),
                            "rmse_h1": rmse_h1,
                            "rmse_h2": rmse_by_h.get(2, float("nan")),
                            "rmse_h3": rmse_by_h.get(3, float("nan")),
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
                            "mae_by_h": mae_by_h,
                            "rmse_by_h": rmse_by_h,
                            "mape_by_h": mape_by_h,
                        }

                    # last evaluated model as active
                    if target_models:
                        st.session_state["active_model"] = target_models[0]

                    st.toast(f"Evaluation completed for {len(target_models)} model(s)")
                    st.sidebar.success("Model evaluation completed successfully.")

                    # If user requested Eval & Forecast, automatically create forecast for selected model
                    if eval_fc_pressed:
                        st.sidebar.write(_t("Generating forecast…"))
                        models_to_fc = target_models  # already respects all_models_chk
                        for mdl in models_to_fc:
                            generate_and_store_forecast(mdl, horizon=HORIZON_DAYS)
                        st.toast(f"Forecast generated for {len(models_to_fc)} model(s)")
                        st.sidebar.success("Forecast generation completed successfully.")

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
        if chosen_model != st.session_state.get("active_model"):
            st.toast(f"Selected model: {chosen_model}")
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
                    st.toast("Generating forecast for selected model...")
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

    # Per-horizon metrics up to HORIZON_DAYS
    mae_by_h = res.get("mae_by_h", {})
    rmse_by_h = res.get("rmse_by_h", {})
    mape_by_h = res.get("mape_by_h", {})

    # ---- Overview row (conf / acc) -----------------------------------
    overview_cols = st.columns(2)
    with overview_cols[0]:
        st.metric(_t("Conf (%)"), "--" if pd.isna(conf_val) else f"{conf_val:.2f}")
    with overview_cols[1]:
        st.metric(_t("Acc (%)"), "--" if pd.isna(acc_val) else f"{acc_val:.2f}")

    # ---- MAE per horizon ---------------------------------------------
    st.markdown(f"#### {_t('MAE per horizon')}")
    cols_mae = st.columns(len(HORIZON_TUPLE))
    for idx, h in enumerate(HORIZON_TUPLE):
        v = mae_by_h.get(h, float("nan"))
        with cols_mae[idx]:
            st.metric(f"MAE h+{h}", "--" if pd.isna(v) else f"{v:.2f}")

    # ---- RMSE per horizon --------------------------------------------
    st.markdown(f"#### {_t('RMSE per horizon')}")
    cols_rmse = st.columns(len(HORIZON_TUPLE))
    for idx, h in enumerate(HORIZON_TUPLE):
        v = rmse_by_h.get(h, float("nan"))
        with cols_rmse[idx]:
            st.metric(f"RMSE h+{h}", "--" if pd.isna(v) else f"{v:.2f}")

    # ---- MAPE per horizon (optional) ----------------------------------
    st.markdown(f"#### {_t('MAPE per horizon')}")
    cols_mape = st.columns(len(HORIZON_TUPLE))
    for idx, h in enumerate(HORIZON_TUPLE):
        v = mape_by_h.get(h, float("nan"))
        with cols_mape[idx]:
            st.metric(f"MAPE h+{h}", "--" if pd.isna(v) else f"{v:.2f}")

    # ---------------- Explanation caption -----------------------------
    st.caption(
        _t(
            "Row-wise horizon metrics (above) average the error of each h-day-ahead prediction across all evaluation rows.\n"
            "For a real-world, bulletin-style view of performance, switch to the 'Anchor 7-day' tab, where metrics are computed by freezing predictions on an anchor day and comparing them with observations that occur h days later."
        )
    )

    st.markdown("---")

    tab_labels = [_t("Summary"), _t("Predictions"), _t("3D Grid"), _t("Time Series"), _t("Anchor 7-day"), _t("Uncertainty"), _t("Extremes"), _t("Debug")]
    summary_tab, pred_tab, grid_tab, ts_tab, anchor_tab, uncertainty_tab, extremes_tab, debug_tab = st.tabs(tab_labels)

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
                st.dataframe(fi_df, use_container_width=True, key=f"feat_imp_{model_name}")

    with pred_tab:
        _st_dataframe_safe(df_predplot_all, key=f"pred_df_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")

    with grid_tab:
        # Build list of unique dates present in evaluation subset only
        unique_dates = sorted(pd.to_datetime(df_eval["detection_time"], errors="coerce").dt.floor("D").unique())
        date_choice = st.selectbox(
            _t("Select date"),
            options=[d.strftime("%Y-%m-%d") for d in unique_dates],
            key=f"day_{model_name}_grid_{len(unique_dates)}",
        )
        sel_date = pd.to_datetime(date_choice)
        df_predplot = df_predplot_all[pd.to_datetime(df_predplot_all["detection_time"], errors="coerce").dt.floor("D") == sel_date]
        plot_3d_grid(
            df_predplot,
            key=f"grid_{model_name}_{date_choice}",
            color_by_delta=True,
        )

    with ts_tab:
        # ---------------- Horizon selector -----------------------
        horizon_opts = list(range(1, HORIZON_DAYS + 1))
        sel_horizon = st.selectbox(
            _t("Forecast horizon (h + ? days)"),
            options=horizon_opts,
            index=0,
            key=f"ts_horizon_{model_name}",
        )

        # Merge past (training+eval) with future predictions so the trend is continuous
        hist_df = res.get("df_predplot_all")
        if hist_df is None:
            hist_df = pd.DataFrame()
        combined_df = pd.concat([hist_df, df_eval], ignore_index=True, sort=False)
        plot_time_series(
            combined_df,
            key=f"time_{model_name}_{sel_horizon}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}",
            horizon_day=sel_horizon,
        )

    # -------------- ANCHOR 7-DAY TAB -------------------
    with anchor_tab:
        st.subheader("7-Day Forecast from Anchor Day (forecast_day=1)")

        # Add explanatory caption once at top of tab
        st.caption(
            _t(
                "Anchor metrics emulate operational use: predictions are frozen on the selected anchor day (forecast_day = 1) "
                "and each horizon h is scored against the real temperature measured h days later.\n"
                "This gives the most realistic estimate of future-forecast performance."
            )
        )

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
                    import plotly.graph_objects as go
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

                # ------------------------------------------------------
                # NEW: Sensor-level error tables per horizon (Δ > 0.5 °C)
                # ------------------------------------------------------
                THRESH_DIFF = 0.5  # degrees C
                st.markdown("---")
                st.markdown("### Sensor-level discrepancies (> 0.5 °C)")

                # Pre-compute yesterday date lookup once for performance
                full_df_dates = df_eval.copy()
                full_df_dates["_date"] = pd.to_datetime(full_df_dates["detection_time"]).dt.floor("D")

                for h in HORIZON_TUPLE:
                    target_date = anchor_date + pd.Timedelta(days=h)
                    pred_col = f"pred_h{h}d"
                    if pred_col not in anchor_rows.columns:
                        continue  # skip horizon not available

                    # Predicted rows (anchor rows already filtered by sensor keys)
                    preds_h = anchor_rows.copy()
                    preds_h = preds_h.assign(predicted_temp=preds_h[pred_col])

                    # Actual rows at target date
                    actual_h = df_eval[pd.to_datetime(df_eval["detection_time"]).dt.floor("D") == target_date].copy()
                    actual_h = actual_h.assign(actual_temp=actual_h[TARGET_TEMP_COL])

                    key_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in preds_h.columns and c in actual_h.columns]

                    merged_h = preds_h[key_cols + ["predicted_temp"]].merge(
                        actual_h[key_cols + ["actual_temp", "detection_time"]], on=key_cols, how="inner"
                    )

                    if merged_h.empty:
                        st.info(f"h+{h}: No matching sensor readings available.")
                        continue

                    # Yesterday temp (target_date − 1 day)
                    yest_date = target_date - pd.Timedelta(days=1)
                    yest_rows = full_df_dates[full_df_dates["_date"] == yest_date].copy()
                    yest_rows = yest_rows.assign(yesterday_temp=yest_rows[TARGET_TEMP_COL])

                    merged_h = merged_h.merge(
                        yest_rows[key_cols + ["yesterday_temp"]], on=key_cols, how="left"
                    )

                    merged_h["diff"] = (merged_h["predicted_temp"] - merged_h["actual_temp"]).abs()
                    merged_h = merged_h[merged_h["diff"] > THRESH_DIFF]

                    if merged_h.empty:
                        st.info(f"h+{h}: No sensor differences > {THRESH_DIFF} °C.")
                        continue

                    show_cols = key_cols + ["predicted_temp", "actual_temp", "yesterday_temp", "diff"]
                    st.markdown(f"#### h+{h}")
                    _st_dataframe_safe(merged_h[show_cols].sort_values("diff", ascending=False), key=f"anchor_sensor_h{h}_{model_name}")
                # End new block

        # ---- Overall MAE across ALL anchor dates (aggregated) ----
        mae_vals_global: list[float] = []
        for anchor_val in sorted(df_eval["forecast_day"].unique()):
            anchor_rows_all = df_eval[df_eval["forecast_day"] == anchor_val].copy()
            if anchor_rows_all.empty:
                continue
            anchor_date_all = pd.to_datetime(anchor_rows_all["detection_time"]).dt.floor("D").min()
            for h in HORIZON_TUPLE:
                pred_col = f"pred_h{h}d"
                if pred_col not in anchor_rows_all.columns:
                    continue
                pred_subset_all = anchor_rows_all.assign(pred_val=anchor_rows_all[pred_col])

                target_date_all = anchor_date_all + pd.Timedelta(days=h)
                act_subset_all = df_eval[pd.to_datetime(df_eval["detection_time"]).dt.floor("D") == target_date_all].copy()
                act_subset_all = act_subset_all.assign(actual_val=act_subset_all[TARGET_TEMP_COL])

                key_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in pred_subset_all.columns and c in act_subset_all.columns]
                merged_all = pred_subset_all[key_cols + ["pred_val"]].merge(
                    act_subset_all[key_cols + ["actual_val"]], on=key_cols, how="inner"
                )

                if not merged_all.empty:
                    mae_val = (merged_all["pred_val"] - merged_all["actual_val"]).abs().mean()
                    if pd.notna(mae_val):
                        mae_vals_global.append(mae_val)

        if mae_vals_global:
            avg_mae_global = float(np.nanmean(mae_vals_global))
            max_mae_global = float(np.nanmax(mae_vals_global))
            st.markdown("---")
            st.markdown("### Aggregate MAE Metrics Across All Anchors")
            col_avg, col_max = st.columns(2)
            with col_avg:
                st.metric("Avg MAE (all anchors × 7 days)", f"{avg_mae_global:.2f}")
            with col_max:
                st.metric("Max MAE (all anchors × 7 days)", f"{max_mae_global:.2f}")

            # -------- Per-horizon aggregate MAE -----------------------
            mae_by_h: dict[int, list[float]] = {h: [] for h in HORIZON_TUPLE}
            maxerr_by_h: dict[int, list[float]] = {h: [] for h in HORIZON_TUPLE}

            for anchor_val in sorted(df_eval["forecast_day"].unique()):
                anchor_rows_all = df_eval[df_eval["forecast_day"] == anchor_val].copy()
                if anchor_rows_all.empty:
                    continue
                anchor_date_all = pd.to_datetime(anchor_rows_all["detection_time"]).dt.floor("D").min()

                for h in HORIZON_TUPLE:
                    pred_col = f"pred_h{h}d"
                    if pred_col not in anchor_rows_all.columns:
                        continue

                    pred_subset_all = anchor_rows_all.assign(pred_val=anchor_rows_all[pred_col])
                    target_date_all = anchor_date_all + pd.Timedelta(days=h)

                    act_subset_all = df_eval[
                        pd.to_datetime(df_eval["detection_time"]).dt.floor("D") == target_date_all
                    ].copy()
                    act_subset_all = act_subset_all.assign(actual_val=act_subset_all[TARGET_TEMP_COL])

                    key_cols = [
                        c
                        for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"]
                        if c in pred_subset_all.columns and c in act_subset_all.columns
                    ]

                    merged_all = pred_subset_all[key_cols + ["pred_val"]].merge(
                        act_subset_all[key_cols + ["actual_val"]], on=key_cols, how="inner"
                    )

                    if not merged_all.empty:
                        diffs = (merged_all["pred_val"] - merged_all["actual_val"]).abs()
                        mae_val = diffs.mean()
                        max_val = diffs.max()
                        if pd.notna(mae_val):
                            mae_by_h[h].append(mae_val)
                        if pd.notna(max_val):
                            maxerr_by_h[h].append(max_val)

            # Display horizon-wise MAE metrics if any values present
            any_vals = any(len(v) > 0 for v in mae_by_h.values())
            if any_vals:
                st.markdown("### MAE by Forecast Horizon (All Anchors)")
                cols_h = st.columns(len(HORIZON_TUPLE))
                for idx, h in enumerate(HORIZON_TUPLE):
                    vals = mae_by_h[h]
                    mae_h = np.nan if not vals else float(np.nanmean(vals))
                    with cols_h[idx]:
                        lbl = f"h+{h} MAE"
                        st.metric(lbl, "--" if pd.isna(mae_h) else f"{mae_h:.2f}")

                # ---------- Max absolute error per horizon -------------
                cols_maxh = st.columns(len(HORIZON_TUPLE))
                for idx, h in enumerate(HORIZON_TUPLE):
                    vals = maxerr_by_h[h]
                    max_h = np.nan if not vals else float(np.nanmax(vals))
                    with cols_maxh[idx]:
                        lbl = f"h+{h} Max |Error|"
                        st.metric(lbl, "--" if pd.isna(max_h) else f"{max_h:.2f}")

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
                import plotly.graph_objects as go
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
                    import plotly.graph_objects as go
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

    # ------------------ UNCERTAINTY ANALYSIS TAB ------------------
    with uncertainty_tab:
        st.subheader("🔬 Uncertainty Analysis & Prediction Confidence")
        
        # Load the model to check if it has uncertainty estimation capabilities
        try:
            mdl = load_trained_model(model_name)
            
            # Conservative system status display

                
            # Uncertainty estimation status
            if hasattr(mdl, 'uncertainty_estimation') and mdl.uncertainty_estimation:
                st.success("**Uncertainty Estimation**: ACTIVE")
                
                # Try to get uncertainty intervals from the most recent prediction
                prediction_intervals = getattr(mdl, '_last_prediction_intervals', None)
                
                if prediction_intervals and isinstance(prediction_intervals, dict):
                    st.markdown("### 📈 Prediction Confidence Intervals")
                    
                    # Display uncertainty statistics
                    uncertainties = prediction_intervals.get('uncertainties')
                    if uncertainties is not None:
                        avg_uncertainty = np.mean(uncertainties)
                        max_uncertainty = np.max(uncertainties)
                        min_uncertainty = np.min(uncertainties)
                        
                        # Uncertainty metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Uncertainty", f"{avg_uncertainty:.3f}°C")
                        with col2:
                            st.metric("Max Uncertainty", f"{max_uncertainty:.3f}°C")
                        with col3:
                            st.metric("Min Uncertainty", f"{min_uncertainty:.3f}°C")
                        with col4:
                            reliability = "High" if avg_uncertainty < 0.5 else "Medium" if avg_uncertainty < 1.0 else "Low"
                            st.metric("Reliability", reliability)
                        
                        # Uncertainty by horizon
                        if uncertainties.ndim > 1 and uncertainties.shape[1] > 1:
                            st.markdown("#### 📊 Uncertainty by Forecast Horizon")
                            
                            horizon_uncertainty = []
                            for h in range(min(uncertainties.shape[1], len(HORIZON_TUPLE))):
                                horizon_h = HORIZON_TUPLE[h]
                                unc_h = np.mean(uncertainties[:, h])
                                horizon_uncertainty.append({
                                    'Horizon': f'h+{horizon_h}',
                                    'Days Ahead': horizon_h,
                                    'Avg Uncertainty (°C)': round(unc_h, 3),
                                    'Confidence Level': "High" if unc_h < 0.5 else "Medium" if unc_h < 1.0 else "Low"
                                })
                            
                            uncertainty_df = pd.DataFrame(horizon_uncertainty)
                            st.dataframe(uncertainty_df, use_container_width=True, key=f"uncertainty_by_horizon_{model_name}")
                            
                            # Plot uncertainty by horizon
                            import plotly.graph_objects as go
                            fig_unc = go.Figure()
                            fig_unc.add_trace(go.Scatter(
                                x=[f"h+{h}" for h in HORIZON_TUPLE[:len(horizon_uncertainty)]],
                                y=[row['Avg Uncertainty (°C)'] for row in horizon_uncertainty],
                                mode='lines+markers',
                                name='Average Uncertainty',
                                line=dict(color='orange', width=3),
                                marker=dict(size=8)
                            ))
                            fig_unc.update_layout(
                                title="Prediction Uncertainty by Forecast Horizon",
                                xaxis_title="Forecast Horizon",
                                yaxis_title="Average Uncertainty (°C)",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_unc, use_container_width=True, key=f"uncertainty_plot_{model_name}")
                        
                        # Confidence intervals
                        st.markdown("#### 🎯 Available Confidence Intervals")
                        
                        ci_info = []
                        for key in prediction_intervals.keys():
                            if key.startswith('lower_') or key.startswith('upper_'):
                                confidence_level = key.split('_')[1] + '%'
                                ci_type = 'Lower Bound' if key.startswith('lower_') else 'Upper Bound'
                                ci_info.append({
                                    'Confidence Level': confidence_level,
                                    'Type': ci_type,
                                    'Available': '✅'
                                })
                        
                        if ci_info:
                            ci_df = pd.DataFrame(ci_info)
                            st.dataframe(ci_df, use_container_width=True, key=f"confidence_intervals_{model_name}")
                        
                        # Explanation
                        st.markdown("#### 💡 How to Interpret Uncertainty")
                        st.markdown("""
                        - **Low uncertainty** (< 0.5°C): High confidence predictions
                        - **Medium uncertainty** (0.5-1.0°C): Moderate confidence  
                        - **High uncertainty** (> 1.0°C): Lower confidence, use with caution
                        - **68% CI**: ~2/3 of actual values should fall within this range
                        - **95% CI**: ~19/20 of actual values should fall within this range
                        """)
                        
                else:
                    st.info("📊 Uncertainty intervals will be available after making predictions")
                    
            else:
                st.warning("📊 **Uncertainty Estimation**: ❌ DISABLED")
                st.info("💡 Enable uncertainty estimation in model training to get confidence intervals")
                
        except Exception as e:
            st.error(f"Could not load model for uncertainty analysis: {e}")
            st.info("Please ensure the model is properly trained and saved.")

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

        # Offer download of raw predictions CSV if available
        csv_path = forecast_data.get("csv_path")
        if csv_path and pathlib.Path(csv_path).exists():
            with open(csv_path, "rb") as _f:
                st.download_button(
                    label="Download predictions CSV",
                    data=_f.read(),
                    file_name=pathlib.Path(csv_path).name,
                    mime="text/csv",
                )

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
        
        # Log uncertainty estimation during forecasting
        if isinstance(mdl, MultiLGBMRegressor) and hasattr(mdl, 'uncertainty_estimation'):
            if mdl.uncertainty_estimation:
                _d(f"[FORECAST-UNCERTAINTY] Applied uncertainty estimation during forecasting (n_bootstrap={getattr(mdl, 'n_bootstrap_samples', 50)})")
                st.toast(f"🔮 Forecast with uncertainty: {getattr(mdl, 'n_bootstrap_samples', 50)} bootstrap samples", icon="🔮")
                

            else:
                _d(f"[FORECAST-UNCERTAINTY] No uncertainty estimation - point forecast only")
                st.toast("⚠️ Forecast without uncertainty - no confidence intervals", icon="⚠️")

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
            day_df = features.add_time_since_last_measurement(day_df)
            day_df = features.add_multi_lag_parallel(day_df, lags=(1,2,3,4,5,6,7,14,30))
            day_df = features.add_rolling_stats_parallel(day_df, window_days=7)

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
            
            # Log uncertainty for recursive forecasting (if multi-output)
            if isinstance(mdl, MultiLGBMRegressor) and hasattr(mdl, 'uncertainty_estimation') and preds.ndim == 2:
                if mdl.uncertainty_estimation:
                    _d(f"[RECURSIVE-UNCERTAINTY] Day {d}: Uncertainty estimation applied to {preds.shape[1]} horizons")
                else:
                    _d(f"[RECURSIVE-UNCERTAINTY] Day {d}: No uncertainty estimation - point predictions only")

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

    # ---------------- Persist predictions to CSV -----------------
    try:
        # Keep only essential columns for the user-facing CSV
        core_cols = [
            c for c in [
                "granary_id",
                "heap_id",
                "grid_x",
                "grid_y",
                "grid_z",
                "detection_time",
                "forecast_day",
                "predicted_temp",
            ]
            if c in future_df.columns
        ]
        out_df = future_df[core_cols].copy()

        # Ensure output directory exists
        out_dir = pathlib.Path("data/forecasts")
        out_dir.mkdir(parents=True, exist_ok=True)

        forecast_name = f"{pathlib.Path(model_name).stem}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        forecast_path = out_dir / forecast_name
        
        # Save as Parquet with snappy compression (60-80% smaller, 10x faster)
        from granarypredict.ingestion import save_granary_data
        parquet_path = save_granary_data(
            df=out_df,
            filepath=forecast_path,
            format='parquet',
            compression='snappy'
        )

        # Store path for UI download
        st.session_state["forecasts"][model_name]["parquet_path"] = str(parquet_path)
        _d(f"[FORECAST] Parquet written to {parquet_path}")
    except Exception as exc:
        _d(f"Could not write forecast CSV: {exc}")

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
    df = insert_calendar_gaps(df)
    _d("insert_calendar_gaps: added rows for missing dates")

    # -------------------------------------------------------------
    # 2️⃣ Interpolate numeric columns per sensor across the now-complete
    #    timeline so gap rows take the average of surrounding real values.
    # -------------------------------------------------------------
    df = interpolate_sensor_numeric(df)
    _d("interpolate_sensor_numeric: linear interpolation applied per sensor")

    # -------------------------------------------------------------
    # NEW – Subsample to one record every 4 hours per sensor to
    #       mitigate over-representation of densely sampled probes.


    # Subsampling per sensor removed for this pipeline version

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
    # Add time-since-last-measurement features for data quality assessment
    df = features.add_time_since_last_measurement(df)
    _d("add_time_since_last_measurement: added data quality temporal features")
    # lag features will be created inside add_multi_lag (includes 1-day)

    # -------------------------------------------------------------
    # 4️⃣ Extra temperature features (multi-lag, rolling stats, delta) - PARALLEL VERSION
    # -------------------------------------------------------------
    df = features.add_multi_lag_parallel(df, lags=(1,2,3,4,5,6,7,14,30))
    df = features.add_rolling_stats_parallel(df, window_days=7)
    _d("add_multi_lag_parallel & add_rolling_stats_parallel: extra features added with multiprocessing")
    
    # -------------------------------------------------------------
    # 4.5️⃣ Lean directional features for temperature movement prediction
    # -------------------------------------------------------------
    df = features.add_directional_features_lean(df)
    _d("add_directional_features_lean: 6 directional features added for trend prediction")
    
    # -------------------------------------------------------------
    # 4.6️⃣ Stability features for conservative temperature prediction - PARALLEL VERSION
    # -------------------------------------------------------------
    df = features.add_stability_features_parallel(df)
    _d("add_stability_features_parallel: 8 stability features added for conservative predictions with multiprocessing")
    
    # -------------------------------------------------------------
    # 4.7️⃣ Horizon-specific directional features for multi-model accuracy
    # -------------------------------------------------------------
    df = features.add_horizon_specific_directional_features(df, max_horizon=HORIZON_DAYS)
    _d(f"add_horizon_specific_directional_features: Enhanced directional features for {HORIZON_DAYS}-day forecasting added")

    # -------------------------------------------------------------
    # 5️⃣ Multi-horizon targets (1–3 days ahead)
    # -------------------------------------------------------------
    df = features.add_multi_horizon_targets(df, horizons=HORIZON_TUPLE)  # NEW
    _d("add_multi_horizon_targets: future target columns added")

    # Ensure group identifiers available for downstream splitting/evaluation
    df = assign_group_id(df)
    _d("assign_group_id: _group_id column added to dataframe")
    df = comprehensive_sort(df)
    _d("comprehensive_sort: dataframe sorted by granary/heap/grid/date")

    return df


# ---------------------------------------------------------------------
# Helper: insert rows for missing calendar dates per sensor
# ---------------------------------------------------------------------




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
    _d("_preprocess_cached: running full pipeline with parellel processing (no Streamlit cache)")
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
            _d("✅ Detected preprocessed file – loading directly, skipping heavy pipeline")
            if isinstance(uploaded_file, (str, pathlib.Path)):
                file_path = pathlib.Path(uploaded_file)
                if file_path.suffix.lower() == '.parquet':
                    df_fast = pd.read_parquet(uploaded_file)
                else:
                    df_fast = pd.read_csv(uploaded_file, encoding="utf-8")
            else:
                # Handle StreamlitUploadedFile
                if uploaded_file.name.lower().endswith('.parquet'):
                    df_fast = pd.read_parquet(uploaded_file)
                else:
                    df_fast = pd.read_csv(uploaded_file, encoding="utf-8")
            st.session_state["processed_df"] = df_fast.copy()
            return df_fast
        except Exception as exc:
            _d(f"Could not load preprocessed file fast-path: {exc}; falling back to pipeline")

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

        processed_dir = pathlib.Path("data/preloaded")
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = processed_dir / f"{orig_name}_processed.parquet"
        if not out_parquet.exists():
            from granarypredict.ingestion import save_granary_data
            save_granary_data(proc, out_parquet, format="parquet", compression="snappy")
            _d(f"💾 Saved processed Parquet to {out_parquet}")
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
        if "data/processed" in p.as_posix() or p.name.endswith("_processed.csv") or p.name.endswith("_processed.parquet"):
            return True
    elif hasattr(upload, "name"):
        name = upload.name
        if name.endswith("_processed.csv") or name.endswith("_processed.parquet"):
            return True
    return False


# ---------------------------------------------------------------------
# Helper: numeric interpolation per sensor across calendar-completed frame
# ---------------------------------------------------------------------


from granarypredict.cleaning_helpers import (
    insert_calendar_gaps,
    interpolate_sensor_numeric,
)


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

    unique_dates = sorted(df["_date"].dropna().unique())
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