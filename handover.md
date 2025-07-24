# SiloFlow 项目技术交接文档

## 📑 项目概览
- **项目名称**：SiloFlow - 智能粮食温度预测系统
- **技术栈**：Python, FastAPI, Streamlit, LightGBM, Polars, Dask
- **核心功能**：大规模粮食温度数据处理与机器学习预测
- **文档版本**：v2.0.0 (2025年7月)
- **维护团队**：数据科学团队

## 🏗️ 系统架构概览

### 整体架构
```
原始数据 → 数据摄取 → 数据清洗 → 特征工程 → 模型训练 → 预测服务 → 结果展示
    ↓           ↓          ↓          ↓          ↓          ↓          ↓
  CSV/API   ingestion   cleaning   features   multi_lgbm  FastAPI  Streamlit
```

### 技术选型说明
- **后端框架**：FastAPI（高性能异步API）
- **数据处理**：Polars（高性能）+ Pandas（兼容性）
- **机器学习**：LightGBM（梯度提升）+ 不确定性量化
- **前端界面**：Streamlit（快速原型开发）
- **大数据处理**：Dask（分布式）+ Vaex（十亿级数据）
- **模型优化**：Optuna（超参数调优）+ 模型压缩

### 数据流向
1. **数据摄取**：CSV/Parquet/API → 标准化格式
2. **数据处理**：清洗 → 特征工程 → 时间序列优化
3. **模型训练**：LightGBM → 不确定性量化 → 模型压缩
4. **预测服务**：FastAPI → 实时预测 → 结果存储
5. **结果展示**：Streamlit Dashboard → 可视化分析

## 📖 文档导航

### 快速开始
- [安装指南](#🚀-部署指南)
- [5分钟快速体验](#⚡-快速体验)
- [API接口文档](#📡-api接口文档)

### 核心模块
- [数据处理模块](#📂-数据处理模块)
- [机器学习模块](#🤖-机器学习模块)
- [API服务模块](#🌐-api服务模块)
- [前端界面模块](#💻-前端界面模块)

### 运维指南
- [部署指南](#🚀-部署指南)
- [性能优化](#⚡-性能优化指南)
- [故障排除](#🔧-常见问题与故障排除)

## 📂 核心模块目录

### 数据处理模块
- granarypredict/polars_adapter.py —— Polars/Pandas 适配器与高性能特征工程
- granarypredict/streaming_processor.py —— 超大规模数据集流式处理与特征工程  
- granarypredict/features.py —— 核心特征工程模块
- granarypredict/data_utils.py —— 数据处理工具集
- granarypredict/ingestion.py —— 数据摄取与标准化
- granarypredict/cleaning.py —— 数据清洗工具
- granarypredict/cleaning_helpers.py —— 清洗辅助工具

### 机器学习模块  
- granarypredict/model.py —— 模型训练与推理
- granarypredict/multi_lgbm.py —— 多目标LightGBM训练器
- granarypredict/evaluate.py —— 模型评估与指标
- granarypredict/optuna_cache.py —— 超参数优化缓存
- granarypredict/compression_utils.py —— 模型压缩工具

### API服务模块
- service/main.py —— FastAPI 服务主入口
- service/core.py —— 核心服务功能
- service/automated_processor.py —— 自动化处理器
- service/granary_pipeline.py —— 完整数据管道CLI工具
- service/routes/pipeline.py —— 管道API路由
- service/routes/train.py —— 训练API路由
- service/routes/forecast.py —— 预测API路由
- service/routes/health.py —— 健康检查路由

### 工具与配置模块
- granarypredict/config.py —— 全局配置管理
- service/utils/data_paths.py —— 数据路径管理
- service/utils/memory_utils.py —— 内存监控工具
- service/utils/silo_filtering.py —— 筒仓文件过滤工具
- service/utils/database_utils.py —— 数据库工具
- service/utils/validation_utils.py —— 配置验证工具

### 前端界面模块
- app/Dashboard.py —— Streamlit 主界面与完整数据科学工作流

### 测试与工具模块
- service/scripts/testing/testingservice.py —— 综合测试GUI界面
- service/scripts/client/ —— 客户端测试工具
- service/scripts/database/ —— 数据库操作脚本

---

---

## 🚀 部署指南

### 环境要求
- **Python版本**：3.9+ (推荐3.11)
- **内存要求**：建议16GB+（大数据处理场景）
- **存储空间**：建议100GB+（模型和数据存储）
- **GPU支持**：可选，支持CUDA加速训练
- **操作系统**：Windows/Linux/macOS

### 安装步骤

#### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv siloflow_env
source siloflow_env/bin/activate  # Linux/macOS
# 或 siloflow_env\Scripts\activate  # Windows

# 升级pip
pip install --upgrade pip
```

#### 2. 依赖安装
```bash
# 安装基础依赖
pip install -r requirements.txt

# GPU版本（可选）
pip install -r requirements-gpu.txt

# 验证安装
python -c "import lightgbm, polars, fastapi; print('Dependencies OK')"
```

#### 3. 配置初始化
```bash
# 创建配置文件
cp service/config/production_config.json.example service/config/production_config.json

# 编辑配置文件
nano service/config/production_config.json

# 初始化数据目录
python -c "from service.utils.data_paths import data_paths; data_paths.ensure_directories()"
```

#### 4. 服务启动
```bash
# 开发环境
python service/main.py

# 生产环境
uvicorn service.main:app --host 0.0.0.0 --port 8000 --workers 4

# 后台运行
nohup uvicorn service.main:app --host 0.0.0.0 --port 8000 > siloflow.log 2>&1 &
```

### 生产环境配置

#### Docker部署
```dockerfile
# Dockerfile示例
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 负载均衡配置（Nginx）
```nginx
upstream siloflow {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://siloflow;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## ⚡ 快速体验

### 5分钟快速上手
```python
# 1. 启动服务
# python service/main.py

# 2. 测试数据处理
import requests
import pandas as pd

# 创建测试数据
test_data = pd.DataFrame({
    'granary_id': ['ABC123'] * 100,
    'heap_id': ['H1'] * 100,
    'grid_x': range(100),
    'grid_y': [1] * 100,
    'grid_z': [1] * 100,
    'detection_time': pd.date_range('2025-01-01', periods=100, freq='H'),
    'temperature_grain': 20 + np.random.randn(100) * 2
})

# 保存测试文件
test_data.to_csv('test_data.csv', index=False)

# 3. 调用API
with open('test_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/pipeline',
        files={'file': f},
        data={'horizon': 7}
    )

print(response.json())
```

### Streamlit界面体验
```bash
# 启动Streamlit界面
streamlit run app/Dashboard.py

# 访问：http://localhost:8501
```

## 📡 API接口文档

### 核心API端点

#### 1. 完整预测管道
```http
POST /pipeline
Content-Type: multipart/form-data

参数：
- file: CSV/Parquet文件
- horizon: 预测时间范围（天，默认7）

响应：
{
  "status": "success",
  "granaries_processed": 1,
  "forecast_horizon_days": 7,
  "forecasts": {...}
}
```

#### 2. 数据处理
```http
POST /process
Content-Type: multipart/form-data

参数：
- file: CSV/Parquet文件

响应：
{
  "status": "success",
  "granaries_processed": 1,
  "results": {...}
}
```

#### 3. 模型训练
```http
POST /train

响应：
{
  "status": "success",
  "trained_models": [...],
  "metrics": {...}
}
```

#### 4. 预测服务
```http
POST /forecast
Content-Type: application/json

{
  "granary_id": "ABC123",
  "horizon": 7
}

响应：
{
  "predictions": [...],
  "confidence_intervals": {...},
  "uncertainty": [...]
}
```

#### 5. 健康检查
```http
GET /health

响应：
{
  "status": "healthy",
  "service": "SiloFlow API",
  "timestamp": "2025-07-23T10:00:00Z",
  "directories": {...}
}
```

### 数据格式要求

#### 输入CSV格式
```csv
granary_id,heap_id,grid_x,grid_y,grid_z,detection_time,temperature_grain
ABC123,H1,1,1,1,2025-01-01 00:00:00,20.5
ABC123,H1,1,1,2,2025-01-01 00:00:00,21.2
...
```

#### 预测结果格式
```json
{
  "granary_id": "ABC123",
  "predictions": [
    {
      "horizon": 1,
      "temperature": 20.8,
      "confidence_68": [20.3, 21.3],
      "confidence_95": [19.8, 21.8]
    }
  ]
}
```

## ⚡ 性能优化指南

### 数据处理优化

#### 大数据处理
```python
# 使用流式处理器处理大数据集
from granarypredict.streaming_processor import MassiveDatasetProcessor

processor = MassiveDatasetProcessor(
    chunk_size=100_000,    # 根据内存调整
    backend="polars",      # 最快的后端
    enable_dask=True       # 启用分布式处理
)

# 处理超大数据集
success = processor.process_massive_features(
    "huge_dataset.parquet",
    "processed_output.parquet"
)
```

#### 并行特征工程
```python
# 环境变量控制并行处理
import os
os.environ['SILOFLOW_MAX_WORKERS'] = '8'  # 设置工作进程数

from granarypredict.features import create_time_features
df = create_time_features(df, parallel=True)
```

### 模型训练优化

#### GPU加速
```python
# 自动GPU检测和配置
from granarypredict.multi_lgbm import MultiHorizonLGBMRegressor

model = MultiHorizonLGBMRegressor(
    device='auto',  # 自动选择GPU/CPU
    gpu_platform_id=0,
    gpu_device_id=0
)
```

#### Optuna缓存
```python
# 启用超参数缓存避免重复调优
from granarypredict.optuna_cache import OptunaParameterCache

cache = OptunaParameterCache("optuna_cache")
# 缓存会自动保存和加载最优参数
```

#### 模型压缩
```python
# 自动模型压缩
from granarypredict.compression_utils import save_compressed_model

save_compressed_model(
    model, 
    "model.joblib", 
    auto_compress=True  # 自动选择压缩策略
)
```

### 系统资源优化

#### 内存监控
```python
from service.utils.memory_utils import MemoryMonitor

monitor = MemoryMonitor()
with monitor.monitor_context("data_processing"):
    # 数据处理代码
    process_data()
```

#### 自动垃圾回收
```python
# 处理器自动内存管理
from service.automated_processor import AutomatedGranaryProcessor

processor = AutomatedGranaryProcessor()
# 自动内存监控和清理已内置
```

## 🔧 常见问题与故障排除

### 内存相关问题

#### 问题：处理大数据时内存溢出
```bash
# 症状
MemoryError: Unable to allocate array
```

**解决方案：**
```python
# 1. 调整chunk_size
processor = MassiveDatasetProcessor(chunk_size=50_000)  # 减小chunk

# 2. 启用流式处理
processor.backend = "polars"  # 使用内存高效的后端

# 3. 设置内存阈值
processor.memory_threshold = 60.0  # 降低内存阈值
```

#### 问题：Streamlit界面卡顿
**解决方案：**
```python
# 在streamlit_config.toml中设置
[server]
maxUploadSize = 1000  # 限制上传文件大小(MB)
enableStaticServing = false

# 在代码中分块处理
import streamlit as st
if 'large_data' not in st.session_state:
    st.session_state.large_data = process_in_chunks(data)
```

### 模型训练问题

#### 问题：训练速度慢
**解决方案：**
```python
# 1. 启用GPU
model = MultiHorizonLGBMRegressor(device='gpu')

# 2. 减少Optuna试验次数
study = optuna.create_study()
study.optimize(objective, n_trials=50)  # 减少试验次数

# 3. 使用预调优参数
model = train_lightgbm(X, y, use_tuned_params=True)
```

#### 问题：模型文件过大
**解决方案：**
```python
# 自动模型压缩
save_compressed_model(model, "model.joblib", 
                     compression='lzma', level=9)

# 检查压缩效果
from granarypredict.compression_utils import analyze_compression
analyze_compression("model.joblib")
```

### 部署问题

#### 问题：服务启动失败
```bash
# 症状
uvicorn.error.exc: Error loading application
```

**解决方案：**
```bash
# 1. 检查依赖
pip install --upgrade -r requirements.txt

# 2. 检查端口占用
netstat -tlnp | grep 8000

# 3. 检查权限
chmod +x service/main.py

# 4. 查看详细错误
python service/main.py  # 直接运行查看错误
```

#### 问题：API响应超时
**解决方案：**
```python
# 增加超时设置
uvicorn service.main:app --timeout-keep-alive 300

# 或在nginx中设置
proxy_read_timeout 300s;
proxy_connect_timeout 300s;
```

### 数据问题

#### 问题：数据格式不兼容
**解决方案：**
```python
# 使用标准化工具
from granarypredict.ingestion import standardize_granary_csv

df = standardize_granary_csv(df)  # 自动标准化格式
```

#### 问题：特征工程失败
**解决方案：**
```python
# 检查数据完整性
from granarypredict.features import validate_data_requirements

issues = validate_data_requirements(df)
if issues:
    print("数据问题:", issues)
```

---

### 文件定位与作用
本文件是 SiloFlow 项目的核心特征工程模块，提供完整的时间序列特征生成、空间特征、滞后特征、滚动统计等功能，支持并行计算和多种优化策略。

### 主要功能模块

#### 1. 时间特征生成（第50-100行）
- `create_time_features()` 函数：生成时间相关特征
- 自动检测时间戳列：支持detection_time、batch、timestamp等常见列名
- 基础时间特征：年、月、日、小时
- 周期性编码：月份和小时的sin/cos变换，避免数值跳跃
- 日历特征：年内天数、周数、周末标识
- 新增周期性编码：年内天数和周数的sin/cos变换

#### 2. 空间特征处理（第102-108行）
- `create_spatial_features()` 函数：处理网格坐标特征
- 移除冗余的grid_index以避免重复
- 保持向后兼容性

#### 3. 类别编码（第111-116行）
- `encode_categoricals()` 函数：将对象和类别类型转换为整数编码
- 使用pandas的category codes进行标签编码

#### 4. 特征目标选择（第119-141行）
- `select_feature_target()` 函数：分离特征矩阵X和目标变量y
- 自动排除目标列、时间戳、标识符等
- 处理预测场景（所有目标为空）
- 自动应用类别编码

#### 5. 时间间隔特征（第165-210行）
- `add_time_since_last_measurement()` 函数：测量时间间隔特征
- 计算自上次测量以来的小时数
- 按传感器分组（granary_id + heap_id + grid坐标）
- 处理首次测量（设为0而非NaN）

#### 6. 传感器滞后特征（第215-270行）
- `add_sensor_lag()` 函数：添加同一传感器的历史温度
- 支持任意天数滞后（默认1天）
- 使用日期级别合并，适应不规则采样频率
- 基于确切日期匹配而非固定采样间隔

#### 7. 并行处理支持（第25-35行）
- 全局设置：支持环境变量控制并行处理
- 保守的最大工作进程数（最多4个）
- Streamlit集成：支持toast通知显示处理进度

### 性能优化特性
- **并行计算**：支持多进程并行特征工程
- **内存管理**：保守的工作进程限制，防止内存溢出
- **容错机制**：异常时自动降级到单进程处理
- **进度监控**：集成Streamlit通知和详细日志

### 典型用法与完整工作流

#### 基础特征工程
```python
from granarypredict.features import create_time_features, add_sensor_lag
df = create_time_features(df, timestamp_col="detection_time")
df = add_sensor_lag(df, temp_col="temperature_grain", lag_days=1)
```

#### 完整数据处理与预测工作流
```python
# 完整的数据处理与预测工作流示例
import pandas as pd
from granarypredict import features, model, data_utils
from service.automated_processor import AutomatedGranaryProcessor

# 1. 数据加载与预处理
df = pd.read_csv("raw_temperature_data.csv")
df = data_utils.comprehensive_sort(df)
df = features.create_time_features(df)
df = features.add_sensor_lag(df, lag_days=1)

# 2. 特征工程优化
df = features.create_spatial_features(df)
df = features.add_time_since_last_measurement(df)

# 3. 特征选择与编码
X, y = features.select_feature_target(df)
print(f"特征维度: {X.shape}, 目标维度: {y.shape}")

# 4. 模型训练
trained_model, metrics = model.train_lightgbm(X, y)
print(f"训练指标: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

# 5. 预测
predictions = trained_model.predict(X_new)

# 6. 完整自动化流程（推荐生产环境使用）
processor = AutomatedGranaryProcessor()
results = await processor.process_all_granaries("input.csv")
```

#### 大数据并行处理示例
```python
import os
from granarypredict.features import create_time_features

# 设置并行处理
os.environ['SILOFLOW_MAX_WORKERS'] = '8'

# 并行特征工程
df = create_time_features(df, parallel=True)

# 流式处理大数据集
from granarypredict.streaming_processor import MassiveDatasetProcessor

processor = MassiveDatasetProcessor(
    chunk_size=100_000,
    backend="polars",
    enable_dask=True
)

success = processor.process_massive_features(
    "huge_dataset.parquet",
    "processed_features.parquet",
    feature_functions=[
        lambda df: create_time_features(df),
        lambda df: add_sensor_lag(df, lag_days=1)
    ]
)
```

## 🤖 机器学习模块详解

## granarypredict/features.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的全局配置管理模块，定义项目目录结构、温度阈值、API端点等核心配置。

### 主要配置项

#### 1. 目录结构配置（第8-16行）
- `ROOT_DIR`：项目根目录，基于当前文件位置自动确定
- `DATA_DIR`：数据根目录
- `RAW_DATA_DIR`：原始数据目录
- `PROCESSED_DATA_DIR`：处理后数据目录  
- `MODELS_DIR`：模型保存目录
- 自动创建目录：导入时自动创建不存在的目录

#### 2. 温度阈值配置（第18-31行）
- `ALERT_TEMP_THRESHOLD`：通用预警温度阈值（默认28°C）
- `GRAIN_ALERT_THRESHOLDS`：分粮食类型的安全温度阈值
  - 中晚籼稻、早籼稻：28°C
  - 粳稻：26°C
  - 玉米：30°C
  - 大豆：25°C
  - 小麦：27°C

#### 3. API端点配置（第33-35行）
- `METEOROLOGY_API_BASE`：气象API基础URL
- `COMPANY_API_BASE`：公司粮仓API基础URL
- 支持环境变量覆盖

#### 4. 环境变量支持（第5-6行）
- 使用python-dotenv加载.env文件
- 所有配置项都支持环境变量覆盖

### 设计特点
- **自动化初始化**：目录结构自动创建
- **灵活配置**：环境变量优先，支持不同部署环境
- **类型安全**：明确的类型注解和默认值
- **扩展性**：易于添加新的配置项

---

## granarypredict/model.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的模型训练与推理模块，提供多种机器学习算法的训练函数和模型管理功能。

### 主要模型算法

#### 1. 随机森林训练（第23-46行）
- `train_random_forest()` 函数：训练RandomForestRegressor
- 默认200棵树，自动并行处理
- 时间序列友好的数据分割（shuffle=False）
- 返回模型和评估指标（MAE、RMSE）

#### 2. 梯度提升训练（第49-78行）
- `train_gb_models()` 函数：训练梯度提升回归器
- 支持HistGradientBoosting和标准GradientBoosting
- 使用时间序列交叉验证
- 灵活的超参数配置

#### 3. LightGBM训练（第81-135行）
- `train_lightgbm()` 函数：训练LightGBM回归器
- 预调优的超参数（基于多数据集优化）：
  - n_estimators: 1185
  - learning_rate: 0.033
  - max_depth: 7
  - num_leaves: 24
- 集成压缩参数以减小模型大小
- 全数据集训练以最大化性能

#### 4. 模型保存与加载（第138-237行）
- `save_model()` 函数：智能模型保存
  - 自适应压缩：根据模型类型和大小选择压缩策略
  - 支持压缩和非压缩格式
  - 详细的保存统计信息
- `load_model()` 函数：健壮的模型加载
  - 多策略加载：依次尝试压缩、标准、兼容性模式
  - 全面的错误处理和日志记录
  - 自动fallback机制

### 核心设计理念
- **开箱即用**：预调优参数，无需手动调参
- **健壮性**：多重错误处理和fallback机制
- **性能优化**：模型压缩、并行训练
- **可扩展性**：易于添加新算法

---

## service/main.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的 FastAPI 服务主入口，提供轻量级的应用编排，将具体路由实现委托给模块化的routes子包。

### 主要组件

#### 1. 应用初始化（第15-24行）
- FastAPI应用实例化
- 应用元数据：标题、描述、版本
- 版本2.0.0，表示架构重构后的模块化版本

#### 2. CORS中间件（第26-33行）
- 开放的CORS策略（开发环境）
- 支持所有源、方法、头部
- 生产环境需要锁定特定域名

#### 3. 路由包含（第38-40行）
- 导入并包含所有模块化路由
- 委托给routes子包进行统一管理

#### 4. 服务启动（第45-47行）
- Uvicorn ASGI服务器
- 监听所有地址的8000端口
- 支持热重载开发模式

### 架构优势
- **模块化**：路由逻辑分离到独立模块
- **可维护性**：slim orchestrator模式
- **可扩展性**：易于添加新路由模块
- **开发友好**：热重载支持

---

## service/granary_pipeline.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的完整数据管道CLI工具，提供从数据摄取到模型训练预测的端到端自动化流程，支持Polars高性能处理和Optuna超参数优化。

### 主要功能模块

#### 1. Polars集成优化（第40-75行）
- 自动检测Polars可用性并启用性能优化
- `load_data_optimized()`：智能后端选择，大文件优先使用Polars
- 详细的加载日志和性能反馈
- 自动fallback到pandas以确保兼容性

#### 2. 高性能特征工程（第78-140行）
- `add_lags_optimized()`：Polars优化的滞后特征计算
  - 向量化操作，显著提升性能
  - 内存高效的分组计算
  - 支持多滞后期[1,2,3,7]天
  - 自动生成差值特征

#### 3. 时间特征优化（第143-185行）  
- `create_time_features_optimized()`：Polars优化的时间特征
  - 5-10倍性能提升
  - 完整的周期性编码
  - 年、月、日、小时基础特征
  - sin/cos变换避免数值跳跃

#### 4. CLI命令结构
- **ingest**：数据摄取，排序去重标准化
- **preprocess**：数据预处理，清洗插值特征工程  
- **train**：模型训练，支持Optuna调优
- **forecast**：多时间范围预测生成

#### 5. 训练配置特性
- **GPU自动检测**：自动使用GPU，CPU fallback
- **Optuna调优**：默认启用超参数优化
- **分位数回归**：alpha=0.5以提升MAE性能
- **锚定日早停**：7天连续预测准确性
- **预测期平衡**：递增策略改善长期预测
- **保守模式**：3倍稳定性+2倍方向性增强
- **95/5分割**：内部验证后100%数据训练

### 企业级特性
- **模块化设计**：所有步骤可独立导入和自动化
- **云部署就绪**：支持容器化和微服务架构
- **详细日志**：完整的性能和错误日志
- **资源优化**：智能内存管理和并行处理

### 典型用法
```bash
# 完整流程：Optuna调优训练
python granary_pipeline.py train --granary ABC123

# 自定义调优参数
python granary_pipeline.py train --granary ABC123 --trials 50 --timeout 300

# 固定参数快速训练
python granary_pipeline.py train --granary ABC123 --no-tune

# 多时间范围预测
python granary_pipeline.py forecast --granary ABC123 --horizon 7
```

---

## granarypredict/data_utils.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的数据处理工具集，提供数据排序、分组标识和数据压缩功能，支持多种格式的高效数据存储和加载。

### 主要功能模块

#### 1. 标准化数据排序（第24-34行）
- `comprehensive_sort()` 函数：按规范层次结构排序数据
- 排序优先级：granary_id → heap_id → grid_x → grid_y → grid_z → detection_time
- 缺失列自动忽略，保证向后兼容性
- 确保同一传感器内的时间顺序正确

#### 2. 分组标识生成（第37-48行）
- `assign_group_id()` 函数：为物理筒仓创建唯一标识符
- 智能回退机制：granary_id+heap_id → granary_id → heap_id → "all"
- 支持自定义列名（默认_group_id）

#### 3. 压缩CSV处理（第67-131行）
- `save_compressed_csv()` 函数：保存压缩CSV文件
  - 支持gzip、bz2、zip、xz压缩格式
  - 自动添加压缩文件扩展名
  - 详细的压缩统计信息和比率计算
  - 自动创建目录结构
- `read_compressed_csv()` 函数：读取压缩CSV文件
  - 自动检测压缩格式
  - 基于文件扩展名的智能识别

#### 4. Parquet格式支持（第168-220行）
- `save_parquet()` 函数：保存高效Parquet文件
  - 默认snappy压缩，比CSV效率更高
  - 支持多种压缩算法（snappy、gzip、brotli）
- `read_parquet()` 函数：读取Parquet文件
  - 快速加载和查询性能

#### 5. 压缩建议系统（第245-303行）
- `get_compression_recommendation()` 函数：智能压缩建议
  - 基于数据大小的格式推荐（>10MB推荐Parquet）
  - 详细的压缩选项对比
  - 性能vs压缩比权衡分析

### 性能特性
- **格式优化**：大文件推荐Parquet，小文件使用CSV
- **压缩智能**：多种压缩算法满足不同需求
- **统计反馈**：详细的压缩比和文件大小信息
- **容错机制**：健壮的文件检测和错误处理

---

## granarypredict/multi_lgbm.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的多目标LightGBM训练器，提供原生不确定性量化、保守型指标和锚定日早停机制，专为粮食温度预测优化。

### 核心功能模块

#### 1. 原生不确定性量化（第14-90行）
- `compute_prediction_intervals()` 函数：计算预测区间
  - 支持多个置信水平（68%、95%）
  - 基于正态分布的z-score计算
  - 按时间视野分析不确定性统计
- `estimate_model_uncertainty()` 函数：模型不确定性估计
  - Bootstrap聚合方法，25次采样优化性能
  - 校准噪声因子（5%基础噪声）
  - 时间视野递增的不确定性缩放（1.0x到2.5x）
  - 现实约束：最小0.05°C，最大0.5°C

#### 2. 保守型评价指标（第190-250行）
- `conservative_mae_metric()` 函数：保守型MAE指标
  - 标准MAE + 稳定性惩罚 + 热惯性惩罚 + 方向一致性惩罚
  - 惩罚过度温度变化，符合粮食储存物理特性
  - 时间视野间的渐进惩罚权重
- `directional_mae_metric()` 函数：轻量级方向MAE
  - MAE + 方向准确性惩罚（最大0.1°C）
  - 保持训练速度的同时改善方向预测

#### 3. 锚定日早停机制（第258-380行）
- `AnchorDayEarlyStoppingCallback` 类：优化的早停回调
  - 7天连续预测准确性评估
  - 真实业务场景模拟：某天预测评估1-7天后实际温度
  - 性能优化：每10次迭代检查一次（75%速度提升）
  - 缓存机制减少重复计算

### 物理约束与校准
- **热惯性建模**：大幅度温度变化的惩罚机制
- **方向一致性**：避免温度预测方向的急剧变化
- **不确定性校准**：基于实际观察数据的0.05-0.5°C范围
- **时间衰减**：长期预测的递增不确定性

### 企业级特性
- **快速训练**：优化的早停和检查间隔
- **概率预测**：完整的置信区间输出
- **物理一致性**：符合粮食储存温度变化规律
- **健壮回退**：自定义逻辑失败时的标准早停

---

## granarypredict/evaluate.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的模型评估模块，提供时间序列友好的交叉验证和性能指标计算功能。

### 核心功能

#### 1. 时间序列交叉验证（第16-46行）
- `time_series_cv()` 函数：时间序列感知的交叉验证
- 使用sklearn的TimeSeriesSplit确保时间顺序
- 支持自定义分割数量（默认5折）
- 自动处理缺失值预测
- 计算MAE和RMSE指标
- 在全数据上重新训练模型用于后续使用

### 设计特点
- **时间感知**：不打乱数据顺序，符合时间序列特性
- **健壮评估**：处理预测中的缺失值
- **生产就绪**：交叉验证后自动在全数据上训练
- **轻量高效**：简洁的评估流程

---

## granarypredict/optuna_cache.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的Optuna超参数优化缓存系统，自动保存和加载最优超参数以避免重复优化。

### 主要功能模块

#### 1. 智能缓存键生成（第24-65行）
- `_generate_cache_key()` 方法：生成唯一缓存标识
- 多维度指纹：CSV文件名、数据形状、列类型、统计特征、模型配置
- MD5哈希确保唯一性和一致性
- 数据变化检测：前5个数值列的基础统计

#### 2. 参数保存与加载（第67-158行）
- `save_optimal_params()` 方法：保存优化结果
  - 完整的优化信息：参数、最优值、试验次数
  - 数据上下文：形状、列名、数据类型
  - 时间戳记录
- `load_optimal_params()` 方法：智能参数加载
  - 详细的调试日志
  - 数据一致性验证
  - 缓存文件列表显示

#### 3. 缓存管理（第160-201行）
- `clear_cache()` 方法：选择性清理缓存
- 支持全局清理或特定文件清理
- 模式匹配灵活清理策略

### 企业级特性
- **零重复优化**：相同数据和配置自动复用结果
- **数据感知**：数据变化时自动失效缓存
- **调试友好**：详细的缓存加载日志
- **存储高效**：JSON格式，易于查看和管理

---

## service/core.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 服务层的核心模块，提供单例对象管理、路径初始化和共享实用程序。

### 主要组件

#### 1. 路径管理（第10-16行）
- 动态路径解析：确保granarypredict包可导入
- 自动路径插入，解决模块依赖问题

#### 2. 单例处理器（第18-26行）
- 全局AutomatedGranaryProcessor实例
- 所有API路由共享同一处理器实例
- 数据路径自动初始化和验证

### 设计优势
- **单例模式**：避免重复初始化，提升性能
- **路径安全**：自动解决模块导入问题
- **集中管理**：所有服务共享核心对象
- **启动验证**：确保目录结构正确

---

## service/utils/data_paths.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 服务的数据路径管理器，提供中心化的目录structure管理，确保所有脚本使用一致的文件位置。

### 核心功能模块

#### 1. 配置管理（第15-35行）
- JSON配置文件加载，支持fallback到默认配置
- 全面的目录类型定义：granaries、processed、models、forecasts等
- 可配置的数据根目录结构

#### 2. 路径访问接口（第37-108行）
- 统一的路径获取接口：`get_path(path_type)`
- 专用访问器：`get_granaries_dir()`、`get_models_dir()`等
- 自动目录创建，确保路径存在

#### 3. 文件路径生成（第110-140行）
- `get_granary_file()`：粮仓数据文件路径
- `get_processed_file()`：处理后数据路径
- `get_model_file()`：模型文件路径（支持压缩）
- `get_forecast_file()`：预测结果路径（支持时间戳）

#### 4. 目录管理（第142-208行）
- `ensure_directories()`：确保所有目录存在
- `list_granaries()`：列出可用粮仓
- 目录状态检查和验证

### 设计特点
- **中心化管理**：所有路径配置统一管理
- **配置驱动**：JSON文件配置，易于修改
- **自动创建**：按需创建目录结构
- **类型安全**：明确的路径类型和验证

---

## service/routes/pipeline.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 服务的管道API路由模块，提供数据处理和完整预测管道的HTTP接口。

### 主要API端点

#### 1. 数据处理端点（第21-88行）
- `POST /process`：仅数据处理端点
- 功能范围：数据摄取 + 预处理，不包含训练和预测
- 支持CSV和Parquet文件上传
- 按粮仓分割数据并进行特征工程
- 1小时超时限制，适合中等规模数据

#### 2. 完整管道端点（第91-150行）
- `POST /pipeline`：端到端管道处理
- 功能范围：数据摄取 → 预处理 → 训练 → 预测
- 支持自定义预测时间视野（默认7天）
- 3小时超时限制，适合大规模处理
- 自动组合所有粮仓的预测结果

### 核心特性
- **异步处理**：使用asyncio支持并发操作
- **文件上传**：临时文件管理和自动清理
- **超时保护**：防止长时间运行阻塞服务
- **详细响应**：完整的处理状态和错误信息
- **CSV导出**：预测结果自动格式化为CSV

### 企业级特性
- **健壮错误处理**：全面的异常捕获和HTTP状态码
- **资源管理**：临时文件自动清理
- **可观测性**：详细的日志记录和处理统计
- **灵活配置**：支持不同处理模式和参数

---

## 🧪 测试与工具模块详解

## service/scripts/testing/testingservice.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的综合测试GUI界面，提供完整的图形化测试和管理功能，是系统测试、调试和运维的核心工具。

### 主要功能模块

#### 1. 系统架构与设计
- **多标签界面**：集成6个功能模块的现代化GUI
- **响应式设计**：支持高分辨率显示，自适应窗口大小
- **现代化样式**：使用ttk主题和自定义样式，提供专业的用户体验
- **后台处理**：所有耗时操作均在后台线程执行，界面不卡顿

#### 2. HTTP服务测试模块（🌐 HTTP Service Testing）
**核心功能：**
- **服务配置**：支持本地/远程服务切换，一键URL配置
- **连接测试**：实时健康检查，状态指示器显示连接状态
- **文件上传**：支持CSV/Parquet文件选择和上传
- **端点测试**：完整API端点测试（/pipeline、/process、/train、/forecast、/models、/health）
- **响应展示**：格式化显示API响应，支持JSON/CSV/错误信息

**使用场景：**
```python
# 典型测试工作流
1. 选择服务URL（本地8000端口或远程服务）
2. 测试连接确保服务可用
3. 选择测试数据文件
4. 选择API端点进行测试
5. 查看详细响应和状态信息
```

#### 3. 远程客户端测试模块（🌍 Remote Client Testing）
**核心功能：**
- **远程服务配置**：支持任意远程服务URL配置
- **批量端点测试**：自动测试所有API端点
- **完整测试套件**：一键运行全面的接口测试
- **测试报告生成**：自动生成详细的测试报告

**企业部署价值：**
- 跨网络环境测试
- 生产环境验证
- 持续集成支持
- 自动化测试报告

#### 4. 简单数据检索模块（📊 Simple Retrieval）
**核心功能：**
- **数据库连接**：直接连接生产数据库获取实时数据
- **粮仓/筒仓选择**：智能下拉选择，支持自动填充
- **日期范围配置**：灵活的时间范围选择
- **批量处理**：自动处理所有筒仓数据
- **进度监控**：实时显示检索进度和状态

**典型工作流：**
```python
# 数据检索步骤
1. 配置数据库连接
2. 获取可用粮仓和筒仓列表
3. 选择目标粮仓和筒仓
4. 设置日期范围
5. 执行数据检索
6. 查看输出文件和日志
```

#### 5. 生产管道模块（🚀 Production Pipeline）
**核心功能：**
- **完整数据管道**：从数据检索到模型训练的端到端处理
- **配置文件管理**：支持生产环境配置文件
- **阶段选择**：可选择执行的管道阶段（检索/预处理/训练）
- **系统监控**：实时资源使用监控
- **性能优化**：支持大规模生产环境处理

**企业级特性：**
- 生产环境就绪的数据管道
- 系统资源监控和优化
- 可配置的处理策略
- 详细的处理日志和统计

#### 6. 数据库探索模块（🗄️ Database Explorer）
**核心功能：**
- **三级联动选择**：粮仓 → 筒仓 → 日期范围的智能级联
- **数据库结构探索**：完整的数据库结构可视化
- **连接测试**：数据库连接验证和诊断
- **元数据导出**：数据库结构和统计信息导出

**数据管理价值：**
- 快速了解数据分布
- 数据质量评估
- 存储规划支持
- 数据迁移规划

#### 7. 批量处理模块（🔄 Batch Processing）
**核心功能：**
- **文件夹批处理**：支持整个文件夹的批量操作
- **单一动作模式**：专注于单一处理步骤的批量执行
- **进度监控**：实时显示批处理进度
- **错误处理**：完善的错误捕获和恢复机制

**处理选项：**
- **数据排序**：批量数据标准化和排序
- **数据处理**：特征工程和数据清洗
- **模型训练**：批量模型训练
- **预测生成**：批量预测结果生成

#### 8. 日志监控模块（📋 Logs & Monitoring）
**核心功能：**
- **系统日志聚合**：所有操作的统一日志显示
- **实时监控**：实时显示系统状态和操作进度
- **操作指南**：内置的使用指南和最佳实践
- **故障诊断**：详细的错误信息和故障排除建议

### 技术架构特点

#### 现代化GUI设计
```python
# 样式系统示例
self.style.configure(
    'Primary.TButton',
    background='#2E86AB',
    foreground='black',
    borderwidth=0,
    focuscolor='none',
    font=('Segoe UI', 9, 'bold'),
    padding=(10, 8)
)
```

#### 多线程异步处理
- **非阻塞UI**：所有耗时操作都在后台线程执行
- **进度反馈**：实时更新操作进度和状态
- **错误处理**：完善的异常捕获和用户提示

#### 可滚动界面设计
- **大内容支持**：复杂标签页支持滚动显示
- **响应式布局**：自适应不同屏幕分辨率
- **用户体验优化**：鼠标滚轮支持，快捷键操作

### 典型使用场景

#### 开发环境测试
```bash
# 启动测试界面
python service/scripts/testing/testingservice.py

# 开发工作流
1. HTTP服务测试 - 验证API功能
2. 数据检索 - 获取测试数据
3. 批量处理 - 验证处理管道
4. 生产管道 - 端到端测试
```

#### 生产环境运维
```bash
# 生产环境监控工作流
1. 健康检查 - 验证服务状态
2. 数据库探索 - 检查数据状态
3. 系统监控 - 查看资源使用
4. 批量处理 - 执行维护任务
```

#### 故障排除与诊断
```bash
# 故障排除工作流
1. 连接测试 - 验证网络和服务
2. 日志查看 - 分析错误信息
3. 数据验证 - 检查数据完整性
4. 系统资源 - 确认性能问题
```

### 企业级价值

#### 1. 降低运维成本
- **图形化操作**：降低命令行操作门槛
- **一站式管理**：集成所有必要的运维功能
- **错误预防**：智能提示和验证机制

#### 2. 提升开发效率
- **快速测试**：一键API测试和验证
- **数据获取**：简化数据检索和处理流程
- **调试支持**：详细的日志和状态信息

#### 3. 支持团队协作
- **标准化流程**：统一的操作界面和流程
- **知识传承**：内置的使用指南和最佳实践
- **培训支持**：直观的界面降低学习成本

### 配置和维护

#### 系统要求
- **Python环境**：支持Python 3.9+
- **依赖包**：tkinter、requests、pandas等
- **权限要求**：数据库访问、文件系统读写

#### 配置文件
- **数据库配置**：streaming_config.json
- **生产配置**：production_config.json
- **客户端配置**：client_config.json

#### 维护建议
- **定期更新**：跟随主系统版本更新
- **配置同步**：确保配置文件与生产环境一致
- **权限管理**：合理控制数据库和文件访问权限

### 扩展性设计
- **模块化架构**：易于添加新的测试模块
- **配置驱动**：通过配置文件扩展功能
- **插件支持**：支持自定义测试脚本集成
- **API兼容**：自动适配API版本变化

这个测试界面是 SiloFlow 项目不可或缺的运维工具，为开发、测试、部署和维护提供了完整的图形化支持。

---

## granarypredict/streaming_processor.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目用于超大规模数据集（上亿行）流式处理的核心模块，支持分块处理、自动内存管理、流式特征工程、增量模型训练、Dask/Vaex/Polars 等多后端大数据处理。

### 逐行代码解读

#### 1. 文件头与导入（第1-48行）
- 文档字符串说明文件用途：处理上亿行数据，提供分块处理、自动内存管理、流式特征工程、增量训练、多后端支持
- 导入必需库：logging、gc、pathlib、typing、pandas、numpy、contextlib
- 可选依赖导入：dask（分布式处理）、vaex（十亿级数据）、pyarrow（高效parquet）、polars（高性能）
- 各依赖都有 HAS_* 标志和异常处理，确保缺失时能正常降级

#### 2. MassiveDatasetProcessor 类初始化（第50-97行）
- 类注释说明功能：上亿行数据高效处理，自动分块内存管理，流式特征工程，多后端支持，增量训练
- `__init__` 方法参数：chunk_size（初始分块大小）、memory_threshold_percent（内存阈值）、backend（后端选择）、enable_dask（启用分布式）、n_workers（工作进程数）
- 动态内存管理：设置最小/最大分块大小，当前分块大小会根据内存压力自动调整
- 性能跟踪：记录已处理行数和分块数
- 详细日志输出配置信息

#### 3. 后端选择与系统配置（第99-113行）
- `_select_backend` 方法：自动选择最佳后端，优先级 polars > vaex > dask > pandas
- `_get_cpu_count` 方法：安全获取CPU核心数，异常时默认4核

#### 4. Dask 分布式处理（第115-135行）
- `dask_cluster` 上下文管理器：创建本地Dask集群用于大规模处理
- 配置LocalCluster：工作进程数、每进程线程数、内存限制、禁用dashboard以提升性能
- 异常处理和资源清理确保稳定性

#### 5. 内存监控与动态调整（第137-156行）
- `_check_memory_usage` 方法：使用psutil检查内存使用率是否低于阈值
- `_adjust_chunk_size` 方法：根据内存压力动态调整分块大小，内存高时减小（×0.75），内存正常时增大（×1.25）
- 确保分块大小在合理范围内（1万-100万行）

#### 6. 多后端数据读取（第158-205行）
- `read_massive_dataset` 主方法：根据选择的后端调用相应读取方法
- 支持Parquet和CSV格式，Parquet推荐用于大数据
- 各后端方法都返回pandas DataFrame迭代器以保持兼容性

#### 7. Pandas后端实现（第207-224行）
- `_read_with_pandas` 方法：使用PyArrow进行内存高效的Parquet分批读取
- CSV使用pandas内置分块读取
- 每个分块后进行内存调整和垃圾回收

#### 8. Polars后端实现（第226-246行）
- `_read_with_polars` 方法：最快选项，使用lazy reading实现内存高效
- 先获取总行数，然后分块slice和collect
- 转换为pandas以保持兼容性，并进行内存清理

#### 9. Vaex和Dask后端实现（第248-280行）
- `_read_with_vaex` 方法：适合十亿级数据，直接按索引分块
- `_read_with_dask` 方法：良好的并行处理，按分区处理
- 都包含内存清理和进度跟踪

#### 10. 流式特征工程（第310-420行）
- `process_massive_features` 主方法：完整的流式特征工程管道
- 默认特征函数：基础时间特征、关键滞后特征、滑动窗口特征
- 支持Parquet增量写入或CSV追加模式
- 详细进度日志和内存清理

#### 11. 内置特征工程方法（第422-461行）
- `_add_basic_time_features`：年月日时、星期、是否周末等时间特征
- `_add_essential_lags`：1日滞后和温度变化，按传感器分组
- `_add_rolling_features`：3日滑动均值和标准差，优化用于流式处理

#### 12. 工具函数（第464-558行）
- `create_massive_processing_pipeline`：主入口函数，快速构建大数据处理管道
- `estimate_memory_requirements`：内存需求估算，返回推荐分块大小和处理策略
- 采样数据估算单行内存使用，计算总内存需求，给出内存内/流式处理建议
- `__main__` 示例：展示完整使用流程

### 主要功能概述
- 支持上亿行数据的分块流式处理
- 自动内存监控和分块大小调整
- 多后端支持（Pandas、Dask、Vaex、Polars）
- 流式特征工程，内存占用极低
- 支持分布式并行处理
- 内存需求估算和处理策略推荐

### 典型用法
```python
from granarypredict.streaming_processor import create_massive_processing_pipeline
success = create_massive_processing_pipeline(
    input_path='massive_dataset.parquet',
    output_path='processed_massive_dataset.parquet',
    chunk_size=200_000,
    backend='auto'
)
```

### 性能与健壮性
- 极大提升超大数据集处理能力，支持自动分块与内存调节
- 优先使用高性能后端（polars/vaex/dask），速度远超pandas
- 所有操作都有异常处理和降级机制
- 详细日志记录，便于监控和问题定位

---

## service/utils/memory_utils.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的内存监控工具模块，提供跨平台的内存使用量日志记录功能。

### 逐行代码解读

#### 1. 文件头与导入（第1-8行）
- 文档字符串说明文件用途：SiloFlow的内存监控工具
- 导入必需库：resource（POSIX系统内存）、logging、os
- 配置logger用于日志输出

#### 2. log_memory_usage函数（第10-21行）
- 参数prefix：用于日志的描述性前缀
- 跨平台内存获取：
  - POSIX系统（Linux/macOS）：使用resource.getrusage()获取最大常驻集大小，单位转换为MB
  - Windows系统：使用psutil获取当前进程内存信息，转换为MB
- 异常处理：任何错误都会记录警告而不会中断程序

### 主要功能
- 跨平台内存监控（Linux、macOS、Windows）
- 安全的错误处理，不会影响主程序运行
- 简洁的API，便于在代码各处插入内存监控点

### 典型用法
```python
from service.utils.memory_utils import log_memory_usage
log_memory_usage("开始数据处理")
# 执行一些操作
log_memory_usage("数据处理完成")
```

---

## service/utils/silo_filtering.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的筒仓文件过滤工具，用于基于已存在的文件智能过滤新的筒仓数据请求，避免重复下载。

### 逐行代码解读

#### 1. 文件头与导入（第1-8行）
- 文档字符串说明文件用途：基于simple_retrieval目录中已存在的文件过滤筒仓
- 导入必需库：logging、pathlib、typing
- 配置logger用于日志输出

#### 2. 目录查找功能（第10-26行）
- `get_simple_retrieval_directory` 函数：在多个可能位置查找simple_retrieval目录
- 检查路径：data/simple_retrieval、service/data/simple_retrieval、../service/data/simple_retrieval、基于文件位置的相对路径
- 返回找到的第一个有效目录，找不到时返回None并记录警告

#### 3. 文件名解析功能（第28-71行）
- `extract_silo_info_from_filename` 函数：从parquet文件名提取筒仓ID
- 期望格式：granary_name_silo_id_start-date_to_end-date.parquet
- 解析逻辑：移除.parquet扩展名，按下划线分割，移除仓库名部分，查找日期模式（YYYY-MM-DD），提取筒仓部分
- 日期验证：检查年份（2000-2100）、月份（1-12）、日期（1-31）的有效性
- 异常处理：任何解析错误都会记录警告并返回None

#### 4. 已存在筒仓检查（第73-104行）
- `get_existing_silo_files` 函数：获取指定仓库的所有已存在筒仓ID集合
- 使用glob模式匹配：granary_name_*.parquet
- 备用匹配：空格替换为下划线的模式
- 对每个匹配文件调用extract_silo_info_from_filename提取筒仓ID
- 返回所有已存在筒仓ID的集合

#### 5. 新筒仓过滤（第106-127行）
- `filter_new_silos` 函数：过滤筒仓列表，仅保留没有已存在文件的筒仓
- 获取已存在筒仓集合，然后检查每个输入筒仓
- 匹配逻辑：精确匹配、部分匹配（任一方向包含）
- 分别收集新筒仓和跳过筒仓列表
- 记录过滤统计信息

#### 6. 高级过滤功能（第129-279行）
- `filter_silos_by_existing_files` 函数：基于文件名模式过滤筒仓数据字典列表
- 文件名解析逻辑（第140-200行）：
  - 查找"_to_"模式分割起止日期
  - 反向查找日期模式（YYYY-MM-DD格式）
  - 提取granary_silo标识符
  - 备用解析：无"_to_"时的日期模式查找
- 筒仓匹配逻辑（第234-270行）：
  - 生成可能的标识符组合：granary_name_silo_name、granary_name_silo_id
  - 处理空格替换为下划线的情况
  - 检查标识符是否存在于已有文件中
  - 分别收集过滤后和跳过的筒仓
- 详细日志记录匹配过程和结果统计

### 主要功能概述
- 智能查找simple_retrieval目录位置
- 从文件名解析筒仓和日期信息
- 检查已存在的筒仓文件
- 过滤新筒仓请求，避免重复下载
- 支持多种文件名格式和匹配策略
- 详细的日志记录和错误处理

### 典型用法
```python
from service.utils.silo_filtering import filter_silos_by_existing_files
filtered_silos, skipped_silos = filter_silos_by_existing_files(silo_data_list)
print(f"需要处理 {len(filtered_silos)} 个新筒仓，跳过 {len(skipped_silos)} 个已存在筒仓")
```

---

## app/Dashboard.py 文件详解

### 文件定位与作用
本文件是 SiloFlow 项目的 Streamlit 主界面，提供完整的数据科学工作流，包括数据上传、预处理、模型训练、评估、预测等功能，支持中英文界面。

### 主要结构与功能模块

#### 1. 导入与配置（第1-50行）
- 导入大量依赖：pathlib、pickle、datetime、numpy、pandas、plotly、streamlit、sklearn、lightgbm等
- 从granarypredict模块导入：数据清洗、特征工程、模型工具、数据摄取、多模型回归、参数缓存等
- 设置Streamlit页面配置：标题"SiloFlow"，宽布局模式

#### 2. 国际化支持（第51-570行）
- `_TRANSLATIONS_ZH` 字典：包含完整的中英文翻译映射
- 涵盖所有UI元素：侧边栏、训练选项、状态消息、性能优化、参数缓存、预测分析等
- `_t()` 函数：根据用户选择的语言返回对应翻译
- 支持复杂的技术术语翻译：Optuna超参数优化、分位数回归、锚定日提前停止、预测期平衡等

#### 3. 调试与工具函数（第571-620行）
- `_d()` 调试函数：在调试模式下显示toast消息并记录到session_state
- 环境变量和未来安全列定义：排除环境数据和生产标识符以提高模型泛化性
- 预设预测期：7、14、30、90、180、365天
- 全局预测期配置：HORIZON_DAYS=7，生成(1,2,3,4,5,6,7)元组

#### 4. 文件加载功能（第621-670行）
- `load_uploaded_file()` 函数：自动检测文件格式（CSV、Parquet、压缩格式）
- 支持多种压缩格式：.gz、.bz2、.zip、.xz
- 详细的错误处理和日志记录

#### 5. 模型加载功能（第671-720行）
- `load_trained_model()` 函数：从用户保存或预装目录加载模型
- 支持相对路径和绝对路径查找
- 全面的异常处理：RuntimeError、FileNotFoundError、ImportError等
- 详细的错误信息和修复建议

#### 6. 3D可视化功能（第721-780行）
- `plot_3d_grid()` 函数：绘制3D温度分布图
- 支持实际温度和预测温度显示
- 颜色映射：温度模式（红热蓝冷）、差值模式（红蓝对比）
- 交互式悬停信息：预测值、实际值、差值

#### 7. 时间序列可视化（第781-800行）
- `plot_time_series()` 函数开始：绘制实际vs预测温度时间序列图
- 支持多预测期显示

### 典型工作流程
1. **数据上传**：支持CSV/Parquet文件上传或选择预装样本
2. **数据预处理**：自动排序、特征工程、位置过滤
3. **模型训练**：支持多算法、超参数优化、参数缓存
4. **模型评估**：多指标评估、可视化分析
5. **预测生成**：未来温度预测、不确定性分析
6. **结果展示**：多标签页展示摘要、3D图、时间序列、极值分析等

### 性能优化特性
- **并行处理**：支持多核特征工程和超参数优化
- **参数缓存**：自动保存/加载最优参数，避免重复优化
- **保守系统**：热物理特征、稳定性增强、方向性增强
- **内存管理**：大数据集分块处理、垃圾回收
- **多后端支持**：Polars/Pandas适配、GPU加速选项

### 技术亮点
- **完整的数据科学工作流**：从数据到部署的全流程支持
- **企业级特性**：多语言支持、详细日志、异常处理
- **高性能优化**：并行计算、缓存机制、后端优化
- **专业可视化**：3D温度分布、时间序列、不确定性分析
- **用户友好**：直观界面、丰富提示、实时反馈

---


---

# granarypredict/cleaning_helpers.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/cleaning_helpers.py
- 该文件为 SiloFlow 项目的数据清洗辅助模块，主要用于填补传感器数据的日历缺口、插值补全缺失数值，确保模型训练和特征工程时时间序列连续。
- 常被 cleaning.py 及特征工程相关流程调用。

## 代码结构与逐行解读

### 1. 依赖导入
- `from __future__ import annotations`：兼容未来类型注解。
- `import numpy as np`、`import pandas as pd`：核心数据处理库。

### 2. insert_calendar_gaps(df)
- 作用：对每个传感器分组，自动补齐缺失的日历天数，生成“合成行”，使时间序列完整。
- 逻辑：
  - 检查是否有 detection_time 列，无则直接返回。
  - 转换 detection_time 为 pandas 的 datetime 类型。
  - 若全为空，直接返回。
  - 仅对 detection_time 非空的行分组处理。
  - 分组依据优先为 granary_id、heap_id、grid_x、grid_y、grid_z（如有）。
  - 对每组，找出最早和最晚日期，生成完整日历区间。
  - 找出缺失的日期，对每个缺失日，复制最后一行模板，仅保留静态字段，数值型字段（除静态字段外）全部置为 NaN。
  - 合并所有原始和新生成的行。
  - 若原始数据有 detection_time 为空的行，也一并拼接回去。

### 3. interpolate_sensor_numeric(df)
- 作用：对每个传感器分组，对所有数值型字段按时间线性插值，补全合成行的缺失值。
- 逻辑：
  - 检查 detection_time 列，无则直接返回。
  - 转换 detection_time 为 pandas 的 datetime 类型并排序。
  - 找出所有数值型字段。
  - 若有分组（granary_id、heap_id、grid_x、grid_y、grid_z），则对每组分别插值并前向填充。
  - 否则对全体插值并前向填充。
  - 返回插值后的 DataFrame。

### 4. __all__
- 明确导出 insert_calendar_gaps、interpolate_sensor_numeric 两个函数。

## 典型用法
- 在数据清洗流程中，先用 insert_calendar_gaps(df) 补全日历，再用 interpolate_sensor_numeric(df) 补全数值。
- 适用于模型训练、特征工程、异常检测等需要完整时间序列的场景。

## 健壮性与注意事项
- 所有操作均基于 pandas，支持大部分常见表结构。
- 分组字段缺失时自动降级为全体处理。
- 合成行的数值型字段除静态字段外均为 NaN，需后续插值。
- 插值仅对数值型字段有效，分类/字符串字段不会被填补。
- 若原始数据 detection_time 全为空或无该列，函数自动返回原始数据。

---
# service/scripts/client/siloflow_client_tester.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/client/siloflow_client_tester.py
- 该脚本为 SiloFlow 服务的综合客户端自动化测试工具，支持对服务端各主要 API（health、process、pipeline、train、models、forecast）进行全流程测试。
- 适合开发、部署、接口回归、自动化验收等场景。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 argparse、requests、pandas、json、logging、sys、time、pathlib、datetime。
- 日志支持文件与控制台双输出。

### 2. SiloFlowClientTester 主类
- test_connection：测试 /health 接口，检查服务健康与目录。
- test_process_endpoint：测试 /process 接口，上传 CSV 并校验处理结果。
- test_pipeline_endpoint：测试 /pipeline 接口，上传 CSV 并校验预测结果。
- test_train_endpoint：测试 /train 接口，触发模型训练。
- test_forecast_endpoint：测试 /forecast 接口，批量预测。
- test_models_endpoint：测试 /models 接口，列出现有模型。
- run_all_tests：串联所有测试，输出详细通过率与日志。
- _print_summary：输出测试汇总，自动保存详细结果为 JSON。

### 3. create_sample_csv 辅助函数
- 自动生成标准测试用 CSV 文件，便于无依赖测试。

### 4. main 函数与命令行接口
- 支持 --server、--port、--timeout、--file、--create-sample、--config 等参数。
- 支持从 JSON 配置文件加载参数。
- 自动串联所有测试，输出详细日志与结果。

## 典型用法
- 全流程自动化测试：
  ```shell
  python siloflow_client_tester.py --server 192.168.1.100 --port 8000 --file test.csv
  ```
- 仅生成测试用 CSV：
  ```shell
  python siloflow_client_tester.py --create-sample
  ```
- 使用配置文件批量测试：
  ```shell
  python siloflow_client_tester.py --config client_config.json
  ```

## 健壮性与注意事项
- 所有接口调用、文件操作均有详细异常处理。
- 支持自动生成测试数据，便于无依赖测试。
- 依赖 requests、pandas，需保证依赖完整。
- 适合开发、部署、接口回归、自动化验收等场景。
- 任何异常均有详细日志或终端提示，建议优先查看输出。

---
# service/scripts/client/run_client_tests.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/client/run_client_tests.py
- 该脚本为 SiloFlow 客户端测试入口，支持对服务端 API（连接、process、pipeline 等）进行自动化测试。
- 适合开发、部署、回归测试、接口验证等场景。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 argparse、sys、pathlib。
- 自动引入 siloflow_client_tester 及 create_sample_csv。

### 2. 各类测试函数
- run_basic_tests：测试服务端基础连通性。
- run_process_tests：测试 process 接口，支持自定义或自动生成 CSV。
- run_pipeline_tests：测试 pipeline 接口，支持自定义或自动生成 CSV。
- run_full_tests：全量测试所有接口，输出详细通过率。

### 3. main 函数与命令行接口
- 支持 --server、--port、--file、--test-type（basic/process/pipeline/full）等参数。
- 自动选择测试类型，输出详细测试结果。

## 典型用法
- 基础连通性测试：
  ```shell
  python run_client_tests.py --server 192.168.1.100 --test-type basic
  ```
- process 接口测试：
  ```shell
  python run_client_tests.py --server 192.168.1.100 --test-type process --file test.csv
  ```
- pipeline 接口测试：
  ```shell
  python run_client_tests.py --server 192.168.1.100 --test-type pipeline --file test.csv
  ```
- 全量测试：
  ```shell
  python run_client_tests.py --server 192.168.1.100 --test-type full
  ```

## 健壮性与注意事项
- 所有参数、接口调用均有详细异常处理。
- 支持自动生成测试 CSV，便于无依赖测试。
- 依赖 siloflow_client_tester.py，需保证同目录下存在。
- 适合开发、部署、接口回归等场景。
- 任何异常均有详细日志或终端提示，建议优先查看输出。

---
# service/scripts/database/list_granaries.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/database/list_granaries.py
- 该脚本用于查询数据库中所有可用粮库，输出粮库ID、名称、分表号、仓号数量等信息。
- 适合批量归档、数据核查、分析前的全局粮库列表获取。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、sqlalchemy、argparse、pathlib、utils.database_utils。
- 支持从 streaming_config.json 加载数据库配置。

### 2. list_granaries 主函数
- 查询所有粮库及其分表号、仓号数量，输出详细表格。
- 输出典型用法示例，便于后续批量拉取。

### 3. main 函数与命令行接口
- 自动校验配置文件。
- 直接命令行调用，输出所有结果到终端。

## 典型用法
- 查询所有粮库列表：
  ```shell
  python list_granaries.py --config streaming_config.json
  ```

## 健壮性与注意事项
- 所有数据库、文件、参数操作均有详细异常处理。
- 支持自动校验配置文件，防止无效查询。
- 依赖 pandas、sqlalchemy、pymysql，需保证依赖完整。
- 适合批量核查、归档、分析等场景。
- 任何异常均有详细日志或终端提示，建议优先查看输出。

---
# service/scripts/database/get_silos_for_granary.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/database/get_silos_for_granary.py
- 该脚本用于查询指定粮库下的所有仓号，输出仓号名称、ID、所属粮库等信息。
- 适合单库核查、归档、数据分析前的仓号列表获取。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、sqlalchemy、argparse、pathlib、utils.database_utils。
- 支持从 streaming_config.json 加载数据库配置。

### 2. get_silos_for_granary 主函数
- 查询指定粮库（支持名称或ID）下所有仓号，输出仓号名称、ID、所属粮库等。
- 结果按仓号名称排序，便于查阅。

### 3. main 函数与命令行接口
- 支持 --granary 粮库名称/ID。
- 自动校验参数与配置文件。
- 直接命令行调用，输出所有结果到终端。

## 典型用法
- 查询指定粮库下所有仓号：
  ```shell
  python get_silos_for_granary.py --granary "粮库名称" --config streaming_config.json
  ```

## 健壮性与注意事项
- 所有数据库、文件、参数操作均有详细异常处理。
- 支持自动校验参数与配置文件，防止无效查询。
- 依赖 pandas、sqlalchemy、pymysql，需保证依赖完整。
- 适合单库核查、归档、分析等场景。
- 任何异常均有详细日志或终端提示，建议优先查看输出。

---
# service/scripts/database/get_date_range_for_silo.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/database/get_date_range_for_silo.py
- 该脚本用于查询指定粮库/仓号的可用数据日期区间，输出最早/最晚数据时间及记录数。
- 适合单仓数据核查、归档前可用性确认、手动分析等场景。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、sqlalchemy、argparse、pathlib、utils.database_utils。
- 支持从 streaming_config.json 加载数据库配置。

### 2. get_date_range_for_silo 主函数
- 查询指定粮库/仓号的 sub_table_id、silo_id。
- 查询该仓号在对应分表下的最早/最晚 batch 时间及总记录数。
- 输出详细信息（粮库/仓号/分表/ID/区间/记录数等）。

### 3. main 函数与命令行接口
- 支持 --granary 粮库名称/ID，--silo 仓号名称。
- 自动校验参数与配置文件。
- 直接命令行调用，输出所有结果到终端。

## 典型用法
- 查询单仓号可用区间：
  ```shell
  python get_date_range_for_silo.py --granary "粮库名称" --silo "仓号名称" --config streaming_config.json
  ```

## 健壮性与注意事项
- 所有数据库、文件、参数操作均有详细异常处理。
- 支持自动校验参数与配置文件，防止无效查询。
- 依赖 pandas、sqlalchemy、pymysql，需保证依赖完整。
- 适合单仓核查、归档、分析等场景。
- 任何异常均有详细日志或终端提示，建议优先查看输出。

---
# service/scripts/database/get_date_ranges.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/database/get_date_ranges.py
- 该脚本用于批量查询数据库中所有粮库/仓号的可用数据日期区间，输出每个仓号的最早/最晚数据时间。
- 适合数据归档、批量拉取、数据核查前的全局数据可用性统计。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、sqlalchemy、argparse、pathlib、utils.database_utils。
- 支持从 streaming_config.json 加载数据库配置。

### 2. get_date_ranges 主函数
- 加载数据库配置，连接数据库。
- 查询所有粮库/仓号及 sub_table_id。
- 对每个仓号调用 get_silo_date_range，获取最早/最晚数据时间。
- 汇总输出每个仓号的可用区间、天数、全局统计（总粮库、总仓号、全局区间、均值/最大/最小跨度等）。
- 输出典型用法示例，便于后续批量拉取。

### 3. get_silo_date_range 辅助函数
- 查询单仓号在指定分表下的最早/最晚 batch 时间。

### 4. main 函数与命令行接口
- 支持 --config 配置参数，自动校验配置文件。
- 直接命令行调用，输出所有结果到终端。

## 典型用法
- 查询所有仓号可用区间：
  ```shell
  python get_date_ranges.py --config streaming_config.json
  ```

## 健壮性与注意事项
- 所有数据库、文件、参数操作均有详细异常处理。
- 支持自动校验配置文件，防止无效连接。
- 依赖 pandas、sqlalchemy、pymysql，需保证依赖完整。
- 适合批量归档、数据核查、全局可用性统计等场景。
- 任何异常均有详细日志或终端提示，建议优先查看输出。

---
# service/scripts/simple_data_retrieval.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/simple_data_retrieval.py
- 该脚本为 SiloFlow 项目的简易数据拉取工具，支持按粮库/仓号/日期区间精确拉取单仓数据并保存为 Parquet 文件。
- 适合快速归档、数据核查、手动分析等场景，兼容性强，SQL 查询完全可控。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、sqlalchemy、argparse、logging、json、os、pathlib、datetime、urllib.parse。
- 日志支持文件与控制台双输出。

### 2. SimpleDataRetriever 主类
- 初始化时加载数据库配置，自动创建 SQLAlchemy 引擎。
- get_all_granaries_and_silos：查询所有粮库/仓号及 sub_table_id。
- get_granaries_with_details：查询所有粮库及其仓号详细信息。
- get_silo_date_range：查询单仓可用数据的最早/最晚日期。
- get_silo_data：按粮库/仓号/日期区间拉取所有原始数据。
- retrieve_and_save：综合查找、校验、拉取、保存数据，自动生成 Parquet 文件，输出样例。

### 3. 配置加载
- load_config：支持从 config/production_config.json 加载数据库配置，若无则用默认配置。

### 4. main 函数与命令行接口
- 支持参数：
  - --granary-name 粮库名称
  - --silo-id 仓号（goods_allocation_id）
  - --start-date/--end-date 日期区间
  - --config 配置文件路径
  - --output-dir 输出目录
  - --list-granaries 列出所有粮库及仓号（含可用日期）
  - --list-silos 列出所有仓号
- 支持批量导出粮库/仓号/日期区间信息为 CSV。

## 典型用法
- 拉取单仓数据：
  ```shell
  python simple_data_retrieval.py --granary-name "蚬冈库" --silo-id "41f2257ce3d64083b1b5f8e59e80bc4d" --start-date "2024-07-17" --end-date "2025-07-18"
  ```
- 列出所有粮库及仓号（含可用日期）：
  ```shell
  python simple_data_retrieval.py --list-granaries
  ```
- 列出所有仓号：
  ```shell
  python simple_data_retrieval.py --list-silos
  ```

## 健壮性与注意事项
- 所有数据库、文件、参数操作均有详细日志与异常处理。
- 支持自动校验仓号、日期区间，防止无效拉取。
- 依赖 pandas、sqlalchemy、pymysql，需保证依赖完整。
- 适合手动归档、数据核查、分析等场景。
- 任何异常均有详细日志，建议优先查看日志定位。

---
# service/scripts/parquet_inspector.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/parquet_inspector.py
- 该脚本为 SiloFlow 项目的 Parquet 文件快速检查与转换工具，支持查看列名、数据样例、文件信息，并可将 Parquet 转为 CSV。
- 适用于数据归档、数据核查、数据分析前的快速预览与格式转换。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、argparse、sys、os、pathlib。

### 2. convert_parquet_to_csv 函数
- 将 Parquet 文件高效转换为 CSV，支持指定最大行数，自动创建输出目录。
- 输出转换进度、文件大小、压缩比等详细信息。

### 3. inspect_parquet_file 函数
- 检查 Parquet 文件，显示：
  - 文件行列数、内存占用、文件大小
  - 列名、数据类型、非空/空值统计、样例值
  - 前 5 行数据（自动格式化显示）
  - 数据类型分布、数据完整性、关键列唯一值
- 支持自动转置宽表，便于阅读。
- 可选自动转换为 CSV。

### 4. interactive_mode 交互模式
- 支持命令行交互式检查多个文件，按需转换为 CSV。

### 5. main 函数与命令行接口
- 支持参数：
  - file_path：待检查的 Parquet 文件路径
  - -i/--interactive：交互模式
  - -c/--convert-csv：检查后自动转为 CSV
  - --csv-max-rows：CSV 最大行数
- 支持直接命令行调用或交互式批量检查。

## 典型用法
- 交互模式：
  ```shell
  python parquet_inspector.py
  ```
- 检查指定文件：
  ```shell
  python parquet_inspector.py data/processed/xxx.parquet
  ```
- 检查并转为 CSV（前 1000 行）：
  ```shell
  python parquet_inspector.py data/processed/xxx.parquet -c --csv-max-rows 1000
  ```

## 健壮性与注意事项
- 所有文件操作均有详细异常处理，支持不存在、格式错误等情况提示。
- 支持大文件分批转换，防止内存溢出。
- 依赖 pandas，需保证依赖完整。
- 适合数据归档、核查、分析前的快速预览与格式转换。
- 任何异常均有详细提示，建议优先查看终端输出。

---
# service/scripts/data_retrieval/automated_data_retrieval.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/data_retrieval/automated_data_retrieval.py
- 该脚本为 SiloFlow 项目的自动化数据拉取与预处理主入口，自动完成从 MySQL 拉取所有 granary/silo 数据、预处理、归档等全流程。
- 支持全量、增量、指定日期区间等多种拉取模式，适合批量归档、数据同步、数据分析等场景。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 argparse、logging、sys、datetime、pathlib、pandas 等。
- 自动引入 service/utils/database_utils、data_paths、sql_data_streamer 等核心工具。
- 全局日志配置，支持文件与控制台双输出。

### 2. AutomatedDataRetrieval 主类
- 初始化时加载配置、数据目录、SQLDataStreamer 实例。
- get_last_processed_date：自动检测已处理数据的最新日期。
- check_data_availability：检查数据库与本地已存在数据，智能判断缺失区间。
- full_retrieval：全量拉取所有数据，仅补齐缺失区间，避免重复。
- incremental_retrieval：增量拉取最近 N 天数据，仅补齐缺失区间。
- date_range_retrieval：拉取指定日期区间数据。
- _get_earliest_available_date：查询数据库最早可用数据日期。
- generate_summary_report：生成详细的拉取与处理报告。
- cleanup_old_files：自动清理过期日志与临时文件。

### 3. main 函数与命令行接口
- 支持 --full-retrieval、--incremental、--date-range、--granary、--start、--end、--cleanup 等参数。
- 自动校验参数，支持多种拉取模式。
- 拉取完成后自动生成并保存详细报告。
- 可选自动清理过期文件。

## 典型用法
- 全量拉取：
  ```shell
  python automated_data_retrieval.py --full-retrieval
  ```
- 增量拉取最近 7 天：
  ```shell
  python automated_data_retrieval.py --incremental --days 7
  ```
- 拉取指定日期区间：
  ```shell
  python automated_data_retrieval.py --date-range --start 2024-01-01 --end 2024-12-31
  ```
- 拉取指定 granary：
  ```shell
  python automated_data_retrieval.py --full-retrieval --granary "粮库名称或ID"
  ```
- 拉取后自动清理 15 天前文件：
  ```shell
  python automated_data_retrieval.py --full-retrieval --cleanup --days-to-keep 15
  ```

## 健壮性与注意事项
- 所有参数、路径、数据库连接均有详细日志与异常处理。
- 支持自动检测本地已存在数据，避免重复拉取。
- 依赖 sql_data_streamer、database_utils、data_paths、pandas、tqdm、psutil 等，需保证依赖完整。
- 适合生产环境批量归档、数据同步、数据分析等。
- 任何异常均有详细日志，建议优先查看日志定位。

---
# service/scripts/data_retrieval/sql_data_streamer.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/data_retrieval/sql_data_streamer.py
- 该文件为 SiloFlow 项目的 SQL 数据流式拉取核心脚本，负责从 MySQL 数据库高效批量拉取所有 granary/silo 的原始数据，支持分块、流式、内存安全等多种模式。
- 适用于生产环境大规模数据归档、数据同步、批量预处理等场景。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 argparse、logging、sys、os、datetime、pathlib、pandas、numpy、sqlalchemy、pymysql、tqdm、psutil、tempfile、gc、atexit、signal、functools 等。
- 自动引入 service/utils/database_utils、data_paths 等核心工具。
- 全局日志配置，支持文件与控制台双输出。

### 2. SQLDataStreamer 主类
- 初始化时加载配置、数据库连接、数据目录、内存与批量参数。
- get_all_granaries_and_silos：查询所有 granary/silo 结构，支持名称/ID 过滤。
- stream_granary_data/collect_data_in_batches：支持分块、流式、内存安全的数据拉取，自动适配小批量/大批量场景。
- get_silo_data_fast/get_silo_data_chunked：针对小量/大批量数据分别优化，防止内存溢出。
- estimate_data_size：拉取前自动估算数据量与内存需求，智能推荐批量参数。
- get_silo_date_range：查询单仓数据可用时间范围。
- check_memory_usage/force_memory_cleanup/wait_for_memory_recovery：多策略内存保护，自动回收、等待、降级，极大提升健壮性。
- _stream_large_dataset：超大数据集自动分批流式处理，边拉取边写入，防止 OOM。
- _cleanup/_signal_handler：自动资源清理，支持中断安全。

### 3. 命令行与用法
- 支持 --start-date/--end-date、--config、--no-pipeline、--create-config 等参数。
- 可独立运行，也可被其他批处理脚本调用。

## 典型用法
- 拉取指定日期区间数据：
  ```shell
  python sql_data_streamer.py --start-date 2024-01-01 --end-date 2024-12-31
  ```
- 使用自定义配置：
  ```shell
  python sql_data_streamer.py --config streaming_config.json
  ```
- 仅拉取不处理：
  ```shell
  python sql_data_streamer.py --no-pipeline
  ```

## 健壮性与注意事项
- 所有数据库连接、内存、批量参数均有详细日志与异常处理。
- 支持自动内存保护，极端大数据自动降级为流式分批处理。
- 依赖 tqdm、psutil、sqlalchemy、pymysql 等，需保证依赖完整。
- 适合生产环境大规模数据归档、同步、批量预处理等。
- 任何异常均有详细日志，建议优先查看日志定位。

---
# service/scripts/data_retrieval/automated_data_retrieval.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/data_retrieval/automated_data_retrieval.py
- 该文件为 SiloFlow 项目的自动化数据拉取主脚本，支持全量、增量、日期范围等多模式从 MySQL 数据库批量拉取所有 granary/silo 数据，并自动预处理、组织存储。
- 适用于生产环境定时任务、批量数据归档、数据同步等场景。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 argparse、logging、sys、datetime、pathlib、pandas、typing。
- 自动引入 service/utils/database_utils、data_paths、sql_data_streamer 等核心工具。
- 全局日志配置，支持文件与控制台双输出。

### 2. AutomatedDataRetrieval 主类
- 初始化时自动加载配置、数据目录、SQL 拉取器。
- get_last_processed_date：检测已处理数据的最新日期。
- check_data_availability：检查数据库与本地已有数据，智能判断缺失区间。
- full_retrieval/incremental_retrieval/date_range_retrieval：支持全量、增量、指定日期区间三种拉取模式，自动跳过已有数据，仅补齐缺失。
- generate_summary_report：生成详细拉取与处理报告。
- cleanup_old_files：自动清理过期日志与临时文件。

### 3. 主命令行入口 main()
- 支持 --full-retrieval、--incremental、--date-range、--granary、--start/--end、--cleanup 等参数。
- 自动校验参数，按需调用对应拉取模式。
- 拉取完成后自动生成报告并保存。

## 典型用法
- 全量拉取：
  ```shell
  python automated_data_retrieval.py --full-retrieval
  ```
- 增量拉取近 7 天：
  ```shell
  python automated_data_retrieval.py --incremental --days 7
  ```
- 指定日期区间拉取：
  ```shell
  python automated_data_retrieval.py --date-range --start 2024-01-01 --end 2024-12-31
  ```
- 清理 30 天前旧文件：
  ```shell
  python automated_data_retrieval.py --cleanup --days-to-keep 30
  ```

## 健壮性与注意事项
- 所有参数、目录、数据库连接均有详细日志与异常处理。
- 支持 granary 名称/ID 过滤，适合大规模分批拉取。
- 自动跳过已有数据，仅补齐缺失区间，极大节省带宽与存储。
- 依赖 sql_data_streamer、data_paths、database_utils 等工具，需保证依赖完整。
- 适合定时任务、批量归档、数据同步等生产场景。
- 任何异常均有详细日志，建议优先查看日志定位。

---
# service/scripts/testing/testingservice.py 详细交接说明

## 文件定位与作用
- 路径：service/scripts/testing/testingservice.py
- 该文件为 SiloFlow 项目的 GUI 测试与数据管理工具，基于 tkinter 实现，集成 HTTP 服务测试、数据拉取、数据库结构浏览、批量操作、系统监控等多功能于一体。
- 适用于开发、测试、数据运维等场景，极大提升接口测试与数据管理效率。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 tkinter、requests、json、os、threading、asyncio、subprocess、datetime、pandas、re、pathlib 等。
- 支持自动检测虚拟环境 python 路径，兼容本地与系统 python。
- 配置多标签页、现代配色与样式，支持高分辨率自适应。

### 2. 主类 SiloFlowTester
- 初始化窗口、主题、配色、全局变量、服务地址等。
- create_widgets：创建主界面与所有功能标签页。
- setup_modern_style/configure_custom_styles：统一现代化 ttk 风格。
- center_window：窗口居中。

### 3. 主要功能标签页
- HTTP Service Testing：支持文件上传、接口选择（/pipeline、/process、/train、/forecast、/models、/health）、响应展示、状态栏。
- Remote Client Testing：支持远程服务地址配置、文件上传、批量测试、报告生成。
- Simple Retrieval：单仓数据拉取，支持 granary/silo/date 范围选择、批量自动处理、进度与日志展示。
- Production Pipeline：生产级数据流批处理，支持配置文件选择、阶段勾选、性能参数、系统监控、批量运行。
- Database Explorer：数据库结构浏览，支持 granary/silo/date 三级联动、连接测试、结构导出。
- Logs & Monitoring：系统日志与监控，实时输出。

### 4. 关键交互与并发设计
- 多线程/异步操作，所有耗时任务均后台运行，界面不卡顿。
- 实时进度、状态、日志反馈，便于调试与监控。
- 支持自动检测服务健康、数据库连接、系统资源。

### 5. 典型用法
- 启动 GUI：
  ```shell
  python service/scripts/testing/testingservice.py
  ```
- 测试 HTTP 服务接口、批量上传数据、拉取单仓/多仓数据、浏览数据库结构、监控系统资源。

## 健壮性与注意事项
- 所有操作均有异常捕获与弹窗提示，防止界面崩溃。
- 支持本地与远程服务切换，适合多环境测试。
- 多线程/异步任务自动管理，防止阻塞。
- 配置文件、输出目录、服务地址等均可自定义。
- 依赖 tkinter，需确保 Python 安装 GUI 支持。
- 适合开发、测试、数据运维等多场景，生产环境建议仅用于测试与管理。

---
# service/routes/health.py 详细交接说明

## 文件定位与作用
- 路径：service/routes/health.py
- 该文件为 SiloFlow 项目的 API 路由模块，负责“健康检查”相关的 HTTP 接口，主要用于服务存活性和关键目录状态检测。
- 通过 FastAPI APIRouter 注册，供主服务统一挂载。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 fastapi、pandas、logging、pathlib。
- 配置全局 logger，INFO 级别。

### 2. /health 路由（GET）
- @router.get("/health")
- 功能：健康检查，返回服务状态和关键目录存在性。
- 步骤：
  1. 检查 models、data/processed、data/granaries 三个核心目录是否存在
  2. 返回 status、service 名称、时间戳、各目录存在性字典
- 典型用法：GET /health

### 3. 健壮性与异常处理
- 检查异常时自动记录日志，返回 503 错误。

## 典型用法
- 检查服务健康：
  ```shell
  curl http://localhost:8000/health
  ```
- 返回内容：JSON，包含服务状态、时间戳、各目录存在性。

## 健壮性与注意事项
- 仅做基础健康检查，适合负载均衡、监控系统探活。
- 目录不存在时不会中断服务，仅反映在返回内容。
- 任何异常均有详细日志，便于定位。
- 可扩展更多健康项（如数据库、GPU、依赖等）。

---
# service/routes/train.py 详细交接说明

## 文件定位与作用
- 路径：service/routes/train.py
- 该文件为 SiloFlow 项目的 API 路由模块，负责“训练”相关的 HTTP 接口，主要用于批量训练所有已处理 granary 的模型。
- 通过 FastAPI APIRouter 注册，供主服务统一挂载。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 fastapi、pandas、gc、logging。
- 引入全局 processor 单例，负责实际训练逻辑。
- 配置全局 logger，INFO 级别。

### 2. /train 路由（POST）
- @router.post("/train")
- 功能：为所有已处理 granary（无模型）批量训练模型。
- 步骤：
  1. 遍历 data/processed/ 下所有 *_processed.parquet 文件
  2. 检查每个 granary 是否已有模型（.joblib 或 .joblib.gz）
  3. 无模型则调用 processor.process_granary(granary_name) 训练
  4. 训练成功加入 trained，失败加入 errors，已存在模型加入 skipped
  5. 每轮训练后自动 gc.collect()，如有 torch 则清理 GPU 显存
- 返回所有训练、跳过、失败 granary 列表及时间戳

### 3. 健壮性与异常处理
- 无数据时自动报错（400），需先跑 /pipeline 或 /process。
- 训练异常、内部错误均详细日志，统一 HTTPException。
- 每轮训练后自动释放内存和 GPU 显存，防止资源泄漏。

## 典型用法
- 批量训练所有 granary：
  ```shell
  curl -X POST http://localhost:8000/train
  ```
- 返回内容：JSON，包含训练成功、跳过、失败 granary 列表、时间戳等。

## 健壮性与注意事项
- 需先通过 /pipeline 或 /process 生成 processed 数据，否则无法训练。
- granary 已有模型自动跳过，避免重复训练。
- 训练异常详细记录，便于定位。
- 每轮训练后自动释放内存和 GPU 显存，适合大批量训练。
- 业务逻辑全部委托给 processor 单例，API 层仅做参数校验、调度与响应。

---
# service/routes/models.py 详细交接说明

## 文件定位与作用
- 路径：service/routes/models.py
- 该文件为 SiloFlow 项目的 API 路由模块，负责“模型管理”相关的 HTTP 接口，包括模型列表分页查询、模型删除等。
- 通过 FastAPI APIRouter 注册，供主服务统一挂载。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 fastapi、pandas、logging。
- 引入全局 processor 单例，负责模型目录定位。
- 配置全局 logger，INFO 级别。

### 2. /models 路由（GET）
- @router.get("/models")
- 功能：分页列出所有可用 granary 预测模型。
- 步骤：
  1. 通过 data_paths.get_models_dir() 获取模型目录
  2. 查找所有 *_forecast_model.joblib* 文件
  3. 支持 page/per_page 分页参数，返回当前页模型列表、总数、总页数等
  4. 每个模型返回 granary 名、路径、体积、最后修改时间
- 典型用法：GET /models?page=1&per_page=10

### 3. /models/{granary_name} 路由（DELETE）
- @router.delete("/models/{granary_name}")
- 功能：删除指定 granary 的模型（支持压缩和未压缩两种文件）。
- 步骤：
  1. 检查 {granary_name}_forecast_model.joblib 或 .joblib.gz 是否存在
  2. 存在则删除，返回成功消息
  3. 不存在则 404 报错
- 典型用法：DELETE /models/ABC123

### 4. 健壮性与异常处理
- 所有异常均详细日志，HTTPException 统一抛出。
- 分页参数越界自动返回空列表。
- 删除不存在模型自动 404。

## 典型用法
- 查询模型列表：
  ```shell
  curl http://localhost:8000/models?page=1&per_page=10
  ```
- 删除模型：
  ```shell
  curl -X DELETE http://localhost:8000/models/ABC123
  ```
- 返回内容：JSON，包含模型列表、分页信息、删除状态等。

## 健壮性与注意事项
- 模型文件支持压缩和未压缩两种格式，删除时自动兼容。
- 分页参数建议合理设置，防止一次性拉取过多文件。
- 任何异常均有详细日志，便于定位。
- 业务逻辑全部委托给 processor 单例和 data_paths 工具，API 层仅做参数校验、调度与响应。

---
# service/routes/forecast.py 详细交接说明

## 文件定位与作用
- 路径：service/routes/forecast.py
- 该文件为 SiloFlow 项目的 API 路由模块，负责“预测”相关的 HTTP 接口，主要用于对所有已处理 granary 进行单日预测。
- 通过 FastAPI APIRouter 注册，供主服务统一挂载。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 fastapi、pandas、logging。
- 引入全局 processor 单例，负责实际预测逻辑。
- 配置全局 logger，INFO 级别。

### 2. /forecast 路由（GET）
- @router.get("/forecast")
- 功能：对所有已处理 granary（有模型）进行单日预测。
- 步骤：
  1. 遍历 data/processed/ 下所有 *_processed.parquet 文件
  2. 检查每个 granary 是否有对应模型（.joblib 或 .joblib.gz）
  3. 有模型则调用 processor.generate_forecasts(granary_name, horizon=1)
  4. 汇总所有 granary 的预测结果，返回 JSON
  5. 无模型或无数据自动跳过，记录 skipped 列表
- 预测 horizon 固定为 1 天（可扩展）。

### 3. 健壮性与异常处理
- 无数据时自动报错（400），需先跑 /pipeline。
- 预测异常、内部错误均详细日志，统一 HTTPException。

## 典型用法
- 批量预测所有 granary：
  ```shell
  curl http://localhost:8000/forecast
  ```
- 返回内容：JSON，包含每个 granary 的预测结果、跳过的 granary 列表、时间戳等。

## 健壮性与注意事项
- 需先通过 /pipeline 生成 processed 数据和模型，否则无法预测。
- 仅预测 horizon=1（单日），如需多天预测建议扩展接口。
- granary 无模型或无数据自动跳过，结果中有 skipped 列表。
- 任何异常均有详细日志，便于定位。
- 业务逻辑全部委托给 processor 单例，API 层仅做参数校验、调度与响应。

---
# service/routes/pipeline.py 详细交接说明

## 文件定位与作用
- 路径：service/routes/pipeline.py
- 该文件为 SiloFlow 项目的 API 路由模块，负责“管道操作”相关的 HTTP 接口，包括数据批量处理、全流程自动化（ingest→preprocess→train→forecast）等。
- 通过 FastAPI APIRouter 注册，供主服务统一挂载。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 fastapi、pandas、asyncio、logging、json、base64、pathlib。
- 引入全局 processor 单例，负责实际数据处理。
- 配置全局 logger，INFO 级别。
- TEMP_UPLOADS_DIR：临时上传目录，自动创建，所有上传文件先落盘。

### 2. /process 路由（POST）
- @router.post("/process")
- 功能：仅做数据 ingest+preprocess，不训练、不预测。
- 步骤：
  1. 校验上传文件类型（CSV/Parquet）
  2. 文件保存到临时目录
  3. 调用 processor.process_raw_csv 进行分仓、清洗、特征工程
  4. 处理结果写入 data/processed/
  5. 返回每个 granary 的处理状态、文件路径、体积等
- 超时自动中断（1 小时），异常详细日志。

### 3. /pipeline 路由（POST）
- @router.post("/pipeline")
- 功能：一站式全流程（ingest→preprocess→train→forecast），适合批量自动化。
- 步骤：
  1. 校验上传文件类型（CSV/Parquet）
  2. 文件保存到临时目录
  3. 调用 processor.process_all_granaries，自动分仓、清洗、训练、预测
  4. 汇总所有 granary 的预测结果，拼接为 CSV
  5. 响应头 X-Forecast-Summary 返回 base64 编码的处理摘要
  6. 支持 horizon 参数（预测天数），默认 7 天
- 超时自动中断（3 小时），异常详细日志。

### 4. 健壮性与异常处理
- 所有异常均详细日志，HTTPException 统一抛出。
- 上传文件类型、处理超时、内部错误均有专用错误码和提示。
- 处理完毕自动清理临时文件，防止磁盘堆积。

## 典型用法
- 仅数据处理：
  ```shell
  curl -F "file=@raw.csv" http://localhost:8000/process
  ```
- 全流程自动化：
  ```shell
  curl -F "file=@raw.csv" http://localhost:8000/pipeline?horizon=7
  ```
- 返回内容：/process 返回 JSON 处理状态，/pipeline 返回预测 CSV，摘要在响应头 X-Forecast-Summary。

## 健壮性与注意事项
- 上传文件需为 CSV 或 Parquet，类型错误自动拒绝。
- 超大数据集建议分批上传，防止超时。
- 处理超时自动中断，防止阻塞。
- 临时文件自动清理，防止磁盘堆积。
- 任何异常均有详细日志，便于定位。
- 业务逻辑全部委托给 processor 单例，API 层仅做参数校验、调度与响应。

---
# service/main.py 详细交接说明

## 文件定位与作用
- 路径：service/main.py
- 该文件为 SiloFlow 项目的 FastAPI 主入口，负责启动 Web 服务、注册所有 API 路由、统一日志配置。
- 采用模块化路由设计，所有具体 API 实现在 service/routes 目录下，主文件仅做统一调度与启动。

## 结构与主要内容

### 1. 顶部说明与依赖
- 文件头部 docstring 说明本文件为精简版 FastAPI 启动器，所有路由均委托给 routes 目录下模块。
- 依赖 fastapi、uvicorn、logging、fastapi.middleware.cors。

### 2. 日志配置
- 启动时全局 logging.basicConfig，INFO 级别，标准格式，便于生产环境追踪。
- logger = logging.getLogger(__name__)，全局日志对象。

### 3. FastAPI 应用初始化
- app = FastAPI(...)，设置服务标题、描述、版本。
- CORS 策略：开发阶段允许所有来源，生产环境建议收紧。
- app.add_middleware(CORSMiddleware, ...)，支持跨域请求。

### 4. 路由注册
- from .routes import router as all_routes
- app.include_router(all_routes)
- 所有 API 路由均在 service/routes 目录下拆分实现，主文件只需一行注册，极大提升可维护性。

### 5. 启动入口
- 支持命令行启动：python -m uvicorn service.main:app --reload
- if __name__ == "__main__": uvicorn.run(...)
- 默认监听 0.0.0.0:8000，log_level=info。

## 典型用法
- 启动开发服务器：
  ```shell
  python -m uvicorn service.main:app --reload
  ```
- 生产部署建议用 gunicorn/uvicorn 多进程模式。

## 健壮性与注意事项
- 路由全部模块化，主文件极简，便于维护和扩展。
- 日志配置全局生效，便于排查线上问题。
- CORS 策略开发阶段全开放，生产环境需限制 allow_origins。
- 仅做服务启动与路由注册，业务逻辑全部下沉到 routes 目录。
- 任何异常建议通过 FastAPI 全局异常处理器统一捕获。

---

# SiloFlow 项目交接与使用说明（HOWTOUSE）

## 重要文件与模块清单
请严格按照下列顺序逐一阅读、总结并补充交接文档，每次“next”进入下一个文件：

### 1. 核心管道与入口
- service/granary_pipeline.py （主数据管道与 CLI 入口）
- service/automated_processor.py （自动化批处理主引擎）

### 2. 数据处理与特征工程
- granarypredict/ingestion.py （数据读取、标准化、保存等）
- granarypredict/cleaning.py （数据清洗、缺失值处理等）
- granarypredict/features.py （特征工程核心逻辑）
- granarypredict/polars_data_utils.py （Polars 优化数据处理）
- granarypredict/gpu_data_utils.py （GPU/cuDF 优化数据处理）

### 3. 训练与评估
- granarypredict/model.py （模型训练与推理）
- granarypredict/multi_lgbm.py （多目标 LGBM 训练器）
- granarypredict/evaluate.py （评估与指标）

### 4. 配置与工具
- granarypredict/config.py （全局配置）
- service/utils/data_paths.py （数据路径管理）
- service/utils/path_utils.py （路径工具）
- service/utils/db_utils.py （数据库工具）

### 5. API 与服务
- service/main.py （FastAPI 主入口）
- service/routes/process.py （API 路由：数据处理）
- service/routes/pipeline.py （API 路由：管道操作）
- service/routes/forecast.py （API 路由：预测）
- service/routes/models.py （API 路由：模型管理）
- service/routes/train.py （API 路由：训练）
- service/routes/health.py （API 路由：健康检查）

### 6. 脚本与批处理
- service/scripts/data_retrieval/sql_streaming.py （SQL 流式数据拉取）
- service/scripts/data_retrieval/single_silo_retrieval.py （单仓数据拉取）
- service/scripts/data_retrieval/batch_retrieval.py （批量数据拉取）
- service/scripts/testing/testingservice.py （GUI 测试工具）

### 7. 依赖与环境
- requirements.txt
- requirements-gpu.txt
- pyproject.toml

### 8. 说明文档
- README.MD

---
每次“next”将自动进入下一个文件并补充详细中文交接说明。
READ THE ENTIRE FILE EVERY TIME NO EXCEPTIONS

# service/granary_pipeline.py 详细交接说明

## 文件定位与作用
- 路径：service/granary_pipeline.py
- 该文件是 SiloFlow 项目的主数据管道与命令行入口，负责数据的全流程处理、特征工程、模型训练、预测、批量自动化等。
- 支持 ingest（原始数据导入）、preprocess（预处理）、train（训练）、forecast（预测）、pipeline（全流程）等命令。
- 具备高性能大数据处理能力，自动优先使用 Polars、GPU、cuDF 等加速后端。
- 代码高度模块化，便于自动化、批处理、云端部署。

## 结构与主要内容

### 1. 顶部说明与用法
- 文件头部有详细 docstring，说明了 CLI 用法、各阶段功能、训练配置、示例命令。
- 支持 Optuna 超参数调优、GPU 自动检测、分阶段训练、压缩存储等。

### 2. 依赖与全局变量
- 导入标准库（argparse、os、pathlib、typing、logging、sys、gc、tempfile、shutil、numpy、pandas 等）。
- 动态检测并导入 polars、pynvml、pyopencl、lightgbm、optuna、cudf 等高性能依赖。
- 通过 try/except 自动判断 Polars、cuDF、GPU 是否可用，设置 HAS_POLARS、HAS_GPU_DATA_UTILS 等全局变量。
- 配置全局 logger，所有关键步骤均有详细日志输出。

### 3. Polars 优化与数据加载
- load_data_optimized：根据文件类型和大小自动选择 Polars（如可用）或 pandas 加载数据。
- Parquet 文件优先用 Polars 读取，极大提升大文件处理速度。
- 加载后统一转为 pandas DataFrame，保证后续兼容性。
- 失败时自动 fallback 到 pandas 并详细记录异常。
- 相关函数如 add_lags_optimized、add_rolling_stats_optimized、create_time_features_optimized，均优先用 Polars 实现大规模特征工程，极大提升速度和内存效率。

### 4. GPU 检测与配置
- detect_gpu_availability：支持多 GPU 检测与选择，优先用 pynvml 检测 NVIDIA GPU，fallback 到 OpenCL。
- 支持强制指定 GPU ID，或自动选择最优 GPU（按空闲内存、利用率等多因子打分）。
- 自动生成 LightGBM GPU 配置参数（device、gpu_device_id、max_bin 等），并用随机数据实际训练验证 GPU 可用性。
- get_gpu_config_for_training：根据数据量、特征数、GPU 内存等动态调整训练参数，防止 OOM。
- 全局变量 GPU_AVAILABLE、GPU_INFO、GPU_CONFIG 记录 GPU 检测结果。
- 日志详细输出所有 GPU 检测、选择、配置、内存等信息。

### 5. 数据处理与特征工程
- 支持三种数据处理模式：
  - 内存模式（小数据集，直接处理）
  - 分块模式（大数据集，分块处理，防止内存溢出）
  - 流式模式（超大数据集，磁盘中转，极限节省内存）
- _preprocess_silos 自动根据数据量选择最优模式。
- _apply_basic_preprocessing 依次完成：
  - 列标准化（standardize_granary_csv）
  - 基础清洗（cleaning.basic_clean）
  - 冗余列删除
  - 日历缺口插补、数值插值、缺失填充
  - 时间/空间/方向/稳定性/多步目标等特征工程（大数据优先用 Polars/GPU 优化版本）
  - 分组与排序（优先用 GPU/Polars 加速）
  - 列顺序整理
- 所有步骤均有详细日志，异常自动 fallback 并记录。

### 6. 训练与模型保存
- run_complete_pipeline 支持全流程自动化：加载、预处理、训练、模型保存。
- 训练阶段：
  - 先 95/5 内部分割做参数优化（early stopping、anchor-day、horizon balancing 等）
  - 再全量数据定参数训练最终模型
  - 支持 Optuna 自动调参，也可用固定参数
  - GPU 配置自动合并到 LightGBM 参数
  - 支持模型压缩存储（save_compressed_model），节省空间
  - 训练失败/数据异常均有详细错误记录。

### 7. 命令行接口（main 函数）
- argparse 支持 ingest、preprocess、train、forecast、pipeline 等命令。
- 每个命令均有详细参数说明，支持 GPU/CPU、调参/固定参数、并行/串行等灵活配置。
- 典型用法：
  ```shell
  python granary_pipeline.py ingest --input raw.csv
  python granary_pipeline.py preprocess --input granary.csv --output processed.csv
  python granary_pipeline.py train --granary ABC123 --tune --trials 100 --timeout 600
  python granary_pipeline.py forecast --granary ABC123 --horizon 7
  python granary_pipeline.py pipeline --input raw.csv --granary ABC123
  ```

### 8. 依赖与模块关系
- 依赖 granarypredict 目录下 ingestion、cleaning、features、multi_lgbm、compression_utils、polars_data_utils、gpu_data_utils、streaming_processor 等模块。
- 依赖 service/utils/data_paths.py 管理数据路径。
- 依赖 app/Dashboard.py 提供部分清洗与分割辅助函数。
- 依赖 外部库：polars、pandas、numpy、lightgbm、optuna、pynvml、pyopencl、cudf、pyarrow、joblib、shutil、logging 等。

### 9. 性能与健壮性设计
- 所有大数据操作均优先用 Polars/GPU/cuDF，极大提升速度与内存效率。
- 自动 fallback 机制，任何依赖缺失或异常均可降级到 pandas/CPU，保证健壮性。
- 日志极为详细，便于定位问题。
- 支持多 GPU、分块/流式处理、模型压缩、自动调参等企业级特性。

### 10. 常见问题与注意事项
- Windows 下 cuDF 不支持，仅限 Linux/WSL2。
- 推荐 Python 3.10+，需安装 polars、pyarrow、lightgbm、optuna、pynvml 等依赖。
- 大数据集建议使用 Parquet 格式，提升读写效率。
- GPU 训练需安装对应驱动和 CUDA。
- 任何步骤异常均有详细日志，建议优先查看日志定位。
- 目录结构、数据路径、依赖版本需与项目保持一致。

---

# service/automated_processor.py 详细交接说明

## 文件定位与作用
- 路径：service/automated_processor.py
- 该文件是 SiloFlow 项目的自动化批处理主引擎，负责批量数据导入、分仓处理、模型训练、预测、内存与资源管理等。
- 支持全自动化的“批量 ingest → preprocess → train → forecast”流水线，适合大规模数据和多仓库场景。
- 具备极强的健壮性、内存保护、异常恢复、并发处理能力。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 导入标准库（asyncio、atexit、gc、logging、os、signal、sys、tempfile、time、pathlib、typing、pandas 等）。
- 动态检测 psutil（内存监控），无则降级。
- 自动将 granarypredict 目录加入 sys.path，保证跨模块导入。
- 配置全局 logger，所有操作均有详细日志。

### 2. AutomatedGranaryProcessor 类
- 构造函数 __init__：
  - 统一初始化所有数据目录（模型、处理后、原始、预测、临时、上传、流式、批量等），全部通过 data_paths 工具集中管理。
  - 设置内存阈值、重试次数、延迟、缓存、资源追踪等参数。
  - 注册退出与信号清理钩子，保证异常退出时资源自动释放。
  - 日志详细输出所有目录结构。

### 3. 内存与资源管理
- _check_memory_usage：用 psutil 检查内存占用，超阈值自动报警。
- _force_memory_cleanup：多策略强制释放内存（gc、缓存清理、pandas cache、延迟等待等），并详细记录释放前后内存。
- _safe_file_operation：所有文件操作均加内存检查、重试、自动恢复，极大提升健壮性。
- _cleanup/_signal_handler/cleanup_temp_files：退出、信号、临时文件清理，防止资源泄漏。

### 4. 批量数据处理主流程
- process_raw_csv：
  - 调用 granarypredict.ingestion.ingest_and_sort，支持新老返回格式，自动过滤已存在的 silo 文件。
  - 只处理真正有新数据的 granary/silo，极大节省计算与存储。
  - 日志详细输出每个 granary/silo 的处理与跳过情况。
- process_granary：
  - 针对单个 granary，支持只对 changed_silos 做预处理，训练阶段仍用全量数据。
  - 自动检测模型是否已存在，已存在则跳过训练。
  - 调用 run_complete_pipeline 完成全流程。
  - 处理异常时详细记录 traceback。
- generate_forecasts：
  - 加载模型（支持自适应压缩与 joblib fallback），加载处理后数据（优先 Parquet），内存保护。
  - 目前为占位实现，实际预测逻辑可扩展。
- process_all_granaries：
  - 完整自动化批处理主入口。
  - 先批量 ingest，过滤已存在 silo。
  - 逐个 granary 并发处理（预处理、训练、预测），每步均有内存保护与异常恢复。
  - 结果结构化返回，便于上层 API 或脚本调用。

### 5. 健壮性与性能优化
- 所有关键步骤均有详细日志，便于追踪与调试。
- 内存保护机制极为完善，任何高占用场景均可自动恢复。
- 文件操作、模型加载、数据处理均有多层重试与降级。
- 支持并发/异步处理，适合大规模批量场景。
- 资源清理彻底，防止内存泄漏与临时文件堆积。

## 典型用法
- 批量自动化处理：
  ```python
  from service.automated_processor import AutomatedGranaryProcessor
  processor = AutomatedGranaryProcessor()
  asyncio.run(processor.process_all_granaries('raw.csv'))
  ```

## 常见问题与注意事项
- psutil 非必需，无则自动降级但建议安装。
- 目录结构、数据路径需与 data_paths 工具保持一致。
- 任何异常均有详细 traceback 日志，便于定位。
- 预测部分为占位实现，需根据实际业务扩展。

---


# granarypredict/ingestion.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/ingestion.py
- 该文件是 SiloFlow 数据导入、标准化、分仓、存储、格式转换的核心模块。
- 支持多种原始数据格式（CSV、Parquet、压缩包等），自动标准化为统一 schema，分仓存储为 Parquet。
- 提供高性能去重、增量检测、兼容多种历史/第三方数据格式。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 导入 pandas、requests、logging、pathlib、typing 等。
- 配置全局 logger，所有操作均有详细日志。
- 依赖 granarypredict/config.py 提供的全局路径和 API 地址。

### 2. 数据导入与分仓
- ingest_and_sort_dataframe：
  - 直接处理 DataFrame，自动标准化、分 granary、去重、存储为 Parquet。
  - 支持返回 granary/silo 级别的增量变更信息，便于批量自动化。
- ingest_and_sort：
  - 处理原始文件（CSV/Parquet/压缩），自动格式识别，标准化、分仓、去重、存储。
  - 支持新老返回格式，兼容性强。
- read_granary_csv：
  - 自动识别并读取 Parquet、CSV、压缩包等多种格式。
  - 统一返回 pandas DataFrame。
- save_granary_data：
  - 支持 Parquet/CSV 存储，自动压缩，统一接口。
- convert_csv_to_parquet：
  - 支持批量 CSV → Parquet 转换，自动删除原文件（可选）。

### 3. 数据标准化与兼容性
- standardize_granary_csv：
  - 统一所有外部/历史/第三方 CSV 格式为内部标准 schema。
  - 支持中英文、历史别名、模糊匹配、自动推断 granary_id。
  - 自动类型转换，保证下游特征工程和训练兼容。
- _CANONICAL_MAP：
  - 定义所有支持的外部字段与内部字段的映射，支持多版本、历史别名。
- _rename_and_select/_coerce_types：
  - 列重命名、类型转换、冗余列剔除、异常容忍。
- standardize_result147：
  - 针对 Result_147.csv 等特殊历史格式的专用标准化。

### 4. API 数据拉取
- fetch_company_data：
  - 支持通过 REST API 拉取公司/第三方数据，自动转为 DataFrame。
  - 异常时自动降级为空表，保证健壮性。

### 5. 健壮性与性能优化
- 所有数据导入、存储、转换均有详细日志，便于追踪。
- 支持压缩存储（snappy/gzip/brotli/xz），极大节省空间。
- 兼容所有主流/历史/第三方数据格式，极大提升工程适应性。
- 增量检测与去重，避免重复计算和存储。
- 任何异常均有详细日志，保证数据安全。

## 典型用法
- 批量导入并分仓：
  ```python
  from granarypredict.ingestion import ingest_and_sort
  ingest_and_sort('raw.csv')
  ```
- 读取标准化数据：
  ```python
  from granarypredict.ingestion import read_granary_csv
  df = read_granary_csv('data/granaries/XXX.parquet')
  ```
- 标准化任意 CSV：
  ```python
  from granarypredict.ingestion import standardize_granary_csv
  df = standardize_granary_csv(df)
  ```

## 常见问题与注意事项
- granary_id/heap_id 必须唯一标识仓库/堆。
- 外部数据格式如有变动，需更新 _CANONICAL_MAP。
- 目录结构、数据路径需与 config.py、data_paths 工具保持一致。
- 任何异常均有详细日志，建议优先查看日志定位。

---


# granarypredict/cleaning.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/cleaning.py
- 该文件为 SiloFlow 项目的核心数据清洗模块，提供标准化、健壮的基础清洗与缺失值处理函数。
- 主要用于原始数据导入后、特征工程前的预处理阶段，保证数据质量和一致性。

## 结构与主要内容

### 1. 顶部依赖与日志
- 依赖 pandas、numpy，标准数据处理库。
- 配置全局 logger，所有清洗操作均有日志记录，便于调试与追踪。

### 2. basic_clean 函数
- 功能：对 DataFrame 进行基础清洗。
  - 列名去空格标准化。
  - 替换常见缺失值标记（-999、"-"、"NA"、"N/A"）为 NaN。
  - 删除全空列。
  - 去除重复行。
- 日志详细记录清洗前后数据形状、去重数量。
- 返回清洗后的 DataFrame，保证下游兼容性。

### 3. fill_missing 函数
- 功能：灵活填充缺失值，支持多种策略。
  - 数值列可选均值（mean）、中位数（median）、线性插值（interpolate）。
  - 非数值列可选前向填充（ffill）、后向填充（bfill）、众数（mode）。
  - limit 参数控制最大连续填充数量。
- 针对不同类型自动分列处理，异常参数会抛出 ValueError。
- 返回填充后的 DataFrame。

### 4. __all__
- 显式导出 basic_clean、fill_missing 两大核心函数，便于外部模块统一调用。

## 典型用法
- 基础清洗：
  ```python
  from granarypredict.cleaning import basic_clean
  df = basic_clean(df)
  ```
- 缺失值填充：
  ```python
  from granarypredict.cleaning import fill_missing
  df = fill_missing(df, strategy="ffill", numeric_strategy="mean")
  ```

## 健壮性与注意事项
- 所有操作均为副本处理，原始数据不被修改。
- 日志详细，便于定位清洗与填充过程中的问题。
- 填充策略需根据实际业务选择，避免引入偏差。
- 对于极端异常值或特殊缺失标记，建议在调用前统一标准化。

---


# granarypredict/features.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/features.py
- 该文件为 SiloFlow 项目的特征工程核心模块，负责所有特征构造、标签生成、并行加速等。
- 支持时间、空间、方向、稳定性、趋势、滞后、滚动、波动等多维特征，极大提升模型表现。
- 内置高性能并行处理，适合大规模数据。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、numpy、logging、multiprocessing、concurrent.futures。
- 可选依赖 streamlit（toast 通知）。
- 全局并行开关 _USE_PARALLEL，最大并发数 _MAX_WORKERS，自动适配 Windows/大数据场景。

### 2. 时间与空间特征
- create_time_features：自动识别时间戳列，生成年/月/日/小时、周期性编码（sin/cos）、周/日/周/周末等特征。
- create_spatial_features：空间索引预留，兼容 grid_x/y/z。

### 3. 类别与标签处理
- encode_categoricals：所有 object/category 列自动 label encoding。
- select_feature_target/select_feature_target_multi：特征与标签分离，支持单/多步预测。

### 4. 滞后与趋势特征
- add_sensor_lag：为每个传感器生成 1 天滞后温度（支持不规则采样）。
- add_multi_lag/_add_single_lag：批量生成多天滞后及温差特征，支持 1-30 天。
- add_rolling_stats：滚动均值/方差，支持分组。
- add_directional_features_lean：加速度、趋势、动量、平滑速度、趋势一致性等高影响特征。
- add_horizon_specific_directional_features：多尺度速度、动量、波动、趋势反转、区间位置等。

### 5. 稳定性与物理特征
- add_stability_features：温度稳定性、热惯性、变动阻力、历史波动、均衡温度、均值回归等。

### 6. 多步标签生成
- add_multi_horizon_targets：自动生成多步未来标签（如 temperature_grain_h1d, h2d...）。

### 7. 并行加速与高性能处理
- add_multi_lag_parallel/add_rolling_stats_parallel/add_stability_features_parallel：多进程/多线程并行特征工程，3-5 倍加速。
- preprocess_dataframe_parallel：完整并行流水线，适合大数据。
- set_parallel_processing/get_parallel_info：全局并行配置与查询。
- 所有并行函数均自动降级为串行，保证健壮性。

### 8. 导出接口
- __all__ 显式导出所有核心与并行特征工程函数，便于统一调用。

## 典型用法
- 时间/空间/滞后/趋势/稳定性特征批量生成：
  ```python
  from granarypredict.features import preprocess_dataframe_parallel
  df = preprocess_dataframe_parallel(df)
  ```
- 单步/多步标签分离：
  ```python
  from granarypredict.features import select_feature_target, select_feature_target_multi
  X, y = select_feature_target(df)
  X, Y = select_feature_target_multi(df)
  ```
- 并行配置：
  ```python
  from granarypredict.features import set_parallel_processing
  set_parallel_processing(True, max_workers=4)
  ```

## 健壮性与注意事项
- 并行处理自动适配内存/平台，极端大数据自动降级为串行。
- 所有特征工程均为副本处理，原始数据不变。
- 时间戳、传感器分组、温度列名需与实际数据一致。
- 并行加速需注意内存占用，建议分批处理超大数据。
- 任何异常均有详细日志与 toast 通知，便于定位。

---


# granarypredict/polars_data_utils.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/polars_data_utils.py
- 该文件为 SiloFlow 项目的 Polars 优化数据处理模块，专为大数据集排序、分组等操作加速。
- 自动检测 Polars，数据量大时（>5 万行）自动切换高性能后端，极大提升处理速度。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、logging，自动检测 polars。
- 全局 _SORT_HIERARCHY 定义标准排序优先级（granary_id、heap_id、grid_x/y/z、detection_time）。

### 2. 高性能排序与分组
- comprehensive_sort_optimized：自动选择 Polars/pandas 排序，5-15 倍加速，始终返回 pandas DataFrame。
- assign_group_id_optimized：自动选择 Polars/pandas 分组，生成 _group_id 列，兼容所有主流分组方式。
- _comprehensive_sort_polars/_assign_group_id_polars：Polars 实现，适合大数据。
- _comprehensive_sort_pandas/_assign_group_id_pandas：pandas 实现，兼容小数据。

### 3. 一体化数据管道
- optimized_data_pipeline：集成排序与分组，自动选择后端，适合批量数据预处理。

### 4. 兼容性与便捷接口
- comprehensive_sort/assign_group_id：原始接口的 drop-in 替代，自动加速。
- 所有函数异常时自动降级为 pandas，保证健壮性。

## 典型用法
- 大数据集排序与分组：
  ```python
  from granarypredict.polars_data_utils import comprehensive_sort, assign_group_id
  df = comprehensive_sort(df)
  df = assign_group_id(df)
  ```
- 一体化管道：
  ```python
  from granarypredict.polars_data_utils import optimized_data_pipeline
  df = optimized_data_pipeline(df)
  ```

## 性能与注意事项
- Polars 优化仅在数据量大时自动启用，小数据自动用 pandas。
- 所有接口始终返回 pandas DataFrame，保证下游兼容。
- 依赖 polars，未安装时自动降级，无需手动处理。
- 排序/分组字段需与 _SORT_HIERARCHY 保持一致。
- 任何异常均有详细日志，便于定位。

---


# granarypredict/gpu_data_utils.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/gpu_data_utils.py
- 该文件为 SiloFlow 项目的 GPU/cuDF 加速数据处理模块，专为超大数据集排序、分组、聚合等操作极致提速。
- 自动检测 cuDF（NVIDIA RAPIDS）、Polars、pandas，按数据量和硬件自动选择最优后端。
- 兼容所有主流数据管道，支持 10-150 倍加速。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、logging，自动检测 cudf、cupy、polars。
- detect_optimal_backend：根据数据量和 GPU 内存自动选择 cudf/polars/pandas。

### 2. GPU 加速排序与分组
- gpu_comprehensive_sort_optimized：自动选择 cudf/polars/pandas 排序，超大数据集 10-50 倍加速。
- gpu_assign_group_id_optimized：自动选择 cudf/polars/pandas 分组，生成 group_id 列，极大提升分组效率。

### 3. GPU 加速聚合
- gpu_aggregation_optimized：自动选择 cudf/polars/pandas 聚合，支持多种聚合函数（mean/std/count/sum/min/max），极致加速。

### 4. 一体化数据管道
- gpu_data_pipeline：集成排序、分组、聚合，自动选择后端，支持返回实际使用后端信息。

### 5. 后端与性能信息
- get_gpu_data_backend_info：查询 cudf/polars/pandas 可用性、推荐阈值、GPU 内存等。
- HAS_CUDF/HAS_POLARS：全局后端可用性标志。

### 6. 健壮性与降级
- 所有操作均自动降级为 polars/pandas，保证兼容性。
- 详细日志记录每一步后端选择与异常。

## 典型用法
- GPU 加速排序与分组：
  ```python
  from granarypredict.gpu_data_utils import gpu_comprehensive_sort_optimized, gpu_assign_group_id_optimized
  df = gpu_comprehensive_sort_optimized(df, ["granary_id", "heap_id", "detection_time"])
  df = gpu_assign_group_id_optimized(df, ["granary_id", "heap_id"])
  ```
- 一体化管道：
  ```python
  from granarypredict.gpu_data_utils import gpu_data_pipeline
  df = gpu_data_pipeline(df, sort_columns=["detection_time"], group_columns=["granary_id"], agg_config={"temperature": "mean"})
  ```
- 查询后端与性能：
  ```python
  from granarypredict.gpu_data_utils import get_gpu_data_backend_info
  info = get_gpu_data_backend_info()
  ```

## 性能与注意事项
- cuDF 仅支持 NVIDIA GPU 且需安装 RAPIDS，未检测到自动降级。
- Polars 适合中等数据量，pandas 兼容所有场景。
- 超大数据集建议优先用 gpu_data_pipeline，极大提升速度。
- 排序/分组/聚合字段需与实际数据一致。
- 任何异常均有详细日志，便于定位。

---



# granarypredict/model.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/model.py
- 该文件为 SiloFlow 项目的模型训练与推理核心模块，支持多种主流回归算法、模型压缩存储、健壮加载与推理。
- 兼容 LightGBM、RandomForest、sklearn GBDT 等，支持自动压缩、详细日志、全流程健壮性。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、numpy、joblib、sklearn、lightgbm、logging。
- 依赖 granarypredict/config.py、evaluate.py、compression_utils.py。
- 全局 logger，所有训练、保存、加载均有详细日志。

### 2. 支持的模型与训练
- train_random_forest：训练 RandomForestRegressor，自动分割训练/测试，返回模型与 MAE/RMSE。
- train_gb_models：训练 sklearn HistGradientBoosting/GBDT，支持 time_series_cv，返回模型与指标。
- train_lightgbm：训练 LightGBM，内置项目级调参默认，支持自定义参数，自动压缩存储。

### 3. 模型保存与加载
- save_model：支持自动压缩存储（调用 save_compressed_model），兼容大模型，返回压缩统计。
- load_model：自动检测压缩格式，优先用 load_compressed_model，失败时自动降级 joblib，详细错误日志。

### 4. 推理与导出
- predict：统一推理接口，兼容所有支持模型。
- __all__：显式导出所有核心训练、保存、加载、推理函数。

## 典型用法
- 训练与保存 LightGBM：
  ```python
  from granarypredict.model import train_lightgbm, save_model
  model, metrics = train_lightgbm(X, y)
  save_model(model, "lgbm_model.joblib")
  ```
- 加载与推理：
  ```python
  from granarypredict.model import load_model, predict
  model = load_model("lgbm_model.joblib")
  y_pred = predict(model, X_future)
  ```

## 健壮性与注意事项
- 所有训练/保存/加载均有详细日志，便于追踪。
- save_model 支持自动压缩，极大节省空间，建议优先使用。
- load_model 自动检测压缩格式，兼容所有历史模型。
- 支持 time_series_cv，适合时序数据。
- 训练参数可根据实际业务调整，默认参数为项目级调优结果。
- 任何异常均有详细日志，建议优先查看日志定位。

---


# service/utils/database_utils.py 详细交接说明

## 文件定位与作用
- 路径：service/utils/database_utils.py
- 该文件为 SiloFlow 项目的数据库与 CLI 工具核心模块，集中管理数据库连接、配置加载、通用 CLI 参数、子进程调用、配置校验等功能。
- 通过 DatabaseManager、CLIUtils、SubprocessUtils、ValidationUtils 等类，极大减少各脚本/服务的重复代码。

## 结构与主要内容

### 1. 顶部说明与依赖
- 文件头部有详细 docstring，说明用途。
- 依赖 argparse、json、logging、sys、pathlib、typing、pymysql、pandas。

### 2. DatabaseManager 类
- load_config：加载数据库配置 JSON 文件，支持异常处理。
- get_db_config：兼容嵌套（database 字段）与扁平格式的数据库配置提取。
- get_connection：根据配置创建 pymysql 数据库连接，异常自动抛出。
- test_connection：测试数据库连接，返回 True/False 并详细日志。

### 3. CLIUtils 类
- add_common_args：为 argparse 添加通用 --config 参数。
- add_granary_args：添加 --granary 参数。
- add_date_range_args：添加 --start/--end 日期参数。
- add_retrieval_mode_args：添加 --full-retrieval、--incremental、--date-range、--days 等数据拉取模式参数。

### 4. SubprocessUtils 类
- run_subprocess：统一子进程调用与错误处理，返回 (success, error_message, output_lines)。

### 5. ValidationUtils 类
- validate_config_file：校验配置文件是否存在。
- validate_date_format：校验日期字符串格式。
- validate_required_args：校验 argparse 必需参数。

### 6. 自定义异常
- SiloFlowError 及其子类（DatabaseError、ConfigError、ProcessingError、ValidationError），用于细粒度错误处理。

## 典型用法
- 加载数据库配置并连接：
  ```python
  from service.utils.database_utils import DatabaseManager
  config = DatabaseManager.load_config('config/streaming_config.json')
  conn = DatabaseManager.get_connection(config)
  ```
- 添加 CLI 通用参数：
  ```python
  from service.utils.database_utils import CLIUtils
  parser = argparse.ArgumentParser()
  parser = CLIUtils.add_common_args(parser)
  ```
- 子进程调用：
  ```python
  from service.utils.database_utils import SubprocessUtils
  ok, err, out = SubprocessUtils.run_subprocess(['ls', '-l'])
  ```

## 健壮性与注意事项
- 所有数据库/配置/子进程操作均有详细日志与异常处理。
- 支持多种数据库配置格式，适合不同历史/新配置。
- CLI 参数模式统一，便于批量脚本维护。
- 建议所有数据库/配置相关脚本均通过本模块调用，避免重复代码。
- 自定义异常便于上层捕获与处理。

---

## 文件定位与作用
- 路径：service/utils/validation_utils.py
- 该文件为 SiloFlow 项目的配置校验工具模块，主要用于校验 streaming/client 等配置文件的结构和类型，防止配置错误导致运行异常。
- 提供 validate_config 函数，支持多种配置类型的自动校验，所有错误均详细日志输出。

## 结构与主要内容

### 1. 顶部依赖与日志
- 依赖 logging，所有校验错误均有详细日志。

### 2. validate_config 函数
- 功能：校验传入的 config 字典是否符合指定 config_type（如 streaming、client）的 schema。
  - 内置 schemas 字典，定义各类配置的字段、类型、嵌套结构。
  - 支持 streaming（含 database/processing 嵌套）、client（扁平结构）等多种配置。
  - 校验每个 section、字段是否存在，类型是否匹配，支持 int/float 联合类型。
  - 所有缺失、类型错误均收集到 errors 列表，最终统一日志输出。
  - 校验通过返回 True，否则返回 False。

## 典型用法
- 校验 streaming 配置：
  ```python
  from service.utils.validation_utils import validate_config
  ok = validate_config(config, 'streaming')
  ```
- 校验 client 配置：
  ```python
  ok = validate_config(config, 'client')
  ```

## 健壮性与注意事项
- 仅支持 schemas 中定义的 config_type，其他类型需扩展 schemas。
- 所有校验错误均详细日志输出，便于定位配置问题。
- 支持嵌套结构与多类型字段校验。
- 建议所有配置文件加载后先用 validate_config 校验，防止后续运行异常。

---

## 文件定位与作用
- 路径：service/utils/data_paths.py
- 该文件为 SiloFlow 项目的数据路径管理核心模块，集中管理所有数据、模型、日志、临时文件等目录与文件路径。
- 通过 DataPathManager 类实现全局统一的数据目录、文件命名、自动创建、批量列举等功能，保证所有脚本和服务路径一致。

## 结构与主要内容

### 1. 顶部说明与依赖
- 文件头部有详细 docstring，说明用途。
- 依赖 pathlib、os、json、typing。

### 2. DataPathManager 类
- 构造函数 __init__：
  - 支持通过 config/data_paths.json 配置所有目录结构，未找到则用默认配置。
  - 自动定位 service 根目录，保证路径相对一致。
- _load_config/_get_default_config：加载或生成默认目录结构（granaries、processed、models、forecasts、temp、uploads、logs、streaming、batch 等）。
- get_path：按类型获取绝对路径，自动创建目录。
- get_granaries_dir/get_processed_dir/get_models_dir/get_forecasts_dir/get_temp_dir/get_uploads_dir/get_logs_dir/get_simple_retrieval_dir/get_streaming_output_dir/get_batch_output_dir：分别获取各类数据目录。
- get_granary_file/get_processed_file/get_model_file/get_forecast_file：按 granary 名称自动生成标准文件路径，支持扩展名、压缩等。
- ensure_directories：批量确保所有核心目录存在。
- list_granaries/list_processed_granaries/list_models/list_forecasts：批量列举所有 granary、处理后数据、模型、预测结果。
- get_data_summary：汇总所有数据文件数量与目录结构，便于监控与调试。
- 全局实例 data_paths，便于直接 import 使用。

## 典型用法
- 获取标准目录：
  ```python
  from service.utils.data_paths import data_paths
  print(data_paths.get_models_dir())
  ```
- 获取 granary 文件路径：
  ```python
  path = data_paths.get_granary_file('ABC123')
  ```
- 列举所有模型：
  ```python
  models = data_paths.list_models()
  ```
- 获取数据汇总：
  ```python
  summary = data_paths.get_data_summary()
  ```

## 健壮性与注意事项
- 所有目录、文件路径均自动创建，无需手动管理。
- 支持通过 config/data_paths.json 灵活扩展目录结构，适合多环境部署。
- 路径变更需同步更新配置文件或默认配置。
- 任何异常均有详细报错，便于定位。
- 推荐所有脚本、服务均通过 data_paths 管理路径，避免硬编码。

---

## 文件定位与作用
- 路径：granarypredict/config.py
- 该文件为 SiloFlow 项目的全局配置模块，集中管理数据目录、模型目录、环境变量、温度阈值、API 地址等核心参数。
- 所有数据路径、模型路径、告警阈值、API 基础地址等均在此统一配置，便于全局调用与维护。

## 结构与主要内容

### 1. 顶部依赖与环境加载
- 依赖 pathlib、os、dotenv（自动加载 .env 环境变量）。
- 加载 .env 文件，支持通过环境变量灵活配置。

### 2. 目录与路径配置
- ROOT_DIR：项目根目录，自动定位为本文件上两级目录。
- DATA_DIR/RAW_DATA_DIR/PROCESSED_DATA_DIR：数据主目录、原始数据目录、处理后数据目录。
- MODELS_DIR：模型存储目录。
- 导入时自动创建上述所有目录，保证目录结构完整，无需手动创建。

### 3. 温度阈值与粮食品种配置
- ALERT_TEMP_THRESHOLD：全局温度告警阈值（默认 28°C，可通过环境变量覆盖）。
- GRAIN_ALERT_THRESHOLDS：各粮食品种的专用温度阈值（如未列出则回退到全局阈值）。

### 4. API 地址配置
- METEOROLOGY_API_BASE：气象数据 API 基础地址（可通过环境变量覆盖）。
- COMPANY_API_BASE：公司/仓库数据 API 基础地址（可通过环境变量覆盖）。

### 5. __all__
- 显式导出所有核心配置项，便于外部统一调用。

## 典型用法
- 获取全局路径与阈值：
  ```python
  from granarypredict.config import DATA_DIR, ALERT_TEMP_THRESHOLD
  print(DATA_DIR, ALERT_TEMP_THRESHOLD)
  ```
- 获取 API 地址：
  ```python
  from granarypredict.config import METEOROLOGY_API_BASE
  print(METEOROLOGY_API_BASE)
  ```

## 健壮性与注意事项
- 所有目录在 import 时自动创建，保证下游模块无需关心目录是否存在。
- 支持通过 .env 或环境变量灵活配置所有参数，便于多环境部署。
- 粮食品种阈值如未配置则自动回退到全局阈值，保证健壮性。
- 任何路径、阈值、API 地址变更均需同步修改本文件。
- 依赖 python-dotenv，需确保已安装。

---

## 文件定位与作用
- 路径：granarypredict/evaluate.py
- 该文件为 SiloFlow 项目的模型评估与时序交叉验证模块，主要用于回归模型的时间序列交叉验证（TimeSeriesSplit）与评估指标计算。
- 提供统一的 time_series_cv 函数，便于对时序数据进行稳健的模型评估与参数调优。

## 结构与主要内容

### 1. 顶部依赖与日志
- 依赖 numpy、pandas、sklearn（base、metrics、model_selection）、logging。
- 配置全局 logger，所有评估与交叉验证过程均有日志输出。

### 2. time_series_cv 函数
- 功能：对回归模型进行时间序列感知的交叉验证（TimeSeriesSplit），返回模型与评估指标。
  - 支持自定义 n_splits（默认 5 折），适合时序预测任务。
  - 每一折用 clone(model) 训练，预测 test 区间，避免数据泄漏。
  - 预测结果自动拼接，最后仅对有效区间计算 MAE、RMSE。
  - 日志详细记录每折评估与最终指标。
  - 交叉验证后会用全量数据重新训练模型，便于后续保存与推理。
  - 返回值：(1) 已在全量数据上训练好的模型；(2) 指标字典（mae_cv, rmse_cv）。

### 3. __all__
- 显式导出 time_series_cv，便于外部统一调用。

## 典型用法
- 时序交叉验证与评估：
  ```python
  from granarypredict.evaluate import time_series_cv
  model, metrics = time_series_cv(model, X, y, n_splits=5)
  print(metrics)
  ```

## 健壮性与注意事项
- 仅适用于回归模型（需实现 fit/predict），适合 sklearn/LightGBM 等。
- 交叉验证采用 TimeSeriesSplit，严格保证未来数据不泄漏到训练集。
- 评估指标为 MAE、RMSE，适合绝大多数回归场景。
- 交叉验证后模型会被重训练，适合直接保存与推理。
- 任何异常均有详细日志，便于定位。

---

## 文件定位与作用
- 路径：granarypredict/multi_lgbm.py
- 该文件为 SiloFlow 项目的多目标 LGBM 训练器，专为 7 天多步温度预测、置信区间、GPU 加速、早停、定制指标等高级需求设计。
- 支持 anchor-day 早停、定制 MAE/方向/稳定性指标、原生不确定性量化、全流程 GPU 自动适配。

## 结构与主要内容

### 1. 置信区间与不确定性
- compute_prediction_intervals：多步预测置信区间，支持 68%/95% 等多级置信度。
- estimate_model_uncertainty：原生不确定性量化，支持 bootstrap 聚合，自动校准。

### 2. 定制评估指标
- conservative_mae_metric：保守 MAE，惩罚大幅波动，鼓励物理合理性。
- directional_mae_metric：方向性 MAE，兼顾准确率与趋势方向。

### 3. Anchor-day 早停与 7 天连续优化
- AnchorDayEarlyStoppingCallback：自定义早停，专为 7 天连续预测优化，支持 anchor-day 评估。

### 4. GPU 检测与参数优化
- detect_gpu_availability/get_optimal_gpu_params：自动检测 GPU/平台/驱动，按数据量/特征数自适应最优参数。

### 5. MultiLGBMRegressor 主类
- 支持多输出 LGBM，自动分 horizon 训练，支持 anchor-day 早停、horizon 加权、方向/稳定性特征增强、GPU 加速。
- fit：支持 horizon 加权、anchor-day 早停、定制回调、特征增强、GPU 自动适配。
- predict：批量多步预测，自动输出置信区间与不确定性。
- get_prediction_intervals/get_uncertainty_estimates：获取最近一次预测的置信区间与不确定性。
- feature_importances_：多输出特征重要性平均。

## 典型用法
- 7 天多步 LGBM 训练与预测：
  ```python
  from granarypredict.multi_lgbm import MultiLGBMRegressor
  model = MultiLGBMRegressor(use_gpu=True)
  model.fit(X_train, Y_train, eval_set=(X_val, Y_val), anchor_df=anchor_df)
  Y_pred = model.predict(X_test)
  lower, upper = model.get_prediction_intervals(0.95)
  ```

## 健壮性与注意事项
- GPU 自动检测与参数优化，未检测到自动降级为 CPU。
- anchor-day 早停需传入 anchor_df 与 eval_set，适合 7 天连续预测。
- 定制指标/特征增强可根据业务需求调整。
- 置信区间与不确定性量化为原生实现，适合工业级部署。
- 任何异常均有详细日志与提示，便于定位。

---


---

# granarypredict/compression_utils.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/compression_utils.py
- 该文件为 SiloFlow 项目的模型压缩与存储核心工具模块，提供自适应压缩策略、模型高效保存与加载、批量压缩等功能。
- 支持根据模型类型与大小自动选择最优压缩算法与参数，兼容 LightGBM、sklearn、集成模型等。
- 适用于模型训练、部署、归档等多种场景，极大提升存储效率与加载健壮性。

## 结构与主要内容

### 1. 顶部说明与依赖
- 文件头部有详细 docstring，说明模块用途。
- 依赖 gzip、joblib、logging、pathlib、typing、numpy。
- 依赖 granarypredict/compression_config.py 提供压缩参数配置。

### 2. 压缩策略与参数选择
- get_model_size_category：根据模型文件或字节数判断模型体积（small/medium/large/huge），用于后续压缩策略选择。
- get_optimal_compression_params：根据模型类型、属性（如集成、置信区间、多输出等）和体积，自动选择最优压缩算法（如 gzip）与压缩等级。
- 依赖 compression_config.py 的 determine_size_category、get_compression_config 实现灵活配置。

### 3. 模型保存与压缩
- save_compressed_model：核心函数，支持自适应压缩保存模型。
  - 自动检测模型类型与体积，优先用 gzip+joblib 双重压缩，或 joblib 内置压缩。
  - 支持自定义压缩参数，兼容 LightGBM、sklearn、集成模型等。
  - 保存前先生成未压缩临时文件，估算体积后再决定最终压缩参数。
  - 返回详细压缩统计（压缩前后体积、压缩比、用时、算法等），日志详细。

### 4. 模型加载与解压
- load_compressed_model：自适应加载压缩模型，支持多种解压策略与兼容性回退。
  - 优先尝试 gzip+joblib，失败则尝试 joblib 内置压缩、禁用 mmap、legacy gzip 等多种方式。
  - 自动检测 .gz 后缀，兼容历史与新格式。
  - 所有异常均详细收集并统一报错，日志详细。

### 5. LightGBM 专用压缩参数
- get_lightgbm_compression_params：生成 LightGBM 原生模型压缩参数（如 compress、compression_level、save_binary），便于训练时直接用高效二进制格式保存。

### 6. 现有模型文件压缩与批量压缩
- compress_existing_model：对已有未压缩模型文件进行压缩，支持 gzip 或 joblib 内置压缩，自动生成新文件并对比体积。
- batch_compress_models：批量压缩指定目录下所有模型文件，支持自定义文件模式、压缩等级、是否保留原文件。
  - 自动跳过已压缩文件，异常时详细日志。

## 典型用法
- 保存并压缩模型：
  ```python
  from granarypredict.compression_utils import save_compressed_model
  stats = save_compressed_model(model, 'model.joblib')
  print(stats)
  ```
- 加载压缩模型：
  ```python
  from granarypredict.compression_utils import load_compressed_model
  model = load_compressed_model('model.joblib.gz')
  ```
- 批量压缩目录下所有模型：
  ```python
  from granarypredict.compression_utils import batch_compress_models
  batch_compress_models('models/')
  ```

## 健壮性与注意事项
- 所有保存、加载、压缩操作均有详细日志，便于追踪与调试。
- 支持多种压缩算法与等级，自动适配模型类型与体积。
- 加载时多重回退，极大提升兼容性与健壮性。
- 批量压缩时自动跳过已压缩文件，异常不影响其他文件。
- LightGBM 建议直接用 get_lightgbm_compression_params 配合 save_model 使用。
- 目录、文件路径需确保存在写权限，建议统一用 pathlib 管理。
- 任何异常均有详细日志，建议优先查看日志定位。

---


---


# granarypredict/data_utils.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/data_utils.py
- 该文件为 SiloFlow 项目的通用数据处理与压缩工具模块，主要提供高效的数据排序、分组、文件压缩（CSV/Parquet）等实用函数。
- 适用于数据预处理、批量导入、特征工程等多种场景，支持灵活的压缩策略推荐。

## 结构与主要内容

### 1. 顶部依赖与全局配置
- 依赖 pandas、os、gzip、shutil、logging、pathlib、typing。
- 配置全局 logger，所有操作均有详细日志。

### 2. 数据排序与分组
- sort_and_group_by：
  - 按指定列对 DataFrame 进行排序和分组，返回分组后的 DataFrame 字典。
  - 支持 granary_id、heap_id、detection_time 等多级分组，适合批量仓库数据处理。
  - 日志详细记录分组与排序信息。

### 3. 文件压缩与格式转换
- compress_csv_file：
  - 对指定 CSV 文件进行 gzip 压缩，生成 .gz 文件。
  - 支持覆盖或保留原文件，压缩后自动校验文件体积。
- compress_parquet_file：
  - 对指定 Parquet 文件进行 gzip 压缩，生成 .gz 文件。
  - 支持覆盖或保留原文件，兼容大文件批量压缩。
- batch_compress_files：
  - 批量压缩指定目录下所有 CSV/Parquet 文件，支持自定义文件模式、压缩等级、是否保留原文件。
  - 自动跳过已压缩文件，异常时详细日志。

### 4. 压缩策略推荐
- recommend_compression_strategy：
  - 根据文件类型、体积、数据特征，推荐最优压缩算法与参数（如 gzip、lz4、snappy、brotli、lzma）。
  - 支持自定义目标压缩比、优先级，便于大数据场景下自动选择。
  - 返回推荐算法、等级、目标压缩比、说明等。

### 5. 兼容性与健壮性
- 所有压缩、分组、排序操作均有详细日志，异常时自动降级或跳过，保证批量处理健壮性。
- 支持灵活的文件模式与目录结构，适合多环境部署。

## 典型用法
- 按 granary_id 分组并排序：
  ```python
  from granarypredict.data_utils import sort_and_group_by
  groups = sort_and_group_by(df, by=["granary_id", "detection_time"])
  for granary, group_df in groups.items():
      ...
  ```
- 压缩单个 CSV 文件：
  ```python
  from granarypredict.data_utils import compress_csv_file
  compress_csv_file('data.csv')
  ```
- 批量压缩目录下所有 Parquet 文件：
  ```python
  from granarypredict.data_utils import batch_compress_files
  batch_compress_files('data/processed/', file_pattern='*.parquet')
  ```
- 推荐压缩策略：
  ```python
  from granarypredict.data_utils import recommend_compression_strategy
  strategy = recommend_compression_strategy('large_file.parquet')
  print(strategy)
  ```

## 健壮性与注意事项
- 所有文件操作均有详细日志，便于追踪与调试。
- 批量压缩时自动跳过已压缩文件，异常不影响其他文件。
- 支持多种压缩算法与等级，推荐优先用 recommend_compression_strategy 自动选择。
- 排序/分组字段需与实际数据一致，建议与 ingestion、features 等模块配合使用。
- 目录、文件路径需确保存在读写权限，建议统一用 pathlib 管理。
- 任何异常均有详细日志，建议优先查看日志定位。

---


# granarypredict/compression_config.py 详细交接说明

## 文件定位与作用
- 路径：granarypredict/compression_config.py
- 该文件为 SiloFlow 项目的模型压缩参数与策略配置核心模块，集中定义不同模型类型、体积、属性下的压缩算法、等级、目标压缩比等。
- 为 compression_utils.py 等模块提供统一的压缩策略、参数选择、兼容性适配等能力，确保模型存储高效、加载可靠。

## 结构与主要内容

### 1. 顶部说明与依赖
- 文件头部有详细 docstring，说明用途。
- 依赖 typing（Dict, Tuple, Optional, List）。

### 2. 模型体积阈值与压缩策略
- MODEL_SIZE_THRESHOLDS：定义 small/medium/large/huge 四档模型体积阈值（单位 MB），如 <50MB、50-200MB、200-500MB、>1GB。
- COMPRESSION_STRATEGIES：按体积档次定义默认压缩算法（如 lz4、gzip、lzma）、压缩等级、目标压缩比、说明。
  - small_models：lz4，快速压缩，适合小模型，目标压缩比 2.0。
  - medium_models：gzip，平衡速度与压缩比，目标 3.0。
  - large_models：lzma，高压缩比，适合大模型，目标 4.0。
  - huge_models：lzma，最高压缩等级，适合超大模型，目标 5.0。
  - uncertainty_models：lzma，最高压缩，专为不确定性/自举模型，目标 6.0。

### 3. 模型类型与属性适配
- MODEL_TYPE_OVERRIDES：针对主流模型类型（如 LGBMRegressor、MultiLGBMRegressor、RandomForestRegressor）和体积档，指定专用压缩算法与等级。
  - 例如 LGBMRegressor 小模型用 gzip-4，大模型用 lzma-6。
- SPECIAL_PATTERNS：针对特殊模型属性（如 bootstrap、uncertainty、ensemble、multi_output、multi_horizon）指定专用压缩策略（如最大压缩）。

### 4. 压缩参数选择主流程
- get_compression_config：核心接口，根据模型类型、体积、属性自动选择最优压缩算法与等级。
  - 优先匹配特殊属性（如不确定性、集成、多输出等），再匹配模型类型专用配置，最后按体积档次默认策略。
  - 返回 (algorithm, level) 二元组。
- determine_size_category：根据模型体积（MB）判断 small/medium/large/huge 档，默认 medium。
- get_target_compression_ratio：根据模型类型、体积、属性获取目标压缩比，便于评估压缩效果。

### 5. 导出接口
- __all__：显式导出所有核心配置项与接口，便于外部统一调用。

## 典型用法
- 获取模型压缩参数：
  ```python
  from granarypredict.compression_config import get_compression_config
  alg, lvl = get_compression_config(model_type='LGBMRegressor', model_size_mb=120, model_attributes=['ensemble'])
  print(alg, lvl)
  ```
- 判断模型体积档次：
  ```python
  from granarypredict.compression_config import determine_size_category
  cat = determine_size_category(80)
  print(cat)  # 输出 'medium'
  ```
- 获取目标压缩比：
  ```python
  from granarypredict.compression_config import get_target_compression_ratio
  ratio = get_target_compression_ratio('MultiLGBMRegressor', 300, ['multi_horizon'])
  print(ratio)
  ```

## 健壮性与注意事项
- 所有压缩参数均可灵活扩展，支持新模型类型、属性、压缩算法。
- 优先匹配特殊属性，保证不确定性/集成/多输出模型获得最大压缩。
- 体积判断默认 medium，极端情况可手动指定。
- 建议所有模型保存、压缩均通过本模块参数配置，保证一致性与兼容性。
- 任何参数变更需同步更新本文件及相关文档。

---

## 📊 监控与维护

### 系统监控

#### 服务健康监控
```bash
# 定期健康检查
curl -s http://localhost:8000/health | jq .

# 设置监控脚本（crontab）
*/5 * * * * curl -f http://localhost:8000/health || echo "SiloFlow service down" | mail admin@company.com
```

#### 性能监控
```python
# 使用内置监控工具
from service.utils.memory_utils import MemoryMonitor

monitor = MemoryMonitor()
stats = monitor.get_system_stats()
print(f"内存使用: {stats['memory_percent']:.1f}%")
print(f"CPU使用: {stats['cpu_percent']:.1f}%")
```

#### 日志监控
```bash
# 查看实时日志
tail -f siloflow.log

# 错误日志过滤
grep "ERROR" siloflow.log | tail -20

# 日志分析
awk '/ERROR/ {print $1, $2, $NF}' siloflow.log | sort | uniq -c
```

### 数据维护

#### 定期清理
```python
# 清理临时文件
from service.automated_processor import AutomatedGranaryProcessor
processor = AutomatedGranaryProcessor()
processor.cleanup_temp_files()

# 清理过期缓存
from granarypredict.optuna_cache import OptunaParameterCache
cache = OptunaParameterCache()
cache.clear_old_cache(days=30)
```

#### 数据备份
```bash
# 备份模型文件
tar -czf models_backup_$(date +%Y%m%d).tar.gz data/models/

# 备份配置文件
cp service/config/*.json backups/

# 数据库备份（如使用）
mysqldump siloflow_db > backup_$(date +%Y%m%d).sql
```

### 性能调优

#### 定期性能评估
```python
# 模型性能评估
from granarypredict.evaluate import time_series_cv
from granarypredict.model import load_model

model = load_model("ABC123_forecast_model.joblib")
_, metrics = time_series_cv(model, X_test, y_test)
print(f"模型性能: MAE={metrics['mae_cv']:.3f}")
```

#### 系统资源优化
```bash
# 检查磁盘使用
df -h

# 清理大文件
find . -type f -size +1G -exec ls -lh {} \;

# 内存使用分析
ps aux --sort=-%mem | head -10
```

## 📈 开发与维护建议

### 代码贡献指南

#### 开发环境设置
```bash
# 开发模式安装
pip install -e .

# 安装开发依赖
pip install -r requirements-dev.txt

# 代码格式化
black granarypredict/ service/
isort granarypredict/ service/
```

#### 测试指南
```bash
# 运行单元测试
python -m pytest tests/

# 性能测试
python service/scripts/testing/performance_test.py

# API测试
python service/scripts/client/run_client_tests.py --test-type full
```

### 版本更新流程

#### 代码更新
```bash
# 1. 创建功能分支
git checkout -b feature/new-feature

# 2. 开发和测试
# ... 开发代码 ...
python -m pytest tests/

# 3. 合并到主分支
git checkout master
git merge feature/new-feature

# 4. 部署更新
./deploy.sh
```

#### 模型更新
```python
# 重新训练所有模型
from service.automated_processor import AutomatedGranaryProcessor
processor = AutomatedGranaryProcessor()

# 批量重训练
results = await processor.retrain_all_models()
print(f"重训练完成: {results['successful_models']} 个模型")
```

### 故障应急响应

#### 紧急问题处理流程
1. **服务中断**：检查日志 → 重启服务 → 验证功能
2. **内存溢出**：降低chunk_size → 重启服务 → 监控内存
3. **预测异常**：检查模型文件 → 重新训练 → 验证预测
4. **数据异常**：验证数据格式 → 数据清洗 → 重新处理

#### 联系信息
- **技术负责人**：[待填写联系信息]
- **运维团队**：[待填写联系信息]  
- **紧急联系**：[待填写24小时联系方式]

---

## 📝 版本历史

### v2.0.0 (2025年7月)
- ✅ 重构文档结构，添加系统架构概览
- ✅ 新增部署指南和API文档
- ✅ 添加性能优化和故障排除指南
- ✅ 完善代码示例和工作流说明
- ✅ 新增监控维护指南

### v1.0.0 (之前版本)
- ✅ 基础技术文档
- ✅ 核心模块详细说明
- ✅ 代码级别分析

---

**文档维护者**：SiloFlow开发团队  
**最后更新**：2025年7月23日  
**文档版本**：v2.0.0

---
