# 🚀 SiloFlow 快速入门指南

**SiloFlow** 是一个智能粮仓温度预测系统，提供自动化数据处理、机器学习管道和实时预测服务。

## 📋 目录
- [系统要求](#-系统要求)
- [安装指南](#️-安装指南)
- [初始配置](#️-初始配置)
- [启动服务](#-启动服务)
- [使用测试GUI](#-使用测试gui)
- [数据管道工作流](#-数据管道工作流)
- [API使用指南](#-api使用指南)
- [仪表板访问](#-仪表板访问)
- [常用操作](#-常用操作)
- [故障排除](#-故障排除)

---

## 🔧 系统要求

### 最低要求
- **Python**: 3.8+ (推荐: 3.11+)
- **内存**: 8GB (大数据集推荐16GB+)
- **存储**: 10GB可用空间
- **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

### 可选GPU支持
- **NVIDIA GPU**: 用于加速数据处理
- **CUDA**: 11.2+ (GPU加速所需)
- **内存**: 使用GPU时推荐16GB+

---

## 🛠️ 安装指南

### 步骤1: 克隆仓库
```bash
git clone https://github.com/kaiyuanli7/siloflow.git
cd siloflow
```

### 步骤2: 创建Python虚拟环境
```bash
# Windows
python -m venv siloflow-env
siloflow-env\Scripts\activate

# macOS/Linux
python3 -m venv siloflow-env
source siloflow-env/bin/activate
```

### 步骤3: 安装依赖

#### 标准安装 (CPU)
```bash
pip install -r requirements.txt
```

#### GPU加速 (可选)
```bash
# 首先安装标准依赖
pip install -r requirements.txt

# 然后安装GPU依赖
pip install -r requirements-gpu.txt
```

### 步骤4: 以开发模式安装包
```bash
pip install -e .
```

### 步骤5: 验证安装
```bash
python -c "import granarypredict; print('✅ SiloFlow安装成功!')"
```

---

## ⚙️ 初始配置

### 步骤1: 数据库配置
创建或更新 `service/config/streaming_config.json`:
```json
{
  "database": {
    "host": "your-database-host",
    "port": 3306,
    "user": "your-username",
    "password": "your-password", 
    "database": "your-database-name"
  },
  "data_paths": {
    "raw_data": "data/raw",
    "processed_data": "data/processed",
    "models": "models"
  }
}
```

### 步骤2: 生产环境配置
更新 `service/config/production_config.json`:
```json
{
  "service": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "model": {
    "retrain_interval_hours": 24,
    "forecast_horizon_days": 7
  }
}
```

### 步骤3: 客户端测试配置
更新 `service/config/client_config.json`:
```json
{
  "server": "localhost",
  "port": 8000,
  "timeout": 300,
  "file": "sample_sensor_data.csv"
}
```

---

## 🚀 启动服务

### 方法1: 直接启动FastAPI服务
```bash
# 导航到项目根目录
cd siloflow

# 启动服务
python -m uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
```

### 方法2: 使用启动脚本
```bash
# 启动生产服务
python service/start_service.py

# 或使用自定义配置
python service/start_service.py --config service/config/production_config.json
```

### 方法3: 后台服务
```bash
# 在后台启动
nohup python -m uvicorn service.main:app --host 0.0.0.0 --port 8000 > service.log 2>&1 &
```

### 验证服务运行
打开浏览器访问: http://localhost:8000/docs
您应该看到FastAPI交互式文档。

---

## 🧪 使用测试GUI

测试GUI是您进行系统交互和测试的主要工具。

### 步骤1: 启动测试界面
```bash
python service/scripts/testing/testingservice.py
```

### 步骤2: GUI概览
界面提供6个主要标签页:

#### 🌐 HTTP服务测试
- **目的**: 测试API端点和上传文件
- **快速开始**:
  1. 选择服务URL (本地/远程)
  2. 测试连接
  3. 选择测试文件 (CSV/Parquet)
  4. 选择要测试的端点
  5. 查看响应

#### 🌍 远程客户端测试  
- **目的**: 测试远程服务部署
- **快速开始**:
  1. 输入远程服务URL
  2. 运行全面的端点测试
  3. 查看详细测试报告

#### 📊 简单检索
- **目的**: 从数据库提取数据
- **快速开始**:
  1. 配置数据库连接
  2. 选择粮仓和筒仓
  3. 设置日期范围
  4. 执行检索
  5. 检查输出文件

#### 🚀 生产管道
- **目的**: 运行完整的数据处理管道
- **快速开始**:
  1. 加载生产配置
  2. 选择管道阶段
  3. 监控系统资源
  4. 查看处理日志

#### 🗄️ 数据库浏览器
- **目的**: 探索数据库结构和数据
- **快速开始**:
  1. 测试数据库连接
  2. 浏览粮仓和筒仓
  3. 探索数据分布
  4. 导出元数据

#### 🔄 批量处理
- **目的**: 处理多个文件或操作
- **快速开始**:
  1. 选择输入文件夹
  2. 选择处理操作
  3. 监控批处理进度
  4. 查看结果

---

## 📊 数据管道工作流

### 工作流1: 初始数据设置
```bash
# 1. 从数据库检索数据
python service/scripts/testing/testingservice.py
# → 使用"简单检索"标签页
# → 选择您的粮仓和日期范围
# → 执行检索

# 2. 验证data/raw/中的数据
ls data/raw/
```

### 工作流2: 训练新模型
```bash
# 选项A: 通过API
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"granary_id": "your_granary", "retrain": true}'

# 选项B: 通过测试GUI
# → 使用"HTTP服务测试"标签页
# → 选择"/train"端点
# → 配置参数
# → 执行训练
```

### 工作流3: 运行预测
```bash
# 上传数据并获取预测
curl -X POST "http://localhost:8000/forecast" \
  -F "file=@your_data.csv" \
  -F "granary_id=your_granary"
```

### 工作流4: 完整管道处理
```bash
# 通过测试GUI
# → 使用"生产管道"标签页
# → 加载生产配置
# → 选择所有管道阶段
# → 执行完整工作流
```

---

## 🔗 API使用指南

### 健康检查
```bash
curl http://localhost:8000/health
```

### 上传和处理数据
```bash
curl -X POST "http://localhost:8000/pipeline" \
  -F "file=@sensor_data.csv" \
  -F "granary_id=granary_001" \
  -F "operation=process"
```

### 训练模型
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "granary_id": "granary_001",
    "retrain": true,
    "hyperparameter_tuning": true
  }'
```

### 生成预测
```bash
curl -X POST "http://localhost:8000/forecast" \
  -F "file=@current_data.csv" \
  -F "granary_id=granary_001" \
  -F "forecast_days=7"
```

### 列出可用模型
```bash
curl http://localhost:8000/models
```

### 获取处理状态
```bash
curl http://localhost:8000/status
```

---

## 📱 仪表板访问

### 步骤1: 启动仪表板
```bash
# 导航到app目录
cd app

# 启动Streamlit仪表板
streamlit run Dashboard.py

# 或指定端口
streamlit run Dashboard.py --server.port 8501
```

### 步骤2: 访问仪表板
打开浏览器访问: http://localhost:8501

### 仪表板功能
- **数据可视化**: 实时温度监控
- **模型性能**: 准确性指标和验证结果
- **预测图表**: 交互式预测可视化
- **系统状态**: 服务健康和资源监控

---

## 🔄 常用操作

### 日常操作检查清单

#### 1. 系统健康检查
```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查仪表板
curl http://localhost:8501

# 查看系统日志
tail -f service.log
```

#### 2. 数据刷新
```bash
# 检索最新数据
python service/scripts/testing/testingservice.py
# → 使用"简单检索"获取今天的数据
```

#### 3. 模型更新
```bash
# 使用新数据重新训练模型
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"retrain": true}'
```

### 每周操作

#### 1. 批量处理
```bash
# 处理一周的数据
python service/scripts/testing/testingservice.py
# → 使用"批量处理"标签页
# → 选择每周数据文件夹
```

#### 2. 系统维护
```bash
# 清理临时文件
find data/processed/temp -type f -mtime +7 -delete

# 归档旧日志
mkdir -p logs/archive
mv service.log logs/archive/service_$(date +%Y%m%d).log
```

---

## 🐛 故障排除

### 常见问题

#### 问题1: 服务无法启动
```bash
# 检查端口可用性
netstat -tulpn | grep :8000

# 杀死现有进程
pkill -f "uvicorn"

# 重启服务
python -m uvicorn service.main:app --host 0.0.0.0 --port 8000
```

#### 问题2: 数据库连接失败
```bash
# 测试数据库连接
python -c "
import json
from granarypredict.config import get_streaming_config
config = get_streaming_config()
print('数据库配置加载成功')
"

# 检查配置中的数据库凭据
cat service/config/streaming_config.json
```

#### 问题3: GUI无法启动
```bash
# 检查tkinter安装
python -c "import tkinter; print('GUI库可用')"

# 如果缺少tkinter则安装 (Ubuntu)
sudo apt-get install python3-tk

# 强制重新安装GUI依赖
pip install --force-reinstall -r requirements.txt
```

#### 问题4: 内存不足
```bash
# 检查内存使用
free -h

# 在配置中减少批处理大小
# 编辑 service/config/production_config.json
# 减少"batch_size"参数
```

#### 问题5: 模型训练失败
```bash
# 检查训练数据
python -c "
import pandas as pd
df = pd.read_csv('data/processed/latest_data.csv')
print(f'数据形状: {df.shape}')
print(f'列名: {df.columns.tolist()}')
"

# 清理模型缓存
rm -rf models/cache/*
rm -rf optuna_cache/*
```

### 获取帮助

#### 日志分析
```bash
# 查看详细服务日志
tail -f -n 100 service.log

# 搜索错误
grep -i error service.log | tail -20

# 检查API访问日志
grep "POST\|GET" service.log | tail -10
```

#### 系统诊断
```bash
# 运行内置诊断
python service/scripts/testing/testingservice.py
# → 使用"数据库浏览器"标签页
# → 测试所有连接
```

#### 性能监控
```bash
# 监控系统资源
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'内存: {psutil.virtual_memory().percent}%')
print(f'磁盘: {psutil.disk_usage(\"/\").percent}%')
"
```

---

## 🎯 下一步

### 高级配置
- 查看 `handover.md` 获取详细模块文档
- 为大数据集配置GPU加速
- 设置自动监控和警报
- 实施备份和灾难恢复

### 生产部署
- 配置反向代理 (nginx/Apache)
- 设置SSL证书
- 实施身份验证和授权
- 配置负载均衡以实现高可用性

### 开发
- 使用pytest设置开发环境
- 配置CI/CD管道
- 实施自定义特征工程
- 为特定用例扩展API端点

---

## 📞 支持

- **文档**: 查看 `handover.md` 获取详细技术文档
- **问题**: 通过GitHub issues报告错误
- **性能**: 使用测试GUI进行性能分析
- **配置**: 检查 `service/config/` 中的所有配置文件

**祝您使用SiloFlow预测愉快! 🌾📈**
