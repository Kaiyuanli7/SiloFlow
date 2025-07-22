# SiloFlow Service - Grain Temperature Forecasting API

## 🌾 Overview

The SiloFlow Service is an automated grain temperature forecasting platform that provides REST API endpoints for processing grain sensor data and generating multi-horizon temperature forecasts. The service is designed for enterprise-scale operations with automated data retrieval from MySQL databases, intelligent preprocessing, model training, and forecasting capabilities.

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Data Retrieval System](#-data-retrieval-system)
- [Configuration](#-configuration)
- [Testing & Development](#-testing--development)
- [File Structure](#-file-structure)
- [Troubleshooting](#-troubleshooting)
- [Production Deployment](#-production-deployment)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SiloFlow Service Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │  Data Sources   │    │   Processing    │    │    Outputs      │ │
│  │                 │    │    Pipeline     │    │                 │ │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤ │
│  │ • MySQL DB      │───▶│ • Data Ingestion│───▶│ • REST API      │ │
│  │ • CSV/Parquet   │    │ • Preprocessing │    │ • Forecasts     │ │
│  │ • Manual Upload │    │ • Model Training│    │ • Model Files   │ │
│  │                 │    │ • Forecasting   │    │                 │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Component Details                            │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ • FastAPI Service (main.py, routes/)                       │ │
│  │ • Automated Processor (automated_processor.py)             │ │
│  │ • Granary Pipeline (granary_pipeline.py)                   │ │
│  │ • Data Retrieval System (scripts/data_retrieval/)          │ │
│  │ • Testing GUI (scripts/testing/testingservice.py)          │ │
│  │ • Utilities (utils/)                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MySQL database access (optional for database retrieval)
- Dependencies: `pip install -r requirements.txt`

### 1. Start the Service
```bash
# Option 1: Using the startup script (recommended)
python start_service.py

# Option 2: Direct FastAPI startup
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Using Python directly
python main.py
```

The service will start on `http://localhost:8000`

### 2. Verify Service Health
```bash
curl http://localhost:8000/health
```

### 3. View API Documentation
Open your browser to: `http://localhost:8000/docs`

### 4. Quick Test with Sample Data
```bash
# Upload a CSV file for processing
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_data.csv"

# Generate forecasts for all processed granaries
curl http://localhost:8000/forecast
```

## 📋 API Endpoints

### Core Processing Endpoints

#### `POST /process`
**Purpose**: Process-only endpoint for data ingestion and preprocessing without training
- **Input**: CSV or Parquet file with grain sensor data
- **Process**: Ingestion → Granary splitting → Preprocessing → Feature engineering
- **Output**: Processing status for each granary

```bash
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sensor_data.csv"
```

#### `POST /pipeline`
**Purpose**: Full pipeline including processing, training, and forecasting
- **Input**: CSV or Parquet file + optional horizon parameter
- **Process**: Ingestion → Preprocessing → Model Training → Forecasting
- **Output**: Complete pipeline results with forecasts

```bash
curl -X POST "http://localhost:8000/pipeline" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sensor_data.csv" \
     -F "horizon=7"
```

#### `GET /forecast`
**Purpose**: Generate forecasts for all granaries with trained models
- **Input**: None (uses existing processed data and models)
- **Process**: Loads models → Generates 1-day forecasts
- **Output**: Forecast data for all available granaries

```bash
curl http://localhost:8000/forecast
```

### Utility Endpoints

#### `GET /health`
**Purpose**: Service health check
```bash
curl http://localhost:8000/health
```

#### `GET /models`
**Purpose**: List all available trained models
```bash
curl http://localhost:8000/models
```

#### `POST /train`
**Purpose**: Train models for specific or all granaries
```bash
curl -X POST http://localhost:8000/train
```

### Response Format
All endpoints return JSON responses with this structure:
```json
{
  "status": "success|error",
  "timestamp": "2025-07-22T10:30:00",
  "data": {...},
  "errors": [...],
  "granaries_processed": 2,
  "processing_time": "120.5s"
}
```

## 📊 Data Retrieval System

The service includes automated data retrieval from MySQL databases with memory-optimized streaming.

### Configuration
Configure database access in `config/streaming_config.json`:
```json
{
  "database": {
    "host": "your-db-host",
    "port": 3306,
    "database": "cloud_lq",
    "user": "your-username",
    "password": "your-password"
  }
}
```

### Data Retrieval Scripts

#### Full Database Streaming
```bash
cd scripts/data_retrieval
python sql_data_streamer.py --start-date 2024-01-01 --end-date 2024-12-31
```

#### Simple Single-Silo Retrieval
```bash
python simple_data_retrieval.py \
  --granary-name "蚬冈库" \
  --silo-id "41f2257ce3d64083b1b5f8e59e80bc4d" \
  --start-date "2024-07-17" \
  --end-date "2024-07-18"
```

#### Automated Batch Retrieval
```bash
python automated_data_retrieval.py \
  --date-range \
  --start 2024-12-01 \
  --end 2024-12-07 \
  --granary "蚬冈库"
```

### Database Utilities
- `list_granaries.py` - List all available granaries in the database
- `get_silos_for_granary.py` - Get all silos for a specific granary
- `get_date_ranges.py` - Check available date ranges for each silo

## ⚙️ Configuration

### Main Configuration Files

#### `config/streaming_config.json` - Database Configuration
```json
{
  "database": {
    "host": "your-host",
    "port": 3306,
    "database": "cloud_lq",
    "user": "username",
    "password": "password"
  },
  "processing": {
    "initial_chunk_size": 50000,
    "memory_threshold_percent": 75,
    "output_dir": "data/streaming"
  }
}
```

#### `config/data_paths.json` - Directory Structure
```json
{
  "granaries_dir": "data/granaries",
  "processed_dir": "data/processed",
  "models_dir": "data/models",
  "forecasts_dir": "data/forecasts"
}
```

### Directory Structure
```
service/
├── data/
│   ├── granaries/          # Raw granary-specific data files
│   ├── processed/          # Cleaned and preprocessed data
│   ├── models/            # Trained ML models (.joblib files)
│   ├── forecasts/         # Generated forecast files
│   └── temp/              # Temporary processing files
├── temp_uploads/          # Incoming file uploads
└── logs/                  # Service logs
```

## 🧪 Testing & Development

### Interactive Testing GUI
Launch the comprehensive testing interface:
```bash
cd scripts/testing
python testingservice.py
```

Features:
- **HTTP Service Testing**: Test all API endpoints
- **Simple Data Retrieval**: GUI for database retrieval
- **Database Explorer**: Browse available granaries and silos
- **Batch Processing**: Process multiple files
- **Logs & Monitoring**: View service logs and performance

### Client Testing
Test remote service instances:
```bash
cd scripts/client
python siloflow_client_tester.py --server your-server-ip --create-sample
```

### Command Line Testing
```bash
# Test service health
curl http://localhost:8000/health

# Test with sample file
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_data.csv"
```

## 📁 File Structure

```
service/
├── main.py                     # FastAPI application entry point
├── start_service.py           # Service startup script
├── core.py                    # Core singleton instances
├── automated_processor.py     # Main processing engine
├── granary_pipeline.py        # Data pipeline orchestrator
├── routes/                    # API route definitions
│   ├── __init__.py           # Router aggregation
│   ├── pipeline.py           # /process and /pipeline endpoints
│   ├── forecast.py           # /forecast endpoint
│   ├── health.py             # /health endpoint
│   ├── models.py             # /models and /train endpoints
│   └── train.py              # Training-specific endpoints
├── utils/                     # Utility modules
│   ├── data_paths.py         # Centralized path management
│   └── database_utils.py     # Database utilities
├── scripts/                   # Standalone scripts
│   ├── data_retrieval/       # Database retrieval scripts
│   │   ├── sql_data_streamer.py          # Main streaming script
│   │   ├── simple_data_retrieval.py     # Simple retrieval
│   │   └── automated_data_retrieval.py  # Automated batch retrieval
│   ├── database/             # Database utility scripts
│   │   ├── list_granaries.py            # List available granaries
│   │   ├── get_silos_for_granary.py     # Get silos for granary
│   │   └── get_date_ranges.py           # Check date ranges
│   ├── testing/              # Testing and development tools
│   │   └── testingservice.py            # GUI testing interface
│   ├── client/               # Client testing scripts
│   └── parquet_inspector.py  # Parquet file inspection utility
├── config/                    # Configuration files
│   ├── streaming_config.json # Database configuration
│   ├── data_paths.json       # Directory paths
│   └── production_config.json# Production settings
├── docs/                      # Additional documentation
└── README.md                  # This file
```

## 🔧 Core Components Explained

### `automated_processor.py` - Processing Engine
The heart of the system that handles:
- File ingestion and format detection
- Granary separation and data splitting
- Preprocessing pipeline (cleaning, gap insertion, interpolation)
- Feature engineering (temporal, spatial, lag features)
- Model training with hyperparameter optimization
- Multi-horizon forecasting (1-7 days)
- Memory management and error handling

### `granary_pipeline.py` - Pipeline Orchestrator
Modular pipeline for processing individual granaries:
- **Ingest**: Data sorting, deduplication, standardization
- **Preprocess**: Cleaning, gap insertion, feature engineering
- **Train**: Model fitting with Dashboard-optimized settings
- **Forecast**: Multi-horizon prediction

### `routes/` - API Layer
FastAPI route definitions:
- **pipeline.py**: Main processing endpoints (`/process`, `/pipeline`)
- **forecast.py**: Forecasting endpoint (`/forecast`)
- **health.py**: Health check endpoint (`/health`)
- **models.py**: Model management (`/models`, `/train`)

### `scripts/data_retrieval/` - Database Integration
- **sql_data_streamer.py**: Memory-optimized streaming from MySQL
- **simple_data_retrieval.py**: Single-silo retrieval
- **automated_data_retrieval.py**: Batch retrieval with date ranges

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
- **Check dependencies**: `pip install -r requirements.txt`
- **Check ports**: Ensure port 8000 is available
- **Check logs**: `service.log` in the service directory

#### Memory Issues During Processing
- **Reduce chunk size** in streaming_config.json
- **Lower memory threshold** (e.g., from 75% to 60%)
- **Close other applications** to free memory

#### Database Connection Failed
- **Check configuration** in `config/streaming_config.json`
- **Test connection** using `scripts/testing/testingservice.py`
- **Verify credentials** and network access

#### Models Not Found
- **Run preprocessing first**: Use `/process` endpoint
- **Check model directory**: `data/models/` should contain `.joblib` files
- **Retrain models**: Use `/train` endpoint

### Log Files
- **service.log**: Main service logs
- **sql_data_streamer.log**: Database retrieval logs
- **simple_data_retrieval.log**: Simple retrieval logs

### Performance Monitoring
```bash
# Check service health
curl http://localhost:8000/health

# Monitor resource usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Check disk space
python -c "import psutil; print(f'Disk: {psutil.disk_usage('.').free/1024**3:.1f}GB free')"
```

## 🌐 Production Deployment

### System Requirements
- **OS**: Windows/Linux with Python 3.8+
- **Memory**: 8GB+ recommended (16GB+ for large datasets)
- **Storage**: 50GB+ free space for data and models
- **Network**: Access to MySQL database

### Production Checklist
1. **Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate.bat  # Windows
   pip install -r requirements.txt
   ```

2. **Configuration**:
   - Update `config/streaming_config.json` with production database
   - Configure `config/data_paths.json` for production paths
   - Set up SSL certificates for HTTPS

3. **Security**:
   - Change default passwords
   - Configure CORS properly
   - Set up authentication if needed
   - Use environment variables for sensitive data

4. **Monitoring**:
   - Set up log rotation
   - Monitor disk space and memory usage
   - Configure health check alerts
   - Set up backup procedures for models and data

5. **Service Management**:
   ```bash
   # For production, use a process manager like systemd or supervisor
   # Example systemd service file:
   [Unit]
   Description=SiloFlow Service
   After=network.target
   
   [Service]
   Type=simple
   User=siloflow
   WorkingDirectory=/path/to/service
   ExecStart=/path/to/.venv/bin/python start_service.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

### Performance Optimization
- **Use SSD storage** for faster I/O operations
- **Configure memory limits** based on available RAM
- **Use database connection pooling** for high-load scenarios
- **Enable model compression** to save storage space
- **Set up data retention policies** to manage disk usage

### Scaling Options
- **Horizontal scaling**: Deploy multiple service instances behind a load balancer
- **Database optimization**: Use read replicas, indexing, and query optimization
- **Model serving**: Use dedicated model serving infrastructure for high-throughput forecasting
- **Containerization**: Use Docker for consistent deployments

## 📝 API Usage Examples

### Complete Workflow Example
```python
import requests
import pandas as pd

# 1. Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# 2. Process data
with open('sensor_data.csv', 'rb') as f:
    files = {'file': ('sensor_data.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8000/process', files=files)
    print(response.json())

# 3. Train models
response = requests.post('http://localhost:8000/train')
print(response.json())

# 4. Generate forecasts
response = requests.get('http://localhost:8000/forecast')
forecasts = response.json()
print(f"Generated forecasts for {forecasts['forecasts_count']} granaries")
```

### Data Retrieval and Processing Workflow
```bash
# 1. Retrieve data from database
cd scripts/data_retrieval
python sql_data_streamer.py --start-date 2024-01-01 --end-date 2024-01-31

# 2. Process the retrieved data
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/granaries/combined_granaries.csv"

# 3. Train models
curl -X POST http://localhost:8000/train

# 4. Generate forecasts
curl http://localhost:8000/forecast
```

## 📞 Support & Maintenance

### Daily Operations
1. **Health Check**: `curl http://localhost:8000/health`
2. **Log Review**: Check service logs for errors
3. **Data Retrieval**: Run incremental data updates
4. **Forecast Generation**: Generate daily forecasts

### Weekly Maintenance
1. **Model Performance Review**: Check forecast accuracy
2. **Data Quality Check**: Review input data statistics
3. **System Resource Check**: Monitor CPU, memory, disk usage
4. **Backup**: Backup models and configuration files

### Monthly Maintenance
1. **Model Retraining**: Retrain with fresh data
2. **Performance Optimization**: Review and optimize settings
3. **Security Updates**: Update dependencies and credentials
4. **Capacity Planning**: Review growth and resource needs

---

**SiloFlow Service v2.0** - Automated Grain Temperature Forecasting Platform  
For additional support, check the documentation in the `docs/` directory or review the inline code comments.
