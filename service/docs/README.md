# SiloFlow Automated Pipeline Service

Automated HTTP service for grain temperature forecasting that processes raw CSV data, trains models, and returns h+1 to h+7 forecasts.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd service
pip install -r requirements.txt
```

### 2. Start the Service
```bash
python start_service.py
```

The service will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📋 API Endpoints

### GET `/forecast`
**Silo-specific forecasting endpoint** - Generates forecasts for a specific silo within a granary.

**Parameters:**
- `granary_name` (required): Name or ID of the granary
- `silo_id` (required): ID of the silo within the granary  
- `horizon_days` (optional): Number of days to forecast (default: 7, max: 30)

**Example:**
```bash
curl -X GET "http://localhost:8000/forecast?granary_name=中正粮食储备库&silo_id=H1&horizon_days=7"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-01-15T14:30:22",
  "granary_name": "中正粮食储备库",
  "silo_id": "H1",
  "horizon_days": 7,
  "latest_data_date": "2025-01-14T23:00:00",
  "sensors_used": 15,
  "forecasts": [
    {
      "granary_id": "中正粮食储备库",
      "silo_id": "H1",
      "forecast_date": "2025-01-15",
      "forecast_horizon": 1,
      "grid_x": 1,
      "grid_y": 1,
      "grid_z": 1,
      "predicted_temperature_celsius": 22.5,
      "base_date": "2025-01-14",
      "sensor_location": "(1,1,1)",
      "uncertainty_std": 0.8,
      "confidence_lower_95": 21.1,
      "confidence_upper_95": 23.9
    }
  ],
  "summary": {
    "total_predictions": 105,
    "sensors_forecasted": 15,
    "avg_predicted_temperature": 22.3,
    "min_predicted_temperature": 21.8,
    "max_predicted_temperature": 23.1,
    "temperature_range": 1.3,
    "forecast_horizons": [1, 2, 3, 4, 5, 6, 7],
    "model_type": "MultiLGBMRegressor"
  }
}
```

### GET `/forecast/all`
**Legacy endpoint** - Previously generated forecasts for all granaries (now deprecated).

### POST `/pipeline`
**Full pipeline endpoint** - Processes raw CSV and returns forecasts for all granaries.

**Request:**
- `file`: Raw CSV file with data from multiple granaries
- `horizon`: Forecast horizon in days (default: 7)

**Example:**
```bash
curl -X POST "http://localhost:8000/forecast" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@combined_sensor_data.csv" \
     -F "horizon=7"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-01-15T14:30:22",
  "granaries_processed": 2,
  "successful_granaries": 2,
  "forecast_horizon_days": 7,
  "granaries": {
    "中正粮食储备库": {
      "granary_name": "中正粮食储备库",
      "forecast_horizon_days": 7,
      "last_historical_date": "2025-01-15",
      "total_sensors": 8,
      "sensor_coordinates": [
        {"x": 1, "y": 1, "z": 1},
        {"x": 1, "y": 1, "z": 2},
        {"x": 1, "y": 2, "z": 1},
        {"x": 1, "y": 2, "z": 2},
        {"x": 2, "y": 1, "z": 1},
        {"x": 2, "y": 1, "z": 2},
        {"x": 2, "y": 2, "z": 1},
        {"x": 2, "y": 2, "z": 2}
      ],
      "daily_forecasts": {
        "day_1": {
          "date": "2025-01-16",
          "sensors": [
            {
              "sensor_location": {"x": 1, "y": 1, "z": 1},
              "temperature_celsius": 18.5,
              "uncertainty": 0.2,
              "prediction_interval": {
                "lower_bound": 18.1,
                "upper_bound": 18.9
              }
            },
            {
              "sensor_location": {"x": 1, "y": 1, "z": 2},
              "temperature_celsius": 18.7,
              "uncertainty": 0.3,
              "prediction_interval": {
                "lower_bound": 18.3,
                "upper_bound": 19.1
              }
            }
          ]
        },
        "day_2": {
          "date": "2025-01-17",
          "sensors": [
            {
              "sensor_location": {"x": 1, "y": 1, "z": 1},
              "temperature_celsius": 18.7,
              "uncertainty": 0.3,
              "prediction_interval": {
                "lower_bound": 18.3,
                "upper_bound": 19.1
              }
            }
          ]
        }
      },
      "summary": {
        "min_temperature": 18.5,
        "max_temperature": 20.1,
        "average_temperature": 19.3
      }
    },
    "蚬冈库": {
      "granary_name": "蚬冈库",
      "forecast_horizon_days": 7,
      "last_historical_date": "2025-01-15",
      "total_sensors": 6,
      "sensor_coordinates": [
        {"x": 1, "y": 1, "z": 1},
        {"x": 1, "y": 1, "z": 2},
        {"x": 1, "y": 2, "z": 1},
        {"x": 1, "y": 2, "z": 2},
        {"x": 2, "y": 1, "z": 1},
        {"x": 2, "y": 1, "z": 2}
      ],
      "daily_forecasts": {
        "day_1": {
          "date": "2025-01-16",
          "sensors": [
            {
              "sensor_location": {"x": 1, "y": 1, "z": 1},
              "temperature_celsius": 19.2,
              "uncertainty": 0.3,
              "prediction_interval": {
                "lower_bound": 18.6,
                "upper_bound": 19.8
              }
            }
          ]
        }
      },
      "summary": {
        "min_temperature": 19.2,
        "max_temperature": 20.9,
        "average_temperature": 20.1
      }
    }
  }
}
```

### POST `/process`
**Process-only endpoint** - Ingests and processes granaries without forecasting.

**Request:**
- `file`: Raw CSV file with data from multiple granaries

**Example:**
```bash
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@combined_sensor_data.csv"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-01-15T14:30:22",
  "granaries_processed": 2,
  "successful_granaries": 2,
  "results": {
    "中正粮食储备库": {
      "success": true,
      "steps_completed": ["preprocess", "train"],
      "model_path": "models/中正粮食储备库_forecast_model.joblib"
    },
    "蚬冈库": {
      "success": true,
      "steps_completed": ["preprocess"],
      "model_path": "models/蚬冈库_forecast_model.joblib"
    }
  }
}
```

### GET `/health`
**Health check endpoint** for monitoring and load balancers.

**Response:**
```json
{
  "status": "healthy",
  "service": "SiloFlow Automated Pipeline",
  "timestamp": "2025-01-15T14:30:22",
  "directories": {
    "models": true,
    "data/processed": true,
    "data/granaries": true
  }
}
```

### GET `/models`
**List all available trained models.**

**Response:**
```json
{
  "status": "success",
  "models_count": 2,
  "models": [
    {
      "granary": "中正粮食储备库",
      "model_path": "models/中正粮食储备库_forecast_model.joblib",
      "size_mb": 45.2,
      "modified": "2025-01-15T14:30:22"
    },
    {
      "granary": "蚬冈库",
      "model_path": "models/蚬冈库_forecast_model.joblib",
      "size_mb": 52.1,
      "modified": "2025-01-15T14:30:22"
    }
  ]
}
```

### DELETE `/models/{granary_name}`
**Delete a specific granary's model.**

**Example:**
```bash
curl -X DELETE "http://localhost:8000/models/中正粮食储备库"
```

**Response:**
```json
{
  "status": "success",
  "message": "Model deleted for granary: 中正粮食储备库"
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the service directory:

```env
# Service Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Model Configuration
MAX_WORKERS=4
MODEL_CACHE_SIZE=10

# File Upload Configuration
MAX_FILE_SIZE=100MB
TEMP_UPLOAD_DIR=temp_uploads
```

### Production Deployment

**Using Docker:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_service.py"]
```

**Using systemd service:**
```ini
[Unit]
Description=SiloFlow Automated Pipeline Service
After=network.target

[Service]
Type=simple
User=siloflow
WorkingDirectory=/opt/siloflow/service
ExecStart=/usr/bin/python3 start_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 📊 Input Data Format

Your CSV should contain these columns (or equivalents):

| Column | Description | Required |
|--------|-------------|----------|
| `granary_id` or `storepointName` | Granary identifier | ✅ |
| `heap_id` or `storeName` | Silo identifier | ✅ |
| `detection_time` or `batch` | Timestamp | ✅ |
| `temperature_grain` or `temp` | Target temperature | ✅ |
| `grid_x`, `grid_y`, `grid_z` or `x`, `y`, `z` | Sensor coordinates | ✅ |
| `temperature_inside` | Indoor temperature | ❌ |
| `humidity_warehouse` | Warehouse humidity | ❌ |
| `temperature_outside` | Outdoor temperature | ❌ |

**Example CSV:**
```csv
storepointName,storeName,batch,temp,x,y,z,indoor_temp,outdoor_temp
中正粮食储备库,101,2023-03-06 14:47:04,18.0,1,1,1,18.88,28.0
中正粮食储备库,101,2023-03-06 14:47:04,18.0,1,1,2,18.88,28.0
蚬冈库,P1-1-01堆位,2023-03-06 14:47:04,19.5,2,1,1,19.2,28.0
```

## 🔄 Automated Workflow

The service automatically:

1. **Ingests** raw CSV and splits by granary
2. **Preprocesses** each granary's data:
   - Standardizes column names
   - Cleans and deduplicates
   - Drops redundant columns
   - Inserts calendar gaps
   - Interpolates missing data
   - Applies comprehensive feature engineering
3. **Manages models**:
   - Uses existing models if available
   - Trains new models if missing
   - Auto-retrains if needed
4. **Generates forecasts**:
   - h+1 to h+7 temperature predictions
   - Uncertainty estimates
   - Confidence intervals
5. **Returns results** via HTTP response

## 🚨 Error Handling

The service provides comprehensive error handling:

- **File validation**: Ensures CSV format
- **Data validation**: Checks required columns
- **Processing errors**: Graceful handling of pipeline failures
- **Model errors**: Fallback for missing or corrupted models
- **HTTP errors**: Proper status codes and error messages

## 📈 Monitoring

### Health Checks
```bash
curl http://localhost:8000/health
```

### Logs
Service logs are written to `service.log` and stdout.

### Metrics
Monitor these key metrics:
- Request processing time
- Success/failure rates
- Model training time
- Memory usage
- Disk space usage

## 🔒 Security Considerations

For production deployment:

1. **HTTPS**: Use SSL/TLS certificates
2. **Authentication**: Implement API key or JWT authentication
3. **Rate Limiting**: Add request rate limiting
4. **File Validation**: Strict file type and size validation
5. **CORS**: Configure CORS appropriately for your domain
6. **Input Sanitization**: Validate and sanitize all inputs

## 🐛 Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check dependencies
pip install -r requirements.txt

# Check Python path
python -c "import granary_pipeline; print('OK')"

# Check directories
ls -la models/ data/processed/ data/granaries/
```

**Forecast generation fails:**
```bash
# Check model files
ls -la models/*.joblib

# Check processed data
ls -la data/processed/*.csv

# Check logs
tail -f service.log
```

**Memory issues:**
- Increase system memory
- Reduce batch size
- Use model caching
- Implement data streaming

## 📞 Support

For issues and questions:
1. Check the logs in `service.log`
2. Review the API documentation at `/docs`
3. Test with sample data first
4. Verify input data format

## 📄 License

This service is part of the SiloFlow project and follows the same license terms. 