# SiloFlow Client Testing Guide

This guide explains how to use the new `/process` endpoint and the comprehensive client testing tools for the SiloFlow service.

## 🆕 New `/process` Endpoint

The `/process` endpoint is designed for data ingestion and preprocessing only, without training or forecasting. This is useful when you want to:

1. **Prepare data** for later processing
2. **Split large datasets** into granary-specific files
3. **Preprocess data** without the overhead of model training
4. **Validate data** before running the full pipeline

### Endpoint Details

- **URL**: `POST /process`
- **Input**: CSV or Parquet file
- **Output**: JSON response with processing status
- **Timeout**: 1 hour (configurable)

### What `/process` Does

1. **Ingests** the uploaded CSV/Parquet file
2. **Splits data** by granary into separate Parquet files
3. **Preprocesses** each granary's data (cleaning, feature engineering)
4. **Saves** processed files to `data/processed/`
5. **Returns** processing status for each granary

### Example Response

```json
{
  "status": "success",
  "timestamp": "2025-01-15T14:30:22",
  "granaries_processed": 2,
  "successful_granaries": 2,
  "results": {
    "中正粮食储备库": {
      "success": true,
      "steps_completed": ["preprocess"],
      "processed_file": "data/processed/中正粮食储备库_processed.parquet",
      "file_size_mb": 45.2
    },
    "蚬冈库": {
      "success": true,
      "steps_completed": ["preprocess"],
      "processed_file": "data/processed/蚬冈库_processed.parquet",
      "file_size_mb": 52.1
    }
  }
}
```

## 🧪 Client Testing Tools

### 1. Main Client Tester (`siloflow_client_tester.py`)

A comprehensive testing tool that mimics the functionality of the local testing service but designed for network clients.

#### Features

- **Connection Testing**: Verify service health and accessibility
- **Endpoint Testing**: Test all available endpoints (`/process`, `/pipeline`, `/train`, `/forecast`, `/models`)
- **File Upload Testing**: Test with real CSV files
- **Sample Data Generation**: Auto-generate sample data for testing
- **Detailed Reporting**: Comprehensive test results with JSON output
- **Error Handling**: Robust error handling and timeout management

#### Usage

```bash
# Basic usage with server IP
python siloflow_client_tester.py --server 192.168.1.100 --port 8000

# Use configuration file
python siloflow_client_tester.py --config client_config.json

# Test with specific file
python siloflow_client_tester.py --server 192.168.1.100 --file your_data.csv

# Create sample data and test
python siloflow_client_tester.py --server 192.168.1.100 --create-sample
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--server` | Server IP address | Required |
| `--port` | Server port | 8000 |
| `--timeout` | Request timeout (seconds) | 300 |
| `--file` | CSV file to test with | None |
| `--create-sample` | Create sample CSV file | False |
| `--config` | Configuration file (JSON) | None |

### 2. Example Runner (`run_client_tests.py`)

A simplified script that demonstrates different testing scenarios.

#### Usage

```bash
# Run all tests
python run_client_tests.py --server 192.168.1.100 --test-type full

# Run only basic connectivity tests
python run_client_tests.py --server 192.168.1.100 --test-type basic

# Run only process endpoint tests
python run_client_tests.py --server 192.168.1.100 --test-type process

# Run only pipeline endpoint tests
python run_client_tests.py --server 192.168.1.100 --test-type pipeline
```

#### Test Types

- **`basic`**: Connection and health checks only
- **`process`**: Test `/process` endpoint
- **`pipeline`**: Test `/pipeline` endpoint
- **`full`**: Run all available tests

### 3. Configuration File (`client_config.json`)

A JSON configuration file for easy setup and reuse.

```json
{
  "server": "192.168.1.100",
  "port": 8000,
  "timeout": 300,
  "file": "sample_sensor_data.csv",
  "description": "SiloFlow Client Testing Configuration"
}
```

## 📊 Testing Workflow

### 1. Basic Connectivity Test

```bash
python run_client_tests.py --server 192.168.1.100 --test-type basic
```

**Expected Output:**
```
🔍 Running Basic Tests...
==================================================
🔍 Testing connection to SiloFlow service...
✅ Service is healthy!
   Status: healthy
   Service: SiloFlow Automated Pipeline
   models: ✅
   data/processed: ✅
   data/granaries: ✅
✅ Basic connectivity test passed!
```

### 2. Process Endpoint Test

```bash
python run_client_tests.py --server 192.168.1.100 --test-type process
```

**Expected Output:**
```
📊 Running Process Tests...
==================================================
📄 Creating sample CSV file for testing...
📄 Created sample CSV file: sample_sensor_data.csv
📊 Testing /process endpoint with sample_sensor_data.csv...
✅ Processing successful!
   Granaries processed: 2
   Successful: 2
   ✅ 中正粮食储备库: 0.15 MB
   ✅ 蚬冈库: 0.12 MB
✅ Process endpoint test passed!
```

### 3. Full Pipeline Test

```bash
python run_client_tests.py --server 192.168.1.100 --test-type pipeline
```

**Expected Output:**
```
🔄 Running Pipeline Tests...
==================================================
🔄 Testing /pipeline endpoint with sample_sensor_data.csv (horizon: 7 days)...
✅ Pipeline processing successful!
   Granaries processed: 2
   Successful: 2
   Total forecast records: 28
   Forecast saved: forecast_20250115_143022.csv
   ✅ 中正粮食储备库: 14 records
   ✅ 蚬冈库: 14 records
✅ Pipeline endpoint test passed!
```

### 4. Complete Test Suite

```bash
python siloflow_client_tester.py --server 192.168.1.100 --create-sample
```

**Expected Output:**
```
🚀 Starting comprehensive SiloFlow client tests...
   Server: http://192.168.1.100:8000
   Timeout: 300 seconds
------------------------------------------------------------
🔍 Testing connection to SiloFlow service...
✅ Service is healthy!
📊 Testing /process endpoint with sample_sensor_data.csv...
✅ Processing successful!
🔄 Testing /pipeline endpoint with sample_sensor_data.csv (horizon: 7 days)...
✅ Pipeline processing successful!
🏋️ Testing /train endpoint...
✅ Training successful!
📋 Testing /models endpoint...
✅ Models listing successful!
🔮 Testing /forecast endpoint for ['中正粮食储备库', '蚬冈库'] (horizon: 7 days)...
✅ Forecast generation successful!

📊 TEST SUMMARY
============================================================
Total Tests: 6
Passed: 6 ✅
Failed: 0 ❌

📄 Detailed results saved to: test_results_20250115_143022.json
🎉 All tests passed! SiloFlow service is working correctly.
```

## 🔧 Advanced Usage

### Custom Client Script

```python
from siloflow_client_tester import SiloFlowClientTester

# Create tester instance
tester = SiloFlowClientTester("192.168.1.100", 8000, timeout=600)

# Test specific endpoints
if tester.test_connection():
    print("Service is accessible")
    
    # Test process endpoint
    success = tester.test_process_endpoint("your_data.csv")
    if success:
        print("Data processing successful")
    
    # Test pipeline endpoint
    success = tester.test_pipeline_endpoint("your_data.csv", horizon=14)
    if success:
        print("Full pipeline successful")
```

### Error Handling

The client tester includes comprehensive error handling:

- **Network errors**: Connection timeouts, DNS resolution failures
- **HTTP errors**: 4xx and 5xx status codes
- **File errors**: Missing files, permission issues
- **Processing errors**: Timeouts, validation failures

### Logging

All tests generate detailed logs:

- **Console output**: Real-time progress and results
- **Log file**: `client_test.log` with detailed information
- **Results file**: JSON file with complete test results

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if server is running: `python start_service.py`
   - Verify IP address and port
   - Check firewall settings

2. **Timeout Errors**
   - Increase timeout value: `--timeout 600`
   - Check network speed and file size
   - Consider using smaller test files

3. **File Not Found**
   - Verify file path is correct
   - Use `--create-sample` to generate test data
   - Check file permissions

4. **Processing Errors**
   - Verify CSV format matches expected schema
   - Check for missing required columns
   - Ensure data quality (no corrupted values)

### Debug Mode

For detailed debugging, modify the logging level in the script:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📋 Requirements

### Server Requirements
- SiloFlow service running on port 8000
- Network accessibility from client machines
- Sufficient disk space for processed files

### Client Requirements
- Python 3.8+
- Required packages:
  ```bash
  pip install requests pandas
  ```

## 🔄 Integration with Existing Workflows

The `/process` endpoint can be integrated into existing workflows:

1. **Data Validation**: Use `/process` to validate data before full pipeline
2. **Batch Processing**: Process multiple files sequentially
3. **Quality Assurance**: Check processing results before training
4. **Development**: Test data preprocessing without full pipeline overhead

This comprehensive testing suite ensures reliable operation of the SiloFlow service in network environments! 