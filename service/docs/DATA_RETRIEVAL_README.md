# Automated Data Retrieval System for SiloFlow

This system automatically retrieves raw CSV data from your MySQL database for all granaries and silos, and processes it through the pipeline for analysis and forecasting.

## Overview

The automated data retrieval system consists of three main components:

1. **SQL Data Streamer** (`sql_data_streamer.py`) - Core data retrieval engine
2. **Automated Data Retrieval** (`automated_data_retrieval.py`) - High-level automation orchestrator
3. **Setup Script** (`setup_data_retrieval.py`) - System setup and testing

## Features

- **Automatic Discovery**: Discovers all granaries and silos in your database
- **Intelligent Date Range**: Automatically determines available data ranges
- **Memory Management**: Optimized for large datasets with memory monitoring
- **Incremental Processing**: Only processes new data since last run
- **Pipeline Integration**: Seamlessly connects to existing SiloFlow pipeline
- **Comprehensive Logging**: Detailed logs for monitoring and debugging

## Quick Start

### 1. Setup and Testing

```bash
# Navigate to service directory
cd service

# Run setup script
python setup_data_retrieval.py
```

This will:
- Create necessary directories
- Set up configuration file with your database credentials
- Test database connection
- Test data retrieval with a small sample

### 2. List Available Granaries

```bash
# List all available granaries with their IDs and silo counts
python list_granaries.py
```

### 3. Full Data Retrieval

```bash
# Retrieve all available data from all granaries
python automated_data_retrieval.py --full-retrieval

# Retrieve data from a specific granary only (much faster)
python automated_data_retrieval.py --full-retrieval --granary "蚬冈库"
python automated_data_retrieval.py --full-retrieval --granary "中正粮食储备库"
```

### 4. Incremental Retrieval

```bash
# Retrieve data from the last 7 days (all granaries)
python automated_data_retrieval.py --incremental --days 7

# Retrieve data from the last 7 days (specific granary only)
python automated_data_retrieval.py --incremental --days 7 --granary "蚬冈库"

# Retrieve data from the last 30 days (specific granary only)
python automated_data_retrieval.py --incremental --days 30 --granary "蚬冈库"
```

### 5. Specific Date Range

```bash
# Retrieve data for specific date range (all granaries)
python automated_data_retrieval.py --date-range --start 2024-01-01 --end 2024-12-31

# Retrieve data for specific date range (specific granary only)
python automated_data_retrieval.py --date-range --start 2024-01-01 --end 2024-12-31 --granary "蚬冈库"
```

## Configuration

The system uses `streaming_config.json` for configuration:

```json
{
  "database": {
    "host": "rm-wz9805aymm47k9z3qxo.mysql.rds.aliyuncs.com",
    "port": 3306,
    "database": "cloud_lq",
    "user": "userQuey",
    "password": "UserQ@20240807soft"
  },
  "processing": {
    "initial_chunk_size": 50000,
    "min_chunk_size": 10000,
    "max_chunk_size": 150000,
    "memory_threshold_percent": 75,
    "output_dir": "data/streaming",
    "log_level": "INFO"
  },
  "advanced": {
    "max_retries": 3,
    "retry_delay": 5,
    "enable_progress_bar": true
  }
}
```

## Database Schema

The system works with your existing MySQL database structure:

### Key Tables:
- `cloud_server.base_store_location_other` - Granary locations
- `cloud_server.v_store_list` - Store hierarchy (granaries and silos)
- `cloud_lq.lq_point_history_[XX]` - Sensor data (XX = granary table suffix)
- `cloud_lq.lq_store_history_[XX]` - Environmental data

### Data Flow:
1. **Discovery**: Queries granary and silo information
2. **Date Range**: Determines available data for each silo
3. **Retrieval**: Fetches sensor and environmental data
4. **Processing**: Runs through SiloFlow pipeline (preprocessing only)
5. **Storage**: Saves as Parquet files for efficient access

## Output Structure

```
data/
├── streaming/
│   └── granaries/
│       ├── 蚬冈库.parquet
│       ├── 中正粮食储备库.parquet
│       └── [other_granaries].parquet
├── processed/
│   ├── 蚬冈库_processed.parquet
│   ├── 中正粮食储备库_processed.parquet
│   └── [other_granaries]_processed.parquet
└── models/
    ├── 蚬冈库_forecast_model.joblib
    ├── 中正粮食储备库_forecast_model.joblib
    └── [other_granaries]_forecast_model.joblib
```

## Usage Patterns

### Development/Testing
```bash
# List available granaries first
python list_granaries.py

# Small date range for testing (single granary)
python automated_data_retrieval.py --date-range --start 2024-12-01 --end 2024-12-07 --granary "蚬冈库"
```

### Production (Full Processing)
```bash
# Complete data retrieval and processing (all granaries)
python automated_data_retrieval.py --full-retrieval

# Complete data retrieval for single granary (much faster)
python automated_data_retrieval.py --full-retrieval --granary "蚬冈库"
```

### Daily Incremental Updates
```bash
# Daily cron job for incremental updates (all granaries)
python automated_data_retrieval.py --incremental --days 1

# Daily cron job for specific granary
python automated_data_retrieval.py --incremental --days 1 --granary "蚬冈库"
```

### Data Only (No Pipeline)
```bash
# Retrieve data without processing
python sql_data_streamer.py --start-date 2024-01-01 --end-date 2024-12-31 --no-pipeline
```

## Performance Optimization

### Memory Management
- **Chunked Processing**: Processes data in configurable chunks
- **Memory Monitoring**: Automatically adjusts chunk size based on memory usage
- **Garbage Collection**: Cleans up memory during processing

### Processing Optimization
- **Parquet Format**: 60-80% smaller file sizes, 10x faster I/O
- **Incremental Updates**: Only processes new data
- **Parallel Processing**: Efficient handling of multiple granaries

### Recommended Settings
```json
{
  "processing": {
    "initial_chunk_size": 50000,    // Start with 50K records
    "min_chunk_size": 10000,        // Minimum 10K records
    "max_chunk_size": 150000,       // Maximum 150K records
    "memory_threshold_percent": 75  // Reduce chunk size at 75% memory
  }
}
```

## Monitoring and Logging

### Log Files
- `sql_data_streamer.log` - Core data retrieval logs
- `automated_data_retrieval.log` - Automation orchestration logs
- `data_retrieval_report.txt` - Summary reports

### Key Metrics
- **Processing Time**: Total time for data retrieval and processing
- **Records Processed**: Number of records retrieved and processed
- **Memory Usage**: Peak memory usage during processing
- **Success Rate**: Percentage of successful granary processing

### Example Log Output
```
2024-12-01 10:00:00 - INFO - Starting data streaming from 2024-01-01 to 2024-12-31
2024-12-01 10:00:05 - INFO - Found 15 silos across 3 granaries
2024-12-01 10:00:10 - INFO - Processing granary: 蚬冈库 (ID: aa1259ce34644e799aa1dcb08a79ee87)
2024-12-01 10:00:15 - INFO - Successfully processed silo 41f2257ce3d64083b1b5f8e59e80bc4d: 125,430 records
2024-12-01 10:05:00 - INFO - Data streaming completed: 3 processed granaries, 1,250,000 total records
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database credentials in streaming_config.json
   # Test connection manually
   python setup_data_retrieval.py
   ```

2. **Memory Issues**
   ```bash
   # Reduce chunk sizes in configuration
   # Monitor memory usage during processing
   # Use smaller date ranges for testing
   ```

3. **No Data Found**
   ```bash
   # Check date ranges in database
   # Verify granary and silo IDs
   # Test with smaller date range
   ```

4. **Pipeline Processing Failed**
   ```bash
   # Check granary_pipeline.py dependencies
   # Verify processed data format
   # Review pipeline logs
   ```

### Debug Mode
```bash
# Enable debug logging
# Edit streaming_config.json: "log_level": "DEBUG"
python automated_data_retrieval.py --incremental --days 1
```

## Integration with Existing Pipeline

The data retrieval system integrates seamlessly with your existing SiloFlow pipeline:

1. **Data Collection**: Retrieves raw data from MySQL database
2. **Preprocessing**: Runs through existing cleaning and feature engineering
3. **Storage**: Saves processed data in Parquet format
4. **Model Training**: Can trigger model training (optional)
5. **Forecasting**: Ready for forecasting pipeline

### Pipeline Integration Points
- Uses existing `granary_pipeline.py` for processing
- Maintains same data format and structure
- Compatible with existing models and forecasts
- Supports incremental updates

## Security Considerations

- **Database Credentials**: Stored in configuration file (ensure file permissions)
- **Network Security**: Uses SSL/TLS for database connections
- **Data Privacy**: Processes data locally, no external transmission
- **Access Control**: Follows database user permissions

## Future Enhancements

1. **Real-time Streaming**: Continuous data ingestion
2. **Cloud Integration**: AWS S3, Azure Blob storage
3. **Advanced Scheduling**: Cron jobs, task queues
4. **Monitoring Dashboard**: Web-based monitoring interface
5. **Alerting**: Email/SMS notifications for failures

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Review the troubleshooting section
3. Test with smaller date ranges
4. Verify database connectivity and permissions

## License

This system is part of the SiloFlow project and follows the same licensing terms. 