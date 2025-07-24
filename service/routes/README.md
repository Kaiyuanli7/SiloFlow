# SiloFlow Route Structure

## Overview
The SiloFlow service routes have been reorganized into focused, single-purpose endpoints for better separation of concerns and clearer API structure.

## Route Files and Their Purpose

### `/health` - System Health Check
**File:** `health.py`
- **Purpose:** Service health monitoring
- **Endpoints:** `GET /health`
- **Function:** Check service status and directory availability

### `/models` - Model Management
**File:** `models.py`  
- **Purpose:** Model listing and information
- **Endpoints:** `GET /models`
- **Function:** List available trained models with pagination

### `/sort` - Data Ingestion & Sorting
**File:** `sort.py`
- **Purpose:** Raw data ingestion and sorting only
- **Endpoints:** `POST /sort`
- **Function:** 
  - Ingests uploaded CSV/Parquet files
  - Sorts and deduplicates data
  - Splits by granary into separate files
  - Saves to `data/granaries/`

### `/process` - Data Preprocessing
**File:** `process.py`
- **Purpose:** Data preprocessing only (no training/forecasting)
- **Endpoints:** `POST /process`
- **Function:**
  - Ingests and sorts data (calls sort functionality)
  - Applies data cleaning and feature engineering
  - Saves processed files to `data/processed/`

### `/train` - Model Training
**File:** `train.py`
- **Purpose:** Model training only
- **Endpoints:** `POST /train`
- **Function:**
  - Trains models for processed granaries
  - Saves trained models to `models/`
  - Skips granaries that already have models

### `/forecast` - Forecast Generation
**File:** `forecast.py`
- **Purpose:** Forecast generation only
- **Endpoints:** `GET /forecast`
- **Function:**
  - Generates forecasts for granaries with trained models
  - Uses existing processed data and models
  - Returns forecast results

### `/pipeline` - Full End-to-End Pipeline
**File:** `pipeline.py`
- **Purpose:** Complete workflow orchestration
- **Endpoints:** `POST /pipeline`
- **Function:**
  - Runs complete pipeline: sort → process → train → forecast
  - End-to-end one-stop solution
  - Returns combined forecast CSV with processing summary

## Workflow Patterns

### Incremental Workflow (Recommended for Large Datasets)
1. `POST /sort` - Ingest and sort raw data
2. `POST /process` - Clean and preprocess data
3. `POST /train` - Train models
4. `GET /forecast` - Generate forecasts

### Quick Workflow (For Smaller Datasets)
1. `POST /pipeline` - Complete end-to-end processing

### Selective Operations
- `POST /sort` only - When you just need sorted data
- `POST /process` only - When you need processed data for external use
- `POST /train` only - When you have processed data and need models
- `GET /forecast` only - When you have models and need fresh forecasts

## Benefits of This Structure

1. **Single Responsibility:** Each endpoint has one clear purpose
2. **Better Error Handling:** Easier to isolate and debug issues
3. **Resource Management:** Can process large datasets incrementally
4. **Flexibility:** Choose the right endpoint for your specific needs
5. **Clearer API:** More intuitive endpoint names and purposes
6. **Monitoring:** Better tracking of which step failed in complex workflows
