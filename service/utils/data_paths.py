#!/usr/bin/env python3
"""
Data Path Manager for SiloFlow
==============================

Centralized data path management to ensure all scripts use the same
directory structure and file locations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

class DataPathManager:
    """Centralized data path management for SiloFlow service."""
    
    def __init__(self, config_file: str = "config/data_paths.json"):
        """Initialize with data paths configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.service_root = Path(__file__).parent.parent
        
    def _load_config(self) -> Dict:
        """Load data paths configuration."""
        config_path = Path(__file__).parent.parent / self.config_file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get optimized default data paths configuration."""
        return {
            "data_root": "data",
            "granaries_dir": "data/granaries",
            "processed_dir": "data/processed",
            "models_dir": "data/models", 
            "forecasts_dir": "data/forecasts",
            "temp_dir": "data/temp",
            "uploads_dir": "temp_uploads",
            "logs_dir": "logs",
            "simple_retrieval_dir": "data/simple_retrieval",
            "streaming_output_dir": "data/streaming",
            "batch_output_dir": "data/batch"
        }
    
    def get_path(self, path_type: str) -> Path:
        """Get absolute path for a specific data directory type."""
        if path_type not in self.config:
            raise ValueError(f"Unknown path type: {path_type}")
        
        relative_path = self.config[path_type]
        absolute_path = self.service_root / relative_path
        
        # Ensure directory exists
        absolute_path.mkdir(parents=True, exist_ok=True)
        
        return absolute_path
    
    def get_granaries_dir(self) -> Path:
        """Get granaries directory path."""
        return self.get_path("granaries_dir")
    
    def get_processed_dir(self) -> Path:
        """Get processed data directory path."""
        return self.get_path("processed_dir")
    
    def get_models_dir(self) -> Path:
        """Get models directory path."""
        return self.get_path("models_dir")
    
    def get_forecasts_dir(self) -> Path:
        """Get forecasts directory path."""
        return self.get_path("forecasts_dir")
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory path."""
        return self.get_path("temp_dir")
    
    def get_uploads_dir(self) -> Path:
        """Get uploads directory path."""
        return self.get_path("uploads_dir")
    
    def get_logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.get_path("logs_dir")
    
    def get_simple_retrieval_dir(self) -> Path:
        """Get simple retrieval directory path."""
        return self.get_path("simple_retrieval_dir")
    
    def get_streaming_output_dir(self) -> Path:
        """Get streaming output directory path."""
        return self.get_path("streaming_output_dir")
    
    def get_batch_output_dir(self) -> Path:
        """Get batch output directory path."""
        return self.get_path("batch_output_dir")
    
    def get_granary_file(self, granary_name: str, extension: str = ".parquet") -> Path:
        """Get path for a specific granary file."""
        granaries_dir = self.get_granaries_dir()
        return granaries_dir / f"{granary_name}{extension}"
    
    def get_processed_file(self, granary_name: str) -> Path:
        """Get path for a processed granary file."""
        processed_dir = self.get_processed_dir()
        return processed_dir / f"{granary_name}_processed.parquet"
    
    def get_model_file(self, granary_name: str, compressed: bool = False) -> Path:
        """Get path for a model file."""
        models_dir = self.get_models_dir()
        if compressed:
            return models_dir / f"{granary_name}_forecast_model.joblib.gz"
        else:
            return models_dir / f"{granary_name}_forecast_model.joblib"
    
    def get_forecast_file(self, granary_name: str, timestamp: Optional[str] = None) -> Path:
        """Get path for a forecast file."""
        forecasts_dir = self.get_forecasts_dir()
        if timestamp:
            return forecasts_dir / f"{granary_name}_forecast_{timestamp}.parquet"
        else:
            return forecasts_dir / f"{granary_name}_forecast.parquet"
    
    def ensure_directories(self):
        """Ensure all data directories exist."""
        directories = [
            "granaries_dir",
            "processed_dir", 
            "models_dir",
            "forecasts_dir",
            "temp_dir",
            "uploads_dir",
            "logs_dir"
        ]
        
        for dir_type in directories:
            self.get_path(dir_type)
    
    def list_granaries(self) -> list:
        """List all available granaries."""
        granaries_dir = self.get_granaries_dir()
        granaries = []
        
        for file_path in granaries_dir.glob("*.parquet"):
            granary_name = file_path.stem
            granaries.append(granary_name)
        
        return sorted(granaries)
    
    def list_processed_granaries(self) -> list:
        """List all processed granaries."""
        processed_dir = self.get_processed_dir()
        processed = []
        
        for file_path in processed_dir.glob("*_processed.parquet"):
            granary_name = file_path.stem.replace("_processed", "")
            processed.append(granary_name)
        
        return sorted(processed)
    
    def list_models(self) -> list:
        """List all available models."""
        models_dir = self.get_models_dir()
        models = []
        
        for file_path in models_dir.glob("*_forecast_model.joblib*"):
            granary_name = file_path.stem.replace("_forecast_model", "")
            if granary_name.endswith(".joblib"):
                granary_name = granary_name[:-7]
            models.append(granary_name)
        
        return sorted(models)
    
    def list_forecasts(self) -> list:
        """List all available forecasts."""
        forecasts_dir = self.get_forecasts_dir()
        forecasts = []
        
        for file_path in forecasts_dir.glob("*_forecast*.parquet"):
            forecasts.append(file_path.name)
        
        return sorted(forecasts)
    
    def get_data_summary(self) -> Dict:
        """Get summary of all data files."""
        return {
            "granaries": len(self.list_granaries()),
            "processed": len(self.list_processed_granaries()),
            "models": len(self.list_models()),
            "forecasts": len(self.list_forecasts()),
            "directories": {
                "granaries": str(self.get_granaries_dir()),
                "processed": str(self.get_processed_dir()),
                "models": str(self.get_models_dir()),
                "forecasts": str(self.get_forecasts_dir()),
                "temp": str(self.get_temp_dir()),
                "uploads": str(self.get_uploads_dir()),
                "logs": str(self.get_logs_dir())
            }
        }

# Global instance for easy access
data_paths = DataPathManager() 