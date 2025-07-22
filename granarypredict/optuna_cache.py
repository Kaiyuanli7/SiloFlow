"""
Optuna Parameter Caching System

Automatically saves and loads optimal hyperparameters to avoid redundant optimization.
"""

import json
import pathlib
import hashlib
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


class OptunaParameterCache:
    """Manages caching of Optuna optimization results."""
    
    def __init__(self, cache_dir: str = "optuna_cache"):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _generate_cache_key(self, 
                          csv_filename: str, 
                          df: pd.DataFrame,
                          model_config: Dict[str, Any]) -> str:
        """Generate a unique cache key based on data and configuration."""
        # Create a hash based on:
        # 1. CSV filename
        # 2. Data shape and basic statistics
        # 3. Model configuration
        # 4. Enhanced data fingerprinting
        
        # Create a more comprehensive data fingerprint
        data_fingerprint = {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "column_names": sorted(df.columns.tolist()),
            "numeric_cols_stats": {}
        }
        
        # Add statistics for numeric columns to detect data changes
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # First 5 numeric columns
        for col in numeric_cols:
            if len(df[col]) > 0:
                col_data = df[col]
                has_data = bool(not col_data.isna().all())
                data_fingerprint["numeric_cols_stats"][col] = {
                    "mean": float(col_data.mean()) if has_data else 0.0,
                    "std": float(col_data.std()) if has_data else 0.0,
                    "min": float(col_data.min()) if has_data else 0.0,
                    "max": float(col_data.max()) if has_data else 0.0
                }
        
        key_components = {
            "csv_filename": csv_filename,
            "data_fingerprint": data_fingerprint,
            "model_config": model_config
        }
        
        # Create deterministic hash
        key_string = json.dumps(key_components, sort_keys=True, default=str)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> pathlib.Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"optuna_params_{cache_key}.json"
    
    def save_optimal_params(self, 
                          csv_filename: str,
                          df: pd.DataFrame,
                          model_config: Dict[str, Any],
                          optimal_params: Dict[str, Any],
                          best_value: float,
                          n_trials: int) -> str:
        """Save optimal parameters to cache."""
        cache_key = self._generate_cache_key(csv_filename, df, model_config)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            "csv_filename": csv_filename,
            "optimal_params": optimal_params,
            "best_value": best_value,
            "n_trials": n_trials,
            "model_config": model_config,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict()
            }
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        return cache_key
    
    def load_optimal_params(self, 
                          csv_filename: str,
                          df: pd.DataFrame,
                          model_config: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], float, str]]:
        """Load optimal parameters from cache if available."""
        cache_key = self._generate_cache_key(csv_filename, df, model_config)
        cache_path = self._get_cache_path(cache_key)
        
        print(f"[CACHE-LOADER] Loading cache for: {csv_filename}")
        print(f"[CACHE-LOADER] Generated cache key: {cache_key}")
        print(f"[CACHE-LOADER] Cache path: {cache_path}")
        print(f"[CACHE-LOADER] Cache file exists: {cache_path.exists()}")
        
        # List all available cache files for debugging
        available_files = list(self.cache_dir.glob("optuna_params_*.json"))
        print(f"[CACHE-LOADER] Available cache files: {len(available_files)}")
        for i, file in enumerate(available_files[:5]):  # Show first 5
            print(f"[CACHE-LOADER] Cache file {i+1}: {file.name}")
        
        if not cache_path.exists():
            print(f"[CACHE-LOADER] Cache file not found for key: {cache_key}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Verify cache data is valid
            if (cache_data.get("csv_filename") == csv_filename and
                cache_data.get("data_info", {}).get("shape") == list(df.shape)):
                
                optimal_params = cache_data["optimal_params"]
                best_value = cache_data["best_value"]
                timestamp = cache_data.get("timestamp", "unknown")
                
                return optimal_params, best_value, timestamp
            else:
                # Data mismatch, cache is stale
                return None
                
        except (json.JSONDecodeError, KeyError, TypeError):
            # Cache file is corrupted
            return None
    
    def clear_cache(self, csv_filename: Optional[str] = None):
        """Clear cache files. If csv_filename is provided, only clear for that file."""
        if csv_filename:
            # Clear only files matching this CSV
            pattern = f"*{csv_filename.split('.')[0]}*"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("optuna_params_*.json"):
                cache_file.unlink()
    
    def list_cached_params(self) -> Dict[str, Dict[str, Any]]:
        """List all cached parameter sets."""
        cached_params = {}
        
        for cache_file in self.cache_dir.glob("optuna_params_*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                key = cache_file.stem.replace("optuna_params_", "")
                cached_params[key] = {
                    "csv_filename": cache_data.get("csv_filename"),
                    "best_value": cache_data.get("best_value"),
                    "n_trials": cache_data.get("n_trials"),
                    "timestamp": cache_data.get("timestamp"),
                    "data_shape": cache_data.get("data_info", {}).get("shape")
                }
            except (json.JSONDecodeError, KeyError):
                continue
        
        return cached_params


# Global cache instance
_global_cache = OptunaParameterCache()

def get_cache() -> OptunaParameterCache:
    """Get the global parameter cache instance."""
    return _global_cache

def save_optimal_params(csv_filename: str, df: pd.DataFrame, model_config: Dict[str, Any], 
                       optimal_params: Dict[str, Any], best_value: float, n_trials: int) -> str:
    """Convenience function to save optimal parameters."""
    return _global_cache.save_optimal_params(csv_filename, df, model_config, optimal_params, best_value, n_trials)

def load_optimal_params(csv_filename: str, df: pd.DataFrame, model_config: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], float, str]]:
    """Convenience function to load optimal parameters."""
    return _global_cache.load_optimal_params(csv_filename, df, model_config)

def clear_cache(csv_filename: Optional[str] = None):
    """Convenience function to clear cache."""
    return _global_cache.clear_cache(csv_filename)

def list_cached_params() -> Dict[str, Dict[str, Any]]:
    """Convenience function to list cached parameters."""
    return _global_cache.list_cached_params() 