"""
Compression configuration for granarypredict models.

This module defines the compression strategies and thresholds for different
model types and sizes to optimize storage efficiency while maintaining
loading reliability and performance.
"""

from typing import Dict, Tuple, Optional, List

# Model size thresholds in MB
MODEL_SIZE_THRESHOLDS = {
    'small': 50,     # Models under 50MB - fast compression for quick loading
    'medium': 200,   # Models 50-200MB - balanced compression
    'large': 500,    # Models 200-500MB - higher compression
    'huge': 1000,    # Models over 1GB - maximum compression
}

# Compression strategies by model size category
COMPRESSION_STRATEGIES = {
    'small_models': {
        'algorithm': 'lz4',
        'level': 3,
        'description': 'Fast compression for quick loading',
        'target_ratio': 2.0
    },
    'medium_models': {
        'algorithm': 'gzip', 
        'level': 6,
        'description': 'Balanced compression and speed',
        'target_ratio': 3.0
    },
    'large_models': {
        'algorithm': 'lzma',
        'level': 6,
        'description': 'Higher compression for storage efficiency',
        'target_ratio': 4.0
    },
    'huge_models': {
        'algorithm': 'lzma',
        'level': 9,
        'description': 'Maximum compression for very large models',
        'target_ratio': 5.0
    },
    'uncertainty_models': {
        'algorithm': 'lzma',
        'level': 9,
        'description': 'Maximum compression for bootstrap/uncertainty models',
        'target_ratio': 6.0
    }
}

# Model-type specific overrides
MODEL_TYPE_OVERRIDES = {
    'LGBMRegressor': {
        'small': ('gzip', 4),   # LightGBM compresses well with gzip
        'medium': ('gzip', 6),
        'large': ('lzma', 6),
        'huge': ('lzma', 9)
    },
    'MultiLGBMRegressor': {
        'small': ('gzip', 6),   # Multi-horizon models need better compression
        'medium': ('lzma', 6),
        'large': ('lzma', 8),
        'huge': ('lzma', 9)
    },
    'RandomForestRegressor': {
        'small': ('lz4', 3),    # Random forests are less compressible
        'medium': ('gzip', 4),
        'large': ('gzip', 6),
        'huge': ('lzma', 6)
    }
}

# Special model patterns that require specific handling
SPECIAL_PATTERNS = {
    'bootstrap': 'uncertainty_models',    # Bootstrap models
    'uncertainty': 'uncertainty_models',  # Uncertainty estimation models
    'ensemble': 'large_models',          # Ensemble models
    'multi_output': 'large_models',      # Multi-output models
    'multi_horizon': 'medium_models'     # Multi-horizon forecasting models
}

def get_compression_config(
    model_type: Optional[str] = None,
    model_size_mb: Optional[float] = None,
    model_attributes: Optional[List[str]] = None
) -> Tuple[str, int]:
    """
    Get optimal compression configuration for a model.
    
    Parameters:
    -----------
    model_type : str, optional
        Type of the model (e.g., 'LGBMRegressor')
    model_size_mb : float, optional
        Model size in megabytes
    model_attributes : list, optional
        List of model attributes or patterns
        
    Returns:
    --------
    Tuple[str, int]
        Compression algorithm and level
    """
    # Check for special patterns first
    if model_attributes:
        for attr in model_attributes:
            attr_lower = str(attr).lower()
            for pattern, strategy in SPECIAL_PATTERNS.items():
                if pattern in attr_lower:
                    config = COMPRESSION_STRATEGIES[strategy]
                    return config['algorithm'], config['level']
    
    # Check model type overrides
    if model_type and model_type in MODEL_TYPE_OVERRIDES:
        size_category = determine_size_category(model_size_mb)
        if size_category in MODEL_TYPE_OVERRIDES[model_type]:
            return MODEL_TYPE_OVERRIDES[model_type][size_category]
    
    # Use default size-based strategy
    size_category = determine_size_category(model_size_mb)
    strategy_key = f'{size_category}_models'
    
    if strategy_key in COMPRESSION_STRATEGIES:
        config = COMPRESSION_STRATEGIES[strategy_key]
        return config['algorithm'], config['level']
    
    # Fallback to medium compression
    config = COMPRESSION_STRATEGIES['medium_models']
    return config['algorithm'], config['level']

def determine_size_category(size_mb: Optional[float]) -> str:
    """Determine size category based on model size in MB."""
    if size_mb is None:
        return 'medium'  # Safe default
    
    if size_mb < MODEL_SIZE_THRESHOLDS['small']:
        return 'small'
    elif size_mb < MODEL_SIZE_THRESHOLDS['medium']:
        return 'medium'
    elif size_mb < MODEL_SIZE_THRESHOLDS['large']:
        return 'large'
    else:
        return 'huge'

def get_target_compression_ratio(
    model_type: Optional[str] = None,
    model_size_mb: Optional[float] = None,
    model_attributes: Optional[List[str]] = None
) -> float:
    """Get target compression ratio for a model configuration."""
    algorithm, level = get_compression_config(model_type, model_size_mb, model_attributes)
    
    # Find the strategy that matches this algorithm/level
    for strategy in COMPRESSION_STRATEGIES.values():
        if strategy['algorithm'] == algorithm and strategy['level'] == level:
            return strategy['target_ratio']
    
    return 3.0  # Default target ratio

# Export configuration for external use
__all__ = [
    'MODEL_SIZE_THRESHOLDS',
    'COMPRESSION_STRATEGIES', 
    'MODEL_TYPE_OVERRIDES',
    'SPECIAL_PATTERNS',
    'get_compression_config',
    'determine_size_category',
    'get_target_compression_ratio'
]
