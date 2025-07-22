import logging
from typing import Union, Tuple, Dict

logger = logging.getLogger(__name__)

def validate_config(config: dict, config_type: str) -> bool:
    """Validate configuration against schema"""
    schemas = {
        "streaming": {
            "database": {
                "host": str, "port": int, "database": str, 
                "user": str, "password": str
            },
            "processing": {
                "initial_chunk_size": int, "min_chunk_size": int, 
                "max_chunk_size": int, "memory_threshold_percent": (int, float),
                "max_records_per_batch": int, "memory_safety_threshold_percent": (int, float),
                "batch_timeout_seconds": int, "force_gc_threshold_percent": (int, float),
                "pause_duration_seconds": int, "max_retries_per_chunk": int
            }
        },
        "client": {
            "server": str, "port": int, "timeout": int, "file": str
        }
    }
    
    if config_type not in schemas:
        raise ValueError(f"Unknown config type: {config_type}")
    
    schema = schemas[config_type]
    errors = []
    
    for section, fields in schema.items():
        if section not in config:
            errors.append(f"Missing section: {section}")
            continue
            
        for field, field_type in fields.items():
            if field not in config[section]:
                errors.append(f"Missing field: {section}.{field}")
            elif not isinstance(config[section][field], field_type):
                if not (isinstance(field_type, tuple) and 
                        isinstance(config[section][field], field_type)):
                    errors.append(
                        f"Invalid type for {section}.{field}: "
                        f"Expected {field_type}, got {type(config[section][field])}"
                    )
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True