"""
Compression utilities for model saving and loading.

This module provides functions to save and load models with compression
to reduce storage space while maintaining model performance.
"""

import gzip
import joblib
import logging
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


def save_compressed_model(
    model: Any, 
    path: Union[str, Path], 
    compression_level: int = 6,
    use_gzip: bool = True,
    **joblib_kwargs
) -> Path:
    """
    Save a model with compression to reduce file size.
    
    Parameters:
    -----------
    model : Any
        The model to save (LightGBM, scikit-learn, etc.)
    path : Union[str, Path]
        Path where to save the model
    compression_level : int, default=6
        Compression level (1-9, higher = more compression but slower)
    use_gzip : bool, default=True
        Whether to use gzip compression on top of joblib compression
    **joblib_kwargs
        Additional arguments passed to joblib.dump
        
    Returns:
    --------
    Path
        Path where the model was saved
    """
    path = Path(path)
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add compression to joblib kwargs
    joblib_kwargs.setdefault('compress', ('gzip', compression_level))
    
    if use_gzip:
        # Use gzip compression on top of joblib compression
        with gzip.open(path, 'wb', compresslevel=compression_level) as f:
            joblib.dump(model, f, **joblib_kwargs)
    else:
        # Use only joblib compression
        joblib.dump(model, path, **joblib_kwargs)
    
    # Log file size
    file_size = path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    logger.info(f"Saved compressed model to {path} ({file_size_mb:.2f} MB)")
    
    return path


def load_compressed_model(
    path: Union[str, Path], 
    use_gzip: bool = True,
    **joblib_kwargs
) -> Any:
    """
    Load a compressed model.
    
    Parameters:
    -----------
    path : Union[str, Path]
        Path to the compressed model file
    use_gzip : bool, default=True
        Whether the file was saved with gzip compression
    **joblib_kwargs
        Additional arguments passed to joblib.load
        
    Returns:
    --------
    Any
        The loaded model
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if use_gzip:
        # Load with gzip decompression
        with gzip.open(path, 'rb') as f:
            model = joblib.load(f, **joblib_kwargs)
    else:
        # Load with joblib decompression only
        model = joblib.load(path, **joblib_kwargs)
    
    file_size = path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    logger.info(f"Loaded compressed model from {path} ({file_size_mb:.2f} MB)")
    
    return model


def get_lightgbm_compression_params(
    compression_level: int = 6,
    enable_compression: bool = True
) -> dict:
    """
    Get LightGBM parameters for model compression.
    
    Parameters:
    -----------
    compression_level : int, default=6
        Compression level (1-9, higher = more compression but slower)
    enable_compression : bool, default=True
        Whether to enable LightGBM's built-in compression
        
    Returns:
    --------
    dict
        LightGBM parameters for compression
    """
    if not enable_compression:
        return {}
    
    return {
        'compress': True,
        'compression_level': compression_level,
        'save_binary': True,  # Save in binary format (more compact)
    }


def compress_existing_model(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    compression_level: int = 6,
    use_gzip: bool = True
) -> Path:
    """
    Compress an existing uncompressed model file.
    
    Parameters:
    -----------
    input_path : Union[str, Path]
        Path to the uncompressed model file
    output_path : Optional[Union[str, Path]], default=None
        Path for the compressed model (if None, adds .gz extension)
    compression_level : int, default=6
        Compression level (1-9)
    use_gzip : bool, default=True
        Whether to use gzip compression
        
    Returns:
    --------
    Path
        Path to the compressed model file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        if use_gzip:
            output_path = input_path.with_suffix(input_path.suffix + '.gz')
        else:
            output_path = input_path.with_suffix(input_path.suffix + '.compressed')
    
    output_path = Path(output_path)
    
    # Load the uncompressed model
    logger.info(f"Loading uncompressed model from {input_path}")
    model = joblib.load(input_path)
    
    # Save with compression
    logger.info(f"Compressing model to {output_path}")
    save_compressed_model(
        model, 
        output_path, 
        compression_level=compression_level,
        use_gzip=use_gzip
    )
    
    # Compare file sizes
    original_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    logger.info(f"Compression complete:")
    logger.info(f"  Original: {original_size / (1024*1024):.2f} MB")
    logger.info(f"  Compressed: {compressed_size / (1024*1024):.2f} MB")
    logger.info(f"  Compression ratio: {compression_ratio:.1f}%")
    
    return output_path


def batch_compress_models(
    models_dir: Union[str, Path],
    pattern: str = "*.joblib",
    compression_level: int = 6,
    use_gzip: bool = True,
    backup_original: bool = True
) -> list[Path]:
    """
    Compress all model files in a directory.
    
    Parameters:
    -----------
    models_dir : Union[str, Path]
        Directory containing model files
    pattern : str, default="*.joblib"
        File pattern to match
    compression_level : int, default=6
        Compression level (1-9)
    use_gzip : bool, default=True
        Whether to use gzip compression
    backup_original : bool, default=True
        Whether to keep original files as backup
        
    Returns:
    --------
    list[Path]
        List of compressed model file paths
    """
    models_dir = Path(models_dir)
    compressed_files = []
    
    for model_file in models_dir.glob(pattern):
        if model_file.suffix == '.gz' or '.compressed' in model_file.name:
            logger.info(f"Skipping already compressed file: {model_file}")
            continue
            
        try:
            compressed_path = compress_existing_model(
                model_file,
                compression_level=compression_level,
                use_gzip=use_gzip
            )
            compressed_files.append(compressed_path)
            
            # Optionally remove original file
            if not backup_original:
                model_file.unlink()
                logger.info(f"Removed original file: {model_file}")
                
        except Exception as e:
            logger.error(f"Failed to compress {model_file}: {e}")
    
    logger.info(f"Compressed {len(compressed_files)} model files")
    return compressed_files 