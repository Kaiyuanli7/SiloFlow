"""
Optimized compression utilities for model saving and loading.

This module provides adaptive compression strategies based on model size and type
to achieve optimal balance between file size and loading reliability.
"""

import gzip
import joblib
import logging
from pathlib import Path
from typing import Any, Optional, Union, Dict, Tuple
import numpy as np

from .compression_config import get_compression_config

logger = logging.getLogger(__name__)

def get_model_size_category(model_path_or_size: Union[str, Path, int]) -> str:
    """
    Determine model size category for compression strategy selection.
    
    Parameters:
    -----------
    model_path_or_size : Union[str, Path, int]
        Either path to model file or size in bytes
        
    Returns:
    --------
    str
        Size category: 'small', 'medium', 'large', 'huge'
    """
    from .compression_config import determine_size_category
    
    if isinstance(model_path_or_size, (str, Path)):
        path = Path(model_path_or_size)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
        else:
            return 'medium'  # Default fallback
    else:
        size_mb = model_path_or_size / (1024 * 1024)
    
    return determine_size_category(size_mb)

def get_optimal_compression_params(model: Any = None, size_category: Optional[str] = None) -> Tuple[str, int]:
    """
    Get optimal compression parameters based on model type and size.
    
    Parameters:
    -----------
    model : Any, optional
        The model object to analyze
    size_category : str, optional
        Pre-determined size category
        
    Returns:
    --------
    Tuple[str, int]
        Compression algorithm and level
    """
    # Analyze model characteristics
    model_type = None
    model_attributes = []
    
    if model is not None:
        model_type = type(model).__name__
        
        # Check for special model characteristics
        if hasattr(model, 'bootstrap_models_'):
            model_attributes.append('bootstrap')
        if hasattr(model, 'uncertainty_estimation'):
            model_attributes.append('uncertainty')
        if hasattr(model, 'estimators_'):
            n_estimators = len(getattr(model, 'estimators_', []))
            if n_estimators > 5:
                model_attributes.append('ensemble')
            if n_estimators > 1:
                model_attributes.append('multi_output')
        if hasattr(model, 'n_outputs_') and getattr(model, 'n_outputs_', 1) > 1:
            model_attributes.append('multi_horizon')
    
    # Use the new configuration system
    return get_compression_config(
        model_type=model_type,
        model_size_mb=None,  # Will be determined later if needed
        model_attributes=model_attributes
    )


def save_compressed_model(
    model: Any, 
    path: Union[str, Path], 
    compression_config: Optional[Dict] = None,
    **joblib_kwargs
) -> Dict:
    """
    Save a model with adaptive compression based on size and type.
    
    Parameters:
    -----------
    model : Any
        The model to save (LightGBM, scikit-learn, etc.)
    path : Union[str, Path]
        Path where to save the model
    compression_config : Optional[Dict]
        Custom compression configuration (overrides adaptive selection)
    **joblib_kwargs
        Additional arguments passed to joblib.dump
        
    Returns:
    --------
    Dict
        Compression statistics and metadata
    """
    import time
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Get optimal compression parameters
    if compression_config:
        compression_alg, compression_level = compression_config.get('algorithm', 'gzip'), compression_config.get('level', 6)
    else:
        compression_alg, compression_level = get_optimal_compression_params(model)
    
    # First get uncompressed size estimate
    temp_path = path.with_suffix('.temp')
    try:
        joblib.dump(model, temp_path, compress=0)
        uncompressed_size = temp_path.stat().st_size
        temp_path.unlink()  # Clean up
        
        # Refine compression based on actual size
        size_category = get_model_size_category(uncompressed_size)
        final_compression_alg, final_compression_level = get_optimal_compression_params(model, size_category)
        
    except Exception:
        # Fallback if temp file creation fails
        uncompressed_size = 0
        final_compression_alg, final_compression_level = compression_alg, compression_level
    
    # Save with optimal compression
    joblib_kwargs.setdefault('compress', (final_compression_alg, final_compression_level))
    
    if final_compression_alg == 'gzip':
        # Use double compression for gzip (gzip + joblib)
        with gzip.open(f"{path}.gz", 'wb', compresslevel=final_compression_level) as f:
            joblib.dump(model, f, compress=0, **joblib_kwargs)
        final_path = Path(f"{path}.gz")
    else:
        # Use joblib's built-in compression
        joblib.dump(model, path, **joblib_kwargs)
        final_path = path
    
    # Calculate compression stats
    compressed_size = final_path.stat().st_size
    compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 and uncompressed_size > 0 else 1.0
    save_time = time.time() - start_time
    
    stats = {
        'path': str(final_path),
        'compression_algorithm': final_compression_alg,
        'compression_level': final_compression_level,
        'size_category': get_model_size_category(uncompressed_size) if uncompressed_size > 0 else 'unknown',
        'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
        'compressed_size_mb': compressed_size / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'space_saved_mb': (uncompressed_size - compressed_size) / (1024 * 1024) if uncompressed_size > 0 else 0,
        'save_time_seconds': save_time
    }
    
    logger.info(f"Model saved with {final_compression_alg} compression:")
    logger.info(f"  Size: {stats['compressed_size_mb']:.2f} MB "
               f"(was {stats['uncompressed_size_mb']:.2f} MB)" if uncompressed_size > 0 else f"  Size: {stats['compressed_size_mb']:.2f} MB")
    logger.info(f"  Compression ratio: {compression_ratio:.2f}x" if compression_ratio > 1 else "  No compression ratio available")
    logger.info(f"  Save time: {save_time:.2f} seconds")
    
    return stats


def load_compressed_model(
    path: Union[str, Path], 
    **joblib_kwargs
) -> Any:
    """
    Load a compressed model with adaptive decompression and comprehensive fallbacks.
    
    Parameters:
    -----------
    path : Union[str, Path]
        Path to the compressed model file
    **joblib_kwargs
        Additional arguments passed to joblib.load
        
    Returns:
    --------
    Any
        The loaded model
    """
    path = Path(path)
    
    if not path.exists():
        # Try with .gz extension if original doesn't exist
        gz_path = Path(f"{path}.gz")
        if gz_path.exists():
            path = gz_path
        else:
            raise FileNotFoundError(f"Model file not found: {path} or {gz_path}")
    
    # Try loading with different strategies in order of likelihood
    errors = []
    file_size_mb = path.stat().st_size / (1024 * 1024)
    
    # Strategy 1: Gzip + joblib (most common for our new compression)
    if str(path).endswith('.gz'):
        try:
            with gzip.open(path, 'rb') as f:
                model = joblib.load(f, **joblib_kwargs)
            logger.info(f"Loaded gzip-compressed model from {path} ({file_size_mb:.2f} MB)")
            return model
        except Exception as e:
            errors.append(f"Gzip decompression failed: {str(e)}")
    
    # Strategy 2: Pure joblib with built-in compression
    try:
        model = joblib.load(path, **joblib_kwargs)
        logger.info(f"Loaded joblib-compressed model from {path} ({file_size_mb:.2f} MB)")
        return model
    except Exception as e:
        errors.append(f"Joblib with compression failed: {str(e)}")
    
    # Strategy 3: Force no memory mapping (compatibility fallback)
    try:
        joblib_kwargs_no_mmap = joblib_kwargs.copy()
        joblib_kwargs_no_mmap['mmap_mode'] = None
        model = joblib.load(path, **joblib_kwargs_no_mmap)
        logger.info(f"Loaded model from {path} ({file_size_mb:.2f} MB) with mmap disabled")
        return model
    except Exception as e:
        errors.append(f"No-mmap fallback failed: {str(e)}")
    
    # Strategy 4: Try as gzip even without .gz extension (legacy compatibility)
    if not str(path).endswith('.gz'):
        try:
            with gzip.open(path, 'rb') as f:
                model = joblib.load(f, mmap_mode=None)
            logger.info(f"Loaded legacy gzip model from {path} ({file_size_mb:.2f} MB)")
            return model
        except Exception as e:
            errors.append(f"Legacy gzip fallback failed: {str(e)}")
    
    # All strategies failed
    logger.error(f"All loading strategies failed for {path} ({file_size_mb:.2f} MB)")
    error_msg = f"Failed to load compressed model from {path}. Tried {len(errors)} strategies. Errors: {'; '.join(errors)}"
    raise RuntimeError(error_msg)


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