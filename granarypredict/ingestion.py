
from __future__ import annotations
import difflib
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import requests

from .config import RAW_DATA_DIR, METEOROLOGY_API_BASE, COMPANY_API_BASE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ingest_and_sort_dataframe(df: pd.DataFrame, return_new_data_status=False, output_dir: Optional[Union[str, Path]] = None):
    """
    Ingests a DataFrame directly, splits by granary, deduplicates, and writes to data/granaries/<granary>.parquet.
    This is an optimized version that skips the temporary CSV file step.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data DataFrame from database
    return_new_data_status : bool, default False
        Whether to return detailed status information
    output_dir : str or Path, optional
        Output directory for granary files (defaults to 'data/granaries')
    
    Returns:
    --------
    dict or list
        If return_new_data_status=True: dict with granary_status and silo_changes
        Otherwise: list of granary names
    """
    import pandas as pd
    from pathlib import Path
    import os
    
    # Use the existing standardization function to handle various column formats
    df = standardize_granary_csv(df)
    
    # Now we should have a standardized granary_id column
    if 'granary_id' not in df.columns:
        # Fallback: try to identify granary column from common names
        possible_granary_cols = ['storepointName', 'storeName', '仓库名称', 'granary_id', 'Granary_ID']
        granary_col = None
        for col in possible_granary_cols:
            if col in df.columns:
                granary_col = col
                break
        
        if granary_col is None:
            # If no granary column found, create a default one
            df['granary_id'] = 'unknown_granary'
        else:
            df['granary_id'] = df[granary_col]
    
    granary_names = df['granary_id'].unique().tolist()
    
    granary_status = {}
    silo_changes = {}  # Track which silos changed for each granary
    
    # Set output directory
    if output_dir is None:
        out_dir = Path('data/granaries')
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for granary in granary_names:
        granary_df = df[df['granary_id'] == granary].copy()
        out_path = out_dir / f"{granary}.parquet"
        
        if out_path.exists():
            # Read existing Parquet file
            old_df = read_granary_csv(out_path)
            
            # Track silo-level changes
            changed_silos = []
            for silo in granary_df['heap_id'].unique():
                old_silo_data = old_df[old_df['heap_id'] == silo]
                new_silo_data = granary_df[granary_df['heap_id'] == silo]
                
                # Check if this silo has new data
                combined_silo = pd.concat([old_silo_data, new_silo_data], ignore_index=True)
                deduped_silo = combined_silo.drop_duplicates()
                
                if len(deduped_silo) > len(old_silo_data):
                    changed_silos.append(silo)
            
            # Combine all data for the granary
            combined = pd.concat([old_df, granary_df], ignore_index=True)
            deduped = combined.drop_duplicates()
            
            # Save as Parquet with compression
            save_granary_data(deduped, out_path, format='parquet', compression='snappy')
            
            # Granary has new data if any silo changed
            granary_status[granary] = len(changed_silos) > 0
            silo_changes[granary] = changed_silos
            
        else:
            # New granary - all silos are "new"
            save_granary_data(granary_df, out_path, format='parquet', compression='snappy')
            granary_status[granary] = True
            silo_changes[granary] = list(granary_df['heap_id'].unique())
    
    if return_new_data_status:
        return {
            'granary_status': granary_status,
            'silo_changes': silo_changes
        }
    else:
        return list(granary_status.keys())


def ingest_and_sort(csv_path, return_new_data_status=False):
    """
    Ingests the raw data file (CSV or Parquet), splits by granary, deduplicates, and writes to data/granaries/<granary>.parquet.
    If return_new_data_status is True, returns a dict with both granary and silo change information.
    Otherwise, returns a list of granary names.
    """
    from .ingestion import read_granary_csv
    import pandas as pd
    from pathlib import Path
    import os
    
    # Use automatic format detection (CSV or Parquet)
    df = read_granary_csv(csv_path)
    
    # Use the existing standardization function to handle various column formats
    df = standardize_granary_csv(df)
    
    # Now we should have a standardized granary_id column
    if 'granary_id' not in df.columns:
        # Fallback: try to identify granary column from common names
        possible_granary_cols = ['storepointName', 'storeName', '仓库名称', 'granary_id', 'Granary_ID']
        granary_col = None
        for col in possible_granary_cols:
            if col in df.columns:
                granary_col = col
                break
        
        if granary_col is None:
            # If no granary column found, create a default one
            df['granary_id'] = 'unknown_granary'
        else:
            df['granary_id'] = df[granary_col]
    
    granary_names = df['granary_id'].unique().tolist()
    
    granary_status = {}
    silo_changes = {}  # Track which silos changed for each granary
    
    out_dir = Path('data/granaries')
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for granary in granary_names:
        granary_df = df[df['granary_id'] == granary].copy()
        out_path = out_dir / f"{granary}.parquet"
        
        if out_path.exists():
            # Read existing Parquet file
            old_df = read_granary_csv(out_path)
            
            # Track silo-level changes
            changed_silos = []
            for silo in granary_df['heap_id'].unique():
                old_silo_data = old_df[old_df['heap_id'] == silo]
                new_silo_data = granary_df[granary_df['heap_id'] == silo]
                
                # Check if this silo has new data
                combined_silo = pd.concat([old_silo_data, new_silo_data], ignore_index=True)
                deduped_silo = combined_silo.drop_duplicates()
                
                if len(deduped_silo) > len(old_silo_data):
                    changed_silos.append(silo)
            
            # Combine all data for the granary
            combined = pd.concat([old_df, granary_df], ignore_index=True)
            deduped = combined.drop_duplicates()
            
            # Save as Parquet with compression
            save_granary_data(deduped, out_path, format='parquet', compression='snappy')
            
            # Granary has new data if any silo changed
            granary_status[granary] = len(changed_silos) > 0
            silo_changes[granary] = changed_silos
            
        else:
            # New granary - all silos are "new"
            save_granary_data(granary_df, out_path, format='parquet', compression='snappy')
            granary_status[granary] = True
            silo_changes[granary] = list(granary_df['heap_id'].unique())
    
    if return_new_data_status:
        return {
            'granary_status': granary_status,
            'silo_changes': silo_changes
        }
    else:
        return list(granary_status.keys())


def read_granary_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Read granary data file with automatic format detection.
    
    This function automatically detects and handles:
    - Compressed CSV files (.gz, .bz2, .zip, .xz)
    - Regular CSV files (.csv)
    - Parquet files (.parquet)
    
    Parameters:
    -----------
    filepath : str or pathlib.Path
        Path to data file (CSV or Parquet)
    
    Returns:
    --------
    pd.DataFrame
        Loaded granary data
    """
    import pathlib
    from .data_utils import read_compressed_csv, read_parquet
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Granary data file not found: {filepath}")
    
    # Check file format based on extension
    if filepath.suffix == '.parquet':
        logger.info(f"Reading Parquet file: {filepath}")
        df = read_parquet(filepath)
    elif filepath.suffix.endswith(('.gz', '.gzip', '.bz2', '.zip', '.xz')):
        logger.info(f"Reading compressed CSV: {filepath}")
        df = read_compressed_csv(filepath)
    elif filepath.suffix == '.csv':
        logger.info(f"Reading regular CSV: {filepath}")
        df = pd.read_csv(filepath)
    else:
        # Try to read as CSV if no recognized extension
        logger.info(f"Attempting to read as CSV: {filepath}")
        df = pd.read_csv(filepath)
    
    logger.info(f"Loaded granary data: {len(df)} rows, {len(df.columns)} columns")
    return df

def save_granary_data(
    df: pd.DataFrame, 
    filepath: Union[str, Path], 
    format: str = 'parquet',
    compression: str = 'snappy',
    **kwargs
) -> Path:
    """
    Save granary data in the specified format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or pathlib.Path
        Base filepath (extension will be added automatically)
    format : str, default 'parquet'
        Output format: 'parquet' or 'csv'
    compression : str, default 'snappy'
        Compression method for Parquet: 'snappy', 'gzip', 'brotli'
        For CSV: 'gzip', 'bz2', 'zip', 'xz'
    **kwargs :
        Additional arguments passed to save functions
    
    Returns:
    --------
    pathlib.Path
        Path to the saved file
    """
    from .data_utils import save_parquet, save_compressed_csv
    
    filepath = Path(filepath)
    
    if format.lower() == 'parquet':
        logger.info(f"Saving as Parquet: {filepath}")
        return save_parquet(df, filepath, compression=compression, **kwargs)
    elif format.lower() == 'csv':
        logger.info(f"Saving as compressed CSV: {filepath}")
        return save_compressed_csv(df, filepath, compression=compression, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'")

def convert_csv_to_parquet(
    csv_filepath: Union[str, Path],
    parquet_filepath: Union[str, Path] = None,
    compression: str = 'snappy',
    delete_original: bool = False
) -> Path:
    """
    Convert existing CSV file to Parquet format.
    
    Parameters:
    -----------
    csv_filepath : str or pathlib.Path
        Path to existing CSV file
    parquet_filepath : str or pathlib.Path, optional
        Path for output Parquet file (auto-generated if None)
    compression : str, default 'snappy'
        Parquet compression method
    delete_original : bool, default False
        Whether to delete the original CSV file after conversion
    
    Returns:
    --------
    pathlib.Path
        Path to the created Parquet file
    """
    csv_filepath = Path(csv_filepath)
    
    if not csv_filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
    
    # Generate Parquet filepath if not provided
    if parquet_filepath is None:
        parquet_filepath = csv_filepath.with_suffix('.parquet')
    else:
        parquet_filepath = Path(parquet_filepath)
    
    logger.info(f"Converting CSV to Parquet: {csv_filepath} → {parquet_filepath}")
    
    # Read CSV (handles compressed formats automatically)
    df = read_granary_csv(csv_filepath)
    
    # Save as Parquet
    from .data_utils import save_parquet
    saved_path = save_parquet(df, parquet_filepath, compression=compression)
    
    # Delete original if requested
    if delete_original:
        csv_filepath.unlink()
        logger.info(f"Deleted original CSV file: {csv_filepath}")
    
    return saved_path




def fetch_company_data(
    endpoint: Literal[
        "granaries",
        "heaps",
        "sensors",
        "operations",
    ],
    params: Optional[dict[str, str]] = None,
    *,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """Generic GET request to company's data service."""
    url = f"{COMPANY_API_BASE}/{endpoint}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception as exc:
        logger.warning("Company API unavailable (%s), returning empty frame.", exc)
        return pd.DataFrame()


def standardize_result147(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Result_147.csv format to standard columns used in pipeline."""
    # Column mapping for multiple export formats (Result_147 and the new StorePoint format)
    mapping = {
        # ----- core timestamp / spatial / target -----
        "batch": "detection_time",        # timestamp
        "temp": "temperature_grain",     # target variable
        "x": "grid_x",
        "y": "grid_y",
        "z": "grid_z",

        # ----- environmental variables -----
        "indoor_temp": "temperature_inside",
        "outdoor_temp": "temperature_outside",
        "indoor_humidity": "humidity_warehouse",
        "outdoor_humidity": "humidity_outside",

        # ----- warehouse / grain metadata -----
        "storeType": "warehouse_type",

        # ----- new StorePoint header additions -----
        "storepoint_id": "storepoint_id",
        "storepointName": "granary_id",
        "storeName": "heap_id",
        "kqdz": "address_cn",           # 库区地址 (Chinese)
        "storeId": "warehouse_id",
        "locatType": "location_type",
        "line_no": "line_no",
        "layer_no": "layer_no",
        "avg_in_temp": "avg_in_temp",
        "max_temp": "max_temp",
        "min_temp": "min_temp",
    }
    df = df.rename(columns=mapping)
    # ensure correct dtypes
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    nums = [
        "temperature_grain",
        "temperature_inside",
        "temperature_outside",
        "humidity_warehouse",
        "humidity_outside",
        "avg_in_temp",
        "max_temp",
        "min_temp",
    ]
    for col in nums:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


__all__ = [
    "read_granary_csv",
    "fetch_meteorology",
    "fetch_company_data",
    "standardize_result147",
    "standardize_granary_csv",
]


# ------------------------------------------------------------
# NEW – Canonical schema standardisation helper  (May-2025)
# ------------------------------------------------------------

# Desired external CSV header (user-facing):
#   Granary_ID,longitude,latitude,address,Heap_ID,batch,temp,x,y,z,
#   avg_grain_temp,max_temp,min_temp,indoor_temp,indoor_humidity,
#   outdoor_temp,outdoor_humidity,storeType

# Internal pipeline keeps its original snake-case column names so that the
# rest of the codebase (features/, dashboard, etc.) remains unchanged.

_CANONICAL_MAP: dict[str, str] = {
    # ----- ID / location -----
    "Granary_ID": "granary_id",
    "Heap_ID": "heap_id",
    "address": "address_cn",  # keep same field semantics – Chinese address OK

    # ----- timestamp & spatial coords -----
    "batch": "detection_time",
    "x": "grid_x",
    "y": "grid_y",
    "z": "grid_z",

    # ----- sensor & temperature measurements -----
    "temp": "temperature_grain",      # individual probe reading
    "avg_grain_temp": "avg_grain_temp",  # pile-wide daily average
    "max_temp": "max_temp",
    "min_temp": "min_temp",

    # ----- environmental -----
    "indoor_temp": "temperature_inside",
    "indoor_humidity": "humidity_warehouse",
    "outdoor_temp": "temperature_outside",
    "outdoor_humidity": "humidity_outside",

    # ----- misc -----
    "storeType": "warehouse_type",

    # ----- Backward-compatibility aliases (old StorePoint / Result_147) -----
    "storepointName": "granary_id",
    "storeName": "heap_id",
    "kqdz": "address_cn",
    "avg_in_temp": "avg_grain_temp",  # old name for average grain temperature
}


def _rename_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns via ``_CANONICAL_MAP`` and keep only recognised ones."""
    # First, explicitly drop longitude/latitude columns that might appear in original CSVs
    longitude_latitude_cols = ["kdwd", "kdjd", "longitude", "latitude"]
    df = df.drop(columns=[col for col in longitude_latitude_cols if col in df.columns])
    
    # Flexible column mapping: case-insensitive, fuzzy, and Chinese support
    rename_map = {}
    for c in df.columns:
        # Try exact case-insensitive match
        for k in _CANONICAL_MAP:
            if c.lower() == k.lower():
                rename_map[c] = _CANONICAL_MAP[k]
                break
        else:
            # Try fuzzy match (threshold 0.8)
            matches = difflib.get_close_matches(c, _CANONICAL_MAP.keys(), n=1, cutoff=0.8)
            if matches:
                rename_map[c] = _CANONICAL_MAP[matches[0]]
    df = df.rename(columns=rename_map)
    keep_cols = set(_CANONICAL_MAP.values())
    # Log columns that will be kept/dropped
    logger.debug("Columns kept: %s", [c for c in df.columns if c in keep_cols])
    logger.debug("Columns dropped: %s", [c for c in df.columns if c not in keep_cols])
    # Keep all canonical columns, but also retain extra columns for debugging
    selected = [c for c in df.columns if c in keep_cols]
    extra = [c for c in df.columns if c not in keep_cols]
    logger.debug("Extra columns retained for debugging: %s", extra)
    # Return canonical columns plus extras
    return df[selected + extra].copy()


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast datatypes for numeric columns and parse datetimes."""
    # Parse timestamp
    if "detection_time" in df.columns:
        df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")

    # Numeric coercion list – everything except IDs / address / types
    numeric_like = [
        "grid_x",
        "grid_y",
        "grid_z",
        "temperature_grain",
        "avg_grain_temp",
        "max_temp",
        "min_temp",
        "temperature_inside",
        "temperature_outside",
        "humidity_warehouse",
        "humidity_outside",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_granary_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any recognised CSV variant to the canonical internal schema.

    Steps:
    1. Rename columns via ``_CANONICAL_MAP`` (case-sensitive).
    2. Drop columns that are not part of the canonical internal schema.
    3. Parse *detection_time* to datetime and coerce numeric columns.
    """

    logger.info("Standardising CSV to canonical schema. Incoming cols=%s", list(df.columns))
    df = _rename_and_select(df)
    # Try to infer granary_id if missing
    if "granary_id" not in df.columns or df["granary_id"].isnull().all():
        # Try to infer from other columns (e.g., address_cn, storepointName, etc.)
        possible_cols = [c for c in df.columns if "granary" in c.lower() or "库" in c or "仓" in c]
        if possible_cols:
            df["granary_id"] = df[possible_cols[0]]
            logger.info(f"Inferred granary_id from column: {possible_cols[0]}")
        else:
            df["granary_id"] = "unknown_granary"
            logger.warning("Could not infer granary_id, using 'unknown_granary'")
    df = _coerce_types(df)
    logger.info("After standardisation cols=%s", list(df.columns))
    return df