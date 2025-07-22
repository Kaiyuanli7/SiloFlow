"""
Utility functions for filtering silos based on existing files in simple_retrieval directory.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)

def get_simple_retrieval_directory() -> Optional[Path]:
    """
    Find the simple_retrieval directory, checking multiple possible locations.
    
    Returns:
        Path to simple_retrieval directory if found, None otherwise
    """
    possible_paths = [
        Path("data/simple_retrieval"),
        Path("service/data/simple_retrieval"), 
        Path("../service/data/simple_retrieval"),
        Path(__file__).parent.parent / "data" / "simple_retrieval"
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            logger.debug(f"Found simple_retrieval directory at: {path}")
            return path
    
    logger.warning("Simple_retrieval directory not found in any expected location")
    return None

def extract_silo_info_from_filename(filename: str, granary_name: str) -> Optional[str]:
    """
    Extract silo ID from a parquet filename.
    
    Expected format: granary_name_silo_id_start-date_to_end-date.parquet
    
    Args:
        filename: Name of the parquet file
        granary_name: Name of the granary to match against
        
    Returns:
        Extracted silo ID or None if extraction fails
    """
    try:
        # Remove the .parquet extension
        filename_stem = Path(filename).stem
        parts = filename_stem.split('_')
        
        if len(parts) < 2:
            return None
        
        # Remove granary name parts from the beginning
        granary_parts = granary_name.split('_')
        
        # Find where the granary name ends in the filename
        remaining_parts = parts[len(granary_parts):]
        
        if not remaining_parts:
            return None
        
        # Find where the date pattern starts (YYYY-MM-DD format)
        silo_parts = []
        for i, part in enumerate(remaining_parts):
            # Check if this part looks like a 4-digit year
            if len(part) == 4 and part.isdigit() and 2000 <= int(part) <= 2100:
                # Check if the pattern continues with MM-DD
                if (i + 2 < len(remaining_parts) and 
                    len(remaining_parts[i + 1]) == 2 and remaining_parts[i + 1].isdigit() and
                    len(remaining_parts[i + 2]) == 2 and remaining_parts[i + 2].isdigit() and
                    1 <= int(remaining_parts[i + 1]) <= 12 and  # Valid month
                    1 <= int(remaining_parts[i + 2]) <= 31):    # Valid day
                    break
            silo_parts.append(part)
        
        if silo_parts:
            return '_'.join(silo_parts)
        
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting silo info from filename {filename}: {e}")
        return None

def get_existing_silo_files(granary_name: str) -> Set[str]:
    """
    Get set of existing silo IDs for a granary from the simple_retrieval directory.
    
    Args:
        granary_name: Name of the granary to check
        
    Returns:
        Set of silo IDs that already have files
    """
    existing_silos = set()
    
    simple_retrieval_dir = get_simple_retrieval_directory()
    if not simple_retrieval_dir:
        return existing_silos
    
    try:
        # Look for parquet files matching this granary
        pattern = f"{granary_name}_*.parquet"
        matching_files = list(simple_retrieval_dir.glob(pattern))
        
        # Also try with spaces replaced by underscores
        if not matching_files:
            pattern_alt = f"{granary_name.replace(' ', '_')}_*.parquet"
            matching_files = list(simple_retrieval_dir.glob(pattern_alt))
        
        logger.debug(f"Found {len(matching_files)} existing files for granary {granary_name}")
        
        for file_path in matching_files:
            silo_id = extract_silo_info_from_filename(file_path.name, granary_name)
            if silo_id:
                existing_silos.add(silo_id)
                logger.debug(f"  Existing silo: {silo_id}")
        
        return existing_silos
        
    except Exception as e:
        logger.error(f"Error checking existing silo files for {granary_name}: {e}")
        return existing_silos

def filter_new_silos(granary_name: str, changed_silos: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter a list of changed silos to only include those that don't already have files.
    
    Args:
        granary_name: Name of the granary
        changed_silos: List of silo IDs that have changed
        
    Returns:
        Tuple of (new_silos, skipped_silos) lists
    """
    existing_silos = get_existing_silo_files(granary_name)
    
    new_silos = []
    skipped_silos = []
    
    for silo in changed_silos:
        # Check if this silo already has a file (exact match or partial match)
        silo_exists = False
        for existing_silo in existing_silos:
            if (silo == existing_silo or 
                existing_silo in silo or 
                silo in existing_silo):
                silo_exists = True
                skipped_silos.append(silo)
                break
        
        if not silo_exists:
            new_silos.append(silo)
    
    logger.info(f"Granary {granary_name}: {len(new_silos)} new silos, {len(skipped_silos)} skipped silos")
    return new_silos, skipped_silos

def filter_silos_by_existing_files(silos_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter a list of silo data dictionaries to exclude silos that already have files.
    
    Args:
        silos_data: List of silo dictionaries with 'granary_name', 'silo_name', etc.
        
    Returns:
        Tuple of (filtered_silos, skipped_silos) lists
    """
    simple_retrieval_dir = get_simple_retrieval_directory()
    if not simple_retrieval_dir:
        logger.warning("Simple_retrieval directory not found - processing all silos")
        return silos_data, []
    
    existing_files = list(simple_retrieval_dir.glob("*.parquet"))
    existing_file_names = {f.stem.lower() for f in existing_files}
    
    filtered_silos = []
    skipped_silos = []
    
    logger.info(f"Checking {len(silos_data)} silos against {len(existing_files)} existing files")
    
    for silo in silos_data:
        granary_name = silo.get('granary_name', '')
        silo_name = silo.get('silo_name', '')
        silo_id = silo.get('silo_id', '')
        
        # Generate possible filename patterns for this silo
        patterns_to_check = [
            f"{granary_name}_{silo_name}",
            f"{granary_name}_{silo_id}",
            f"{granary_name.replace(' ', '_')}_{silo_name.replace(' ', '_')}",
            f"{granary_name.replace(' ', '_')}_{silo_id.replace(' ', '_')}"
        ]
        
        silo_exists = False
        for pattern in patterns_to_check:
            pattern_lower = pattern.lower()
            # Check if any existing file contains this pattern or vice versa
            for existing_name in existing_file_names:
                if (pattern_lower in existing_name or 
                    existing_name in pattern_lower or
                    any(part in existing_name for part in pattern_lower.split('_') if len(part) > 2)):
                    silo_exists = True
                    break
            if silo_exists:
                break
        
        if not silo_exists:
            filtered_silos.append(silo)
        else:
            skipped_silos.append(silo)
    
    logger.info(f"Filtering complete: {len(filtered_silos)} new silos, {len(skipped_silos)} existing silos")
    return filtered_silos, skipped_silos
