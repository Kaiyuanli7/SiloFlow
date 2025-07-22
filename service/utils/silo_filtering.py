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
    
    # Build a set of existing silo identifiers from filenames
    # Expected format: granary_name_silo_name_YYYY-MM-DD_to_YYYY-MM-DD.parquet
    existing_silo_identifiers = set()
    
    for file_path in existing_files:
        filename_stem = file_path.stem.lower()
        
        # Look for the "_to_" pattern which separates start and end dates
        if "_to_" in filename_stem:
            # Split at "_to_" and take the part before it
            before_to = filename_stem.split("_to_")[0]
            # Split by underscore to get parts
            parts = before_to.split('_')
            
            # Look for date pattern (YYYY-MM-DD format) working backwards
            # The date part will be something like "2023-01-17"
            date_start_index = -1
            for i in range(len(parts) - 1, -1, -1):  # Work backwards
                part = parts[i]
                # Check if this part looks like a date (YYYY-MM-DD)
                if '-' in part:
                    date_parts = part.split('-')
                    if (len(date_parts) == 3 and
                        len(date_parts[0]) == 4 and date_parts[0].isdigit() and
                        len(date_parts[1]) == 2 and date_parts[1].isdigit() and
                        len(date_parts[2]) == 2 and date_parts[2].isdigit() and
                        2000 <= int(date_parts[0]) <= 2100 and
                        1 <= int(date_parts[1]) <= 12 and
                        1 <= int(date_parts[2]) <= 31):
                        date_start_index = i
                        break
            
            if date_start_index > 0:
                # Extract the granary_silo part before the date
                granary_silo_parts = parts[:date_start_index]
                granary_silo_identifier = '_'.join(granary_silo_parts)
                existing_silo_identifiers.add(granary_silo_identifier)
                logger.debug(f"Found existing silo: {granary_silo_identifier}")
        else:
            # Fallback: try to find date pattern without "_to_"
            parts = filename_stem.split('_')
            date_start_index = -1
            for i, part in enumerate(parts):
                # Check if this part looks like a date (YYYY-MM-DD)
                if '-' in part:
                    date_parts = part.split('-')
                    if (len(date_parts) == 3 and
                        len(date_parts[0]) == 4 and date_parts[0].isdigit() and
                        len(date_parts[1]) == 2 and date_parts[1].isdigit() and
                        len(date_parts[2]) == 2 and date_parts[2].isdigit() and
                        2000 <= int(date_parts[0]) <= 2100 and
                        1 <= int(date_parts[1]) <= 12 and
                        1 <= int(date_parts[2]) <= 31):
                        date_start_index = i
                        break
            
            if date_start_index > 0:
                granary_silo_parts = parts[:date_start_index]
                granary_silo_identifier = '_'.join(granary_silo_parts)
                existing_silo_identifiers.add(granary_silo_identifier)
                logger.debug(f"Found existing silo (fallback): {granary_silo_identifier}")
    
    filtered_silos = []
    skipped_silos = []
    
    logger.info(f"Checking {len(silos_data)} silos against {len(existing_silo_identifiers)} existing silo identifiers")
    
    for silo in silos_data:
        granary_name = silo.get('granary_name', '').lower()
        silo_name = silo.get('silo_name', '').lower()
        silo_id = silo.get('silo_id', '').lower()
        
        # Generate the expected silo identifier patterns
        # Try both silo_name and silo_id as they might be used interchangeably
        possible_identifiers = [
            f"{granary_name}_{silo_name}",
            f"{granary_name}_{silo_id}",
        ]
        
        # Also try with spaces replaced by underscores
        if ' ' in granary_name or ' ' in silo_name:
            possible_identifiers.extend([
                f"{granary_name.replace(' ', '_')}_{silo_name.replace(' ', '_')}",
                f"{granary_name.replace(' ', '_')}_{silo_id.replace(' ', '_')}"
            ])
        
        silo_exists = False
        matched_identifier = None
        
        for identifier in possible_identifiers:
            if identifier in existing_silo_identifiers:
                silo_exists = True
                matched_identifier = identifier
                break
        
        if silo_exists:
            skipped_silos.append(silo)
            logger.debug(f"Skipping silo {granary_name}_{silo_name} - found existing file with identifier: {matched_identifier}")
        else:
            filtered_silos.append(silo)
    
    logger.info(f"Filtering complete: {len(filtered_silos)} new silos, {len(skipped_silos)} existing silos")
    return filtered_silos, skipped_silos
