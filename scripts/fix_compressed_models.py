#!/usr/bin/env python3
"""
Model Compression Repair Tool

This script helps fix model loading issues caused by compression problems.
It can:
1. List all model files and their status
2. Test model loading with different strategies
3. Repair compressed models by re-saving them
4. Convert between compressed/uncompressed formats

Usage:
    python scripts/fix_compressed_models.py --list          # List all models
    python scripts/fix_compressed_models.py --test MODEL    # Test loading a specific model
    python scripts/fix_compressed_models.py --repair MODEL  # Repair a specific model
    python scripts/fix_compressed_models.py --repair-all    # Repair all models
"""

import argparse
import logging
from pathlib import Path
import sys
import traceback

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
from granarypredict.model import load_model, save_model
from granarypredict.compression_utils import load_compressed_model, save_compressed_model
from granarypredict.config import MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_files():
    """Get all model files from the models directory."""
    model_files = []
    
    # Common model file patterns
    patterns = ["*.joblib", "*.pkl", "*.pickle", "*.gz", "*.compressed"]
    
    for pattern in patterns:
        model_files.extend(MODELS_DIR.glob(pattern))
    
    # Also check preloaded models
    preloaded_dir = MODELS_DIR / "preloaded"
    if preloaded_dir.exists():
        for pattern in patterns:
            model_files.extend(preloaded_dir.glob(pattern))
    
    return sorted(set(model_files))


def test_model_loading(model_path: Path):
    """Test different loading strategies for a model file."""
    print(f"\n🔍 Testing model: {model_path}")
    print(f"   File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Strategy 1: Our robust loader
    try:
        model = load_model(model_path)
        print("   ✅ SUCCESS: Robust loader worked")
        return model, "robust"
    except Exception as e:
        print(f"   ❌ FAILED: Robust loader - {str(e)}")
    
    # Strategy 2: Standard joblib
    try:
        model = joblib.load(model_path)
        print("   ✅ SUCCESS: Standard joblib worked")
        return model, "joblib"
    except Exception as e:
        print(f"   ❌ FAILED: Standard joblib - {str(e)}")
    
    # Strategy 3: Compressed loader with gzip
    try:
        model = load_compressed_model(model_path, use_gzip=True)
        print("   ✅ SUCCESS: Compressed loader (gzip) worked")
        return model, "compressed_gzip"
    except Exception as e:
        print(f"   ❌ FAILED: Compressed loader (gzip) - {str(e)}")
    
    # Strategy 4: Compressed loader without gzip
    try:
        model = load_compressed_model(model_path, use_gzip=False)
        print("   ✅ SUCCESS: Compressed loader (no gzip) worked")
        return model, "compressed_no_gzip"
    except Exception as e:
        print(f"   ❌ FAILED: Compressed loader (no gzip) - {str(e)}")
    
    print("   💥 ALL LOADING STRATEGIES FAILED")
    return None, "failed"


def repair_model(model_path: Path, backup: bool = True):
    """Attempt to repair a problematic model file."""
    print(f"\n🔧 Repairing model: {model_path}")
    
    # First, test if we can load it
    model, strategy = test_model_loading(model_path)
    
    if model is None:
        print("   💥 Cannot repair: Model cannot be loaded with any strategy")
        return False
    
    if strategy == "robust":
        print("   ✅ No repair needed: Model loads fine with robust loader")
        return True
    
    # Create backup if requested
    if backup:
        backup_path = model_path.with_suffix(model_path.suffix + '.backup')
        backup_path.write_bytes(model_path.read_bytes())
        print(f"   📝 Created backup: {backup_path}")
    
    # Re-save the model in a standard format
    try:
        # Save without compression first
        temp_path = model_path.with_suffix('.temp.joblib')
        joblib.dump(model, temp_path)
        
        # Replace the original
        model_path.unlink()
        temp_path.rename(model_path)
        
        print(f"   ✅ Repaired: Re-saved model as standard joblib file")
        
        # Test the repaired model
        test_model, test_strategy = test_model_loading(model_path)
        if test_model is not None:
            print(f"   ✅ Verified: Repaired model loads successfully")
            return True
        else:
            print(f"   ❌ Repair failed: Model still cannot be loaded")
            return False
            
    except Exception as e:
        print(f"   ❌ Repair failed: {str(e)}")
        return False


def list_models():
    """List all model files and their loading status."""
    print("\n📋 Model Files in System:")
    print("=" * 60)
    
    model_files = get_model_files()
    
    if not model_files:
        print("No model files found.")
        return
    
    working_count = 0
    broken_count = 0
    
    for model_file in model_files:
        try:
            # Quick test - just try to load
            model = load_model(model_file)
            status = "✅ Working"
            working_count += 1
        except Exception:
            status = "❌ Broken"
            broken_count += 1
        
        size_mb = model_file.stat().st_size / (1024*1024)
        rel_path = model_file.relative_to(Path.cwd()) if model_file.is_relative_to(Path.cwd()) else model_file
        
        print(f"{status:12} {size_mb:6.1f}MB  {rel_path}")
    
    print("=" * 60)
    print(f"Summary: {working_count} working, {broken_count} broken, {len(model_files)} total")


def repair_all_models():
    """Repair all broken models."""
    print("\n🔧 Repairing All Models:")
    print("=" * 60)
    
    model_files = get_model_files()
    repaired_count = 0
    
    for model_file in model_files:
        try:
            # Test if it needs repair
            model = load_model(model_file)
            print(f"✅ {model_file.name}: Already working, skipping")
        except Exception:
            print(f"🔧 {model_file.name}: Needs repair, attempting...")
            if repair_model(model_file):
                repaired_count += 1
    
    print("=" * 60)
    print(f"Repair complete: {repaired_count} models repaired")


def main():
    parser = argparse.ArgumentParser(description="Fix compressed model loading issues")
    parser.add_argument("--list", action="store_true", help="List all model files and their status")
    parser.add_argument("--test", type=str, help="Test loading a specific model file")
    parser.add_argument("--repair", type=str, help="Repair a specific model file")
    parser.add_argument("--repair-all", action="store_true", help="Repair all broken models")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups when repairing")
    
    args = parser.parse_args()
    
    if not any([args.list, args.test, args.repair, args.repair_all]):
        parser.print_help()
        return
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        list_models()
    
    if args.test:
        model_path = Path(args.test)
        if not model_path.exists():
            # Try relative to models dir
            model_path = MODELS_DIR / args.test
        
        if not model_path.exists():
            print(f"❌ Model file not found: {args.test}")
            return
        
        test_model_loading(model_path)
    
    if args.repair:
        model_path = Path(args.repair)
        if not model_path.exists():
            # Try relative to models dir
            model_path = MODELS_DIR / args.repair
        
        if not model_path.exists():
            print(f"❌ Model file not found: {args.repair}")
            return
        
        repair_model(model_path, backup=not args.no_backup)
    
    if args.repair_all:
        repair_all_models()


if __name__ == "__main__":
    main()
