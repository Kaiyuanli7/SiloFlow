#!/usr/bin/env python3
"""
Setup Script for Data Retrieval System
======================================

This script helps set up and test the automated data retrieval system.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import urllib.parse

def create_config():
    """Create the default configuration file."""
    config = {
        "database": {
            "host": "rm-wz9805aymm47k9z3qxo.mysql.rds.aliyuncs.com",
            "port": 3306,
            "database": "cloud_lq",
            "user": "userQuey",
            "password": "UserQ@20240807soft"
        },
        "processing": {
            "initial_chunk_size": 50000,
            "min_chunk_size": 10000,
            "max_chunk_size": 150000,
            "memory_threshold_percent": 75,
            "output_dir": "data/streaming",
            "log_level": "INFO"
        },
        "advanced": {
            "max_retries": 3,
            "retry_delay": 5,
            "enable_progress_bar": True
        }
    }
    
    config_path = Path("streaming_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Configuration file created: {config_path}")
    return config_path

def test_database_connection():
    """Test the database connection."""
    try:
        from sqlalchemy import create_engine, text
        
        config_path = Path("streaming_config.json")
        if not config_path.exists():
            print("✗ Configuration file not found. Creating default config...")
            create_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        db_config = config['database']
        password_encoded = urllib.parse.quote_plus(db_config['password'])
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{password_encoded}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful")
            
            # Test a simple query
            result = conn.execute(text("SELECT COUNT(*) FROM cloud_server.base_store_location_other"))
            count = result.fetchone()[0]
            print(f"✓ Found {count} granary locations")
            
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_data_retrieval():
    """Test data retrieval with a small sample."""
    try:
        from sql_data_streamer import SQLDataStreamer
        
        print("\nTesting data retrieval...")
        
        # Test with a small date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        streamer = SQLDataStreamer()
        
        # Get granaries info
        granaries_df = streamer.get_all_granaries_and_silos()
        print(f"✓ Found {len(granaries_df)} silos across {granaries_df['sub_table_id'].nunique()} granaries")
        
        # Test with first granary
        if not granaries_df.empty:
            first_granary = granaries_df.groupby(['storepoint_id', 'store_name', 'sub_table_id']).first()
            granary_name = first_granary['store_name'].iloc[0]
            print(f"✓ Testing with granary: {granary_name}")
            
            # Get a small sample of data
            results = streamer.stream_all_data(
                start_date=start_date,
                end_date=end_date,
                run_pipeline=False  # Skip pipeline for testing
            )
            
            if results['success']:
                print(f"✓ Data retrieval test successful: {results['total_records']} records")
                return True
            else:
                print("✗ Data retrieval test failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data retrieval test failed: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/streaming",
        "data/streaming/granaries",
        "data/processed",
        "models",
        "temp_uploads"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def main():
    """Main setup function."""
    print("SiloFlow Data Retrieval System Setup")
    print("=" * 40)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Create configuration
    print("\n2. Setting up configuration...")
    config_path = create_config()
    
    # Test database connection
    print("\n3. Testing database connection...")
    if not test_database_connection():
        print("✗ Setup failed: Database connection test failed")
        sys.exit(1)
    
    # Test data retrieval
    print("\n4. Testing data retrieval...")
    if not test_data_retrieval():
        print("✗ Setup failed: Data retrieval test failed")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("✓ Setup completed successfully!")
    print("=" * 40)
    
    print("\nNext steps:")
    print("1. Run full data retrieval:")
    print("   python automated_data_retrieval.py --full-retrieval")
    print()
    print("2. Run incremental retrieval (last 7 days):")
    print("   python automated_data_retrieval.py --incremental --days 7")
    print()
    print("3. Run for specific date range:")
    print("   python automated_data_retrieval.py --date-range --start 2024-01-01 --end 2024-12-31")
    print()
    print("4. View usage examples:")
    print("   python run_data_streaming.py")

if __name__ == "__main__":
    main() 