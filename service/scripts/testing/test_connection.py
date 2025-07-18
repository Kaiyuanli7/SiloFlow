#!/usr/bin/env python3
"""
Test Database Connection
=======================

This script tests the database connection using the configuration file.
"""

import sys
import urllib.parse
from pathlib import Path

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(service_dir))

from utils.database_utils import DatabaseManager, ValidationUtils, ValidationError

def test_connection():
    """Test database connection with different methods."""
    
    # Database configuration - using the same format as streaming_config.json
    config = {
        "database": {
            "host": "59.36.210.145",
            "port": 3306,
            "database": "cloud_lq",
            "user": "userQuey",
            "password": "UserQ@20240807soft"
        }
    }
    
    print("Testing database connection...")
    print(f"Host: {config['host']}")
    print(f"Port: {config['port']}")
    print(f"Database: {config['database']}")
    print(f"User: {config['user']}")
    
    try:
        # Test with PyMySQL directly
        print("\n1. Testing with PyMySQL...")
        connection = DatabaseManager.get_connection(config)
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"[OK] PyMySQL connection successful: {result}")
        
        # Test a simple query
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM cloud_server.base_store_location_other")
            result = cursor.fetchone()
            if result:
                count = result[0]
                print(f"[OK] Found {count} granary locations")
            else:
                print("[WARNING] No result from count query")
        
        connection.close()
        
    except Exception as e:
        print(f"[ERROR] PyMySQL connection failed: {e}")
        return False
    
    try:
        # Test with SQLAlchemy
        print("\n2. Testing with SQLAlchemy...")
        from sqlalchemy import create_engine, text
        
        db_config = DatabaseManager.get_db_config(config)
        password_encoded = urllib.parse.quote_plus(db_config['password'])
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{password_encoded}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"[OK] SQLAlchemy connection successful: {result.fetchone()}")
        
        print("[OK] All connection tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] SQLAlchemy connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection() 