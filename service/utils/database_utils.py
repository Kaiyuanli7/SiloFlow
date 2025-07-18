#!/usr/bin/env python3
"""
Shared utilities for SiloFlow service layer.
Eliminates redundant code across multiple modules.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import pymysql
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Centralized database connection and configuration management."""
    
    @staticmethod
    def load_config(config_file: str) -> Dict:
        """Load database configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            raise ConfigError(f"Failed to load config: {e}")
    
    @staticmethod
    def get_db_config(config: Dict) -> Dict:
        """Extract database configuration from config dict, handling nested formats."""
        # Handle nested database config (like in streaming_config.json)
        if 'database' in config and isinstance(config['database'], dict):
            return config['database']
        # Handle flat config (direct database settings)
        elif all(key in config for key in ['host', 'port', 'user', 'password', 'database']):
            return config
        else:
            raise ConfigError("Invalid database configuration format")
    
    @staticmethod
    def get_connection(config: Dict) -> pymysql.Connection:
        """Create database connection using configuration."""
        try:
            db_config = DatabaseManager.get_db_config(config)
            connection = pymysql.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                charset='utf8mb4'
            )
            return connection
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}")
    
    @staticmethod
    def test_connection(config: Dict) -> bool:
        """Test database connection."""
        try:
            connection = DatabaseManager.get_connection(config)
            connection.close()
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

class CLIUtils:
    """Common CLI argument patterns and utilities."""
    
    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add common arguments to CLI parsers."""
        parser.add_argument('--config', default="streaming_config.json", 
                          help='Path to database configuration JSON file')
        return parser
    
    @staticmethod
    def add_granary_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add granary-specific arguments."""
        parser.add_argument('--granary', help='Granary name or ID to filter by')
        return parser
    
    @staticmethod
    def add_date_range_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add date range arguments."""
        parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', help='End date (YYYY-MM-DD)')
        return parser
    
    @staticmethod
    def add_retrieval_mode_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add data retrieval mode arguments."""
        parser.add_argument('--full-retrieval', action='store_true', 
                          help='Perform full data retrieval')
        parser.add_argument('--incremental', action='store_true', 
                          help='Perform incremental retrieval')
        parser.add_argument('--date-range', action='store_true', 
                          help='Retrieve data for specific date range')
        parser.add_argument('--days', type=int, default=7, 
                          help='Days to look back for incremental retrieval')
        return parser

class SubprocessUtils:
    """Utilities for running subprocesses with consistent error handling."""
    
    @staticmethod
    def run_subprocess(cmd: list, description: str = "subprocess") -> tuple[bool, str, list]:
        """
        Run a subprocess with consistent error handling.
        
        Returns:
            tuple: (success, error_message, output_lines)
        """
        try:
            import subprocess
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            if process.stdout is None:
                return False, "No output captured from subprocess", []
                
            # Stream output in real-time
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line)
            
            process.wait()
            
            if process.returncode == 0:
                return True, "", output_lines
            else:
                return False, f"Process failed with exit code {process.returncode}", output_lines
                
        except Exception as exc:
            return False, f"Subprocess error: {exc}", []

class ValidationUtils:
    """Common validation utilities."""
    
    @staticmethod
    def validate_config_file(config_file: str) -> bool:
        """Validate that config file exists and is readable."""
        if not config_file:
            return False
        return Path(config_file).exists()
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """Validate date string format (YYYY-MM-DD)."""
        try:
            from datetime import datetime
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_required_args(args: argparse.Namespace, required: list) -> bool:
        """Validate that required arguments are provided."""
        for arg in required:
            if not getattr(args, arg, None):
                return False
        return True

# Custom exceptions for better error handling
class SiloFlowError(Exception):
    """Base exception for SiloFlow service."""
    pass

class DatabaseError(SiloFlowError):
    """Database-related errors."""
    pass

class ConfigError(SiloFlowError):
    """Configuration-related errors."""
    pass

class ProcessingError(SiloFlowError):
    """Data processing errors."""
    pass

class ValidationError(SiloFlowError):
    """Validation errors."""
    pass 