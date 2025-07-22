#!/usr/bin/env python3
"""
SiloFlow Client Testing Script
==============================

A comprehensive client testing tool for the SiloFlow service that mimics the functionality
of the local testing service but designed for network clients.

Usage:
    python siloflow_client_tester.py --server 192.168.1.100 --port 8000
    python siloflow_client_tester.py --config client_config.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SiloFlowClientTester:
    """Comprehensive client testing tool for SiloFlow service."""
    
    def __init__(self, server_ip: str, port: int = 8000, timeout: int = 300):
        self.base_url = f"http://{server_ip}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test results storage
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "server": f"{server_ip}:{port}",
            "tests": {}
        }
    
    def test_connection(self) -> bool:
        """Test basic connectivity to the service."""
        try:
            logger.info("Testing connection to SiloFlow service...")
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info("Service is healthy!")
                logger.info(f"   Status: {health_data.get('status', 'unknown')}")
                logger.info(f"   Service: {health_data.get('service', 'unknown')}")
                logger.info(f"   Timestamp: {health_data.get('timestamp', 'unknown')}")
                
                # Check directories
                directories = health_data.get('directories', {})
                for dir_name, exists in directories.items():
                    status = "PASS" if exists else "FAIL"
                    logger.info(f"   {dir_name}: {status}")
                
                self.test_results["tests"]["connection"] = {
                    "status": "PASS",
                    "details": health_data
                }
                return True
            else:
                logger.error(f"Service returned error: {response.status_code}")
                self.test_results["tests"]["connection"] = {
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}"
                }
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to service: {e}")
            self.test_results["tests"]["connection"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_process_endpoint(self, file_path: str) -> bool:
        """Test the /process endpoint with a CSV file."""
        try:
            logger.info(f"Testing /process endpoint with {file_path}...")
            
            if not Path(file_path).exists():
                logger.error(f"❌ File not found: {file_path}")
                self.test_results["tests"]["process"] = {
                    "status": "FAIL",
                    "error": f"File not found: {file_path}"
                }
                return False
            
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = self.session.post(
                    f"{self.base_url}/process",
                    files=files,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Processing successful!")
                logger.info(f"   Granaries processed: {result.get('granaries_processed', 0)}")
                logger.info(f"   Successful: {result.get('successful_granaries', 0)}")
                
                # Show details for each granary
                results = result.get('results', {})
                for granary_name, granary_data in results.items():
                    if granary_data.get('success'):
                        file_size = granary_data.get('file_size_mb', 0)
                        logger.info(f"   PASS {granary_name}: {file_size} MB")
                    else:
                        error = granary_data.get('error', 'Unknown error')
                        logger.info(f"   ❌ {granary_name}: {error}")
                
                self.test_results["tests"]["process"] = {
                    "status": "PASS",
                    "details": result
                }
                return True
            else:
                logger.error(f"❌ Processing failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.test_results["tests"]["process"] = {
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing process endpoint: {e}")
            self.test_results["tests"]["process"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_pipeline_endpoint(self, file_path: str, horizon: int = 7) -> bool:
        """Test the /pipeline endpoint with a CSV file."""
        try:
            logger.info(f"🔄 Testing /pipeline endpoint with {file_path} (horizon: {horizon} days)...")
            
            if not Path(file_path).exists():
                logger.error(f"❌ File not found: {file_path}")
                self.test_results["tests"]["pipeline"] = {
                    "status": "FAIL",
                    "error": f"File not found: {file_path}"
                }
                return False
            
            with open(file_path, 'rb') as file:
                files = {'file': file}
                data = {'horizon': horizon}
                response = self.session.post(
                    f"{self.base_url}/pipeline",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                # Save the forecast results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                forecast_filename = f"forecast_{timestamp}.csv"
                
                with open(forecast_filename, 'wb') as f:
                    f.write(response.content)
                
                # Get summary from headers
                summary_b64 = response.headers.get('X-Forecast-Summary')
                if summary_b64:
                    import base64
                    summary = json.loads(base64.b64decode(summary_b64).decode('utf-8'))
                    logger.info("✅ Pipeline processing successful!")
                    logger.info(f"   Granaries processed: {summary.get('granaries_processed', 0)}")
                    logger.info(f"   Successful: {summary.get('successful_granaries', 0)}")
                    logger.info(f"   Total forecast records: {summary.get('total_forecast_records', 0)}")
                    logger.info(f"   Forecast saved: {forecast_filename}")
                    
                    # Show details for each granary
                    granaries = summary.get('granaries', {})
                    for granary_name, granary_data in granaries.items():
                        if 'total_records' in granary_data:
                            logger.info(f"   ✅ {granary_name}: {granary_data['total_records']} records")
                        else:
                            error = granary_data.get('error', 'Unknown error')
                            logger.info(f"   ❌ {granary_name}: {error}")
                    
                    self.test_results["tests"]["pipeline"] = {
                        "status": "PASS",
                        "details": summary,
                        "forecast_file": forecast_filename
                    }
                else:
                    logger.info("✅ Pipeline processing completed!")
                    logger.info(f"   Forecast saved: {forecast_filename}")
                    self.test_results["tests"]["pipeline"] = {
                        "status": "PASS",
                        "forecast_file": forecast_filename
                    }
                
                return True
            else:
                logger.error(f"❌ Pipeline failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.test_results["tests"]["pipeline"] = {
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing pipeline endpoint: {e}")
            self.test_results["tests"]["pipeline"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_train_endpoint(self) -> bool:
        """Test the /train endpoint."""
        try:
            logger.info("🏋️ Testing /train endpoint...")
            
            response = self.session.post(
                f"{self.base_url}/train",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Training successful!")
                logger.info(f"   Trained granaries: {len(result.get('trained_granaries', []))}")
                logger.info(f"   Skipped granaries: {len(result.get('skipped_granaries', []))}")
                
                if result.get('trained_granaries'):
                    logger.info(f"   Trained: {', '.join(result['trained_granaries'][:5])}")
                    if len(result['trained_granaries']) > 5:
                        logger.info(f"   ... and {len(result['trained_granaries']) - 5} more")
                
                if result.get('errors'):
                    logger.warning(f"   Errors: {result['errors']}")
                
                self.test_results["tests"]["train"] = {
                    "status": "PASS",
                    "details": result
                }
                return True
            else:
                logger.error(f"❌ Training failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.test_results["tests"]["train"] = {
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing train endpoint: {e}")
            self.test_results["tests"]["train"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_forecast_endpoint(self, granary_names: List[str], horizon_days: int = 7) -> bool:
        """Test the /forecast endpoint."""
        try:
            logger.info(f"🔮 Testing /forecast endpoint for {granary_names} (horizon: {horizon_days} days)...")
            
            data = {
                "granaries": granary_names,
                "horizon_days": horizon_days
            }
            
            response = self.session.post(
                f"{self.base_url}/forecast",
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Forecast generation successful!")
                
                granaries = result.get('granaries', {})
                for granary_name, granary_data in granaries.items():
                    if 'total_sensors' in granary_data:
                        sensors = granary_data['total_sensors']
                        logger.info(f"   ✅ {granary_name}: {sensors} sensors")
                    else:
                        error = granary_data.get('error', 'Unknown error')
                        logger.info(f"   ❌ {granary_name}: {error}")
                
                self.test_results["tests"]["forecast"] = {
                    "status": "PASS",
                    "details": result
                }
                return True
            else:
                logger.error(f"❌ Forecast failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.test_results["tests"]["forecast"] = {
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing forecast endpoint: {e}")
            self.test_results["tests"]["forecast"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test the /models endpoint."""
        try:
            logger.info("📋 Testing /models endpoint...")
            
            response = self.session.get(f"{self.base_url}/models", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Models listing successful!")
                logger.info(f"   Total models: {result.get('models_count', 0)}")
                
                models = result.get('models', [])
                for model in models[:5]:  # Show first 5 models
                    granary = model.get('granary', 'Unknown')
                    size = model.get('size_mb', 0)
                    logger.info(f"   📦 {granary}: {size} MB")
                
                if len(models) > 5:
                    logger.info(f"   ... and {len(models) - 5} more models")
                
                self.test_results["tests"]["models"] = {
                    "status": "PASS",
                    "details": result
                }
                return True
            else:
                logger.error(f"❌ Models listing failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.test_results["tests"]["models"] = {
                    "status": "FAIL",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing models endpoint: {e}")
            self.test_results["tests"]["models"] = {
                "status": "FAIL",
                "error": str(e)
            }
            return False
    
    def run_all_tests(self, test_file: Optional[str] = None) -> Dict:
        """Run all available tests."""
        logger.info("🚀 Starting comprehensive SiloFlow client tests...")
        logger.info(f"   Server: {self.base_url}")
        logger.info(f"   Timeout: {self.timeout} seconds")
        logger.info("-" * 60)
        
        # Test 1: Connection
        connection_ok = self.test_connection()
        if not connection_ok:
            logger.error("❌ Cannot proceed with tests - service is not accessible")
            return self.test_results
        
        logger.info("-" * 60)
        
        # Test 2: Process endpoint (if file provided)
        if test_file:
            self.test_process_endpoint(test_file)
            logger.info("-" * 60)
        
        # Test 3: Pipeline endpoint (if file provided)
        if test_file:
            self.test_pipeline_endpoint(test_file)
            logger.info("-" * 60)
        
        # Test 4: Train endpoint
        self.test_train_endpoint()
        logger.info("-" * 60)
        
        # Test 5: Models endpoint
        self.test_models_endpoint()
        logger.info("-" * 60)
        
        # Test 6: Forecast endpoint (with sample granaries)
        sample_granaries = ["中正粮食储备库", "蚬冈库"]
        self.test_forecast_endpoint(sample_granaries)
        logger.info("-" * 60)
        
        # Summary
        self._print_summary()
        
        return self.test_results
    
    def _print_summary(self):
        """Print test summary."""
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"].values() if test["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        
        if failed_tests > 0:
            logger.info("\nFailed Tests:")
            for test_name, test_result in self.test_results["tests"].items():
                if test_result["status"] == "FAIL":
                    error = test_result.get("error", "Unknown error")
                    logger.info(f"   ❌ {test_name}: {error}")
        
        # Save results to file
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! SiloFlow service is working correctly.")
        else:
            logger.warning("⚠️ Some tests failed. Check the details above.")

def create_sample_csv():
    """Create a sample CSV file for testing."""
    sample_data = {
        'storepointName': ['中正粮食储备库', '中正粮食储备库', '蚬冈库', '蚬冈库'],
        'storeName': ['101', '102', 'P1-1-01堆位', 'P1-1-02堆位'],
        'batch': ['2023-03-06 14:47:04', '2023-03-06 14:47:04', '2023-03-06 14:47:04', '2023-03-06 14:47:04'],
        'temp': [18.0, 18.5, 19.5, 19.8],
        'x': [1, 1, 2, 2],
        'y': [1, 1, 1, 1],
        'z': [1, 2, 1, 2],
        'indoor_temp': [18.88, 18.88, 19.2, 19.2],
        'outdoor_temp': [28.0, 28.0, 28.0, 28.0]
    }
    
    df = pd.DataFrame(sample_data)
    filename = "sample_sensor_data.csv"
    df.to_csv(filename, index=False)
    logger.info(f"📄 Created sample CSV file: {filename}")
    return filename

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SiloFlow Client Testing Tool")
    parser.add_argument("--server", required=True, help="Server IP address")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (default: 300)")
    parser.add_argument("--file", help="CSV file to test with (optional)")
    parser.add_argument("--create-sample", action="store_true", help="Create a sample CSV file for testing")
    parser.add_argument("--config", help="Configuration file (JSON)")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            server = config.get('server', args.server)
            port = config.get('port', args.port)
            timeout = config.get('timeout', args.timeout)
            test_file = config.get('file', args.file)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)
    else:
        server = args.server
        port = args.port
        timeout = args.timeout
        test_file = args.file
    
    # Create sample file if requested
    if args.create_sample:
        test_file = create_sample_csv()
    
    # Create and run tester
    tester = SiloFlowClientTester(server, port, timeout)
    
    try:
        results = tester.run_all_tests(test_file)
        
        # Exit with appropriate code
        total_tests = len(results["tests"])
        passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "PASS")
        
        if passed_tests == total_tests:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Some tests failed
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 