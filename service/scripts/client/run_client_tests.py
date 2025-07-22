#!/usr/bin/env python3
"""
SiloFlow Client Testing Examples
================================

This script demonstrates how to use the SiloFlow client tester
with different testing scenarios.

Usage:
    python run_client_tests.py --server 192.168.1.100
    python run_client_tests.py --config client_config.json
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path to import the tester
sys.path.insert(0, str(Path(__file__).parent))

from siloflow_client_tester import SiloFlowClientTester, create_sample_csv

def run_basic_tests(server_ip: str, port: int = 8000):
    """Run basic connectivity and health tests."""
    print("🔍 Running Basic Tests...")
    print("=" * 50)
    
    tester = SiloFlowClientTester(server_ip, port)
    
    # Test connection only
    success = tester.test_connection()
    if success:
        print("✅ Basic connectivity test passed!")
        return True
    else:
        print("❌ Basic connectivity test failed!")
        return False

def run_process_tests(server_ip: str, port: int = 8000, test_file: str = None):
    """Run process endpoint tests."""
    print("\n📊 Running Process Tests...")
    print("=" * 50)
    
    if not test_file:
        print("📄 Creating sample CSV file for testing...")
        test_file = create_sample_csv()
    
    tester = SiloFlowClientTester(server_ip, port)
    
    # Test process endpoint
    success = tester.test_process_endpoint(test_file)
    if success:
        print("✅ Process endpoint test passed!")
        return True
    else:
        print("❌ Process endpoint test failed!")
        return False

def run_pipeline_tests(server_ip: str, port: int = 8000, test_file: str = None):
    """Run pipeline endpoint tests."""
    print("\n🔄 Running Pipeline Tests...")
    print("=" * 50)
    
    if not test_file:
        print("📄 Creating sample CSV file for testing...")
        test_file = create_sample_csv()
    
    tester = SiloFlowClientTester(server_ip, port)
    
    # Test pipeline endpoint
    success = tester.test_pipeline_endpoint(test_file)
    if success:
        print("✅ Pipeline endpoint test passed!")
        return True
    else:
        print("❌ Pipeline endpoint test failed!")
        return False

def run_full_tests(server_ip: str, port: int = 8000, test_file: str = None):
    """Run all tests."""
    print("\n🚀 Running Full Test Suite...")
    print("=" * 50)
    
    if not test_file:
        print("📄 Creating sample CSV file for testing...")
        test_file = create_sample_csv()
    
    tester = SiloFlowClientTester(server_ip, port)
    
    # Run all tests
    results = tester.run_all_tests(test_file)
    
    # Check overall success
    total_tests = len(results["tests"])
    passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "PASS")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️ {passed_tests}/{total_tests} tests passed")
        return False

def main():
    """Main function with different testing scenarios."""
    parser = argparse.ArgumentParser(description="SiloFlow Client Testing Examples")
    parser.add_argument("--server", required=True, help="Server IP address")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--file", help="CSV file to test with (optional)")
    parser.add_argument("--test-type", choices=["basic", "process", "pipeline", "full"], 
                       default="full", help="Type of tests to run (default: full)")
    
    args = parser.parse_args()
    
    print("🧪 SiloFlow Client Testing Examples")
    print(f"   Server: {args.server}:{args.port}")
    print(f"   Test Type: {args.test_type}")
    print(f"   Test File: {args.file or 'Auto-generated sample'}")
    print("-" * 60)
    
    try:
        if args.test_type == "basic":
            success = run_basic_tests(args.server, args.port)
        elif args.test_type == "process":
            success = run_process_tests(args.server, args.port, args.file)
        elif args.test_type == "pipeline":
            success = run_pipeline_tests(args.server, args.port, args.file)
        elif args.test_type == "full":
            success = run_full_tests(args.server, args.port, args.file)
        
        if success:
            print("\n🎉 Testing completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 