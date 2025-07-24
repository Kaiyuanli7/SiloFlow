#!/usr/bin/env python3
"""
Simple test script to check if the process endpoint works
"""
import requests
import os
import time

def test_process_endpoint():
    # Check if server is running
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        print(f'Server health check: {response.status_code}')
        if response.status_code == 200:
            print(f'Health response: {response.json()}')
        else:
            print("Server not healthy, skipping test")
            return
    except Exception as e:
        print(f'Server not responding: {e}')
        return
    
    # Check existing files before test
    processed_dir = 'data/processed'
    granaries_dir = 'data/granaries'
    
    print(f"\nBefore test:")
    print(f"Granaries dir exists: {os.path.exists(granaries_dir)}")
    if os.path.exists(granaries_dir):
        granary_files = [f for f in os.listdir(granaries_dir) if '中软粮情验证' in f]
        print(f"Chinese granary files: {granary_files}")
    
    print(f"Processed dir exists: {os.path.exists(processed_dir)}")
    if os.path.exists(processed_dir):
        processed_files = [f for f in os.listdir(processed_dir) if '中软粮情验证' in f]
        print(f"Chinese processed files: {processed_files}")
    
    # Test the process endpoint with existing Chinese file
    chinese_file_path = os.path.join(granaries_dir, '中软粮情验证.parquet')
    if os.path.exists(chinese_file_path):
        print(f"\nTesting process endpoint with existing file: {chinese_file_path}")
        try:
            with open(chinese_file_path, 'rb') as f:
                files = {'file': ('中软粮情验证.parquet', f, 'application/octet-stream')}
                response = requests.post('http://localhost:8000/process', 
                                       files=files, 
                                       timeout=120)
            
            print(f"Process response status: {response.status_code}")
            print(f"Process response: {response.text}")
            
            # Wait a moment for processing to complete
            time.sleep(2)
            
            # Check if processed file was created
            if os.path.exists(processed_dir):
                processed_files_after = [f for f in os.listdir(processed_dir) if '中软粮情验证' in f]
                print(f"Chinese processed files after: {processed_files_after}")
                
                if processed_files_after:
                    for pf in processed_files_after:
                        file_path = os.path.join(processed_dir, pf)
                        file_size = os.path.getsize(file_path)
                        print(f"  - {pf}: {file_size} bytes")
                else:
                    print("No processed files created!")
            else:
                print("Processed directory still doesn't exist")
                
        except Exception as e:
            print(f"Error testing process endpoint: {e}")
    else:
        print(f"Chinese test file not found: {chinese_file_path}")

if __name__ == "__main__":
    test_process_endpoint()
