import requests
import csv
import sys

API_URL = "http://localhost:8502/api/forecast"
CSV_PATH = "../granaries_and_silos.csv"
MODEL_NAME = "三角仓库_forecast_model.joblib"  # Example model name, change as needed

def test_api_from_csv(csv_path, model_name, api_url):
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            granary_id, sub_table_id, silo_id = row[:3]
            payload = {
                "granary_id": granary_id,
                "sub_table_id": sub_table_id,
                "silo_id": silo_id,
                "model_name": model_name
            }
            print(f"Testing: {payload}")
            try:
                response = requests.post(api_url, json=payload, timeout=10)
                print(f"Status: {response.status_code}")
                print(f"Response: {response.json()}")
            except Exception as e:
                print(f"Error: {e}")
            print("-" * 40)

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    model_name = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME
    api_url = sys.argv[3] if len(sys.argv) > 3 else API_URL
    test_api_from_csv(csv_path, model_name, api_url)

if __name__ == "__main__":
    main()
