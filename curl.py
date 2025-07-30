import requests
import json

API_URL = "http://localhost:8502/api/forecast"
payload = {
    "granary_name": "三角仓库",
    "silo_id": "060761ff080d46c082053f97043e79bd"
}
headers = {"Content-Type": "application/json"}
response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
print(response.status_code)
print(response.json())