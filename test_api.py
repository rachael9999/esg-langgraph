import requests
import json

def test_external_api():
    url = "http://192.168.1.100:8000/pack/companyinfo"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "appkey": "zhongTanDify"
    }
    data = {
        "keyword": "山东东方鸿浩厨业有限公司"
    }
    
    print(f"Requesting {url}...")
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"Status Code: {resp.status_code}")
        print("Response Content:")
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_external_api()
