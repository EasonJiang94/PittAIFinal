import os
import time
import hmac
import json
import pandas as pd
from hashlib import sha256
from requests import Session

APIURL = "https://open-api.bingx.com"
BINGX_APIKEY = os.environ.get("BINGX_APIKEY")
BINGX_SECRETKEY = os.environ.get("BINGX_SECRETKEY")
session = Session()

def get_sign(api_secret, payload):
    return hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), sha256).hexdigest()

def send_request(method, path, url_params, payload):
    signed_url = f"{APIURL}{path}?{url_params}&signature={get_sign(BINGX_SECRETKEY, url_params)}"
    headers = {"X-BX-APIKEY": BINGX_APIKEY}
    response = session.request(method, signed_url, headers=headers, data=payload)
    return response.json()

def parse_params(params_map):
    params_str = "&".join(f"{k}={v}" for k, v in sorted(params_map.items()))
    return f"{params_str}&timestamp={int(time.time() * 1000)}"

def demo(start_date, end_date, symbol="BTC-USDT", interval="1M", limit=1440):
    start_time = pd.Timestamp(start_date).timestamp() * 1000
    end_time = pd.Timestamp(end_date).timestamp() * 1000
    path = "/openApi/swap/v3/quote/klines"
    method = "GET"
    his = {}
    total_requests = 0

    print("Starting data collection...")
    while end_time > start_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_time),
            "endTime": int(end_time),
            "limit": limit,
        }
        params_str = parse_params(params)
        response_data = send_request(method, path, params_str, {})
        total_requests += 1
        print(f"Request {total_requests}: Data from {pd.to_datetime(params['startTime'], unit='ms')} to {pd.to_datetime(params['endTime'], unit='ms')}")
        if not response_data['data']:
            print("No more data available.")
            break
        end_time = pd.to_datetime(response_data["data"][-1]["time"], unit="ms").timestamp() * 1000 - 60000
        his.setdefault("data", []).extend(response_data["data"])

    if his:
        file_name = f"{symbol}_data_{interval}.json"
        with open(file_name, "w", encoding="utf-8") as file:
            json.dump(his, file, ensure_ascii=False, indent=4)
        print("Data collection complete. Data written to:", file_name)

if __name__ == "__main__":
    start_date = "2020-04-09"
    end_date = "2024-04-13"
    demo(start_date, end_date, symbol="ETH-USDT", interval="1m", limit=1440)
