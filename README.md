# PittFinalProject
It is the FinalProject of AI course of University of Pittsburgh
# How to start
## 1. Get BingX API KEY
You have to go to Bing X Web and create an account. 

After you have an account, apply a API KEY

## 2. run the get_bingX_data.py
First you need to add `BINGX_APIKEY` and `BINGX_SECRETKEY` into your environment variables.

Then, adjest the download date range 
```python
if __name__ == "__main__":
    start_date = "2020-04-09"
    end_date = "2024-04-13"
    demo(start_date, end_date, symbol="ETH-USDT", interval="1h", limit=1440)
```
