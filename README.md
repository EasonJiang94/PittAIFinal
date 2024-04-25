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

## 3. do the data preprocess
Set the path and run the script `data_preprocess.py`
```python
    file_name = "ETH-USDT_data_1h.json"
```

## 4. run the training code
Notice : You have to install the dependencies of this code. 
```bash
cd Module
python PPO2_test.py | experiment_1.log
```

expected reuslt would be like this:
```text
Round: 201, Step: 350206/349440 (100.22%)
Action: SELL_LONG, current_price: 461.27, prev_price: 459.82, Increase: 1.4499999999999886 (+ 0.003%), current_time: 2020-11-14 20:00:00
operation_history: {'BUY_LONG': 1479, 'SELL_LONG': 1673, 'WAIT': 165}
[DEBUG]  self.prev_captal = 1103573.64, current_price = 461.27, self.price = 0.00,      self.position = 0.00,   self.balance = 1124575.70, [SELL_LONG]
current_captal: 1124575.695355001
current_captal after CPI: 1124569.2765439544
opportunity_cost: 0
reward composition: profit_reward = 1.4807,     op_reward : 0.0000,     opportunityCost_reward: 0.0000, mdd_reward: -0.2217 
Position: 0, Balance: 1124575.695355001, Current Captal: 1124569.2765439544, MDD:  73.90 %
step: 3317 2020-11-14 20:00:00, {'status': (3316, 461.27, 0, 'SELL_LONG'), 'reward': '\x1b[1;32m1.2589927241240966\x1b[0m'}, current_captal: 1124569.2765439544, diff: -0.1114029161199364
reward: 1.2589927241240966, obs[-3] (current cash): 1.0, obs[-2](account value): 5.70779513579171e-06, obs[-1](position): 0.0
===============================Step End===============================
Round: 201, Step: 350207/349440 (100.22%)
Action: BUY_LONG, current_price: 459.74, prev_price: 461.27, Increase: -1.5299999999999727 (-0.003%), current_time: 2020-11-14 21:00:00
operation_history: {'BUY_LONG': 1480, 'SELL_LONG': 1673, 'WAIT': 165}
[DEBUG]  self.prev_captal = 1124569.28, current_price = 459.74, self.price = 459.74,    self.position = 14676.67,       self.balance = 0.00, [BUY_LONG]
[DEBUG][_calculate_temp_balence]self.position = 14676.67, current_price = 459.74, p = 6747454.17, balance = 1117828.24
[DEBUG][_calculate_temp_balence]self.position = 14676.67, current_price = 459.74, p = 6747454.17, balance = 1117828.24
current_captal: 1117828.2411828707
current_captal after CPI: 1117821.8608846904
opportunity_cost: 0
reward composition: profit_reward = -2.6218,    op_reward : 0.0000,     opportunityCost_reward: 0.0000, mdd_reward: -0.2217 
Position: 14676.67414653936, Balance: 0, Current Captal: 1117821.8608846904, MDD:  73.90 %
step: 3318 2020-11-14 21:00:00, {'status': (3317, 459.74, 14676.67414653936, 'BUY_LONG'), 'reward': '\x1b[1;31m-2.8435238708138804\x1b[0m'}, current_captal: 1117821.8608846904, diff: -0.11673449862321696
[DEBUG][_calculate_temp_balence]self.position = 14676.67, current_price = 368.26, p = 5404832.02, balance = -224122.60
reward: -2.8435238708138804, obs[-3] (current cash): -0.0, obs[-2](account value): -1.2004993877018175, obs[-1](position): 1.0
===============================Step End===============================
``` 

# 5. do the backtest
check the `data_json` path and the `model_path`
```python
    data_json = json.load(open("../DataBase/ETH-USDT_data_1h.json", "r"))
    data_json = list(reversed(data_json["data"]))[20000+i*200:]
    print(f"{len(data_json) = }")
    # Example usage
    model_path = './model/PPO2_trading/PPO2_ETH-USDT_1h_leverge6_4week_short.pkl'
```
```
python backtest.py
```