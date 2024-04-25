import time
import ujson as json
import numpy as np
from stable_baselines3 import PPO
from PPO2_test import TreadingEnv, Candlestick, Action, MAX_TRAING_STEP_PER_ROUND
def prepare_backtest_data(data):
    # data = reversed(data["data"])  # Reverse if necessary depending on how the time should be ordered
    processed_data = []
    for d in data:
        # Convert fields to float, handle NaNs as needed (e.g., set to 0 or a specific value)
        if np.isnan(d.get("upperband", 0)) or np.isnan(d.get("macd", 0)):
            continue
        processed_data.append(Candlestick(
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d["volume"]),
            time=int(d["time"]),
            upperband=float(d.get("upperband", 0)),
            middleband=float(d.get("middleband", 0)),
            lowerband=float(d.get("lowerband", 0)),
            macd=float(d.get("macd", 0)),
            signal=float(d.get("signal", 0)),
            hist=float(d.get("hist", 0))
        ))
    print(f"{len(processed_data) = }")
    return processed_data


def load_model(path):
    return PPO.load(path)

def backtest_model(model, backtest_data):
    env = TreadingEnv(backtest_data, total_timesteps=len(backtest_data), test=True)
    obs, info = env.reset()  # Extract both observation and info if your env returns a tuple
    done = False
    init_price = -1
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        tmp = env.step(action)
        # print(f"{tmp = }")
        obs, rewards, done, info = tmp[0], tmp[1], tmp[2], tmp[3]
        
        if init_price == -1:
            init_price = env.current_price
        print(f"Step: {env.current_step}, Captal : {env.prev_captal:.2f}, Reward: {rewards:.2f}, InitPrice = {init_price:.2f}, CurrenPrice = {env.current_price:.2f}, Date: {env.current_time} Action taken: {Action(action).name}")
        # return
        # time.sleep(1)
        # if env.current_step >= 10:
        #     return


def run_backtest(data_json, model_path):
    backtest_data = prepare_backtest_data(data_json)
    model = load_model(model_path)
    backtest_model(model, backtest_data)

for i in range (40):
    # Data JSON loaded from your description or external file
    data_json = json.load(open("../DataBase/ETH-USDT_data_1h.json", "r"))
    data_json = list(reversed(data_json["data"]))[20000+i*200:]
    print(f"{len(data_json) = }")
    # Example usage
    model_path = './model/PPO2_trading/PPO2_ETH-USDT_1h_leverge6_4week_short.pkl'
    run_backtest(data_json, model_path)
    print(f"================== Round {i} =================")
    time.sleep(2)
