from abc import ABC
import enum
import random
import os
import ujson as json

from typing import Any

# import tensorflow as tf
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

# import gym
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
import numpy as np

# from utils import ckeck_folder_and_create

try:
    import pandas as pd
except ImportError:
    import pip

    pip.main(["install", "pandas"])
    import pandas as pd

# TODO: Use Argparser/Configparser to set the leverage
LEVERAGE = 6
INITIAL_BALANCE = 100000.0
MAX_TRAING_STEP_PER_ROUND = 24 * 7 * 52 * 2    # 24 hours * 7 days * 52 weeks * 2 years
CPI = 0.05 / (24 * 365)  # 5% per year, CPI per hour
EPLISON = 1e-8

def ckeck_folder_and_create(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Create folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")


class Action(enum.Enum):
    BUY_LONG = 0
    SELL_LONG = 1
    BUY_SHORT = 2
    SELL_SHORT = 3
    BUY_LONG_HALF = 4
    SELL_LONG_HALF = 5
    WAIT = 6
    # BUY_SHORT_HALF = 7
    # SELL_SHORT_HALF = 8


# abstract class for handling fee
class HandlingFee(ABC):
    def __init__(self) -> None:
        self.market_fee = 0.0
        self.limit_fee = 0.0

    def calculate(self, price: float, amount: float) -> float:
        raise NotImplementedError


class BingxPerpetualFuturesHandlingFee(HandlingFee):
    def __init__(self) -> None:
        super().__init__()
        self.market_fee = 0.0005
        self.limit_fee = 0.0002

    def calculate(self, price: float, amount: float) -> float:
        return price * amount * self.market_fee


class Candlestick(object):
    def __init__(
        self,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        time: int,
        upperband: float,
        middleband: float,
        lowerband: float,
        macd: float,
        signal: float,
        hist: float,
    ) -> None:
        """
        "open": "7333.9",
        "close": "7335.3",
        "high": "7346.7",
        "low": "7325.1",
        "volume": "5.79",
        "time": 1586390460000,
        "upperband": NaN,
        "middleband": NaN,
        "lowerband": NaN,
        "macd": NaN,
        "signal": NaN,
        "hist": NaN
        """
        self.open = float(open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = float(volume)
        self.time = time  # ms
        self.upperband = upperband
        self.middleband = middleband
        self.lowerband = lowerband
        self.macd = macd
        self.signal = signal
        self.hist = hist

    def __str__(self) -> str:
        return (
            f"Open: {self.open}, High: {self.high}, Low: {self.low}, Close: {self.close}, ",
            f"Volume: {self.volume}, Time: {datetime.utcfromtimestamp(self.time/1000)}",
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}>Open: {self.open}, High: {self.high}, Low: {self.low}, Close: {self.close}, ",
            f"Volume: {self.volume}, Time: {datetime.utcfromtimestamp(self.time/1000)}",
        )

    # without time
    def __call__(self):
        return (
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.upperband,
            self.middleband,
            self.lowerband,
            self.macd,
            self.signal,
            self.hist,
        )

    # def __iter__(self):
    #     return iter(self())


# Create stock trading environment
class TreadingEnv(gym.Env):
    def __init__(self, stock_history: list[Candlestick], total_timesteps: int, test: bool=False) -> None:
        super(TreadingEnv, self).__init__()
        self.lookback_window_size = (
            24 * 7 * 4
        )  # 24 hours * 7 days * 4 weeks, this means model can see 24 hours of data.
        # if len(stock_history) <= self.lookback_window_size + MAX_TRAING_STEP_PER_ROUND:
        #     raise ValueError("Insufficient data for the specified training steps and lookback window size.")

        # self.history = stock_history
        # self.__max_start_index = len(self.history) - MAX_TRAING_STEP_PER_ROUND
        # self.__start_index = random.randint(
        #     self.lookback_window_size, self.__max_start_index
        # )

        self.test = test
        self.testing_round = 0
        self.training_step = 0
        self.__total_timesteps = total_timesteps

        self.history = stock_history
        self.normalized_history = self.__data_normalized()

        self.__max_start_index = len(self.history) - MAX_TRAING_STEP_PER_ROUND
        print(f"[DEBUG] {self.lookback_window_size = }, {self.__max_start_index = }")
        self.__start_index = self.lookback_window_size if self.test else random.randint(
            self.lookback_window_size, self.__max_start_index
        ) 
        self.balance_history = []
        self.trade_history = {}
        self.operation_history = {x.name: 0 for x in Action}
        self.trade_fee = BingxPerpetualFuturesHandlingFee()
        self.leverage = LEVERAGE

        self.balance = INITIAL_BALANCE  # 10000 USDT
        self.prev_captal = self.balance
        self.current_step = 0
        self.position = 0
        self.average_cost = 0
        self.price = 0
        self.mdd = 0  # Maximum Drawdown
        self.handling_fee = 0.0002 if (self.leverage > 1) else 0.001
        self.margin = 0.2 if (self.leverage > 1) else 0 * self.average_cost

        # In POC stage, only use market price buy and sell operations. Expand after confirming feasibility.
        self.action_space = spaces.Discrete(len(Action))
        """
        Maybe use multiple dimantion array to represent difference action, like: limit price, stop price, etc.
        e.g. we can define a 3x4x2x1x1 array to represent the action space,
        first dimension represent the type of action, 0: buy, 1: sell, 2: hold
        second dimension represent the type of order, 0: market, 1: limit, 2: stop loss, 3: take profit
        third dimension represent the type of position, 0: long, 1: short
        forth dimension represent the type of leverage, 0: 1x, 1: 2x, 2: 3x, 3: 4x
        fifth dimension represent the type of amount
        self.action_space = spaces.MultiDiscrete([3, 4, 2, 1, 1])
        """

        # The observation space is a lookback_window_size*Candlestick data size + 1 array, with the last value being the current cash balance, account value and position.
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.lookback_window_size * 11 + 3,),
            dtype=np.float64,
        )

    def __data_normalized(self): 
        temp = []
        for idx, Candlestick_data in enumerate(self.history):
            if idx == 0:
                continue
            normalized_open = (Candlestick_data.open - self.history[idx-1].close) / self.history[idx-1].close if self.history[idx-1].close != 0 else 0
            temp.append(Candlestick(open=(normalized_open - self.history[idx-1].close) / self.history[idx-1].close, 
                                    high=(Candlestick_data.high - self.history[idx-1].close) / self.history[idx-1].close, 
                                    low=(Candlestick_data.low - self.history[idx-1].close) / self.history[idx-1].close, 
                                    close=(Candlestick_data.close - self.history[idx-1].close) / self.history[idx-1].close, 
                                    volume=(Candlestick_data.volume - self.history[idx-1].volume) / (self.history[idx-1].volume + 1e-8), 
                                    time=Candlestick_data.time, 
                                    upperband=(Candlestick_data.upperband - self.history[idx-1].close) / self.history[idx-1].close, 
                                    middleband=(Candlestick_data.middleband - self.history[idx-1].close) / self.history[idx-1].close, 
                                    lowerband=(Candlestick_data.lowerband - self.history[idx-1].close) / self.history[idx-1].close, 
                                    macd=(Candlestick_data.macd - self.history[idx-1].macd) / self.history[idx-1].macd, 
                                    signal=(Candlestick_data.signal - self.history[idx-1].signal) / self.history[idx-1].signal, 
                                    hist=(Candlestick_data.hist - self.history[idx-1].hist) / self.history[idx-1].hist))
        self.history = self.history[1:]
        return temp

    def _next_observation(self):
        frame = []
        temp_frame = self.normalized_history[
            self.__start_index
            + self.current_step : self.__start_index
            + self.current_step
            + self.lookback_window_size
        ]
        

        for c in temp_frame:
            data = c()
            if any(np.isnan(data)):
                print("NaN detected in observations")
            frame.extend(data)

        current_price = self.history[self.__start_index + self.current_step].close
        # calculate the value of crypto after deleverage
        temp_balance = self._calculate_temp_balence(current_price) 
        current_captal = temp_balance + self.balance
        frame.append(self.balance / (current_captal))
        frame.append((current_captal - self.prev_captal) / self.prev_captal)
        frame.append(np.abs(self.position)/self.position if self.position != 0 else 0)
        return np.array(frame)

    def reset(self, seed=None) -> tuple[ObsType, dict]:
        self.__start_index = self.lookback_window_size if self.test else random.randint(
            self.lookback_window_size, self.__max_start_index
        ) 
        self.leverage = LEVERAGE
        self.lookback_window_size = (
            24 * 7 * 4
        )  # 60 minutes * 24 hours, this means model can see 24 hours of data.
        self.balance = INITIAL_BALANCE  # 10000 USDT
        self.prev_balance = INITIAL_BALANCE
        self.prev_captal = self.balance

        self.current_step = 0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.average_cost = 0
        self.price = 0
        self.trade_history = {}
        self.mdd = 0
        self.balance_history = []
        self.operation_history = {x.name: 0 for x in Action}
        self.testing_round += 1

        return self._next_observation(), {}

    def step(self, action) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        current_price = self.history[
            self.__start_index + self.current_step + self.lookback_window_size
        ].close

        previous_price = self.history[
            self.__start_index + self.current_step + self.lookback_window_size - 1
        ].close

        current_time = pd.to_datetime(
            self.history[
                self.__start_index + self.current_step + self.lookback_window_size
            ].time,
            unit="ms",
        )

        reward = 0
        deficit = 0
        profit = 0
        opportunity_cost = 0
        miss_operation = False
        increase = current_price - previous_price
        increase_percent = increase / previous_price
        increase_percent_str = (
            f"\033[1;31m{increase} ({increase_percent: .03f}%)\033[0m"
            if increase < 0
            else f"\033[1;32m{increase} (+{increase_percent: .03f}%)\033[0m"
        )
        self.operation_history[Action(action).name] += 1
        print(f"Round: {self.testing_round}, Step: {self.training_step}/{self.__total_timesteps} ({self.training_step/self.__total_timesteps*100:.2f}%)")
        print(
            f"Action: {Action(action).name}, current_price: {current_price}, "
            f"prev_price: {previous_price}, Increase: {increase_percent_str}, "
            f"current_time: {current_time}"
        )
        print(f"operation_history: {self.operation_history}")

        if action == Action.BUY_LONG.value:
            if self.balance > 0:
                self.price = (self.price * self.position + self.balance * self.leverage) / (self.position + (self.balance * self.leverage) / (current_price))
                self.position += (self.balance * self.leverage) / current_price
                self.balance = 0
                self.trade_history[current_time] = {
                    "status": (
                        self.current_step,
                        current_price,
                        self.position,
                        "BUY_LONG",
                    )
                }
                print(f"[DEBUG]\t {self.prev_captal = :.2f}, {current_price = :.2f}, {self.price = :.2f},\t{self.position = :.2f},\t{self.balance = :.2f}, [BUY_LONG]", flush=True)
            else:
                miss_operation = True
        elif action == Action.BUY_LONG_HALF.value:
            if self.balance > 0:
                self.price = (self.price * self.position + self.balance * self.leverage / 2) / (self.position + (self.balance * self.leverage) / (2 * current_price))
                self.position += (self.balance * self.leverage) / (current_price * 2)
                self.balance = self.balance / 2
                self.trade_history[current_time] = {
                    "status": (
                        self.current_step,
                        current_price,
                        self.position,
                        "BUY_LONG_HALF",
                    )
                }
                print(f"[DEBUG]\t {self.prev_captal = :.2f}, {current_price = :.2f}, {self.price = :.2f},\t{self.position = :.2f},\t{self.balance = :.2f}, [BUY_LONG_HALF]", flush=True)
            else:
                miss_operation = True
        elif action == Action.SELL_LONG.value:
            if self.position > 0:
                self.balance += (
                    (self.position * current_price)
                    - (self.leverage - 1) * (self.position / self.leverage) * self.price
                    - self.trade_fee.market_fee * current_price * self.position
                    - self.trade_fee.market_fee * self.price * self.position
                )

                self.position = 0
                self.price = 0
                self.trade_history[current_time] = {
                    "status": (
                        self.current_step,
                        current_price,
                        self.position,
                        "SELL_LONG",
                    )
                }
                print(f"[DEBUG]\t {self.prev_captal = :.2f}, {current_price = :.2f}, {self.price = :.2f},\t{self.position = :.2f},\t{self.balance = :.2f}, [SELL_LONG]", flush=True)

            else:
                miss_operation = True
        elif action == Action.SELL_LONG_HALF.value:
            if self.position > 0:
                self.balance += (
                    (self.position * current_price / 2)
                    - (self.leverage - 1) * (self.position / self.leverage) * self.price / 2
                    - self.trade_fee.market_fee * current_price * self.position / 2
                    - self.trade_fee.market_fee * self.price * self.position / 2
                )

                self.position = self.position / 2
                self.price = self.price        # It should not be changed
                self.trade_history[current_time] = {
                    "status": (
                        self.current_step,
                        current_price,
                        self.position,
                        "SELL_LONG_HALF",
                    )
                }
                print(f"[DEBUG]\t {self.prev_captal = :.2f}, {current_price = :.2f}, {self.price = :.2f},\t{self.position = :.2f},\t{self.balance = :.2f}, [SELL_LONG_HALF]", flush=True)

            else:
                miss_operation = True
        elif action == Action.BUY_SHORT.value:
            if self.balance > 0 and self.position == 0 :
                self.price = current_price
                self.position -= (self.balance * self.leverage) / current_price
                self.balance = 0
                self.trade_history[current_time] = {
                    "status": (
                        self.current_step,
                        current_price,
                        self.position,
                        "BUY_SHORT",
                    )
                }
                print(f"[DUY_SHORT]\t {self.price = :.2f},\t{self.position = :.2f},\t{self.balance = :.2f},\t{current_price = :.2f}, EBUG][B", flush=True)

            else:
                miss_operation = True

        elif action == Action.SELL_SHORT.value:
            if self.position < 0:
                self.position = -self.position
                profit = (self.price - current_price) * self.position

                self.balance += (
                    (self.position / self.leverage) * self.price
                    + profit
                    - self.trade_fee.market_fee * current_price * self.position
                    - self.trade_fee.market_fee * self.price * self.position
                )
                self.position = 0
                self.price = 0
                self.trade_history[current_time] = {
                    "status": (
                        self.current_step,
                        current_price,
                        self.position,
                        "SELL_SHORT",
                    )
                }
                print(f"[DELL_SHORT]\t {self.price = :.2f},\t{self.position = :.2f},\t{self.balance = :.2f},\t{current_price = :.2f}, EBUG][S", flush=True)

            else:
                miss_operation = True
        elif action == Action.WAIT.value:
            opportunity_cost = (
                np.abs(increase) * (self.balance * self.leverage) / current_price
            )
            print(f"[DEBUG]\t {self.prev_captal = :.2f}, {current_price = :.2f}, {self.price = :.2f},\t{self.position = :.2f},\t{self.balance = :.2f}, [WAIT]", flush=True)


        # Check for margin call
        need_force_sell = self._check_margin_call(current_price)
        if need_force_sell:
            self.trade_history[current_time] = {
                "status": (
                    self.current_step,
                    current_price,
                    self.position,
                    "MARGIN_CALL",
                )
            }

        self.current_step += 1
        self.training_step += 1
        # Calculate reward
        # TODO: normalize the reward to -1 ~ 1

        current_captal: float = self.balance + self._calculate_temp_balence(current_price)
        print(f"current_captal: {current_captal}")
        self.balance_history.append(current_captal)
        self.mdd = max(self.mdd, (max(self.balance_history) - current_captal) / max(self.balance_history))
        current_captal = current_captal * (1 - (CPI))
        balance_diff = current_captal - self.prev_captal
        profit_weight = 2 if balance_diff >= 0 else 9
        profit_reward = profit_weight * np.tanh(balance_diff * 50 / self.prev_captal)
        op_reward = -1 if miss_operation else 0.0
        opportunity_reward = 0.01 * opportunity_cost
        mdd_reward = -0.3 * self.mdd
        print(f"current_captal after CPI: {current_captal}")
        print(f"opportunity_cost: {opportunity_cost}")

        base_point = (current_captal) / max(self.balance_history)
        diff = (current_captal - max(self.balance_history)) / max(self.balance_history)
        # TODO : Try add MDD into reward or operation profit
        print(f"reward composition: {profit_reward = :.4f},"\
                f"\top_reward : {op_reward:.4f},"\
                f"\topportunityCost_reward: {opportunity_reward:.04f},"\
                f"\tmdd_reward: {mdd_reward:.4f} ")
        position_str = "Long" if self.position > 0 else "Short" if self.position < 0 else "No position"
        print(f"Position: {self.position}, Balance: {self.balance}, Current Captal: {current_captal}, MDD: {self.mdd * 100: .2f} %")
     
        reward = (profit_reward + op_reward - opportunity_reward + mdd_reward)

        if self.trade_history.get(current_time):
            self.trade_history[current_time]["reward"] = (
                f"\033[1;31m{reward}\033[0m"
                if reward <= 0
                else f"\033[1;32m{reward}\033[0m"
            )
        else:
            self.trade_history[current_time] = {
                "reward": (
                    f"\033[1;31m{reward}\033[0m"
                    if reward <= 0
                    else f"\033[1;32m{reward}\033[0m"
                )
            }

        self.prev_captal = current_captal
        self.prev_balance = self.balance
        terminated = truncated = done = (
            self.current_step >= MAX_TRAING_STEP_PER_ROUND
        ) or current_captal < (INITIAL_BALANCE / 3)
        print(
            f"step: {self.current_step} {current_time}, {self.trade_history[current_time]}, current_captal: {current_captal}, diff: {diff}"
        )
        if done:
            print(f"===============================Done===============================")
            obs = self.reset()
            print(obs)
        else:
            obs = self._next_observation()
            print(
                f"reward: {self.trade_history[current_time]['reward']}, obs[-3] (current cash): {obs[-3]}, obs[-2](account value): {obs[-2]}, obs[-1](position): {obs[-1]}"
            )
            print(
                f"===============================Step End==============================="
            )
        return obs, reward, terminated, truncated, self.trade_history

    def _calculate_temp_balence(self, current_price: float) -> float:
        balance = 0
        if self.position > 0:
            balance = (
                (self.position * current_price)
                - ((self.leverage - 1) * (self.position / self.leverage) * self.price)
                - (self.trade_fee.market_fee * current_price * self.position)
                - (self.trade_fee.market_fee * self.price * self.position)
            )
            print(f"[DEBUG][_calculate_temp_balence]{self.position = :.2f}, \
{current_price = :.2f}, p = {current_price*self.position:.2f}, {balance = :.2f}")
        elif self.position < 0:
            position = -self.position
            profit = (self.price - current_price) * position
            balance = (
                (position / self.leverage) * self.price
                + profit
                - self.trade_fee.market_fee * current_price * position
                - self.trade_fee.market_fee * self.price * position
            )

        return balance

    def _check_margin_call(self, current_price: float) -> bool:
        if self.position == 0:
            return False

        balance = self._calculate_temp_balence(current_price)

        if self.position > 0:
            cost = self.price * self.position / self.leverage
        elif self.position < 0:
            position = -self.position
            cost = self.price * position / self.leverage

        if balance < self.margin * cost:
            self.position = 0
            self.balance = balance

            return True
        return False
    
    @property
    def current_time(self):
        return pd.to_datetime(
            self.history[
                self.__start_index + self.current_step + self.lookback_window_size
            ].time,
            unit="ms",
        )
    @property
    def current_price(self):
        return self.history[
            self.__start_index + self.current_step + self.lookback_window_size
        ].close

    # TODO: need fix
    def render(self):
        profit = (
            self.balance
            + self.position
            * self.history[
                self.__start_index + self.current_step + self.lookback_window_size
            ]
            - INITIAL_BALANCE
        )
        print(f"Step: {self.current_step}, Profit: {profit}")

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __del__(self):
        pass


def create_and_train(
    train_data: list[Candlestick],
    train_step: int = MAX_TRAING_STEP_PER_ROUND * 20,
) -> PPO:
    
    env = TreadingEnv(train_data, total_timesteps=train_step)
    model = PPO(MlpPolicy, env, verbose=1, device='cuda')
    model.learn(total_timesteps=train_step)
    return model


def save_model(model: PPO, path: str) -> None:
    model.save(path)


def prepare_training_data(
    data_path: str = "../DataBase/ETH-USDT_data_1m.json",
) -> list[Candlestick]:
    source_data = json.load(open(data_path, "r", encoding="utf-8"))
    source_data = list(reversed(source_data["data"]))[:20000]
    data = []
    for d in source_data:
        if np.isnan(d["upperband"]) or np.isnan(d["macd"]):
            continue
        data.append(Candlestick(**d))
    return data

def main():
    np.random.seed(52)
    target = "ETH-USDT"
    time_frame = "1h"
    training_data = prepare_training_data(
        f"../DataBase/{target}_data_{time_frame}.json"
    )
    print(f'[DEBUG] {f"../DataBase/{target}_data_{time_frame}.json"}')
    model = create_and_train(training_data)
    save_dir = "./model"
    model_path = os.path.join(
            save_dir,
            "PPO2_trading",
            f"PPO2_{target}_{time_frame}_leverge{LEVERAGE}_4week_short_profit_weight.pkl",
        )
    ckeck_folder_and_create(save_dir)
    save_model(model, model_path)
    print(f"Model has been saved. : {model_path}")


if __name__ == "__main__":
    main()