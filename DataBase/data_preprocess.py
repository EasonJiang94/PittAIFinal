import os
import time
import talib
import json
import numpy as np


def calaulate_bbands(data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    upperband, middleband, lowerband = talib.BBANDS(
        data, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype
    )
    return upperband, middleband, lowerband


def calaulate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, signal, hist = talib.MACD(
        data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )
    return macd, signal, hist


if __name__ == "__main__":
    file_name = "ETH-USDT_data_1m.json"
    datas = json.load(open(file_name, "r", encoding="utf-8"))

    close_price = np.array([float(x["close"]) for x in datas["data"]])
    upperband, middleband, lowerband = calaulate_bbands(close_price)
    for idx, (u, m, l) in enumerate(zip(upperband, middleband, lowerband)):
        datas["data"][idx]["upperband"] = u
        datas["data"][idx]["middleband"] = m
        datas["data"][idx]["lowerband"] = l
    macd, signal, hist = calaulate_macd(close_price)
    # print(f"macd len: {len(macd)}")
    # print(f"signal len: {len(signal)}")
    # print(f"hist len: {len(hist)}")
    for idx, (m, s, h) in enumerate(zip(macd, signal, hist)):
        datas["data"][idx]["macd"] = m
        datas["data"][idx]["signal"] = s
        datas["data"][idx]["hist"] = h

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)

    # 1709251200000 1586390460000
