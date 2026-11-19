import numpy as np
import pandas as pd
from data import load, TICKERS

def close_to_close(df: pd.DataFrame) -> pd.Series:
    r = np.log(df["Close"] / df["Close"].shift(1))
    return (r ** 2).rename("cc")

def overnight(df: pd.DataFrame) -> pd.Series:
    r = np.log(df["Open"] / df["Close"].shift(1))
    return (r ** 2).rename("on")

def parkinson(df: pd.DataFrame) -> pd.Series:
    r = np.log(df["High"] / df["Low"])
    return (r ** 2 / (4 * np.log(2))).rename("parkinson")

def garman_klass(df: pd.DataFrame) -> pd.Series:
    r = np.log(df["Close"] / df["Open"])
    u = np.log(df["High"] / df["Low"])
    return (0.5 * u ** 2 - (2 * np.log(2) - 1) * r ** 2).rename("gk")
