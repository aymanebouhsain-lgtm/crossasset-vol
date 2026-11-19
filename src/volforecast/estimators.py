import numpy as np
import pandas as pd

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

def daily_variance(df: pd.DataFrame, method="gk", include_overnight=True) -> pd.Series:
    if method == "cc":
        var = close_to_close(df)
    elif method == "on":
        var = overnight(df)
    elif method == "parkinson":
        var = parkinson(df)
    elif method == "gk":
        var = garman_klass(df)
    else:
        raise ValueError(f"Unknown method: {method}")

    if include_overnight and (method == "parkinson" or method == "gk"):
        var += overnight(df)

    return var

def annualize(var, periods: int = 252):
    return np.sqrt(var * periods)