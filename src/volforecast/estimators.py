import numpy as np
import pandas as pd


def close_to_close(df: pd.DataFrame) -> pd.Series:
    """Close-to-close variance: ln(C_t / C_{t-1})^2."""
    r = np.log(df["Close"] / df["Close"].shift(1))
    return (r**2).rename("cc")


def overnight(df: pd.DataFrame) -> pd.Series:
    """Overnight variance: ln(O_t / C_{t-1})^2."""
    r = np.log(df["Open"] / df["Close"].shift(1))
    return (r**2).rename("on")


def parkinson(df: pd.DataFrame) -> pd.Series:
    """Parkinson variance: ln(H_t / L_t)^2 / (4 ln 2)."""
    r = np.log(df["High"] / df["Low"])
    return (r**2 / (4 * np.log(2))).rename("parkinson")


def garman_klass(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass variance: 0.5 ln(H_t / L_t)^2 - (2 ln 2 - 1) ln(C_t / O_t)^2."""
    r = np.log(df["Close"] / df["Open"])
    u = np.log(df["High"] / df["Low"])
    return (0.5 * u**2 - (2 * np.log(2) - 1) * r**2).rename("gk")


def daily_variance(
    df: pd.DataFrame, method: str = "gk", include_overnight: bool = True
) -> pd.Series:
    """Daily variance using `method` in {"cc", "on", "parkinson", "gk"}.

    `include_overnight` only applies to the range-based estimators ("parkinson"
    and "gk"), which then add the overnight term ln(O_t / C_{t-1})^2. It is
    ignored for "cc" and "on".
    """
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


def annualize(
    var: pd.Series | pd.DataFrame, periods: int = 252
) -> pd.Series | pd.DataFrame:
    """Annualized volatility from daily variance: sqrt(var * periods)."""
    return np.sqrt(var * periods)
