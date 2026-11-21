import numpy as np
import pandas as pd


def _check_inputs(y: pd.Series, y_hat: pd.Series, positive: bool = False) -> None:
    """Raise if y and y_hat cover different dates, or (if positive) contain values <= 0."""
    if not y.index.equals(y_hat.index):
        raise ValueError("y and y_hat must have the same dates")
    if positive:
        n_bad = int((y <= 0).sum() + (y_hat <= 0).sum())
        if n_bad > 0:
            raise ValueError(f"{n_bad} non-positive values in y or y_hat")


def mse(y: pd.Series, y_hat: pd.Series) -> float:
    """Mean squared error between realized and forecast variance."""
    _check_inputs(y, y_hat)
    return ((y - y_hat) ** 2).mean()


def qlike(y: pd.Series, y_hat: pd.Series) -> float:
    """QLIKE loss: mean of y/y_hat - ln(y/y_hat) - 1 (0 for a perfect forecast)."""
    _check_inputs(y, y_hat, positive=True)
    return (y / y_hat - np.log(y / y_hat)).mean() - 1