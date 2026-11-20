import numpy as np
import pandas as pd


def simulate_ohlc(
    n_days: int,
    steps_per_day: int,
    sigma: float,
    mu: float = 0.0,
    seed: int | None = None,
    s0: float = 1.0,
) -> pd.DataFrame:
    """
    Simulate daily OHLC bars from an intraday Brownian motion in log price.

    There is no overnight gap: each day's open equals the previous day's close
    (the first open is ``s0``). High and Low are taken over the open and every
    intraday step of the day.

    Parameters:
    - n_days: Number of days to simulate.
    - steps_per_day: Number of intraday steps per day.
    - sigma: Daily volatility of log returns.
    - mu: Daily drift of log returns (default 0.0).
    - seed: Random seed for reproducibility (default None).
    - s0: Initial price (default 1.0).

    Returns:
    - DataFrame with columns ['Open', 'High', 'Low', 'Close'], one row per day.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps_per_day

    log_returns = rng.normal(mu * dt, sigma * np.sqrt(dt), size=(n_days, steps_per_day))

    # Log path, one row per day: cumulative log price after each intraday step.
    log_path = np.cumsum(log_returns, axis=None).reshape(n_days, steps_per_day)
    log_close = log_path[:, -1]
    log_open = np.concatenate(([0.0], log_close[:-1]))

    # Extremes include the open, since the price starts there each day.
    log_high = np.maximum(log_path.max(axis=1), log_open)
    log_low = np.minimum(log_path.min(axis=1), log_open)

    return pd.DataFrame(
        s0 * np.exp(np.column_stack((log_open, log_high, log_low, log_close))),
        columns=["Open", "High", "Low", "Close"],
    )
