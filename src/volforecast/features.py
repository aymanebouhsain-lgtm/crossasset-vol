import pandas as pd
from volforecast.estimators import daily_variance

def har_features(v: pd.Series) -> pd.DataFrame:
    features = {
        'rv_d': v,
        'rv_w': v.rolling(window=5).mean(),
        'rv_m': v.rolling(window=22).mean()
    }
    return pd.DataFrame(features)

def target(v: pd.Series, horizon: int = 5) -> pd.Series:
    """Sum of daily variance over the next `horizon` days: v_{t+1} + ... + v_{t+horizon}."""
    # Reverse time so "future" becomes "past": shift(1) brings v_{t+1} to row t,
    # then a backward rolling sum adds v_{t+1}..v_{t+horizon}. Reverse back.
    return v.iloc[::-1].shift(1).rolling(window=horizon).sum().iloc[::-1].rename('target')

def make_dataset(df: pd.DataFrame, method: str = "gk", horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    v = daily_variance(df, method=method, include_overnight=True)
    x = har_features(v)
    y = target(v, horizon=horizon)
    x.dropna(inplace=True)
    y.dropna(inplace=True)
    temp = pd.concat([x, y], axis=1).dropna()
    return temp[x.columns], temp[y.name]




