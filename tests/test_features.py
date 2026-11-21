"""Tests of the HAR features, the forward-looking target and make_dataset."""

import numpy as np
import pandas as pd
import pytest

from volforecast.features import har_features, make_dataset, target
from volforecast.simulate import simulate_ohlc


@pytest.fixture
def v() -> pd.Series:
    """v = 1, 2, ..., 30 on business days, so every window sum is easy to check."""
    index = pd.bdate_range("2020-01-01", periods=30)
    return pd.Series(np.arange(1.0, 31.0), index=index)


def test_target_sums_next_five_days(v):
    y = target(v, horizon=5)
    assert y.iloc[0] == 2 + 3 + 4 + 5 + 6  # = 20, today's value excluded
    assert y.iloc[-5:].isna().all()  # not enough future days
    assert y.iloc[:-5].notna().all()


def test_har_features(v):
    x = har_features(v)
    assert x["rv_w"].iloc[4] == np.mean([1, 2, 3, 4, 5])  # = 3
    assert x["rv_w"].iloc[:4].isna().all()
    assert x["rv_m"].first_valid_index() == v.index[21]  # 22-day window
    pd.testing.assert_series_equal(x["rv_d"], v, check_names=False)


def test_make_dataset_aligned_and_complete():
    df = simulate_ohlc(n_days=300, steps_per_day=50, sigma=0.01, seed=0)
    X, y = make_dataset(df)
    assert X.index.equals(y.index)
    assert len(X) > 0
    assert not X.isna().any().any()
    assert not y.isna().any()
