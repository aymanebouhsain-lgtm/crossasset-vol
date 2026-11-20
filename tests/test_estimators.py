"""Monte Carlo tests of the simulator and the daily variance estimators.

All tests use a fixed seed, so they are deterministic. Tolerances follow from
two sources of error:

- Sampling error. For close-to-close, r^2 has mean sigma^2 and variance
  2 sigma^4, so the relative standard error of the mean over N days is
  sqrt(2 / N) ~ 1% for N = 20,000. A 5% tolerance is about 5 standard errors.
- Discretization bias. With M intraday steps, the observed high and low miss
  the true extremes by ~ 0.5826 sigma / sqrt(M) (Broadie-Glasserman-Kou), so
  the range estimators are biased low by an amount of order 1 / sqrt(M).
"""

import numpy as np
import pandas as pd
import pytest

from volforecast.estimators import close_to_close, garman_klass, parkinson
from volforecast.simulate import simulate_ohlc

SIGMA = 0.01
N_DAYS = 20_000
M = 500
SEED = 0


@pytest.fixture(scope="module")
def ohlc() -> pd.DataFrame:
    return simulate_ohlc(n_days=N_DAYS, steps_per_day=M, sigma=SIGMA, seed=SEED)


def bias_ratio(var: pd.Series) -> float:
    return var.dropna().mean() / SIGMA**2


# --- simulator ---------------------------------------------------------------


def test_open_equals_previous_close(ohlc):
    np.testing.assert_allclose(ohlc["Open"].iloc[1:], ohlc["Close"].iloc[:-1])


def test_high_low_bound_open_and_close(ohlc):
    assert (ohlc["High"] >= ohlc[["Open", "Close"]].max(axis=1)).all()
    assert (ohlc["Low"] <= ohlc[["Open", "Close"]].min(axis=1)).all()


# --- estimators --------------------------------------------------------------


def test_close_to_close_unbiased(ohlc):
    # Sampling error ~ sqrt(2 / N) ~ 1%; 5% is ~5 standard errors.
    assert bias_ratio(close_to_close(ohlc)) == pytest.approx(1.0, abs=0.05)


# NOTE: the task description was cut off after "Parkinson and"; this test is an
# assumed completion (range estimators biased low by order 1 / sqrt(M)).
@pytest.mark.parametrize("estimator", [parkinson, garman_klass])
def test_range_estimators_biased_low(ohlc, estimator):
    # Expected gap 1 - bias ratio ~ c / sqrt(M) with c ~ 1.2 (Parkinson) and
    # ~ 1.6 (Garman-Klass); at M = 500 that is ~5-9%, well above the ~1%
    # sampling error. The band [0.5, 3] / sqrt(M) checks the sign and the order
    # of magnitude without depending on the exact constant.
    gap = 1.0 - bias_ratio(estimator(ohlc))
    assert 0.5 / np.sqrt(M) < gap < 3.0 / np.sqrt(M)
