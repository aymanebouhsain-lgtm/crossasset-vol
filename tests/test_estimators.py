"""Monte Carlo tests of the simulator and the daily variance estimators.

All tests use a fixed seed, so they are deterministic. Tolerances follow from
two sources of error:

- Sampling error. For close-to-close, r^2 has mean sigma^2 and variance
  2 sigma^4, so the relative standard error of the mean over N days is
  sqrt(2 / N) ~ 1% for N = 20,000. The range estimators are less noisy.
- Discretization bias. With M intraday steps, the observed high and low miss
  the true extremes by ~ 0.5826 sigma / sqrt(M) (Broadie-Glasserman-Kou), so
  the range estimators are biased low by an amount of order 1 / sqrt(M):
  about 7% (Parkinson) and 9% (Garman-Klass) at M = 500, 34% and 47% at M = 10.
"""

import numpy as np
import pandas as pd
import pytest

from volforecast.estimators import close_to_close, garman_klass, parkinson
from volforecast.simulate import simulate_ohlc

SIGMA = 0.01
N_DAYS = 20_000
SEED = 0

RANGE_ESTIMATORS = [parkinson, garman_klass]


def simulate(steps_per_day: int, s0: float = 1.0) -> pd.DataFrame:
    return simulate_ohlc(
        n_days=N_DAYS, steps_per_day=steps_per_day, sigma=SIGMA, seed=SEED, s0=s0
    )


@pytest.fixture(scope="module")
def ohlc_500() -> pd.DataFrame:
    return simulate(500)


@pytest.fixture(scope="module")
def ohlc_10() -> pd.DataFrame:
    return simulate(10)


def bias_ratio(var: pd.Series) -> float:
    """E[estimate] / sigma^2 over the simulated days (first-day NaN dropped)."""
    return var.dropna().mean() / SIGMA**2


def fair_efficiency(var: pd.Series, var_cc: pd.Series) -> float:
    """Ratio of squared coefficients of variation, cc over the estimator."""
    both = pd.concat([var, var_cc], axis=1).dropna()
    est, cc = both.iloc[:, 0], both.iloc[:, 1]
    return (cc.var() / cc.mean() ** 2) / (est.var() / est.mean() ** 2)


# --- simulator ---------------------------------------------------------------


def test_open_equals_previous_close(ohlc_500):
    np.testing.assert_allclose(
        ohlc_500["Open"].iloc[1:].to_numpy(), ohlc_500["Close"].iloc[:-1].to_numpy()
    )


def test_high_low_bound_open_and_close(ohlc_500):
    assert (ohlc_500["High"] >= ohlc_500[["Open", "Close"]].max(axis=1)).all()
    assert (ohlc_500["Low"] <= ohlc_500[["Open", "Close"]].min(axis=1)).all()


# --- estimators --------------------------------------------------------------


def test_close_to_close_unbiased(ohlc_500):
    # Sampling error ~ sqrt(2 / N) ~ 1%; 5% is ~5 standard errors.
    assert bias_ratio(close_to_close(ohlc_500)) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("estimator", RANGE_ESTIMATORS)
def test_range_estimators_biased_low(ohlc_500, estimator):
    # Expected bias ratio at M = 500: ~0.93 (Parkinson), ~0.91 (Garman-Klass).
    # The lower bound 0.88 leaves ~3% for sampling error and the constant in
    # the 1 / sqrt(M) law; the upper bound 1.0 checks the sign of the bias.
    assert 0.88 < bias_ratio(estimator(ohlc_500)) < 1.0


@pytest.mark.parametrize("estimator", RANGE_ESTIMATORS)
def test_bias_shrinks_with_finer_sampling(ohlc_10, ohlc_500, estimator):
    # The gap 1 - bias ratio scales like M^(-1/2): ~7x smaller at M = 500 than
    # at M = 10, far beyond the ~1% sampling error.
    gap_10 = 1.0 - bias_ratio(estimator(ohlc_10))
    gap_500 = 1.0 - bias_ratio(estimator(ohlc_500))
    assert gap_500 < gap_10


@pytest.mark.parametrize("estimator", RANGE_ESTIMATORS)
def test_fair_efficiency_above_three(ohlc_500, estimator):
    # Expected at M = 500: ~4.6 (Parkinson), ~6.9 (Garman-Klass), with roughly
    # +/-5% variation across seeds. 3 is a conservative floor.
    cc = close_to_close(ohlc_500)
    assert fair_efficiency(estimator(ohlc_500), cc) > 3.0


@pytest.mark.parametrize("estimator", [close_to_close, *RANGE_ESTIMATORS])
def test_estimators_invariant_to_price_level(estimator):
    # All estimators use log price ratios, so scaling the price by s0 must not
    # change them (up to floating-point rounding). A short path is enough.
    a = simulate_ohlc(n_days=500, steps_per_day=50, sigma=SIGMA, seed=SEED, s0=1.0)
    b = simulate_ohlc(n_days=500, steps_per_day=50, sigma=SIGMA, seed=SEED, s0=100.0)
    np.testing.assert_allclose(estimator(a), estimator(b), rtol=1e-9, atol=1e-15)
