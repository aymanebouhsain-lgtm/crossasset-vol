# crossasset-vol

[![tests](https://github.com/aymanebouhsain-lgtm/crossasset-vol/actions/workflows/tests.yml/badge.svg)](https://github.com/aymanebouhsain-lgtm/crossasset-vol/actions/workflows/tests.yml)

Forecasting next-week realized volatility across five asset classes using daily
OHLC (open, high, low, close) data. The project builds daily variance estimates
from price ranges and returns, and will compare forecasting models for the
average volatility over the following five trading days on five liquid ETFs:
SPY, TLT, FXE, GLD and USO.

## Assets

| Ticker | Asset class | Instrument                          |
|--------|-------------|-------------------------------------|
| SPY    | Equities    | SPDR S&P 500 ETF Trust              |
| TLT    | Rates       | iShares 20+ Year Treasury Bond ETF  |
| FXE    | FX          | Invesco CurrencyShares Euro Trust   |
| GLD    | Gold        | SPDR Gold Shares                    |
| USO    | Oil         | United States Oil Fund              |

Daily data from January 2007 onwards, downloaded with `yfinance` (prices
adjusted for splits and dividends).

## Current status

- **Data pipeline** (`src/volforecast/data.py`): downloads daily OHLCV data
  for the five ETFs and stores it locally as Parquet files.
- **Volatility estimators** (`src/volforecast/estimators.py`): four daily
  variance estimators:
  - close-to-close (squared log return),
  - overnight (squared close-to-open log return),
  - Parkinson (high-low range),
  - Garman-Klass (high-low range and open-to-close return).

  The range-based estimators (Parkinson, Garman-Klass) can optionally add the
  overnight term, so that they cover the full close-to-close day.
- **Monte Carlo validation** (`src/volforecast/simulate.py`,
  `notebooks/03_monte_carlo.ipynb`): simulates daily OHLC bars from an intraday
  Brownian motion and measures the bias and efficiency of each estimator
  against the known true variance. Summary below.
- **HAR-RV baseline** (`src/volforecast/features.py`, `models.py`,
  `evaluate.py`, `notebooks/04_har.ipynb`): daily, weekly and monthly variance
  features, a target equal to the sum of variance over the next 5 trading days,
  the HAR model fitted by OLS, naive and mean benchmarks, and MSE and QLIKE
  losses. Trained on 2007-2016 and tested on 2017 onwards for all five ETFs.
  Summary below.

## Estimator validation (Monte Carlo)

Daily OHLC bars are simulated from a driftless Brownian motion in log price,
observed at M equally spaced points per day, with no overnight gap (20,000 days,
daily volatility 1%). The true daily variance is known, so each estimator can
be checked against it.

![Bias ratio and fair efficiency against M](figures/mc_bias_efficiency.png)

- Close-to-close is unbiased at every M (bias ratio 0.99 to 1.01).
- Parkinson and Garman-Klass are biased low, because the observed high and low
  miss the true extremes between observations. Their bias ratio is 0.57 / 0.41
  at M = 5 and still 0.95 / 0.94 at M = 1000.
- The bias shrinks like M^(-1/2), as predicted by Broadie, Glasserman and Kou
  (1997): the fitted log-log slopes are -0.479 (Parkinson) and -0.465
  (Garman-Klass), averaged over 5 seeds for M = 50 to 1000.
- Measured on squared coefficients of variation, the efficiency relative to
  close-to-close at M = 1000 is 4.8 (Parkinson) and 7.2 (Garman-Klass), below
  the continuous-path values 5.2 and 7.4. The plain variance ratio overstates
  the gain for biased estimators (17.5 for Garman-Klass at M = 5, against 2.9).

On real data, prices are also observed at discrete trades, so range-based
estimators will tend to understate volatility, more so for less liquid assets.

## HAR-RV baseline (out of sample)

The HAR model of Corsi (2009) regresses the sum of daily variance (Garman-Klass
plus the overnight term) over the next 5 trading days on its daily, weekly and
monthly averages. It is fitted by OLS on 2007-2016 and evaluated on 2017 to
November 2026. The last 5 training rows are dropped, since their targets
overlap the test period. The table shows HAR's loss divided by the loss of the
naive forecast (next 5 days = last 5 days), so a value below 1 means HAR does
better.

| Ticker | HAR MSE / naive | HAR QLIKE / naive |
|--------|-----------------|-------------------|
| SPY    | 0.82            | 0.79              |
| TLT    | 1.25            | 0.84              |
| FXE    | 0.88            | 0.64              |
| GLD    | 0.68            | 0.69              |
| USO    | 0.69            | 0.73              |

- HAR beats the naive forecast on QLIKE for all five assets (0.64 to 0.84).
- TLT is the exception on MSE. The loss is driven by March 2020, when realized
  volatility jumped to 70-90% and HAR, whose slopes sum to about 3.5 and pull
  the forecast toward the 2007-2016 average, forecast about 40-50%. 2020 is
  the only test year where HAR loses to naive on MSE (1.34). Excluding it, the
  ratio is 0.75.
- The monthly term carries most of the weight for every asset, and almost all
  of it for FXE (beta_m = 4.02 against 0.06 daily and 0.14 weekly). For SPY
  the coefficients imply a long-run volatility of about 21%.

## Roadmap

1. [x] Monte Carlo validation of the estimators on simulated price paths
2. [x] HAR-RV baseline model
3. [ ] Walk-forward out-of-sample evaluation using the QLIKE loss
4. [ ] GARCH-family models
5. [ ] LightGBM model
6. [ ] Daily live forecasting job

## Results

Coming soon.

## Setup

```bash
git clone https://github.com/aymanebouhsain-lgtm/crossasset-vol.git
cd crossasset-vol

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python src/volforecast/data.py   # downloads data into data/
```

Run the tests and the linter/formatter with:

```bash
pytest
ruff check .     # lint
ruff format .    # format
```

The test suite contains an import smoke test, Monte Carlo tests of the
simulator and estimators (fixed seed), and tests of the HAR features and
target. CI runs it on every push with
Python 3.12.

## Repository structure

```
crossasset-vol/
├── .github/workflows/
│   └── tests.yml           # CI: runs pytest on push
├── src/volforecast/
│   ├── __init__.py
│   ├── data.py             # download and load daily OHLCV data
│   ├── estimators.py       # daily variance estimators and annualization
│   ├── simulate.py         # simulated OHLC bars for Monte Carlo tests
│   ├── features.py         # HAR features and 5-day target
│   ├── models.py           # HAR, naive and mean models
│   └── evaluate.py         # MSE and QLIKE losses
├── notebooks/
│   ├── 01_explore.ipynb    # first look at the data
│   ├── 02_estimators.ipynb # comparing estimators on SPY
│   ├── 03_monte_carlo.ipynb # Monte Carlo validation of the estimators
│   └── 04_har.ipynb        # HAR-RV baseline on the five ETFs
├── tests/
│   ├── test_smoke.py       # placeholder import test
│   ├── test_estimators.py  # Monte Carlo tests of simulator and estimators
│   └── test_features.py    # tests of HAR features, target and dataset
├── figures/                # figures saved by the notebooks
├── data/                   # Parquet files created by data.py (not tracked)
├── pyproject.toml          # pytest and ruff configuration
├── requirements.txt
└── LICENSE
```

## License

MIT. See [LICENSE](LICENSE).
