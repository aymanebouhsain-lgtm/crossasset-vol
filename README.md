# crossasset-vol

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

## Roadmap

1. Monte Carlo validation of the estimators on simulated price paths
2. HAR-RV baseline model
3. Walk-forward out-of-sample evaluation using the QLIKE loss
4. GARCH-family models
5. LightGBM model
6. Daily live forecasting job

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

The notebooks add `../src` to the import path, so start Jupyter from the
repository root or the `notebooks/` folder.

## Repository structure

```
crossasset-vol/
├── src/volforecast/
│   ├── data.py           # download and load daily OHLCV data
│   └── estimators.py     # daily variance estimators and annualization
├── notebooks/
│   ├── 01_explore.ipynb    # first look at the data
│   └── 02_estimators.ipynb # comparing estimators on SPY
├── data/                 # Parquet files created by data.py (not tracked)
├── requirements.txt
└── LICENSE
```

## License

MIT. See [LICENSE](LICENSE).
