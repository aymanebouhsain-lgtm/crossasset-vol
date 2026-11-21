from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {"SPY": "equities", "TLT": "rates", "FXE": "fx", "GLD": "gold", "USO": "oil"}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download(start="2007-01-01"):
    """Download daily OHLCV data for all tickers since `start`.

    Saves one Parquet file per ticker in data/.
    """
    DATA_DIR.mkdir(exist_ok=True)
    for ticker in TICKERS:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.to_parquet(DATA_DIR / f"{ticker}.parquet")
        print(f"{ticker}: {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")


def load(ticker):
    """Load the saved daily OHLCV data for one ticker."""
    return pd.read_parquet(DATA_DIR / f"{ticker}.parquet")


if __name__ == "__main__":
    download()
