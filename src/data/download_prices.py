"""
Download daily ETF price data from Yahoo Finance using yfinance.

Output:
    data/raw/etf_prices.parquet

Fields:
    date, ticker, open, high, low, close, adj_close, volume
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "VNQ",
    "UUP",
]

START_DATE = "2010-01-01"

REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def get_project_root() -> Path:
    """Return the project root from src/data/download_prices.py."""
    return Path(__file__).resolve().parents[2]


def download_raw_prices() -> pd.DataFrame:
    """
    Download raw daily ETF prices from Yahoo Finance.

    Note:
        yfinance treats the end date as exclusive, so this uses tomorrow's date
        to include today's data if available.
    """
    end_date = (date.today() + timedelta(days=1)).isoformat()

    raw = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=end_date,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("No ETF price data was downloaded. Check internet connection or yfinance.")

    return raw


def convert_to_long_format(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance output into a clean long-format DataFrame."""
    frames: list[pd.DataFrame] = []

    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns because multiple tickers were requested.")

    level_0 = set(raw.columns.get_level_values(0))
    level_1 = set(raw.columns.get_level_values(1))

    for ticker in TICKERS:
        if ticker in level_0:
            ticker_df = raw[ticker].copy()
        elif ticker in level_1:
            ticker_df = raw.xs(ticker, level=1, axis=1).copy()
        else:
            raise ValueError(f"Ticker {ticker} was not found in downloaded data.")

        ticker_df = ticker_df.reset_index()
        ticker_df["ticker"] = ticker

        frames.append(ticker_df)

    prices = pd.concat(frames, ignore_index=True)

    prices = prices.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(prices.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prices = prices[REQUIRED_COLUMNS].copy()

    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices["ticker"] = prices["ticker"].astype(str)

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prices


def run_quality_checks(prices: pd.DataFrame) -> None:
    """Run required data quality checks before saving."""

    duplicate_count = prices.duplicated(subset=["ticker", "date"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate ticker-date rows.")

    price_columns = ["open", "high", "low", "close", "adj_close"]
    negative_price_count = (prices[price_columns] < 0).sum().sum()
    if negative_price_count > 0:
        raise ValueError(f"Found {negative_price_count} negative price values.")

    missing_tickers = sorted(set(TICKERS) - set(prices["ticker"].unique()))
    if missing_tickers:
        raise ValueError(f"Missing tickers in final dataset: {missing_tickers}")

    for ticker in TICKERS:
        ticker_data = prices.loc[prices["ticker"] == ticker].copy()

        if ticker_data.empty:
            raise ValueError(f"No data found for ticker {ticker}.")

        first_valid_index = ticker_data["adj_close"].first_valid_index()

        if first_valid_index is None:
            raise ValueError(f"No valid adjusted close values found for ticker {ticker}.")

        after_first_valid = ticker_data.loc[first_valid_index:]

        missing_adj_close = after_first_valid["adj_close"].isna().sum()
        if missing_adj_close > 0:
            raise ValueError(
                f"Ticker {ticker} has {missing_adj_close} missing adjusted close values "
                "after the first valid observation."
            )

        min_date = pd.to_datetime(ticker_data["date"].min())
        max_date = pd.to_datetime(ticker_data["date"].max())
        years_of_data = (max_date - min_date).days / 365.25

        if years_of_data < 10:
            raise ValueError(
                f"Ticker {ticker} has only {years_of_data:.2f} years of data. "
                "Minimum required is 10 years."
            )

    print("Quality checks passed:")
    print("- No duplicate ticker-date rows")
    print("- No negative prices")
    print("- No missing adjusted close after first valid observation")
    print("- Minimum 10 years of data per ETF")


def save_prices(prices: pd.DataFrame) -> Path:
    """Save cleaned ETF prices to Parquet."""
    project_root = get_project_root()
    output_path = project_root / "data" / "raw" / "etf_prices.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output_path, index=False)

    return output_path


def main() -> None:
    """Download, validate, and save ETF price data."""
    raw = download_raw_prices()
    prices = convert_to_long_format(raw)
    run_quality_checks(prices)

    output_path = save_prices(prices)

    print(f"Saved ETF price data to: {output_path}")
    print(f"Rows: {len(prices):,}")
    print(f"Tickers: {prices['ticker'].nunique()}")
    print(f"Date range: {prices['date'].min()} to {prices['date'].max()}")


if __name__ == "__main__":
    main()
