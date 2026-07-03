"""
Build monthly ETF price-based features.

Input:
    data/raw/etf_prices.parquet

Output:
    data/interim/price_features_monthly.parquet

This script creates month-end price, momentum, volatility, drawdown,
liquidity, trend, and next-month target variables for each ETF.

Important:
    All feature columns are computed using only information available
    on or before the feature month-end date. The next-month return target
    is shifted forward only after features are built.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "etf_prices.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "price_features_monthly.parquet"

TRADING_DAYS_PER_YEAR = 252


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )
    return df


def load_price_data(path: Path) -> pd.DataFrame:
    """Load and validate raw ETF price data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\nRun src/data/download_prices.py first."
        )

    df = pd.read_parquet(path)
    df = clean_column_names(df)

    required_columns = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[list(required_columns)].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"],
        keep="last",
    )

    if (df[["open", "high", "low", "close", "adj_close"]] < 0).any().any():
        raise ValueError("Negative price values found in raw price data.")

    return df


def max_drawdown(values: np.ndarray) -> float:
    """
    Compute maximum drawdown over a rolling price window.

    Returns a negative number, for example -0.25 for a 25% drawdown.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0

    return float(np.min(drawdowns))


def build_features_for_one_ticker(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """Build daily rolling features and convert them to month-end observations."""
    df_ticker = df_ticker.sort_values("date").copy()
    ticker = df_ticker["ticker"].iloc[0]

    df_ticker = df_ticker.set_index("date")

    df_ticker["daily_return"] = df_ticker["adj_close"].pct_change()

    # Daily rolling volatility features, annualized.
    df_ticker["realized_vol_21d"] = df_ticker["daily_return"].rolling(
        window=21, min_periods=15
    ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    df_ticker["realized_vol_63d"] = df_ticker["daily_return"].rolling(
        window=63, min_periods=45
    ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    df_ticker["realized_vol_126d"] = df_ticker["daily_return"].rolling(
        window=126, min_periods=90
    ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Six-month drawdown based on roughly 126 trading days.
    df_ticker["max_drawdown_6m"] = (
        df_ticker["adj_close"].rolling(window=126, min_periods=90).apply(max_drawdown, raw=True)
    )

    # Liquidity features.
    df_ticker["avg_volume_21d"] = df_ticker["volume"].rolling(window=21, min_periods=15).mean()

    df_ticker["dollar_volume"] = (
        (df_ticker["adj_close"] * df_ticker["volume"]).rolling(window=21, min_periods=15).mean()
    )

    # Trend feature.
    df_ticker["ma_200d"] = df_ticker["adj_close"].rolling(window=200, min_periods=150).mean()

    df_ticker["price_above_200d_ma"] = np.where(
        df_ticker["ma_200d"].notna(),
        (df_ticker["adj_close"] > df_ticker["ma_200d"]).astype(int),
        np.nan,
    )

    # Select the last trading observation in each calendar month.
    month_end = df_ticker.groupby(df_ticker.index.to_period("M")).tail(1).copy()

    month_end["date"] = month_end.index.to_period("M").to_timestamp("M")
    month_end["ticker"] = ticker

    # Monthly return features based on month-end adjusted close.
    month_end["return_1m"] = month_end["adj_close"].pct_change(1)
    month_end["return_3m"] = month_end["adj_close"].pct_change(3)
    month_end["return_6m"] = month_end["adj_close"].pct_change(6)
    month_end["return_12m"] = month_end["adj_close"].pct_change(12)

    month_end["momentum_12m_minus_1m"] = month_end["return_12m"] - month_end["return_1m"]

    month_end["volume_change_3m"] = month_end["avg_volume_21d"].pct_change(3)

    # Next-month target. This is intentionally forward-looking and should
    # only be used as the prediction target, never as an input feature.
    month_end["next_1m_return"] = month_end["adj_close"].shift(-1) / month_end["adj_close"] - 1.0

    output_columns = [
        "date",
        "ticker",
        "return_1m",
        "return_3m",
        "return_6m",
        "return_12m",
        "momentum_12m_minus_1m",
        "realized_vol_21d",
        "realized_vol_63d",
        "realized_vol_126d",
        "max_drawdown_6m",
        "volume_change_3m",
        "dollar_volume",
        "price_above_200d_ma",
        "next_1m_return",
    ]

    return month_end[output_columns].reset_index(drop=True)


def add_cross_sectional_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional next-month ranking targets.

    next_1m_rank:
        Rank 1 means the highest next-month return in that month.

    top_tercile_next_month:
        1 if the ETF is in the top third of next-month returns for that month,
        otherwise 0.
    """
    df = df.copy()

    df["next_1m_rank"] = df.groupby("date")["next_1m_return"].rank(
        method="first",
        ascending=False,
    )

    valid_count = df.groupby("date")["next_1m_return"].transform("count")
    top_cutoff = np.ceil(valid_count / 3)

    df["top_tercile_next_month"] = np.where(
        df["next_1m_return"].notna(),
        (df["next_1m_rank"] <= top_cutoff).astype(int),
        np.nan,
    )

    return df


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build monthly price features for all ETFs."""
    feature_frames = []

    for ticker, df_ticker in prices.groupby("ticker", sort=True):
        ticker_features = build_features_for_one_ticker(df_ticker)
        feature_frames.append(ticker_features)

    features = pd.concat(feature_frames, ignore_index=True)
    features = add_cross_sectional_targets(features)

    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    return features


def run_quality_checks(features: pd.DataFrame) -> None:
    """Run basic output checks."""
    duplicate_count = features.duplicated(subset=["date", "ticker"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Duplicate date-ticker rows found: {duplicate_count}")

    required_columns = [
        "date",
        "ticker",
        "return_1m",
        "return_3m",
        "return_6m",
        "return_12m",
        "momentum_12m_minus_1m",
        "realized_vol_21d",
        "realized_vol_63d",
        "realized_vol_126d",
        "max_drawdown_6m",
        "volume_change_3m",
        "dollar_volume",
        "price_above_200d_ma",
        "next_1m_return",
        "next_1m_rank",
        "top_tercile_next_month",
    ]

    missing = set(required_columns).difference(features.columns)
    if missing:
        raise ValueError(f"Missing output columns: {sorted(missing)}")

    if features.empty:
        raise ValueError("Feature output is empty.")

    ticker_count = features["ticker"].nunique()
    date_count = features["date"].nunique()

    print("Price feature quality checks passed.")
    print(f"Rows: {len(features):,}")
    print(f"Tickers: {ticker_count}")
    print(f"Monthly dates: {date_count}")
    print(f"Date range: {features['date'].min().date()} to {features['date'].max().date()}")
    print(f"Output path: {OUTPUT_PATH}")


def main() -> None:
    """Main script entry point."""
    prices = load_price_data(INPUT_PATH)
    features = build_price_features(prices)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    run_quality_checks(features)


if __name__ == "__main__":
    main()
