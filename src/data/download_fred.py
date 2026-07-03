"""
Download FRED macroeconomic data and build monthly macro features.

Raw output:
    data/raw/fred_macro.parquet

Monthly feature output:
    data/interim/macro_features_monthly.parquet
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web


START_DATE = "2010-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

RAW_OUTPUT_PATH = Path("data/raw/fred_macro.parquet")
FEATURE_OUTPUT_PATH = Path("data/interim/macro_features_monthly.parquet")

FRED_SERIES = {
    "FEDFUNDS": "Federal funds rate",
    "DGS2": "2-year Treasury yield",
    "DGS10": "10-year Treasury yield",
    "T10Y2Y": "10-year minus 2-year Treasury spread",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "UNRATE": "Unemployment rate",
    "PAYEMS": "Nonfarm payrolls",
    "VIXCLS": "CBOE Volatility Index",
    "BAMLH0A0HYM2": "High-yield credit spread",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "BAA10Y": "Fallback credit spread proxy: Moody's Baa corporate spread over 10-year Treasury",
}

REQUIRED_USER_SERIES = {
    "FEDFUNDS",
    "DGS2",
    "DGS10",
    "T10Y2Y",
    "CPIAUCSL",
    "UNRATE",
    "PAYEMS",
    "VIXCLS",
    "BAMLH0A0HYM2",
    "NFCI",
}

LIMITED_HISTORY_ALLOWED = {"BAMLH0A0HYM2"}


def ensure_output_directories() -> None:
    """Create output directories if they do not already exist."""
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_fred_data() -> pd.DataFrame:
    """Download all required FRED series."""
    frames: list[pd.DataFrame] = []

    for series_id, description in FRED_SERIES.items():
        print(f"Downloading {series_id}: {description}")
        series = web.DataReader(series_id, "fred", START_DATE, END_DATE)
        frames.append(series)

    data = pd.concat(frames, axis=1, sort=True)
    data = data.sort_index()
    data.index.name = "date"

    return data


def validate_raw_data(data: pd.DataFrame) -> None:
    """Run basic quality checks on raw FRED data."""
    missing_required_columns = REQUIRED_USER_SERIES - set(data.columns)
    if missing_required_columns:
        raise ValueError(f"Missing required FRED columns: {sorted(missing_required_columns)}")

    if "BAA10Y" not in data.columns:
        raise ValueError("Missing fallback credit spread column: BAA10Y")

    if data.index.duplicated().any():
        raise ValueError("Duplicate date rows found in raw FRED data.")

    all_missing_columns = data.columns[data.isna().all()].tolist()
    if all_missing_columns:
        raise ValueError(f"These FRED series are completely missing: {all_missing_columns}")

    for column in data.columns:
        first_valid = data[column].first_valid_index()
        last_valid = data[column].last_valid_index()

        if first_valid is None or last_valid is None:
            raise ValueError(f"No valid observations for {column}")

        coverage_years = (last_valid - first_valid).days / 365.25

        if coverage_years < 10 and column not in LIMITED_HISTORY_ALLOWED:
            raise ValueError(
                f"{column} has less than 10 years of valid data: {coverage_years:.2f} years"
            )

        if coverage_years < 10 and column in LIMITED_HISTORY_ALLOWED:
            print(
                f"Warning: {column} has only {coverage_years:.2f} years of data. "
                "This is allowed because FRED currently limits this series history."
            )


def choose_credit_spread_series(monthly: pd.DataFrame) -> pd.Series:
    """
    Select the credit-spread feature source.

    BAMLH0A0HYM2 is the preferred high-yield spread series, but FRED currently
    provides only limited history for it. If it has less than 10 years of valid
    data, use BAA10Y as the long-history credit-spread proxy.
    """
    baml_first = monthly["BAMLH0A0HYM2"].first_valid_index()
    baml_last = monthly["BAMLH0A0HYM2"].last_valid_index()

    if baml_first is not None and baml_last is not None:
        baml_coverage_years = (baml_last - baml_first).days / 365.25
    else:
        baml_coverage_years = 0.0

    if baml_coverage_years >= 10:
        print("Using BAMLH0A0HYM2 as credit_spread_level.")
        return monthly["BAMLH0A0HYM2"]

    print("Using BAA10Y as credit_spread_level because BAMLH0A0HYM2 has limited history.")
    return monthly["BAA10Y"]


def build_monthly_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw mixed-frequency FRED data into month-end macro features.
    """
    monthly = raw_data.resample("ME").last().ffill()

    credit_spread = choose_credit_spread_series(monthly)

    features = pd.DataFrame(index=monthly.index)
    features.index.name = "date"

    features["rate_level"] = monthly["FEDFUNDS"]
    features["rate_change_1m"] = monthly["FEDFUNDS"].diff(1)
    features["rate_change_3m"] = monthly["FEDFUNDS"].diff(3)

    features["yield_curve_slope"] = monthly["T10Y2Y"].combine_first(
        monthly["DGS10"] - monthly["DGS2"]
    )

    features["inflation_momentum"] = (
        ((monthly["CPIAUCSL"] / monthly["CPIAUCSL"].shift(3)) ** 4) - 1
    ) * 100

    features["unemployment_change"] = monthly["UNRATE"].diff(1)

    features["credit_spread_level"] = credit_spread
    features["credit_spread_change"] = credit_spread.diff(1)

    features["vix_level"] = monthly["VIXCLS"]
    features["vix_change"] = monthly["VIXCLS"].diff(1)

    features["financial_conditions_level"] = monthly["NFCI"]
    features["financial_conditions_change"] = monthly["NFCI"].diff(1)

    features = features.dropna().reset_index()

    return features


def validate_monthly_features(features: pd.DataFrame) -> None:
    """Run quality checks on monthly feature output."""
    required_columns = [
        "date",
        "rate_level",
        "rate_change_1m",
        "rate_change_3m",
        "yield_curve_slope",
        "inflation_momentum",
        "unemployment_change",
        "credit_spread_level",
        "credit_spread_change",
        "vix_level",
        "vix_change",
        "financial_conditions_level",
        "financial_conditions_change",
    ]

    missing_columns = set(required_columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"Missing monthly feature columns: {sorted(missing_columns)}")

    if features.empty:
        raise ValueError("Monthly feature dataset is empty.")

    if features["date"].duplicated().any():
        raise ValueError("Duplicate monthly dates found in macro features.")

    feature_columns = [column for column in required_columns if column != "date"]
    if features[feature_columns].isna().any().any():
        raise ValueError("Monthly macro features contain missing values.")

    if len(features) < 120:
        raise ValueError(f"Expected at least 120 monthly rows, but found only {len(features)}.")


def main() -> None:
    """Run full FRED ingestion and monthly feature pipeline."""
    ensure_output_directories()

    raw_data = download_fred_data()
    validate_raw_data(raw_data)

    raw_data.reset_index().to_parquet(RAW_OUTPUT_PATH, index=False)
    print(f"Saved raw FRED data to: {RAW_OUTPUT_PATH}")

    monthly_features = build_monthly_features(raw_data)
    validate_monthly_features(monthly_features)

    monthly_features.to_parquet(FEATURE_OUTPUT_PATH, index=False)
    print(f"Saved monthly macro features to: {FEATURE_OUTPUT_PATH}")

    print("\nRaw data shape:", raw_data.shape)
    print("Monthly feature shape:", monthly_features.shape)
    print("\nMonthly feature preview:")
    print(monthly_features.tail())


if __name__ == "__main__":
    main()
