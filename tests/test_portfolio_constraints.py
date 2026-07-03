from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

PORTFOLIO_WEIGHT_CANDIDATES = [
    ROOT / "results" / "portfolio_weights.csv",
    ROOT / "results" / "raw_signal_weights.csv",
]

MAX_SINGLE_ETF_WEIGHT = 0.35
MIN_SELECTED_ETF_WEIGHT = 0.05


def find_weights_file_or_skip() -> Path:
    for path in PORTFOLIO_WEIGHT_CANDIDATES:
        if path.exists():
            return path

    pytest.skip(
        "No portfolio weight file found. Expected one of: "
        + ", ".join(str(path) for path in PORTFOLIO_WEIGHT_CANDIDATES)
    )


def read_weights_or_skip() -> pd.DataFrame:
    path = find_weights_file_or_skip()
    df = pd.read_csv(path)

    if df.empty:
        pytest.fail(f"File exists but is empty: {path}")

    return df


def get_date_column(df: pd.DataFrame) -> str:
    candidate_columns = [
        "month_end_date",
        "rebalance_date",
        "date",
    ]

    for column in candidate_columns:
        if column in df.columns:
            return column

    pytest.fail(
        "Portfolio weights file must contain one of these date columns: "
        f"{candidate_columns}. Available columns: {list(df.columns)}"
    )


def get_weight_column(df: pd.DataFrame) -> str:
    candidate_columns = [
        "weight",
        "optimized_weight",
        "portfolio_weight",
        "target_weight",
        "allocation",
    ]

    for column in candidate_columns:
        if column in df.columns:
            return column

    pytest.fail(
        "Portfolio weights file must contain a portfolio weight column. "
        f"Expected one of: {candidate_columns}. "
        f"Available columns: {list(df.columns)}"
    )


def clean_weights_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    date_column = get_date_column(df)
    weight_column = get_weight_column(df)

    cleaned = df.copy()
    cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")
    cleaned[weight_column] = pd.to_numeric(cleaned[weight_column], errors="coerce")

    assert cleaned[date_column].notna().all(), (
        f"{date_column} contains invalid dates."
    )

    assert cleaned[weight_column].notna().all(), (
        f"{weight_column} contains nonnumeric values."
    )

    cleaned = cleaned.rename(
        columns={
            date_column: "test_date_column",
            weight_column: "test_weight_column",
        }
    )

    return cleaned


def test_portfolio_weights_sum_to_one_each_month():
    df = clean_weights_dataframe(read_weights_or_skip())

    monthly_weight_sum = df.groupby("test_date_column")["test_weight_column"].sum()

    bad_months = monthly_weight_sum[
        ~np.isclose(monthly_weight_sum, 1.0, atol=1e-6, rtol=1e-6)
    ]

    assert bad_months.empty, (
        "Portfolio weights must sum to 1 for each month. "
        f"Bad months:\n{bad_months.head(10)}"
    )


def test_portfolio_is_long_only():
    df = clean_weights_dataframe(read_weights_or_skip())

    assert (df["test_weight_column"] >= -1e-12).all(), (
        "Long-only portfolio cannot contain negative weights."
    )


def test_no_single_etf_weight_exceeds_maximum_allowed_weight():
    df = clean_weights_dataframe(read_weights_or_skip())

    max_observed_weight = df["test_weight_column"].max()

    assert max_observed_weight <= MAX_SINGLE_ETF_WEIGHT + 1e-8, (
        f"Single ETF weight exceeds maximum allowed weight. "
        f"Observed max={max_observed_weight:.6f}, "
        f"allowed max={MAX_SINGLE_ETF_WEIGHT:.6f}."
    )


def test_selected_etf_weights_respect_minimum_when_present():
    df = clean_weights_dataframe(read_weights_or_skip())

    if "selected_flag" in df.columns:
        selected = df[df["selected_flag"].astype(bool)].copy()
    else:
        selected = df[df["test_weight_column"] > 1e-12].copy()

    if selected.empty:
        pytest.fail("No selected ETFs found in the portfolio weights file.")

    positive_selected_weights = selected["test_weight_column"][
        selected["test_weight_column"] > 1e-12
    ]

    if positive_selected_weights.empty:
        pytest.fail("Selected ETFs exist, but all selected weights are zero.")

    assert (positive_selected_weights >= MIN_SELECTED_ETF_WEIGHT - 1e-8).all(), (
        f"Selected ETF weights must be at least {MIN_SELECTED_ETF_WEIGHT:.2%}."
    )
