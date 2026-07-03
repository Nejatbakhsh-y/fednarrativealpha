from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKTEST_RESULTS_PATH = ROOT / "results" / "backtest_results.csv"


def read_backtest_or_skip() -> pd.DataFrame:
    if not BACKTEST_RESULTS_PATH.exists():
        pytest.skip(f"Required file not found: {BACKTEST_RESULTS_PATH}")
    df = pd.read_csv(BACKTEST_RESULTS_PATH)
    if df.empty:
        pytest.fail(f"File exists but is empty: {BACKTEST_RESULTS_PATH}")
    return df


def test_transaction_costs_reduce_gross_returns():
    df = read_backtest_or_skip()

    required_columns = {"gross_return", "transaction_cost", "net_return"}
    missing = required_columns - set(df.columns)
    assert not missing, f"Missing required backtest accounting columns: {sorted(missing)}"

    gross_return = pd.to_numeric(df["gross_return"], errors="coerce")
    transaction_cost = pd.to_numeric(df["transaction_cost"], errors="coerce")
    net_return = pd.to_numeric(df["net_return"], errors="coerce")

    assert gross_return.notna().all(), "gross_return contains nonnumeric values."
    assert transaction_cost.notna().all(), "transaction_cost contains nonnumeric values."
    assert net_return.notna().all(), "net_return contains nonnumeric values."

    assert (transaction_cost >= -1e-12).all(), (
        "Transaction costs must be nonnegative."
    )

    assert (net_return <= gross_return + 1e-12).all(), (
        "Transaction costs must reduce or leave unchanged gross returns."
    )

    expected_net_return = gross_return - transaction_cost

    assert np.allclose(
        net_return,
        expected_net_return,
        atol=1e-8,
        rtol=1e-6,
        equal_nan=False,
    ), "net_return must equal gross_return minus transaction_cost."


def test_backtest_summary_metrics_are_reasonable_when_present():
    df = read_backtest_or_skip()

    if "hit_rate" in df.columns:
        hit_rate = pd.to_numeric(df["hit_rate"], errors="coerce")
        assert hit_rate.dropna().between(0, 1).all(), (
            "hit_rate must be between 0 and 1."
        )

    if "monthly_turnover" in df.columns:
        monthly_turnover = pd.to_numeric(df["monthly_turnover"], errors="coerce")
        assert (monthly_turnover.dropna() >= -1e-12).all(), (
            "monthly_turnover must be nonnegative."
        )

    if "transaction_cost_drag" in df.columns:
        transaction_cost_drag = pd.to_numeric(
            df["transaction_cost_drag"],
            errors="coerce",
        )
        assert (transaction_cost_drag.dropna() >= -1e-12).all(), (
            "transaction_cost_drag must be nonnegative."
        )
