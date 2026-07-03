"""
Step 14 — Add Transaction Costs

Input:
    results/raw_signal_weights.csv

Optional fallback inputs for actual returns:
    results/walk_forward_predictions.csv
    data/processed/master_modeling_dataset.parquet

Output:
    results/backtest_results.csv

Transaction-cost scenarios:
    low_cost    = 5 bps per one-way trade
    medium_cost = 10 bps per one-way trade
    high_cost   = 25 bps per one-way trade
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIGNAL_WEIGHTS_PATH = PROJECT_ROOT / "results" / "raw_signal_weights.csv"
WALK_FORWARD_PATH = PROJECT_ROOT / "results" / "walk_forward_predictions.csv"
MASTER_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "master_modeling_dataset.parquet"

OUTPUT_PATH = PROJECT_ROOT / "results" / "backtest_results.csv"


COST_SCENARIOS_BPS = {
    "low_cost": 5,
    "medium_cost": 10,
    "high_cost": 25,
}


def _standardize_dates(df: pd.DataFrame, date_col: str = "month_end_date") -> pd.DataFrame:
    """Convert month_end_date to pandas datetime and sort."""
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if df[date_col].isna().any():
        bad_rows = df[df[date_col].isna()].head()
        raise ValueError(f"Could not parse some {date_col} values:\n{bad_rows}")

    return df


def load_signal_weights() -> pd.DataFrame:
    """Load Step 13 portfolio signal weights."""
    if not SIGNAL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {SIGNAL_WEIGHTS_PATH}\n"
            "Run Step 13 first:\n"
            "python src\\backtest\\signal_to_portfolio.py"
        )

    df = pd.read_csv(SIGNAL_WEIGHTS_PATH)
    df = _standardize_dates(df)

    required_cols = {"month_end_date", "ticker", "weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Signal weights file is missing required columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    if "selected_flag" in df.columns:
        selected_numeric = pd.to_numeric(df["selected_flag"], errors="coerce")
        selected_bool = selected_numeric.fillna(0).astype(int).eq(1)
        df = df[selected_bool | df["weight"].ne(0)].copy()
    else:
        df = df[df["weight"].ne(0)].copy()

    if df.empty:
        raise ValueError("No selected or nonzero-weight portfolio rows found in raw_signal_weights.csv.")

    if "portfolio_rule" not in df.columns:
        df["portfolio_rule"] = "unknown_rule"

    df = df.sort_values(["month_end_date", "ticker"]).reset_index(drop=True)
    return df


def load_actual_returns_from_walk_forward() -> pd.DataFrame | None:
    """Load actual_next_return from walk-forward predictions if available."""
    if not WALK_FORWARD_PATH.exists():
        return None

    df = pd.read_csv(WALK_FORWARD_PATH)
    df = _standardize_dates(df)

    required_cols = {"month_end_date", "ticker", "actual_next_return"}
    if not required_cols.issubset(df.columns):
        return None

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["actual_next_return"] = pd.to_numeric(df["actual_next_return"], errors="coerce")

    # If multiple model rows exist for the same month/ticker, actual return should be identical.
    # We aggregate defensively to avoid duplicate rows after merging.
    df = (
        df.groupby(["month_end_date", "ticker"], as_index=False)["actual_next_return"]
        .mean()
        .sort_values(["month_end_date", "ticker"])
    )

    return df


def load_actual_returns_from_master_dataset() -> pd.DataFrame | None:
    """Load next_1m_return from the master modeling dataset if available."""
    if not MASTER_DATASET_PATH.exists():
        return None

    df = pd.read_parquet(MASTER_DATASET_PATH)
    df = _standardize_dates(df)

    required_cols = {"month_end_date", "ticker", "next_1m_return"}
    if not required_cols.issubset(df.columns):
        return None

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["actual_next_return"] = pd.to_numeric(df["next_1m_return"], errors="coerce")

    df = (
        df.groupby(["month_end_date", "ticker"], as_index=False)["actual_next_return"]
        .mean()
        .sort_values(["month_end_date", "ticker"])
    )

    return df


def attach_actual_returns(weights: pd.DataFrame) -> pd.DataFrame:
    """Ensure signal weights contain actual_next_return."""
    weights = weights.copy()

    if "actual_next_return" in weights.columns:
        weights["actual_next_return"] = pd.to_numeric(weights["actual_next_return"], errors="coerce")
        if weights["actual_next_return"].notna().all():
            return weights

    returns = load_actual_returns_from_walk_forward()

    if returns is None:
        returns = load_actual_returns_from_master_dataset()

    if returns is None:
        raise FileNotFoundError(
            "Could not find actual returns.\n\n"
            "The script needs one of the following:\n"
            "1. actual_next_return inside results/raw_signal_weights.csv, or\n"
            "2. results/walk_forward_predictions.csv with actual_next_return, or\n"
            "3. data/processed/master_modeling_dataset.parquet with next_1m_return.\n\n"
            "Recommended fix: rerun Step 12 and Step 13."
        )

    if "actual_next_return" in weights.columns:
        weights = weights.drop(columns=["actual_next_return"])

    merged = weights.merge(
        returns,
        on=["month_end_date", "ticker"],
        how="left",
        validate="many_to_one",
    )

    missing_returns = merged["actual_next_return"].isna()
    if missing_returns.any():
        examples = merged.loc[
            missing_returns, ["month_end_date", "ticker", "weight"]
        ].head(20)

        raise ValueError(
            "Some selected portfolio rows are missing actual_next_return.\n"
            "Examples:\n"
            f"{examples}\n\n"
            "Recommended fix: rerun Step 12 and Step 13 so dates and tickers match."
        )

    return merged


def calculate_monthly_gross_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly gross portfolio returns before transaction costs."""
    df = df.copy()
    df["weighted_return"] = df["weight"] * df["actual_next_return"]

    monthly = (
        df.groupby("month_end_date", as_index=False)
        .agg(
            gross_return=("weighted_return", "sum"),
            total_weight=("weight", "sum"),
            n_holdings=("ticker", "nunique"),
            portfolio_rule=("portfolio_rule", "first"),
        )
        .sort_values("month_end_date")
        .reset_index(drop=True)
    )

    return monthly


def calculate_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate one-way traded notional turnover.

    Formula:
        turnover_t = sum_i |weight_i,t - weight_i,t-1|

    The first month is measured from a zero-weight portfolio.
    Example:
        If the first portfolio is fully invested with total weight 1.0,
        first-month turnover is 1.0.
    """
    weights = (
        df.pivot_table(
            index="month_end_date",
            columns="ticker",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )

    previous_weights = weights.shift(1).fillna(0.0)
    turnover = (weights - previous_weights).abs().sum(axis=1)

    turnover_df = turnover.rename("turnover").reset_index()
    return turnover_df


def apply_transaction_costs(monthly: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    """Create one row per month per cost scenario."""
    base = monthly.merge(turnover, on="month_end_date", how="left")
    base["turnover"] = base["turnover"].fillna(0.0)

    scenario_frames = []

    for scenario_name, bps in COST_SCENARIOS_BPS.items():
        cost_rate = bps / 10_000.0

        temp = base.copy()
        temp["cost_scenario"] = scenario_name
        temp["cost_bps"] = bps
        temp["transaction_cost"] = temp["turnover"] * cost_rate
        temp["net_return"] = temp["gross_return"] - temp["transaction_cost"]

        temp["cumulative_gross_return"] = (1.0 + temp["gross_return"]).cumprod() - 1.0
        temp["cumulative_net_return"] = (1.0 + temp["net_return"]).cumprod() - 1.0

        scenario_frames.append(temp)

    results = pd.concat(scenario_frames, ignore_index=True)

    ordered_cols = [
        "month_end_date",
        "cost_scenario",
        "cost_bps",
        "portfolio_rule",
        "n_holdings",
        "total_weight",
        "turnover",
        "gross_return",
        "transaction_cost",
        "net_return",
        "cumulative_gross_return",
        "cumulative_net_return",
    ]

    results = results[ordered_cols].sort_values(
        ["month_end_date", "cost_bps"]
    ).reset_index(drop=True)

    return results


def print_summary(results: pd.DataFrame) -> None:
    """Print a compact terminal summary."""
    summary = (
        results.groupby("cost_scenario", as_index=False)
        .agg(
            months=("month_end_date", "nunique"),
            avg_turnover=("turnover", "mean"),
            avg_gross_return=("gross_return", "mean"),
            avg_transaction_cost=("transaction_cost", "mean"),
            avg_net_return=("net_return", "mean"),
            final_cumulative_net_return=("cumulative_net_return", "last"),
        )
        .sort_values("cost_scenario")
    )

    print("\nTransaction-cost backtest complete.")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("\nScenario summary:")
    print(summary.to_string(index=False))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    weights = load_signal_weights()
    weights = attach_actual_returns(weights)

    monthly = calculate_monthly_gross_returns(weights)
    turnover = calculate_turnover(weights)

    results = apply_transaction_costs(monthly, turnover)

    results.to_csv(OUTPUT_PATH, index=False)

    print_summary(results)


if __name__ == "__main__":
    main()
