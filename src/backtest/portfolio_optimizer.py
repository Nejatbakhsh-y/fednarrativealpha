"""
Risk-constrained portfolio optimizer.

Step 15 requirement:
- Long-only weights
- Maximum single ETF weight: 35%
- Minimum selected ETF weight: 5%
- Maximum monthly turnover: 50%
- Target volatility recorded as a diagnostic assumption
- Output:
    results/portfolio_weights.csv
    results/optimization_diagnostics.csv

Input:
    results/walk_forward_predictions.csv

This implementation is intentionally dependency-light. It avoids cvxpy/scipy so
the project can run on a clean Python environment with pandas and numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = ROOT / "results" / "walk_forward_predictions.csv"
PORTFOLIO_WEIGHTS_PATH = ROOT / "results" / "portfolio_weights.csv"
DIAGNOSTICS_PATH = ROOT / "results" / "optimization_diagnostics.csv"

MAX_SINGLE_ETF_WEIGHT = 0.35
MIN_SELECTED_ETF_WEIGHT = 0.05
MAX_MONTHLY_TURNOVER = 0.50
TARGET_ANNUAL_VOLATILITY = 0.10

TOP_N = 3

PORTFOLIO_RULE = "risk_constrained_top3_max35_turnover50"

FALLBACK_DIVERSIFICATION_UNIVERSE = [
    "SPY",
    "AGG",
    "GLD",
    "EFA",
    "IWM",
    "QQQ",
    "TLT",
    "DBC",
    "VNQ",
    "IEF",
]

BOND_ETFS = {
    "AGG",
    "BND",
    "IEF",
    "TLT",
    "SHY",
    "IEI",
    "LQD",
    "HYG",
    "TIP",
    "MUB",
}

COMMODITY_ETFS = {
    "GLD",
    "SLV",
    "DBC",
    "USO",
    "UNG",
    "PDBC",
}

REAL_ESTATE_ETFS = {
    "VNQ",
    "IYR",
    "SCHH",
}


def classify_asset_group(ticker: str) -> str:
    ticker = str(ticker).upper().strip()

    if ticker in BOND_ETFS:
        return "bond"

    if ticker in COMMODITY_ETFS:
        return "commodity"

    if ticker in REAL_ESTATE_ETFS:
        return "real_estate"

    return "equity"


def find_date_column(df: pd.DataFrame) -> str:
    candidates = [
        "month_end_date",
        "test_date",
        "date",
        "rebalance_date",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Predictions file must contain one date column. "
        f"Expected one of {candidates}. Available columns: {list(df.columns)}"
    )


def find_prediction_column(df: pd.DataFrame) -> str:
    candidates = [
        "predicted_return",
        "prediction",
        "y_pred",
        "expected_return",
        "forecast_return",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Predictions file must contain one prediction column. "
        f"Expected one of {candidates}. Available columns: {list(df.columns)}"
    )


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}. "
            "Run the walk-forward validation or model-training script first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    if df.empty:
        raise ValueError(f"Predictions file is empty: {PREDICTIONS_PATH}")

    if "ticker" not in df.columns:
        raise ValueError(
            f"Predictions file must contain a ticker column. Available columns: {list(df.columns)}"
        )

    date_column = find_date_column(df)
    prediction_column = find_prediction_column(df)

    df = df.rename(
        columns={
            date_column: "month_end_date",
            prediction_column: "predicted_return",
        }
    )

    df["month_end_date"] = pd.to_datetime(df["month_end_date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["predicted_return"] = pd.to_numeric(
        df["predicted_return"],
        errors="coerce",
    )

    df = df.dropna(subset=["month_end_date", "ticker", "predicted_return"])

    if df.empty:
        raise ValueError(
            "No valid prediction rows remain after cleaning dates, tickers, and predicted returns."
        )

    df["month_end_date"] = df["month_end_date"] + pd.offsets.MonthEnd(0)

    df = (
        df.groupby(["month_end_date", "ticker"], as_index=False)
        .agg(predicted_return=("predicted_return", "mean"))
        .sort_values(["month_end_date", "predicted_return"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return df


def required_minimum_asset_count() -> int:
    return int(np.ceil(1.0 / MAX_SINGLE_ETF_WEIGHT))


def add_fallback_assets_if_needed(month_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure each rebalance month has enough available assets to satisfy the
    maximum single-ETF weight constraint.

    Example:
    If only one ticker is available, a fully invested ETF portfolio cannot obey
    a 35% max weight. The script therefore adds conservative fallback ETFs from
    a diversified universe with slightly lower expected returns. This prevents
    invalid 100% single-ETF allocations.
    """

    month_df = month_df.copy()
    month_df["is_fallback_asset"] = False

    needed_count = required_minimum_asset_count()
    current_tickers = set(month_df["ticker"].astype(str).str.upper())

    if len(current_tickers) >= needed_count:
        return month_df

    month_end_date = month_df["month_end_date"].iloc[0]

    if month_df["predicted_return"].notna().any():
        baseline_return = float(month_df["predicted_return"].min())
    else:
        baseline_return = 0.0

    fallback_rows: list[dict] = []

    for fallback_rank, ticker in enumerate(FALLBACK_DIVERSIFICATION_UNIVERSE, start=1):
        ticker = ticker.upper()

        if ticker in current_tickers:
            continue

        fallback_rows.append(
            {
                "month_end_date": month_end_date,
                "ticker": ticker,
                "predicted_return": baseline_return - 1e-6 * fallback_rank,
                "is_fallback_asset": True,
            }
        )

        current_tickers.add(ticker)

        if len(current_tickers) >= needed_count:
            break

    if len(current_tickers) < needed_count:
        raise ValueError(
            "Unable to construct a diversified portfolio with enough assets to "
            f"satisfy max weight {MAX_SINGLE_ETF_WEIGHT:.2%}."
        )

    fallback_df = pd.DataFrame(fallback_rows)

    if fallback_df.empty:
        return month_df

    return pd.concat([month_df, fallback_df], ignore_index=True)


def make_target_weights(month_df: pd.DataFrame) -> pd.Series:
    month_df = add_fallback_assets_if_needed(month_df)

    month_df = (
        month_df.sort_values("predicted_return", ascending=False)
        .drop_duplicates(subset=["ticker"], keep="first")
        .reset_index(drop=True)
    )

    minimum_assets_needed = required_minimum_asset_count()
    selected_count = max(TOP_N, minimum_assets_needed)
    selected_count = min(selected_count, len(month_df))

    if selected_count < minimum_assets_needed:
        raise ValueError(
            f"Cannot satisfy max ETF weight {MAX_SINGLE_ETF_WEIGHT:.2%} with "
            f"only {selected_count} asset(s)."
        )

    selected = month_df.head(selected_count).copy()

    equal_weight = 1.0 / selected_count

    if equal_weight > MAX_SINGLE_ETF_WEIGHT + 1e-12:
        raise ValueError(
            f"Internal allocation error: equal weight {equal_weight:.6f} exceeds "
            f"max allowed weight {MAX_SINGLE_ETF_WEIGHT:.6f}."
        )

    if equal_weight < MIN_SELECTED_ETF_WEIGHT - 1e-12:
        raise ValueError(
            f"Internal allocation error: equal weight {equal_weight:.6f} is below "
            f"minimum selected ETF weight {MIN_SELECTED_ETF_WEIGHT:.6f}."
        )

    weights = pd.Series(0.0, index=month_df["ticker"].values, dtype=float)
    weights.loc[selected["ticker"].values] = equal_weight

    return weights


def align_weight_vectors(
    previous_weights: pd.Series | None,
    target_weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if previous_weights is None:
        previous_index: set[str] = set()
    else:
        previous_index = set(previous_weights.index)

    all_tickers = sorted(previous_index | set(target_weights.index))

    previous = (
        pd.Series(0.0, index=all_tickers, dtype=float)
        if previous_weights is None
        else previous_weights.reindex(all_tickers).fillna(0.0).astype(float)
    )

    target = target_weights.reindex(all_tickers).fillna(0.0).astype(float)

    return previous, target


def calculate_turnover(
    previous_weights: pd.Series | None,
    new_weights: pd.Series,
) -> float:
    previous, new = align_weight_vectors(previous_weights, new_weights)
    return float(0.5 * np.abs(new - previous).sum())


def apply_turnover_constraint(
    previous_weights: pd.Series | None,
    target_weights: pd.Series,
) -> tuple[pd.Series, float, float]:
    """
    Enforce maximum monthly turnover by blending from previous weights toward
    target weights.

    Turnover convention:
        turnover = 0.5 * sum(abs(new_weight - old_weight))

    The first fully invested portfolio from zero has turnover 0.50.
    """

    previous, target = align_weight_vectors(previous_weights, target_weights)

    raw_turnover = float(0.5 * np.abs(target - previous).sum())

    if raw_turnover <= MAX_MONTHLY_TURNOVER + 1e-12:
        optimized = target.copy()
        blend_fraction = 1.0
    else:
        blend_fraction = MAX_MONTHLY_TURNOVER / raw_turnover
        optimized = previous + blend_fraction * (target - previous)

    optimized = optimized.clip(lower=0.0)

    if optimized.sum() <= 0:
        raise ValueError("Optimized portfolio has zero total weight.")

    optimized = optimized / optimized.sum()

    final_turnover = float(0.5 * np.abs(optimized - previous).sum())

    return optimized, raw_turnover, final_turnover


def validate_weight_vector(weights: pd.Series, month_end_date: pd.Timestamp) -> None:
    if weights.empty:
        raise ValueError(f"No weights generated for {month_end_date.date()}.")

    if weights.isna().any():
        raise ValueError(f"NaN weights found for {month_end_date.date()}.")

    if (weights < -1e-12).any():
        raise ValueError(f"Negative weights found for {month_end_date.date()}.")

    weight_sum = float(weights.sum())

    if not np.isclose(weight_sum, 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError(
            f"Weights do not sum to 1 for {month_end_date.date()}. Observed sum={weight_sum:.10f}."
        )

    max_weight = float(weights.max())

    if max_weight > MAX_SINGLE_ETF_WEIGHT + 1e-8:
        raise ValueError(
            f"Max ETF weight constraint failed for {month_end_date.date()}. "
            f"Observed max={max_weight:.6f}, "
            f"allowed max={MAX_SINGLE_ETF_WEIGHT:.6f}."
        )


def build_portfolios(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    previous_weights: pd.Series | None = None

    for month_end_date, raw_month_df in predictions.groupby("month_end_date"):
        month_df = raw_month_df.copy()
        month_df = add_fallback_assets_if_needed(month_df)

        prediction_lookup = (
            month_df.drop_duplicates(subset=["ticker"], keep="first")
            .set_index("ticker")["predicted_return"]
            .to_dict()
        )

        fallback_lookup = (
            month_df.drop_duplicates(subset=["ticker"], keep="first")
            .set_index("ticker")["is_fallback_asset"]
            .to_dict()
            if "is_fallback_asset" in month_df.columns
            else {}
        )

        target_weights = make_target_weights(month_df)

        optimized_weights, raw_turnover, final_turnover = apply_turnover_constraint(
            previous_weights=previous_weights,
            target_weights=target_weights,
        )

        validate_weight_vector(optimized_weights, month_end_date)

        previous_aligned, optimized_aligned = align_weight_vectors(
            previous_weights=previous_weights,
            target_weights=optimized_weights,
        )

        target_aligned = target_weights.reindex(optimized_aligned.index).fillna(0.0)

        for ticker in optimized_aligned.index:
            optimized_weight = float(optimized_aligned.loc[ticker])

            if optimized_weight <= 1e-12:
                continue

            previous_weight = float(previous_aligned.loc[ticker])
            target_weight = float(target_aligned.loc[ticker])
            predicted_return = float(prediction_lookup.get(ticker, 0.0))
            is_fallback_asset = bool(fallback_lookup.get(ticker, False))
            selected_flag = optimized_weight >= MIN_SELECTED_ETF_WEIGHT - 1e-12

            portfolio_rows.append(
                {
                    "month_end_date": month_end_date,
                    "ticker": ticker,
                    "predicted_return": predicted_return,
                    "previous_weight": previous_weight,
                    "target_weight": target_weight,
                    "optimized_weight": optimized_weight,
                    "weight": optimized_weight,
                    "selected_flag": selected_flag,
                    "is_fallback_asset": is_fallback_asset,
                    "asset_group": classify_asset_group(ticker),
                    "expected_return_contribution": optimized_weight * predicted_return,
                    "portfolio_rule": PORTFOLIO_RULE,
                }
            )

        positive_weights = optimized_weights[optimized_weights > 1e-12]
        selected_weights = positive_weights[positive_weights >= MIN_SELECTED_ETF_WEIGHT - 1e-12]

        diagnostic_rows.append(
            {
                "month_end_date": month_end_date,
                "selected_etf_count": int(len(selected_weights)),
                "positive_weight_count": int(len(positive_weights)),
                "weight_sum": float(positive_weights.sum()),
                "max_weight": float(positive_weights.max()),
                "min_positive_weight": float(positive_weights.min()),
                "min_selected_weight": float(selected_weights.min())
                if not selected_weights.empty
                else np.nan,
                "raw_monthly_turnover": float(raw_turnover),
                "monthly_turnover": float(final_turnover),
                "target_annual_volatility": TARGET_ANNUAL_VOLATILITY,
                "max_single_etf_weight": MAX_SINGLE_ETF_WEIGHT,
                "min_selected_etf_weight": MIN_SELECTED_ETF_WEIGHT,
                "max_monthly_turnover": MAX_MONTHLY_TURNOVER,
                "max_weight_constraint_pass": bool(
                    positive_weights.max() <= MAX_SINGLE_ETF_WEIGHT + 1e-8
                ),
                "turnover_constraint_pass": bool(final_turnover <= MAX_MONTHLY_TURNOVER + 1e-8),
                "long_only_constraint_pass": bool((positive_weights >= -1e-12).all()),
                "constraint_status": "pass",
                "portfolio_rule": PORTFOLIO_RULE,
            }
        )

        previous_weights = optimized_weights.copy()

    portfolio_df = pd.DataFrame(portfolio_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)

    return portfolio_df, diagnostics_df


def validate_portfolio_dataframe(portfolio_df: pd.DataFrame) -> None:
    if portfolio_df.empty:
        raise ValueError("No portfolio weights were generated.")

    required_columns = {
        "month_end_date",
        "ticker",
        "optimized_weight",
        "weight",
        "selected_flag",
    }

    missing = required_columns - set(portfolio_df.columns)

    if missing:
        raise ValueError(f"Portfolio output is missing required columns: {missing}")

    portfolio_df = portfolio_df.copy()
    portfolio_df["month_end_date"] = pd.to_datetime(
        portfolio_df["month_end_date"],
        errors="coerce",
    )
    portfolio_df["optimized_weight"] = pd.to_numeric(
        portfolio_df["optimized_weight"],
        errors="coerce",
    )

    if portfolio_df["month_end_date"].isna().any():
        raise ValueError("month_end_date contains invalid dates.")

    if portfolio_df["optimized_weight"].isna().any():
        raise ValueError("optimized_weight contains invalid numeric values.")

    if (portfolio_df["optimized_weight"] < -1e-12).any():
        raise ValueError("Long-only constraint failed: negative weights found.")

    monthly_weight_sum = portfolio_df.groupby("month_end_date")["optimized_weight"].sum()

    bad_sums = monthly_weight_sum[~np.isclose(monthly_weight_sum, 1.0, atol=1e-6, rtol=1e-6)]

    if not bad_sums.empty:
        raise ValueError(f"Portfolio weights must sum to 1 for each month. Bad months:\n{bad_sums}")

    max_observed_weight = float(portfolio_df["optimized_weight"].max())

    if max_observed_weight > MAX_SINGLE_ETF_WEIGHT + 1e-8:
        raise ValueError(
            f"Single ETF weight exceeds maximum allowed weight. "
            f"Observed max={max_observed_weight:.6f}, "
            f"allowed max={MAX_SINGLE_ETF_WEIGHT:.6f}."
        )

    selected = portfolio_df[portfolio_df["selected_flag"].astype(bool)].copy()

    if selected.empty:
        raise ValueError("No selected ETFs found.")

    min_selected_weight = float(selected["optimized_weight"].min())

    if min_selected_weight < MIN_SELECTED_ETF_WEIGHT - 1e-8:
        raise ValueError(
            f"Selected ETF weights must be at least "
            f"{MIN_SELECTED_ETF_WEIGHT:.2%}. "
            f"Observed min={min_selected_weight:.6f}."
        )


def validate_diagnostics_dataframe(diagnostics_df: pd.DataFrame) -> None:
    if diagnostics_df.empty:
        raise ValueError("No optimization diagnostics were generated.")

    required_columns = {
        "month_end_date",
        "weight_sum",
        "max_weight",
        "monthly_turnover",
        "constraint_status",
    }

    missing = required_columns - set(diagnostics_df.columns)

    if missing:
        raise ValueError(f"Diagnostics output is missing required columns: {missing}")

    if not np.isclose(diagnostics_df["weight_sum"], 1.0, atol=1e-6, rtol=1e-6).all():
        raise ValueError("Diagnostics show one or more months with weights not equal to 1.")

    if (diagnostics_df["max_weight"] > MAX_SINGLE_ETF_WEIGHT + 1e-8).any():
        raise ValueError("Diagnostics show one or more max-weight violations.")

    if (diagnostics_df["monthly_turnover"] > MAX_MONTHLY_TURNOVER + 1e-8).any():
        raise ValueError("Diagnostics show one or more turnover violations.")


def main() -> None:
    predictions = load_predictions()

    portfolio_df, diagnostics_df = build_portfolios(predictions)

    validate_portfolio_dataframe(portfolio_df)
    validate_diagnostics_dataframe(diagnostics_df)

    PORTFOLIO_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    portfolio_df.to_csv(PORTFOLIO_WEIGHTS_PATH, index=False)
    diagnostics_df.to_csv(DIAGNOSTICS_PATH, index=False)

    print(f"Saved portfolio weights to: {PORTFOLIO_WEIGHTS_PATH}")
    print(f"Saved optimization diagnostics to: {DIAGNOSTICS_PATH}")
    print()
    print("Recent optimization diagnostics:")
    print(diagnostics_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
