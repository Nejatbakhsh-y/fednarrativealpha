from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "walk_forward_predictions.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "raw_signal_weights.csv"
DEFAULT_PRICE_PATH = PROJECT_ROOT / "data" / "raw" / "etf_prices.parquet"

REQUIRED_PREDICTION_COLUMNS = {
    "month_end_date",
    "ticker",
    "predicted_return",
}

OPTIONAL_GROUP_COLUMNS = [
    "model_name",
    "feature_set",
    "training_start",
    "training_end",
    "test_date",
]

DATE_COLUMNS = [
    "month_end_date",
    "training_start",
    "training_end",
    "test_date",
]


def log(message: str) -> None:
    """Print a consistent status message."""
    print(f"[signal_to_portfolio] {message}")


def read_predictions(path: Path) -> pd.DataFrame:
    """Read and validate walk-forward prediction output."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Run Step 12 first to create results/walk_forward_predictions.csv."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_PREDICTION_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Prediction file is missing required columns: " + ", ".join(sorted(missing))
        )

    df = df.copy()

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["predicted_return"] = pd.to_numeric(df["predicted_return"], errors="coerce")

    if "actual_next_return" in df.columns:
        df["actual_next_return"] = pd.to_numeric(df["actual_next_return"], errors="coerce")

    df = df.dropna(subset=["month_end_date", "ticker", "predicted_return"])

    if df.empty:
        raise ValueError("Prediction file contains no usable prediction rows.")

    duplicate_subset = [
        col
        for col in [
            "month_end_date",
            "ticker",
            "model_name",
            "feature_set",
            "training_start",
            "training_end",
            "test_date",
        ]
        if col in df.columns
    ]

    if duplicate_subset:
        before = len(df)
        df = df.sort_values(duplicate_subset).drop_duplicates(
            subset=duplicate_subset,
            keep="last",
        )
        dropped = before - len(df)
        if dropped > 0:
            log(f"Dropped {dropped:,} duplicate prediction rows.")

    return df


def get_group_columns(df: pd.DataFrame) -> list[str]:
    """Group by month and available model metadata."""
    group_cols = ["month_end_date"]
    for col in OPTIONAL_GROUP_COLUMNS:
        if col in df.columns:
            group_cols.append(col)
    return group_cols


def add_signal_ranks(group: pd.DataFrame) -> pd.DataFrame:
    """Rank assets within one monthly prediction group."""
    ranked = group.sort_values(
        by=["predicted_return", "ticker"],
        ascending=[False, True],
    ).copy()
    ranked["signal_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def build_top_equal_weight_signals(
    predictions: pd.DataFrame,
    top_n: int,
    include_zero_weights: bool = True,
) -> pd.DataFrame:
    """
    Convert predictions into raw long-only signal weights.

    Rule:
        For each monthly model/feature-set group:
        1. Sort ETFs by predicted_return from highest to lowest.
        2. Select the top N ETFs.
        3. Assign equal weight across selected ETFs.
        4. Assign zero weight to non-selected ETFs when include_zero_weights=True.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    group_cols = get_group_columns(predictions)
    output_frames: list[pd.DataFrame] = []

    for _, group in predictions.groupby(group_cols, dropna=False, sort=True):
        ranked = add_signal_ranks(group)

        selected_count = min(top_n, len(ranked))
        ranked["selected_flag"] = ranked["signal_rank"] <= selected_count
        ranked["selected_count"] = selected_count
        ranked["raw_weight"] = 0.0

        if selected_count > 0:
            ranked.loc[ranked["selected_flag"], "raw_weight"] = 1.0 / selected_count

        ranked["initial_weight"] = ranked["raw_weight"]
        ranked["weight"] = ranked["raw_weight"]
        ranked["portfolio_rule"] = f"top_{top_n}_equal_weight"
        ranked["top_n"] = top_n
        ranked["weight_method"] = "equal_weight"
        ranked["requires_risk_optimizer"] = True

        if not include_zero_weights:
            ranked = ranked.loc[ranked["selected_flag"]].copy()

        output_frames.append(ranked)

    if not output_frames:
        raise ValueError("No signal weights were created.")

    return pd.concat(output_frames, ignore_index=True)


def read_price_returns(price_path: Path) -> pd.DataFrame | None:
    """
    Read daily ETF prices and return daily adjusted-close returns.

    This is only used by the advanced raw-weight option. The simple top-3
    equal-weight rule does not need price data.
    """
    if not price_path.exists():
        log(
            f"Price file not found at {price_path}. "
            "Advanced covariance weights will fall back to prediction-only weights."
        )
        return None

    prices = pd.read_parquet(price_path)

    required = {"date", "ticker"}
    missing = required.difference(prices.columns)
    if missing:
        log(
            "Price file is missing required columns "
            f"{sorted(missing)}. Advanced weights will use fallback weights."
        )
        return None

    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    if price_col not in prices.columns:
        log(
            "Price file has neither adj_close nor close. "
            "Advanced weights will use fallback weights."
        )
        return None

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")

    prices = prices.dropna(subset=["date", "ticker", price_col])
    prices = prices.sort_values(["date", "ticker"])

    price_matrix = prices.pivot_table(
        index="date",
        columns="ticker",
        values=price_col,
        aggfunc="last",
    ).sort_index()

    returns = price_matrix.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)

    if returns.empty:
        log("Price return matrix is empty. Advanced weights will use fallback weights.")
        return None

    return returns


def estimate_covariance_matrix(
    daily_returns: pd.DataFrame | None,
    tickers: list[str],
    as_of_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame | None:
    """
    Estimate a covariance matrix using only data before the signal month.

    No-lookahead rule:
        Daily returns must be strictly earlier than month_end_date.
    """
    if daily_returns is None or daily_returns.empty:
        return None

    available_tickers = [ticker for ticker in tickers if ticker in daily_returns.columns]
    if len(available_tickers) < 2:
        return None

    history = daily_returns.loc[daily_returns.index < as_of_date, available_tickers].tail(
        lookback_days
    )

    min_obs = max(40, min(80, lookback_days // 3))
    if len(history) < min_obs:
        return None

    cov = history.cov(min_periods=min_obs // 2)
    cov = cov.reindex(index=tickers, columns=tickers)

    diag_values = np.diag(cov.fillna(0.0).to_numpy())
    positive_diag = diag_values[diag_values > 0]

    if len(positive_diag) == 0:
        return None

    fallback_variance = float(np.nanmedian(positive_diag))
    cov = cov.fillna(0.0)

    for ticker in tickers:
        if cov.loc[ticker, ticker] <= 0:
            cov.loc[ticker, ticker] = fallback_variance

    cov_values = cov.to_numpy(dtype=float)
    diagonal = np.diag(np.diag(cov_values))
    shrunk = 0.90 * cov_values + 0.10 * diagonal

    ridge = max(float(np.nanmedian(np.diag(shrunk))) * 1e-6, 1e-10)
    shrunk = shrunk + ridge * np.eye(len(tickers))

    return pd.DataFrame(shrunk, index=tickers, columns=tickers)


def cap_and_normalize_weights(
    weights: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    """Normalize nonnegative weights and apply a simple maximum-weight cap."""
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)

    if weights.sum() <= 0:
        return np.repeat(1.0 / len(weights), len(weights))

    weights = weights / weights.sum()

    if max_weight <= 0 or max_weight >= 1:
        return weights

    max_weight = max(max_weight, 1.0 / len(weights))

    capped = np.zeros_like(weights)
    remaining = np.ones(len(weights), dtype=bool)
    remaining_total = 1.0

    base = weights.copy()

    while remaining.any():
        rem_base = base[remaining]
        if rem_base.sum() <= 0:
            capped[remaining] = remaining_total / remaining.sum()
            break

        proposed = remaining_total * rem_base / rem_base.sum()
        over_cap_local = proposed > max_weight

        if not over_cap_local.any():
            capped[remaining] = proposed
            break

        remaining_indices = np.where(remaining)[0]
        capped_indices = remaining_indices[over_cap_local]
        capped[capped_indices] = max_weight
        remaining[capped_indices] = False
        remaining_total = 1.0 - capped.sum()

        if remaining_total <= 0:
            break

    if capped.sum() <= 0:
        return np.repeat(1.0 / len(weights), len(weights))

    return capped / capped.sum()


def prediction_only_fallback_weights(
    predicted_returns: Iterable[float],
    max_weight: float,
) -> np.ndarray:
    """
    Produce long-only weights from predictions when covariance is unavailable.

    The transformation is shift-based, so it remains valid when all predicted
    returns are negative.
    """
    mu = np.asarray(list(predicted_returns), dtype=float)
    mu = np.where(np.isfinite(mu), mu, np.nan)

    if np.isnan(mu).all():
        raw = np.ones(len(mu), dtype=float)
    else:
        min_mu = np.nanmin(mu)
        raw = mu - min_mu
        raw = np.where(np.isfinite(raw), raw, 0.0)
        raw = raw + 1e-8

    return cap_and_normalize_weights(raw, max_weight=max_weight)


def mean_variance_raw_weights(
    predicted_returns: Iterable[float],
    covariance: pd.DataFrame | None,
    risk_aversion: float,
    max_weight: float,
) -> np.ndarray:
    """
    Compute a long-only raw mean-variance proxy.

    This is intentionally a raw signal allocator, not the final risk optimizer.
    The later risk-optimization step can impose volatility, turnover, drawdown,
    liquidity, sector, or other constraints.
    """
    mu = np.asarray(list(predicted_returns), dtype=float)
    mu = np.where(np.isfinite(mu), mu, np.nan)

    if covariance is None:
        return prediction_only_fallback_weights(mu, max_weight=max_weight)

    cov = covariance.to_numpy(dtype=float)

    if cov.shape[0] != len(mu) or cov.shape[1] != len(mu):
        return prediction_only_fallback_weights(mu, max_weight=max_weight)

    if np.isnan(mu).all():
        return np.repeat(1.0 / len(mu), len(mu))

    mu_shifted = mu - np.nanmin(mu)
    mu_shifted = np.where(np.isfinite(mu_shifted), mu_shifted, 0.0)
    mu_shifted = mu_shifted + 1e-8

    try:
        raw = np.linalg.pinv(cov * risk_aversion) @ mu_shifted
    except np.linalg.LinAlgError:
        return prediction_only_fallback_weights(mu, max_weight=max_weight)

    raw = np.clip(raw, 0.0, None)

    if not np.isfinite(raw).all() or raw.sum() <= 0:
        return prediction_only_fallback_weights(mu, max_weight=max_weight)

    return cap_and_normalize_weights(raw, max_weight=max_weight)


def build_top_mean_variance_signals(
    predictions: pd.DataFrame,
    daily_returns: pd.DataFrame | None,
    top_n: int,
    lookback_days: int,
    risk_aversion: float,
    max_weight: float,
    include_zero_weights: bool = True,
) -> pd.DataFrame:
    """
    Advanced raw signal rule.

    Rule:
        1. Rank ETFs by predicted return.
        2. Select the top N ETFs, normally top 5.
        3. Estimate covariance from prior daily ETF returns only.
        4. Compute long-only mean-variance proxy weights.
        5. Mark the output as requiring the later risk optimizer.
    """
    if top_n <= 1:
        raise ValueError("Advanced mean-variance rule requires top_n >= 2.")

    group_cols = get_group_columns(predictions)
    output_frames: list[pd.DataFrame] = []

    for _, group in predictions.groupby(group_cols, dropna=False, sort=True):
        ranked = add_signal_ranks(group)

        selected_count = min(top_n, len(ranked))
        ranked["selected_flag"] = ranked["signal_rank"] <= selected_count
        ranked["selected_count"] = selected_count
        ranked["raw_weight"] = 0.0

        selected = ranked.loc[ranked["selected_flag"]].copy()
        selected_tickers = selected["ticker"].tolist()

        covariance = estimate_covariance_matrix(
            daily_returns=daily_returns,
            tickers=selected_tickers,
            as_of_date=ranked["month_end_date"].iloc[0],
            lookback_days=lookback_days,
        )

        weights = mean_variance_raw_weights(
            predicted_returns=selected["predicted_return"].to_numpy(dtype=float),
            covariance=covariance,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
        )

        ranked.loc[ranked["selected_flag"], "raw_weight"] = weights
        ranked["initial_weight"] = ranked["raw_weight"]
        ranked["weight"] = ranked["raw_weight"]
        ranked["portfolio_rule"] = f"top_{top_n}_mean_variance_raw"
        ranked["top_n"] = top_n
        ranked["weight_method"] = "mean_variance_proxy"
        ranked["covariance_lookback_days"] = lookback_days
        ranked["risk_aversion"] = risk_aversion
        ranked["max_weight"] = max_weight
        ranked["requires_risk_optimizer"] = True

        if not include_zero_weights:
            ranked = ranked.loc[ranked["selected_flag"]].copy()

        output_frames.append(ranked)

    if not output_frames:
        raise ValueError("No advanced signal weights were created.")

    return pd.concat(output_frames, ignore_index=True)


def validate_weight_sums(signals: pd.DataFrame) -> None:
    """Confirm that each monthly portfolio group sums to one."""
    group_cols = get_group_columns(signals)

    sums = signals.groupby(group_cols, dropna=False)["weight"].sum().reset_index(name="weight_sum")

    bad = sums.loc[~np.isclose(sums["weight_sum"], 1.0, atol=1e-8)]

    if not bad.empty:
        preview = bad.head(10).to_string(index=False)
        raise ValueError(
            "Some monthly portfolio groups do not sum to 1.0.\n"
            f"First problematic groups:\n{preview}"
        )


def order_output_columns(signals: pd.DataFrame) -> pd.DataFrame:
    """Put the most important audit columns first."""
    preferred = [
        "month_end_date",
        "test_date",
        "ticker",
        "predicted_return",
        "predicted_rank",
        "signal_rank",
        "selected_flag",
        "selected_count",
        "raw_weight",
        "initial_weight",
        "weight",
        "actual_next_return",
        "model_name",
        "feature_set",
        "training_start",
        "training_end",
        "portfolio_rule",
        "top_n",
        "weight_method",
        "requires_risk_optimizer",
        "covariance_lookback_days",
        "risk_aversion",
        "max_weight",
        "created_at_utc",
    ]

    existing_preferred = [col for col in preferred if col in signals.columns]
    remaining = [col for col in signals.columns if col not in existing_preferred]
    return signals[existing_preferred + remaining]


def write_signals(signals: pd.DataFrame, output_path: Path) -> None:
    """Write signal weights to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signals = signals.copy()
    signals["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signals = order_output_columns(signals)

    sort_cols = [
        col
        for col in [
            "month_end_date",
            "model_name",
            "feature_set",
            "signal_rank",
            "ticker",
        ]
        if col in signals.columns
    ]

    signals = signals.sort_values(sort_cols).reset_index(drop=True)

    signals.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    selected_rows = int(signals["selected_flag"].sum())
    total_rows = len(signals)
    months = signals["month_end_date"].nunique()

    log(f"Wrote {total_rows:,} rows to {output_path}")
    log(f"Selected nonzero signal rows: {selected_rows:,}")
    log(f"Unique signal months: {months:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert walk-forward model predictions into raw portfolio weights."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to results/walk_forward_predictions.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for results/raw_signal_weights.csv.",
    )
    parser.add_argument(
        "--rule",
        choices=["top_equal_weight", "top_mean_variance"],
        default="top_equal_weight",
        help=(
            "Portfolio signal rule. Use top_equal_weight for the required simple "
            "version. Use top_mean_variance for the optional advanced raw signal."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Number of ETFs to select. Default is 3 for top_equal_weight and 5 "
            "for top_mean_variance."
        ),
    )
    parser.add_argument(
        "--price-data",
        type=Path,
        default=DEFAULT_PRICE_PATH,
        help="Path to daily ETF price parquet file for advanced covariance estimates.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=252,
        help="Daily return lookback window for covariance estimates.",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=10.0,
        help="Risk-aversion scalar for advanced mean-variance proxy weights.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.40,
        help="Maximum single-ETF raw weight for advanced weights.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional exact model_name filter.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        help="Optional exact feature_set filter.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Write only selected ETFs instead of all ETFs with zero weights.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions = read_predictions(args.predictions)

    if args.model_name is not None:
        if "model_name" not in predictions.columns:
            raise ValueError("--model-name was supplied, but model_name is not in predictions.")
        predictions = predictions.loc[predictions["model_name"] == args.model_name].copy()

    if args.feature_set is not None:
        if "feature_set" not in predictions.columns:
            raise ValueError("--feature-set was supplied, but feature_set is not in predictions.")
        predictions = predictions.loc[predictions["feature_set"] == args.feature_set].copy()

    if predictions.empty:
        raise ValueError("No predictions remain after applying filters.")

    if args.top_n is not None:
        top_n = args.top_n
    elif args.rule == "top_mean_variance":
        top_n = 5
    else:
        top_n = 3

    include_zero_weights = not args.selected_only

    log(f"Input predictions: {args.predictions}")
    log(f"Signal rule: {args.rule}")
    log(f"Top N: {top_n}")

    if args.rule == "top_equal_weight":
        signals = build_top_equal_weight_signals(
            predictions=predictions,
            top_n=top_n,
            include_zero_weights=include_zero_weights,
        )
    else:
        daily_returns = read_price_returns(args.price_data)
        signals = build_top_mean_variance_signals(
            predictions=predictions,
            daily_returns=daily_returns,
            top_n=top_n,
            lookback_days=args.lookback_days,
            risk_aversion=args.risk_aversion,
            max_weight=args.max_weight,
            include_zero_weights=include_zero_weights,
        )

    validate_weight_sums(signals)
    write_signals(signals, args.output)


if __name__ == "__main__":
    main()
