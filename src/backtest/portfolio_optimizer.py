"""
Step 15 — Risk-Constrained Portfolio Optimizer

Creates:
    results/portfolio_weights.csv
    results/optimization_diagnostics.csv

Main input:
    results/walk_forward_predictions.csv

Fallback input:
    results/raw_signal_weights.csv

The optimizer uses only return history dated before the current month when
estimating covariance. This helps preserve the no-lookahead structure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_PATH = RESULTS_DIR / "walk_forward_predictions.csv"
RAW_SIGNAL_WEIGHTS_PATH = RESULTS_DIR / "raw_signal_weights.csv"

MASTER_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "master_modeling_dataset.parquet"
PRICE_FEATURES_PATH = PROJECT_ROOT / "data" / "interim" / "price_features_monthly.parquet"

OUTPUT_WEIGHTS_PATH = RESULTS_DIR / "portfolio_weights.csv"
OUTPUT_DIAGNOSTICS_PATH = RESULTS_DIR / "optimization_diagnostics.csv"


# ---------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------

TOP_N = 5
LOOKBACK_MONTHS = 36
MIN_HISTORY_MONTHS = 6

MAX_SINGLE_ETF_WEIGHT = 0.35
MIN_SELECTED_ETF_WEIGHT = 0.05
MAX_MONTHLY_TURNOVER = 0.50
TARGET_VOL_ANNUAL = 0.10

RISK_AVERSION = 3.0
TURNOVER_PENALTY = 0.05

# Optional constraints. Keep as None unless you want to activate them.
MAX_BOND_ALLOCATION: Optional[float] = None
MAX_COMMODITY_ALLOCATION: Optional[float] = None

DEFAULT_FALLBACK_ANNUAL_VOL = 0.20

BOND_ETFS = {
    "AGG",
    "BND",
    "BNDX",
    "BSV",
    "BIL",
    "SHY",
    "IEI",
    "IEF",
    "TLT",
    "TIP",
    "MBB",
    "LQD",
    "HYG",
    "JNK",
    "VCIT",
    "VCSH",
    "VGIT",
    "VGLT",
}

COMMODITY_ETFS = {
    "GLD",
    "IAU",
    "SLV",
    "DBC",
    "PDBC",
    "GSG",
    "COMT",
    "USO",
    "UNG",
    "DBA",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def _read_parquet_if_possible(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"Warning: could not read {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None


def _clean_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "month_end_date" not in out.columns:
        raise ValueError("Missing required column: month_end_date")
    if "ticker" not in out.columns:
        raise ValueError("Missing required column: ticker")

    out["month_end_date"] = pd.to_datetime(out["month_end_date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()

    out = out.dropna(subset=["month_end_date"])
    out = out[out["ticker"].ne("")]

    return out


def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        path = PREDICTIONS_PATH
    elif RAW_SIGNAL_WEIGHTS_PATH.exists():
        path = RAW_SIGNAL_WEIGHTS_PATH
    else:
        raise FileNotFoundError(
            "No prediction file found. Run Step 12 or Step 13 first. Expected one of:\n"
            f"  {PREDICTIONS_PATH}\n"
            f"  {RAW_SIGNAL_WEIGHTS_PATH}"
        )

    df = pd.read_csv(path)
    df = _clean_base_columns(df)

    prediction_candidates = [
        "predicted_return",
        "prediction",
        "predicted_next_return",
        "expected_return",
        "model_prediction",
    ]

    prediction_col = next((c for c in prediction_candidates if c in df.columns), None)

    if prediction_col is None:
        raise ValueError(
            "Could not find a prediction column. Expected one of: "
            + ", ".join(prediction_candidates)
        )

    if prediction_col != "predicted_return":
        df = df.rename(columns={prediction_col: "predicted_return"})

    df["predicted_return"] = pd.to_numeric(df["predicted_return"], errors="coerce")
    df = df.dropna(subset=["predicted_return"])

    actual_candidates = [
        "actual_next_return",
        "next_1m_return",
        "forward_1m_return",
        "realized_next_return",
    ]

    actual_col = next((c for c in actual_candidates if c in df.columns), None)

    if actual_col and actual_col != "actual_next_return":
        df = df.rename(columns={actual_col: "actual_next_return"})

    if "actual_next_return" in df.columns:
        df["actual_next_return"] = pd.to_numeric(df["actual_next_return"], errors="coerce")

    keep = ["month_end_date", "ticker", "predicted_return"]

    if "actual_next_return" in df.columns:
        keep.append("actual_next_return")

    out = df[keep].sort_values(["month_end_date", "ticker"]).reset_index(drop=True)

    print(f"Loaded predictions: {path.relative_to(PROJECT_ROOT)}")
    print(
        f"Rows: {len(out):,} | "
        f"Months: {out['month_end_date'].nunique():,} | "
        f"ETFs: {out['ticker'].nunique():,}"
    )

    return out


def _standardize_return_data(
    df: Optional[pd.DataFrame],
    source_name: str,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    if "month_end_date" not in df.columns or "ticker" not in df.columns:
        return None

    return_col = None

    for c in ["next_1m_return", "actual_next_return", "realized_next_return"]:
        if c in df.columns:
            return_col = c
            break

    if return_col is None:
        return None

    out = df[["month_end_date", "ticker", return_col]].copy()
    out = _clean_base_columns(out)

    out = out.rename(columns={return_col: "realized_return"})
    out["realized_return"] = pd.to_numeric(out["realized_return"], errors="coerce")
    out = out.dropna(subset=["realized_return"])

    out["source"] = source_name

    if out.empty:
        return None

    return out


def load_return_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    master = _standardize_return_data(
        _read_parquet_if_possible(MASTER_DATASET_PATH),
        "master_modeling_dataset",
    )

    if master is not None:
        frames.append(master)

    price = _standardize_return_data(
        _read_parquet_if_possible(PRICE_FEATURES_PATH),
        "price_features_monthly",
    )

    if price is not None:
        frames.append(price)

    if "actual_next_return" in predictions.columns:
        pred_copy = predictions.rename(columns={"actual_next_return": "realized_next_return"})
        pred_hist = _standardize_return_data(pred_copy, "walk_forward_predictions")

        if pred_hist is not None:
            frames.append(pred_hist)

    if not frames:
        print("No realized-return history found. The script will use fallback covariance.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    priority = {
        "master_modeling_dataset": 1,
        "price_features_monthly": 2,
        "walk_forward_predictions": 3,
    }

    combined["priority"] = combined["source"].map(priority).fillna(99)
    combined = combined.sort_values(["month_end_date", "ticker", "priority"])
    combined = combined.drop_duplicates(["month_end_date", "ticker"], keep="first")

    matrix = combined.pivot_table(
        index="month_end_date",
        columns="ticker",
        values="realized_return",
        aggfunc="mean",
    ).sort_index()

    print(f"Loaded covariance return matrix: {matrix.shape[0]:,} months x {matrix.shape[1]:,} ETFs")

    return matrix


# ---------------------------------------------------------------------
# Portfolio math
# ---------------------------------------------------------------------


def is_bond(ticker: str) -> bool:
    return ticker.upper() in BOND_ETFS


def is_commodity(ticker: str) -> bool:
    return ticker.upper() in COMMODITY_ETFS


def choose_candidates(
    month_df: pd.DataFrame,
    previous_weights: Dict[str, float],
) -> List[str]:
    ranked = month_df.sort_values("predicted_return", ascending=False)
    top = ranked["ticker"].head(TOP_N).tolist()

    previous = [t for t, w in previous_weights.items() if abs(w) > 1e-10]

    candidates: List[str] = []

    for ticker in top + previous:
        if ticker not in candidates:
            candidates.append(ticker)

    max_assets = int(math.floor(1.0 / MIN_SELECTED_ETF_WEIGHT))

    return candidates[:max_assets]


def effective_bounds(n_assets: int) -> Tuple[float, float, str]:
    lower = MIN_SELECTED_ETF_WEIGHT
    upper = MAX_SINGLE_ETF_WEIGHT
    notes = []

    if n_assets * lower > 1.0:
        lower = 1.0 / n_assets
        notes.append("minimum_weight_relaxed")

    if n_assets * upper < 1.0:
        upper = 1.0 / n_assets
        notes.append("maximum_weight_relaxed")

    if lower > upper:
        lower = min(upper, 1.0 / n_assets)
        notes.append("bounds_reconciled")

    return lower, upper, ";".join(notes) if notes else "requested_bounds_used"


def estimate_covariance(
    return_matrix: pd.DataFrame,
    month: pd.Timestamp,
    candidates: Sequence[str],
) -> Tuple[np.ndarray, int, str]:
    n = len(candidates)
    fallback_var = (DEFAULT_FALLBACK_ANNUAL_VOL / math.sqrt(12.0)) ** 2

    if return_matrix.empty:
        return np.eye(n) * fallback_var, 0, "fallback_no_history"

    available = [t for t in candidates if t in return_matrix.columns]

    if not available:
        return np.eye(n) * fallback_var, 0, "fallback_no_candidate_history"

    hist = return_matrix.loc[return_matrix.index < month, available].tail(LOOKBACK_MONTHS)
    hist = hist.dropna(how="all")

    obs = len(hist)

    if obs < MIN_HISTORY_MONTHS:
        cov = np.eye(n) * fallback_var

        if obs >= 2:
            sample_var = hist.var(skipna=True)

            for i, ticker in enumerate(candidates):
                v = sample_var.get(ticker, np.nan)

                if pd.notna(v) and v > 0:
                    cov[i, i] = float(v)

        return cov, obs, "fallback_insufficient_history"

    hist = hist.copy()

    for col in hist.columns:
        hist[col] = hist[col].fillna(hist[col].mean())

    partial_cov = hist.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cov = np.eye(n) * fallback_var

    for i, ti in enumerate(candidates):
        for j, tj in enumerate(candidates):
            if ti in partial_cov.index and tj in partial_cov.columns:
                cov[i, j] = float(partial_cov.loc[ti, tj])

    cov = (cov + cov.T) / 2.0

    for i in range(n):
        if not np.isfinite(cov[i, i]) or cov[i, i] <= 0:
            cov[i, i] = fallback_var

    cov += np.eye(n) * 1e-10

    return cov, obs, "sample_covariance"


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(weights.T @ cov @ weights)


def annualized_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return math.sqrt(max(12.0 * portfolio_variance(weights, cov), 0.0))


def turnover(
    weights: np.ndarray,
    candidates: Sequence[str],
    previous_weights: Dict[str, float],
) -> float:
    prev_vec = np.array([previous_weights.get(t, 0.0) for t in candidates])
    candidate_set = set(candidates)

    residual_sold = sum(abs(w) for t, w in previous_weights.items() if t not in candidate_set)

    return 0.5 * (float(np.sum(np.abs(weights - prev_vec))) + residual_sold)


def project_to_bounds_sum(
    weights: np.ndarray,
    lower: float,
    upper: float,
    scores: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, lower, upper)

    s = np.asarray(scores, dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = s - np.min(s)

    if np.sum(s) <= 1e-12:
        s = np.ones_like(s)

    for _ in range(max_iter):
        diff = 1.0 - float(np.sum(w))

        if abs(diff) <= 1e-12:
            break

        if diff > 0:
            room = np.maximum(upper - w, 0.0)

            if np.sum(room) <= 1e-12:
                break

            preference = room * (1.0 + s / np.sum(s))
            w += diff * preference / np.sum(preference)

        else:
            reducible = np.maximum(w - lower, 0.0)

            if np.sum(reducible) <= 1e-12:
                break

            w -= (-diff) * reducible / np.sum(reducible)

        w = np.clip(w, lower, upper)

    total = np.sum(w)

    if total > 0:
        w = w / total

    return np.clip(w, lower, upper)


def initial_weights(
    candidates: Sequence[str],
    previous_weights: Dict[str, float],
    expected_returns: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    prev = np.array([previous_weights.get(t, 0.0) for t in candidates], dtype=float)

    if np.sum(prev) <= 1e-12:
        base = np.ones(len(candidates)) / len(candidates)
    else:
        base = np.maximum(prev, lower)

    return project_to_bounds_sum(base, lower, upper, expected_returns)


def objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov: np.ndarray,
    candidates: Sequence[str],
    previous_weights: Dict[str, float],
) -> float:
    expected = float(weights @ expected_returns)
    variance = portfolio_variance(weights, cov)
    turn = turnover(weights, candidates, previous_weights)

    utility = expected - RISK_AVERSION * variance - TURNOVER_PENALTY * turn

    return -utility


def constraint_list(
    candidates: Sequence[str],
    previous_weights: Dict[str, float],
    cov: np.ndarray,
    include_vol_constraint: bool,
) -> List[dict]:
    constraints: List[dict] = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w)) - 1.0,
        },
        {
            "type": "ineq",
            "fun": lambda w: MAX_MONTHLY_TURNOVER - turnover(w, candidates, previous_weights),
        },
    ]

    if include_vol_constraint:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: TARGET_VOL_ANNUAL**2 - 12.0 * portfolio_variance(w, cov),
            }
        )

    if MAX_BOND_ALLOCATION is not None:
        idx = [i for i, t in enumerate(candidates) if is_bond(t)]

        if idx:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, idx=idx: MAX_BOND_ALLOCATION - float(np.sum(w[idx])),
                }
            )

    if MAX_COMMODITY_ALLOCATION is not None:
        idx = [i for i, t in enumerate(candidates) if is_commodity(t)]

        if idx:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, idx=idx: MAX_COMMODITY_ALLOCATION - float(np.sum(w[idx])),
                }
            )

    return constraints


def optimize_one_month(
    candidates: Sequence[str],
    expected_returns: np.ndarray,
    cov: np.ndarray,
    previous_weights: Dict[str, float],
    lower: float,
    upper: float,
) -> Tuple[np.ndarray, bool, str, str, bool, bool, float]:
    x0 = initial_weights(candidates, previous_weights, expected_returns, lower, upper)

    if minimize is None:
        return (
            x0,
            False,
            "fallback_no_scipy",
            "scipy is unavailable; used fallback weights",
            True,
            True,
            np.nan,
        )

    bounds = [(lower, upper) for _ in candidates]

    def solve(include_vol_constraint: bool):
        return minimize(
            objective,
            x0,
            args=(expected_returns, cov, candidates, previous_weights),
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_list(
                candidates,
                previous_weights,
                cov,
                include_vol_constraint,
            ),
            options={
                "maxiter": 1000,
                "ftol": 1e-10,
                "disp": False,
            },
        )

    first = solve(include_vol_constraint=True)

    if first.success:
        w = project_to_bounds_sum(first.x, lower, upper, expected_returns)

        return (
            w,
            True,
            "optimized",
            str(first.message),
            False,
            False,
            -float(first.fun),
        )

    second = solve(include_vol_constraint=False)

    if second.success:
        w = project_to_bounds_sum(second.x, lower, upper, expected_returns)

        return (
            w,
            True,
            "optimized_target_vol_relaxed",
            (
                "Target volatility constraint was relaxed. "
                f"First message: {first.message}. "
                f"Second message: {second.message}."
            ),
            True,
            False,
            -float(second.fun),
        )

    return (
        x0,
        False,
        "fallback_solver_failed",
        (f"SLSQP failed. First message: {first.message}. Second message: {second.message}."),
        True,
        True,
        np.nan,
    )


# ---------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------


def optimize_all_months(
    predictions: pd.DataFrame,
    return_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weight_rows: List[dict] = []
    diagnostic_rows: List[dict] = []

    previous_weights: Dict[str, float] = {}

    for month in sorted(predictions["month_end_date"].unique()):
        month = pd.Timestamp(month)
        month_df = predictions[predictions["month_end_date"] == month].copy()

        if month_df.empty:
            continue

        candidates = choose_candidates(month_df, previous_weights)

        if not candidates:
            continue

        expected_map = month_df.set_index("ticker")["predicted_return"].to_dict()

        if "actual_next_return" in month_df.columns:
            actual_map = month_df.set_index("ticker")["actual_next_return"].to_dict()
        else:
            actual_map = {}

        expected_returns = np.array(
            [expected_map.get(t, 0.0) for t in candidates],
            dtype=float,
        )

        expected_returns = np.clip(expected_returns, -0.25, 0.25)

        cov, cov_obs, cov_status = estimate_covariance(return_matrix, month, candidates)
        lower, upper, bounds_status = effective_bounds(len(candidates))

        (
            weights,
            success,
            status,
            message,
            vol_relaxed,
            used_fallback,
            objective_value,
        ) = optimize_one_month(
            candidates,
            expected_returns,
            cov,
            previous_weights,
            lower,
            upper,
        )

        weights = project_to_bounds_sum(weights, lower, upper, expected_returns)

        expected_monthly_return = float(weights @ expected_returns)
        expected_annual_return = (1.0 + expected_monthly_return) ** 12 - 1.0

        realized_next_return = np.nan

        if actual_map:
            actual_values = np.array(
                [actual_map.get(t, np.nan) for t in candidates],
                dtype=float,
            )

            if not np.all(np.isnan(actual_values)):
                realized_next_return = float(weights @ np.nan_to_num(actual_values, nan=0.0))

        turn = turnover(weights, candidates, previous_weights)
        variance = portfolio_variance(weights, cov)
        vol = annualized_vol(weights, cov)

        previous_snapshot = previous_weights.copy()
        new_previous_weights: Dict[str, float] = {}

        for ticker, weight, pred in zip(candidates, weights, expected_returns):
            if weight > 1e-10:
                new_previous_weights[ticker] = float(weight)

            if is_bond(ticker):
                asset_group = "bond"
            elif is_commodity(ticker):
                asset_group = "commodity"
            else:
                asset_group = "other"

            weight_rows.append(
                {
                    "month_end_date": month.date().isoformat(),
                    "ticker": ticker,
                    "predicted_return": float(pred),
                    "previous_weight": float(previous_snapshot.get(ticker, 0.0)),
                    "optimized_weight": float(weight),
                    "selected_flag": 1,
                    "asset_group": asset_group,
                    "expected_return_contribution": float(weight * pred),
                    "portfolio_rule": "risk_constrained_optimization",
                }
            )

        bond_allocation = float(sum(w for t, w in zip(candidates, weights) if is_bond(t)))

        commodity_allocation = float(sum(w for t, w in zip(candidates, weights) if is_commodity(t)))

        diagnostic_rows.append(
            {
                "month_end_date": month.date().isoformat(),
                "n_selected_assets": len(candidates),
                "expected_monthly_return": expected_monthly_return,
                "expected_annual_return": expected_annual_return,
                "realized_next_return": realized_next_return,
                "portfolio_variance_monthly": variance,
                "annualized_volatility": vol,
                "target_vol_annual": TARGET_VOL_ANNUAL,
                "target_vol_constraint_relaxed": int(vol_relaxed),
                "turnover": turn,
                "max_monthly_turnover": MAX_MONTHLY_TURNOVER,
                "max_weight": float(np.max(weights)),
                "min_positive_weight": float(np.min(weights[weights > 1e-10])),
                "requested_max_single_etf_weight": MAX_SINGLE_ETF_WEIGHT,
                "requested_min_selected_etf_weight": MIN_SELECTED_ETF_WEIGHT,
                "effective_max_single_etf_weight": upper,
                "effective_min_selected_etf_weight": lower,
                "bounds_status": bounds_status,
                "bond_allocation": bond_allocation,
                "commodity_allocation": commodity_allocation,
                "max_bond_allocation_constraint": MAX_BOND_ALLOCATION,
                "max_commodity_allocation_constraint": MAX_COMMODITY_ALLOCATION,
                "objective_value": objective_value,
                "optimizer_success": int(success),
                "optimizer_status": status,
                "optimizer_message": message,
                "used_fallback": int(used_fallback),
                "covariance_observations": cov_obs,
                "covariance_status": cov_status,
                "risk_aversion": RISK_AVERSION,
                "turnover_penalty": TURNOVER_PENALTY,
            }
        )

        previous_weights = new_previous_weights

    return pd.DataFrame(weight_rows), pd.DataFrame(diagnostic_rows)


def main() -> int:
    print("=" * 72)
    print("Step 15 — Risk-Constrained Portfolio Optimizer")
    print("=" * 72)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        predictions = load_predictions()
        return_matrix = load_return_matrix(predictions)

        weights_df, diagnostics_df = optimize_all_months(predictions, return_matrix)

        if weights_df.empty or diagnostics_df.empty:
            raise ValueError("No optimizer output was produced. Check the prediction file.")

        weights_df = weights_df.sort_values(
            ["month_end_date", "optimized_weight"],
            ascending=[True, False],
        )

        diagnostics_df = diagnostics_df.sort_values("month_end_date")

        weights_df.to_csv(OUTPUT_WEIGHTS_PATH, index=False)
        diagnostics_df.to_csv(OUTPUT_DIAGNOSTICS_PATH, index=False)

        print("\nCreated:")
        print(f"  {OUTPUT_WEIGHTS_PATH.relative_to(PROJECT_ROOT)}")
        print(f"  {OUTPUT_DIAGNOSTICS_PATH.relative_to(PROJECT_ROOT)}")

        print("\nSummary:")
        print(f"  Months optimized: {len(diagnostics_df):,}")
        print(f"  Average selected ETFs: {diagnostics_df['n_selected_assets'].mean():.2f}")
        print(f"  Average turnover: {diagnostics_df['turnover'].mean():.4f}")
        print(
            f"  Average annualized volatility: {diagnostics_df['annualized_volatility'].mean():.4f}"
        )
        print(
            f"  Target-volatility relaxed months: "
            f"{int(diagnostics_df['target_vol_constraint_relaxed'].sum()):,}"
        )
        print(f"  Fallback months: {int(diagnostics_df['used_fallback'].sum()):,}")

        print("\nLast five diagnostics:")
        print(
            diagnostics_df[
                [
                    "month_end_date",
                    "n_selected_assets",
                    "expected_monthly_return",
                    "annualized_volatility",
                    "turnover",
                    "optimizer_status",
                    "covariance_status",
                ]
            ]
            .tail()
            .to_string(index=False)
        )

    except Exception as exc:
        print("\nERROR: portfolio optimization failed.")
        print(exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
