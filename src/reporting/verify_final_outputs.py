from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

BACKTEST_RESULTS_PATH = RESULTS_DIR / "backtest_results.csv"
BACKTEST_SUMMARY_PATH = RESULTS_DIR / "backtest_summary.json"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
WALK_FORWARD_PATH = RESULTS_DIR / "walk_forward_predictions.csv"
PORTFOLIO_WEIGHTS_PATH = RESULTS_DIR / "portfolio_weights.csv"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_date_column(df: pd.DataFrame) -> str | None:
    return find_first_existing_column(
        df,
        [
            "month_end_date",
            "date",
            "period",
            "rebalance_date",
            "test_date",
        ],
    )


def infer_return_column(df: pd.DataFrame) -> str | None:
    return find_first_existing_column(
        df,
        [
            "net_return",
            "portfolio_net_return",
            "strategy_net_return",
            "strategy_return",
            "portfolio_return",
            "monthly_return",
            "return",
            "gross_return",
        ],
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def calculate_performance_summary(returns: pd.Series) -> dict:
    returns = safe_numeric(returns).dropna()

    if returns.empty:
        return {
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "calmar_ratio": None,
            "hit_rate": None,
            "best_month": None,
            "worst_month": None,
            "number_of_months": 0,
        }

    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0

    n_months = len(returns)
    annualized_return = cumulative.iloc[-1] ** (12 / n_months) - 1
    annualized_volatility = returns.std(ddof=1) * np.sqrt(12) if n_months > 1 else 0.0

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility and annualized_volatility > 0
        else None
    )

    downside = returns[returns < 0]
    downside_volatility = downside.std(ddof=1) * np.sqrt(12) if len(downside) > 1 else 0.0

    sortino_ratio = (
        annualized_return / downside_volatility
        if downside_volatility and downside_volatility > 0
        else None
    )

    max_drawdown = drawdown.min()
    calmar_ratio = (
        annualized_return / abs(max_drawdown) if max_drawdown and max_drawdown < 0 else None
    )

    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": None if sharpe_ratio is None else float(sharpe_ratio),
        "sortino_ratio": None if sortino_ratio is None else float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": None if calmar_ratio is None else float(calmar_ratio),
        "hit_rate": float((returns > 0).mean()),
        "best_month": float(returns.max()),
        "worst_month": float(returns.min()),
        "number_of_months": int(n_months),
    }


def create_backtest_summary_and_return_series() -> tuple[pd.DataFrame, str, str]:
    if not BACKTEST_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {BACKTEST_RESULTS_PATH}")

    df = pd.read_csv(BACKTEST_RESULTS_PATH)

    if df.empty:
        raise ValueError("results/backtest_results.csv is empty.")

    date_col = infer_date_column(df)
    return_col = infer_return_column(df)

    if date_col is None:
        raise ValueError(
            "Could not find a date column in backtest_results.csv. "
            "Expected one of: month_end_date, date, period, rebalance_date, test_date."
        )

    if return_col is None:
        raise ValueError(
            "Could not find a return column in backtest_results.csv. "
            "Expected one of: net_return, portfolio_net_return, strategy_return, "
            "portfolio_return, monthly_return, return, gross_return."
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[return_col] = safe_numeric(df[return_col])
    df = df.dropna(subset=[date_col, return_col]).sort_values(date_col)

    summary = calculate_performance_summary(df[return_col])

    if "transaction_cost" in df.columns:
        summary["transaction_cost_drag"] = float(safe_numeric(df["transaction_cost"]).sum())
    elif "transaction_cost_drag" in df.columns:
        summary["transaction_cost_drag"] = float(safe_numeric(df["transaction_cost_drag"]).sum())
    else:
        summary["transaction_cost_drag"] = None

    summary["source_file"] = "results/backtest_results.csv"
    summary["date_column"] = date_col
    summary["return_column"] = return_col

    with BACKTEST_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df, date_col, return_col


def plot_cumulative_returns(df: pd.DataFrame, date_col: str, return_col: str) -> None:
    plot_df = df[[date_col, return_col]].copy()
    plot_df = plot_df.dropna().sort_values(date_col)
    plot_df["cumulative_return"] = (1.0 + plot_df[return_col]).cumprod() - 1.0

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df[date_col], plot_df["cumulative_return"], label="Strategy")
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cumulative_returns.png", dpi=150)
    plt.close()


def plot_drawdown_curve(df: pd.DataFrame, date_col: str, return_col: str) -> None:
    plot_df = df[[date_col, return_col]].copy()
    plot_df = plot_df.dropna().sort_values(date_col)

    cumulative = (1.0 + plot_df[return_col]).cumprod()
    running_max = cumulative.cummax()
    plot_df["drawdown"] = cumulative / running_max - 1.0

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df[date_col], plot_df["drawdown"], label="Drawdown")
    plt.title("Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "drawdown_curve.png", dpi=150)
    plt.close()


def plot_feature_importance() -> None:
    if not FEATURE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {FEATURE_IMPORTANCE_PATH}")

    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    if df.empty:
        raise ValueError("results/feature_importance.csv is empty.")

    feature_col = find_first_existing_column(
        df,
        ["feature", "feature_name", "variable", "column"],
    )

    importance_col = find_first_existing_column(
        df,
        ["importance", "mean_importance", "gain", "coefficient_abs", "abs_coefficient"],
    )

    if feature_col is None:
        feature_col = df.columns[0]

    if importance_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(
                "Could not find a numeric importance column in feature_importance.csv."
            )
        importance_col = numeric_cols[0]

    plot_df = df[[feature_col, importance_col]].copy()
    plot_df[importance_col] = safe_numeric(plot_df[importance_col])
    plot_df = plot_df.dropna()
    plot_df = plot_df.sort_values(importance_col, ascending=False).head(20)
    plot_df = plot_df.sort_values(importance_col, ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df[feature_col].astype(str), plot_df[importance_col])
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()


def plot_walk_forward_rank_ic() -> None:
    if not WALK_FORWARD_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {WALK_FORWARD_PATH}")

    df = pd.read_csv(WALK_FORWARD_PATH)

    if df.empty:
        raise ValueError("results/walk_forward_predictions.csv is empty.")

    date_col = infer_date_column(df)

    if date_col is None:
        raise ValueError("Could not find a date column in walk_forward_predictions.csv.")

    required = ["predicted_return", "actual_next_return"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"walk_forward_predictions.csv is missing columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["predicted_return"] = safe_numeric(df["predicted_return"])
    df["actual_next_return"] = safe_numeric(df["actual_next_return"])
    df = df.dropna(subset=[date_col, "predicted_return", "actual_next_return"])

    ic_rows = []

    for date, group in df.groupby(date_col):
        if len(group) < 3:
            continue

        rank_ic = group["predicted_return"].corr(
            group["actual_next_return"],
            method="spearman",
        )

        ic_rows.append(
            {
                "month_end_date": date,
                "rank_ic": rank_ic,
            }
        )

    ic_df = pd.DataFrame(ic_rows)

    if ic_df.empty:
        raise ValueError(
            "Could not calculate walk-forward rank IC. Not enough cross-sectional data."
        )

    ic_df = ic_df.dropna().sort_values("month_end_date")
    ic_df["rank_ic_6m_average"] = ic_df["rank_ic"].rolling(6, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(ic_df["month_end_date"], ic_df["rank_ic"], label="Monthly Rank IC")
    plt.plot(ic_df["month_end_date"], ic_df["rank_ic_6m_average"], label="6-Month Average")
    plt.axhline(0.0, linewidth=1)
    plt.title("Walk-Forward Rank IC")
    plt.xlabel("Date")
    plt.ylabel("Spearman Rank IC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "walk_forward_rank_ic.png", dpi=150)
    plt.close()


def plot_portfolio_weights_over_time() -> None:
    if not PORTFOLIO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {PORTFOLIO_WEIGHTS_PATH}")

    df = pd.read_csv(PORTFOLIO_WEIGHTS_PATH)

    if df.empty:
        raise ValueError("results/portfolio_weights.csv is empty.")

    date_col = infer_date_column(df)

    if date_col is None:
        raise ValueError("Could not find a date column in portfolio_weights.csv.")

    required = ["ticker", "weight"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"portfolio_weights.csv is missing columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["weight"] = safe_numeric(df["weight"])
    df = df.dropna(subset=[date_col, "ticker", "weight"])

    pivot = (
        df.pivot_table(
            index=date_col,
            columns="ticker",
            values="weight",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )

    top_tickers = pivot.mean().sort_values(ascending=False).head(10).index.tolist()
    plot_df = pivot[top_tickers]

    plt.figure(figsize=(11, 6))
    for ticker in plot_df.columns:
        plt.plot(plot_df.index, plot_df[ticker], label=str(ticker))

    plt.title("Portfolio Weights Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "portfolio_weights_over_time.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()

    backtest_df, date_col, return_col = create_backtest_summary_and_return_series()

    plot_cumulative_returns(backtest_df, date_col, return_col)
    plot_drawdown_curve(backtest_df, date_col, return_col)
    plot_feature_importance()
    plot_walk_forward_rank_ic()
    plot_portfolio_weights_over_time()

    print("Created final GitHub outputs:")
    print(f"- {BACKTEST_SUMMARY_PATH}")
    print(f"- {FIGURES_DIR / 'cumulative_returns.png'}")
    print(f"- {FIGURES_DIR / 'drawdown_curve.png'}")
    print(f"- {FIGURES_DIR / 'feature_importance.png'}")
    print(f"- {FIGURES_DIR / 'walk_forward_rank_ic.png'}")
    print(f"- {FIGURES_DIR / 'portfolio_weights_over_time.png'}")


if __name__ == "__main__":
    main()
