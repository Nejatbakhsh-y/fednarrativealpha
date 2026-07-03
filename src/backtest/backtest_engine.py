"""
Backtest engine for FedNarrativeAlpha.

Run:
    python src/backtest/backtest_engine.py

Outputs:
    results/backtest_engine_monthly_returns.csv
    results/backtest_engine_performance_summary.csv
    results/backtest_engine_aligned_performance_summary.csv
    results/backtest_engine_model_comparison.csv
    results/backtest_engine_diagnostics.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

PREDICTIONS = RESULTS / "walk_forward_predictions.csv"
RAW_WEIGHTS = RESULTS / "raw_signal_weights.csv"
OPT_WEIGHTS = RESULTS / "portfolio_weights.csv"
MASTER = PROCESSED / "master_modeling_dataset.parquet"
PRICE_FEATURES = INTERIM / "price_features_monthly.parquet"

OUT_MONTHLY = RESULTS / "backtest_engine_monthly_returns.csv"
OUT_SUMMARY = RESULTS / "backtest_engine_performance_summary.csv"
OUT_ALIGNED = RESULTS / "backtest_engine_aligned_performance_summary.csv"
OUT_COMPARE = RESULTS / "backtest_engine_model_comparison.csv"
OUT_DIAG = RESULTS / "backtest_engine_diagnostics.csv"

DATE_COLS = ["month_end_date", "date", "period_end", "rebalance_date", "test_date"]
TICKER_COLS = ["ticker", "symbol", "asset", "etf"]
RET_COLS = [
    "actual_next_return",
    "next_1m_return",
    "forward_1m_return",
    "realized_return",
    "net_return",
    "gross_return",
    "return",
    "monthly_return",
]
PRED_COLS = ["predicted_return", "prediction", "y_pred", "forecast", "expected_return"]
WEIGHT_COLS = ["weight", "portfolio_weight", "target_weight", "optimized_weight", "signal_weight"]
STRATEGY_COLS = ["strategy_name", "strategy", "portfolio_rule", "model_strategy", "model_name", "feature_set"]
MOMENTUM_COLS = ["momentum_12m_minus_1m", "return_12m", "return_6m", "return_3m", "return_1m"]
BOND_TICKERS = ["AGG", "BND", "IEF", "TLT", "SHY", "IEI", "LQD", "HYG"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add_diag(diag: list[dict], item: str, status: str, detail: str) -> None:
    diag.append({"item": item, "status": status, "detail": detail})


def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def read_table(path: Path, diag: list[dict]) -> pd.DataFrame:
    if not path.exists():
        add_diag(diag, rel(path), "missing", "Optional input not found.")
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            add_diag(diag, rel(path), "skipped", f"Unsupported file type: {path.suffix}")
            return pd.DataFrame()
        add_diag(diag, rel(path), "loaded", f"{len(df):,} rows and {len(df.columns):,} columns.")
        return df
    except Exception as exc:
        add_diag(diag, rel(path), "error", f"{type(exc).__name__}: {exc}")
        return pd.DataFrame()


def standardize(df: pd.DataFrame, source: str, diag: list[dict]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    date_col = first_col(df, DATE_COLS)
    ticker_col = first_col(df, TICKER_COLS)
    if date_col is None or ticker_col is None:
        add_diag(
            diag,
            source,
            "skipped",
            f"Missing date or ticker column. Date candidates={DATE_COLS}; ticker candidates={TICKER_COLS}.",
        )
        return pd.DataFrame()
    df["month_end_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["ticker"] = df[ticker_col].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["month_end_date"])
    df = df[df["ticker"].ne("") & df["ticker"].ne("NAN")]
    return df


def as_decimal_return(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    non_missing = x.dropna()
    if non_missing.empty:
        return x
    if non_missing.abs().median() > 1.0 or non_missing.abs().quantile(0.95) > 5.0:
        return x / 100.0
    return x


def normalize_weights(w: pd.DataFrame) -> pd.DataFrame:
    if w.empty:
        return w
    w = w.copy()
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
    w = w[w["weight"] > 0].copy()
    if w.empty:
        return w
    total = w.groupby(["strategy_name", "month_end_date"])["weight"].transform("sum")
    w = w[total > 0].copy()
    total = w.groupby(["strategy_name", "month_end_date"])["weight"].transform("sum")
    w["weight"] = w["weight"] / total
    return w


def load_return_panel(diag: list[dict]) -> pd.DataFrame:
    frames = []
    for path, priority in [(PREDICTIONS, 0), (MASTER, 1), (PRICE_FEATURES, 2)]:
        raw = read_table(path, diag)
        df = standardize(raw, rel(path), diag)
        if df.empty:
            continue
        ret_col = first_col(df, RET_COLS)
        if ret_col is None:
            add_diag(diag, rel(path), "skipped", f"No realized return column found. Candidates={RET_COLS}.")
            continue
        tmp = df[["month_end_date", "ticker", ret_col]].rename(columns={ret_col: "actual_next_return"})
        tmp["actual_next_return"] = as_decimal_return(tmp["actual_next_return"])
        tmp = tmp.dropna(subset=["actual_next_return"])
        tmp = tmp.groupby(["month_end_date", "ticker"], as_index=False)["actual_next_return"].mean()
        tmp["source_priority"] = priority
        frames.append(tmp)

    if not frames:
        raise FileNotFoundError(
            "No realized returns found. Expected actual_next_return or next_1m_return "
            "in results/walk_forward_predictions.csv, data/processed/master_modeling_dataset.parquet, "
            "or data/interim/price_features_monthly.parquet."
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["month_end_date", "ticker", "source_priority"])
    panel = panel.drop_duplicates(["month_end_date", "ticker"], keep="first")
    panel = panel[["month_end_date", "ticker", "actual_next_return"]].copy()
    add_diag(diag, "return_panel", "created", f"{len(panel):,} month/ticker rows.")
    return panel


def load_predictions(diag: list[dict]) -> pd.DataFrame:
    raw = read_table(PREDICTIONS, diag)
    df = standardize(raw, rel(PREDICTIONS), diag)
    if df.empty:
        return pd.DataFrame()
    pred_col = first_col(df, PRED_COLS)
    if pred_col is None:
        add_diag(diag, rel(PREDICTIONS), "skipped", f"No prediction column found. Candidates={PRED_COLS}.")
        return pd.DataFrame()
    df["predicted_return"] = as_decimal_return(df[pred_col])
    if "model_name" not in df.columns:
        df["model_name"] = "model"
    if "feature_set" not in df.columns:
        df["feature_set"] = "unknown_feature_set"
    df["model_text"] = (df["model_name"].astype(str) + " " + df["feature_set"].astype(str)).str.lower()
    return df.dropna(subset=["predicted_return"])


def file_weights(path: Path, default_name: str, diag: list[dict]) -> pd.DataFrame:
    raw = read_table(path, diag)
    df = standardize(raw, rel(path), diag)
    if df.empty:
        return pd.DataFrame()
    weight_col = first_col(df, WEIGHT_COLS)
    if weight_col is None:
        add_diag(diag, rel(path), "skipped", f"No weight column found. Candidates={WEIGHT_COLS}.")
        return pd.DataFrame()
    if "selected_flag" in df.columns:
        selected = pd.to_numeric(df["selected_flag"], errors="coerce").fillna(0)
        df = df[(selected > 0) | (pd.to_numeric(df[weight_col], errors="coerce") > 0)].copy()
    strategy_col = first_col(df, STRATEGY_COLS)
    if strategy_col:
        df["strategy_name"] = default_name + " - " + df[strategy_col].astype(str).str.replace("_", " ", regex=False)
    else:
        df["strategy_name"] = default_name
    out = df[["month_end_date", "ticker", "strategy_name", weight_col]].rename(columns={weight_col: "weight"})
    return normalize_weights(out)


def top_n_weights(scores: pd.DataFrame, strategy: str, score_col: str, top_n: int) -> pd.DataFrame:
    if scores.empty or score_col not in scores.columns:
        return pd.DataFrame()
    x = scores[["month_end_date", "ticker", score_col]].copy()
    x[score_col] = pd.to_numeric(x[score_col], errors="coerce")
    x = x.dropna(subset=[score_col])
    if x.empty:
        return pd.DataFrame()
    x = x.sort_values(["month_end_date", score_col, "ticker"], ascending=[True, False, True])
    x["rank"] = x.groupby("month_end_date")[score_col].rank(method="first", ascending=False)
    x = x[x["rank"] <= top_n].copy()
    n = x.groupby("month_end_date")["ticker"].transform("nunique")
    x["strategy_name"] = strategy
    x["weight"] = 1.0 / n
    return x[["month_end_date", "ticker", "strategy_name", "weight"]]


def prediction_strategy(pred: pd.DataFrame, strategy: str, include: list[str], exclude: list[str], top_n: int) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    text = pred["model_text"].fillna("").astype(str)
    mask = np.ones(len(pred), dtype=bool) if not include else np.zeros(len(pred), dtype=bool)
    for p in include:
        mask |= text.str.contains(p.lower(), regex=False).to_numpy()
    for p in exclude:
        mask &= ~text.str.contains(p.lower(), regex=False).to_numpy()
    x = pred[mask].copy()
    if x.empty:
        return pd.DataFrame()
    scores = x.groupby(["month_end_date", "ticker"], as_index=False)["predicted_return"].mean()
    return top_n_weights(scores, strategy, "predicted_return", top_n)


def equal_weight(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel[["month_end_date", "ticker"]].copy()
    n = x.groupby("month_end_date")["ticker"].transform("nunique")
    x["strategy_name"] = "Equal-weight ETF universe"
    x["weight"] = 1.0 / n
    return x


def single_ticker(panel: pd.DataFrame, ticker: str, strategy: str) -> pd.DataFrame:
    x = panel.loc[panel["ticker"] == ticker.upper(), ["month_end_date", "ticker"]].copy()
    if x.empty:
        return x
    x["strategy_name"] = strategy
    x["weight"] = 1.0
    return x


def sixty_forty(panel: pd.DataFrame) -> pd.DataFrame:
    tickers = set(panel["ticker"].unique())
    if "SPY" not in tickers:
        return pd.DataFrame()
    bond = next((t for t in BOND_TICKERS if t in tickers), None)
    if bond is None:
        return pd.DataFrame()
    spy_dates = set(panel.loc[panel["ticker"] == "SPY", "month_end_date"])
    bond_dates = set(panel.loc[panel["ticker"] == bond, "month_end_date"])
    rows = []
    for dt in sorted(spy_dates & bond_dates):
        rows.append({"month_end_date": dt, "ticker": "SPY", "strategy_name": "60/40 portfolio", "weight": 0.60})
        rows.append({"month_end_date": dt, "ticker": bond, "strategy_name": "60/40 portfolio", "weight": 0.40})
    return pd.DataFrame(rows)


def momentum_strategy(diag: list[dict], top_n: int) -> pd.DataFrame:
    for path in [PRICE_FEATURES, MASTER]:
        raw = read_table(path, diag)
        df = standardize(raw, rel(path), diag)
        if df.empty:
            continue
        signal_col = first_col(df, MOMENTUM_COLS)
        if signal_col is None:
            add_diag(diag, rel(path), "skipped", f"No momentum signal column found. Candidates={MOMENTUM_COLS}.")
            continue
        scores = df[["month_end_date", "ticker", signal_col]].copy()
        scores[signal_col] = as_decimal_return(scores[signal_col])
        out = top_n_weights(scores, "Momentum-only strategy", signal_col, top_n)
        if not out.empty:
            add_diag(diag, "Momentum-only strategy", "created", f"Used {signal_col} from {rel(path)}.")
            return out
    return pd.DataFrame()


def add_strategy(collection: list[pd.DataFrame], strategy_df: pd.DataFrame, name: str, diag: list[dict]) -> None:
    if strategy_df is None or strategy_df.empty:
        add_diag(diag, name, "not_created", "Input data unavailable or no valid rows created.")
        return
    collection.append(strategy_df)
    add_diag(
        diag,
        name,
        "created",
        f"{len(strategy_df):,} weight rows across {strategy_df['month_end_date'].nunique():,} months.",
    )


def build_weights(panel: pd.DataFrame, pred: pd.DataFrame, diag: list[dict], top_n: int) -> pd.DataFrame:
    strategies: list[pd.DataFrame] = []

    add_strategy(strategies, single_ticker(panel, "SPY", "SPY"), "SPY", diag)
    add_strategy(strategies, sixty_forty(panel), "60/40 portfolio", diag)
    add_strategy(strategies, equal_weight(panel), "Equal-weight ETF universe", diag)
    add_strategy(strategies, momentum_strategy(diag, top_n), "Momentum-only strategy", diag)

    add_strategy(
        strategies,
        prediction_strategy(pred, "Macro-only strategy", ["macro only", "macro-only", "macro_only"], ["price", "fed", "text", "narrative"], top_n),
        "Macro-only strategy",
        diag,
    )

    price_macro = prediction_strategy(
        pred,
        "Price + macro model",
        ["price + macro", "price_macro", "model b"],
        ["fed", "text", "narrative"],
        top_n,
    )
    if price_macro.empty:
        price_macro = prediction_strategy(pred, "Price + macro model", ["macro"], ["fed", "text", "narrative"], top_n)
    add_strategy(strategies, price_macro, "Price + macro model", diag)

    add_strategy(
        strategies,
        prediction_strategy(
            pred,
            "Price + macro + Fed text model",
            ["price + macro + fed", "price_macro_fed", "fed text", "fed_text", "narrative", "model c"],
            [],
            top_n,
        ),
        "Price + macro + Fed text model",
        diag,
    )

    add_strategy(
        strategies,
        file_weights(OPT_WEIGHTS, "Optimized full AI/narrative portfolio", diag),
        "Optimized full AI/narrative portfolio",
        diag,
    )
    add_strategy(strategies, file_weights(RAW_WEIGHTS, "Raw signal portfolio", diag), "Raw signal portfolio", diag)

    if not strategies:
        raise RuntimeError("No strategy weights could be created.")

    out = pd.concat(strategies, ignore_index=True)
    out = normalize_weights(out)
    out = out.drop_duplicates(["strategy_name", "month_end_date", "ticker"], keep="last")
    add_diag(diag, "all_strategy_weights", "created", f"{out['strategy_name'].nunique():,} strategies and {len(out):,} rows.")
    return out


def turnover(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in weights.groupby("strategy_name"):
        p = g.pivot_table(index="month_end_date", columns="ticker", values="weight", aggfunc="sum", fill_value=0.0).sort_index()
        t = 0.5 * p.diff().abs().sum(axis=1)
        if not t.empty:
            t.iloc[0] = 0.0
        z = t.reset_index()
        z.columns = ["month_end_date", "monthly_turnover"]
        z["strategy_name"] = strategy
        rows.append(z)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def strategy_returns(weights: pd.DataFrame, panel: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    weights = normalize_weights(weights)
    merged = weights.merge(panel, on=["month_end_date", "ticker"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["weighted_return"] = merged["weight"] * merged["actual_next_return"]
    monthly = (
        merged.groupby(["strategy_name", "month_end_date"], as_index=False)
        .agg(gross_return=("weighted_return", "sum"), number_of_holdings=("ticker", "nunique"))
    )
    monthly = monthly.merge(turnover(weights), on=["strategy_name", "month_end_date"], how="left")
    monthly["monthly_turnover"] = monthly["monthly_turnover"].fillna(0.0)
    monthly["transaction_cost"] = monthly["monthly_turnover"] * cost_bps / 10_000.0
    monthly["net_return"] = monthly["gross_return"] - monthly["transaction_cost"]
    monthly["transaction_cost_bps"] = cost_bps
    return monthly.sort_values(["strategy_name", "month_end_date"]).reset_index(drop=True)


def ann_return(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return np.nan
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return np.nan
    return growth ** (12.0 / len(r)) - 1.0


def ann_vol(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    return float(r.std(ddof=1) * math.sqrt(12.0)) if len(r) >= 2 else np.nan


def max_dd(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in monthly.groupby("strategy_name"):
        g = g.sort_values("month_end_date").copy()
        net = g["net_return"]
        gross = g["gross_return"]
        ar = ann_return(net)
        gar = ann_return(gross)
        av = ann_vol(net)
        dd = max_dd(net)
        downside = net[net < 0]
        dvol = downside.std(ddof=1) * math.sqrt(12.0) if len(downside) >= 2 else np.nan
        best_idx = net.idxmax()
        worst_idx = net.idxmin()

        rows.append(
            {
                "strategy_name": strategy,
                "start_date": g["month_end_date"].min(),
                "end_date": g["month_end_date"].max(),
                "n_months": len(g),
                "annualized_return": ar,
                "gross_annualized_return": gar,
                "annualized_volatility": av,
                "sharpe_ratio": ar / av if pd.notna(ar) and pd.notna(av) and av != 0 else np.nan,
                "sortino_ratio": ar / dvol if pd.notna(ar) and pd.notna(dvol) and dvol != 0 else np.nan,
                "max_drawdown": dd,
                "calmar_ratio": ar / abs(dd) if pd.notna(ar) and pd.notna(dd) and dd < 0 else np.nan,
                "monthly_turnover": g["monthly_turnover"].mean(),
                "hit_rate": (net > 0).mean(),
                "best_month": net.max(),
                "best_month_date": g.loc[best_idx, "month_end_date"],
                "worst_month": net.min(),
                "worst_month_date": g.loc[worst_idx, "month_end_date"],
                "transaction_cost_drag": gar - ar if pd.notna(gar) and pd.notna(ar) else np.nan,
                "average_monthly_transaction_cost": g["transaction_cost"].mean(),
                "final_growth_of_1": float((1.0 + net).prod()),
            }
        )
    return pd.DataFrame(rows).sort_values(["annualized_return", "sharpe_ratio"], ascending=[False, False])


def aligned_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    sets = [set(g["month_end_date"]) for _, g in monthly.groupby("strategy_name")]
    if len(sets) < 2:
        return pd.DataFrame()
    common = set.intersection(*sets)
    if len(common) < 3:
        return pd.DataFrame()
    out = summarize(monthly[monthly["month_end_date"].isin(common)].copy())
    out.insert(1, "comparison_type", "common_date_aligned")
    return out


def compare_models(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    x = summary.copy()
    names = x["strategy_name"].astype(str).str.lower()
    full_mask = (
        names.str.contains("fed", regex=False)
        | names.str.contains("text", regex=False)
        | names.str.contains("narrative", regex=False)
        | names.str.contains("optimized", regex=False)
    )
    full = x[full_mask].sort_values("annualized_return", ascending=False)
    non_text = x[~full_mask].sort_values("annualized_return", ascending=False)
    if full.empty or non_text.empty:
        return pd.DataFrame()

    f = full.iloc[0]
    b = non_text.iloc[0]
    return pd.DataFrame(
        [
            {
                "question": "Does the full AI/narrative model outperform the best non-text model after transaction costs?",
                "full_model_strategy": f["strategy_name"],
                "best_non_text_strategy": b["strategy_name"],
                "full_model_annualized_return": f["annualized_return"],
                "best_non_text_annualized_return": b["annualized_return"],
                "annualized_return_difference": f["annualized_return"] - b["annualized_return"],
                "full_model_sharpe_ratio": f["sharpe_ratio"],
                "best_non_text_sharpe_ratio": b["sharpe_ratio"],
                "sharpe_ratio_difference": f["sharpe_ratio"] - b["sharpe_ratio"],
                "full_model_max_drawdown": f["max_drawdown"],
                "best_non_text_max_drawdown": b["max_drawdown"],
                "outperforms_after_transaction_costs": bool(f["annualized_return"] > b["annualized_return"]),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    diag: list[dict] = []
    add_diag(diag, "configuration", "started", f"cost_bps={args.cost_bps}; top_n={args.top_n}")

    panel = load_return_panel(diag)
    pred = load_predictions(diag)
    weights = build_weights(panel, pred, diag, args.top_n)
    monthly = strategy_returns(weights, panel, args.cost_bps)

    if monthly.empty:
        raise RuntimeError("No monthly strategy returns were created. Check date/ticker alignment.")

    summary = summarize(monthly)
    aligned = aligned_summary(monthly)
    comparison = compare_models(aligned if not aligned.empty else summary)

    monthly.to_csv(OUT_MONTHLY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    if aligned.empty:
        pd.DataFrame([{"status": "not_created", "detail": "Fewer than three common dates across all strategies."}]).to_csv(OUT_ALIGNED, index=False)
    else:
        aligned.to_csv(OUT_ALIGNED, index=False)

    if comparison.empty:
        pd.DataFrame([{"status": "not_created", "detail": "Could not identify both a full AI/narrative model and a non-text comparator."}]).to_csv(OUT_COMPARE, index=False)
    else:
        comparison.to_csv(OUT_COMPARE, index=False)

    pd.DataFrame(diag).to_csv(OUT_DIAG, index=False)

    print("Backtest engine complete.")
    print(f"Monthly returns:      {rel(OUT_MONTHLY)}")
    print(f"Performance summary: {rel(OUT_SUMMARY)}")
    print(f"Aligned summary:     {rel(OUT_ALIGNED)}")
    print(f"Model comparison:    {rel(OUT_COMPARE)}")
    print(f"Diagnostics:         {rel(OUT_DIAG)}")

    if not comparison.empty:
        row = comparison.iloc[0]
        verdict = "YES" if row["outperforms_after_transaction_costs"] else "NO"
        print("")
        print("Main result")
        print(f"Question: {row['question']}")
        print(f"Answer: {verdict}")
        print(f"Full model: {row['full_model_strategy']}")
        print(f"Best non-text: {row['best_non_text_strategy']}")
        print(f"Annualized return difference: {row['annualized_return_difference']:.4f}")
        print(f"Sharpe ratio difference: {row['sharpe_ratio_difference']:.4f}")


if __name__ == "__main__":
    main()
