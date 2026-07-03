"""
Generate reproducible Markdown reports for the FedNarrativeAlpha project.

Outputs
-------
reports/final_research_report.md
reports/model_card.md

Run from the repository root:

    python src/reporting/make_research_report.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


PROJECT_NAME = "FedNarrativeAlpha: GenAI-Assisted Alpha Research Pipeline for Liquid ETF Markets"

EXPECTED_FILES: Dict[str, str] = {
    "Price features": "data/interim/price_features_monthly.parquet",
    "Macro features": "data/interim/macro_features_monthly.parquet",
    "Fed text features": "data/interim/fed_text_features_monthly.parquet",
    "Master modeling dataset": "data/processed/master_modeling_dataset.parquet",
    "Selected features": "results/selected_features.csv",
    "Baseline metrics": "results/baseline_model_metrics.json",
    "ML model metrics": "results/model_metrics.json",
    "Walk-forward predictions": "results/walk_forward_predictions.csv",
    "Raw signal weights": "results/raw_signal_weights.csv",
    "Portfolio weights": "results/portfolio_weights.csv",
    "Optimization diagnostics": "results/optimization_diagnostics.csv",
    "Transaction-cost results": "results/backtest_results.csv",
    "Backtest summary": "results/backtest_summary.csv",
}

PERFORMANCE_COLUMNS: List[str] = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "monthly_turnover",
    "hit_rate",
    "best_month",
    "worst_month",
    "transaction_cost_drag",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.4f}"
    text = str(value)
    text = text.replace("\n", " ")
    text = text.replace("|", "\\|")
    return text


def markdown_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = []

    for row in rows:
        row_list = [format_value(value) for value in row]
        if len(row_list) < len(headers):
            row_list += [""] * (len(headers) - len(row_list))
        body_lines.append("| " + " | ".join(row_list[: len(headers)]) + " |")

    if not body_lines:
        body_lines.append("| " + " | ".join([""] * len(headers)) + " |")

    return "\n".join([header_line, separator] + body_lines)


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return None


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
        return {"value": data}
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return None


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def flatten_json(data: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_json(value, full_key))
        else:
            rows.append((full_key, value))
    return rows


def file_status_section(root: Path) -> str:
    rows = []
    for label, relative_path in EXPECTED_FILES.items():
        status = "Available" if (root / relative_path).exists() else "Missing"
        rows.append([label, relative_path, status])
    return markdown_table(["Artifact", "Path", "Status"], rows)


def summarize_json_file(root: Path, relative_path: str) -> str:
    data = read_json(root / relative_path)
    if not data:
        return f"No readable file found at `{relative_path}`."

    rows = flatten_json(data)
    if not rows:
        return f"`{relative_path}` was found, but it did not contain readable metric fields."

    return markdown_table(["Metric", "Value"], rows[:40])


def summarize_master_dataset(root: Path) -> str:
    path = root / "data/processed/master_modeling_dataset.parquet"
    if not path.exists():
        return (
            "No master modeling dataset found at `data/processed/master_modeling_dataset.parquet`."
        )

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return f"The master dataset exists, but it could not be read: `{exc}`."

    rows: List[List[Any]] = [
        ["Rows", len(df)],
        ["Columns", len(df.columns)],
    ]

    if "ticker" in df.columns:
        rows.append(["Unique tickers", df["ticker"].nunique()])

    if "month_end_date" in df.columns:
        dates = safe_to_datetime(df["month_end_date"])
        if dates.notna().any():
            rows.append(["Earliest month", dates.min().date()])
            rows.append(["Latest month", dates.max().date()])

    target_columns = [
        column
        for column in ["next_1m_return", "next_1m_rank", "top_tercile_next_month"]
        if column in df.columns
    ]
    rows.append(
        [
            "Detected target columns",
            ", ".join(target_columns) if target_columns else "None detected",
        ]
    )

    return markdown_table(["Dataset attribute", "Value"], rows)


def summarize_selected_features(root: Path) -> str:
    path = root / "results/selected_features.csv"
    df = read_csv(path)

    if df is None:
        return "No selected-feature file found at `results/selected_features.csv`."

    rows: List[List[Any]] = [
        ["Rows", len(df)],
        ["Columns", len(df.columns)],
    ]

    if "selected_flag" in df.columns:
        selected = df["selected_flag"].astype(str).str.lower().isin(["1", "true", "yes"])
        rows.append(["Selected features", int(selected.sum())])

    if "feature_group" in df.columns:
        rows.append(["Feature groups", df["feature_group"].nunique()])

    output = markdown_table(["Feature-selection attribute", "Value"], rows)

    if "feature_name" in df.columns:
        sort_columns = [
            column
            for column in ["final_score", "selection_frequency", "mutual_information"]
            if column in df.columns
        ]

        if sort_columns:
            top_df = df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(15)
        else:
            top_df = df.head(15)

        display_columns = [
            column
            for column in [
                "feature_name",
                "feature_group",
                "mutual_information",
                "redundancy_penalty",
                "final_score",
                "selection_frequency",
                "selected_flag",
            ]
            if column in top_df.columns
        ]

        if display_columns:
            output += "\n\nTop feature records:\n\n"
            output += markdown_table(display_columns, top_df[display_columns].values.tolist())

    return output


def summarize_walk_forward(root: Path) -> str:
    path = root / "results/walk_forward_predictions.csv"
    df = read_csv(path)

    if df is None:
        return "No walk-forward prediction file found at `results/walk_forward_predictions.csv`."

    rows: List[List[Any]] = [
        ["Rows", len(df)],
        ["Columns", len(df.columns)],
    ]

    if "month_end_date" in df.columns:
        dates = safe_to_datetime(df["month_end_date"])
        if dates.notna().any():
            rows.append(["Earliest test month", dates.min().date()])
            rows.append(["Latest test month", dates.max().date()])

    if "ticker" in df.columns:
        rows.append(["Unique tickers", df["ticker"].nunique()])

    if "model_name" in df.columns:
        models = sorted(df["model_name"].dropna().astype(str).unique())
        rows.append(["Models", ", ".join(models)])

    if "feature_set" in df.columns:
        feature_sets = sorted(df["feature_set"].dropna().astype(str).unique())
        rows.append(["Feature sets", ", ".join(feature_sets)])

    output = markdown_table(["Walk-forward attribute", "Value"], rows)

    if {"actual_next_return", "predicted_return"}.issubset(df.columns):
        temp = df.copy()
        temp["actual_next_return"] = pd.to_numeric(temp["actual_next_return"], errors="coerce")
        temp["predicted_return"] = pd.to_numeric(temp["predicted_return"], errors="coerce")
        temp = temp.dropna(subset=["actual_next_return", "predicted_return"])

        if not temp.empty:
            corr = temp["actual_next_return"].corr(temp["predicted_return"])
            sign_hit = (temp["actual_next_return"] * temp["predicted_return"] > 0).mean()
            output += "\n\nOverall prediction diagnostics:\n\n"
            output += markdown_table(
                ["Metric", "Value"],
                [
                    ["Usable prediction rows", len(temp)],
                    ["Prediction/actual correlation", corr],
                    ["Sign hit rate", sign_hit],
                ],
            )

    return output


def summarize_portfolio_weights(root: Path) -> str:
    path = root / "results/portfolio_weights.csv"
    df = read_csv(path)

    if df is None:
        return "No portfolio-weight file found at `results/portfolio_weights.csv`."

    rows: List[List[Any]] = [
        ["Rows", len(df)],
        ["Columns", len(df.columns)],
    ]

    if "ticker" in df.columns:
        rows.append(["Unique ETFs", df["ticker"].nunique()])

    if "month_end_date" in df.columns:
        dates = safe_to_datetime(df["month_end_date"])
        if dates.notna().any():
            rows.append(["Earliest portfolio month", dates.min().date()])
            rows.append(["Latest portfolio month", dates.max().date()])

    if "weight" in df.columns:
        weights = pd.to_numeric(df["weight"], errors="coerce")
        rows.append(["Maximum observed weight", weights.max()])
        rows.append(["Average absolute weight", weights.abs().mean()])

    return markdown_table(["Portfolio attribute", "Value"], rows)


def summarize_optimization(root: Path) -> str:
    path = root / "results/optimization_diagnostics.csv"
    df = read_csv(path)

    if df is None:
        return "No optimization diagnostic file found at `results/optimization_diagnostics.csv`."

    rows: List[List[Any]] = [
        ["Rows", len(df)],
        ["Columns", len(df.columns)],
    ]

    output = markdown_table(["Optimization attribute", "Value"], rows)

    if "status" in df.columns:
        status_counts = df["status"].astype(str).value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        output += "\n\nOptimization status counts:\n\n"
        output += markdown_table(["Status", "Count"], status_counts.values.tolist())

    return output


def compute_performance(group: pd.DataFrame, return_column: str) -> Dict[str, Any]:
    returns = pd.to_numeric(group[return_column], errors="coerce").dropna()

    if returns.empty:
        return {column: None for column in PERFORMANCE_COLUMNS}

    n_months = len(returns)
    cumulative = (1.0 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1.0
    annualized_return = (1.0 + total_return) ** (12.0 / n_months) - 1.0

    monthly_volatility = returns.std(ddof=1) if n_months > 1 else 0.0
    annualized_volatility = monthly_volatility * math.sqrt(12.0)

    sharpe_ratio = None
    if annualized_volatility != 0:
        sharpe_ratio = annualized_return / annualized_volatility

    downside_returns = returns[returns < 0]
    downside_volatility = (
        downside_returns.std(ddof=1) * math.sqrt(12.0) if len(downside_returns) > 1 else None
    )

    sortino_ratio = None
    if downside_volatility not in [None, 0]:
        sortino_ratio = annualized_return / downside_volatility

    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1.0
    max_drawdown = drawdowns.min()

    calmar_ratio = None
    if max_drawdown != 0:
        calmar_ratio = annualized_return / abs(max_drawdown)

    turnover_column = first_existing_column(group, ["monthly_turnover", "turnover"])
    cost_column = first_existing_column(
        group,
        ["transaction_cost_drag", "transaction_cost", "cost_drag", "trading_cost"],
    )

    monthly_turnover = None
    if turnover_column:
        monthly_turnover = pd.to_numeric(group[turnover_column], errors="coerce").mean()

    transaction_cost_drag = None
    if cost_column:
        transaction_cost_drag = pd.to_numeric(group[cost_column], errors="coerce").mean()

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "monthly_turnover": monthly_turnover,
        "hit_rate": (returns > 0).mean(),
        "best_month": returns.max(),
        "worst_month": returns.min(),
        "transaction_cost_drag": transaction_cost_drag,
    }


def summarize_backtest(root: Path) -> Tuple[str, Optional[pd.DataFrame]]:
    summary_path = root / "results/backtest_summary.csv"
    monthly_path = root / "results/backtest_results.csv"

    summary_df = read_csv(summary_path)
    if summary_df is not None:
        metric_columns = [column for column in PERFORMANCE_COLUMNS if column in summary_df.columns]
        label_columns = [
            column
            for column in [
                "strategy",
                "benchmark",
                "model_name",
                "feature_set",
                "cost_scenario",
                "portfolio_rule",
            ]
            if column in summary_df.columns
        ]

        if metric_columns:
            display_columns = label_columns + metric_columns
            return markdown_table(
                display_columns, summary_df[display_columns].values.tolist()
            ), summary_df

    monthly_df = read_csv(monthly_path)
    if monthly_df is None:
        return (
            "No backtest file found at `results/backtest_results.csv` or `results/backtest_summary.csv`.",
            None,
        )

    return_column = first_existing_column(
        monthly_df,
        [
            "net_return",
            "portfolio_return",
            "strategy_return",
            "monthly_return",
            "return",
            "gross_return",
        ],
    )

    if return_column is None:
        return (
            "A backtest file was found, but no recognizable return column was detected.",
            monthly_df,
        )

    group_columns = [
        column
        for column in [
            "strategy",
            "benchmark",
            "model_name",
            "feature_set",
            "cost_scenario",
            "portfolio_rule",
        ]
        if column in monthly_df.columns
    ]

    rows = []

    if group_columns:
        for keys, group in monthly_df.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            metrics = compute_performance(group, return_column)
            rows.append(list(keys) + [metrics[column] for column in PERFORMANCE_COLUMNS])
        perf_df = pd.DataFrame(rows, columns=group_columns + PERFORMANCE_COLUMNS)
    else:
        metrics = compute_performance(monthly_df, return_column)
        perf_df = pd.DataFrame(
            [[metrics[column] for column in PERFORMANCE_COLUMNS]], columns=PERFORMANCE_COLUMNS
        )

    return markdown_table(list(perf_df.columns), perf_df.values.tolist()), perf_df


def infer_main_result(performance_df: Optional[pd.DataFrame]) -> str:
    if performance_df is None or performance_df.empty:
        return (
            "The current generated files do not yet support a definitive comparison between "
            "the full AI/narrative model and the non-text models after transaction costs."
        )

    ranking_column = None
    for candidate in ["sharpe_ratio", "annualized_return"]:
        if candidate in performance_df.columns:
            ranking_column = candidate
            break

    if ranking_column is None:
        return "The backtest output does not contain Sharpe ratio or annualized return."

    label_columns = [
        column
        for column in ["strategy", "benchmark", "model_name", "feature_set", "portfolio_rule"]
        if column in performance_df.columns
    ]

    if not label_columns:
        return "The backtest output has metrics, but it lacks strategy labels needed for model comparison."

    temp = performance_df.copy()
    temp[ranking_column] = pd.to_numeric(temp[ranking_column], errors="coerce")
    temp = temp.dropna(subset=[ranking_column])

    if temp.empty:
        return "The backtest output does not contain usable numeric performance values."

    labels = temp[label_columns].astype(str).agg(" ".join, axis=1).str.lower()
    full_model_mask = labels.str.contains(
        "text|fed|narrative|full|model c|price macro fed", regex=True
    )

    full_models = temp[full_model_mask]
    non_text_models = temp[~full_model_mask]

    if full_models.empty:
        return (
            "No clearly labeled full AI/narrative model was detected. "
            "Use labels such as `price_macro_fed_text`, `full_ai_narrative`, or `Model C`."
        )

    if non_text_models.empty:
        return (
            "A full AI/narrative model was detected, but no non-text comparison model was detected."
        )

    best_full = full_models.sort_values(ranking_column, ascending=False).iloc[0]
    best_non_text = non_text_models.sort_values(ranking_column, ascending=False).iloc[0]

    full_label = " | ".join(str(best_full[column]) for column in label_columns)
    non_text_label = " | ".join(str(best_non_text[column]) for column in label_columns)

    full_value = float(best_full[ranking_column])
    non_text_value = float(best_non_text[ranking_column])

    if full_value > non_text_value:
        comparison = "outperformed"
    elif full_value < non_text_value:
        comparison = "did not outperform"
    else:
        comparison = "matched"

    return (
        f"Using `{ranking_column}`, the best detected full AI/narrative model "
        f"({full_label}: {full_value:.4f}) {comparison} the best detected non-text model "
        f"({non_text_label}: {non_text_value:.4f}) after the available transaction-cost adjustments."
    )


def build_final_report(root: Path) -> str:
    backtest_section, performance_df = summarize_backtest(root)
    main_result = infer_main_result(performance_df)

    sections = [
        "# Final Research Report",
        "",
        f"**Project:** {PROJECT_NAME}",
        f"**Generated:** {timestamp()}",
        "**Script:** `src/reporting/make_research_report.py`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        (
            "This project evaluates whether Federal Reserve narrative and text-derived features improve "
            "ETF return prediction and portfolio performance beyond price-only and price-plus-macro baselines."
        ),
        "",
        f"**Main after-cost result:** {main_result}",
        "",
        "---",
        "",
        "## 2. Research Question",
        "",
        (
            "Does a full AI/narrative alpha model that combines price features, macroeconomic indicators, "
            "and Federal Reserve text features outperform non-text alternatives in liquid ETF markets after "
            "transaction costs?"
        ),
        "",
        "---",
        "",
        "## 3. Data Sources",
        "",
        file_status_section(root),
        "",
        "Master modeling dataset summary:",
        "",
        summarize_master_dataset(root),
        "",
        "---",
        "",
        "## 4. Hypothesis Registry",
        "",
        markdown_table(
            ["Hypothesis ID", "Hypothesis", "Test Design", "Decision Rule"],
            [
                [
                    "H1",
                    "Fed text features add incremental predictive value beyond price and macro features.",
                    "Compare walk-forward diagnostics across feature groups.",
                    "Support H1 if the text model improves out-of-sample ranking, hit rate, or prediction/actual correlation.",
                ],
                [
                    "H2",
                    "Text-enhanced predictions improve investable portfolio performance.",
                    "Run after-cost backtests from monthly prediction-based ETF weights.",
                    "Support H2 if the text model has stronger after-cost risk-adjusted performance.",
                ],
                [
                    "H3",
                    "The text advantage survives realistic frictions.",
                    "Apply low, medium, and high transaction-cost scenarios.",
                    "Support H3 only if performance survives costs and turnover effects.",
                ],
                [
                    "H4",
                    "Feature selection improves parsimony without destroying performance.",
                    "Compare selected-feature models with full feature-set models.",
                    "Support H4 if selected-feature models retain similar performance with fewer predictors.",
                ],
            ],
        ),
        "",
        "---",
        "",
        "## 5. Feature Construction",
        "",
        (
            "The project constructs monthly ETF features from price, volume, momentum, realized volatility, "
            "drawdown, liquidity, macroeconomic, and text-derived inputs. Forward one-month ETF returns "
            "are used as prediction targets."
        ),
        "",
        summarize_selected_features(root),
        "",
        "---",
        "",
        "## 6. GenAI/Text Feature Design",
        "",
        (
            "The Fed text feature layer is designed to extract structured market signals from Federal Reserve "
            "communications. Candidate feature families include policy tone, inflation concern, growth concern, "
            "labor-market concern, financial-stability concern, hawkish/dovish orientation, topic features, "
            "and embedding-derived narrative features."
        ),
        "",
        "---",
        "",
        "## 7. No-Lookahead Controls",
        "",
        (
            "Features at month t must use only information available on or before month t. Targets are forward "
            "one-month returns. Walk-forward validation is chronological, random train-test splitting is avoided, "
            "and portfolio weights are formed only from predictions available before the return period."
        ),
        "",
        "---",
        "",
        "## 8. Model Design",
        "",
        (
            "The model set includes Ridge regression, Elastic Net, Random Forest, Gradient Boosting, and optional "
            "XGBoost or LightGBM models when those packages are installed. Model groups include price-only, "
            "price-plus-macro, price-plus-macro-plus-Fed-text, and selected-feature specifications."
        ),
        "",
        "Baseline metrics:",
        "",
        summarize_json_file(root, "results/baseline_model_metrics.json"),
        "",
        "Machine-learning model metrics:",
        "",
        summarize_json_file(root, "results/model_metrics.json"),
        "",
        "---",
        "",
        "## 9. Walk-Forward Validation",
        "",
        (
            "The intended validation design uses a 60-month training window, a one-month test window, "
            "a one-month step size, and a five-trading-day embargo."
        ),
        "",
        summarize_walk_forward(root),
        "",
        "---",
        "",
        "## 10. Backtest Design",
        "",
        (
            "The backtest converts monthly ETF return predictions into portfolio weights and evaluates realized "
            "monthly portfolio returns against SPY, 60/40, equal-weight ETF universe, momentum-only, macro-only, "
            "price-plus-macro, and price-plus-macro-plus-text benchmarks."
        ),
        "",
        backtest_section,
        "",
        "---",
        "",
        "## 11. Transaction Cost Assumptions",
        "",
        markdown_table(
            ["Cost scenario", "One-way trading cost"],
            [
                ["Low cost", "5 basis points"],
                ["Medium cost", "10 basis points"],
                ["High cost", "25 basis points"],
            ],
        ),
        "",
        (
            "The key performance measure is after-cost return. A model that performs well before costs but loses "
            "its advantage after costs should not be interpreted as an investable improvement."
        ),
        "",
        "---",
        "",
        "## 12. Risk Constraints",
        "",
        (
            "The optimizer is designed for long-only ETF weights, maximum single ETF weight of 35%, minimum selected "
            "ETF weight of 5%, maximum monthly turnover of 50%, target annualized volatility of 10%, and optional "
            "bond or commodity allocation caps."
        ),
        "",
        "Portfolio-weight summary:",
        "",
        summarize_portfolio_weights(root),
        "",
        "Optimization diagnostic summary:",
        "",
        summarize_optimization(root),
        "",
        "---",
        "",
        "## 13. Results",
        "",
        f"**{main_result}**",
        "",
        (
            "Interpretation should focus on after-cost risk-adjusted performance, especially Sharpe ratio, "
            "Sortino ratio, maximum drawdown, Calmar ratio, turnover, and transaction-cost drag."
        ),
        "",
        (
            "A valid positive result requires the full AI/narrative model to outperform not only SPY or equal "
            "weight, but also the stronger non-text models after transaction costs."
        ),
        "",
        "---",
        "",
        "## 14. Failure Cases",
        "",
        "Potential failure cases include:",
        "",
        "1. Text features improve in-sample fit but do not improve walk-forward performance.",
        "2. The full model selects high-turnover portfolios that lose their advantage after transaction costs.",
        "3. Fed text features are correlated with macro variables and add redundancy rather than new information.",
        "4. The strategy performs well only in a small number of crisis months.",
        "5. The model improves average returns but increases drawdown or volatility.",
        "6. Feature importance is unstable across time.",
        "",
        "---",
        "",
        "## 15. Limitations",
        "",
        (
            "This is a research prototype. Results can be sensitive to ETF universe selection, macro data revisions, "
            "text release timing, prompt design, transaction-cost assumptions, and market-regime dependence. "
            "The project does not establish causal effects of Federal Reserve language."
        ),
        "",
        "---",
        "",
        "## 16. Future Work",
        "",
        "Future extensions should include:",
        "",
        "1. Larger ETF universes and sector-specific subuniverses.",
        "2. More precise Federal Reserve text timestamp alignment.",
        "3. Robustness checks by market regime.",
        "4. Bootstrap or block-bootstrap confidence intervals for performance metrics.",
        "5. Deflated Sharpe ratio or multiple-testing corrections.",
        "6. More detailed transaction-cost and liquidity modeling.",
        "7. Model governance documentation and monitoring thresholds.",
        "8. Out-of-sample paper-trading simulation.",
        "",
        "---",
        "",
        "## Reproducibility Note",
        "",
        "Regenerate this report with:",
        "",
        "```bash",
        "python src/reporting/make_research_report.py",
        "```",
        "",
    ]

    return "\n".join(sections)


def build_model_card(root: Path) -> str:
    sections = [
        "# Model Card",
        "",
        f"**Project:** {PROJECT_NAME}",
        f"**Generated:** {timestamp()}",
        "**Script:** `src/reporting/make_research_report.py`",
        "",
        "---",
        "",
        "## Model Purpose",
        "",
        (
            "The model predicts next-month relative ETF returns using price features, macroeconomic features, "
            "and Federal Reserve text-derived features. The research purpose is to test whether a GenAI-assisted "
            "narrative signal improves liquid ETF allocation decisions after transaction costs and risk constraints."
        ),
        "",
        "---",
        "",
        "## Data Sources",
        "",
        file_status_section(root),
        "",
        "---",
        "",
        "## Prediction Target",
        "",
        "The main prediction target is `next_1m_return`.",
        "",
        "Additional target fields are `next_1m_rank` and `top_tercile_next_month`.",
        "",
        "---",
        "",
        "## Feature Groups",
        "",
        "1. Price and momentum features.",
        "2. Realized volatility and drawdown features.",
        "3. Volume and liquidity features.",
        "4. Macroeconomic features.",
        "5. Federal Reserve text and narrative features.",
        "6. Information-theoretic selected features.",
        "",
        summarize_selected_features(root),
        "",
        "---",
        "",
        "## Validation Method",
        "",
        (
            "The model uses walk-forward validation rather than random train-test splitting. The intended design "
            "uses a 60-month training window, a one-month test window, a one-month step size, and a five-trading-day embargo."
        ),
        "",
        summarize_walk_forward(root),
        "",
        "---",
        "",
        "## Known Limitations",
        "",
        (
            "Known limitations include data revision risk, timestamp alignment risk, model instability, prompt "
            "sensitivity, market-regime dependence, and transaction-cost sensitivity."
        ),
        "",
        "---",
        "",
        "## Model Risk Controls",
        "",
        "1. Chronological walk-forward validation.",
        "2. No-lookahead feature construction.",
        "3. Embargo around validation periods.",
        "4. Benchmark comparisons against non-text strategies.",
        "5. Transaction-cost scenarios.",
        "6. Turnover monitoring.",
        "7. Risk-constrained portfolio optimization.",
        "8. Feature-selection documentation.",
        "9. Reproducible report generation.",
        "10. Explicit model-card documentation.",
        "",
        "---",
        "",
        "## Ethical and Practical Limitations",
        "",
        (
            "This model is a research prototype. It is not designed to replace professional portfolio management, "
            "risk oversight, investment due diligence, or regulatory review."
        ),
        "",
        "---",
        "",
        "## Not Investment Advice Disclaimer",
        "",
        (
            "This project is for educational and research purposes only. It does not provide investment advice, "
            "financial advice, trading recommendations, or an offer to buy or sell securities."
        ),
        "",
    ]

    return "\n".join(sections)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    root = get_project_root()

    final_report_path = root / "reports" / "final_research_report.md"
    model_card_path = root / "reports" / "model_card.md"

    write_text(final_report_path, build_final_report(root))
    write_text(model_card_path, build_model_card(root))

    print("Generated reports:")
    print("- reports/final_research_report.md")
    print("- reports/model_card.md")


if __name__ == "__main__":
    main()
