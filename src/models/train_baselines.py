"""
Train baseline forecasting models for FedNarrativeAlpha.

Input:
    data/processed/master_modeling_dataset.parquet

Output:
    results/baseline_metrics.json
    results/baseline_predictions.parquet

No-lookahead rule:
    For every prediction month t, machine-learning baselines are trained only on
    rows with month_end_date strictly before t.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "next_1m_return"
RANK_TARGET_COL = "next_1m_rank"
TOP_TERCILE_COL = "top_tercile_next_month"

ID_COLS = ["month_end_date", "ticker"]

PREDICTION_COLS = {
    "equal_weight": "pred_equal_weight",
    "previous_12m_momentum": "pred_previous_12m_momentum",
    "previous_1m_reversal": "pred_previous_1m_reversal",
    "low_volatility": "pred_low_volatility",
    "macro_only_ridge": "pred_macro_only_ridge",
    "price_only_random_forest": "pred_price_only_random_forest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train no-lookahead baseline forecasting models.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/master_modeling_dataset.parquet"),
        help="Path to master modeling dataset.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/baseline_metrics.json"),
        help="Path to baseline metrics JSON output.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("results/baseline_predictions.parquet"),
        help="Path to baseline predictions output.",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=60,
        help="Minimum number of historical months before evaluation starts.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=300,
        help="Number of trees for the random forest baseline.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Build a version-compatible OneHotEncoder.

    scikit-learn >= 1.2 uses sparse_output.
    Older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_master_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {path}. Run src/data/build_master_dataset.py first."
        )

    df = pd.read_parquet(path)

    required_cols = ID_COLS + [TARGET_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Master dataset is missing required columns: {missing}")

    df = df.copy()
    df["month_end_date"] = pd.to_datetime(df["month_end_date"])
    df["ticker"] = df["ticker"].astype(str)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    if TOP_TERCILE_COL in df.columns:
        df[TOP_TERCILE_COL] = pd.to_numeric(df[TOP_TERCILE_COL], errors="coerce")

    df = df.sort_values(["month_end_date", "ticker"]).reset_index(drop=True)

    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLS + [TARGET_COL, RANK_TARGET_COL, TOP_TERCILE_COL])
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


def select_feature_group(df: pd.DataFrame, group: str) -> list[str]:
    features = numeric_feature_columns(df)

    macro_keywords = [
        "macro",
        "rate",
        "fedfunds",
        "dgs",
        "yield",
        "curve",
        "slope",
        "inflation",
        "cpi",
        "unemployment",
        "unrate",
        "payems",
        "credit",
        "spread",
        "vix",
        "financial_conditions",
        "nfci",
        "baml",
    ]

    price_keywords = [
        "return_1m",
        "return_3m",
        "return_6m",
        "return_12m",
        "momentum",
        "realized_vol",
        "volatility",
        "drawdown",
        "volume",
        "dollar_volume",
        "price_above",
        "ma",
    ]

    if group == "macro":
        keywords = macro_keywords
    elif group == "price":
        keywords = price_keywords
    else:
        raise ValueError(f"Unknown feature group: {group}")

    selected = [col for col in features if any(keyword in col.lower() for keyword in keywords)]

    return selected


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def evaluation_months(df: pd.DataFrame, min_train_months: int) -> list[pd.Timestamp]:
    months = sorted(df["month_end_date"].dropna().unique())

    if len(months) <= min_train_months:
        raise ValueError(
            f"Not enough monthly observations. Found {len(months)} months, "
            f"but min_train_months={min_train_months}."
        )

    return list(months[min_train_months:])


def expanding_mean_predictions(
    df: pd.DataFrame,
    eval_months: list[pd.Timestamp],
) -> pd.Series:
    preds = pd.Series(index=df.index, dtype=float)

    for month in eval_months:
        train = df[(df["month_end_date"] < month) & df[TARGET_COL].notna()]
        test_idx = df.index[df["month_end_date"] == month]

        mean_return = train[TARGET_COL].mean()
        if not np.isfinite(mean_return):
            mean_return = 0.0

        preds.loc[test_idx] = mean_return

    return preds


def signal_predictions(
    df: pd.DataFrame,
    eval_months: list[pd.Timestamp],
    signal_col: str | None,
    multiplier: float = 1.0,
) -> pd.Series:
    """
    Create no-lookahead signal predictions.

    Missing signal values are filled using the historical median of that signal
    computed only from months before the prediction month.
    """
    preds = pd.Series(index=df.index, dtype=float)

    if signal_col is None:
        return expanding_mean_predictions(df, eval_months)

    for month in eval_months:
        train = df[df["month_end_date"] < month]
        test_idx = df.index[df["month_end_date"] == month]

        historical_median = pd.to_numeric(train[signal_col], errors="coerce").median()
        if not np.isfinite(historical_median):
            historical_median = 0.0

        values = pd.to_numeric(df.loc[test_idx, signal_col], errors="coerce")
        values = values.fillna(historical_median)

        preds.loc[test_idx] = multiplier * values

    return preds


def build_regression_pipeline(
    numeric_features: list[str],
    model: Any,
    scale_numeric: bool,
) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    ticker_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = [
        ("ticker", ticker_pipeline, ["ticker"]),
    ]

    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def walk_forward_ml_predictions(
    df: pd.DataFrame,
    eval_months: list[pd.Timestamp],
    feature_cols: list[str],
    model: Any,
    scale_numeric: bool,
) -> pd.Series:
    preds = pd.Series(index=df.index, dtype=float)

    fallback_preds = expanding_mean_predictions(df, eval_months)

    for month in eval_months:
        train = df[(df["month_end_date"] < month) & df[TARGET_COL].notna()].copy()

        test = df[(df["month_end_date"] == month) & df[TARGET_COL].notna()].copy()

        if train.empty or test.empty:
            continue

        if not feature_cols:
            preds.loc[test.index] = fallback_preds.loc[test.index]
            continue

        x_train = train[["ticker"] + feature_cols]
        y_train = train[TARGET_COL]
        x_test = test[["ticker"] + feature_cols]

        pipeline = build_regression_pipeline(
            numeric_features=feature_cols,
            model=model,
            scale_numeric=scale_numeric,
        )

        try:
            pipeline.fit(x_train, y_train)
            preds.loc[test.index] = pipeline.predict(x_test)
        except Exception as exc:
            print(
                f"WARNING: ML model failed for month {month}. "
                f"Using expanding mean fallback. Error: {exc}"
            )
            preds.loc[test.index] = fallback_preds.loc[test.index]

    return preds


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return np.nan

    return float(x.corr(y, method="pearson"))


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    x_rank = pd.to_numeric(x, errors="coerce").rank(method="average")
    y_rank = pd.to_numeric(y, errors="coerce").rank(method="average")
    return pearson_corr(x_rank, y_rank)


def monthly_average_correlation(
    df: pd.DataFrame,
    pred_col: str,
    method: str,
) -> float:
    values = []

    for _, group in df.groupby("month_end_date"):
        if method == "pearson":
            corr = pearson_corr(group[pred_col], group[TARGET_COL])
        elif method == "spearman":
            corr = spearman_corr(group[pred_col], group[TARGET_COL])
        else:
            raise ValueError(f"Unknown method: {method}")

        if np.isfinite(corr):
            values.append(corr)

    if not values:
        return np.nan

    return float(np.mean(values))


def predicted_top_tercile_mask(group: pd.DataFrame, pred_col: str) -> pd.Series:
    """
    Select predicted top tercile within a month.

    If all predictions are tied, all names are selected. This prevents arbitrary
    ticker ordering from dominating the equal-weight baseline.
    """
    preds = pd.to_numeric(group[pred_col], errors="coerce")

    if preds.nunique(dropna=True) <= 1:
        return pd.Series(True, index=group.index)

    k = max(1, math.ceil(len(group) / 3))
    ranks = preds.rank(method="first", ascending=False)

    return ranks <= k


def top_tercile_precision(df: pd.DataFrame, pred_col: str) -> float:
    monthly_precision = []

    for _, group in df.groupby("month_end_date"):
        selected = predicted_top_tercile_mask(group, pred_col)

        if selected.sum() == 0:
            continue

        if TOP_TERCILE_COL in group.columns and group[TOP_TERCILE_COL].notna().any():
            actual_top = pd.to_numeric(group[TOP_TERCILE_COL], errors="coerce") == 1
        else:
            k = max(1, math.ceil(len(group) / 3))
            actual_ranks = group[TARGET_COL].rank(method="first", ascending=False)
            actual_top = actual_ranks <= k

        monthly_precision.append(float(actual_top[selected].mean()))

    if not monthly_precision:
        return np.nan

    return float(np.mean(monthly_precision))


def selected_positive_return_hit_rate(df: pd.DataFrame, pred_col: str) -> float:
    """
    Hit rate is defined as the average monthly fraction of selected predicted
    top-tercile names with positive next-month returns.
    """
    monthly_hit_rates = []

    for _, group in df.groupby("month_end_date"):
        selected = predicted_top_tercile_mask(group, pred_col)

        if selected.sum() == 0:
            continue

        hits = group.loc[selected, TARGET_COL] > 0
        monthly_hit_rates.append(float(hits.mean()))

    if not monthly_hit_rates:
        return np.nan

    return float(np.mean(monthly_hit_rates))


def evaluate_model(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    eval_df = df[df[pred_col].notna() & df[TARGET_COL].notna()].copy()

    if eval_df.empty:
        return {
            "information_coefficient": None,
            "rank_ic": None,
            "directional_accuracy": None,
            "top_tercile_precision": None,
            "mean_absolute_error": None,
            "hit_rate": None,
            "n_observations": 0,
            "n_months": 0,
        }

    pred = pd.to_numeric(eval_df[pred_col], errors="coerce")
    actual = pd.to_numeric(eval_df[TARGET_COL], errors="coerce")

    directional_accuracy = float(((pred >= 0) == (actual >= 0)).mean())
    mae = float(np.mean(np.abs(pred - actual)))

    metrics = {
        "information_coefficient": monthly_average_correlation(eval_df, pred_col, method="pearson"),
        "rank_ic": monthly_average_correlation(eval_df, pred_col, method="spearman"),
        "directional_accuracy": directional_accuracy,
        "top_tercile_precision": top_tercile_precision(eval_df, pred_col),
        "mean_absolute_error": mae,
        "hit_rate": selected_positive_return_hit_rate(eval_df, pred_col),
        "n_observations": int(len(eval_df)),
        "n_months": int(eval_df["month_end_date"].nunique()),
    }

    clean_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not np.isfinite(value):
            clean_metrics[key] = None
        else:
            clean_metrics[key] = value

    return clean_metrics


def train_baselines(args: argparse.Namespace) -> dict[str, Any]:
    df = load_master_dataset(args.input)

    eval_months = evaluation_months(df, args.min_train_months)

    predictions = df[ID_COLS + [TARGET_COL]].copy()

    if RANK_TARGET_COL in df.columns:
        predictions[RANK_TARGET_COL] = df[RANK_TARGET_COL]

    if TOP_TERCILE_COL in df.columns:
        predictions[TOP_TERCILE_COL] = df[TOP_TERCILE_COL]

    momentum_col = first_existing_column(
        df,
        [
            "momentum_12m_minus_1m",
            "return_12m",
            "return_6m",
            "return_3m",
        ],
    )

    reversal_col = first_existing_column(
        df,
        [
            "return_1m",
        ],
    )

    low_vol_col = first_existing_column(
        df,
        [
            "realized_vol_63d",
            "realized_vol_126d",
            "realized_vol_21d",
        ],
    )

    macro_features = select_feature_group(df, "macro")
    price_features = select_feature_group(df, "price")

    print(f"Using {len(macro_features)} macro features.")
    print(f"Using {len(price_features)} price features.")
    print(f"Momentum feature: {momentum_col}")
    print(f"Reversal feature: {reversal_col}")
    print(f"Low-volatility feature: {low_vol_col}")
    print(f"Evaluation starts at: {pd.Timestamp(eval_months[0]).date()}")

    predictions[PREDICTION_COLS["equal_weight"]] = expanding_mean_predictions(df, eval_months)

    predictions[PREDICTION_COLS["previous_12m_momentum"]] = signal_predictions(
        df=df,
        eval_months=eval_months,
        signal_col=momentum_col,
        multiplier=1.0,
    )

    predictions[PREDICTION_COLS["previous_1m_reversal"]] = signal_predictions(
        df=df,
        eval_months=eval_months,
        signal_col=reversal_col,
        multiplier=-1.0,
    )

    predictions[PREDICTION_COLS["low_volatility"]] = signal_predictions(
        df=df,
        eval_months=eval_months,
        signal_col=low_vol_col,
        multiplier=-1.0,
    )

    predictions[PREDICTION_COLS["macro_only_ridge"]] = walk_forward_ml_predictions(
        df=df,
        eval_months=eval_months,
        feature_cols=macro_features,
        model=Ridge(alpha=args.ridge_alpha),
        scale_numeric=True,
    )

    predictions[PREDICTION_COLS["price_only_random_forest"]] = walk_forward_ml_predictions(
        df=df,
        eval_months=eval_months,
        feature_cols=price_features,
        model=RandomForestRegressor(
            n_estimators=args.rf_estimators,
            min_samples_leaf=3,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        scale_numeric=False,
    )

    metrics = {
        model_name: evaluate_model(predictions, pred_col)
        for model_name, pred_col in PREDICTION_COLS.items()
    }

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "target": TARGET_COL,
        "rank_target": RANK_TARGET_COL if RANK_TARGET_COL in df.columns else None,
        "top_tercile_target": TOP_TERCILE_COL if TOP_TERCILE_COL in df.columns else None,
        "min_train_months": args.min_train_months,
        "evaluation_start_month": str(pd.Timestamp(eval_months[0]).date()),
        "evaluation_end_month": str(pd.Timestamp(eval_months[-1]).date()),
        "feature_sets": {
            "momentum_signal": momentum_col,
            "reversal_signal": reversal_col,
            "low_volatility_signal": low_vol_col,
            "macro_features": macro_features,
            "price_features": price_features,
        },
        "metrics": metrics,
        "metric_definitions": {
            "information_coefficient": (
                "Average monthly Pearson correlation between predicted score and next-month return."
            ),
            "rank_ic": (
                "Average monthly Spearman rank correlation between predicted "
                "score and next-month return."
            ),
            "directional_accuracy": (
                "Row-level share of observations where predicted and realized "
                "returns have the same sign."
            ),
            "top_tercile_precision": (
                "Average monthly share of predicted top-tercile names that are "
                "actually in the realized top tercile."
            ),
            "mean_absolute_error": (
                "Mean absolute error between predicted score/return and next-month return."
            ),
            "hit_rate": (
                "Average monthly share of selected predicted top-tercile names "
                "with positive next-month return."
            ),
        },
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)

    with args.metrics_output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    predictions.to_parquet(args.predictions_output, index=False)

    return output


def main() -> None:
    args = parse_args()
    output = train_baselines(args)

    print("\nBaseline metrics saved to:")
    print(f"  {args.metrics_output}")

    print("\nBaseline predictions saved to:")
    print(f"  {args.predictions_output}")

    print("\nSummary:")
    for model_name, model_metrics in output["metrics"].items():
        print(
            f"  {model_name}: "
            f"rank_ic={model_metrics['rank_ic']}, "
            f"top_tercile_precision={model_metrics['top_tercile_precision']}, "
            f"mae={model_metrics['mean_absolute_error']}"
        )


if __name__ == "__main__":
    main()
