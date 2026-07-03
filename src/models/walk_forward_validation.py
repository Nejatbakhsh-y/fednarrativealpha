"""
Step 12: Walk-Forward Validation Framework

Purpose
-------
Run chronological walk-forward validation for financial time-series modeling.

Validation design:
- Train window: 60 months
- Test window: next 1 month
- Step size: 1 month
- Embargo: 5 trading days before each test month
- No random train-test split

Input
-----
data/processed/master_modeling_dataset.parquet

Optional input
--------------
results/selected_features.csv

Output
------
results/walk_forward_predictions.csv

Output columns:
- month_end_date
- ticker
- actual_next_return
- predicted_return
- predicted_rank
- model_name
- feature_set
- training_start
- training_end
- test_date
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "next_1m_return"
DATE_COLUMN = "month_end_date"
TICKER_COLUMN = "ticker"

TARGET_COLUMNS = {
    "next_1m_return",
    "next_1m_rank",
    "top_tercile_next_month",
}

ID_COLUMNS = {
    "month_end_date",
    "date",
    "ticker",
    "symbol",
    "security",
    "asset",
}

LEAKAGE_PATTERNS = (
    "next_",
    "future_",
    "forward_",
    "target",
    "actual_",
    "predicted_",
)


PRICE_FEATURE_CANDIDATES = [
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "momentum_12m_minus_1m",
    "realized_vol_21d",
    "realized_vol_63d",
    "realized_vol_126d",
    "max_drawdown_6m",
    "volume_change_3m",
    "dollar_volume",
    "price_above_200d_ma",
]


MACRO_FEATURE_CANDIDATES = [
    "rate_level",
    "rate_change_1m",
    "rate_change_3m",
    "yield_curve_slope",
    "inflation_momentum",
    "unemployment_change",
    "credit_spread_level",
    "credit_spread_change",
    "vix_level",
    "vix_change",
    "financial_conditions_level",
    "financial_conditions_change",
    "FEDFUNDS",
    "DGS2",
    "DGS10",
    "T10Y2Y",
    "CPIAUCSL",
    "UNRATE",
    "PAYEMS",
    "VIXCLS",
    "BAMLH0A0HYM2",
    "NFCI",
]


FED_TEXT_FEATURE_CANDIDATES = [
    "fed_embedding_pc1",
    "fed_embedding_pc2",
    "fed_embedding_pc3",
    "fed_embedding_shift_1m",
    "fed_embedding_shift_3m",
    "fed_similarity_to_inflation_theme",
    "fed_similarity_to_recession_theme",
    "fed_similarity_to_financial_stability_theme",
    "fed_similarity_to_tightening_theme",
    "fed_similarity_to_easing_theme",
    "fed_similarity_to_labor_market_theme",
    "fed_similarity_to_higher_for_longer_theme",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation for ETF alpha models."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/master_modeling_dataset.parquet",
        help="Path to the master modeling dataset.",
    )

    parser.add_argument(
        "--selected-features",
        type=str,
        default="results/selected_features.csv",
        help="Path to selected_features.csv from the information filter step.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/walk_forward_predictions.csv",
        help="Path for the walk-forward prediction output CSV.",
    )

    parser.add_argument(
        "--train-months",
        type=int,
        default=60,
        help="Number of monthly observations used for each rolling training window.",
    )

    parser.add_argument(
        "--embargo-days",
        type=int,
        default=5,
        help="Number of trading days embargoed before the test month.",
    )

    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=100,
        help="Minimum number of training rows required for a model fit.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Comma-separated model list. Options: all, ridge, elastic_net, "
            "random_forest, gradient_boosting, xgboost, lightgbm."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible model training.",
    )

    return parser.parse_args()


def normalize_model_names(model_string: str) -> List[str]:
    names = [
        item.strip().lower().replace("-", "_").replace(" ", "_")
        for item in model_string.split(",")
        if item.strip()
    ]

    if not names:
        return ["all"]

    return names


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def is_safe_feature_column(column: str) -> bool:
    lower_col = column.lower()

    if lower_col in ID_COLUMNS:
        return False

    if lower_col in TARGET_COLUMNS:
        return False

    for pattern in LEAKAGE_PATTERNS:
        if pattern in lower_col:
            return False

    return True


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=["number", "bool"]).columns)


def available_exact_features(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    numeric = set(numeric_columns(df))

    return [
        col
        for col in candidates
        if col in df.columns and col in numeric and is_safe_feature_column(col)
    ]


def available_pattern_features(
    df: pd.DataFrame,
    prefixes: Sequence[str] | None = None,
    contains: Sequence[str] | None = None,
) -> List[str]:
    prefixes = prefixes or []
    contains = contains or []

    numeric = set(numeric_columns(df))
    selected = []

    for col in df.columns:
        lower_col = col.lower()

        if col not in numeric:
            continue

        if not is_safe_feature_column(col):
            continue

        prefix_match = any(lower_col.startswith(prefix.lower()) for prefix in prefixes)
        contains_match = any(term.lower() in lower_col for term in contains)

        if prefix_match or contains_match:
            selected.append(col)

    return selected


def load_master_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {path}\nRun Step 9 first: src/data/build_master_dataset.py"
        )

    df = pd.read_parquet(path)

    required_columns = [DATE_COLUMN, TICKER_COLUMN, TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"The master modeling dataset is missing required columns: {missing}")

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN, TICKER_COLUMN])
    df = df.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    return df


def load_selected_features(path: Path, df: pd.DataFrame) -> List[str]:
    if not path.exists():
        return []

    selected_df = pd.read_csv(path)

    if "feature_name" not in selected_df.columns:
        return []

    if "selected_flag" in selected_df.columns:
        flag = selected_df["selected_flag"]

        if flag.dtype == bool:
            selected_df = selected_df[flag]
        else:
            selected_df = selected_df[flag.astype(str).str.lower().isin(["true", "1", "yes", "y"])]

    candidate_features = selected_df["feature_name"].dropna().astype(str).tolist()
    numeric = set(numeric_columns(df))

    selected_features = [
        col
        for col in candidate_features
        if col in df.columns and col in numeric and is_safe_feature_column(col)
    ]

    return unique_preserve_order(selected_features)


def build_feature_sets(df: pd.DataFrame, selected_features_path: Path) -> Dict[str, List[str]]:
    price_exact = available_exact_features(df, PRICE_FEATURE_CANDIDATES)

    price_pattern = available_pattern_features(
        df,
        prefixes=["price_"],
        contains=[
            "return_1m",
            "return_3m",
            "return_6m",
            "return_12m",
            "momentum",
            "realized_vol",
            "drawdown",
            "volume_change",
            "dollar_volume",
            "200d",
        ],
    )

    macro_exact = available_exact_features(df, MACRO_FEATURE_CANDIDATES)

    macro_pattern = available_pattern_features(
        df,
        prefixes=["macro_", "fred_"],
        contains=[
            "rate_",
            "yield_curve",
            "inflation",
            "unemployment",
            "credit_spread",
            "vix",
            "financial_conditions",
            "fedfunds",
            "dgs2",
            "dgs10",
            "t10y2y",
            "cpiaucsl",
            "unrate",
            "payems",
            "bamlh0a0hym2",
            "nfci",
        ],
    )

    fed_text_exact = available_exact_features(df, FED_TEXT_FEATURE_CANDIDATES)

    fed_text_pattern = available_pattern_features(
        df,
        prefixes=["fed_"],
        contains=[
            "embedding",
            "similarity",
            "inflation_theme",
            "recession_theme",
            "financial_stability_theme",
            "tightening_theme",
            "easing_theme",
        ],
    )

    price_features = unique_preserve_order(price_exact + price_pattern)
    macro_features = unique_preserve_order(macro_exact + macro_pattern)
    fed_text_features = unique_preserve_order(fed_text_exact + fed_text_pattern)

    selected_features = load_selected_features(selected_features_path, df)

    feature_sets = {
        "Model A: price only": price_features,
        "Model B: price + macro": unique_preserve_order(price_features + macro_features),
        "Model C: price + macro + Fed text": unique_preserve_order(
            price_features + macro_features + fed_text_features
        ),
    }

    if selected_features:
        feature_sets["Model D: selected features only"] = selected_features

    usable_feature_sets = {
        name: features for name, features in feature_sets.items() if len(features) > 0
    }

    if not usable_feature_sets:
        raise ValueError("No usable feature columns were found. Check the master dataset columns.")

    return usable_feature_sets


def make_models(model_names: Sequence[str], random_state: int) -> Dict[str, Pipeline]:
    requested = set(model_names)
    use_all = "all" in requested

    models: Dict[str, Pipeline] = {}

    def should_include(name: str) -> bool:
        return use_all or name in requested

    if should_include("ridge"):
        models["Ridge regression"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    if should_include("elastic_net"):
        models["Elastic Net"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.001,
                        l1_ratio=0.25,
                        max_iter=20_000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if should_include("random_forest"):
        models["Random Forest"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if should_include("gradient_boosting"):
        models["Gradient Boosting"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=3,
                        subsample=0.85,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if should_include("xgboost"):
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=300,
                            learning_rate=0.03,
                            max_depth=3,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        except ImportError:
            print("WARNING: xgboost is not installed. Skipping XGBoost.")

    if should_include("lightgbm"):
        try:
            from lightgbm import LGBMRegressor

            models["LightGBM"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=300,
                            learning_rate=0.03,
                            max_depth=3,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            random_state=random_state,
                            n_jobs=-1,
                            verbose=-1,
                        ),
                    ),
                ]
            )
        except ImportError:
            print("WARNING: lightgbm is not installed. Skipping LightGBM.")

    if not models:
        raise ValueError(
            "No valid models selected. Use one or more of: "
            "ridge, elastic_net, random_forest, gradient_boosting, xgboost, lightgbm."
        )

    return models


def clean_xy(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = data[list(features)].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    y = pd.to_numeric(data[target], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    usable_features = [col for col in X.columns if X[col].notna().any()]
    X = X[usable_features]

    return X, y


def run_walk_forward_validation(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    models: Dict[str, Pipeline],
    train_months: int,
    embargo_days: int,
    min_train_rows: int,
) -> pd.DataFrame:
    all_dates = sorted(df[DATE_COLUMN].dropna().unique())

    if len(all_dates) <= train_months:
        raise ValueError(
            f"Not enough monthly dates for walk-forward validation. "
            f"Found {len(all_dates)} dates, but train_months={train_months}."
        )

    records = []

    for test_date in all_dates:
        test_date = pd.Timestamp(test_date)
        embargo_cutoff = test_date - BDay(embargo_days)

        eligible_train_dates = [
            pd.Timestamp(date)
            for date in all_dates
            if pd.Timestamp(date) < test_date and pd.Timestamp(date) <= embargo_cutoff
        ]

        if len(eligible_train_dates) < train_months:
            continue

        train_dates = eligible_train_dates[-train_months:]
        training_start = train_dates[0]
        training_end = train_dates[-1]

        train_df = df[df[DATE_COLUMN].isin(train_dates)].copy()
        test_df = df[df[DATE_COLUMN] == test_date].copy()

        test_df = test_df.dropna(subset=[TARGET_COLUMN])

        if train_df.empty or test_df.empty:
            continue

        for feature_set_name, feature_cols in feature_sets.items():
            feature_cols = [
                col for col in feature_cols if col in df.columns and is_safe_feature_column(col)
            ]

            if not feature_cols:
                continue

            X_train, y_train = clean_xy(train_df, feature_cols, TARGET_COLUMN)

            if X_train.empty or len(y_train) < min_train_rows:
                continue

            X_test = test_df[X_train.columns].copy()
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.apply(pd.to_numeric, errors="coerce")

            actual_values = pd.to_numeric(test_df[TARGET_COLUMN], errors="coerce")

            for model_name, model_pipeline in models.items():
                estimator = clone(model_pipeline)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        estimator.fit(X_train, y_train)
                        predictions = estimator.predict(X_test)

                except Exception as exc:
                    print(
                        f"WARNING: Skipping failed fit. "
                        f"model={model_name}, "
                        f"feature_set={feature_set_name}, "
                        f"test_date={test_date.date()}, "
                        f"error={exc}"
                    )
                    continue

                for row_index, predicted_return in zip(test_df.index, predictions):
                    records.append(
                        {
                            "month_end_date": test_date,
                            "ticker": test_df.loc[row_index, TICKER_COLUMN],
                            "actual_next_return": actual_values.loc[row_index],
                            "predicted_return": float(predicted_return),
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "training_start": training_start,
                            "training_end": training_end,
                            "test_date": test_date,
                        }
                    )

    if not records:
        raise ValueError(
            "Walk-forward validation produced no predictions. "
            "Check feature availability, target availability, and training window length."
        )

    predictions_df = pd.DataFrame(records)

    predictions_df["predicted_rank"] = (
        predictions_df.groupby(["test_date", "model_name", "feature_set"])["predicted_return"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    output_columns = [
        "month_end_date",
        "ticker",
        "actual_next_return",
        "predicted_return",
        "predicted_rank",
        "model_name",
        "feature_set",
        "training_start",
        "training_end",
        "test_date",
    ]

    predictions_df = predictions_df[output_columns]

    for col in ["month_end_date", "training_start", "training_end", "test_date"]:
        predictions_df[col] = pd.to_datetime(predictions_df[col]).dt.strftime("%Y-%m-%d")

    predictions_df = predictions_df.sort_values(
        ["test_date", "feature_set", "model_name", "predicted_rank", "ticker"]
    ).reset_index(drop=True)

    return predictions_df


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    selected_features_path = Path(args.selected_features)
    output_path = Path(args.output)

    print("=" * 80)
    print("Step 12: Walk-Forward Validation")
    print("=" * 80)

    print(f"Reading master dataset from: {input_path}")
    df = load_master_dataset(input_path)

    print(f"Rows loaded: {len(df):,}")
    print(f"Date range: {df[DATE_COLUMN].min().date()} to {df[DATE_COLUMN].max().date()}")
    print(f"Tickers: {df[TICKER_COLUMN].nunique():,}")

    feature_sets = build_feature_sets(df, selected_features_path)

    print("\nFeature sets:")
    for name, features in feature_sets.items():
        print(f"  {name}: {len(features)} features")

    model_names = normalize_model_names(args.models)
    models = make_models(model_names, args.random_state)

    print("\nModels:")
    for model_name in models:
        print(f"  {model_name}")

    print("\nRunning walk-forward validation...")
    predictions_df = run_walk_forward_validation(
        df=df,
        feature_sets=feature_sets,
        models=models,
        train_months=args.train_months,
        embargo_days=args.embargo_days,
        min_train_rows=args.min_train_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print("\nDone.")
    print(f"Predictions saved to: {output_path}")
    print(f"Prediction rows: {len(predictions_df):,}")
    print(
        "Prediction date range: "
        f"{predictions_df['test_date'].min()} to {predictions_df['test_date'].max()}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
