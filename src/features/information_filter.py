"""
Information-theoretic feature selection for FedNarrativeAlpha.

Goal
----
Select features that are predictive of the target while penalizing redundancy.

Methods
-------
1. Mutual information with target
2. Correlation-based redundancy penalty
3. Greedy information-theoretic feature selection
4. Stability selection across rolling monthly windows

Output
------
results/selected_features.csv

Output columns
--------------
feature_name
feature_group
mutual_information
redundancy_penalty
final_score
selected_flag
selection_frequency
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "modeling_dataset.parquet",
    PROJECT_ROOT / "data" / "interim" / "modeling_dataset.parquet",
    PROJECT_ROOT / "data" / "interim" / "master_dataset.parquet",
    PROJECT_ROOT / "data" / "interim" / "price_features_monthly.parquet",
]

OUTPUT_PATH = PROJECT_ROOT / "results" / "selected_features.csv"

TARGET_CANDIDATES = [
    "next_1m_return",
    "next_month_return",
    "target_return",
    "top_tercile_next_month",
    "next_1m_rank",
    "target",
]

IDENTIFIER_COLUMNS = {
    "date",
    "month",
    "month_end",
    "month_end_date",
    "feature_date",
    "ticker",
    "symbol",
    "asset",
    "document_date",
    "document_type",
    "title",
    "url",
    "raw_text",
    "clean_text",
}

LEAKAGE_TERMS = [
    "next_",
    "future_",
    "forward_",
    "target",
    "top_tercile",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run information-theoretic feature selection.")

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Optional input parquet file. If omitted, the script searches standard project paths."
        ),
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column. If omitted, the script searches common target names.",
    )

    parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=0.25,
        help="Redundancy penalty weight.",
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=30,
        help="Maximum number of selected features.",
    )

    parser.add_argument(
        "--window-months",
        type=int,
        default=60,
        help="Rolling-window length for stability selection.",
    )

    parser.add_argument(
        "--step-months",
        type=int,
        default=6,
        help="Step size between rolling windows.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for mutual-information estimation.",
    )

    return parser.parse_args()


def find_existing_input(user_input: str | None) -> Path:
    if user_input is not None:
        path = Path(user_input)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        return path

    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path

    expected_paths = "\n".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)

    raise FileNotFoundError(
        f"No input dataset found. Expected one of these files:\n{expected_paths}"
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)

    if df.empty:
        raise ValueError(f"Input dataset is empty: {input_path}")

    return df


def find_target_column(df: pd.DataFrame, user_target: str | None) -> str:
    if user_target is not None:
        if user_target not in df.columns:
            raise ValueError(f"Target column not found: {user_target}")

        return user_target

    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    available_columns = ", ".join(df.columns)

    raise ValueError(
        "No target column found. Use --target with one of your target columns.\n"
        f"Available columns:\n{available_columns}"
    )


def find_date_column(df: pd.DataFrame) -> str | None:
    for candidate in [
        "date",
        "month",
        "month_end",
        "month_end_date",
        "feature_date",
    ]:
        if candidate in df.columns:
            return candidate

    return None


def is_classification_target(y: pd.Series) -> bool:
    clean_y = y.dropna()

    if clean_y.empty:
        raise ValueError("Target column contains no valid observations.")

    unique_values = clean_y.nunique()

    if unique_values <= 10:
        return True

    return False


def infer_feature_group(feature_name: str) -> str:
    name = feature_name.lower()

    if "fed_embedding" in name or "fed_similarity" in name or name.startswith("fed_"):
        return "fed_text_embeddings"

    if "return" in name or "momentum" in name or "drawdown" in name:
        return "price_momentum"

    if "vol" in name:
        return "realized_volatility"

    if "volume" in name or "dollar_volume" in name:
        return "liquidity"

    if "rate" in name or "yield" in name or "slope" in name:
        return "rates_yield_curve"

    if "inflation" in name or "cpi" in name:
        return "inflation"

    if "unemployment" in name or "payems" in name or "labor" in name:
        return "labor_market"

    if "credit" in name or "spread" in name:
        return "credit_spreads"

    if "vix" in name:
        return "market_volatility"

    if "financial_conditions" in name or "nfci" in name:
        return "financial_conditions"

    return "other"


def get_candidate_features(df: pd.DataFrame, target_col: str) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS)
    excluded.add(target_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    features: list[str] = []

    for col in numeric_cols:
        if col in excluded:
            continue

        col_lower = col.lower()

        if any(term in col_lower for term in LEAKAGE_TERMS):
            continue

        if df[col].nunique(dropna=True) <= 1:
            continue

        features.append(col)

    if not features:
        raise ValueError("No usable numeric feature columns found.")

    return features


def prepare_xy(
    df: pd.DataFrame,
    features: Iterable[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_list = list(features)
    cols = feature_list + [target_col]

    clean_df = df[cols].replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=[target_col]).copy()

    if clean_df.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    X_raw = clean_df[feature_list]
    y = clean_df[target_col]

    imputer = SimpleImputer(strategy="median")
    X_array = imputer.fit_transform(X_raw)

    X = pd.DataFrame(
        X_array,
        columns=feature_list,
        index=clean_df.index,
    )

    return X, y


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> pd.Series:
    if is_classification_target(y):
        y_encoded = pd.factorize(y)[0]

        mi_values = mutual_info_classif(
            X,
            y_encoded,
            discrete_features=False,
            random_state=random_state,
        )
    else:
        mi_values = mutual_info_regression(
            X,
            y.astype(float),
            discrete_features=False,
            random_state=random_state,
        )

    mi = pd.Series(
        mi_values,
        index=X.columns,
        name="mutual_information",
    )

    mi = mi.fillna(0.0)

    return mi


def compute_abs_correlation(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute absolute feature correlation.

    Important
    ---------
    Do not use np.fill_diagonal(corr.values, 0.0), because in some pandas/numpy
    versions corr.values can be read-only. This implementation is safe.
    """
    corr = X.corr().abs().fillna(0.0).copy()

    for feature in corr.columns:
        corr.loc[feature, feature] = 0.0

    return corr


def greedy_select_features(
    mi: pd.Series,
    corr: pd.DataFrame,
    lambda_penalty: float,
    max_features: int,
) -> list[str]:
    selected: list[str] = []
    remaining = set(mi.index)

    max_features = min(max_features, len(remaining))

    for _ in range(max_features):
        best_feature = None
        best_score = -np.inf

        for feature in remaining:
            if selected:
                redundancy = float(corr.loc[feature, selected].mean())
            else:
                redundancy = 0.0

            score = float(mi.loc[feature]) - lambda_penalty * redundancy

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def calculate_final_scores(
    mi: pd.Series,
    corr: pd.DataFrame,
    selected: list[str],
    lambda_penalty: float,
) -> pd.DataFrame:
    rows = []

    for feature in mi.index:
        comparison_features = [f for f in selected if f != feature]

        if comparison_features:
            average_redundancy = float(corr.loc[feature, comparison_features].mean())
        else:
            average_redundancy = 0.0

        redundancy_penalty = lambda_penalty * average_redundancy
        final_score = float(mi.loc[feature]) - redundancy_penalty

        rows.append(
            {
                "feature_name": feature,
                "feature_group": infer_feature_group(feature),
                "mutual_information": float(mi.loc[feature]),
                "redundancy_penalty": float(redundancy_penalty),
                "final_score": float(final_score),
                "selected_flag": bool(feature in selected),
            }
        )

    return pd.DataFrame(rows)


def get_rolling_windows(
    df: pd.DataFrame,
    date_col: str | None,
    window_months: int,
    step_months: int,
) -> list[pd.Index]:
    if date_col is None:
        return [df.index]

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).copy()

    if temp.empty:
        return [df.index]

    temp["_month"] = temp[date_col].dt.to_period("M").dt.to_timestamp("M")

    months = sorted(temp["_month"].dropna().unique())

    if len(months) < max(12, window_months):
        return [temp.index]

    windows: list[pd.Index] = []
    start = 0

    while start + window_months <= len(months):
        window_months_selected = months[start : start + window_months]
        window_month_set = set(window_months_selected)

        idx = temp.index[temp["_month"].isin(window_month_set)]

        if len(idx) > 0:
            windows.append(idx)

        start += step_months

    if not windows:
        windows = [temp.index]

    return windows


def compute_selection_frequency(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str | None,
    lambda_penalty: float,
    max_features: int,
    window_months: int,
    step_months: int,
    random_state: int,
) -> pd.Series:
    windows = get_rolling_windows(
        df=df,
        date_col=date_col,
        window_months=window_months,
        step_months=step_months,
    )

    counts = pd.Series(0.0, index=features)
    valid_windows = 0

    for window_number, window_idx in enumerate(windows, start=1):
        window_df = df.loc[window_idx].copy()

        try:
            X_window, y_window = prepare_xy(
                df=window_df,
                features=features,
                target_col=target_col,
            )

            if len(X_window) < 30:
                print(f"Skipping rolling window {window_number}: only {len(X_window)} valid rows.")
                continue

            mi_window = compute_mutual_information(
                X=X_window,
                y=y_window,
                random_state=random_state,
            )

            corr_window = compute_abs_correlation(X_window)

            selected_window = greedy_select_features(
                mi=mi_window,
                corr=corr_window,
                lambda_penalty=lambda_penalty,
                max_features=max_features,
            )

            counts.loc[selected_window] += 1.0
            valid_windows += 1

        except Exception as exc:
            print(f"Skipping rolling window {window_number} because of error: {exc}")

    if valid_windows == 0:
        frequency = pd.Series(0.0, index=features, name="selection_frequency")
    else:
        frequency = counts / valid_windows
        frequency.name = "selection_frequency"

    return frequency


def main() -> None:
    args = parse_args()

    input_path = find_existing_input(args.input)
    df = load_dataset(input_path)

    target_col = find_target_column(df, args.target)
    date_col = find_date_column(df)

    features = get_candidate_features(df, target_col)

    X, y = prepare_xy(
        df=df,
        features=features,
        target_col=target_col,
    )

    mi = compute_mutual_information(
        X=X,
        y=y,
        random_state=args.random_state,
    )

    corr = compute_abs_correlation(X)

    selected = greedy_select_features(
        mi=mi,
        corr=corr,
        lambda_penalty=args.lambda_penalty,
        max_features=args.max_features,
    )

    result = calculate_final_scores(
        mi=mi,
        corr=corr,
        selected=selected,
        lambda_penalty=args.lambda_penalty,
    )

    selection_frequency = compute_selection_frequency(
        df=df,
        features=features,
        target_col=target_col,
        date_col=date_col,
        lambda_penalty=args.lambda_penalty,
        max_features=args.max_features,
        window_months=args.window_months,
        step_months=args.step_months,
        random_state=args.random_state,
    )

    selection_frequency_df = selection_frequency.reset_index().rename(
        columns={"index": "feature_name"}
    )

    result = result.merge(
        selection_frequency_df,
        on="feature_name",
        how="left",
    )

    result["selection_frequency"] = result["selection_frequency"].fillna(0.0)

    result = result[
        [
            "feature_name",
            "feature_group",
            "mutual_information",
            "redundancy_penalty",
            "final_score",
            "selected_flag",
            "selection_frequency",
        ]
    ]

    result = result.sort_values(
        by=["selected_flag", "selection_frequency", "final_score"],
        ascending=[False, False, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Information-theoretic feature selection complete.")
    print(f"Input file: {input_path}")
    print(f"Target column: {target_col}")
    print(f"Date column: {date_col}")
    print(f"Number of candidate features: {len(features)}")
    print(f"Number of selected features: {int(result['selected_flag'].sum())}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
