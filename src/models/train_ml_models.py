"""
Train walk-forward machine learning alpha models.

Step 11 output files:
    results/model_metrics.json
    results/walk_forward_predictions.csv
    results/feature_importance.csv

Inputs:
    data/processed/master_modeling_dataset.parquet
    results/selected_features.csv

Model groups:
    Model A: price features only
    Model B: price + macro features
    Model C: price + macro + Federal Reserve text features
    Model D: selected features only

Targets:
    Regression: next_1m_return
    Classification: top_tercile_next_month
    Ranking proxy: next_1m_rank

No-lookahead design:
    Walk-forward validation trains only on months strictly before the test month.
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_modeling_dataset.parquet"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "results" / "selected_features.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "walk_forward_predictions.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"


# ---------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------

DATE_COL = "month_end_date"
TICKER_COL = "ticker"

REGRESSION_TARGET = "next_1m_return"
CLASSIFICATION_TARGET = "top_tercile_next_month"
RANKING_TARGET = "next_1m_rank"

TARGET_COLS = {
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    RANKING_TARGET,
}

ID_COLS = {
    DATE_COL,
    TICKER_COL,
}

RANDOM_STATE = 42
TRAIN_WINDOW_MONTHS = 60
MIN_TRAIN_MONTHS = 36

# If your earlier code defines rank 1 as best, the script will try to detect it automatically.
# The final ranking score is oriented so that larger prediction_score means "better".
RANK_HIGHER_IS_BETTER_DEFAULT = True


PRICE_FEATURES = [
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

MACRO_FEATURES = [
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
]

FED_TEXT_FEATURES = [
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
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def to_json_safe(value: Any) -> Any:
    """Convert NumPy/pandas values into JSON-safe Python objects."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def make_imputer() -> SimpleImputer:
    """Use keep_empty_features when supported by the installed sklearn version."""
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def unique_preserve_order(cols: list[str]) -> list[str]:
    seen = set()
    output = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            output.append(col)
    return output


def load_master_dataset() -> pd.DataFrame:
    if not MASTER_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {MASTER_DATA_PATH}\n"
            "Run Step 9 first: src/data/build_master_dataset.py"
        )

    df = pd.read_parquet(MASTER_DATA_PATH)

    required_cols = [DATE_COL, TICKER_COL, REGRESSION_TARGET, CLASSIFICATION_TARGET, RANKING_TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "The master modeling dataset is missing required columns: " + ", ".join(missing)
        )

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    # Normalize classification target to 0/1.
    df[CLASSIFICATION_TARGET] = df[CLASSIFICATION_TARGET].astype(float)

    logger.info("Loaded master dataset: %s rows, %s columns", len(df), len(df.columns))
    logger.info("Date range: %s to %s", df[DATE_COL].min().date(), df[DATE_COL].max().date())

    return df


def infer_fed_text_features(df: pd.DataFrame) -> list[str]:
    explicit = existing_columns(df, FED_TEXT_FEATURES)
    inferred = [
        col
        for col in df.columns
        if col.startswith("fed_")
        and col not in ID_COLS
        and col not in TARGET_COLS
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return unique_preserve_order(explicit + inferred)


def load_selected_features(
    df: pd.DataFrame, fallback_features: list[str]
) -> tuple[list[str], list[str]]:
    warnings_list: list[str] = []

    if not SELECTED_FEATURES_PATH.exists():
        warnings_list.append(
            f"Missing {SELECTED_FEATURES_PATH}. Model D will fall back to Model C features."
        )
        return fallback_features, warnings_list

    selected_df = pd.read_csv(SELECTED_FEATURES_PATH)

    if "feature_name" not in selected_df.columns:
        warnings_list.append(
            f"{SELECTED_FEATURES_PATH} does not contain feature_name. "
            "Model D will fall back to Model C features."
        )
        return fallback_features, warnings_list

    if "selected_flag" in selected_df.columns:
        selected_flag = (
            selected_df["selected_flag"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y", "selected"])
        )
        selected = selected_df.loc[selected_flag, "feature_name"].astype(str).tolist()
    elif "final_score" in selected_df.columns:
        selected_df = selected_df.sort_values("final_score", ascending=False)
        selected = selected_df["feature_name"].head(30).astype(str).tolist()
        warnings_list.append(
            "selected_flag was not found. Model D uses the top 30 features by final_score."
        )
    else:
        selected = selected_df["feature_name"].astype(str).tolist()
        warnings_list.append(
            "selected_flag and final_score were not found. Model D uses all listed feature_name values."
        )

    selected = [
        col
        for col in unique_preserve_order(selected)
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not selected:
        warnings_list.append(
            "No valid selected features were found. Model D will fall back to Model C features."
        )
        return fallback_features, warnings_list

    return selected, warnings_list


def build_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    warnings_list: list[str] = []

    price_features = existing_columns(df, PRICE_FEATURES)
    macro_features = existing_columns(df, MACRO_FEATURES)
    fed_features = infer_fed_text_features(df)

    model_a = unique_preserve_order(price_features)
    model_b = unique_preserve_order(price_features + macro_features)
    model_c = unique_preserve_order(price_features + macro_features + fed_features)

    model_d, selected_warnings = load_selected_features(df, fallback_features=model_c)
    warnings_list.extend(selected_warnings)

    feature_sets = {
        "Model A - price only": model_a,
        "Model B - price plus macro": model_b,
        "Model C - price plus macro plus Fed text": model_c,
        "Model D - selected features only": model_d,
    }

    for group_name, features in feature_sets.items():
        if not features:
            warnings_list.append(f"{group_name} has zero usable features and will be skipped.")
        logger.info("%s: %s features", group_name, len(features))

    return feature_sets, warnings_list


def detect_rank_orientation(df: pd.DataFrame) -> bool:
    """
    Return True when larger next_1m_rank means better future return.
    Return False when smaller next_1m_rank means better future return.
    """
    temp = df[[RANKING_TARGET, CLASSIFICATION_TARGET]].dropna()
    if temp.empty:
        return RANK_HIGHER_IS_BETTER_DEFAULT

    top_rank_median = temp.loc[temp[CLASSIFICATION_TARGET] == 1, RANKING_TARGET].median()
    non_top_rank_median = temp.loc[temp[CLASSIFICATION_TARGET] == 0, RANKING_TARGET].median()

    if pd.isna(top_rank_median) or pd.isna(non_top_rank_median):
        return RANK_HIGHER_IS_BETTER_DEFAULT

    return bool(top_rank_median > non_top_rank_median)


def get_walk_forward_splits(df: pd.DataFrame) -> list[tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    months = sorted(df[DATE_COL].dropna().unique())

    if len(months) <= MIN_TRAIN_MONTHS:
        raise ValueError(
            f"Not enough months for walk-forward validation. "
            f"Found {len(months)} months but need more than {MIN_TRAIN_MONTHS}."
        )

    splits = []

    for test_pos in range(MIN_TRAIN_MONTHS, len(months)):
        test_month = months[test_pos]
        train_start_pos = max(0, test_pos - TRAIN_WINDOW_MONTHS)
        train_months = months[train_start_pos:test_pos]

        train_idx = df.index[df[DATE_COL].isin(train_months)].to_numpy()
        test_idx = df.index[df[DATE_COL] == test_month].to_numpy()

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((pd.Timestamp(test_month), train_idx, test_idx))

    logger.info("Created %s walk-forward splits", len(splits))
    return splits


# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------


def make_pipeline(model: Any, scale: bool) -> Pipeline:
    steps = [("imputer", make_imputer())]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_regression_models() -> dict[str, Pipeline]:
    models = {
        "Ridge regression": make_pipeline(Ridge(alpha=1.0), scale=True),
        "Elastic Net": make_pipeline(
            ElasticNet(
                alpha=0.001,
                l1_ratio=0.50,
                max_iter=10000,
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale=False,
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = make_pipeline(
            XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale=False,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = make_pipeline(
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            scale=False,
        )

    return models


def get_classification_models() -> dict[str, Pipeline]:
    models = {
        "Ridge classifier": make_pipeline(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=10000,
            ),
            scale=True,
        ),
        "Elastic Net logistic": make_pipeline(
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.50,
                C=1.0,
                class_weight="balanced",
                max_iter=10000,
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale=False,
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = make_pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale=False,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = make_pipeline(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            scale=False,
        )

    return models


# ---------------------------------------------------------------------
# Prediction and metrics
# ---------------------------------------------------------------------


def get_prediction_score(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    """
    Return continuous model score for ranking.
    For classifiers:
        - use probability of class 1 when available
        - otherwise use decision_function
        - otherwise use predicted class
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_test)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            score = model.decision_function(x_test)
            return np.asarray(score).ravel()
        except Exception:
            pass

    return np.asarray(model.predict(x_test)).ravel()


def extract_feature_importance(
    fitted_model: Pipeline,
    feature_names: list[str],
    task_name: str,
    target_col: str,
    model_group: str,
    model_name: str,
    test_month: pd.Timestamp,
) -> list[dict[str, Any]]:
    model_step = fitted_model.named_steps["model"]

    signed_values: np.ndarray | None = None
    abs_values: np.ndarray | None = None

    if hasattr(model_step, "coef_"):
        coef = np.asarray(model_step.coef_).ravel()
        if len(coef) == len(feature_names):
            signed_values = coef
            abs_values = np.abs(coef)

    elif hasattr(model_step, "feature_importances_"):
        importance = np.asarray(model_step.feature_importances_).ravel()
        if len(importance) == len(feature_names):
            signed_values = importance
            abs_values = importance

    if signed_values is None or abs_values is None:
        return []

    rows = []
    for feature, signed_value, abs_value in zip(feature_names, signed_values, abs_values):
        rows.append(
            {
                "test_month": test_month.date().isoformat(),
                "task": task_name,
                "target_column": target_col,
                "model_group": model_group,
                "model_name": model_name,
                "feature_name": feature,
                "signed_importance": safe_float(signed_value),
                "importance": safe_float(abs_value),
            }
        )

    return rows


def add_monthly_selection_flags(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()

    pred_df["predicted_top_tercile"] = 0
    pred_df["predicted_bottom_tercile"] = 0

    group_cols = ["task", "model_group", "model_name", DATE_COL]

    for _, idx in pred_df.groupby(group_cols).groups.items():
        scores = pred_df.loc[idx, "prediction_score"]

        if scores.notna().sum() < 3:
            continue

        top_cutoff = scores.quantile(2.0 / 3.0)
        bottom_cutoff = scores.quantile(1.0 / 3.0)

        pred_df.loc[idx, "predicted_top_tercile"] = (
            pred_df.loc[idx, "prediction_score"] >= top_cutoff
        ).astype(int)

        pred_df.loc[idx, "predicted_bottom_tercile"] = (
            pred_df.loc[idx, "prediction_score"] <= bottom_cutoff
        ).astype(int)

    return pred_df


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    metrics["n_predictions"] = int(len(y_true))
    metrics["rmse"] = safe_float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["mae"] = safe_float(mean_absolute_error(y_true, y_pred))

    try:
        metrics["r2"] = safe_float(r2_score(y_true, y_pred))
    except Exception:
        metrics["r2"] = None

    try:
        metrics["pearson_corr"] = safe_float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")
        )
    except Exception:
        metrics["pearson_corr"] = None

    try:
        metrics["spearman_corr"] = safe_float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        )
    except Exception:
        metrics["spearman_corr"] = None

    return metrics


def compute_classification_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    metrics["n_predictions"] = int(len(y_true))
    metrics["accuracy"] = safe_float(accuracy_score(y_true, y_pred))
    metrics["precision"] = safe_float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = safe_float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = safe_float(f1_score(y_true, y_pred, zero_division=0))

    if y_true.nunique() == 2:
        try:
            metrics["roc_auc"] = safe_float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = None

        try:
            metrics["average_precision"] = safe_float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def compute_alpha_metrics(group: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if REGRESSION_TARGET in group.columns:
        returns = group[REGRESSION_TARGET]

        predicted_top = group["predicted_top_tercile"] == 1
        predicted_bottom = group["predicted_bottom_tercile"] == 1

        metrics["avg_realized_return_all"] = safe_float(returns.mean())
        metrics["avg_realized_return_predicted_top_tercile"] = safe_float(
            returns.loc[predicted_top].mean()
        )
        metrics["avg_realized_return_predicted_bottom_tercile"] = safe_float(
            returns.loc[predicted_bottom].mean()
        )

        if predicted_top.any() and predicted_bottom.any():
            metrics["avg_long_short_return"] = safe_float(
                returns.loc[predicted_top].mean() - returns.loc[predicted_bottom].mean()
            )
        else:
            metrics["avg_long_short_return"] = None

    if CLASSIFICATION_TARGET in group.columns:
        predicted_top = group["predicted_top_tercile"] == 1
        if predicted_top.any():
            metrics["top_tercile_hit_rate"] = safe_float(
                group.loc[predicted_top, CLASSIFICATION_TARGET].mean()
            )
        else:
            metrics["top_tercile_hit_rate"] = None

    monthly_ics = []
    for _, month_df in group.groupby(DATE_COL):
        if len(month_df) >= 3:
            ic = month_df["prediction_score"].corr(
                month_df[REGRESSION_TARGET],
                method="spearman",
            )
            if pd.notna(ic):
                monthly_ics.append(ic)

    if monthly_ics:
        metrics["monthly_return_spearman_ic_mean"] = safe_float(np.mean(monthly_ics))
        metrics["monthly_return_spearman_ic_std"] = safe_float(np.std(monthly_ics, ddof=1))
        metrics["monthly_return_spearman_ic_positive_rate"] = safe_float(
            np.mean(np.array(monthly_ics) > 0)
        )
    else:
        metrics["monthly_return_spearman_ic_mean"] = None
        metrics["monthly_return_spearman_ic_std"] = None
        metrics["monthly_return_spearman_ic_positive_rate"] = None

    return metrics


def compute_all_metrics(pred_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    grouped = pred_df.groupby(["task", "target_column", "model_group", "model_name"])

    for (task, target_col, model_group, model_name), group in grouped:
        y_true = group["actual"]
        y_pred = group["prediction"]
        y_score = group["prediction_score"]

        row: dict[str, Any] = {
            "task": task,
            "target_column": target_col,
            "model_group": model_group,
            "model_name": model_name,
        }

        if task == "classification_top_tercile":
            row.update(compute_classification_metrics(y_true, y_pred, y_score))
        else:
            row.update(compute_regression_metrics(y_true, y_pred))

        row.update(compute_alpha_metrics(group))
        rows.append(row)

    rows = sorted(
        rows,
        key=lambda x: (
            x["task"],
            x["model_group"],
            x["model_name"],
        ),
    )

    return rows


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------


def train_walk_forward_models(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    rank_higher_is_better: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings_list: list[str] = []

    if not HAS_XGBOOST:
        warnings_list.append("xgboost is not installed. XGBoost models were skipped.")

    if not HAS_LIGHTGBM:
        warnings_list.append("lightgbm is not installed. LightGBM models were skipped.")

    splits = get_walk_forward_splits(df)

    regression_models = get_regression_models()
    classification_models = get_classification_models()

    tasks = {
        "regression_next_1m_return": {
            "target": REGRESSION_TARGET,
            "models": regression_models,
            "task_type": "regression",
        },
        "classification_top_tercile": {
            "target": CLASSIFICATION_TARGET,
            "models": classification_models,
            "task_type": "classification",
        },
        "ranking_proxy_next_1m_rank": {
            "target": RANKING_TARGET,
            "models": regression_models,
            "task_type": "ranking_regression",
        },
    }

    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for model_group, features in feature_sets.items():
        if not features:
            logger.warning("Skipping %s because it has no usable features.", model_group)
            continue

        for task_name, task_config in tasks.items():
            target_col = task_config["target"]
            model_dict = task_config["models"]
            task_type = task_config["task_type"]

            task_df = df.dropna(subset=[target_col]).copy()

            if task_type == "classification":
                task_df = task_df[task_df[target_col].isin([0, 1])].copy()

            for model_name, base_model in model_dict.items():
                logger.info("Training | %s | %s | %s", model_group, task_name, model_name)

                for test_month, train_idx, test_idx in splits:
                    train_df = task_df.loc[task_df.index.intersection(train_idx)].copy()
                    test_df = task_df.loc[task_df.index.intersection(test_idx)].copy()

                    if train_df.empty or test_df.empty:
                        continue

                    x_train = train_df[features]
                    y_train = train_df[target_col]
                    x_test = test_df[features]
                    y_test = test_df[target_col]

                    if task_type == "classification":
                        y_train = y_train.astype(int)
                        y_test = y_test.astype(int)

                        if y_train.nunique() < 2:
                            continue

                    fitted_model = clone(base_model)

                    try:
                        fitted_model.fit(x_train, y_train)
                    except Exception as exc:
                        warnings_list.append(
                            f"Skipped {model_group} | {task_name} | {model_name} | "
                            f"{test_month.date()} because fitting failed: {exc}"
                        )
                        continue

                    try:
                        y_pred = fitted_model.predict(x_test)
                        y_score = get_prediction_score(fitted_model, x_test)
                    except Exception as exc:
                        warnings_list.append(
                            f"Skipped prediction for {model_group} | {task_name} | {model_name} | "
                            f"{test_month.date()} because prediction failed: {exc}"
                        )
                        continue

                    y_pred = np.asarray(y_pred).ravel()
                    y_score = np.asarray(y_score).ravel()

                    if task_type == "classification":
                        y_pred = pd.Series(y_pred).astype(int).to_numpy()

                    if task_type == "ranking_regression" and not rank_higher_is_better:
                        oriented_score = -y_score
                    else:
                        oriented_score = y_score

                    for row_pos, (_, source_row) in enumerate(test_df.iterrows()):
                        prediction_rows.append(
                            {
                                DATE_COL: source_row[DATE_COL].date().isoformat(),
                                TICKER_COL: source_row[TICKER_COL],
                                "task": task_name,
                                "target_column": target_col,
                                "model_group": model_group,
                                "model_name": model_name,
                                "actual": safe_float(y_test.iloc[row_pos]),
                                "prediction": safe_float(y_pred[row_pos]),
                                "prediction_score": safe_float(oriented_score[row_pos]),
                                REGRESSION_TARGET: safe_float(source_row.get(REGRESSION_TARGET)),
                                RANKING_TARGET: safe_float(source_row.get(RANKING_TARGET)),
                                CLASSIFICATION_TARGET: safe_float(
                                    source_row.get(CLASSIFICATION_TARGET)
                                ),
                                "n_features": len(features),
                            }
                        )

                    importance_rows.extend(
                        extract_feature_importance(
                            fitted_model=fitted_model,
                            feature_names=features,
                            task_name=task_name,
                            target_col=target_col,
                            model_group=model_group,
                            model_name=model_name,
                            test_month=test_month,
                        )
                    )

    predictions_df = pd.DataFrame(prediction_rows)

    if predictions_df.empty:
        raise RuntimeError(
            "No predictions were generated. Check input data and feature availability."
        )

    predictions_df[DATE_COL] = pd.to_datetime(predictions_df[DATE_COL])
    predictions_df = add_monthly_selection_flags(predictions_df)

    raw_importance_df = pd.DataFrame(importance_rows)

    if raw_importance_df.empty:
        feature_importance_df = pd.DataFrame(
            columns=[
                "task",
                "target_column",
                "model_group",
                "model_name",
                "feature_name",
                "mean_importance",
                "std_importance",
                "signed_mean_importance",
                "n_folds",
            ]
        )
    else:
        feature_importance_df = (
            raw_importance_df.groupby(
                ["task", "target_column", "model_group", "model_name", "feature_name"],
                as_index=False,
            )
            .agg(
                mean_importance=("importance", "mean"),
                std_importance=("importance", "std"),
                signed_mean_importance=("signed_importance", "mean"),
                n_folds=("test_month", "nunique"),
            )
            .sort_values(
                ["task", "model_group", "model_name", "mean_importance"],
                ascending=[True, True, True, False],
            )
        )

    return predictions_df, feature_importance_df, warnings_list


def save_outputs(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    predictions_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    warnings_list: list[str],
    rank_higher_is_better: bool,
) -> None:
    ensure_results_dir()

    metrics = compute_all_metrics(predictions_df)

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    payload = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(MASTER_DATA_PATH.relative_to(PROJECT_ROOT)),
        "selected_features_file": str(SELECTED_FEATURES_PATH.relative_to(PROJECT_ROOT)),
        "n_rows_input": int(len(df)),
        "n_months_input": int(df[DATE_COL].nunique()),
        "n_predictions": int(len(predictions_df)),
        "train_window_months": TRAIN_WINDOW_MONTHS,
        "min_train_months": MIN_TRAIN_MONTHS,
        "rank_higher_is_better": rank_higher_is_better,
        "feature_sets": {
            group_name: {
                "n_features": len(features),
                "features": features,
            }
            for group_name, features in feature_sets.items()
        },
        "model_availability": {
            "xgboost_available": HAS_XGBOOST,
            "lightgbm_available": HAS_LIGHTGBM,
        },
        "warnings": warnings_list,
        "metrics": metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=to_json_safe)

    logger.info("Saved metrics: %s", METRICS_PATH)
    logger.info("Saved predictions: %s", PREDICTIONS_PATH)
    logger.info("Saved feature importance: %s", FEATURE_IMPORTANCE_PATH)


def main() -> None:
    ensure_results_dir()

    df = load_master_dataset()
    feature_sets, feature_warnings = build_feature_sets(df)
    rank_higher_is_better = detect_rank_orientation(df)

    logger.info("Rank orientation: larger next_1m_rank is better = %s", rank_higher_is_better)

    predictions_df, feature_importance_df, training_warnings = train_walk_forward_models(
        df=df,
        feature_sets=feature_sets,
        rank_higher_is_better=rank_higher_is_better,
    )

    all_warnings = feature_warnings + training_warnings

    save_outputs(
        df=df,
        feature_sets=feature_sets,
        predictions_df=predictions_df,
        feature_importance_df=feature_importance_df,
        warnings_list=all_warnings,
        rank_higher_is_better=rank_higher_is_better,
    )

    print("\nStep 11 complete.")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings[:20]:
            print(f"- {warning}")
        if len(all_warnings) > 20:
            print(
                f"- ... {len(all_warnings) - 20} additional warnings are stored in model_metrics.json"
            )


if __name__ == "__main__":
    main()
