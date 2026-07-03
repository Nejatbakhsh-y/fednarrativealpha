from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MASTER_DATASET_PATH = ROOT / "data" / "processed" / "master_modeling_dataset.parquet"


def read_master_or_skip() -> pd.DataFrame:
    if not MASTER_DATASET_PATH.exists():
        pytest.skip(f"Required file not found: {MASTER_DATASET_PATH}")
    df = pd.read_parquet(MASTER_DATASET_PATH)
    if df.empty:
        pytest.fail(f"File exists but is empty: {MASTER_DATASET_PATH}")
    return df


def test_master_dataset_has_required_time_keys():
    df = read_master_or_skip()

    required_columns = {"month_end_date", "ticker"}
    missing = required_columns - set(df.columns)

    assert not missing, f"Missing required master dataset columns: {sorted(missing)}"

    month_end_date = pd.to_datetime(df["month_end_date"], errors="coerce")
    assert month_end_date.notna().all(), "Some month_end_date values could not be parsed."

    duplicates = df.duplicated(subset=["month_end_date", "ticker"])
    assert not duplicates.any(), (
        "Master dataset contains duplicate month_end_date/ticker rows."
    )


def test_feature_availability_dates_not_after_prediction_date():
    df = read_master_or_skip()

    assert "month_end_date" in df.columns, "month_end_date column is required."

    prediction_date = pd.to_datetime(df["month_end_date"], errors="coerce")
    assert prediction_date.notna().all(), "Some month_end_date values could not be parsed."

    date_column_keywords = (
        "feature_date",
        "available_date",
        "availability_date",
        "release_date",
        "as_of_date",
        "source_date",
        "effective_date",
    )

    feature_date_columns = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in date_column_keywords)
    ]

    if not feature_date_columns:
        pytest.skip(
            "No explicit feature availability date columns found. "
            "Checked for feature_date, available_date, release_date, as_of_date, "
            "source_date, and effective_date columns."
        )

    for column in feature_date_columns:
        feature_date = pd.to_datetime(df[column], errors="coerce")
        valid_rows = feature_date.notna()

        violations = df.loc[valid_rows & (feature_date > prediction_date), [
            "month_end_date",
            "ticker",
            column,
        ]]

        assert violations.empty, (
            f"No-lookahead violation in {column}: feature date is later than "
            f"the prediction month_end_date. Example rows:\n{violations.head(10)}"
        )
