from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "results" / "walk_forward_predictions.csv"


def read_csv_or_skip(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"Required file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        pytest.fail(f"File exists but is empty: {path}")
    return df


def test_walk_forward_train_dates_must_precede_test_dates():
    df = read_csv_or_skip(PREDICTIONS_PATH)

    required_columns = {"training_start", "training_end", "test_date"}
    missing = required_columns - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

    training_start = pd.to_datetime(df["training_start"], errors="coerce")
    training_end = pd.to_datetime(df["training_end"], errors="coerce")
    test_date = pd.to_datetime(df["test_date"], errors="coerce")

    assert training_start.notna().all(), "Some training_start values could not be parsed as dates."
    assert training_end.notna().all(), "Some training_end values could not be parsed as dates."
    assert test_date.notna().all(), "Some test_date values could not be parsed as dates."

    assert (training_start <= training_end).all(), (
        "Found rows where training_start is after training_end."
    )

    assert (training_end < test_date).all(), (
        "Lookahead violation: training_end must be strictly before test_date."
    )


def test_prediction_month_does_not_precede_training_end():
    df = read_csv_or_skip(PREDICTIONS_PATH)

    required_columns = {"training_end"}
    missing = required_columns - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

    prediction_date_column = "month_end_date" if "month_end_date" in df.columns else "test_date"
    assert prediction_date_column in df.columns, (
        "Predictions file must contain either month_end_date or test_date."
    )

    training_end = pd.to_datetime(df["training_end"], errors="coerce")
    prediction_date = pd.to_datetime(df[prediction_date_column], errors="coerce")

    assert training_end.notna().all(), "Some training_end values could not be parsed as dates."
    assert prediction_date.notna().all(), (
        f"Some {prediction_date_column} values could not be parsed as dates."
    )

    assert (training_end < prediction_date).all(), (
        f"Lookahead violation: training_end must be before {prediction_date_column}."
    )
