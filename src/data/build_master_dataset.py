"""
Build the master no-lookahead modeling dataset.

Inputs
------
data/interim/price_features_monthly.parquet
data/interim/macro_features_monthly.parquet
data/interim/fed_text_features_monthly.parquet

Output
------
data/processed/master_modeling_dataset.parquet

Design rule
-----------
For each ETF-month row, macro and Fed text features are merged using only
records with month_end_date <= the ETF feature month_end_date.

Important limitation
--------------------
This script prevents future-date joins. It does not reconstruct real-time
macroeconomic data vintages or official release-lag calendars. For a stricter
institutional-grade no-lookahead setup, use ALFRED/FRED vintages and release
dates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


TARGET_COLUMNS = [
    "next_1m_return",
    "next_1m_rank",
    "top_tercile_next_month",
]

DATE_COLUMN_CANDIDATES = [
    "month_end_date",
    "date",
    "month",
    "month_date",
    "feature_date",
    "document_month",
    "period",
]

TICKER_COLUMN_CANDIDATES = [
    "ticker",
    "symbol",
    "asset",
    "etf",
]

LEAKAGE_PATTERNS = [
    r"^future_",
    r"_future_",
    r"future$",
    r"^forward_",
    r"_forward_",
    r"forward$",
    r"^lead_",
    r"_lead_",
    r"lead$",
    r"lookahead",
    r"leakage",
    r"^target$",
    r"^target_",
    r"_target$",
    r"^label$",
    r"^label_",
    r"_label$",
    r"^next_",
    r"_next_",
    r"next$",
]


def project_root() -> Path:
    """
    Return repository root assuming this file is located at src/data/.
    """
    return Path(__file__).resolve().parents[2]


def read_table(path: Path) -> pd.DataFrame:
    """
    Read parquet or csv input.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path}. Use .parquet or .csv.")


def normalize_month_end_date(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Ensure a month_end_date column exists and is normalized to calendar month-end.
    """
    df = df.copy()

    if "month_end_date" not in df.columns:
        found = None
        for candidate in DATE_COLUMN_CANDIDATES:
            if candidate in df.columns:
                found = candidate
                break

        if found is None:
            raise KeyError(
                f"{table_name} must contain one of these date columns: {DATE_COLUMN_CANDIDATES}"
            )

        df = df.rename(columns={found: "month_end_date"})

    dates = pd.to_datetime(df["month_end_date"], errors="coerce", utc=True)
    dates = dates.dt.tz_convert(None)
    df["month_end_date"] = dates.dt.to_period("M").dt.to_timestamp("M")

    bad_dates = df["month_end_date"].isna().sum()
    if bad_dates > 0:
        raise ValueError(f"{table_name} has {bad_dates:,} rows with invalid month_end_date values.")

    return df


def normalize_ticker(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Ensure a ticker column exists and is consistently formatted.
    """
    df = df.copy()

    if "ticker" not in df.columns:
        found = None
        for candidate in TICKER_COLUMN_CANDIDATES:
            if candidate in df.columns:
                found = candidate
                break

        if found is None:
            raise KeyError(
                f"{table_name} must contain one of these ticker columns: {TICKER_COLUMN_CANDIDATES}"
            )

        df = df.rename(columns={found: "ticker"})

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    if (df["ticker"] == "").any():
        raise ValueError(f"{table_name} contains blank ticker values.")

    return df


def collapse_duplicate_keys(
    df: pd.DataFrame,
    key_columns: list[str],
    table_name: str,
    allow_collapse: bool,
) -> pd.DataFrame:
    """
    Either raise on duplicate keys or collapse duplicate rows by key.

    Price features should not have duplicates because each ticker-month row is
    one modeling observation. Macro and Fed text features may have multiple rows
    per month, so they can be safely collapsed by mean for numeric columns and
    first value for nonnumeric columns.
    """
    duplicate_count = df.duplicated(key_columns).sum()

    if duplicate_count == 0:
        return df

    if not allow_collapse:
        sample = df.loc[df.duplicated(key_columns, keep=False), key_columns].head(10)
        raise ValueError(
            f"{table_name} contains {duplicate_count:,} duplicate rows for keys "
            f"{key_columns}. Sample:\n{sample}"
        )

    agg_map: dict[str, str] = {}
    for column in df.columns:
        if column in key_columns:
            continue

        if is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            agg_map[column] = "mean"
        else:
            agg_map[column] = "first"

    return df.groupby(key_columns, as_index=False).agg(agg_map)


def leakage_regex() -> re.Pattern[str]:
    """
    Compile leakage column-name patterns.
    """
    return re.compile("|".join(LEAKAGE_PATTERNS), flags=re.IGNORECASE)


def drop_leakage_columns(
    df: pd.DataFrame,
    table_name: str,
    allowed_targets: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Drop columns that look like future/target/label columns unless explicitly allowed.
    """
    df = df.copy()
    allowed = set(allowed_targets)
    pattern = leakage_regex()

    leakage_columns = [
        column
        for column in df.columns
        if column not in allowed
        and column not in {"month_end_date", "ticker"}
        and pattern.search(column)
    ]

    if leakage_columns:
        print(f"[INFO] Dropping possible leakage columns from {table_name}: {leakage_columns}")
        df = df.drop(columns=leakage_columns)

    return df


def keep_model_ready_columns(
    df: pd.DataFrame,
    table_name: str,
    key_columns: Iterable[str],
    allowed_targets: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Keep keys, allowed targets, and numeric/bool model features.

    This intentionally removes raw text, titles, URLs, and other object columns
    from the modeling table.
    """
    key_set = set(key_columns)
    target_set = set(allowed_targets)

    keep_columns: list[str] = []

    for column in df.columns:
        if column in key_set or column in target_set:
            keep_columns.append(column)
            continue

        if is_numeric_dtype(df[column]) or is_bool_dtype(df[column]):
            keep_columns.append(column)

    dropped_columns = [column for column in df.columns if column not in keep_columns]

    if dropped_columns:
        print(f"[INFO] Dropping non-model columns from {table_name}: {dropped_columns}")

    return df[keep_columns].copy()


def resolve_right_side_conflicts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_prefix: str,
) -> pd.DataFrame:
    """
    Rename right-side columns if they would collide with existing left-side columns.
    """
    right = right.copy()

    protected = {"month_end_date"}
    conflicts = sorted((set(left.columns) & set(right.columns)) - protected)

    rename_map = {column: f"{right_prefix}_{column}" for column in conflicts}

    if rename_map:
        print(f"[INFO] Renaming conflicting columns: {rename_map}")
        right = right.rename(columns=rename_map)

    return right


def asof_merge_monthly_features(
    left: pd.DataFrame,
    right: pd.DataFrame,
    table_name: str,
    right_prefix: str,
) -> tuple[pd.DataFrame, str]:
    """
    Merge monthly features using backward as-of logic.

    For each left row with month t, the right-side row must have
    month_end_date <= t.
    """
    if right.empty:
        raise ValueError(f"{table_name} is empty.")

    right = resolve_right_side_conflicts(left, right, right_prefix)

    source_date_column = f"{right_prefix}_source_month_end_date"
    right[source_date_column] = right["month_end_date"]

    left_sorted = left.sort_values(["month_end_date", "ticker"]).reset_index(drop=True)
    right_sorted = right.sort_values("month_end_date").reset_index(drop=True)

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="month_end_date",
        direction="backward",
    )

    future_join_mask = merged[source_date_column].notna() & (
        merged[source_date_column] > merged["month_end_date"]
    )

    if future_join_mask.any():
        raise AssertionError(f"No-lookahead violation detected after merging {table_name}.")

    return merged, source_date_column


def validate_required_targets(df: pd.DataFrame) -> None:
    """
    Confirm that the price feature table contains the required modeling targets.
    """
    missing = [column for column in TARGET_COLUMNS if column not in df.columns]

    if missing:
        raise KeyError(
            "The price feature table is missing required target columns: "
            f"{missing}. Re-run src/features/price_features.py first."
        )


def validate_final_dataset(df: pd.DataFrame) -> None:
    """
    Final quality checks for the master modeling dataset.
    """
    required = ["month_end_date", "ticker", *TARGET_COLUMNS]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Final dataset is missing required columns: {missing}")

    duplicate_count = df.duplicated(["month_end_date", "ticker"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Final dataset contains {duplicate_count:,} duplicate ticker-month rows.")

    if df["month_end_date"].isna().any():
        raise ValueError("Final dataset contains missing month_end_date values.")

    if df["ticker"].isna().any():
        raise ValueError("Final dataset contains missing ticker values.")

    if df.empty:
        raise ValueError("Final dataset is empty.")


def build_master_dataset(
    price_features_path: Path,
    macro_features_path: Path,
    fed_text_features_path: Path,
    output_path: Path,
    drop_missing_targets: bool = True,
) -> pd.DataFrame:
    """
    Build and save the master modeling dataset.
    """
    print("[INFO] Loading input feature tables.")

    price = read_table(price_features_path)
    macro = read_table(macro_features_path)
    fed_text = read_table(fed_text_features_path)

    print(f"[INFO] Price features shape: {price.shape}")
    print(f"[INFO] Macro features shape: {macro.shape}")
    print(f"[INFO] Fed text features shape: {fed_text.shape}")

    price = normalize_month_end_date(price, "price_features")
    price = normalize_ticker(price, "price_features")
    validate_required_targets(price)

    macro = normalize_month_end_date(macro, "macro_features")
    fed_text = normalize_month_end_date(fed_text, "fed_text_features")

    price = collapse_duplicate_keys(
        price,
        key_columns=["month_end_date", "ticker"],
        table_name="price_features",
        allow_collapse=False,
    )

    macro = collapse_duplicate_keys(
        macro,
        key_columns=["month_end_date"],
        table_name="macro_features",
        allow_collapse=True,
    )

    fed_text = collapse_duplicate_keys(
        fed_text,
        key_columns=["month_end_date"],
        table_name="fed_text_features",
        allow_collapse=True,
    )

    price = drop_leakage_columns(
        price,
        table_name="price_features",
        allowed_targets=TARGET_COLUMNS,
    )

    macro = drop_leakage_columns(
        macro,
        table_name="macro_features",
        allowed_targets=[],
    )

    fed_text = drop_leakage_columns(
        fed_text,
        table_name="fed_text_features",
        allowed_targets=[],
    )

    price = keep_model_ready_columns(
        price,
        table_name="price_features",
        key_columns=["month_end_date", "ticker"],
        allowed_targets=TARGET_COLUMNS,
    )

    macro = keep_model_ready_columns(
        macro,
        table_name="macro_features",
        key_columns=["month_end_date"],
        allowed_targets=[],
    )

    fed_text = keep_model_ready_columns(
        fed_text,
        table_name="fed_text_features",
        key_columns=["month_end_date"],
        allowed_targets=[],
    )

    print("[INFO] Merging macro features with backward no-lookahead logic.")
    master, macro_source_col = asof_merge_monthly_features(
        left=price,
        right=macro,
        table_name="macro_features",
        right_prefix="macro",
    )

    print("[INFO] Merging Fed text features with backward no-lookahead logic.")
    master, fed_source_col = asof_merge_monthly_features(
        left=master,
        right=fed_text,
        table_name="fed_text_features",
        right_prefix="fed_text",
    )

    if drop_missing_targets:
        before = len(master)
        master = master.dropna(subset=TARGET_COLUMNS).copy()
        after = len(master)
        print(f"[INFO] Dropped {before - after:,} rows with missing targets.")

    source_columns = [macro_source_col, fed_source_col]

    feature_columns = [
        column
        for column in master.columns
        if column not in {"month_end_date", "ticker", *TARGET_COLUMNS, *source_columns}
    ]

    ordered_columns = [
        "month_end_date",
        "ticker",
        *feature_columns,
        *TARGET_COLUMNS,
    ]

    master = master[ordered_columns].copy()
    master = master.sort_values(["month_end_date", "ticker"]).reset_index(drop=True)

    validate_final_dataset(master)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(output_path, index=False)

    print("[INFO] Master modeling dataset created successfully.")
    print(f"[INFO] Output path: {output_path}")
    print(f"[INFO] Final shape: {master.shape}")
    print(
        f"[INFO] Date range: {master['month_end_date'].min().date()} to {master['month_end_date'].max().date()}"
    )
    print(f"[INFO] Number of tickers: {master['ticker'].nunique():,}")
    print(f"[INFO] Number of feature columns: {len(feature_columns):,}")
    print(f"[INFO] Target columns: {TARGET_COLUMNS}")

    return master


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Build the master no-lookahead ETF modeling dataset."
    )

    parser.add_argument(
        "--price-features",
        type=Path,
        default=root / "data" / "interim" / "price_features_monthly.parquet",
        help="Path to monthly ETF price features.",
    )

    parser.add_argument(
        "--macro-features",
        type=Path,
        default=root / "data" / "interim" / "macro_features_monthly.parquet",
        help="Path to monthly macro features.",
    )

    parser.add_argument(
        "--fed-text-features",
        type=Path,
        default=root / "data" / "interim" / "fed_text_features_monthly.parquet",
        help="Path to monthly Fed text features.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "master_modeling_dataset.parquet",
        help="Output path for the master modeling dataset.",
    )

    parser.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help="Keep rows with missing next-month targets. Default is to drop them.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_master_dataset(
        price_features_path=args.price_features,
        macro_features_path=args.macro_features,
        fed_text_features_path=args.fed_text_features,
        output_path=args.output,
        drop_missing_targets=not args.keep_missing_targets,
    )


if __name__ == "__main__":
    main()
