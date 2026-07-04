from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_OUTPUTS = [
    "data/processed/master_modeling_dataset.parquet",
    "results/selected_features.csv",
    "results/walk_forward_predictions.csv",
    "results/model_metrics.json",
    "results/feature_importance.csv",
    "results/portfolio_weights.csv",
    "results/backtest_results.csv",
    "results/backtest_summary.json",
    "reports/final_research_report.md",
    "reports/model_card.md",
    "reports/figures/cumulative_returns.png",
    "reports/figures/drawdown_curve.png",
    "reports/figures/feature_importance.png",
    "reports/figures/walk_forward_rank_ic.png",
    "reports/figures/portfolio_weights_over_time.png",
]


def check_file(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return {
            "file": relative_path,
            "exists": False,
            "non_empty": False,
            "size_bytes": 0,
            "status": "MISSING",
        }

    size = path.stat().st_size

    return {
        "file": relative_path,
        "exists": True,
        "non_empty": size > 0,
        "size_bytes": size,
        "status": "OK" if size > 0 else "EMPTY",
    }


def validate_json(relative_path: str) -> list[str]:
    errors = []
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        errors.append(f"{relative_path}: missing.")
        return errors

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        errors.append(f"{relative_path}: invalid JSON: {exc}")
        return errors

    if not data:
        errors.append(f"{relative_path}: JSON is empty.")

    return errors


def validate_csv(relative_path: str, required_columns: list[str] | None = None) -> list[str]:
    errors = []
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        errors.append(f"{relative_path}: missing.")
        return errors

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"{relative_path}: cannot read CSV: {exc}")
        return errors

    if df.empty:
        errors.append(f"{relative_path}: CSV is empty.")

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            errors.append(f"{relative_path}: missing columns {missing}")

    return errors


def validate_parquet(relative_path: str, required_columns: list[str] | None = None) -> list[str]:
    errors = []
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        errors.append(f"{relative_path}: missing.")
        return errors

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        errors.append(f"{relative_path}: cannot read Parquet: {exc}")
        return errors

    if df.empty:
        errors.append(f"{relative_path}: Parquet file is empty.")

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            errors.append(f"{relative_path}: missing columns {missing}")

    return errors


def validate_markdown(relative_path: str, required_terms: list[str]) -> list[str]:
    errors = []
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        errors.append(f"{relative_path}: missing.")
        return errors

    text = path.read_text(encoding="utf-8", errors="ignore")

    if len(text.strip()) < 300:
        errors.append(f"{relative_path}: file appears too short.")

    missing = [term for term in required_terms if term.lower() not in text.lower()]
    if missing:
        errors.append(f"{relative_path}: missing expected terms {missing}")

    return errors


def main() -> None:
    rows = [check_file(path) for path in REQUIRED_OUTPUTS]
    audit_df = pd.DataFrame(rows)

    errors = []

    missing_or_empty = audit_df[audit_df["status"] != "OK"]
    for _, row in missing_or_empty.iterrows():
        errors.append(f"{row['file']}: {row['status']}")

    errors.extend(
        validate_parquet(
            "data/processed/master_modeling_dataset.parquet",
            ["month_end_date", "ticker"],
        )
    )

    errors.extend(validate_csv("results/selected_features.csv"))
    errors.extend(
        validate_csv(
            "results/walk_forward_predictions.csv",
            [
                "month_end_date",
                "ticker",
                "actual_next_return",
                "predicted_return",
                "model_name",
                "feature_set",
                "training_start",
                "training_end",
                "test_date",
            ],
        )
    )
    errors.extend(validate_json("results/model_metrics.json"))
    errors.extend(validate_csv("results/feature_importance.csv"))
    errors.extend(
        validate_csv("results/portfolio_weights.csv", ["month_end_date", "ticker", "weight"])
    )
    errors.extend(validate_csv("results/backtest_results.csv"))
    errors.extend(validate_json("results/backtest_summary.json"))

    errors.extend(
        validate_markdown(
            "reports/final_research_report.md",
            [
                "Executive Summary",
                "Research Question",
                "Data Sources",
                "No-Lookahead",
                "Walk-Forward",
                "Backtest",
                "Results",
                "Limitations",
            ],
        )
    )

    errors.extend(
        validate_markdown(
            "reports/model_card.md",
            [
                "Model purpose",
                "Data sources",
                "Prediction target",
                "Validation method",
                "Known limitations",
                "Model risk controls",
            ],
        )
    )

    audit_path = PROJECT_ROOT / "reports" / "final_output_audit.md"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Final Output Audit",
        "",
        "## Required Output Files",
        "",
        audit_df.to_markdown(index=False),
        "",
        "## Validation Result",
        "",
    ]

    if errors:
        lines.append("Status: FAILED")
        lines.append("")
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("Status: PASSED")
        lines.append("")
        lines.append("All required final GitHub outputs exist and passed basic validation checks.")

    audit_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nFinal Output Audit")
    print("==================")
    print(audit_df.to_string(index=False))
    print(f"\nAudit report written to: {audit_path}")

    if errors:
        print("\nFAILED VALIDATION:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nPASSED: all required final outputs are present and valid.")


if __name__ == "__main__":
    main()
