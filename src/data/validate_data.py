"""Validate raw VNAT monthly segment arrivals.

Run from the repository root:
    python src/data/validate_data.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import (
    END_DATE,
    FIGURES,
    RAW_SEGMENTS,
    SEGMENT_COLUMNS,
    START_DATE,
    TABLES,
    TARGET_COLUMNS,
)


@dataclass(frozen=True)
class ValidationResult:
    checks: pd.DataFrame
    issues: pd.DataFrame
    monthly_frame: pd.DataFrame


def _issue(date, issue_type: str, column: str, value, detail: str) -> dict:
    return {
        "date": pd.NaT if pd.isna(date) else pd.Timestamp(date),
        "issue_type": issue_type,
        "column": column,
        "value": value,
        "detail": detail,
    }


def load_raw(path=RAW_SEGMENTS) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", *TARGET_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.loc[:, ["date", *TARGET_COLUMNS]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in TARGET_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def find_repeated_vectors(df: pd.DataFrame, min_repeats: int = 2) -> pd.Series:
    """Flag exact repeated segment vectors after the first occurrence.

    Repeated full vectors are unlikely in a monthly segment table and often
    indicate stale rendered pages from the crawler. The first occurrence is
    kept as the best available observation; later copies are marked.
    """

    vector_cols = TARGET_COLUMNS
    rounded = df[vector_cols].round(0).astype("Int64").astype("string").fillna("<NA>")
    keys = rounded.apply(lambda row: "|".join(row.astype(str).tolist()), axis=1)
    counts = keys.groupby(keys).transform("size")
    duplicate_after_first = keys.duplicated(keep="first")
    return (counts >= min_repeats) & duplicate_after_first


def validate(df: pd.DataFrame, tolerance: float = 0.05) -> ValidationResult:
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="MS")
    observed_dates = df["date"].dropna()
    issues: list[dict] = []

    duplicate_mask = df.duplicated("date", keep=False)
    for _, row in df.loc[duplicate_mask].iterrows():
        issues.append(_issue(row["date"], "duplicated_month", "date", row["date"], "Duplicate month in raw file."))

    missing_dates = expected_dates.difference(observed_dates)
    for date in missing_dates:
        issues.append(_issue(date, "missing_month", "date", "", "Month absent from raw file."))

    for col in TARGET_COLUMNS:
        bad = df[col].notna() & (df[col] <= 0)
        for _, row in df.loc[bad, ["date", col]].iterrows():
            issues.append(_issue(row["date"], "non_positive_value", col, row[col], "Value must be strictly positive."))

    segment_sum = df[SEGMENT_COLUMNS].sum(axis=1, min_count=len(SEGMENT_COLUMNS))
    pct_gap = (segment_sum - df["international_arrivals"]).abs() / df["international_arrivals"]
    inconsistent = pct_gap.notna() & (pct_gap > tolerance)
    for idx, row in df.loc[inconsistent].iterrows():
        issues.append(
            _issue(
                row["date"],
                "segment_sum_inconsistency",
                "segment_sum",
                float(segment_sum.loc[idx]),
                f"Segment sum differs from total by {pct_gap.loc[idx]:.2%}.",
            )
        )

    repeated = find_repeated_vectors(df)
    for _, row in df.loc[repeated].iterrows():
        issues.append(
            _issue(
                row["date"],
                "suspicious_repeated_values",
                "all_target_columns",
                row["international_arrivals"],
                "Exact repeated target vector after an earlier month; likely stale crawler output.",
            )
        )

    monthly = pd.DataFrame({"date": expected_dates}).merge(df, on="date", how="left", indicator=True)
    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(columns=["date", "issue_type", "column", "value", "detail"])

    checks = pd.DataFrame(
        [
            {"check": "required_columns", "status": "pass", "value": len(TARGET_COLUMNS) + 1},
            {"check": "date_range_start", "status": "pass" if observed_dates.min() == expected_dates.min() else "warn", "value": observed_dates.min()},
            {"check": "date_range_end", "status": "pass" if observed_dates.max() == expected_dates.max() else "warn", "value": observed_dates.max()},
            {"check": "expected_months", "status": "pass" if len(monthly) == 168 else "warn", "value": len(expected_dates)},
            {"check": "observed_rows", "status": "pass" if len(df) == len(expected_dates) else "warn", "value": len(df)},
            {"check": "duplicated_months", "status": "pass" if not duplicate_mask.any() else "fail", "value": int(duplicate_mask.sum())},
            {"check": "missing_months", "status": "pass" if len(missing_dates) == 0 else "fail", "value": len(missing_dates)},
            {"check": "non_positive_values", "status": "pass" if not (df[TARGET_COLUMNS] <= 0).any().any() else "fail", "value": int((df[TARGET_COLUMNS] <= 0).sum().sum())},
            {"check": "segment_sum_tolerance", "status": "pass" if not inconsistent.any() else "warn", "value": int(inconsistent.sum())},
            {"check": "suspicious_repeated_vectors", "status": "pass" if not repeated.any() else "warn", "value": int(repeated.sum())},
        ]
    )
    return ValidationResult(checks=checks, issues=issues_df, monthly_frame=monthly)


def save_validation_outputs(result: ValidationResult) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    result.checks.to_csv(TABLES / "data_validation_checks.csv", index=False)
    result.issues.to_csv(TABLES / "data_validation_issues.csv", index=False)

    summary = [
        "# VNAT Monthly Segment Data Validation",
        "",
        "This report validates the raw VNAT monthly segment data before forecasting.",
        "",
        "## Check Summary",
        "",
        result.checks.to_markdown(index=False),
        "",
        "## Issue Counts",
        "",
    ]
    counts = result.issues["issue_type"].value_counts().rename_axis("issue_type").reset_index(name="count")
    summary.append(counts.to_markdown(index=False) if not counts.empty else "No issues detected.")
    summary.append("")
    (TABLES / "data_validation_report.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw VNAT monthly segment data.")
    parser.add_argument("--input", default=RAW_SEGMENTS, help="Input raw CSV path.")
    args = parser.parse_args()

    df = load_raw(args.input)
    result = validate(df)
    save_validation_outputs(result)
    print(result.checks.to_string(index=False))
    print(f"\nIssues written to {TABLES / 'data_validation_issues.csv'}")


if __name__ == "__main__":
    main()
