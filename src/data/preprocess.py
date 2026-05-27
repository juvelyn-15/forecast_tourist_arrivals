"""Clean VNAT monthly segment data for forecasting.

Run from the repository root:
    python src/data/preprocess.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import CLEAN_SEGMENTS, END_DATE, RAW_SEGMENTS, SEGMENT_COLUMNS, START_DATE, TABLES, TARGET_COLUMNS
from src.data.validate_data import find_repeated_vectors, load_raw, validate, save_validation_outputs


def _time_interpolate(series: pd.Series) -> pd.Series:
    out = series.astype(float).interpolate(method="time", limit_direction="both")
    month_medians = out.groupby(out.index.month).transform("median")
    out = out.fillna(month_medians).ffill().bfill()
    return out


def clean_segments(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a complete monthly modeling table for 2012-2025.

    The raw file is preserved. This function flags stale repeated vectors and
    structurally inconsistent rows as missing for the affected target columns,
    then imputes monthly values with deterministic time interpolation.
    """

    raw = raw.drop_duplicates("date", keep="last").sort_values("date").copy()
    expected = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="MS")})
    df = expected.merge(raw, on="date", how="left")

    repeated_dates = set(raw.loc[find_repeated_vectors(raw), "date"])
    segment_sum = df[SEGMENT_COLUMNS].sum(axis=1, min_count=len(SEGMENT_COLUMNS))
    pct_gap = (segment_sum - df["international_arrivals"]).abs() / df["international_arrivals"]
    inconsistent = pct_gap > 0.05
    missing_segments = df[SEGMENT_COLUMNS].isna().any(axis=1)

    df["was_imputed"] = False
    df["quality_flag"] = "observed"

    stale_mask = df["date"].isin(repeated_dates)
    df.loc[stale_mask, TARGET_COLUMNS] = np.nan
    df.loc[stale_mask, ["was_imputed", "quality_flag"]] = [True, "stale_repeated_vector"]

    segment_problem = inconsistent | missing_segments
    df.loc[segment_problem, TARGET_COLUMNS] = np.nan
    df.loc[segment_problem & ~stale_mask, ["was_imputed", "quality_flag"]] = [True, "segment_reconciliation"]

    missing_month = df[TARGET_COLUMNS].isna().any(axis=1)
    df.loc[missing_month & ~stale_mask & ~segment_problem, ["was_imputed", "quality_flag"]] = [True, "missing_or_partial_month"]

    df = df.set_index("date")
    for col in TARGET_COLUMNS:
        df[col] = _time_interpolate(df[col])

    # Preserve exact additivity by deriving other markets as the residual when feasible.
    known_segments = df[["asia_arrivals", "europe_arrivals", "americas_arrivals", "oceania_arrivals"]].sum(axis=1)
    residual_other = df["international_arrivals"] - known_segments
    df["other_markets_arrivals"] = np.where(residual_other > 0, residual_other, df["other_markets_arrivals"])

    for col in TARGET_COLUMNS:
        df[col] = df[col].round().astype(int)
        df[f"log_{col}"] = np.log(df[col])
        df[f"{col}_mom_growth_pct"] = df[col].pct_change(1) * 100
        df[f"{col}_yoy_growth_pct"] = df[col].pct_change(12) * 100

    df["covid_period"] = ((df.index >= "2020-03-01") & (df.index <= "2022-02-01")).astype(int)
    df["recovery_period"] = (df.index >= "2022-03-01").astype(int)
    df["month"] = df.index.month
    df["year"] = df.index.year

    ordered = ["year", "month", *TARGET_COLUMNS, "was_imputed", "quality_flag", "covid_period", "recovery_period"]
    derived = [c for c in df.columns if c.startswith("log_") or c.endswith("_growth_pct")]
    return df.reset_index()[["date", *ordered, *derived]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess VNAT segment data for forecasting.")
    parser.add_argument("--input", default=RAW_SEGMENTS)
    parser.add_argument("--output", default=CLEAN_SEGMENTS)
    args = parser.parse_args()

    raw = load_raw(args.input)
    validation = validate(raw)
    save_validation_outputs(validation)

    clean = clean_segments(raw)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    clean.to_csv(path, index=False)
    clean[["date", "quality_flag", "was_imputed"]].to_csv(TABLES / "preprocessing_flags.csv", index=False)
    print(f"Clean data saved to {path}")
    print(f"Rows: {len(clean)}")
    print(clean["quality_flag"].value_counts().to_string())


if __name__ == "__main__":
    main()
