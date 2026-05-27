"""Build anti-leakage features for the XGBoost benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import MAIN_TARGET


LAG_FEATURES = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
EVENT_FEATURES = [
    "covid_period",
    "border_reopening_period",
    "recovery_period",
    "russia_ukraine_war",
    "china_us_tension",
    "iran_israel_conflict",
]


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["covid_period"] = ((idx >= "2020-03-01") & (idx <= "2022-02-01")).astype(int)
    out["border_reopening_period"] = (idx >= "2022-03-01").astype(int)
    out["recovery_period"] = (idx >= "2023-01-01").astype(int)
    out["russia_ukraine_war"] = (idx >= "2022-02-01").astype(int)
    out["china_us_tension"] = (idx >= "2018-07-01").astype(int)
    out["iran_israel_conflict"] = (idx >= "2024-04-01").astype(int)
    return out


def build_feature_frame(df: pd.DataFrame, target: str = MAIN_TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Create supervised features using only lagged target information.

    Same-month segment arrivals are intentionally excluded to avoid leakage.
    """

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    out = out.sort_index()
    out.index = pd.DatetimeIndex(out.index)

    y = out[target].astype(float)
    features = pd.DataFrame(index=out.index)
    for lag in LAG_FEATURES:
        features[f"lag_{lag}"] = y.shift(lag)
    shifted = y.shift(1)
    for window in ROLLING_WINDOWS:
        features[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        features[f"rolling_std_{window}"] = shifted.rolling(window).std()

    features["month"] = features.index.month
    features["quarter"] = features.index.quarter
    features["year"] = features.index.year
    features["time_index"] = range(len(features))
    features = add_event_features(features)

    data = features.join(y.rename(target)).dropna()
    return data.drop(columns=[target]), data[target]


def feature_columns() -> list[str]:
    rolling_cols: list[str] = []
    for window in ROLLING_WINDOWS:
        rolling_cols.extend([f"rolling_mean_{window}", f"rolling_std_{window}"])
    return [
        *[f"lag_{lag}" for lag in LAG_FEATURES],
        *rolling_cols,
        "month",
        "quarter",
        "year",
        "time_index",
        *EVENT_FEATURES,
    ]
