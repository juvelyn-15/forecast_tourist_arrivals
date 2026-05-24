"""
preprocessing.py
----------------
Cleaning, transformation, and train/test splitting utilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_monthly_data(exclude_covid: bool = False) -> pd.DataFrame:
    """Load the master monthly dataset."""
    df = pd.read_csv(PROC_DIR / "monthly_arrivals.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)
    df.index.freq = "MS"   # Monthly start frequency
    if exclude_covid:
        df = df[df["covid_period"] == 0]
    return df


def get_series(
    df: pd.DataFrame,
    use_log: bool = False,
    col: str = "international_arrivals"
) -> pd.Series:
    """Extract the target series, optionally log-transformed."""
    if use_log:
        return df["log_arrivals"].dropna()
    return df[col].astype(float)


def train_test_split_ts(
    series: pd.Series,
    test_periods: int = 24
) -> tuple[pd.Series, pd.Series]:
    """
    Chronological train/test split for time series.
    Default: last 24 months (2 years) as test set.
    """
    train = series.iloc[:-test_periods]
    test  = series.iloc[-test_periods:]
    return train, test


def handle_covid_outliers(
    series: pd.Series,
    method: str = "interpolate"
) -> pd.Series:
    """
    Handle COVID-period outliers.

    Methods:
    - 'interpolate': Replace 2020-03 to 2022-02 with linear interpolation
                     from 2019 peak to 2022 recovery. Use for pre-COVID models.
    - 'dummy':       Keep values but add dummy variable (for ARIMAX/SARIMAX).
    - 'keep':        No modification — for models that handle breaks (e.g., GARCH).

    Returns cleaned series.
    """
    s = series.copy()
    if method == "interpolate":
        covid_start = pd.Timestamp("2020-03-01")
        covid_end   = pd.Timestamp("2022-03-01")
        mask = (s.index >= covid_start) & (s.index <= covid_end)
        s[mask] = np.nan
        s = s.interpolate(method="linear")
    return s


def create_covid_dummy(series: pd.Series) -> pd.Series:
    """Binary dummy: 1 during COVID disruption (2020-03 to 2022-02)."""
    dummy = pd.Series(0, index=series.index, name="covid_dummy")
    mask = (series.index >= "2020-03-01") & (series.index <= "2022-02-28")
    dummy[mask] = 1
    return dummy
