"""Shared project configuration.

All paths are resolved relative to the repository root so scripts can be run
from the root directory without machine-specific absolute paths.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

RAW_SEGMENTS = DATA_RAW / "vnat_monthly_segments.csv"
CLEAN_SEGMENTS = DATA_PROCESSED / "vnat_monthly_segments_clean.csv"

START_DATE = "2012-01-01"
END_DATE = "2025-12-01"
TRAIN_END = "2022-12-01"
TEST_START = "2023-01-01"

TARGET_COLUMNS = [
    "international_arrivals",
    "asia_arrivals",
    "europe_arrivals",
    "americas_arrivals",
    "oceania_arrivals",
    "other_markets_arrivals",
]

SEGMENT_COLUMNS = [
    "asia_arrivals",
    "europe_arrivals",
    "americas_arrivals",
    "oceania_arrivals",
    "other_markets_arrivals",
]

SERIES_LABELS = {
    "international_arrivals": "International arrivals",
    "asia_arrivals": "Asia",
    "europe_arrivals": "Europe",
    "americas_arrivals": "Americas",
    "oceania_arrivals": "Oceania",
    "other_markets_arrivals": "Other markets",
}

SERIES_COLORS = {
    "international_arrivals": "#2f3437",
    "asia_arrivals": "#4e79a7",
    "europe_arrivals": "#59a14f",
    "americas_arrivals": "#b66d2d",
    "oceania_arrivals": "#8f6aa8",
    "other_markets_arrivals": "#7f7f7f",
}

EVENTS = {
    "COVID-19 outbreak": "2020-03-01",
    "Vietnam border reopening": "2022-03-01",
    "Tourism recovery period": "2023-01-01",
    "Russia-Ukraine war": "2022-02-01",
    "China-US economic tensions": "2018-07-01",
    "Iran-Israel conflict": "2024-04-01",
}
