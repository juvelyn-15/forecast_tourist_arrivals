"""
crawl_worldbank.py
------------------
Fetches international tourist arrivals to Vietnam from the World Bank API.

Indicator: ST.INT.ARVL  — International tourism, number of arrivals
Country:   VNM (Vietnam)
Frequency: Annual (World Bank only provides annual; monthly sourced from VNAT)

This script:
1. Downloads annual arrivals (2000–2023) via wbgapi
2. Downloads supplementary macro variables for VAR model:
   - NY.GDP.MKTP.CD  — Vietnam GDP (current USD)
   - PA.NUS.FCRF     — Official exchange rate VND/USD
   - FP.CPI.TOTL.ZG  — CPI inflation
3. Saves to data/raw/worldbank_annual.csv

Usage:
    python src/crawl/crawl_worldbank.py
"""

import wbgapi as wb
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDICATORS = {
    "ST.INT.ARVL":   "tourist_arrivals",
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "PA.NUS.FCRF":   "exchange_rate_vnd_usd",
    "FP.CPI.TOTL.ZG": "cpi_inflation_pct",
}
COUNTRY = "VNM"
YEARS = range(2000, 2024)


def fetch_indicator(indicator_code: str, label: str) -> pd.Series:
    """Fetch a single World Bank indicator for Vietnam."""
    logger.info(f"Fetching {label} ({indicator_code})...")
    try:
        df = wb.data.DataFrame(indicator_code, economy=COUNTRY, time=YEARS, numericTimeKeys=True)
        series = df.T.squeeze()
        series.index = series.index.astype(int)
        series.name = label
        return series
    except Exception as e:
        logger.error(f"Failed to fetch {indicator_code}: {e}")
        return pd.Series(name=label, dtype=float)


def main():
    all_series = []
    for code, label in INDICATORS.items():
        s = fetch_indicator(code, label)
        all_series.append(s)

    df = pd.concat(all_series, axis=1)
    df.index.name = "year"
    df.sort_index(inplace=True)

    out_path = os.path.join(OUTPUT_DIR, "worldbank_annual.csv")
    df.to_csv(out_path)
    logger.info(f"Saved {df.shape[0]} rows × {df.shape[1]} cols → {out_path}")
    print(df.tail(10).to_string())


if __name__ == "__main__":
    main()
