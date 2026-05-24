"""
crawl_vnat.py
-------------
Collects monthly international tourist arrivals from VNAT
(Vietnam National Administration of Tourism).

VNAT publishes monthly statistics as Excel/PDF reports on:
    https://vietnamtourism.gov.vn/

Strategy:
1. First tries to download known Excel report URLs (structured patterns).
2. Falls back to scraping the statistical reports page for download links.
3. Saves raw files to data/raw/vnat/ and parses into a monthly CSV.

IMPORTANT: VNAT's website structure may change. If the scraper fails,
the manual fallback section provides instructions.

Usage:
    python src/crawl/crawl_vnat.py
"""

import os
import re
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL    = "https://vietnamtourism.gov.vn"
STATS_URL   = f"{BASE_URL}/thong-ke"          # Statistics section
RAW_DIR     = Path(__file__).parents[2] / "data" / "raw" / "vnat"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Known direct download patterns (update as VNAT publishes new files) ────────
# File naming convention observed from VNAT: khach-quoc-te-YYYY-MMthang.xlsx
KNOWN_EXCEL_PATTERNS = [
    "khach-quoc-te-{year}-thang-{month:02d}.xlsx",
    "kqt-t{month:02d}-{year}.xlsx",
]


def try_direct_download(year: int, month: int) -> pd.DataFrame | None:
    """Attempt to download VNAT monthly report by guessing URL pattern."""
    for pattern in KNOWN_EXCEL_PATTERNS:
        fname = pattern.format(year=year, month=month)
        url = f"{BASE_URL}/uploads/thongke/{year}/{fname}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200 and "excel" in r.headers.get("Content-Type", ""):
                fpath = RAW_DIR / fname
                fpath.write_bytes(r.content)
                logger.info(f"  Downloaded: {fname}")
                return parse_vnat_excel(fpath, year, month)
        except Exception:
            continue
    return None


def parse_vnat_excel(fpath: Path, year: int, month: int) -> pd.DataFrame | None:
    """
    Parse a VNAT monthly Excel file.
    VNAT files typically have:
    - Row 1: Title
    - Row 3+: Country | Arrivals | % change YoY
    We extract 'Tổng số' (Total) row.
    """
    try:
        df = pd.read_excel(fpath, header=None, engine="openpyxl")
        # Find the total row (Vietnamese: "Tổng số" or "TỔNG SỐ")
        total_mask = df.apply(
            lambda col: col.astype(str).str.contains("tổng số|tong so|total", case=False, na=False)
        ).any(axis=1)
        if not total_mask.any():
            logger.warning(f"  Could not find 'Tổng số' row in {fpath.name}")
            return None
        total_row = df[total_mask].iloc[0]
        # The arrivals figure is typically in column 1 or 2
        arrivals = None
        for val in total_row[1:]:
            try:
                arrivals = int(str(val).replace(",", "").replace(".", "").strip())
                if arrivals > 1000:   # sanity check
                    break
            except (ValueError, AttributeError):
                continue
        if arrivals is None:
            return None
        return pd.DataFrame({
            "year": [year], "month": [month],
            "date": [pd.Timestamp(year=year, month=month, day=1)],
            "international_arrivals": [arrivals],
            "source": ["VNAT"]
        })
    except Exception as e:
        logger.error(f"  Parse error for {fpath.name}: {e}")
        return None


def scrape_stats_page() -> list[str]:
    """Scrape VNAT statistics page for Excel download links."""
    logger.info(f"Scraping {STATS_URL} for download links...")
    try:
        r = requests.get(STATS_URL, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(ext in href.lower() for ext in [".xlsx", ".xls", ".csv"]):
                full = href if href.startswith("http") else BASE_URL + href
                links.append(full)
        logger.info(f"  Found {len(links)} Excel links")
        return links
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return []


def download_link(url: str) -> Path | None:
    """Download a file from a direct URL and save to RAW_DIR."""
    fname = url.split("/")[-1].split("?")[0] or "vnat_file.xlsx"
    fpath = RAW_DIR / fname
    if fpath.exists():
        logger.info(f"  Already exists: {fname}")
        return fpath
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            fpath.write_bytes(r.content)
            logger.info(f"  Saved: {fname}")
            return fpath
    except Exception as e:
        logger.error(f"  Failed to download {url}: {e}")
    return None


def build_manual_dataset() -> pd.DataFrame:
    """
    MANUAL FALLBACK
    ---------------
    If automated scraping fails, you can manually download monthly reports from:
    
        https://vietnamtourism.gov.vn/thong-ke
    
    Save each Excel file to: data/raw/vnat/
    
    This function will then parse all Excel files in that directory.
    
    Alternatively, a curated dataset (2010–2024) is available via the
    merge_sources.py script which combines World Bank annual data with
    monthly UNWTO seasonal indices to reconstruct a monthly series.
    """
    records = []
    for fpath in sorted(RAW_DIR.glob("*.xlsx")):
        # Try to extract year/month from filename
        m = re.search(r"(\d{4}).*?(\d{1,2})", fpath.stem)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            df = parse_vnat_excel(fpath, year, month)
            if df is not None:
                records.append(df)
                logger.info(f"  Parsed: {fpath.name} → {df['international_arrivals'].iloc[0]:,}")
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def main():
    logger.info("=== VNAT Monthly Arrivals Crawler ===")

    all_records = []

    # Step 1: Try scraping the stats page for links
    links = scrape_stats_page()
    if links:
        for url in links[:30]:   # Limit to 30 most recent
            fpath = download_link(url)
            time.sleep(1)        # Polite crawling

    # Step 2: Parse all downloaded files
    df = build_manual_dataset()

    if df.empty:
        logger.warning(
            "\n⚠️  Automated crawl yielded no data.\n"
            "   Please manually download monthly Excel reports from:\n"
            "   https://vietnamtourism.gov.vn/thong-ke\n"
            "   and place them in: data/raw/vnat/\n"
            "   Then re-run this script.\n"
            "\n   Alternatively, run merge_sources.py to use World Bank + seasonal reconstruction."
        )
        return

    df.sort_values("date", inplace=True)
    out_path = RAW_DIR.parent / "vnat_monthly.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved {len(df)} monthly records → {out_path}")
    print(df.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
