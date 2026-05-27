"""
crawl_vnat_segments.py

Crawl VNAT monthly international tourist arrivals by aggregate market segment:
- Total international arrivals (Tổng số)
- Asia
- Europe
- Americas
- Oceania/Australia continent
- Other markets

This version is based on the controlled Playwright crawler. It avoids stale SPA
state by opening a fresh page per target month, optionally forcing visible
controls, and validating the rendered page before extraction.

Install:
    pip install playwright beautifulsoup4 pandas numpy lxml
    python -m playwright install chromium

Run examples:
    python .\src\crawl\crawl_vnat_segments.py --year 2016 --month 4 --show-browser --debug --no-cache
    python .\src\crawl\crawl_vnat_segments.py --start 2012 --end 2025 --no-cache --debug

Output:
    data/raw/vnat_monthly_segments.csv
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

BASE_URL = "https://vietnamtourism.gov.vn/statistic/international"
OUT_DIR = Path("data/raw")
HTML_DIR = OUT_DIR / "vnat_pages_segments"
DEBUG_DIR = OUT_DIR / "vnat_debug_segments"
OUT_CSV = OUT_DIR / "vnat_monthly_segments.csv"

PAGE_TIMEOUT_MS = 45_000
TABLE_TIMEOUT_MS = 20_000
REQUEST_DELAY_SEC = 0.8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vnat_segments")


@dataclass(frozen=True)
class CrawlTarget:
    year: int
    month: int

    @property
    def period(self) -> str:
        return f"t{self.month}"

    @property
    def url(self) -> str:
        # cache-busting parameter helps prevent stale SPA/intermediate-cache state
        return f"{BASE_URL}?year={self.year}&period={self.period}&_={self.year}{self.month:02d}"

    @property
    def clean_url(self) -> str:
        return f"{BASE_URL}?year={self.year}&period={self.period}"

    @property
    def cache_name(self) -> str:
        return f"{self.year}_t{self.month:02d}.html"


# Canonical output columns and the row labels we expect on VNAT tables.
# These aliases are intentionally broad because the rendered labels may vary
# between Vietnamese, English, accent/no-accent forms, or historical periods.
SEGMENT_ALIASES: dict[str, list[str]] = {
    "international_arrivals": [
        "tổng số", "tong so", "total", "grand total", "tổng cộng", "tong cong"
    ],
    "asia_arrivals": [
        "châu á", "chau a", "asia"
    ],
    "europe_arrivals": [
        "châu âu", "chau au", "europe", "europa"
    ],
    "americas_arrivals": [
        "châu mỹ", "chau my", "america", "americas", "north america", "south america"
    ],
    "oceania_arrivals": [
        "châu úc", "chau uc", "châu đại dương", "chau dai duong", "oceania", "australia"
    ],
    "other_markets_arrivals": [
        "các thị trường khác", "cac thi truong khac",
        "thị trường khác", "thi truong khac",
        "thị trường còn lại", "thi truong con lai",
        "others", "other markets", "other market"
    ],
}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def remove_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def norm_key(value: object) -> str:
    text = normalize_text(value).lower()
    text = remove_accents(text)
    text = re.sub(r"[^a-z0-9% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_int(value: object) -> int | None:
    """Parse VNAT integer cells safely.

    Important: labels such as "1. Châu Á" contain digits, but they are not
    numeric values. Reject any cell containing alphabetic characters before
    extracting digits.
    """
    text = normalize_text(value)
    if not text or text.lower() in {"nan", "none", "-"} or "%" in text:
        return None
    if re.search(r"[A-Za-zÀ-ỹ]", text):
        return None
    cleaned = re.sub(r"[^0-9.,\s]", "", text)
    cleaned = re.sub(r"[.,\s]", "", cleaned)
    if not cleaned:
        return None
    try:
        number = int(cleaned)
    except ValueError:
        return None
    return number if number > 0 else None


def table_rows(html: str) -> list[list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[list[str]] = []
    for tr in soup.select("table tr"):
        cells = [normalize_text(td.get_text(" ", strip=True)) for td in tr.select("th,td")]
        if cells:
            rows.append(cells)
    return rows


def row_label(row: list[str]) -> str:
    """Return the most likely label cell for a VNAT row.

    VNAT aggregate rows may appear as "1. Châu Á" in the first column, not
    as a separate STT + label pair. Therefore we scan the early cells for text
    with letters and remove any leading row number.
    """
    for cell in row[:3]:
        cell_text = normalize_text(cell)
        if re.search(r"[A-Za-zÀ-ỹ]", cell_text):
            return re.sub(r"^\s*\d+\s*[\.)\-:]\s*", "", cell_text).strip()
    return row[1] if len(row) > 1 else (row[0] if row else "")


def first_large_number(row: list[str]) -> int | None:
    """Extract the current-period value from a table row.

    In VNAT rows, the first large integer after STT/label is the target month's
    arrival count. Percentage/change columns are excluded by parse_int().
    """
    candidates: list[int] = []
    for cell in row:
        n = parse_int(cell)
        if n is not None and n >= 10:  # some segment rows may be small historically
            candidates.append(n)
    if not candidates:
        return None

    # Drop likely STT/index values if they appear first.
    large_candidates = [n for n in candidates if n >= 1_000]
    if large_candidates:
        return large_candidates[0]
    return candidates[0]


def aliases_for_matching() -> dict[str, list[str]]:
    return {col: [norm_key(alias) for alias in aliases] for col, aliases in SEGMENT_ALIASES.items()}


def find_segment_values(rows: list[list[str]]) -> tuple[dict[str, int | None], dict[str, str | None]]:
    aliases = aliases_for_matching()
    values: dict[str, int | None] = {col: None for col in SEGMENT_ALIASES}
    matched_labels: dict[str, str | None] = {col: None for col in SEGMENT_ALIASES}

    for row in rows:
        label = row_label(row)
        label_norm = norm_key(label)
        if not label_norm:
            continue

        for col, patterns in aliases.items():
            if values[col] is not None:
                continue
            if any(pat and pat in label_norm for pat in patterns):
                val = first_large_number(row)
                if val is not None:
                    values[col] = val
                    matched_labels[col] = label

    return values, matched_labels


def validate_rendered_period(html: str, target: CrawlTarget, strict: bool = True) -> bool:
    all_text = norm_key(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
    month_tokens = [
        f"thang {target.month}",
        f"t{target.month}",
        f"/{target.month}/",
    ]
    has_year = str(target.year) in all_text
    has_month = any(tok in all_text for tok in month_tokens)
    if strict and not (has_year and has_month):
        log.warning(
            "STALE? target %04d-%02d not visible in rendered page text. Refusing extraction.",
            target.year,
            target.month,
        )
        return False
    return True


def extract_segments(html: str, target: CrawlTarget, strict: bool = True) -> dict | None:
    if not validate_rendered_period(html, target, strict=strict):
        return None

    rows = table_rows(html)
    values, matched_labels = find_segment_values(rows)

    # Total is the only mandatory field. Segment labels may be unavailable on
    # some VNAT pages/periods, so they are kept as missing rather than failing.
    if values["international_arrivals"] is None:
        return None

    rec: dict[str, object] = {
        "date": f"{target.year}-{target.month:02d}-01",
        "year": target.year,
        "month": target.month,
        "period": target.period,
        "url": target.clean_url,
    }
    rec.update(values)

    # Keep matched labels for auditing. These columns let you verify exactly
    # which VNAT row was mapped into each segment.
    for col, label in matched_labels.items():
        rec[f"{col.replace('_arrivals', '')}_matched_label"] = label

    return rec


async def force_year_month_controls(page, target: CrawlTarget) -> None:
    await page.evaluate(
        """
        async ({year, month}) => {
          const sleep = (ms) => new Promise(r => setTimeout(r, ms));
          const fire = (el) => {
            for (const ev of ['input', 'change']) {
              el.dispatchEvent(new Event(ev, { bubbles: true }));
            }
          };
          const yearSelect = document.querySelector('select[name="year"], #statistic-year, select.statistic-year');
          const monthSelect = document.querySelector('select[name="period"], #statistic-month, select.statistic-month');

          if (yearSelect) {
            const yearOpt = Array.from(yearSelect.options || []).find(
              o => (o.value || '').trim() === String(year) || (o.textContent || '').trim() === String(year)
            );
            if (yearOpt) {
              yearSelect.value = yearOpt.value;
              fire(yearSelect);
              await sleep(500);
            }
          }

          if (monthSelect) {
            const period = 't' + month;
            const monthOpt = Array.from(monthSelect.options || []).find(
              o => (o.value || '').trim().toLowerCase() === period ||
                   (o.textContent || '').trim().toLowerCase() === ('tháng ' + month)
            );
            if (monthOpt) {
              monthSelect.value = monthOpt.value;
              fire(monthSelect);
              await sleep(800);
            }
          }
        }
        """,
        {"year": target.year, "month": target.month},
    )


async def fetch_html(context, target: CrawlTarget, use_cache: bool, debug: bool) -> str | None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = HTML_DIR / target.cache_name

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    page = await context.new_page()
    log.info("Fetching %s", target.url)
    try:
        await page.goto(target.url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        await page.wait_for_timeout(1500)
        await force_year_month_controls(page, target)
        try:
            await page.wait_for_selector("table", timeout=TABLE_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            log.warning("No table detected: %s", target.url)
        await page.wait_for_timeout(1500)
        html = await page.content()
    except Exception as exc:
        log.warning("Failed %04d-%02d: %s", target.year, target.month, exc)
        html = None
    finally:
        if debug:
            try:
                await page.screenshot(path=str(DEBUG_DIR / f"{target.year}_t{target.month:02d}.png"), full_page=True)
            except Exception:
                pass
        await page.close()

    if html:
        cache_path.write_text(html, encoding="utf-8")
    await asyncio.sleep(REQUEST_DELAY_SEC)
    return html


async def crawl(targets: Iterable[CrawlTarget], use_cache: bool, headless: bool, debug: bool, strict: bool) -> pd.DataFrame:
    records: list[dict] = []
    failures: list[CrawlTarget] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            locale="vi-VN",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            extra_http_headers={"Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8"},
        )

        for target in targets:
            html = await fetch_html(context, target, use_cache=use_cache, debug=debug)
            if not html:
                failures.append(target)
                continue

            rec = extract_segments(html, target, strict=strict)
            if rec is None:
                log.warning("MISS %04d-%02d", target.year, target.month)
                failures.append(target)
            else:
                msg = [f"total={rec.get('international_arrivals'):,}"]
                for col in ["asia_arrivals", "europe_arrivals", "americas_arrivals", "oceania_arrivals", "other_markets_arrivals"]:
                    val = rec.get(col)
                    msg.append(f"{col.replace('_arrivals','')}={val:,}" if isinstance(val, int) else f"{col.replace('_arrivals','')}=NA")
                log.info("OK %04d-%02d -> %s", target.year, target.month, "; ".join(msg))
                records.append(rec)

        await browser.close()

    if failures:
        log.warning("Failed/stale pages: %s", ", ".join(f"{t.year}-t{t.month}" for t in failures))

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(["year", "month"], keep="last")

    numeric_cols = [col for col in SEGMENT_ALIASES if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"log_{col}"] = np.where(df[col] > 0, np.log(df[col]), np.nan)
        df[f"{col}_mom_growth_pct"] = df[col].pct_change(1) * 100
        df[f"{col}_yoy_growth_pct"] = df[col].pct_change(12) * 100

    return df


def build_targets(args: argparse.Namespace) -> list[CrawlTarget]:
    if args.year and args.month:
        return [CrawlTarget(args.year, args.month)]
    if args.year:
        return [CrawlTarget(args.year, m) for m in range(1, 13)]
    return [CrawlTarget(y, m) for y in range(args.start, args.end + 1) for m in range(1, 13)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl VNAT international arrivals by aggregate market segment")
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int, choices=range(1, 13))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Save screenshots into data/raw/vnat_debug_segments")
    parser.add_argument("--non-strict", action="store_true", help="Allow extraction even if target date is not visible")
    parser.add_argument("--output", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = build_targets(args)
    log.info("Targets: %d pages", len(targets))

    df = asyncio.run(crawl(
        targets,
        use_cache=not args.no_cache,
        headless=not args.show_browser,
        debug=args.debug,
        strict=not args.non_strict,
    ))
    if df.empty:
        raise SystemExit("No valid data extracted. Run one page with --show-browser --debug and inspect data/raw/vnat_debug_segments.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("\nSaved:", args.output)
    print("Rows:", len(df))
    print("Period:", df["date"].min().date(), "to", df["date"].max().date())

    preview_cols = [
        "date", "international_arrivals", "asia_arrivals", "europe_arrivals",
        "americas_arrivals", "oceania_arrivals", "other_markets_arrivals",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].tail(12).to_string(index=False))

    label_cols = [c for c in df.columns if c.endswith("_matched_label")]
    if label_cols:
        print("\nMatched row labels in last extracted row:")
        label_summary = df[["date", *label_cols]].tail(1).to_string(index=False)
        print(label_summary.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
