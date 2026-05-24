"""
eda.py
------
Exploratory Data Analysis for Vietnam tourism arrivals.

Covers:
- Time series plot with COVID annotation
- Seasonal decomposition (additive + multiplicative)
- ACF and PACF plots
- Unit root tests: ADF, PP, KPSS
- Summary statistics

Usage:
    python src/models/eda.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import pacf as compute_pacf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import load_monthly_data, get_series, train_test_split_ts

FIG_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── Plot Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
BLUE   = "#1a5fa8"
RED    = "#c0392b"
GREEN  = "#27ae60"
ORANGE = "#e67e22"


def plot_raw_series(series: pd.Series):
    """Full series plot with COVID shading and key event annotations."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series.values / 1e6, color=BLUE, lw=1.8, label="Monthly arrivals")

    # COVID shading
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-03-01"),
               alpha=0.15, color=RED, label="COVID-19 disruption")

    # SARS annotation
    ax.annotate("SARS\n(2003)", xy=(pd.Timestamp("2003-06-01"), 0.05),
                fontsize=8, ha="center", color="grey")

    # Record high annotation
    peak_idx = series.idxmax()
    ax.annotate(f"Record\n{series[peak_idx]/1e6:.1f}M",
                xy=(peak_idx, series[peak_idx] / 1e6),
                xytext=(0, 20), textcoords="offset points",
                fontsize=8, ha="center", arrowprops=dict(arrowstyle="-|>", color="black"))

    ax.set_title("International Tourist Arrivals to Vietnam (2000–2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Arrivals (millions)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_raw_series.png"))
    plt.close()
    print("✔ Saved: 01_raw_series.png")


def plot_decomposition(series: pd.Series, model: str = "multiplicative"):
    """STL decomposition plot."""
    # Use pre-COVID series for clean decomposition
    pre_covid = series[:"2020-01"]
    stl = STL(pre_covid, period=12, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        (pre_covid, "Observed", BLUE),
        (result.trend, "Trend", ORANGE),
        (result.seasonal, "Seasonal", GREEN),
        (result.resid, "Residual", RED),
    ]
    for ax, (comp, label, color) in zip(axes, components):
        ax.plot(comp, color=color, lw=1.5)
        ax.set_ylabel(label, fontsize=10)
        if label == "Seasonal":
            ax.axhline(0, color="black", lw=0.5, ls="--")

    axes[0].set_title("STL Decomposition — Vietnam International Arrivals (2000–2020)", 
                       fontsize=13, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_stl_decomposition.png"))
    plt.close()
    print("✔ Saved: 02_stl_decomposition.png")


def plot_seasonal_pattern(series: pd.Series):
    """Box plots by month to visualize seasonal pattern."""
    pre_covid = series[:"2019-12"].copy()
    df = pd.DataFrame({
        "arrivals": pre_covid.values / 1e6,
        "month": pre_covid.index.month
    })
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 5))
    df.boxplot(column="arrivals", by="month", ax=ax,
               boxprops=dict(color=BLUE), medianprops=dict(color=RED, lw=2))
    ax.set_xticklabels(month_names)
    ax.set_title("Monthly Seasonal Pattern (2000–2019)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Arrivals (millions)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_seasonal_boxplot.png"))
    plt.close()
    print("✔ Saved: 03_seasonal_boxplot.png")


def plot_acf_pacf(series: pd.Series, lags: int = 48, title_suffix: str = ""):
    """ACF and PACF plots side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=ax1, color=BLUE,
             title=f"ACF{' — ' + title_suffix if title_suffix else ''}")
    plot_pacf(series.dropna(), lags=lags, ax=ax2, color=BLUE, method="yw",
              title=f"PACF{' — ' + title_suffix if title_suffix else ''}")
    plt.tight_layout()
    fname = f"04_acf_pacf{'_' + title_suffix.lower().replace(' ', '_') if title_suffix else ''}.png"
    plt.savefig(os.path.join(FIG_DIR, fname))
    plt.close()
    print(f"✔ Saved: {fname}")


def run_unit_root_tests(series: pd.Series, name: str = "Series") -> pd.DataFrame:
    """
    Run ADF, KPSS unit root tests.
    Also tests on first-differenced and seasonally-differenced series.
    """
    results = []

    def run_adf(s, label):
        stat, pval, _, nobs, crit, _ = adfuller(s.dropna(), autolag="AIC")
        results.append({
            "Series": label,
            "Test": "ADF",
            "H0": "Unit root (non-stationary)",
            "Test Stat": round(stat, 4),
            "p-value": round(pval, 4),
            "1% CV": round(crit["1%"], 4),
            "5% CV": round(crit["5%"], 4),
            "Decision (5%)": "Reject H0 → Stationary" if pval < 0.05 else "Fail to Reject → Non-stationary",
        })

    def run_kpss(s, label):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, _, crit = kpss(s.dropna(), regression="ct", nlags="auto")
        results.append({
            "Series": label,
            "Test": "KPSS",
            "H0": "Stationary",
            "Test Stat": round(stat, 4),
            "p-value": round(pval, 4),
            "1% CV": round(crit["1%"], 4),
            "5% CV": round(crit["5%"], 4),
            "Decision (5%)": "Reject H0 → Non-stationary" if pval < 0.05 else "Fail to Reject → Stationary",
        })

    # Level
    run_adf(series, f"{name} (level)")
    run_kpss(series, f"{name} (level)")

    # First difference
    d1 = series.diff().dropna()
    run_adf(d1, f"Δ{name}")
    run_kpss(d1, f"Δ{name}")

    # Seasonal difference
    ds = series.diff(12).dropna()
    run_adf(ds, f"Δ₁₂{name}")
    run_kpss(ds, f"Δ₁₂{name}")

    # First + seasonal difference
    d1ds = series.diff(12).diff().dropna()
    run_adf(d1ds, f"Δ·Δ₁₂{name}")
    run_kpss(d1ds, f"Δ·Δ₁₂{name}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(TAB_DIR, "unit_root_tests.csv"), index=False)
    return df


def summary_statistics(series: pd.Series) -> pd.DataFrame:
    """Descriptive statistics by period."""
    periods = {
        "Full sample (2000–2024)": series,
        "Pre-COVID (2000–2019)":   series[:"2019-12"],
        "COVID (2020–2022)":       series["2020-01":"2022-02"],
        "Recovery (2022–2024)":    series["2022-03":],
    }
    rows = []
    for label, s in periods.items():
        rows.append({
            "Period": label,
            "N (months)": len(s),
            "Mean": f"{s.mean():,.0f}",
            "Std Dev": f"{s.std():,.0f}",
            "Min": f"{s.min():,.0f}",
            "Max": f"{s.max():,.0f}",
            "CV (%)": f"{s.std() / s.mean() * 100:.1f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB_DIR, "summary_statistics.csv"), index=False)
    return df


def main():
    print("=" * 60)
    print("  EDA — Vietnam International Tourist Arrivals")
    print("=" * 60)

    df = load_monthly_data()
    y  = get_series(df)

    print(f"\nData loaded: {y.index[0].date()} → {y.index[-1].date()} ({len(y)} months)\n")

    # 1. Raw series plot
    plot_raw_series(y)

    # 2. STL decomposition
    plot_decomposition(y)

    # 3. Seasonal pattern
    plot_seasonal_pattern(y)

    # 4. ACF/PACF on level series
    pre_covid = y[:"2019-12"]
    plot_acf_pacf(pre_covid, title_suffix="Level (pre-COVID)")

    # 5. ACF/PACF on differenced series
    d1_ds = pre_covid.diff(12).diff().dropna()
    plot_acf_pacf(d1_ds, title_suffix="Δ·Δ₁₂ (pre-COVID)")

    # 6. Unit root tests
    print("\n--- Unit Root Tests (pre-COVID series) ---")
    ur = run_unit_root_tests(pre_covid, "log_arrivals")
    print(ur[["Series", "Test", "Test Stat", "p-value", "Decision (5%)"]].to_string(index=False))
    print(f"\nSaved → {TAB_DIR}/unit_root_tests.csv")

    # 7. Summary statistics
    print("\n--- Summary Statistics ---")
    ss = summary_statistics(y)
    print(ss.to_string(index=False))
    print(f"\nSaved → {TAB_DIR}/summary_statistics.csv")

    print("\n✔ EDA complete. All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
