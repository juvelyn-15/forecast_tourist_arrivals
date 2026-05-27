"""Generate project figures, model tables, and summary report.

Run from the repository root:
    python src/reporting/generate_outputs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import CLEAN_SEGMENTS, FIGURES, SEGMENT_COLUMNS, SERIES_COLORS, SERIES_LABELS, TABLES, TARGET_COLUMNS
from src.data.preprocess import clean_segments
from src.data.validate_data import load_raw, save_validation_outputs, validate
from src.models.evaluation import load_clean, run_all
from src.plotting import annotate_events, save_figure, set_academic_style


def _format_axis(ax, ylabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="both", length=3, colors="#444444")


def ensure_clean_data() -> pd.DataFrame:
    raw = load_raw()
    validation = validate(raw)
    save_validation_outputs(validation)
    clean = clean_segments(raw)
    CLEAN_SEGMENTS.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_SEGMENTS, index=False)
    return load_clean()


def figure_total_series(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(df.index, df["international_arrivals"], color=SERIES_COLORS["international_arrivals"])
    annotate_events(ax, ["COVID-19 outbreak", "Vietnam border reopening", "Tourism recovery period"])
    ax.set_title("Structural Shocks in International Tourist Arrivals")
    _format_axis(ax, "Monthly arrivals")
    save_figure(fig, FIGURES / "01_total_arrivals_timeseries.png")
    plt.close(fig)


def figure_segments(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    for col in SEGMENT_COLUMNS:
        ax.plot(df.index, df[col], label=SERIES_LABELS[col], color=SERIES_COLORS[col], alpha=0.9)
    ax.set_title("Regional Composition of Vietnam International Arrivals")
    _format_axis(ax, "Monthly arrivals")
    ax.legend(ncol=3, fontsize=8)
    save_figure(fig, FIGURES / "02_segment_arrivals.png")
    plt.close(fig)


def figure_shares(df: pd.DataFrame) -> None:
    shares = df[SEGMENT_COLUMNS].div(df["international_arrivals"], axis=0) * 100
    fig, ax = plt.subplots()
    bottom = np.zeros(len(shares))
    for col in SEGMENT_COLUMNS:
        values = shares[col].to_numpy()
        ax.fill_between(shares.index, bottom, bottom + values, color=SERIES_COLORS[col], alpha=0.75, label=SERIES_LABELS[col])
        bottom += values
    ax.set_ylim(0, 100)
    ax.set_title("Changing Market Shares by Source Region")
    _format_axis(ax, "Share of total arrivals (%)")
    ax.legend(ncol=3, fontsize=8)
    save_figure(fig, FIGURES / "03_segment_share_over_time.png")
    plt.close(fig)


def figure_seasonality(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    monthly = df.assign(month=df.index.month).groupby("month")["international_arrivals"]
    med = monthly.median()
    q1 = monthly.quantile(0.25)
    q3 = monthly.quantile(0.75)
    ax.plot(med.index, med.values, color=SERIES_COLORS["international_arrivals"])
    ax.fill_between(med.index, q1.values, q3.values, color=SERIES_COLORS["international_arrivals"], alpha=0.18)
    ax.set_xticks(range(1, 13))
    ax.set_title("Seasonal Dynamics of International Tourist Arrivals")
    _format_axis(ax, "Median arrivals")
    save_figure(fig, FIGURES / "04_monthly_seasonality.png")
    plt.close(fig)


def figure_yearly_trend(df: pd.DataFrame) -> None:
    annual = df["international_arrivals"].resample("YS").sum()
    fig, ax = plt.subplots()
    ax.plot(annual.index.year, annual.values, color=SERIES_COLORS["international_arrivals"], marker="o")
    ax.set_title("Annual Trend and Post-Pandemic Recovery Path")
    _format_axis(ax, "Annual arrivals")
    save_figure(fig, FIGURES / "05_yearly_trend.png")
    plt.close(fig)


def figure_growth(df: pd.DataFrame) -> None:
    growth = df["international_arrivals"].pct_change(12) * 100
    fig, ax = plt.subplots()
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.plot(growth.index, growth, color=SERIES_COLORS["international_arrivals"])
    annotate_events(ax, ["COVID-19 outbreak", "Vietnam border reopening"])
    ax.set_title("Year-on-Year Growth and Shock Transmission")
    _format_axis(ax, "Year-on-year growth (%)")
    save_figure(fig, FIGURES / "06_growth_rate_analysis.png")
    plt.close(fig)


def figure_event_context(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(df.index, df["international_arrivals"], color=SERIES_COLORS["international_arrivals"])
    annotate_events(
        ax,
        [
            "China-US economic tensions",
            "COVID-19 outbreak",
            "Russia-Ukraine war",
            "Vietnam border reopening",
            "Iran-Israel conflict",
        ],
    )
    ax.set_title("Vietnam Tourism Demand Under Global and Domestic Shocks")
    _format_axis(ax, "Monthly arrivals")
    save_figure(fig, FIGURES / "07_event_context.png")
    plt.close(fig)


def figure_correlation(df: pd.DataFrame) -> None:
    corr = df[TARGET_COLUMNS].pct_change().corr()
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(corr, cmap="Greys", vmin=-1, vmax=1)
    labels = [SERIES_LABELS[c] for c in TARGET_COLUMNS]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Structure of Monthly Arrival Growth")
    save_figure(fig, FIGURES / "08_correlation_heatmap.png")
    plt.close(fig)


def figure_decomposition(df: pd.DataFrame) -> None:
    series = np.log(df["international_arrivals"])
    decomp = seasonal_decompose(series, model="additive", period=12, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    parts = [("Observed", series), ("Trend", decomp.trend), ("Seasonal", decomp.seasonal), ("Residual", decomp.resid)]
    for ax, (name, values) in zip(axes, parts):
        ax.plot(values.index, values.values, color=SERIES_COLORS["international_arrivals"])
        ax.set_ylabel(name)
    axes[0].set_title("Additive Decomposition of Log International Arrivals")
    save_figure(fig, FIGURES / "09_decomposition.png")
    plt.close(fig)


def figure_acf_pacf(df: pd.DataFrame) -> None:
    series = np.log(df["international_arrivals"]).diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_acf(series, lags=36, ax=axes[0], color=SERIES_COLORS["international_arrivals"], zero=False)
    plot_pacf(series, lags=36, ax=axes[1], color=SERIES_COLORS["international_arrivals"], zero=False, method="ywm")
    axes[0].set_title("ACF of Log-Differenced Arrivals")
    axes[1].set_title("PACF of Log-Differenced Arrivals")
    for ax in axes:
        ax.grid(False)
    save_figure(fig, FIGURES / "10_acf_pacf.png")
    plt.close(fig)


def figure_model_forecast(df: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    if forecasts.empty:
        return
    sub = forecasts[(forecasts["target"] == "international_arrivals") & (forecasts["model"] == "Holt-Winters")].copy()
    if sub.empty:
        sub = forecasts[forecasts["target"] == "international_arrivals"].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    fig, ax = plt.subplots()
    ax.plot(df.loc["2018":].index, df.loc["2018":, "international_arrivals"], color=SERIES_COLORS["international_arrivals"], label="Observed")
    ax.plot(sub["date"], sub["forecast"], color="#8f6aa8", label="Forecast")
    ax.fill_between(sub["date"], sub["lower"], sub["upper"], color="#8f6aa8", alpha=0.18)
    ax.set_title("Out-of-Sample Forecast of Total International Arrivals")
    _format_axis(ax, "Monthly arrivals")
    ax.legend()
    save_figure(fig, FIGURES / "11_total_forecast.png")
    plt.close(fig)


def stationarity_table(df: pd.DataFrame) -> pd.DataFrame:
    series = np.log(df["international_arrivals"]).dropna()
    diff = series.diff().dropna()
    rows = []
    for name, values in [("log level", series), ("first log difference", diff)]:
        adf = adfuller(values, autolag="AIC")
        try:
            kpss_stat = kpss(values, regression="c", nlags="auto")
            kpss_p = kpss_stat[1]
        except Exception:
            kpss_p = np.nan
        rows.append({"series": name, "ADF p-value": adf[1], "KPSS p-value": kpss_p})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stationarity_tests.csv", index=False)
    return out


def write_summary_report(df: pd.DataFrame, metrics_df: pd.DataFrame, best_df: pd.DataFrame, stationarity: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    validation_counts = pd.read_csv(TABLES / "data_validation_issues.csv")["issue_type"].value_counts()
    annual_peak = df["international_arrivals"].resample("YS").sum().idxmax().year
    annual_low = df["international_arrivals"].resample("YS").sum().idxmin().year
    total_best = best_df.loc[best_df["target"] == "international_arrivals"].iloc[0]

    report = f"""# Model Summary

## Research Objective

This project forecasts Vietnam's monthly international tourist arrivals using official VNAT segment data. The empirical focus is forecasting accuracy, seasonal structure, volatility, and structural uncertainty after COVID-19.

## Data and Validation

The modeling dataset is `data/processed/vnat_monthly_segments_clean.csv`, derived from `data/raw/vnat_monthly_segments.csv`. The sample contains monthly observations from January 2012 to December 2025 for total arrivals and five regional segments. The validation layer detected {int(validation_counts.sum())} raw-data issues, mainly missing months, segment-sum inconsistencies, and stale repeated crawler values. These observations were flagged and imputed deterministically for modeling.

## Exploratory Findings

Observed arrivals show strong monthly seasonality, a sharp pandemic collapse, and a volatile recovery after the March 2022 border reopening. Annual arrivals reach their sample minimum in {annual_low} and their maximum in {annual_peak}. Asia dominates the regional composition, while Europe, the Americas, Oceania, and other markets display different recovery speeds.

## Stationarity

ADF and KPSS tests indicate that log levels are structurally non-stationary, while first log differences are more stable. This supports the use of differenced seasonal models and motivates residual diagnostics for volatility clustering.

{stationarity.round(4).to_markdown(index=False)}

## Forecast Evaluation

The primary target is total international arrivals. On the 2023-2025 test period, the best total-arrivals model by sMAPE is **{total_best['model']}**, with MAE = {total_best['MAE']:.0f}, RMSE = {total_best['RMSE']:.0f}, MAPE = {total_best['MAPE']:.2f}%, and sMAPE = {total_best['sMAPE']:.2f}%.

{best_df[['target', 'model', 'MAE', 'RMSE', 'MAPE', 'sMAPE']].round(2).to_markdown(index=False)}

## Interpretation and Implications

Observation: Vietnam tourism demand is seasonal and shock-sensitive. Statistical implication: models must account for recurring monthly patterns and unstable post-shock residual variance. Tourism implication: capacity planning should use prediction intervals, not point forecasts alone.

Observation: source-region segments recover unevenly. Statistical implication: aggregate forecasts conceal heterogeneous dynamics. Tourism implication: market-specific promotion and aviation-capacity planning should prioritize segment-level recovery evidence.

## Limitations and Future Work

The cleaned dataset relies on deterministic imputation where raw crawler output is missing or stale. Future work should refresh the VNAT scrape, add exogenous predictors such as flight capacity and exchange rates, and compare structural-break or regime-switching models.
"""
    (ROOT / "reports" / "model_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    set_academic_style()
    df = ensure_clean_data()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    figure_total_series(df)
    figure_segments(df)
    figure_shares(df)
    figure_seasonality(df)
    figure_yearly_trend(df)
    figure_growth(df)
    figure_event_context(df)
    figure_correlation(df)
    figure_decomposition(df)
    figure_acf_pacf(df)
    stationarity = stationarity_table(df)

    metrics_df, best_df, forecasts_df = run_all(df)
    metrics_df.to_csv(TABLES / "model_metrics.csv", index=False)
    best_df.to_csv(TABLES / "best_models.csv", index=False)
    forecasts_df.to_csv(TABLES / "model_forecasts.csv", index=False)
    figure_model_forecast(df, forecasts_df)
    write_summary_report(df, metrics_df, best_df, stationarity)
    print(f"Figures saved to {FIGURES}")
    print(f"Tables saved to {TABLES}")
    print("Summary report saved to reports/model_summary.md")


if __name__ == "__main__":
    main()
