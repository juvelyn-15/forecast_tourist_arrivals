"""Create project notebooks with reproducible code and interpretation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip().splitlines(True)}


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


SETUP = """
from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLEAN_SEGMENTS, FIGURES, TABLES, TARGET_COLUMNS, SEGMENT_COLUMNS, SERIES_COLORS, SERIES_LABELS
from src.plotting import annotate_events, save_figure, set_academic_style

set_academic_style()
df = pd.read_csv(CLEAN_SEGMENTS, parse_dates=["date"]).set_index("date").sort_index()
df.index.freq = "MS"
"""


def eda() -> list[dict]:
    return [
        md("# Exploratory Data Analysis\n\nThis notebook examines the cleaned VNAT monthly segment dataset. The narrative emphasizes seasonality, shock sensitivity, and market heterogeneity."),
        code(SETUP),
        md("## Total Arrivals\n\nObservation: total arrivals rise before the pandemic, collapse during COVID-19, and recover after reopening. Statistical implication: the mean process is structurally unstable. Tourism implication: forecasts should be interpreted with explicit shock uncertainty."),
        code("""
fig, ax = plt.subplots()
ax.plot(df.index, df["international_arrivals"], color=SERIES_COLORS["international_arrivals"])
annotate_events(ax, ["COVID-19 outbreak", "Vietnam border reopening", "Tourism recovery period"])
ax.set_title("Structural Shocks in International Tourist Arrivals")
ax.set_ylabel("Monthly arrivals")
save_figure(fig, FIGURES / "01_total_arrivals_timeseries.png")
plt.show()
"""),
        md("## Segment Arrivals\n\nObservation: Asia contributes the largest volume, while other regions have smaller but distinct trajectories. Statistical implication: aggregate arrivals mask segment-level heterogeneity. Tourism implication: recovery strategies should not assume uniform market behavior."),
        code("""
fig, ax = plt.subplots()
for col in SEGMENT_COLUMNS:
    ax.plot(df.index, df[col], label=SERIES_LABELS[col], color=SERIES_COLORS[col])
ax.set_title("Regional Composition of Vietnam International Arrivals")
ax.set_ylabel("Monthly arrivals")
ax.legend(ncol=3, fontsize=8)
save_figure(fig, FIGURES / "02_segment_arrivals.png")
plt.show()
"""),
        md("## Segment Shares\n\nObservation: source-market shares vary through time, especially during reopening. Statistical implication: composition shifts are relevant for forecast interpretation. Tourism implication: aviation and marketing resources should be aligned with changing source-region shares."),
        code("""
shares = df[SEGMENT_COLUMNS].div(df["international_arrivals"], axis=0) * 100
fig, ax = plt.subplots()
bottom = np.zeros(len(shares))
for col in SEGMENT_COLUMNS:
    values = shares[col].to_numpy()
    ax.fill_between(shares.index, bottom, bottom + values, color=SERIES_COLORS[col], alpha=0.75, label=SERIES_LABELS[col])
    bottom += values
ax.set_ylim(0, 100)
ax.set_title("Changing Market Shares by Source Region")
ax.set_ylabel("Share of total arrivals (%)")
ax.legend(ncol=3, fontsize=8)
save_figure(fig, FIGURES / "03_segment_share_over_time.png")
plt.show()
"""),
        md("## Monthly Seasonality\n\nObservation: arrivals follow a recurring monthly profile. Statistical implication: seasonal terms are necessary in forecasting models. Tourism implication: capacity planning should account for predictable intra-year peaks."),
        code("""
monthly = df.assign(month=df.index.month).groupby("month")["international_arrivals"]
med = monthly.median()
q1 = monthly.quantile(0.25)
q3 = monthly.quantile(0.75)
fig, ax = plt.subplots()
ax.plot(med.index, med.values, color=SERIES_COLORS["international_arrivals"])
ax.fill_between(med.index, q1.values, q3.values, color=SERIES_COLORS["international_arrivals"], alpha=0.18)
ax.set_xticks(range(1, 13))
ax.set_title("Seasonal Dynamics of International Tourist Arrivals")
ax.set_ylabel("Median arrivals")
save_figure(fig, FIGURES / "04_monthly_seasonality.png")
plt.show()
"""),
        md("## Yearly Trend\n\nObservation: the annual series separates the pre-pandemic expansion, pandemic trough, and recovery. Statistical implication: the trend component is interrupted by a structural break. Tourism implication: long-run planning should distinguish trend recovery from temporary rebound effects."),
        code("""
annual = df["international_arrivals"].resample("YS").sum()
fig, ax = plt.subplots()
ax.plot(annual.index.year, annual.values, marker="o", color=SERIES_COLORS["international_arrivals"])
ax.set_title("Annual Trend and Post-Pandemic Recovery Path")
ax.set_ylabel("Annual arrivals")
save_figure(fig, FIGURES / "05_yearly_trend.png")
plt.show()
"""),
        md("## Growth Rates and COVID Shock\n\nObservation: year-on-year growth is extremely volatile around the pandemic and reopening. Statistical implication: variance is time-varying after large shocks. Tourism implication: policy decisions should avoid treating rebound growth as stable baseline demand."),
        code("""
growth = df["international_arrivals"].pct_change(12) * 100
fig, ax = plt.subplots()
ax.axhline(0, color="#777777", linewidth=0.8)
ax.plot(growth.index, growth, color=SERIES_COLORS["international_arrivals"])
annotate_events(ax, ["COVID-19 outbreak", "Vietnam border reopening"])
ax.set_title("Year-on-Year Growth and Shock Transmission")
ax.set_ylabel("Year-on-year growth (%)")
save_figure(fig, FIGURES / "06_growth_rate_analysis.png")
plt.show()
"""),
        md("## Correlation Structure\n\nObservation: segment growth rates are positively correlated but not identical. Statistical implication: common shocks dominate, while segment-specific dynamics remain. Tourism implication: diversification across source markets may reduce exposure to region-specific demand shocks."),
        code("""
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
save_figure(fig, FIGURES / "07_correlation_heatmap.png")
plt.show()
"""),
    ]


def decomposition() -> list[dict]:
    return [
        md("# Decomposition and Stationarity\n\nThis notebook evaluates trend, seasonality, integration, and structural break behavior in total international arrivals."),
        code(SETUP + "\nfrom statsmodels.tsa.seasonal import seasonal_decompose\nfrom statsmodels.tsa.stattools import adfuller, kpss\nfrom statsmodels.graphics.tsaplots import plot_acf, plot_pacf"),
        md("## Seasonal Decomposition\n\nObservation: the log series contains a persistent trend and stable monthly pattern before the pandemic. Statistical implication: additive decomposition on logs is appropriate for separating proportional seasonal effects. Tourism implication: predictable seasonal pressure should be planned separately from crisis-driven demand changes."),
        code("""
series = np.log(df["international_arrivals"])
decomp = seasonal_decompose(series, model="additive", period=12, extrapolate_trend="freq")
fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
for ax, (name, values) in zip(axes, [("Observed", series), ("Trend", decomp.trend), ("Seasonal", decomp.seasonal), ("Residual", decomp.resid)]):
    ax.plot(values.index, values.values, color=SERIES_COLORS["international_arrivals"])
    ax.set_ylabel(name)
axes[0].set_title("Additive Decomposition of Log International Arrivals")
save_figure(fig, FIGURES / "08_decomposition.png")
plt.show()
"""),
        md("## Unit Root Tests\n\nObservation: stationarity tests distinguish log levels from log differences. Statistical implication: differencing is needed for stable mean dynamics. Tourism implication: forecasts should focus on growth dynamics rather than assuming fixed levels."),
        code("""
rows = []
for name, values in [("log level", series.dropna()), ("first log difference", series.diff().dropna())]:
    adf = adfuller(values, autolag="AIC")
    try:
        kpss_result = kpss(values, regression="c", nlags="auto")
        kpss_p = kpss_result[1]
    except Exception:
        kpss_p = np.nan
    rows.append({"series": name, "ADF p-value": adf[1], "KPSS p-value": kpss_p})
stationarity = pd.DataFrame(rows)
stationarity.to_csv(TABLES / "stationarity_tests.csv", index=False)
stationarity
"""),
        md("## ACF and PACF\n\nObservation: autocorrelation remains visible after first differencing, especially at seasonal lags. Statistical implication: SARIMA terms are justified. Tourism implication: recurring calendar effects improve forecast discipline for operational planning."),
        code("""
diff = series.diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
plot_acf(diff, lags=36, ax=axes[0], color=SERIES_COLORS["international_arrivals"], zero=False)
plot_pacf(diff, lags=36, ax=axes[1], color=SERIES_COLORS["international_arrivals"], zero=False, method="ywm")
axes[0].set_title("ACF of Log-Differenced Arrivals")
axes[1].set_title("PACF of Log-Differenced Arrivals")
for ax in axes:
    ax.grid(False)
save_figure(fig, FIGURES / "09_acf_pacf.png")
plt.show()
"""),
        md("## Log Transformation and Structural Break\n\nObservation: the log transformation stabilizes proportional variation but cannot remove the COVID break. Statistical implication: transformation handles scale, while model evaluation must still account for regime change. Tourism implication: post-pandemic uncertainty should remain central in forecast interpretation."),
        code("""
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(df.index, df["international_arrivals"], color=SERIES_COLORS["international_arrivals"])
axes[0].set_title("Level Series")
axes[1].plot(series.index, series, color=SERIES_COLORS["international_arrivals"])
axes[1].set_title("Log Series")
for ax in axes:
    annotate_events(ax, ["COVID-19 outbreak", "Vietnam border reopening"])
save_figure(fig, FIGURES / "11_log_transformation_comparison.png")
plt.show()
"""),
    ]


def modeling_total() -> list[dict]:
    return [
        md("# Modeling Total International Arrivals\n\nThis notebook compares Holt-Winters, SARIMA, and SARIMA-GARCH for the primary target: total international arrivals."),
        code(SETUP + "\nfrom statsmodels.graphics.tsaplots import plot_acf\nfrom scipy import stats\nfrom src.models.evaluation import evaluate_target, split_series"),
        md("## Train/Test Split\n\nObservation: the test period covers the recovery phase from 2023 to 2025. Statistical implication: out-of-sample evaluation is intentionally difficult because the structure differs from the pre-2023 training sample. Tourism implication: forecast errors measure recovery uncertainty, not only model weakness."),
        code("""
target = "international_arrivals"
train, test = split_series(df[target])
train.index.min(), train.index.max(), test.index.min(), test.index.max(), len(train), len(test)
"""),
        md("## Model Estimation and Forecasts\n\nObservation: each method extrapolates seasonality differently. Statistical implication: accuracy comparisons reveal whether smoothing, SARIMA dependence, or residual volatility modeling is more useful. Tourism implication: the preferred model should support planning under post-COVID uncertainty."),
        code("""
rows, results, forecasts = evaluate_target(df[target], target)
metrics = pd.DataFrame(rows)
metrics.to_csv(TABLES / "model_metrics_total.csv", index=False)
metrics
"""),
        code("""
fig, ax = plt.subplots()
ax.plot(df.loc["2018":].index, df.loc["2018":, target], color=SERIES_COLORS[target], label="Observed")
for model in forecasts["model"].unique():
    sub = forecasts[forecasts["model"] == model].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    ax.plot(sub["date"], sub["forecast"], label=model, alpha=0.85)
ax.set_title("Out-of-Sample Forecasts for Total Arrivals")
ax.set_ylabel("Monthly arrivals")
ax.legend(fontsize=8)
save_figure(fig, FIGURES / "12_total_model_comparison.png")
plt.show()
"""),
        md("## Confidence Intervals and Residual Diagnostics\n\nObservation: forecast intervals widen under residual uncertainty. Statistical implication: residual autocorrelation and volatility clustering indicate remaining structure. Tourism implication: decision makers should use ranges for staffing and capacity decisions."),
        code("""
best_model = metrics[metrics["status"].eq("ok")].sort_values("sMAPE").iloc[0]["model"]
best = results[best_model]
best_fc = forecasts[forecasts["model"].eq(best_model)].copy()
best_fc["date"] = pd.to_datetime(best_fc["date"])
fig, ax = plt.subplots()
ax.plot(df.loc["2018":].index, df.loc["2018":, target], color=SERIES_COLORS[target], label="Observed")
ax.plot(best_fc["date"], best_fc["forecast"], color="#8f6aa8", label=f"{best_model} forecast")
ax.fill_between(best_fc["date"], best_fc["lower"], best_fc["upper"], color="#8f6aa8", alpha=0.18)
ax.set_title("Best-Model Forecast Interval for Total Arrivals")
ax.set_ylabel("Monthly arrivals")
ax.legend()
save_figure(fig, FIGURES / "10_total_forecast.png")
plt.show()
"""),
        code("""
resid = best["residuals"].dropna()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].plot(resid.index, resid, color=SERIES_COLORS[target])
axes[0].axhline(0, color="#777777", linewidth=0.8)
axes[0].set_title("Residuals")
plot_acf(resid, lags=30, ax=axes[1], zero=False, color=SERIES_COLORS[target])
axes[1].set_title("Residual ACF")
stats.probplot(resid, dist="norm", plot=axes[2])
axes[2].set_title("Normal Q-Q")
for ax in axes:
    ax.grid(False)
save_figure(fig, FIGURES / "13_total_residual_diagnostics.png")
plt.show()
"""),
        md("## Interpretation\n\nObservation: the best model minimizes test-period forecast error under recovery volatility. Statistical implication: seasonality is forecastable, but residual shocks remain material. Tourism implication: forecasts should be updated frequently as new VNAT observations arrive."),
    ]


def modeling_segments() -> list[dict]:
    return [
        md("# Modeling Regional Segments\n\nSegment models are supporting evidence. The emphasis is heterogeneity rather than unnecessary model complexity."),
        code(SETUP + "\nfrom src.models.evaluation import run_all"),
        md("## Segment Forecast Evaluation\n\nObservation: regions differ in recovery level and volatility. Statistical implication: a model that performs well for total arrivals may not dominate every segment. Tourism implication: source-market planning should use segment-specific forecast risk."),
        code("""
metrics, best, forecasts = run_all(df, targets=SEGMENT_COLUMNS)
metrics.to_csv(TABLES / "model_metrics_segments.csv", index=False)
best.to_csv(TABLES / "best_models_segments.csv", index=False)
best
"""),
        code("""
fig, axes = plt.subplots(len(SEGMENT_COLUMNS), 1, figsize=(9, 11), sharex=True)
for ax, col in zip(axes, SEGMENT_COLUMNS):
    ax.plot(df.loc["2019":].index, df.loc["2019":, col], color=SERIES_COLORS[col], label="Observed")
    row = best[best["target"].eq(col)].iloc[0]
    sub = forecasts[(forecasts["target"].eq(col)) & (forecasts["model"].eq(row["model"]))].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    ax.plot(sub["date"], sub["forecast"], color="#2f3437", alpha=0.85, label=row["model"])
    ax.set_title(f"{SERIES_LABELS[col]}: Best Out-of-Sample Forecast")
    ax.set_ylabel("Arrivals")
axes[0].legend(fontsize=8)
save_figure(fig, FIGURES / "14_segment_forecasts.png")
plt.show()
"""),
        md("## Interpretation\n\nObservation: Asia dominates the total path, while smaller regions show sharper proportional swings. Statistical implication: segment residual variance is heterogeneous. Tourism implication: market-specific policies are necessary for resilient recovery planning."),
    ]


def results() -> list[dict]:
    return [
        md("# Results Interpretation\n\nThis notebook synthesizes the project in an academic forecasting-paper style."),
        code(SETUP + "\nfrom src.models.evaluation import run_all\nvalidation = pd.read_csv(TABLES / 'data_validation_checks.csv')\nmetrics = pd.read_csv(TABLES / 'model_metrics.csv') if (TABLES / 'model_metrics.csv').exists() else run_all(df)[0]\nbest = pd.read_csv(TABLES / 'best_models.csv') if (TABLES / 'best_models.csv').exists() else run_all(df)[1]"),
        md("## Research Objective\n\nThe objective is to forecast Vietnam's monthly international tourist arrivals using VNAT segment data, with attention to seasonality, volatility, and structural breaks."),
        code("validation"),
        md("## Data Source and Variables\n\nThe dataset is `data/raw/vnat_monthly_segments.csv`, cleaned to `data/processed/vnat_monthly_segments_clean.csv`. Variables include total arrivals and five regional segments: Asia, Europe, Americas, Oceania, and other markets. ASEAN is not used."),
        md("## EDA Findings\n\nObservation: arrivals are seasonal, shock-sensitive, and compositionally uneven. Statistical implication: models need seasonal structure and robust post-shock evaluation. Tourism implication: planning should combine aggregate and regional evidence."),
        code("""
fig, ax = plt.subplots()
ax.plot(df.index, df["international_arrivals"], color=SERIES_COLORS["international_arrivals"])
annotate_events(ax, ["COVID-19 outbreak", "Vietnam border reopening", "Russia-Ukraine war", "Iran-Israel conflict"])
ax.set_title("Vietnam Tourism Demand Under Global and Domestic Shocks")
ax.set_ylabel("Monthly arrivals")
save_figure(fig, FIGURES / "15_final_narrative_series.png")
plt.show()
"""),
        md("## Model Performance\n\nObservation: out-of-sample accuracy differs across model classes and segments. Statistical implication: no single specification should be assumed dominant without test-period evidence. Tourism implication: model choice should be revised when recovery dynamics change."),
        code("metrics.round(2)"),
        code("best[['target', 'model', 'MAE', 'RMSE', 'MAPE', 'sMAPE']].round(2)"),
        md("## Forecast Interpretation\n\nObservation: the recovery path remains volatile despite the return of strong arrival volumes. Statistical implication: confidence intervals and residual diagnostics are central outputs. Tourism implication: policymakers and firms should treat forecasts as risk ranges for capacity, promotion, and infrastructure decisions."),
        md("## Limitations and Future Improvements\n\nThe analysis depends on crawler-derived VNAT data and deterministic imputation for missing or stale observations. Future work should refresh raw VNAT pages, incorporate exogenous indicators such as flight capacity and exchange rates, and test structural-break or regime-switching models."),
    ]


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOKS / "01_eda.ipynb", eda())
    write_notebook(NOTEBOOKS / "02_decomposition_stationarity.ipynb", decomposition())
    write_notebook(NOTEBOOKS / "03_modeling_total_arrivals.ipynb", modeling_total())
    write_notebook(NOTEBOOKS / "04_modeling_segments.ipynb", modeling_segments())
    write_notebook(NOTEBOOKS / "05_results_interpretation.ipynb", results())
    print(f"Notebooks written to {NOTEBOOKS}")


if __name__ == "__main__":
    main()
