# 🏖️ Forecasting International Tourist Arrivals to Vietnam

**Course:** Time Series Analysis and Forecast in Economics and Finance  
**Topic:** Forecasting international tourist arrivals to Vietnam  
**Student:** Duong Thi Huyen Trang — 11230593  
**University:** National Economics University (NEU)

---

## 📋 Project Overview

This project builds a complete forecasting pipeline for Vietnam's monthly international tourist arrivals, applying the full spectrum of time series methods taught in the course: exploratory decomposition, Holt-Winters smoothing, Box-Jenkins SARIMA, and GARCH volatility modeling.

### Economic Motivation

Tourism is a critical pillar of Vietnam's economy — contributing roughly 10% of GDP in pre-pandemic years. Accurate forecasts are essential for:
- **Government planning:** Ministry of Culture, Sports and Tourism allocates infrastructure and marketing budgets.
- **Hospitality industry:** Hotels, airlines, and tour operators rely on arrival projections for capacity planning.
- **Macroeconomic analysis:** Tourism earnings are a major source of foreign exchange.

The series is analytically rich: it exhibits strong **monthly seasonality** (Tet holidays, summer peaks), a clear **upward trend** (2010–2019), a catastrophic **structural break** during COVID-19 (2020–2021), and a volatile **recovery phase** (2022–2024) — making it ideal for demonstrating SARIMA and GARCH.

---

## 📁 Repository Structure

```
vietnam_tourism_forecast/
│
├── data/
│   ├── raw/                  # Raw scraped / downloaded data
│   └── processed/            # Cleaned, formatted time series
│
├── src/
│   ├── crawl/
│   │   ├── crawl_vnat.py         # Scrape VNAT official statistics
│   │   ├── crawl_worldbank.py    # World Bank API (backup source)
│   │   └── merge_sources.py      # Merge & reconcile data sources
│   ├── models/
│   │   ├── eda.py                # Decomposition, ACF/PACF, unit root tests
│   │   ├── holt_winters.py       # Holt-Winters (additive & multiplicative)
│   │   ├── sarima.py             # Auto ARIMA + manual SARIMA grid search
│   │   ├── garch.py              # GARCH on SARIMA residuals
│   │   └── var_model.py          # Optional VAR with macro covariates
│   └── utils/
│       ├── preprocessing.py      # Cleaning, log transform, outlier handling
│       └── evaluation.py         # MAE, RMSE, MAPE, Diebold-Mariano test
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_stationarity.ipynb
│   ├── 03_holt_winters.ipynb
│   ├── 04_sarima.ipynb
│   ├── 05_garch.ipynb
│   └── 06_comparison_forecast.ipynb
│
├── outputs/
│   ├── figures/              # All plots
│   ├── tables/               # Model diagnostics, test results
│   └── forecasts/            # Forecast CSV files
│
├── reports/
│   └── final_report.md       # Written analysis & interpretation
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Crawl & collect data
python src/crawl/crawl_worldbank.py     # Reliable API-based source
python src/crawl/crawl_vnat.py          # VNAT official stats (requires manual step)
python src/crawl/merge_sources.py       # Merge into master dataset

# 3. Run EDA
python src/models/eda.py

# 4. Run models in order
python src/models/holt_winters.py
python src/models/sarima.py
python src/models/garch.py

# Or use the Jupyter notebooks for interactive analysis
jupyter notebook notebooks/
```

---

## 📊 Models Applied

| Model  | Purpose |
|-------|---------|
| Decomposition + Holt-Winters | Baseline: trend + seasonality smoothing |
| ADF / PP / KPSS Tests | Determine integration order `d`, `D` |
| SARIMA(p,d,q)(P,D,Q)₁₂ | Primary forecasting model |
| ARIMA on log-differenced | Handles unit root; log-returns interpretation |
| GARCH(1,1) on residuals | Models volatility clustering in recovery |
| VAR (extension) | Multi-equation model with USD/VND, oil price |

---

## 📈 Expected Key Findings

1. The series is **I(1)** with seasonal integration — requiring one regular and one seasonal difference.
2. **SARIMA(1,1,1)(1,1,1)₁₂** or similar is expected to outperform Holt-Winters on the test set.
3. **GARCH effects** are significant in the post-COVID recovery residuals, indicating excess volatility not captured by SARIMA alone.
4. **Forecast horizon:** 12–24 months ahead (2025–2026), with 95% confidence intervals.

---

## 📚 Data Sources

- **VNAT (Vietnam National Administration of Tourism):** https://vietnamtourism.gov.vn/
- **World Bank:** `ST.INT.ARVL` indicator via `wbgapi`
- **UNWTO:** Supplementary validation
