# Forecasting Vietnam International Tourist Arrivals

## Abstract

This repository contains a reproducible academic forecasting study of Vietnam's monthly international tourist arrivals. The empirical analysis uses monthly segment-level data from the Vietnam National Administration of Tourism (VNAT), with `international_arrivals` as the sole primary forecasting target. Regional segment variables are retained as supporting evidence for market composition, source-market structure, and heterogeneous recovery interpretation.

The central forecasting comparison evaluates Holt-Winters, SARIMA, SARIMA-GARCH, and XGBoost on the same target and the same train/test split. This design frames the repository as one coherent forecasting study rather than separate forecasting exercises for each segment.

## Research Objective

The objective is to assess how classical time-series models and a machine-learning benchmark forecast Vietnam's international tourism demand under strong seasonality, structural disruption, and post-COVID recovery uncertainty.

Main forecasting target:

- `international_arrivals`

Supporting segment variables:

- `asia_arrivals`
- `europe_arrivals`
- `americas_arrivals`
- `oceania_arrivals`
- `other_markets_arrivals`

Segment variables are used for EDA, market composition, segment-share analysis, heterogeneous recovery interpretation, and tourism source-market structure. They are not default forecasting targets in the main model-comparison pipeline. ASEAN arrivals are not used. Annual World Bank arrivals are not used as the primary modeling dataset.

## Data

Raw data:

```text
data/raw/vnat_monthly_segments.csv
```

Processed modeling data:

```text
data/processed/vnat_monthly_segments_clean.csv
```

The cleaned dataset covers January 2012 through December 2025 at monthly frequency.

Model evaluation split:

- Training sample: 2012-01 to 2022-12
- Test sample: 2023-01 to 2025-12

### Data Validation Summary

The raw VNAT file contains 155 rows. The cleaned dataset contains the complete 168-month panel. Validation identifies the following raw-data issues:

| Check | Result |
|---|---:|
| Missing months | 13 |
| Segment-sum consistency warnings | 29 |
| Suspicious repeated target vectors | 24 |
| Duplicated months | 0 |
| Non-positive values | 0 |

The preprocessing stage preserves the raw file, flags problematic observations, imputes affected values deterministically, and writes the cleaned monthly panel. Missing or invalid target values are filled by time-based interpolation with calendar-month median fallback.

Current clean-data quality flags:

| Quality flag | Rows |
|---|---:|
| `observed` | 109 |
| `segment_reconciliation` | 35 |
| `stale_repeated_vector` | 24 |

## Repository Structure

```text
forecast_tourist_arrivals/
├── data/
│   ├── raw/
│   │   └── vnat_monthly_segments.csv
│   └── processed/
│       └── vnat_monthly_segments_clean.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_decomposition_stationarity.ipynb
│   ├── 03_modeling_total_arrivals.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_segment_analysis.ipynb
│   └── 06_results_interpretation.ipynb
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── model_summary.md
│   └── report.md
├── src/
│   ├── config.py
│   ├── plotting.py
│   ├── crawl/
│   │   └── crawl_vnat_segments.py
│   ├── data/
│   │   ├── validate_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       ├── evaluation.py
│       ├── holt_winters.py
│       ├── sarima.py
│       ├── sarima_garch.py
│       └── xgboost_model.py
├── requirements.txt
└── README.md
```

## Environment

The project is written for Python and should be executed from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Playwright is required only when refreshing VNAT crawler output:

```powershell
python -m playwright install chromium
```

## Reproducible Workflow

### 1. Validate Raw Data

```powershell
python src\data\validate_data.py
```

Outputs:

```text
reports/tables/data_validation_checks.csv
reports/tables/data_validation_issues.csv
reports/tables/data_validation_report.md
```

### 2. Preprocess Data

```powershell
python src\data\preprocess.py
```

Outputs:

```text
data/processed/vnat_monthly_segments_clean.csv
reports/tables/preprocessing_flags.csv
```

### 3. Evaluate Main Forecasting Models

```powershell
python src\models\evaluation.py
```

Outputs:

```text
reports/tables/model_metrics.csv
reports/tables/best_models.csv
reports/tables/model_forecasts.csv
reports/tables/all_model_comparison.csv
reports/tables/best_model_summary.csv
```

### 4. Execute Central Model-Comparison Notebook

```powershell
jupyter notebook notebooks\04_model_comparison.ipynb
```

Notebook outputs:

```text
reports/tables/all_model_comparison.csv
reports/tables/best_model_summary.csv
reports/figures/model_comparison_forecast.png
reports/figures/model_comparison_metrics.png
reports/figures/xgboost_feature_importance.png
```

### 5. Additional Empirical Diagnostics

```powershell
python src\analysis\empirical_diagnostics.py
```

Outputs:

```text
reports/tables/structural_break_tests.csv
reports/tables/seasonal_significance_tests.csv
reports/tables/seasonal_stability_tests.csv
reports/tables/seasonal_strength_metrics.csv
reports/tables/sarima_estimation_results.csv
reports/tables/garch_estimation_results.csv
reports/tables/residual_diagnostics.csv
reports/tables/diebold_mariano_tests.csv
reports/tables/latex/
reports/figures/seasonal_stability_boxplot.png
reports/figures/structural_break_zivot_andrews.png
reports/figures/seasonal_strength_comparison.png
reports/figures/residual_diagnostics_updated.png
```

The Zivot-Andrews test provides formal evidence on stationarity under a possible structural break in log arrivals. The Friedman seasonal-significance test evaluates whether monthly seasonal effects are statistically meaningful in the pre-COVID period. The Kruskal-Wallis monthly stability tests compare pre-COVID and post-reopening seasonal distributions. STL strength metrics quantify the relative importance of seasonal and trend components. SARIMA and GARCH parameter tables support model interpretation and volatility persistence claims. Residual diagnostics assess autocorrelation, ARCH effects, normality, and first-order serial correlation. Diebold-Mariano tests evaluate whether forecast accuracy differences are statistically significant under absolute and squared error loss.

### 6. Optional Raw Data Refresh

```powershell
python src\crawl\crawl_vnat_segments.py --start 2012 --end 2025 --no-cache
```

Raw HTML and screenshot caches generated by the crawler are ignored by Git.

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Exploratory analysis of total arrivals, segment volumes, segment shares, yearly patterns, monthly seasonality, pre-COVID/COVID/post-reopening regimes, rolling volatility, growth rates, and segment correlations. |
| `02_decomposition_stationarity.ipynb` | Seasonal decomposition, ADF and KPSS tests, ACF/PACF analysis, log transformation comparison, and structural-break interpretation. |
| `03_modeling_total_arrivals.ipynb` | Primary time-series modeling notebook for total arrivals, including fitted values, forecasts, confidence intervals, residual diagnostics, and model interpretation. |
| `04_model_comparison.ipynb` | Central comparison of Holt-Winters, SARIMA, SARIMA-GARCH, and XGBoost on `international_arrivals`. |
| `05_segment_analysis.ipynb` | Supporting segment analysis for source-market composition and heterogeneous recovery interpretation. |
| `06_results_interpretation.ipynb` | Final academic interpretation of data, validation results, EDA findings, model performance, implications, limitations, and future extensions. |

## Methods

### Holt-Winters

Holt-Winters exponential smoothing is estimated on log arrivals to capture trend and seasonal demand patterns. Forecast intervals are approximated from residual variation in the transformed series.

### SARIMA

SARIMA/SARIMAX models are estimated on log arrivals. A compact AIC-based grid search selects non-seasonal and seasonal orders with monthly seasonality.

### SARIMA-GARCH

SARIMA-GARCH combines a SARIMA conditional mean model with a GARCH(1,1) model fitted to SARIMA residuals. GARCH is applied to residual volatility, not raw arrivals.

### XGBoost

XGBoost is used as a machine-learning benchmark for `international_arrivals` only. The benchmark uses lagged target values, rolling target statistics, calendar variables, and event indicators.

Allowed XGBoost features:

- `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`
- `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_12`
- `rolling_std_3`, `rolling_std_6`, `rolling_std_12`
- `month`, `quarter`, `year`, `time_index`
- `covid_period`
- `border_reopening_period`
- `recovery_period`
- `russia_ukraine_war`
- `china_us_tension`
- `iran_israel_conflict`

Same-month segment arrivals are excluded from the XGBoost feature set to avoid contemporaneous leakage.

## Evaluation

Forecast accuracy is evaluated on the 2023-2025 test period using:

- MAE
- RMSE
- MAPE
- sMAPE

The main model comparison evaluates only `international_arrivals`. Best models are selected by sMAPE, with RMSE used as a secondary ordering criterion.

## Current Empirical Results

Current main-target model comparison:

| Model | MAE | RMSE | MAPE | sMAPE |
|---|---:|---:|---:|---:|
| Holt-Winters | 7,283,652 | 11,303,800 | 411.07% | 99.49% |
| SARIMA | 679,227 | 775,568 | 42.12% | 55.90% |
| SARIMA-GARCH | 679,227 | 775,568 | 42.12% | 55.90% |
| XGBoost | 444,851 | 491,967 | 28.74% | 34.07% |

XGBoost is the best-performing benchmark in the current run. Segment variables remain analytically important for source-market structure and recovery interpretation, but they are not treated as separate primary forecasting targets.

## Reporting Standards

Figures and tables are designed for academic reporting. The plotting style uses:

- minimal axes and no gridlines;
- muted, consistent colors across figures;
- event annotations only where analytically useful;
- publication-style figure titles;
- reproducible saved outputs in `reports/figures/`.

The fixed color mapping is:

| Series | Color role |
|---|---|
| `international_arrivals` | charcoal |
| `asia_arrivals` | muted blue |
| `europe_arrivals` | muted green |
| `americas_arrivals` | muted orange |
| `oceania_arrivals` | muted purple |
| `other_markets_arrivals` | muted gray |

## Interpretation

Vietnam's international tourism demand is strongly seasonal, shock-sensitive, and structurally affected by the COVID-19 period. The border reopening phase introduces recovery volatility that reduces forecast stability. Segment-level patterns indicate heterogeneous recovery across source markets, supporting the use of segment data as interpretive market-structure evidence rather than as separate primary forecasting targets.

## Limitations

The analysis depends on crawler-derived VNAT segment data. Missing months, stale repeated pages, and segment reconciliation issues require deterministic interpolation before modeling. Forecast results should therefore be interpreted as conditional on the cleaned dataset rather than as direct estimates from a fully observed raw series.

Future extensions should refresh and manually verify the VNAT scrape, incorporate exogenous variables such as flight capacity, visa policy, exchange rates, and source-market macroeconomic indicators, and compare structural-break, intervention, or regime-switching models.
