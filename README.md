# Forecasting Vietnam International Tourist Arrivals

Academic-style time-series forecasting project for Vietnam's monthly international tourist arrivals using VNAT regional segment data.

The project validates raw crawler output, builds a complete monthly modeling dataset, explores seasonality and structural breaks, estimates Holt-Winters, SARIMA, and SARIMA-GARCH models, evaluates 2023-2025 test forecasts, and writes figures, tables, notebooks, and a summary report.

## Research Scope

Primary target:

- `international_arrivals`

Supporting segment targets:

- `asia_arrivals`
- `europe_arrivals`
- `americas_arrivals`
- `oceania_arrivals`
- `other_markets_arrivals`

ASEAN is intentionally excluded. Annual World Bank arrivals are not used as the main modeling dataset.

## Data

Raw dataset:

```text
data/raw/vnat_monthly_segments.csv
```

Clean modeling dataset:

```text
data/processed/vnat_monthly_segments_clean.csv
```

Current data status:

- Raw rows: 155
- Clean monthly rows: 168
- Sample period: January 2012 to December 2025
- Train period: 2012-2022
- Test period: 2023-2025

The raw file contains crawler-quality issues. The validation pipeline reports:

- 13 missing months
- 29 segment-sum consistency warnings
- 24 suspicious repeated target vectors
- 0 duplicated months
- 0 non-positive values

The preprocessing pipeline keeps the raw file unchanged, flags problematic rows, imputes affected values deterministically, and saves a complete monthly panel. Current clean quality flags:

- `observed`: 109 rows
- `segment_reconciliation`: 35 rows
- `stale_repeated_vector`: 24 rows

## Repository Layout

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
│   ├── 04_modeling_segments.ipynb
│   └── 05_results_interpretation.ipynb
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
│   ├── models/
│   │   ├── evaluation.py
│   │   ├── holt_winters.py
│   │   ├── sarima.py
│   │   └── sarima_garch.py
│   └── reporting/
│       ├── create_notebooks.py
│       └── generate_outputs.py
├── requirements.txt
└── README.md
```

## Environment Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If refreshing VNAT data with Playwright:

```powershell
python -m playwright install chromium
```

## Reproducible Pipeline

Run these commands from the repository root.

Validate the raw VNAT file:

```powershell
python src\data\validate_data.py
```

Create the clean modeling dataset:

```powershell
python src\data\preprocess.py
```

Evaluate all models:

```powershell
python src\models\evaluation.py
```

Generate figures, tables, and the written report:

```powershell
python src\reporting\generate_outputs.py
```

Regenerate notebooks:

```powershell
python src\reporting\create_notebooks.py
```

Optional raw data refresh:

```powershell
python src\crawl\crawl_vnat_segments.py --start 2012 --end 2025 --no-cache
```

## Notebooks

The notebooks are written as an academic analysis sequence:

1. `01_eda.ipynb`
   Exploratory analysis of total arrivals, segment volumes, segment shares, seasonality, yearly trend, growth rates, event annotations, and correlation structure.

2. `02_decomposition_stationarity.ipynb`
   Seasonal decomposition, ADF test, KPSS test, ACF/PACF analysis, log transformation comparison, and COVID structural-break discussion.

3. `03_modeling_total_arrivals.ipynb`
   Primary forecast notebook for `international_arrivals`: train/test split, fitted values, forecasts, confidence intervals, residual diagnostics, and model comparison.

4. `04_modeling_segments.ipynb`
   Segment-level modeling for Asia, Europe, Americas, Oceania, and other markets, focused on heterogeneous recovery patterns.

5. `05_results_interpretation.ipynb`
   Final interpretation of objective, data validation, EDA findings, COVID shock, recovery pattern, segment heterogeneity, model performance, implications, limitations, and future work.

## Models

Implemented models:

- Holt-Winters / Exponential Smoothing on log arrivals.
- SARIMA/SARIMAX selected by compact AIC grid search on log arrivals.
- SARIMA-GARCH with GARCH(1,1) fitted to SARIMA residuals, not raw arrivals.

Evaluation metrics:

- MAE
- RMSE
- MAPE
- sMAPE

Outputs:

```text
reports/tables/model_metrics.csv
reports/tables/best_models.csv
reports/tables/model_forecasts.csv
reports/tables/stationarity_tests.csv
reports/model_summary.md
reports/figures/
```

## Current Model Results

Best models by sMAPE in the current run:

| Target | Best model | MAE | RMSE | MAPE | sMAPE |
|---|---:|---:|---:|---:|---:|
| `international_arrivals` | SARIMA | 679,227 | 775,568 | 42.12% | 55.90% |
| `asia_arrivals` | SARIMA | 560,118 | 631,949 | 44.20% | 59.30% |
| `europe_arrivals` | Holt-Winters | 57,828 | 76,346 | 25.89% | 32.19% |
| `americas_arrivals` | SARIMA | 337,475 | 394,276 | 397.07% | 119.38% |
| `oceania_arrivals` | Holt-Winters | 20,392 | 22,209 | 46.16% | 61.76% |
| `other_markets_arrivals` | SARIMA | 1,811 | 2,034 | 45.94% | 62.13% |

Interpretation: SARIMA is currently selected for the primary total-arrivals target, while segment-level results vary. Large percentage errors for smaller segments reflect low denominators, imputation uncertainty, and unstable post-COVID recovery dynamics.

## Figure and Table Style

Figures use a consistent muted academic color palette:

- total arrivals: charcoal
- Asia: muted blue
- Europe: muted green
- Americas: muted orange
- Oceania: muted purple
- other markets: muted gray

Generated plots avoid gridlines, decorative effects, 3D charts, and unnecessary chart elements. Event annotations are used sparingly for COVID-19, border reopening, recovery, Russia-Ukraine war, China-US tensions, and Iran-Israel conflict.

## Important Implementation Notes

- All scripts are designed to run from the repository root.
- Project paths are centralized in `src/config.py`.
- The cleaned dataset is reproducible from the raw VNAT CSV.
- `reports/model_summary.md` is the current generated summary report.
- `reports/report.md` is an older report artifact retained for reference.
- Raw crawler HTML and screenshot caches are intentionally ignored by Git.

## Limitations

The analysis depends on crawler-derived VNAT monthly segment data. The raw file contains missing months, stale repeated pages, and segment-sum inconsistencies. The preprocessing pipeline flags and imputes affected observations, but model results should be interpreted as conditional on this cleaned dataset.

Future work should refresh the VNAT scrape, manually verify problematic months, add exogenous predictors such as flight capacity, exchange rates, visa policy, and source-market macroeconomic indicators, and compare structural-break or regime-switching models.
