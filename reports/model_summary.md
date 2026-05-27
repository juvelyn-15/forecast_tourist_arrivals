# Model Summary

## Research Objective

This project forecasts Vietnam's monthly international tourist arrivals using official VNAT monthly segment data. The empirical target is `international_arrivals`. Segment variables support source-market composition and heterogeneous recovery interpretation but are not primary forecasting targets.

## Data and Validation

The modeling dataset is `data/processed/vnat_monthly_segments_clean.csv`, derived from `data/raw/vnat_monthly_segments.csv`. The sample contains monthly observations from January 2012 to December 2025. The validation layer detects 66 raw-data issues, mainly missing months, segment-sum inconsistencies, and stale repeated crawler values. These observations are flagged and imputed through deterministic time interpolation with calendar-month median fallback.

## Forecast Evaluation

The central comparison evaluates Holt-Winters, SARIMA, SARIMA-GARCH, and XGBoost on the same target and split. The training period is 2012-2022 and the test period is 2023-2025.

| Model | MAE | RMSE | MAPE | sMAPE |
|---|---:|---:|---:|---:|
| Holt-Winters | 7,283,652 | 11,303,800 | 411.07% | 99.49% |
| SARIMA | 679,227 | 775,568 | 42.12% | 55.90% |
| SARIMA-GARCH | 679,227 | 775,568 | 42.12% | 55.90% |
| XGBoost | 444,851 | 491,967 | 28.74% | 34.07% |

The current best model by sMAPE is XGBoost.

## Interpretation

Observation: Vietnam tourism demand is seasonal and shock-sensitive. Statistical implication: lagged arrivals, seasonal structure, and event indicators contain substantial predictive information. Tourism implication: operational forecasts should be updated as new monthly arrivals become available.

Observation: source-market segments recover unevenly. Statistical implication: aggregate forecasts conceal composition changes. Tourism implication: segment data are most useful for market-structure interpretation rather than as separate primary forecasting targets.

## Limitations

The cleaned dataset relies on deterministic interpolation where raw crawler output is missing, stale, or internally inconsistent. Forecast results should be interpreted as conditional on the cleaned VNAT monthly panel. Future work should refresh and manually verify raw VNAT observations and incorporate external demand drivers such as air capacity, visa policy, exchange rates, and source-market macroeconomic indicators.
