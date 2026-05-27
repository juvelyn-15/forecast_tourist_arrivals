# Model Summary

## Research Objective

This project forecasts Vietnam's monthly international tourist arrivals using official VNAT segment data. The empirical focus is forecasting accuracy, seasonal structure, volatility, and structural uncertainty after COVID-19.

## Data and Validation

The modeling dataset is `data/processed/vnat_monthly_segments_clean.csv`, derived from `data/raw/vnat_monthly_segments.csv`. The sample contains monthly observations from January 2012 to December 2025 for total arrivals and five regional segments. The validation layer detected 66 raw-data issues, mainly missing months, segment-sum inconsistencies, and stale repeated crawler values. These observations were flagged and imputed deterministically using time-based interpolation with calendar-month median fallback.

## Exploratory Findings

Observed arrivals show strong monthly seasonality, a sharp pandemic collapse, and a volatile recovery after the March 2022 border reopening. Annual arrivals reach their sample minimum in 2021 and their maximum in 2025. Asia dominates the regional composition, while Europe, the Americas, Oceania, and other markets display different recovery speeds.

## Stationarity

ADF and KPSS tests indicate that log levels are structurally non-stationary, while first log differences are more stable. This supports the use of differenced seasonal models and motivates residual diagnostics for volatility clustering.

| series               |   ADF p-value |   KPSS p-value |
|:---------------------|--------------:|---------------:|
| log level            |        0.3531 |            0.1 |
| first log difference |        0      |            0.1 |

## Forecast Evaluation

The primary target is total international arrivals. On the 2023-2025 test period, the best total-arrivals model by sMAPE is **SARIMA**, with MAE = 679227, RMSE = 775568, MAPE = 42.12%, and sMAPE = 55.90%.

| target                 | model        |       MAE |      RMSE |   MAPE |   sMAPE |
|:-----------------------|:-------------|----------:|----------:|-------:|--------:|
| americas_arrivals      | SARIMA       | 337475    | 394276    | 397.07 |  119.38 |
| asia_arrivals          | SARIMA       | 560118    | 631949    |  44.2  |   59.3  |
| europe_arrivals        | Holt-Winters |  57828    |  76346.4  |  25.89 |   32.19 |
| international_arrivals | SARIMA       | 679227    | 775568    |  42.12 |   55.9  |
| oceania_arrivals       | Holt-Winters |  20391.9  |  22208.8  |  46.16 |   61.76 |
| other_markets_arrivals | SARIMA       |   1811.43 |   2033.84 |  45.94 |   62.13 |

## Interpretation and Implications

Observation: Vietnam tourism demand is seasonal and shock-sensitive. Statistical implication: models must account for recurring monthly patterns and unstable post-shock residual variance. Tourism implication: capacity planning should use prediction intervals, not point forecasts alone.

Observation: source-region segments recover unevenly. Statistical implication: aggregate forecasts conceal heterogeneous dynamics. Tourism implication: market-specific promotion and aviation-capacity planning should prioritize segment-level recovery evidence.

## Limitations and Future Work

The cleaned dataset relies on deterministic interpolation where raw crawler output is missing or stale. Future work should refresh the VNAT scrape, add exogenous predictors such as flight capacity and exchange rates, and compare structural-break or regime-switching models.
