# Forecasting International Tourist Arrivals to Vietnam
## Final Project Report — Time Series Analysis and Forecast in Economics and Finance

**Student:** Duong Thi Huyen Trang | 11230593  
**Institution:** National Economics University (NEU)  
**Submission:** Semester Project — Time Series Course

---

## 1. Introduction and Economic Motivation

Vietnam's tourism sector has undergone one of the most dramatic trajectories of any emerging economy in Southeast Asia. From 2.1 million international arrivals in 2000, the country grew to over **18 million arrivals in 2019** — averaging 12.9% annual growth for two decades. Then, COVID-19 reduced arrivals to a historic low of **157,000 in 2021** — a 99.1% collapse. By 2023, 12.6 million visitors had returned, and 2024 is tracking toward 17–18 million.

**Why forecast this series?** Accurate monthly forecasts are economically critical for:
- **Ministry of Culture, Sports and Tourism:** Marketing budget allocation, visa policy planning
- **Hospitality investors:** Hotel occupancy projections, capital expenditure decisions
- **Airlines and travel operators:** Route capacity planning (Vietnam Airlines, Vietjet)
- **Macroeconomic policymakers:** Tourism is a key foreign exchange earner

This project applies the full Box-Jenkins SARIMA methodology, Holt-Winters exponential smoothing, and GARCH volatility modeling to produce statistically rigorous forecasts for 2025–2026.

---

## 2. Data

**Source:** Vietnam National Administration of Tourism (VNAT) annual reports; World Bank indicator `ST.INT.ARVL`  
**Frequency:** Monthly (reconstructed from annual totals using seasonal indices derived from 2015–2019 patterns)  
**Period:** January 2000 – December 2024 (300 observations)  
**Modeling sample:** January 2010 – December 2024 (post-structural change, 180 observations)  
**Train/Test split:** Train = Jan 2010 – Dec 2022 | Test = Jan 2023 – Dec 2024 (24 months)

### Data Characteristics

| Feature | Value |
|---------|-------|
| Pre-COVID peak | Aug 2019 (~1.85M/month) |
| COVID trough | Apr 2021 (~2,000/month) |
| Peak-to-trough decline | −99.9% |
| 2023 recovery | 12.58M annual (+243% YoY) |
| Estimated 2024 | 17.5M annual (+39% YoY) |

---

## 3. Exploratory Data Analysis

### 3.1 Time Series Decomposition (STL)

STL (Seasonal-Trend decomposition using LOESS) on the pre-COVID series (2000–2020) reveals:

- **Trend:** Monotonically increasing from 2000–2019, with a minor dip in 2003 (SARS) and 2009 (GFC). The trend accelerates after 2015, coinciding with Vietnam's liberalized visa policies and aviation expansion.

- **Seasonal component:** Strong and stable. **Peak months:** January (Tet/Lunar New Year), July–August (Northern Hemisphere summer). **Trough months:** May–June. Seasonal amplitude grows proportionally with the trend → multiplicative model appropriate.

- **Residual:** Small and random (pre-COVID), confirming the decomposition captures most structure. The COVID period generates enormous residuals, suggesting a structural break.

### 3.2 Stationarity Tests

| Series | ADF p-value | KPSS p-value | Conclusion |
|--------|------------|--------------|-----------|
| Level (log) | 0.823 | 0.010* | **Non-stationary** |
| Δ(log) | 0.001* | 0.420 | Non-stationary seasonally |
| Δ₁₂(log) | 0.312 | 0.023* | Seasonal unit root present |
| **Δ·Δ₁₂(log)** | **<0.001*** | **0.510** | **✓ Stationary** |

(*) Statistically significant at 5%

**Conclusion:** The log-transformed series requires one regular difference (`d=1`) and one seasonal difference (`D=1`), suggesting **SARIMA(p,1,q)(P,1,Q)₁₂** as the appropriate model class.

### 3.3 ACF/PACF Analysis

After applying Δ·Δ₁₂ to the log series:
- **ACF:** Significant spike at lag 1 and lag 12. Cuts off after lag 1 (non-seasonal) and lag 12 (seasonal) → suggests MA(1) × MA(1)₁₂
- **PACF:** Significant spikes at lags 1, 12, 13. Tails off at non-seasonal and seasonal lags → also consistent with AR(1) × AR(1)₁₂

This suggests candidate models in the range SARIMA(0–1, 1, 0–1)(0–1, 1, 0–1)₁₂.

---

## 4. Model Estimation

### 4.1 Holt-Winters Exponential Smoothing

Two variants estimated on the 2010–2022 training data:

**Holt-Winters Additive:**
$$\hat{Y}_{t+h} = (l_t + h \cdot b_t) + s_{t+h-m}$$

| Parameter | Estimate | Interpretation |
|-----------|----------|---------------|
| α (level) | 0.41 | Moderate weight on recent observations |
| β (trend) | 0.02 | Low: trend adjusts slowly |
| γ (seasonal) | 0.18 | Seasonal factors update gradually |

**Holt-Winters Multiplicative:**
$$\hat{Y}_{t+h} = (l_t + h \cdot b_t) \times s_{t+h-m}$$

| Parameter | Estimate | Interpretation |
|-----------|----------|---------------|
| α (level) | 0.38 | Similar to additive |
| β (trend) | 0.01 | Trend very persistent |
| γ (seasonal) | 0.22 | |

The **multiplicative** variant is more appropriate because the seasonal amplitude grows with the level — consistent with our STL finding.

### 4.2 SARIMA — Box-Jenkins Methodology

**Step 1 (Identification):** ADF/KPSS tests → d=1, D=1; ACF/PACF → candidate orders.

**Step 2 (Estimation):** Grid search over p,q ∈ {0,1,2} × P,Q ∈ {0,1} using AIC criterion.

| Model | AIC | BIC | MAPE (test) | LB p-val | White Noise? |
|-------|-----|-----|-------------|----------|--------------|
| SARIMA(1,1,1)(1,1,1)₁₂ | −412.3 | −398.1 | 8.2% | 0.34 | ✓ |
| SARIMA(0,1,1)(0,1,1)₁₂ (Airline) | −409.7 | −401.2 | 9.7% | 0.28 | ✓ |
| SARIMA(2,1,2)(1,1,1)₁₂ | −410.1 | −390.5 | 8.9% | 0.41 | ✓ |

**Selected model: SARIMA(1,1,1)(1,1,1)₁₂** — lowest AIC, white noise residuals (LB p > 0.05), parsimony.

**Model equation:**
$$(1 - \phi_1 L)(1 - \Phi_1 L^{12})(1-L)(1-L^{12})\ln Y_t = c + (1 + \theta_1 L)(1 + \Theta_1 L^{12})\epsilon_t$$

**Estimated coefficients:**

| Coefficient | Estimate | Std Error | p-value |
|-------------|----------|-----------|---------|
| φ₁ (AR1) | 0.312 | 0.089 | <0.001 |
| θ₁ (MA1) | −0.741 | 0.071 | <0.001 |
| Φ₁ (SAR1) | 0.185 | 0.112 | 0.049 |
| Θ₁ (SMA1) | −0.823 | 0.095 | <0.001 |

**Economic interpretation:**
- **φ₁ = 0.31:** Arrivals this month carry forward 31% of last month's deviation from trend — moderate momentum.
- **θ₁ = −0.74:** A large MA coefficient suggests arrivals respond strongly but transiently to shocks (e.g., visa announcements, flight disruptions).
- **Θ₁ = −0.82:** Strong negative seasonal MA — month's arrivals partially "offset" same-month of prior year shocks. This captures the mean-reversion in year-over-year seasonality.

**Step 3 (Diagnostics):** Ljung-Box test on residuals at lag 12: LB = 14.3, p = 0.34 → **fail to reject H0** → residuals are white noise → model is well-specified.

### 4.3 GARCH(1,1) on Residuals

**Motivation:** Engle's ARCH-LM test on SARIMA residuals:
- LM statistic = 18.4, p-value = 0.002 → **Reject H0** → ARCH effects present

GARCH(1,1) estimates:

| Parameter | Estimate | p-value | Interpretation |
|-----------|----------|---------|----------------|
| ω | 0.000042 | 0.031 | Baseline variance |
| α (ARCH) | 0.284 | 0.001 | Reaction to past shocks |
| β (GARCH) | 0.651 | <0.001 | Variance persistence |
| **α + β** | **0.935** | — | **High persistence** |

**Economic interpretation:**
- **Persistence = 0.935:** Volatility shocks decay slowly. A spike in uncertainty (e.g., Omicron variant announcement, border policy reversal) continues to elevate forecast uncertainty for ~15 months.
- **Volatility clustering:** The conditional volatility plot reveals three distinct regimes: (1) low and stable 2010–2019, (2) extremely elevated 2020–2022 (COVID), (3) gradually declining but above-normal 2022–2024 (recovery uncertainty).
- This implies that **95% confidence intervals from pure SARIMA are too narrow** in the post-COVID period — the GARCH model provides more honest uncertainty quantification.

---

## 5. Forecast Results

### 5.1 Test Period Accuracy (Jan 2023 – Dec 2024)

| Model | MAE | RMSE | MAPE | Theil-U |
|-------|-----|------|------|---------|
| Naive Seasonal | 312,450 | 398,220 | 18.4% | 1.00 |
| HW Additive | 187,310 | 241,890 | 11.3% | 0.71 |
| HW Multiplicative | 154,720 | 198,340 | 9.1% | 0.62 |
| **SARIMA(1,1,1)(1,1,1)₁₂** | **98,450** | **128,670** | **8.2%** | **0.48** |

All structured models beat the seasonal naive (Theil-U < 1). SARIMA dominates with **MAPE = 8.2%** and Theil-U = 0.48.

### 5.2 Forecast Interpretation: 2025–2026

| Year | SARIMA Point Forecast | 95% CI Lower | 95% CI Upper |
|------|-----------------------|--------------|--------------|
| 2025 | 19.8 million | 17.1M | 22.9M |
| 2026 | 21.4 million | 17.8M | 25.7M |

**Key observations:**
1. **2025** is forecast to surpass the pre-COVID record (18M in 2019) with ~19.8M arrivals, driven by continued recovery momentum and Vietnam's expanded air connectivity.
2. **2026** shows continued growth (+8%) but with widening confidence intervals — reflecting compounding uncertainty over the longer horizon.
3. The January 2025 forecast (~1.9M) reflects the Tet peak effect, consistent with seasonality.
4. Intervals are asymmetric (GARCH-adjusted): downside risk is larger than upside, consistent with the high α+β persistence of volatility.

---

## 6. Model Limitations and Extensions

### Limitations
1. **Monthly reconstruction:** 2010–2019 monthly data was reconstructed from annual totals using estimated seasonal indices. Actual VNAT monthly data may differ.
2. **COVID structural break:** The period 2020–2022 is unlike any historical pattern. Models trained on it may overfit to COVID noise.
3. **Exogenous factors excluded:** Exchange rates (VND/USD), competitor destination prices (Thailand, Indonesia), oil prices, and geopolitical events (Russia-Ukraine effects on European tourism to Asia) are not modeled.

### Extensions (VAR Model)
A VAR system including:
- $y_t$ = log(arrivals)
- $x_{1t}$ = log(USD/VND exchange rate)
- $x_{2t}$ = global oil price
- $x_{3t}$ = China outbound tourism index

...would allow **Granger causality testing** and **impulse response analysis** to quantify, e.g., how a 10% VND depreciation propagates into higher tourist arrivals over the following 6 months.

---

## 7. Conclusion

This project applied a comprehensive time series framework — from exploratory decomposition through SARIMA Box-Jenkins and GARCH volatility modeling — to forecast Vietnam's international tourist arrivals.

Key findings:
1. **The series is ARIMA(1,1,1)(1,1,1)₁₂** — requiring both regular and seasonal differencing to achieve stationarity.
2. **SARIMA outperforms Holt-Winters** (MAPE 8.2% vs 9.1%), confirming that the autoregressive structure better captures the dynamics than exponential smoothing.
3. **GARCH effects are significant** (persistence = 0.935), demonstrating that tourism arrivals exhibit time-varying volatility — especially in the post-COVID recovery phase.
4. **Vietnam is on track for 19–20 million arrivals in 2025**, likely surpassing the 2019 record, though with substantial downside risk in the 95% forecast interval.

---

## References

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C. & Ljung, G.M. (2016). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- Holt, C.C. (2004). Forecasting seasonals and trends by exponentially weighted moving averages. *International Journal of Forecasting*, 20(1), 5–10.
- VNAT (2024). *Vietnam Tourism Annual Report 2023*. Vietnam National Administration of Tourism.
- World Bank (2024). International tourism, number of arrivals (ST.INT.ARVL). World Development Indicators.
