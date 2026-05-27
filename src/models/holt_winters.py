"""Holt-Winters exponential smoothing model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_forecast(
    train: pd.Series,
    steps: int,
    seasonal_periods: int = 12,
    trend: str = "add",
    seasonal: str = "mul",
) -> dict:
    """Fit Holt-Winters on log arrivals and return level forecasts."""

    y = np.log(train.astype(float).clip(lower=1))
    model = ExponentialSmoothing(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, remove_bias=True)
    pred_log = fitted.forecast(steps)
    fitted_values = np.exp(fitted.fittedvalues).rename("fitted")
    forecast = np.exp(pred_log).rename("forecast")

    resid = y - fitted.fittedvalues
    sigma = float(resid.std(ddof=1)) if resid.notna().sum() > 2 else 0.0
    lower = np.exp(pred_log - 1.96 * sigma).rename("lower")
    upper = np.exp(pred_log + 1.96 * sigma).rename("upper")

    return {
        "model_name": "Holt-Winters",
        "result": fitted,
        "fitted": fitted_values,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "residuals": (train - fitted_values).rename("residual"),
    }
