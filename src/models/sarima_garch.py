"""SARIMA-GARCH model.

The conditional mean is estimated with SARIMA on log arrivals. GARCH is fitted
to the SARIMA residuals, not to raw arrivals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from src.models.sarima import select_sarima


def fit_forecast(train: pd.Series, steps: int) -> dict:
    y = np.log(train.astype(float).clip(lower=1))
    order, seasonal_order, sarima_result = select_sarima(train)
    mean_pred = sarima_result.get_forecast(steps=steps)
    mean_log = mean_pred.predicted_mean

    resid_scale = 10.0
    resid = (sarima_result.resid.dropna() * resid_scale).astype(float)
    if len(resid) < 36:
        raise RuntimeError("Too few residual observations for GARCH.")

    garch = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    garch_result = garch.fit(disp="off")
    variance = garch_result.forecast(horizon=steps, reindex=False).variance.iloc[-1].to_numpy()
    sigma_log = np.sqrt(np.maximum(variance, 0.0)) / resid_scale

    fitted = np.exp(sarima_result.fittedvalues).rename("fitted")
    forecast = np.exp(mean_log).rename("forecast")
    lower = np.exp(mean_log - 1.96 * sigma_log).rename("lower")
    upper = np.exp(mean_log + 1.96 * sigma_log).rename("upper")

    return {
        "model_name": f"SARIMA-GARCH{order}x{seasonal_order}",
        "result": sarima_result,
        "garch_result": garch_result,
        "order": order,
        "seasonal_order": seasonal_order,
        "fitted": fitted,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "residuals": (train - fitted).rename("residual"),
        "conditional_volatility": pd.Series(garch_result.conditional_volatility / resid_scale, index=resid.index),
    }
