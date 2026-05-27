"""SARIMA/SARIMAX model selection and forecasting."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_ORDERS = list(itertools.product([0, 1], [0, 1], [0, 1]))
DEFAULT_SEASONAL_ORDERS = [(p, d, q, 12) for p, d, q in itertools.product([0, 1], [0, 1], [0, 1])]


def select_sarima(
    train: pd.Series,
    orders: list[tuple[int, int, int]] | None = None,
    seasonal_orders: list[tuple[int, int, int, int]] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], object]:
    """Select a compact SARIMA specification by AIC on log arrivals."""

    y = np.log(train.astype(float).clip(lower=1))
    best = None
    best_aic = np.inf
    for order in orders or DEFAULT_ORDERS:
        for seasonal_order in seasonal_orders or DEFAULT_SEASONAL_ORDERS:
            if order == (0, 0, 0) and seasonal_order[:3] == (0, 0, 0):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(
                        y,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=150)
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best = (order, seasonal_order, res)
                    best_aic = res.aic
            except Exception:
                continue
    if best is None:
        raise RuntimeError("No SARIMA specification converged.")
    return best


def fit_forecast(train: pd.Series, steps: int) -> dict:
    order, seasonal_order, result = select_sarima(train)
    pred = result.get_forecast(steps=steps)
    pred_log = pred.predicted_mean
    ci_log = pred.conf_int(alpha=0.05)

    fitted = np.exp(result.fittedvalues).rename("fitted")
    forecast = np.exp(pred_log).rename("forecast")
    lower = np.exp(ci_log.iloc[:, 0]).rename("lower")
    upper = np.exp(ci_log.iloc[:, 1]).rename("upper")

    return {
        "model_name": f"SARIMA{order}x{seasonal_order}",
        "result": result,
        "order": order,
        "seasonal_order": seasonal_order,
        "fitted": fitted,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "residuals": (train - fitted).rename("residual"),
    }
