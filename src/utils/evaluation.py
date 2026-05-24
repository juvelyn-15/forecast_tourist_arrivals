"""
evaluation.py
-------------
Forecast accuracy metrics and model comparison utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return np.mean(np.abs(actual - forecast))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - forecast) ** 2))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Excludes zeros in actual."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE — handles near-zero values better."""
    denom = (np.abs(actual) + np.abs(forecast)) / 2
    return np.mean(np.abs(actual - forecast) / denom) * 100


def theil_u(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Theil's U statistic — U < 1 means better than naive forecast."""
    naive = np.roll(actual, 1)[1:]
    act   = actual[1:]
    fcast = forecast[1:]
    rmse_model = rmse(act, fcast)
    rmse_naive = rmse(act, naive)
    return rmse_model / (rmse_naive + 1e-10)


def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    h: int = 1
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.
    H0: Both models have equal forecast accuracy.
    Returns test stat and p-value.
    """
    e1 = actual - forecast1
    e2 = actual - forecast2
    d  = e1**2 - e2**2    # Loss differential (MSE-based)
    n  = len(d)
    d_bar = np.mean(d)

    # Newey-West variance estimate
    gamma0 = np.var(d, ddof=1)
    gamma_sum = sum(
        (1 - j / (h + 1)) * np.cov(d[j:], d[:-j])[0, 1]
        for j in range(1, h + 1)
        if len(d[j:]) > 1
    )
    lrv = gamma0 + 2 * gamma_sum
    dm_stat = d_bar / np.sqrt(lrv / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "DM_stat": round(dm_stat, 4),
        "p_value": round(p_value, 4),
        "reject_H0_5pct": p_value < 0.05,
    }


def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a comparison table from a dict of model results.

    Args:
        results: {model_name: {"actual": ..., "forecast": ...}}

    Returns:
        DataFrame with MAE, RMSE, MAPE, Theil-U for each model.
    """
    rows = []
    for name, res in results.items():
        actual   = np.array(res["actual"])
        forecast = np.array(res["forecast"])
        rows.append({
            "Model":   name,
            "MAE":     f"{mae(actual, forecast):,.0f}",
            "RMSE":    f"{rmse(actual, forecast):,.0f}",
            "MAPE (%)": f"{mape(actual, forecast):.2f}",
            "SMAPE (%)": f"{smape(actual, forecast):.2f}",
            "Theil-U":  f"{theil_u(actual, forecast):.3f}",
        })
    return pd.DataFrame(rows).set_index("Model")
