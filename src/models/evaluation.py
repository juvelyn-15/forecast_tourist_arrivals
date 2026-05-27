"""Forecast evaluation runner for all VNAT target series.

Run from the repository root:
    python src/models/evaluation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import CLEAN_SEGMENTS, TABLES, TARGET_COLUMNS, TRAIN_END
from src.models import holt_winters, sarima, sarima_garch


ModelFunc = Callable[[pd.Series, int], dict]


MODELS: dict[str, ModelFunc] = {
    "Holt-Winters": holt_winters.fit_forecast,
    "SARIMA": sarima.fit_forecast,
    "SARIMA-GARCH": sarima_garch.fit_forecast,
}


def load_clean(path=CLEAN_SEGMENTS) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df.index.freq = "MS"
    return df


def split_series(series: pd.Series, train_end: str = TRAIN_END) -> tuple[pd.Series, pd.Series]:
    train = series.loc[:train_end].dropna()
    test = series.loc[pd.Timestamp(train_end) + pd.offsets.MonthBegin(1) :].dropna()
    return train, test


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned = pd.concat([y_true.rename("actual"), y_pred.rename("forecast")], axis=1).dropna()
    if aligned.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan}
    actual = aligned["actual"].to_numpy(dtype=float)
    forecast = aligned["forecast"].to_numpy(dtype=float)
    denom = np.where(actual == 0, np.nan, actual)
    smape_denom = np.abs(actual) + np.abs(forecast)
    return {
        "MAE": float(mean_absolute_error(actual, forecast)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, forecast))),
        "MAPE": float(np.nanmean(np.abs((actual - forecast) / denom)) * 100),
        "sMAPE": float(np.nanmean(2 * np.abs(forecast - actual) / np.where(smape_denom == 0, np.nan, smape_denom)) * 100),
    }


def evaluate_target(series: pd.Series, target: str) -> tuple[list[dict], dict[str, dict], pd.DataFrame]:
    train, test = split_series(series)
    rows: list[dict] = []
    results: dict[str, dict] = {}
    forecasts: list[pd.DataFrame] = []

    for model_name, func in MODELS.items():
        try:
            result = func(train, steps=len(test))
            forecast = result["forecast"].copy()
            forecast.index = test.index
            lower = result.get("lower", pd.Series(index=test.index, dtype=float)).copy()
            upper = result.get("upper", pd.Series(index=test.index, dtype=float)).copy()
            lower.index = test.index
            upper.index = test.index

            row = {"target": target, "model": model_name, "status": "ok", **metrics(test, forecast)}
            rows.append(row)
            results[model_name] = result
            forecasts.append(
                pd.DataFrame(
                    {
                        "date": test.index,
                        "target": target,
                        "model": model_name,
                        "actual": test.values,
                        "forecast": forecast.values,
                        "lower": lower.values,
                        "upper": upper.values,
                    }
                )
            )
        except Exception as exc:
            rows.append({"target": target, "model": model_name, "status": f"failed: {exc}", "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan})

    forecast_df = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    return rows, results, forecast_df


def run_all(df: pd.DataFrame, targets: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    forecast_frames: list[pd.DataFrame] = []

    for target in targets or TARGET_COLUMNS:
        rows, _, forecasts = evaluate_target(df[target], target)
        metric_rows.extend(rows)
        if not forecasts.empty:
            forecast_frames.append(forecasts)

    metrics_df = pd.DataFrame(metric_rows)
    ok = metrics_df[metrics_df["status"] == "ok"].copy()
    best = ok.sort_values(["target", "sMAPE", "RMSE"]).groupby("target", as_index=False).first()
    forecasts_df = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    return metrics_df, best, forecasts_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Holt-Winters, SARIMA, and SARIMA-GARCH models.")
    parser.add_argument("--input", default=CLEAN_SEGMENTS)
    args = parser.parse_args()

    df = load_clean(args.input)
    metrics_df, best_df, forecasts_df = run_all(df)
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLES / "model_metrics.csv", index=False)
    best_df.to_csv(TABLES / "best_models.csv", index=False)
    forecasts_df.to_csv(TABLES / "model_forecasts.csv", index=False)
    print(metrics_df.round(3).to_string(index=False))
    print(f"\nSaved metrics to {TABLES / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
