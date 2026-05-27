"""XGBoost benchmark for total international arrivals only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MAIN_TARGET
from src.features.build_features import EVENT_FEATURES, LAG_FEATURES, ROLLING_WINDOWS, add_event_features, build_feature_frame, feature_columns


def _load_xgboost():
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError("xgboost is not installed. Run: python -m pip install -r requirements.txt") from exc
    return XGBRegressor


def _calendar_event_row(date: pd.Timestamp, time_index: int) -> dict[str, float]:
    row = pd.DataFrame(index=[date])
    row["month"] = date.month
    row["quarter"] = date.quarter
    row["year"] = date.year
    row["time_index"] = time_index
    row = add_event_features(row)
    return row.iloc[0].to_dict()


def _history_features(history: pd.Series, date: pd.Timestamp, time_index: int) -> pd.DataFrame:
    values: dict[str, float] = {}
    for lag in LAG_FEATURES:
        values[f"lag_{lag}"] = float(history.iloc[-lag])
    shifted = history.astype(float)
    for window in ROLLING_WINDOWS:
        window_values = shifted.iloc[-window:]
        values[f"rolling_mean_{window}"] = float(window_values.mean())
        values[f"rolling_std_{window}"] = float(window_values.std(ddof=1))
    values.update(_calendar_event_row(date, time_index))
    return pd.DataFrame([values], columns=feature_columns(), index=[date])


def fit_forecast(train: pd.Series, steps: int) -> dict:
    """Fit XGBoost on lagged total-arrivals features and forecast recursively."""

    XGBRegressor = _load_xgboost()
    train_df = train.rename(MAIN_TARGET).to_frame()
    x_train, y_train = build_feature_frame(train_df, target=MAIN_TARGET)
    if x_train.empty:
        raise RuntimeError("Insufficient training rows after lag feature construction.")

    model = XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(x_train, y_train)

    fitted = pd.Series(model.predict(x_train), index=x_train.index, name="fitted")
    resid = (y_train - fitted).rename("residual")

    history = train.astype(float).copy()
    forecasts: list[float] = []
    future_index = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    for date in future_index:
        row = _history_features(history, date, len(history))
        pred = float(model.predict(row)[0])
        pred = max(pred, 1.0)
        forecasts.append(pred)
        history.loc[date] = pred

    forecast = pd.Series(forecasts, index=future_index, name="forecast")
    sigma = float(resid.std(ddof=1)) if resid.notna().sum() > 2 else 0.0
    lower = (forecast - 1.96 * sigma).clip(lower=1.0).rename("lower")
    upper = (forecast + 1.96 * sigma).rename("upper")
    importance = pd.Series(model.feature_importances_, index=feature_columns(), name="importance").sort_values(ascending=False)

    return {
        "model_name": "XGBoost",
        "result": model,
        "fitted": fitted,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "residuals": resid,
        "feature_importance": importance,
    }
