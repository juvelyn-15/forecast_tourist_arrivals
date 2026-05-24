"""
holt_winters.py
---------------
Holt-Winters Exponential Smoothing for Vietnam tourism arrivals.

Models:
- HW Additive:       Y_t = T_t + S_t + e_t
- HW Multiplicative: Y_t = T_t × S_t × e_t

Both use optimized α (level), β (trend), γ (seasonal) parameters.
Forecast horizon: 24 months ahead.

Usage:
    python src/models/holt_winters.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import load_monthly_data, get_series, train_test_split_ts
from utils.evaluation    import compare_models, mae, rmse, mape

FIG_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/tables")
FCT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/forecasts")

BLUE   = "#1a5fa8"
RED    = "#c0392b"
GREEN  = "#27ae60"
ORANGE = "#e67e22"


def fit_holt_winters(train: pd.Series, model_type: str = "multiplicative") -> object:
    """Fit Holt-Winters model with Box-Cox transformation for stability."""
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=model_type,
        seasonal_periods=12,
        initialization_method="estimated",
        use_boxcox=(model_type == "multiplicative"),  # log-transform stabilizes variance
        freq="MS",
    )
    return model.fit(optimized=True, remove_bias=True)


def print_hw_params(fit, model_name: str):
    """Print optimized smoothing parameters."""
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  α (level):    {fit.params['smoothing_level']:.4f}")
    print(f"  β (trend):    {fit.params['smoothing_trend']:.4f}")
    print(f"  γ (seasonal): {fit.params['smoothing_seasonal']:.4f}")
    print(f"  AIC:          {fit.aic:.2f}")
    print(f"  BIC:          {fit.bic:.2f}")
    print(f"  SSE:          {fit.sse:.2f}")


def plot_hw_results(train, test, forecast_test, forecast_future, model_name, fname):
    """Plot fit and forecast."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train.index, train.values / 1e6, color="grey", lw=1, alpha=0.7, label="Train")
    ax.plot(test.index, test.values / 1e6, color=BLUE, lw=2, label="Actual (test)")
    ax.plot(test.index, forecast_test / 1e6, color=RED, lw=2, ls="--", label="Forecast (test)")

    # Future forecast
    ax.plot(forecast_future.index, forecast_future.values / 1e6,
            color=GREEN, lw=2, ls="-.", label="Forecast (future)")

    # Confidence interval (rough ±1.96 RMSE)
    err = rmse(test.values, forecast_test)
    ax.fill_between(forecast_future.index,
                    (forecast_future.values - 1.96 * err) / 1e6,
                    (forecast_future.values + 1.96 * err) / 1e6,
                    alpha=0.2, color=GREEN, label="95% PI (approx)")

    ax.set_title(f"{model_name} — Vietnam International Arrivals", fontsize=13, fontweight="bold")
    ax.set_ylabel("Arrivals (millions)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname))
    plt.close()
    print(f"✔ Saved: {fname}")


def main():
    print("=" * 60)
    print("  Holt-Winters Exponential Smoothing")
    print("=" * 60)

    df = load_monthly_data()
    y  = get_series(df)

    # Use post-2010 series for modeling (better data quality + relevant regime)
    y = y["2010-01":]

    train, test = train_test_split_ts(y, test_periods=24)
    print(f"\nTrain: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} months)")
    print(f"Test:  {test.index[0].date()}  → {test.index[-1].date()} ({len(test)} months)")

    results = {}

    for model_type, label in [("additive", "HW Additive"), ("multiplicative", "HW Multiplicative")]:
        fit = fit_holt_winters(train, model_type)
        print_hw_params(fit, label)

        # In-sample forecast over test period
        forecast_test = fit.forecast(len(test))
        forecast_test = pd.Series(forecast_test.values, index=test.index)

        # Future forecast (24 months beyond the data)
        fit_full = fit_holt_winters(y, model_type)   # Refit on full series
        future_idx = pd.date_range(
            start=y.index[-1] + pd.offsets.MonthBegin(1),
            periods=24, freq="MS"
        )
        forecast_future = fit_full.forecast(24)
        forecast_future = pd.Series(forecast_future.values, index=future_idx)

        # Metrics
        mape_val = mape(test.values, forecast_test.values)
        rmse_val = rmse(test.values, forecast_test.values)
        print(f"\n  Test Set Accuracy:")
        print(f"  MAE  = {mae(test.values, forecast_test.values):,.0f}")
        print(f"  RMSE = {rmse_val:,.0f}")
        print(f"  MAPE = {mape_val:.2f}%")

        results[label] = {"actual": test.values, "forecast": forecast_test.values}

        # Plot
        fname = f"05_hw_{model_type}.png"
        plot_hw_results(train, test, forecast_test.values, forecast_future, label, fname)

        # Save forecast CSV
        forecast_future.to_frame("forecast").to_csv(
            os.path.join(FCT_DIR, f"hw_{model_type}_forecast.csv")
        )

    # Comparison table
    print("\n--- Model Comparison ---")
    comp = compare_models(results)
    print(comp.to_string())
    comp.to_csv(os.path.join(TAB_DIR, "hw_comparison.csv"))

    print("\n✔ Holt-Winters complete.")


if __name__ == "__main__":
    main()
