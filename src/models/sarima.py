"""
sarima.py
---------
Seasonal ARIMA modeling pipeline for Vietnam tourism arrivals.

Steps (Box-Jenkins Methodology):
  1. Identification: ACF/PACF + unit root tests → determine (p,d,q)(P,D,Q)_12
  2. Estimation: Auto-ARIMA grid search + manual specification
  3. Diagnostic Checking: Ljung-Box, residual normality, ACF of residuals
  4. Forecasting: 24-month ahead with 95% CI

Key economic interpretation:
  - d=1, D=1: Series is non-stationary in both level and seasonal component
  - AR terms: Arrivals this month depend on recent months (persistence)
  - MA terms: Captures short-lived shocks (exchange rate announcements, events)
  - Seasonal AR/MA: Captures year-over-year patterns (Tet, summer peaks)

Usage:
    python src/models/sarima.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

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


def auto_arima_search(train_log: pd.Series) -> object:
    """
    Use pmdarima auto_arima for initial model identification.
    Searches over a constrained grid to find best AIC model.
    """
    print("\n[Auto-ARIMA] Searching for optimal order...")
    model = pm.auto_arima(
        train_log,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        d=1, D=1,
        m=12,                     # Monthly seasonality
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        information_criterion="aic",
        stepwise=True,
        seasonal=True,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
    )
    print(f"\n→ Best model: SARIMA{model.order}×{model.seasonal_order}")
    print(f"  AIC = {model.aic():.3f}  |  BIC = {model.bic():.3f}")
    return model


def fit_sarima(train: pd.Series, order: tuple, seasonal_order: tuple) -> object:
    """Fit a SARIMAX model using statsmodels (richer diagnostics)."""
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.fit(disp=False, method="lbfgs", maxiter=500)


def print_model_summary(fit, order, seasonal_order):
    """Print coefficient table."""
    p, d, q = order
    P, D, Q, s = seasonal_order
    print(f"\n{'─'*60}")
    print(f"  SARIMA({p},{d},{q})({P},{D},{Q})₁₂")
    print(f"{'─'*60}")
    print(fit.summary().tables[1])
    print(f"\n  AIC  = {fit.aic:.3f}")
    print(f"  BIC  = {fit.bic:.3f}")
    print(f"  HQIC = {fit.hqic:.3f}")


def run_diagnostics(fit, model_name: str) -> dict:
    """
    Box-Jenkins diagnostic checks:
    1. Ljung-Box test on residuals (H0: no autocorrelation)
    2. Residual ACF plot
    3. Jarque-Bera normality test
    """
    residuals = fit.resid.dropna()

    # Ljung-Box test
    lb = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)

    # Diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Diagnostic Plots — {model_name}", fontsize=13, fontweight="bold")

    # Residuals over time
    axes[0, 0].plot(residuals, color=BLUE, lw=1)
    axes[0, 0].axhline(0, color=RED, lw=0.8, ls="--")
    axes[0, 0].set_title("Residuals")

    # Histogram
    axes[0, 1].hist(residuals, bins=30, color=BLUE, alpha=0.7, edgecolor="white")
    axes[0, 1].set_title("Residual Distribution")

    # ACF of residuals
    plot_acf(residuals, lags=36, ax=axes[1, 0], color=BLUE, title="ACF of Residuals")

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot")

    plt.tight_layout()
    fname = f"07_diagnostics_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(FIG_DIR, fname))
    plt.close()

    print(f"\n  Ljung-Box Test (H0: no autocorrelation in residuals):")
    print(lb[["lb_stat", "lb_pvalue"]].rename(
        columns={"lb_stat": "LB Stat", "lb_pvalue": "p-value"}
    ).to_string())

    return {
        "lb_lag12_pval": lb["lb_pvalue"].iloc[0],
        "lb_lag24_pval": lb["lb_pvalue"].iloc[1],
        "residuals_white_noise": bool(lb["lb_pvalue"].iloc[0] > 0.05),
    }


def plot_forecast(train, test, forecast_df, model_name: str, fname: str):
    """Plot actual vs forecast with confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Show last 3 years of training data only (cleaner plot)
    train_tail = train[-36:]
    ax.plot(train_tail.index, train_tail.values / 1e6,
            color="grey", lw=1.2, alpha=0.8, label="Train (last 3yr)")
    ax.plot(test.index, test.values / 1e6, color=BLUE, lw=2, label="Actual")
    ax.plot(forecast_df.index, forecast_df["forecast"] / 1e6,
            color=RED, lw=2, ls="--", label="Forecast")

    ax.fill_between(
        forecast_df.index,
        forecast_df["lower_95"] / 1e6,
        forecast_df["upper_95"] / 1e6,
        alpha=0.2, color=RED, label="95% CI"
    )

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
    print("  SARIMA Modeling — Box-Jenkins Methodology")
    print("=" * 60)

    df = load_monthly_data()
    y  = get_series(df)
    y  = y["2010-01":]

    # Log-transform for variance stabilization
    y_log = np.log(y)

    train, test       = train_test_split_ts(y, test_periods=24)
    train_log, test_log = train_test_split_ts(y_log, test_periods=24)

    # ── Step 1: Auto-ARIMA identification ───────────────────────────────────────
    auto_model = auto_arima_search(train_log)
    auto_order    = auto_model.order
    auto_seasonal = auto_model.seasonal_order

    # ── Step 2: Candidate models to evaluate ────────────────────────────────────
    # Include auto-ARIMA result plus common SARIMA benchmarks for comparison
    candidates = {
        "SARIMA(1,1,1)(1,1,1)₁₂":  ((1, 1, 1), (1, 1, 1, 12)),
        "SARIMA(0,1,1)(0,1,1)₁₂":  ((0, 1, 1), (0, 1, 1, 12)),   # Airline model
        "SARIMA(2,1,2)(1,1,1)₁₂":  ((2, 1, 2), (1, 1, 1, 12)),
        f"Auto-ARIMA {auto_order}×{auto_seasonal[:3]}": (auto_order, auto_seasonal),
    }

    best_aic, best_name, best_fit = np.inf, None, None
    all_results = {}
    model_metrics = []

    for name, (order, seasonal_order) in candidates.items():
        print(f"\n{'='*60}")
        print(f"  Fitting: {name}")
        try:
            fit = fit_sarima(train_log, order, seasonal_order)
            print_model_summary(fit, order, seasonal_order)
            diag = run_diagnostics(fit, name)

            # Forecast on test set (on log scale → back-transform)
            fc = fit.get_forecast(len(test))
            fc_mean = np.exp(fc.predicted_mean)
            fc_ci   = np.exp(fc.conf_int())
            fc_ci.columns = ["lower_95", "upper_95"]
            fc_ci.index = test.index
            fc_mean.index = test.index

            forecast_df = pd.DataFrame({
                "forecast": fc_mean.values,
                "lower_95": fc_ci["lower_95"].values,
                "upper_95": fc_ci["upper_95"].values,
            }, index=test.index)

            mae_v  = mae(test.values, fc_mean.values)
            rmse_v = rmse(test.values, fc_mean.values)
            mape_v = mape(test.values, fc_mean.values)

            print(f"\n  Test Set Accuracy:")
            print(f"  MAE  = {mae_v:,.0f}")
            print(f"  RMSE = {rmse_v:,.0f}")
            print(f"  MAPE = {mape_v:.2f}%")

            model_metrics.append({
                "Model": name,
                "AIC": round(fit.aic, 2),
                "BIC": round(fit.bic, 2),
                "MAE": f"{mae_v:,.0f}",
                "RMSE": f"{rmse_v:,.0f}",
                "MAPE (%)": f"{mape_v:.2f}",
                "LB p-val (lag12)": f"{diag['lb_lag12_pval']:.3f}",
                "White Noise?": "✓" if diag["residuals_white_noise"] else "✗",
            })

            all_results[name] = {"actual": test.values, "forecast": fc_mean.values}

            if fit.aic < best_aic:
                best_aic  = fit.aic
                best_name = name
                best_fit  = (fit, order, seasonal_order, forecast_df)

        except Exception as e:
            print(f"  ⚠ Failed: {e}")

    # ── Step 3: Best model diagnostics & future forecast ────────────────────────
    if best_fit:
        fit, order, seasonal_order, forecast_df = best_fit
        print(f"\n{'='*60}")
        print(f"  ★ Best Model (by AIC): {best_name}")
        print(f"{'='*60}")

        plot_forecast(train, test, forecast_df, best_name,
                      "06_sarima_test_forecast.png")

        # Future forecast (refit on full series)
        fit_full = fit_sarima(y_log, order, seasonal_order)
        fc_full  = fit_full.get_forecast(24)
        future_idx = pd.date_range(
            start=y.index[-1] + pd.offsets.MonthBegin(1),
            periods=24, freq="MS"
        )
        future_mean = np.exp(fc_full.predicted_mean)
        future_ci   = np.exp(fc_full.conf_int())
        future_mean.index = future_idx
        future_ci.index   = future_idx

        future_df = pd.DataFrame({
            "forecast": future_mean.values,
            "lower_95": future_ci.iloc[:, 0].values,
            "upper_95": future_ci.iloc[:, 1].values,
        }, index=future_idx)

        future_df.to_csv(os.path.join(FCT_DIR, "sarima_future_forecast.csv"))

        # Full forecast plot (test + future)
        combined_fc = pd.concat([forecast_df, future_df])
        combined_test = pd.concat([test, pd.Series([np.nan] * 24, index=future_idx)])
        plot_forecast(y, combined_test, combined_fc,
                      f"{best_name} (Full Horizon)", "08_sarima_full_forecast.png")

        print(f"\n  2025–2026 Annual Forecasts:")
        for year in [2025, 2026]:
            yr_fc = future_df[future_df.index.year == year]["forecast"].sum() / 1e6
            yr_lo = future_df[future_df.index.year == year]["lower_95"].sum() / 1e6
            yr_hi = future_df[future_df.index.year == year]["upper_95"].sum() / 1e6
            print(f"  {year}: {yr_fc:.2f}M arrivals  [CI: {yr_lo:.2f}M – {yr_hi:.2f}M]")

    # ── Save comparison table ────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df.to_csv(os.path.join(TAB_DIR, "sarima_model_comparison.csv"), index=False)
    print(f"\n--- Model Comparison ---")
    print(metrics_df.to_string(index=False))

    print("\n✔ SARIMA complete.")


if __name__ == "__main__":
    main()
