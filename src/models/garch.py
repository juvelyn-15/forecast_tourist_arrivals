"""
garch.py
--------
GARCH volatility modeling on SARIMA residuals for Vietnam tourism arrivals.

Economic motivation:
  After the COVID shock, the recovery path of tourism is highly uncertain.
  The variance of monthly arrivals is NOT constant — it spikes during
  geopolitical events (Omicron, border policy changes) and calms during
  stable periods. This "volatility clustering" is the signature GARCH condition.

  GARCH(1,1) on SARIMA residuals:
    ε_t = σ_t · z_t,     z_t ~ iid N(0,1)
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

  Persistence: α + β → 1 implies high volatility memory (common in tourism post-COVID)

Usage:
    python src/models/garch.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import load_monthly_data, get_series, train_test_split_ts

FIG_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/tables")

BLUE   = "#1a5fa8"
RED    = "#c0392b"
GREEN  = "#27ae60"
ORANGE = "#e67e22"


def test_arch_effects(residuals: pd.Series) -> dict:
    """
    Engle's ARCH-LM test.
    H0: No ARCH effects (homoskedastic residuals).
    If p < 0.05 → ARCH effects present → GARCH is warranted.
    """
    lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals.dropna(), nlags=12)
    print(f"\n  Engle's ARCH-LM Test (H0: no ARCH effects):")
    print(f"  LM Stat = {lm_stat:.4f}  |  p-value = {lm_pval:.4f}")
    if lm_pval < 0.05:
        print("  → ARCH effects detected! GARCH modeling is warranted.")
    else:
        print("  → No significant ARCH effects.")
    return {"lm_stat": lm_stat, "lm_pval": lm_pval}


def plot_volatility(residuals: pd.Series, cond_vol: pd.Series, series_name: str):
    """Plot residuals and conditional volatility side by side."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(residuals, color=BLUE, lw=1, label="SARIMA Residuals")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-03-01"),
                alpha=0.15, color=RED)
    ax1.set_title("SARIMA Residuals & GARCH(1,1) Conditional Volatility",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Residual (log scale)")
    ax1.legend()

    ax2.plot(cond_vol, color=ORANGE, lw=1.5, label="Conditional σ_t")
    ax2.fill_between(cond_vol.index, 0, cond_vol, alpha=0.3, color=ORANGE)
    ax2.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-03-01"),
                alpha=0.15, color=RED, label="COVID period")
    ax2.set_ylabel("Conditional Std Dev")
    ax2.set_xlabel("")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_garch_volatility.png"))
    plt.close()
    print("✔ Saved: 09_garch_volatility.png")


def fit_garch(residuals: pd.Series, p: int = 1, q: int = 1) -> object:
    """
    Fit GARCH(p,q) model on SARIMA residuals.
    Scale residuals by 100 for numerical stability.
    """
    scaled = residuals * 100
    model = arch_model(scaled, vol="GARCH", p=p, q=q, dist="normal")
    result = model.fit(disp="off", show_warning=False)
    return result


def print_garch_results(garch_fit):
    """Print GARCH parameter table with economic interpretation."""
    params = garch_fit.params
    pvals  = garch_fit.pvalues

    print(f"\n{'─'*60}")
    print(f"  GARCH(1,1) Parameter Estimates")
    print(f"{'─'*60}")
    print(f"  ω (omega):  {params['omega']:.6f}   [p = {pvals['omega']:.4f}]  — baseline variance")
    print(f"  α (alpha1): {params['alpha[1]']:.4f}  [p = {pvals['alpha[1]']:.4f}]  — ARCH: impact of past shocks")
    print(f"  β (beta1):  {params['beta[1]']:.4f}  [p = {pvals['beta[1]']:.4f}]  — GARCH: variance persistence")
    persistence = params["alpha[1]"] + params["beta[1]"]
    print(f"\n  Persistence (α + β) = {persistence:.4f}")
    if persistence > 0.95:
        print("  → Near-unit-root in variance: shocks to volatility are very long-lived.")
        print("    Economically: uncertainty in tourism recovery persists for many months.")
    elif persistence > 0.80:
        print("  → High persistence: volatility mean-reverts slowly.")
    else:
        print("  → Moderate persistence: volatility mean-reverts relatively quickly.")
    print(f"\n  Log-Likelihood: {garch_fit.loglikelihood:.2f}")
    print(f"  AIC: {garch_fit.aic:.2f}")
    print(f"  BIC: {garch_fit.bic:.2f}")


def main():
    print("=" * 60)
    print("  GARCH Volatility Modeling on SARIMA Residuals")
    print("=" * 60)

    df = load_monthly_data()
    y  = get_series(df)
    y  = y["2010-01":]
    y_log = np.log(y)

    # ── Step 1: Get SARIMA residuals ────────────────────────────────────────────
    print("\n[Step 1] Fitting SARIMA(1,1,1)(1,1,1)₁₂ to extract residuals...")
    sarima = SARIMAX(
        y_log,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend="c",
    ).fit(disp=False)

    residuals = pd.Series(sarima.resid, index=y_log.index).dropna()

    # ── Step 2: Test for ARCH effects ───────────────────────────────────────────
    print("\n[Step 2] Testing for ARCH effects in SARIMA residuals...")
    arch_test = test_arch_effects(residuals)

    # Ljung-Box on squared residuals
    lb_sq = acorr_ljungbox(residuals**2, lags=[6, 12], return_df=True)
    print(f"\n  Ljung-Box on squared residuals (H0: no serial correlation):")
    print(lb_sq[["lb_stat", "lb_pvalue"]].to_string())

    # ── Step 3: Fit GARCH(1,1) ──────────────────────────────────────────────────
    print("\n[Step 3] Fitting GARCH(1,1) model...")
    garch_fit = fit_garch(residuals)
    print_garch_results(garch_fit)

    # ── Step 4: Extract and plot conditional volatility ─────────────────────────
    cond_vol = pd.Series(
        garch_fit.conditional_volatility / 100,
        index=residuals.index
    )
    plot_volatility(residuals, cond_vol, "log arrivals")

    # ── Step 5: Volatility forecast (12 months) ─────────────────────────────────
    vol_forecast = garch_fit.forecast(horizon=12)
    future_vol = np.sqrt(vol_forecast.variance.iloc[-1]) / 100

    print(f"\n  12-Month Ahead Volatility Forecast (σ_t):")
    future_idx = pd.date_range(
        start=y.index[-1] + pd.offsets.MonthBegin(1),
        periods=12, freq="MS"
    )
    vol_df = pd.DataFrame({
        "date": future_idx,
        "forecast_std_log": future_vol.values
    })
    print(vol_df.to_string(index=False))

    vol_df.to_csv(os.path.join(TAB_DIR, "garch_vol_forecast.csv"), index=False)

    # ── Step 6: GARCH summary table ─────────────────────────────────────────────
    params = garch_fit.params
    summary = pd.DataFrame({
        "Parameter": ["ω (omega)", "α (alpha)", "β (beta)", "α + β"],
        "Estimate":  [
            f"{params['omega']:.6f}",
            f"{params['alpha[1]']:.4f}",
            f"{params['beta[1]']:.4f}",
            f"{params['alpha[1]'] + params['beta[1]']:.4f}",
        ],
        "p-value": [
            f"{garch_fit.pvalues['omega']:.4f}",
            f"{garch_fit.pvalues['alpha[1]']:.4f}",
            f"{garch_fit.pvalues['beta[1]']:.4f}",
            "—",
        ],
        "Interpretation": [
            "Baseline (unconditional) variance",
            "ARCH: sensitivity to past shocks",
            "GARCH: variance persistence",
            "Total persistence (→1 = unit root)",
        ]
    })
    summary.to_csv(os.path.join(TAB_DIR, "garch_parameters.csv"), index=False)
    print(f"\n  Saved parameter table → outputs/tables/garch_parameters.csv")

    print("\n✔ GARCH complete.")


if __name__ == "__main__":
    main()
