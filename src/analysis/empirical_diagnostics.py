"""Additional empirical diagnostics for the main VNAT forecasting target.

Run from the repository root:
    python src/analysis/empirical_diagnostics.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import zivot_andrews

from src.config import CLEAN_SEGMENTS, FIGURES, MAIN_TARGET, TABLES, TEST_START, TRAIN_END
from src.models import holt_winters, sarima, sarima_garch, xgboost_model


LATEX_DIR = TABLES / "latex"
MODEL_FUNCS = {
    "Holt-Winters": holt_winters.fit_forecast,
    "SARIMA": sarima.fit_forecast,
    "SARIMA-GARCH": sarima_garch.fit_forecast,
    "XGBoost": xgboost_model.fit_forecast,
}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def ensure_output_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)


def load_clean(path: Path = CLEAN_SEGMENTS) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError("Input data must contain a 'date' column.")
    if MAIN_TARGET not in df.columns:
        raise ValueError(f"Input data must contain the main target '{MAIN_TARGET}'.")
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index, freq="MS")
    df[MAIN_TARGET] = pd.to_numeric(df[MAIN_TARGET], errors="coerce")
    return df


def main_series(df: pd.DataFrame) -> pd.Series:
    series = df[MAIN_TARGET].astype(float).dropna().clip(lower=1.0)
    series.name = MAIN_TARGET
    return series


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    train = series.loc[:TRAIN_END].dropna()
    test = series.loc[TEST_START:].dropna()
    return train, test


def log_series(series: pd.Series) -> pd.Series:
    return np.log(series.astype(float).clip(lower=1.0)).rename(f"log_{series.name}")


def save_table(df: pd.DataFrame, stem: str, float_format: str = "%.4f") -> None:
    csv_path = TABLES / f"{stem}.csv"
    tex_path = LATEX_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, float_format=lambda x: float_format % x, escape=True)
    tex_path.write_text(latex, encoding="utf-8")


def fit_models(train: pd.Series, test: pd.Series) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for model_name, fit_func in MODEL_FUNCS.items():
        print(f"Fitting {model_name}...")
        results[model_name] = fit_func(train, steps=len(test))
        forecast = results[model_name]["forecast"].copy()
        forecast.index = test.index
        results[model_name]["forecast"] = forecast
        for interval_name in ("lower", "upper"):
            if interval_name in results[model_name]:
                interval = results[model_name][interval_name].copy()
                interval.index = test.index
                results[model_name][interval_name] = interval
    return results


def structural_break_test(log_y: pd.Series) -> pd.DataFrame:
    stat, pvalue, crit, used_lag, breakpoint = zivot_andrews(log_y.dropna(), maxlag=12, regression="ct", autolag="AIC")
    breakpoint = int(breakpoint)
    break_date = log_y.dropna().index[breakpoint]
    return pd.DataFrame(
        [
            {
                "target": MAIN_TARGET,
                "transformation": "log",
                "test": "Zivot-Andrews",
                "regression": "constant and trend",
                "test_statistic": float(stat),
                "p_value": float(pvalue) if pvalue is not None else np.nan,
                "critical_value_1pct": float(crit.get("1%", np.nan)),
                "critical_value_5pct": float(crit.get("5%", np.nan)),
                "critical_value_10pct": float(crit.get("10%", np.nan)),
                "used_lag": int(used_lag),
                "break_index": breakpoint,
                "break_date": break_date.strftime("%Y-%m-%d"),
                "interpretation": "Reject unit root with one structural break at 5%." if pvalue < 0.05 else "Do not reject unit root with one structural break at 5%.",
            }
        ]
    )


def seasonal_significance_test(log_y: pd.Series) -> pd.DataFrame:
    sample = log_y.loc["2012-01-01":"2019-12-01"].dropna().to_frame("log_arrivals")
    sample["year"] = sample.index.year
    sample["month"] = sample.index.month
    matrix = sample.pivot(index="year", columns="month", values="log_arrivals").dropna()
    statistic, pvalue = stats.friedmanchisquare(*[matrix[col].to_numpy() for col in matrix.columns])
    return pd.DataFrame(
        [
            {
                "test": "Friedman test across calendar months",
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "sample_period": "2012-01 to 2019-12",
                "n_years": int(matrix.shape[0]),
                "interpretation": "Seasonal month effects are statistically significant at 5%." if pvalue < 0.05 else "Seasonal month effects are not statistically significant at 5%.",
            }
        ]
    )


def seasonal_stability_tests(log_y: pd.Series) -> pd.DataFrame:
    frame = log_y.dropna().to_frame("log_arrivals")
    frame["month"] = frame.index.month
    pre = frame.loc["2012-01-01":"2019-12-01"]
    post = frame.loc["2022-01-01":"2025-12-01"]
    rows = []
    for month in range(1, 13):
        pre_values = pre.loc[pre["month"] == month, "log_arrivals"].dropna()
        post_values = post.loc[post["month"] == month, "log_arrivals"].dropna()
        if len(pre_values) < 2 or len(post_values) < 2:
            statistic, pvalue = np.nan, np.nan
        else:
            statistic, pvalue = stats.kruskal(pre_values, post_values)
        rows.append(
            {
                "month": month,
                "month_name": MONTH_ABBR[month - 1],
                "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "reject_at_5_percent": bool(pvalue < 0.05) if np.isfinite(pvalue) else False,
            }
        )
    return pd.DataFrame(rows)


def stl_strength(log_y: pd.Series, label: str, start: str, end: str) -> dict[str, float | str]:
    sample = log_y.loc[start:end].dropna()
    stl = STL(sample, period=12, robust=True).fit()
    remainder = pd.Series(stl.resid, index=sample.index)
    seasonal = pd.Series(stl.seasonal, index=sample.index)
    trend = pd.Series(stl.trend, index=sample.index)
    seasonal_denom = float(np.nanvar(remainder + seasonal, ddof=1))
    trend_denom = float(np.nanvar(remainder + trend, ddof=1))
    remainder_var = float(np.nanvar(remainder, ddof=1))
    seasonal_strength = max(0.0, 1.0 - remainder_var / seasonal_denom) if seasonal_denom > 0 else np.nan
    trend_strength = max(0.0, 1.0 - remainder_var / trend_denom) if trend_denom > 0 else np.nan
    return {
        "sample": label,
        "period": f"{sample.index.min().strftime('%Y-%m')} to {sample.index.max().strftime('%Y-%m')}",
        "seasonal_strength": float(seasonal_strength),
        "trend_strength": float(trend_strength),
    }


def seasonal_strength_metrics(log_y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            stl_strength(log_y, "pre-COVID", "2012-01-01", "2019-12-01"),
            stl_strength(log_y, "full sample", "2012-01-01", "2025-12-01"),
        ]
    )


def sarima_estimation_results(result: dict) -> pd.DataFrame:
    res = result["result"]
    params = res.params
    bse = res.bse.reindex(params.index)
    z_stats = params / bse.replace(0, np.nan)
    pvalues = res.pvalues.reindex(params.index)
    conf_int = res.conf_int()
    rows = []
    for name in params.index:
        rows.append(
            {
                "order": str(result["order"]),
                "seasonal_order": str(result["seasonal_order"]),
                "log_likelihood": float(res.llf),
                "AIC": float(res.aic),
                "BIC": float(res.bic),
                "coefficient": name,
                "estimate": float(params[name]),
                "standard_error": float(bse[name]),
                "z_statistic": float(z_stats[name]),
                "p_value": float(pvalues[name]),
                "ci_lower": float(conf_int.loc[name].iloc[0]),
                "ci_upper": float(conf_int.loc[name].iloc[1]),
            }
        )
    return pd.DataFrame(rows)


def garch_estimation_results(sarima_result: dict) -> tuple[pd.DataFrame, object]:
    resid_scale = 10.0
    residuals = (sarima_result["result"].resid.dropna() * resid_scale).astype(float)
    garch = arch_model(residuals, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    garch_result = garch.fit(disp="off")
    params = garch_result.params
    bse = garch_result.std_err.reindex(params.index)
    tvalues = garch_result.tvalues.reindex(params.index)
    pvalues = garch_result.pvalues.reindex(params.index)
    conf_int = garch_result.conf_int()
    alpha = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    rows = []
    for param_name in ["omega", "alpha[1]", "beta[1]"]:
        rows.append(
            {
                "parameter": param_name,
                "estimate": float(params.get(param_name, np.nan)),
                "alpha_plus_beta": alpha + beta,
                "standard_error": float(bse.get(param_name, np.nan)),
                "t_statistic": float(tvalues.get(param_name, np.nan)),
                "p_value": float(pvalues.get(param_name, np.nan)),
                "ci_lower": float(conf_int.loc[param_name].iloc[0]) if param_name in conf_int.index else np.nan,
                "ci_upper": float(conf_int.loc[param_name].iloc[1]) if param_name in conf_int.index else np.nan,
                "log_likelihood": float(garch_result.loglikelihood),
                "AIC": float(garch_result.aic),
                "BIC": float(garch_result.bic),
            }
        )
    return pd.DataFrame(rows), garch_result


def residual_series(model_name: str, result: dict, train: pd.Series) -> pd.Series:
    y_log = log_series(train)
    if model_name == "Holt-Winters":
        fitted = pd.Series(result["result"].fittedvalues, index=train.index)
        residuals = y_log - fitted
    else:
        residuals = pd.Series(result["result"].resid, index=train.index)
    return residuals.replace([np.inf, -np.inf], np.nan).dropna()


def verdict_for_pvalue(pvalue: float, reject_text: str, fail_text: str) -> str:
    if not np.isfinite(pvalue):
        return "Not available"
    return reject_text if pvalue < 0.05 else fail_text


def durbin_watson_verdict(statistic: float) -> str:
    if not np.isfinite(statistic):
        return "Not available"
    if statistic < 1.5:
        return "Evidence of positive first-order autocorrelation."
    if statistic > 2.5:
        return "Evidence of negative first-order autocorrelation."
    return "Near 2 suggests limited first-order autocorrelation."


def residual_diagnostics(results: dict[str, dict], train: pd.Series) -> pd.DataFrame:
    rows = []
    for model_name in ["Holt-Winters", "SARIMA", "SARIMA-GARCH"]:
        resid = residual_series(model_name, results[model_name], train)
        for lag in [6, 12, 24]:
            if len(resid) <= lag + 1:
                statistic, pvalue = np.nan, np.nan
            else:
                lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
                statistic = float(lb["lb_stat"].iloc[0])
                pvalue = float(lb["lb_pvalue"].iloc[0])
            rows.append(
                {
                    "model": model_name,
                    "test": "Ljung-Box",
                    "lag": lag,
                    "statistic": statistic,
                    "p_value": pvalue,
                    "verdict": verdict_for_pvalue(pvalue, "Reject no autocorrelation at 5%.", "No residual autocorrelation detected at 5%."),
                }
            )
        for lag in [6, 12]:
            if len(resid) <= lag + 1:
                statistic, pvalue = np.nan, np.nan
            else:
                statistic, pvalue, _, _ = het_arch(resid, nlags=lag)
            rows.append(
                {
                    "model": model_name,
                    "test": "ARCH-LM",
                    "lag": lag,
                    "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                    "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
                    "verdict": verdict_for_pvalue(float(pvalue), "Reject homoskedastic residuals at 5%.", "No ARCH effects detected at 5%."),
                }
            )
        jb_stat, jb_pvalue, _, _ = jarque_bera(resid)
        rows.append(
            {
                "model": model_name,
                "test": "Jarque-Bera",
                "lag": np.nan,
                "statistic": float(jb_stat),
                "p_value": float(jb_pvalue),
                "verdict": verdict_for_pvalue(float(jb_pvalue), "Reject normal residuals at 5%.", "Normality not rejected at 5%."),
            }
        )
        dw_stat = float(durbin_watson(resid))
        rows.append(
            {
                "model": model_name,
                "test": "Durbin-Watson",
                "lag": np.nan,
                "statistic": dw_stat,
                "p_value": np.nan,
                "verdict": durbin_watson_verdict(dw_stat),
            }
        )
    return pd.DataFrame(rows)


def diebold_mariano(actual: pd.Series, forecast_1: pd.Series, forecast_2: pd.Series, loss: str) -> tuple[float, float]:
    aligned = pd.concat([actual.rename("actual"), forecast_1.rename("f1"), forecast_2.rename("f2")], axis=1).dropna()
    e1 = aligned["actual"] - aligned["f1"]
    e2 = aligned["actual"] - aligned["f2"]
    if loss == "absolute":
        differential = np.abs(e1) - np.abs(e2)
    elif loss == "squared":
        differential = e1.pow(2) - e2.pow(2)
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
    d = differential.to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    mean_d = float(np.mean(d))
    centered = d - mean_d
    max_lag = min(int(np.floor(n ** (1 / 3))), n - 1)
    variance = float(np.sum(centered * centered) / n)
    for lag in range(1, max_lag + 1):
        covariance = float(np.sum(centered[lag:] * centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        variance += 2.0 * weight * covariance
    if variance <= 1e-12:
        return (0.0, 1.0) if abs(mean_d) <= 1e-12 else (np.nan, np.nan)
    dm_stat = mean_d / np.sqrt(variance / n)
    pvalue = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(pvalue)


def forecast_comparison_tests(results: dict[str, dict], test: pd.Series) -> pd.DataFrame:
    comparisons = [
        ("XGBoost", "SARIMA"),
        ("XGBoost", "Holt-Winters"),
        ("SARIMA", "Holt-Winters"),
        ("SARIMA", "SARIMA-GARCH"),
    ]
    rows = []
    for model_1, model_2 in comparisons:
        for loss in ["absolute", "squared"]:
            dm_stat, pvalue = diebold_mariano(test, results[model_1]["forecast"], results[model_2]["forecast"], loss)
            if np.isfinite(dm_stat) and np.isfinite(pvalue):
                direction = model_1 if dm_stat < 0 else model_2
                interpretation = f"{direction} has lower {loss} error loss; difference is statistically significant at 5%." if pvalue < 0.05 else f"No statistically significant {loss} loss difference at 5%."
            else:
                interpretation = "Test not available because loss differentials have insufficient variation."
            rows.append(
                {
                    "model_1": model_1,
                    "model_2": model_2,
                    "loss_function": loss,
                    "dm_statistic": dm_stat,
                    "p_value": pvalue,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def figure_seasonal_stability(log_y: pd.Series) -> None:
    frame = log_y.to_frame("log_arrivals").dropna()
    frame["month"] = frame.index.month
    frame["period"] = np.select(
        [frame.index.year <= 2019, frame.index.year >= 2022],
        ["Pre-COVID (2012-2019)", "Post-reopening (2022-2025)"],
        default="Other",
    )
    frame = frame[frame["period"] != "Other"]
    data = []
    positions = []
    labels = []
    for month in range(1, 13):
        for offset, period in [(-0.18, "Pre-COVID (2012-2019)"), (0.18, "Post-reopening (2022-2025)")]:
            data.append(frame[(frame["month"] == month) & (frame["period"] == period)]["log_arrivals"])
            positions.append(month + offset)
        labels.append(MONTH_ABBR[month - 1])
    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
    colors = ["#d9d9d9", "#8c8c8c"] * 12
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
    for median in box["medians"]:
        median.set_color("#111111")
    ax.set_xticks(range(1, 13), labels)
    ax.set_ylabel("Log international arrivals")
    ax.set_title("Monthly Seasonal Distributions Before and After Reopening")
    ax.legend([box["boxes"][0], box["boxes"][1]], ["2012-2019", "2022-2025"], frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "seasonal_stability_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def figure_structural_break(log_y: pd.Series, break_date: pd.Timestamp) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(log_y.index, log_y.values, color="#2f3437", linewidth=1.8)
    ax.axvline(break_date, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(break_date, ax.get_ylim()[0], f" Break: {break_date:%Y-%m}", va="bottom", ha="left", color="#333333")
    ax.set_title("Zivot-Andrews Structural Break in Log Arrivals")
    ax.set_ylabel("Log international arrivals")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES / "structural_break_zivot_andrews.png", bbox_inches="tight")
    plt.close(fig)


def figure_strength(strength_df: pd.DataFrame) -> None:
    labels = strength_df["sample"].tolist()
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(x - width / 2, strength_df["seasonal_strength"], width, label="Seasonal strength", color="#7f7f7f")
    ax.bar(x + width / 2, strength_df["trend_strength"], width, label="Trend strength", color="#c7c7c7", edgecolor="#555555")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Strength metric")
    ax.set_title("STL Seasonal and Trend Strength")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "seasonal_strength_comparison.png", bbox_inches="tight")
    plt.close(fig)


def figure_residual_diagnostics(results: dict[str, dict], train: pd.Series) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex="col")
    for row, model_name in enumerate(["Holt-Winters", "SARIMA", "SARIMA-GARCH"]):
        resid = residual_series(model_name, results[model_name], train)
        axes[row, 0].plot(resid.index, resid.values, color="#2f3437", linewidth=1.1)
        axes[row, 0].axhline(0, color="#999999", linewidth=0.8)
        axes[row, 0].set_title(f"{model_name} residuals")
        stats.probplot(resid, dist="norm", plot=axes[row, 1])
        axes[row, 1].get_lines()[0].set_markerfacecolor("#7f7f7f")
        axes[row, 1].get_lines()[0].set_markeredgecolor("#7f7f7f")
        axes[row, 1].get_lines()[1].set_color("#333333")
        axes[row, 1].set_title(f"{model_name} Q-Q plot")
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_diagnostics_updated.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_output_dirs()
    set_plot_style()

    df = load_clean()
    series = main_series(df)
    train, test = split_series(series)
    log_y = log_series(series)

    print(f"Loaded {len(series)} monthly observations for {MAIN_TARGET}.")
    print(f"Training period: {train.index.min():%Y-%m} to {train.index.max():%Y-%m}; test period: {test.index.min():%Y-%m} to {test.index.max():%Y-%m}.")

    fitted_results = fit_models(train, test)

    structural_df = structural_break_test(log_y)
    seasonal_sig_df = seasonal_significance_test(log_y)
    stability_df = seasonal_stability_tests(log_y)
    strength_df = seasonal_strength_metrics(log_y)
    sarima_df = sarima_estimation_results(fitted_results["SARIMA"])
    garch_df, _ = garch_estimation_results(fitted_results["SARIMA"])
    residual_df = residual_diagnostics(fitted_results, train)
    dm_df = forecast_comparison_tests(fitted_results, test)

    tables = {
        "structural_break_tests": structural_df,
        "seasonal_significance_tests": seasonal_sig_df,
        "seasonal_stability_tests": stability_df,
        "seasonal_strength_metrics": strength_df,
        "sarima_estimation_results": sarima_df,
        "garch_estimation_results": garch_df,
        "residual_diagnostics": residual_df,
        "diebold_mariano_tests": dm_df,
    }
    for stem, table in tables.items():
        save_table(table, stem)

    break_date = pd.Timestamp(structural_df.loc[0, "break_date"])
    figure_seasonal_stability(log_y)
    figure_structural_break(log_y, break_date)
    figure_strength(strength_df)
    figure_residual_diagnostics(fitted_results, train)

    print("\nCompleted empirical diagnostics.")
    print("CSV tables:")
    for stem in tables:
        print(f"  - {TABLES / f'{stem}.csv'}")
    print("LaTeX tables:")
    for stem in tables:
        print(f"  - {LATEX_DIR / f'{stem}.tex'}")
    print("Figures:")
    for name in [
        "seasonal_stability_boxplot.png",
        "structural_break_zivot_andrews.png",
        "seasonal_strength_comparison.png",
        "residual_diagnostics_updated.png",
    ]:
        print(f"  - {FIGURES / name}")


if __name__ == "__main__":
    main()
