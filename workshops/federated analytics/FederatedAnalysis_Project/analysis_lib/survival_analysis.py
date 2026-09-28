"""Helper functions for the survival analysis notebook."""

# ====================================
# Imports
# ====================================
# Standard library

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.duration.survfunc import SurvfuncRight


# ====================================
# Survival curves
# ====================================
def plot_survival_curves(
    curves: dict[str, SurvfuncRight], follow_up_days: int, title: str, y_min: float = 0.0
) -> None:
    """Plot one Kaplan-Meier survival curve per group, over the whole follow-up period.

    Args:
        curves: Each group's fitted survival curve, keyed by the label shown in the legend.
        follow_up_days: Last day of follow-up, where the curves end.
        title: Plot title.
        y_min: Lowest survival probability shown on the vertical axis, in percent.
    """
    _, ax = plt.subplots(figsize=(8, 5))
    for label, curve in curves.items():
        # Start at 100% on day 0 and keep the last value until the end of follow-up
        days = np.concatenate([[0], curve.surv_times, [follow_up_days]])
        probabilities = np.concatenate([[1], curve.surv_prob, [curve.surv_prob[-1]]]) * 100
        # Survival only changes on the days with deaths, so it's drawn as steps
        ax.step(days, probabilities, where="post", label=label)

    ax.set_xlim(0, follow_up_days)
    ax.set_ylim(y_min, 100.05)
    ax.set_xlabel("Days since symptom onset")
    ax.set_ylabel("Survival probability (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    # A legend is only needed when groups are compared
    if len(curves) > 1:
        ax.legend()
    plt.tight_layout()
    plt.show()


def survival_table(curves: dict[str, SurvfuncRight], day: int) -> pd.DataFrame:
    """Summarize each survival curve on a given day: patients, deaths and survival with its standard error and 95% CI.

    Args:
        curves: Each group's fitted survival curve, keyed by the row label.
        day: Day on which the curves are read (e.g. the end of follow-up).

    Returns:
        One row per curve, with survival, its standard error and its 95% CI in percent.
    """
    z_critical = norm.ppf(0.975)
    rows = {}
    for label, curve in curves.items():
        # Survival on a day is the value of the last step on or before it (100% if no death yet)
        reached = curve.surv_times <= day
        survival = curve.surv_prob[reached][-1] if reached.any() else 1.0
        std_err = curve.surv_prob_se[reached][-1] if reached.any() else 0.0
        rows[label] = {
            "Patients": len(curve.time),
            "Deaths": int(curve.status[curve.time <= day].sum()),
            "Survival (%)": 100 * survival,
            "StdErr (%)": 100 * std_err,
            "CI95_low (%)": 100 * (survival - z_critical * std_err),
            "CI95_high (%)": 100 * (survival + z_critical * std_err),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# ====================================
# Federated aggregation
# ====================================
def pool_survival(site_table: pd.DataFrame) -> dict[str, float]:
    """Combine one group's per-site survival on a given day, weighting each site by its number of patients.

    Args:
        site_table: One row per site, with columns "Patients", "Deaths",
            "Survival (%)" and "StdErr (%)" (survival_table()'s output).

    Returns:
        The total patients and deaths, the combined survival, its standard error and 95% CI, in percent.
    """
    # Each site's share of all patients, so larger sites count more
    weights = site_table["Patients"] / site_table["Patients"].sum()

    # Weighted average of the survival probabilities; the sites are independent,
    # so the variance of that average is the sum of each site's variance times its squared weight
    survival = (weights * site_table["Survival (%)"]).sum()
    std_err = np.sqrt((weights ** 2 * site_table["StdErr (%)"] ** 2).sum())

    z_critical = norm.ppf(0.975)
    return {
        "Patients": site_table["Patients"].sum(),
        "Deaths": site_table["Deaths"].sum(),
        "Survival (%)": survival,
        "StdErr (%)": std_err,
        "CI95_low (%)": survival - z_critical * std_err,
        "CI95_high (%)": survival + z_critical * std_err,
    }


def pool_hazard_ratio(site_table: pd.DataFrame) -> dict[str, float]:
    """Combine one predictor's per-site Cox results by inverse-variance weighting of their coefficients.

    Args:
        site_table: One row per site, with columns "Coefficient" and "StdErr".

    Returns:
        The combined coefficient, standard error, hazard ratio, 95% CI and p-value.
    """
    # One over the squared standard error, so more precise sites count more
    weights = 1 / site_table["StdErr"] ** 2

    # Weighted average of the coefficients, and the standard error of that average
    coefficient = (site_table["Coefficient"] * weights).sum() / weights.sum()
    std_err = np.sqrt(1 / weights.sum())

    # Same critical value statsmodels uses for its 95% CI (about 1.96)
    z_critical = norm.ppf(0.975)
    z = coefficient / std_err

    return {
        "Coefficient": coefficient,
        "StdErr": std_err,
        "HazardRatio": np.exp(coefficient),
        "CI95_low": np.exp(coefficient - z_critical * std_err),
        "CI95_high": np.exp(coefficient + z_critical * std_err),
        "p-value": 2 * norm.sf(abs(z)),
    }


# ====================================
# Result formatting
# ====================================
def _format_survival(results: pd.DataFrame) -> pd.Series:
    """Format each row's survival and 95% CI as "S% [low; high]", with 2 decimals."""
    return results.apply(
        lambda row: f"{row['Survival (%)']:.2f}% [{row['CI95_low (%)']:.2f}; {row['CI95_high (%)']:.2f}]", axis=1
    )


def format_survival_comparison(centralized: pd.DataFrame, federated: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side table of centralized and federated survival, with their 95% CI.

    Args:
        centralized: survival_table()'s output over the whole dataset.
        federated: One row per group with the same columns (the combined results).

    Returns:
        A table with the patients and deaths per group, and the survival [95% CI] of both approaches.
    """
    federated = federated.reindex(centralized.index)
    return pd.DataFrame({
        ("", "Patients"): centralized["Patients"],
        ("", "Deaths"): centralized["Deaths"],
        ("Centralized", "Survival [95% CI]"): _format_survival(centralized),
        ("Federated", "Survival [95% CI]"): _format_survival(federated),
    })


def _format_hazard_ratio(results: pd.DataFrame) -> pd.Series:
    """Format each row's hazard ratio and 95% CI as "HR [low; high]", with 2 decimals."""
    return results.apply(
        lambda row: f"{row['HazardRatio']:.2f} [{row['CI95_low']:.2f}; {row['CI95_high']:.2f}]", axis=1
    )


def _format_p_value(p_value: float) -> str:
    """Format a p-value with 3 decimals ("<0.001" below that), starred when below 0.05."""
    text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    return f"{text}*" if p_value < 0.05 else text


def format_hazard_ratio_comparison(centralized: pd.DataFrame, federated: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side table of centralized and federated hazard ratios.

    Args:
        centralized: compute_hazard_ratio()'s output over the whole dataset.
        federated: One row per predictor with the same columns (the combined results).

    Returns:
        A table with grouped "Centralized" and "Federated" columns: HR [95% CI] and p for both.
    """
    federated = federated.reindex(centralized.index)
    return pd.DataFrame({
        ("Centralized", "HR [95% CI]"): _format_hazard_ratio(centralized),
        ("Centralized", "p"): centralized["p-value"].map(_format_p_value),
        ("Federated", "HR [95% CI]"): _format_hazard_ratio(federated),
        ("Federated", "p"): federated["p-value"].map(_format_p_value),
    })