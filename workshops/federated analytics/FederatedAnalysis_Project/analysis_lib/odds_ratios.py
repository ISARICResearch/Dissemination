"""Helper functions for the odds ratios notebook."""

# ====================================
# Imports
# ====================================
# Standard library

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


# ====================================
# Distribution analysis
# ====================================
def plot_distributions_by_outcome(
    df: pd.DataFrame, outcome: str, predictors: list[str], event_value: str
) -> None:
    """Plot each predictor's association with the outcome, to spot unstable predictors before modelling.

    A categorical predictor is plotted as the event rate within each of its categories
    (e.g. the death rate among "Yes" vs among "No"), not as raw counts, since this
    dataset's outcome is heavily imbalanced and raw counts would hide the pattern.
    A numeric predictor is plotted as one normalized histogram per outcome value.

    Args:
        df: Dataset to plot from.
        outcome: Name of the outcome column to split by.
        predictors: Names of the predictor columns to plot, one subplot each.
        event_value: The outcome value whose rate is plotted for categorical predictors
            (e.g. "Dead").
    """
    ncols = 4
    nrows = -(-len(predictors) // ncols)  # Ceil division, to fit every predictor in the grid
    _, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for ax, predictor in zip(axes, predictors):
        if pd.api.types.is_numeric_dtype(df[predictor]):
            # A numeric predictor (age): normalized so the very unequal group sizes stay comparable
            for value, group in df.groupby(outcome)[predictor]:
                ax.hist(group, bins=15, density=True, alpha=0.6, label=str(value))
            ax.legend(fontsize=7)
        else:
            # A categorical predictor: event rate within each category, not raw counts,
            # so the outcome's class imbalance doesn't hide a category with too little variation
            # Each subplot keeps its own y-axis scale (not a shared 0-100), so a predictor
            # with generally low event rates isn't flattened out by one with much higher ones
            event_rate = df.groupby(predictor)[outcome].apply(lambda s: (s == event_value).mean())
            ax.bar(event_rate.index.astype(str), event_rate.values * 100)
            ax.set_ylabel(f"% {event_value}")
        ax.set_title(predictor, fontsize=9)
        ax.set_xlabel("")

    # Hide unused subplot slots when the predictors don't fill the grid exactly
    for ax in axes[len(predictors):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ====================================
# Per-site separation
# ====================================
def find_separated_predictors(
    site_data: dict[str, pd.DataFrame], outcome: str, predictors: list[str]
) -> pd.DataFrame:
    """Find, per site, the Yes/No predictors with a zero count in their predictor-by-outcome table.

    A zero count (e.g. no deaths among "Yes") is complete separation: the logistic
    regression's coefficient grows without bound and its standard error explodes.
    Predictors and outcome are expected encoded as 0/1 (No/Yes, Alive/Dead).

    Args:
        site_data: Each site's dataset, keyed by site name.
        outcome: Name of the 0/1 outcome column.
        predictors: Names of the predictor columns to check; non-0/1 ones (e.g. age) are skipped.

    Returns:
        One row per (site, predictor) with a zero count, holding the 4 counts of its table.
    """
    count_columns = ["Alive (No)", "Dead (No)", "Alive (Yes)", "Dead (Yes)"]
    rows = []

    for site, df in site_data.items():
        for predictor in predictors:
            # Only 0/1 predictors form a 2x2 table with the outcome
            if not df[predictor].isin([0, 1]).all():
                continue

            # Reindex so a combination with no records shows up as 0 instead of missing
            counts = pd.crosstab(df[predictor], df[outcome]).reindex(
                index=[0, 1], columns=[0, 1], fill_value=0
            )
            if (counts == 0).any().any():
                rows.append({
                    "site": site,
                    "predictor": predictor,
                    **dict(zip(count_columns, counts.to_numpy().flatten())),
                })

    return pd.DataFrame(rows, columns=["site", "predictor", *count_columns]).set_index(["site", "predictor"])


# ====================================
# Federated aggregation
# ====================================
def pool_odds_ratios(site_table: pd.DataFrame) -> dict[str, float]:
    """Combine one predictor's per-site results by inverse-variance weighting on the log-odds scale.

    Args:
        site_table: One row per site, with columns "Coefficient" and "StdErr".

    Returns:
        The combined coefficient, standard error, odds ratio, 95% CI and p-value,
        plus the number of sites combined ("Sites").
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
        "OddsRatio": np.exp(coefficient),
        "CI95_low": np.exp(coefficient - z_critical * std_err),
        "CI95_high": np.exp(coefficient + z_critical * std_err),
        "p-value": 2 * norm.sf(abs(z)),
        "Sites": len(site_table),
    }


# ====================================
# Result formatting
# ====================================
def format_odds_ratio_table(results: pd.DataFrame) -> pd.DataFrame:
    """Format a univariate odds ratio table for display, sorted by p-value.

    Args:
        results: One row per predictor, with columns "OddsRatio", "CI95_low",
            "CI95_high" and "p-value" (e.g. compute_univariate_odds_ratios()'s output).

    Returns:
        A table with the odds ratio and its 95% CI combined into one column.
    """
    results = results.sort_values("p-value")
    table = results.round(3).astype(str)
    table["Odds Ratio (95% CI)"] = (
        table["OddsRatio"] + " (" + table["CI95_low"] + ", " + table["CI95_high"] + ")"
    )
    return table[["Odds Ratio (95% CI)", "p-value"]]


def _format_estimate(results: pd.DataFrame) -> pd.Series:
    """Format each row's odds ratio and 95% CI as "OR [low; high]", with 2 decimals."""
    return results.apply(
        lambda row: f"{row['OddsRatio']:.2f} [{row['CI95_low']:.2f}; {row['CI95_high']:.2f}]", axis=1
    )


def _format_p_value(p_value: float) -> str:
    """Format a p-value with 3 decimals ("<0.001" below that), starred when below 0.05."""
    text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    return f"{text}*" if p_value < 0.05 else text


def format_comparison_table(centralized: pd.DataFrame, federated: pd.DataFrame) -> pd.DataFrame:
    """Build a side-by-side table of centralized and federated odds ratios, sorted by centralized p-value.

    Args:
        centralized: compute_univariate_odds_ratios()'s output over the whole dataset.
        federated: One row per predictor with the same columns, plus "Sites"
            (the number of institutions combined).

    Returns:
        A table with grouped "Centralized" and "Federated" columns: OR [95% CI] and p
        for both, plus Sites for the federated approach.
    """
    centralized = centralized.sort_values("p-value")
    federated = federated.reindex(centralized.index)

    return pd.DataFrame({
        ("Centralized", "OR [95% CI]"): _format_estimate(centralized),
        ("Centralized", "p"): centralized["p-value"].map(_format_p_value),
        ("Federated", "OR [95% CI]"): _format_estimate(federated),
        ("Federated", "p"): federated["p-value"].map(_format_p_value),
        ("Federated", "Sites"): federated["Sites"],
    })


# ====================================
# Forest plot
# ====================================
def plot_forest(
    results: pd.DataFrame,
    title: str = "Forest Plot",
    comparison: pd.DataFrame | None = None,
    labels: tuple[str, str] = ("Centralized", "Federated"),
) -> None:
    """Plot each predictor's odds ratio and 95% confidence interval, optionally against a second set of results.

    Args:
        results: One row per predictor, with columns "OddsRatio", "CI95_low"
            and "CI95_high" (e.g. compute_univariate_odds_ratios()'s output).
        title: Plot title.
        comparison: Optional second set of results with the same columns, drawn
            next to `results` for each predictor.
        labels: Legend labels for `results` and `comparison`, used only with `comparison`.
    """
    results = results.sort_values("OddsRatio")
    series = [(results, labels[0], "tab:blue")]
    if comparison is not None:
        # Aligned to results' predictor order, so both points of a predictor share a row
        series.append((comparison.reindex(results.index), labels[1], "tab:orange"))

    # Each predictor gets one row; with a comparison, its two points are shifted
    # slightly up and down so both confidence intervals stay readable
    y = np.arange(len(results))
    offset = 0.15 if comparison is not None else 0
    row_height = 0.6 if comparison is not None else 0.4
    _, ax = plt.subplots(figsize=(8, row_height * len(results) + 1))

    for i, (table, label, color) in enumerate(series):
        # One point per predictor at its odds ratio, with a horizontal line spanning its 95% CI
        ax.errorbar(
            table["OddsRatio"],
            y + offset * (1 - 2 * i),
            xerr=[table["OddsRatio"] - table["CI95_low"], table["CI95_high"] - table["OddsRatio"]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            label=label,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(results.index)
    if comparison is not None:
        # Bottom right stays empty: the lowest odds ratios sit left of the line of no effect
        ax.legend(loc="lower right")

    # Line of no effect: an odds ratio of 1
    ax.axvline(x=1, color="red", linestyle="--")

    # A log scale keeps odds ratios below and above 1 (e.g. 0.5 and 2) equally
    # spaced from the line of no effect, matching how odds ratios are interpreted
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()