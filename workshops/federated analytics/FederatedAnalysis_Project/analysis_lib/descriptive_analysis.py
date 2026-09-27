"""Helper functions for the descriptive analysis notebook."""

# ====================================
# Imports
# ====================================
# Standard library
from typing import Any

# Third-party
import numpy as np
import pandas as pd


# ====================================
# Constants
# ====================================
# Same age bins used in 01_eda.ipynb's distribution plots, kept consistent across the workshop.
AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
AGE_BIN_LABELS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]


# ====================================
# Descriptive statistics
# ====================================
def descript_stats(column: pd.Series, name: str) -> dict[str, Any]:
    """Compute summary statistics for a single column, tailored to its dtype.

    Args:
        column: Column to summarize.
        name: Label identifying the group this column belongs to (e.g. a site name or "Centralized").

    Returns:
        For a numeric column: mean, sd, sum, sum_squares, median, q1, q3, min, max,
        missing and n_data, plus one count per age bin when `column.name == "age"`.
        For a categorical/binary column: missing and n_data, plus one count per value present.
    """
    if pd.api.types.is_numeric_dtype(column):
        result = {
            "name": name,
            "mean": column.mean(),
            "sd": column.std(),
            "sum": column.sum(),
            "sum_squares": (column**2).sum(),
            "median": column.median(),
            "q1": column.quantile(0.25),
            "q3": column.quantile(0.75),
            "min": column.min(),
            "max": column.max(),
            "missing": column.isna().sum(),
            "n_data": column.count(),
        }

        # --- Age bins ---
        if column.name == "age":
            age_bins = pd.cut(column, bins=AGE_BINS, labels=AGE_BIN_LABELS, right=False)
            result.update(age_bins.value_counts().reindex(AGE_BIN_LABELS).to_dict())

    # Categorical or binary column
    else:
        result = {
            "name": name,
            "missing": column.isna().sum(),
            "n_data": column.count(),
        }
        for value in column.dropna().unique():
            result[str(value)] = str((column == value).sum())

    return result


# ====================================
# Result table assembly
# ====================================
def result_df(result: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of per-group descript_stats() results into a single indexed DataFrame.

    Args:
        result: One descript_stats() output per group (e.g. one per site).

    Returns:
        A DataFrame with one row per group, indexed by group name.
    """
    df_result = pd.DataFrame(result)
    df_result = df_result.rename(columns={"name": "Characteristic"})
    df_result = df_result.set_index("Characteristic")
    return df_result


# ====================================
# Federated aggregation
# ====================================
def weighted_avg(df: pd.DataFrame, metric: str) -> float:
    """Compute the n_data-weighted average of a metric across groups.

    Args:
        df: Table with one row per group, including columns `metric` and `n_data`.
        metric: Column to average.

    Returns:
        The n_data-weighted average of `metric` across all groups.
    """
    return (df[metric] * df["n_data"]).sum() / df["n_data"].sum()


def weighted_std(df: pd.DataFrame) -> float:
    """Compute the exact pooled standard deviation across groups.

    Reconstructs the standard deviation the centralized approach would have
    computed over the pooled data, from each group's sum and sum of squares.

    Args:
        df: Table with one row per group, including columns `n_data`, `sum` and `sum_squares`.

    Returns:
        The pooled standard deviation across all groups.
    """
    n = df["n_data"].sum()
    total_sum = df["sum"].sum()
    total_sum_squares = df["sum_squares"].sum()
    variance = total_sum_squares / (n - 1) - (total_sum**2) / (n * (n - 1))
    return np.sqrt(variance)


# ====================================
# Comparison formatting
# ====================================
def percents(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Format s1/s2 as a percentage string, for appending to count columns.

    Args:
        s1: Numerator counts.
        s2: Denominator counts (e.g. total records).

    Returns:
        Series of strings like " (12.34%)", aligned to s1's index.
    """
    percentage = (s1.astype(float) / s2.astype(float)) * 100
    return " (" + percentage.round(2).astype(str) + "%)"


def process_comparison(df: pd.DataFrame, is_numeric: bool, column: str) -> pd.DataFrame:
    """Format a descript_stats() comparison table for display, appending percentages.

    Args:
        df: Concatenated per-group results for a single column (output of result_df()).
        is_numeric: Whether the original column is numeric, selects the formatting path.
        column: Name of the original column; special-cased for "age" to include its bins.

    Returns:
        Transposed, string-formatted table ready for display, with one column per group
        and one row per characteristic (e.g. "mean (sd)", or each categorical value with its %).
    """
    df = df.round(2)
    df_result = df.astype(str)

    if is_numeric:
        column_order = ["mean (sd)", "median", "q1 - q3", "min-max", "missing", "n_data"]
        df_result["mean (sd)"] = df_result["mean"] + " (" + df_result["sd"] + ")"
        df_result["q1 - q3"] = df_result["q1"] + " - " + df_result["q3"]
        df_result["min-max"] = df_result["min"] + " - " + df_result["max"]
        if column == "age":
            for bin_label in AGE_BIN_LABELS:
                df_result[bin_label] = df_result[bin_label] + percents(df_result[bin_label], df_result["n_data"])
            column_order = column_order + AGE_BIN_LABELS
    else:
        column_order = ["missing", "n_data"]
        for col in df_result.columns:
            if col not in ("missing", "n_data"):
                df_result[col] = df_result[col] + percents(df_result[col], df_result["n_data"])
                column_order.append(col)

    df_result = df_result.fillna("")
    df_result = df_result[column_order]
    df_result = df_result.T

    return df_result