"""Helper functions for the exploratory data analysis notebook."""

# ====================================
# Imports
# ====================================
# Standard library
from typing import Any

# Third-party
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display


# ====================================
# Constants
# ====================================
# Only this category has a numeric variable (age), so only it gets a Range column.
CATEGORY_WITH_RANGE = "Demographic"


# ====================================
# Variable overview
# ====================================
def _summarize_missing(column: pd.Series, total_records: int) -> tuple[int, float]:
    """Return a column's missing value count and percentage."""
    missing_count = int(column.isna().sum())
    return missing_count, 100 * missing_count / total_records


def _summarize_unique_values(column: pd.Series, possible_values: list[str] | None) -> str:
    """List each possible value's count and percentage, or the number of unique values."""
    if not possible_values:
        return f"{column.nunique()} unique"

    value_counts = column.value_counts()
    total = value_counts.sum()
    return ", ".join(
        f"{value}: {count} ({100 * count / total:.1f}%)" for value, count in value_counts.items()
    )


def _summarize_range(column: pd.Series) -> str:
    """Describe a numeric column's mean and interquartile range, empty otherwise."""
    if not pd.api.types.is_numeric_dtype(column):
        return ""

    q1, q3 = column.quantile([0.25, 0.75])
    return f"Mean: {column.mean():.1f}, IQR: {q1:.1f}–{q3:.1f}"


def display_variable_overview(df: pd.DataFrame, data_dictionary: dict[str, dict[str, Any]]) -> None:
    """Display each column's meaning and key statistics, grouped by category.

    Args:
        df: Dataset the statistics (missing values, unique values, range) are computed from.
        data_dictionary: Mapping of column name to its metadata (description,
            possible_values, data_type and category), as loaded from the
            project's data dictionary JSON file.
    """
    rows_by_category: dict[str, list[dict[str, str]]] = {}
    total_records = len(df)

    for column_name, metadata in data_dictionary.items():
        column = df[column_name]
        category = metadata["category"]
        missing_count, missing_pct = _summarize_missing(column, total_records)

        row = {
            "Column": column_name,
            "Description": metadata["description"] or "",
            "Data type": metadata["data_type"],
            "Missing (n)": missing_count,
            "Missing (%)": f"{missing_pct:.1f}",
            "Unique values": _summarize_unique_values(column, metadata["possible_values"]),
        }
        if category == CATEGORY_WITH_RANGE:
            row["Range"] = _summarize_range(column)

        rows_by_category.setdefault(category, []).append(row)

    for category, rows in rows_by_category.items():
        display(Markdown(f"#### {category}"))
        display(pd.DataFrame(rows))


# ====================================
# Distribution plots
# ====================================
def _plot_category_bar(
    ax: plt.Axes,
    series: pd.Series,
    categories: pd.Index,
    normalize: bool,
    subplot_title: str,
    show_values: bool = False,
) -> None:
    """Draw one subplot's bar chart for a category's counts or percentages."""
    counts = series.value_counts(normalize=normalize).reindex(categories).fillna(0)
    if normalize:
        counts *= 100

    positions = range(len(counts))
    bars = ax.bar(positions, counts.to_numpy())
    if show_values:
        labels = [f"{value:.1f}%" if normalize else f"{value:,.0f}" for value in counts]
        ax.bar_label(bars, labels=labels, fontsize=8)
        # Headroom so the label above the tallest bar stays inside the axes
        ax.margins(y=0.15)
    ax.set_xticks(positions)
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
    ax.set_title(subplot_title)
    ax.set_ylabel("%" if normalize else "Count")


def plot_distribution_by_site(
    df: pd.DataFrame,
    column: str,
    site_column: str = "institution",
    bins: list[float] | None = None,
    bin_labels: list[str] | None = None,
    normalize: bool = True,
    title: str | None = None,
    show_values: bool = False,
) -> None:
    """Plot a column's distribution overall and broken down by site.

    Args:
        df: Dataset to plot from.
        column: Column whose distribution is plotted.
        site_column: Column identifying each site (e.g. institution).
        bins: Bin edges to group a numeric column into ranges before plotting.
        bin_labels: Label for each bin in `bins`.
        normalize: Show percentages within each group when True, raw counts otherwise.
        title: Figure title. Defaults to `column`.
        show_values: Write each bar's value above it.
    """
    plot_series = df[column]
    if bins is not None:
        plot_series = pd.cut(plot_series, bins=bins, labels=bin_labels, right=False)

    categories = plot_series.cat.categories if bins is not None else plot_series.value_counts().index
    sites = sorted(df[site_column].unique())

    fig = plt.figure(figsize=(3 * len(sites), 6))
    grid = fig.add_gridspec(2, len(sites))

    ax_total = fig.add_subplot(grid[0, :])
    _plot_category_bar(ax_total, plot_series, categories, normalize, "Total", show_values)

    # Sites share a y-axis with each other, not with the total, so the largest
    # site sets the scale instead of the (much bigger) combined total.
    first_site_ax = None
    for i, site in enumerate(sites):
        ax = fig.add_subplot(grid[1, i], sharey=first_site_ax)
        first_site_ax = first_site_ax or ax
        _plot_category_bar(ax, plot_series[df[site_column] == site], categories, normalize, site, show_values)

    fig.suptitle(title or column)
    fig.tight_layout()
    plt.show()