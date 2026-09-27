"""Shared data preprocessing helpers for the project's workshop notebooks."""

# ====================================
# Imports
# ====================================
# Standard library
from typing import Any

# Third-party
import pandas as pd


# ====================================
# Column selection
# ====================================
def drop_columns_by_category(
    df: pd.DataFrame, data_dictionary: dict[str, dict[str, Any]], categories_to_drop: list[str]
) -> pd.DataFrame:
    """Drop every column whose data-dictionary category is in `categories_to_drop`.

    Args:
        df: Dataset to filter.
        data_dictionary: Mapping of column name to its metadata, as loaded from
            the project's data dictionary JSON file.
        categories_to_drop: Category names (e.g. "Date") whose columns are removed.

    Returns:
        A copy of `df` without the dropped columns.
    """
    columns_to_drop = [
        column
        for column, metadata in data_dictionary.items()
        if metadata["category"] in categories_to_drop
    ]
    return df.drop(columns=columns_to_drop)


# ====================================
# Missing values
# ====================================
def fillna_by_category(
    df: pd.DataFrame,
    data_dictionary: dict[str, dict[str, Any]],
    categories: list[str],
    fill_value: str,
) -> pd.DataFrame:
    """Fill missing values with `fill_value` in every column whose data-dictionary category is in `categories`.

    Args:
        df: Dataset to fill.
        data_dictionary: Mapping of column name to its metadata, as loaded from
            the project's data dictionary JSON file.
        categories: Category names (e.g. "Symptom") whose columns get filled.
        fill_value: Value used to replace missing values in the selected columns.

    Returns:
        A copy of `df` with missing values filled in the selected columns.
    """
    columns = [
        column
        for column, metadata in data_dictionary.items()
        if metadata["category"] in categories and column in df.columns
    ]
    df = df.copy()
    df[columns] = df[columns].fillna(fill_value)
    return df


# ====================================
# Encoding
# ====================================
def encode_yes_no(
    df: pd.DataFrame, data_dictionary: dict[str, dict[str, Any]], categories: list[str]
) -> pd.DataFrame:
    """Encode every "Yes"/"No" column whose data-dictionary category is in `categories` to 1/0.

    Args:
        df: Dataset to encode.
        data_dictionary: Mapping of column name to its metadata, as loaded from
            the project's data dictionary JSON file.
        categories: Category names (e.g. "Symptom") whose columns get encoded.

    Returns:
        A copy of `df` with the selected columns mapped from "Yes"/"No" to 1/0.
    """
    columns = [
        column
        for column, metadata in data_dictionary.items()
        if metadata["category"] in categories and column in df.columns
    ]
    df = df.copy()
    df[columns] = df[columns].apply(lambda column: column.map({"Yes": 1, "No": 0}).astype(int))
    return df