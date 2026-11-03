"""Reusable, read-only profiling helpers for uploaded datasets."""

from typing import Any

import pandas as pd


def profile_dataset(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return basic dataset-wide metrics without modifying the DataFrame."""
    row_count, column_count = dataframe.shape

    return {
        "rows": row_count,
        "columns": column_count,
        "total_cells": row_count * column_count,
        "total_missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def profile_columns(dataframe: pd.DataFrame, sample_size: int = 3) -> pd.DataFrame:
    """Return a general profile for every column in a DataFrame.

    Missing values are excluded from the unique-value count and the sample
    values, so the profile focuses on the values that are actually present.
    """
    row_count = len(dataframe)
    profile_rows: list[dict[str, Any]] = []

    # Using positions means duplicate column labels are still profiled safely.
    for position, column_name in enumerate(dataframe.columns):
        column = dataframe.iloc[:, position]
        non_null_count = int(column.notna().sum())
        missing_count = int(column.isna().sum())
        unique_count = int(column.nunique(dropna=True))

        profile_rows.append(
            {
                "Column": str(column_name),
                "Data Type": str(column.dtype),
                "Non-Null Values": non_null_count,
                "Missing Values": missing_count,
                "Missing %": _percentage(missing_count, row_count),
                "Unique Values": unique_count,
                "Unique %": _percentage(unique_count, row_count),
                "Sample Values": column.dropna().head(sample_size).tolist(),
            }
        )

    return pd.DataFrame(profile_rows)


def profile_numerical_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numerical columns only."""
    numerical_data = dataframe.select_dtypes(include="number")
    profile_rows: list[dict[str, Any]] = []

    for position, column_name in enumerate(numerical_data.columns):
        column = numerical_data.iloc[:, position]
        profile_rows.append(
            {
                "Column": str(column_name),
                "Minimum": column.min(),
                "Maximum": column.max(),
                "Mean": column.mean(),
                "Median": column.median(),
                "Standard Deviation": column.std(),
            }
        )

    return pd.DataFrame(
        profile_rows,
        columns=[
            "Column",
            "Minimum",
            "Maximum",
            "Mean",
            "Median",
            "Standard Deviation",
        ],
    )


def profile_categorical_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return frequency details for object and string columns only."""
    profile_rows: list[dict[str, Any]] = []

    for position, column_name in enumerate(dataframe.columns):
        column = dataframe.iloc[:, position]
        if not (pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column)):
            continue

        value_counts = column.value_counts(dropna=True)
        if value_counts.empty:
            most_frequent_value: Any = None
            most_frequent_frequency = 0
        else:
            most_frequent_value = value_counts.index[0]
            most_frequent_frequency = int(value_counts.iloc[0])

        profile_rows.append(
            {
                "Column": str(column_name),
                "Unique Values": int(column.nunique(dropna=True)),
                "Most Frequent Value": most_frequent_value,
                "Frequency": most_frequent_frequency,
            }
        )

    return pd.DataFrame(
        profile_rows,
        columns=["Column", "Unique Values", "Most Frequent Value", "Frequency"],
    )


def _percentage(part: int, total: int) -> float:
    """Calculate a percentage safely for normal and empty DataFrames."""
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)
