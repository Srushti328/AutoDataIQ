"""Data Quality Engine module for AutoDataIQ.

Provides read-only detection and reporting of data quality issues:
- Missing values & empty columns
- Duplicate rows
- Constant & near-constant columns
- High-cardinality categorical/text fields
- Data-type inconsistencies & invalid domain values
- Potential outliers using IQR statistical method
"""

from typing import Any

import numpy as np
import pandas as pd

from modules.semantic_detector import detect_semantic_types


def analyze_data_quality(
    dataframe: pd.DataFrame,
    semantic_types_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Inspect a DataFrame and return structured data quality reports.

    Returns a dictionary containing:
    - summary: dict of dataset-level quality metrics
    - issues_df: DataFrame of all detected quality issues
    - outliers_df: DataFrame of numerical outlier details (IQR method)
    """
    if dataframe is None or dataframe.empty:
        return {
            "summary": {
                "total_issues_count": 0,
                "missing_value_columns_count": 0,
                "total_missing_cells": 0,
                "duplicate_rows_count": 0,
                "outlier_columns_count": 0,
                "high_cardinality_columns_count": 0,
                "empty_columns_count": 0,
                "inconsistent_columns_count": 0,
            },
            "issues_df": pd.DataFrame(
                columns=[
                    "Issue Type",
                    "Column",
                    "Severity",
                    "Count",
                    "Percentage",
                    "Description",
                ]
            ),
            "outliers_df": pd.DataFrame(
                columns=[
                    "Column",
                    "Outlier Count",
                    "Outlier %",
                    "Lower Bound",
                    "Upper Bound",
                    "Min Value",
                    "Max Value",
                ]
            ),
        }

    # If semantic classifications are not provided, run Phase 3 semantic detector
    if semantic_types_df is None or semantic_types_df.empty:
        semantic_types_df = detect_semantic_types(dataframe)

    # Build semantic type map: col_name -> semantic_type
    semantic_map = dict(
        zip(semantic_types_df["Column"], semantic_types_df["Semantic Type"])
    )

    row_count = len(dataframe)
    issues_list: list[dict[str, Any]] = []
    outlier_list: list[dict[str, Any]] = []

    # Track overall summary counters
    missing_value_cols = 0
    empty_cols = 0
    outlier_cols = 0
    high_cardinality_cols = 0
    inconsistent_cols: set[str] = set()

    # -------------------------------------------------------------
    # 1. Dataset-level Duplicate Detection
    # -------------------------------------------------------------
    duplicate_rows_count = int(dataframe.duplicated().sum())
    duplicate_pct = (duplicate_rows_count / row_count * 100) if row_count > 0 else 0.0

    if duplicate_rows_count > 0:
        if duplicate_pct >= 20.0:
            dup_severity = "High"
        elif duplicate_pct >= 5.0:
            dup_severity = "Medium"
        else:
            dup_severity = "Low"

        issues_list.append(
            {
                "Issue Type": "Duplicate Rows",
                "Column": "Dataset-level",
                "Severity": dup_severity,
                "Count": duplicate_rows_count,
                "Percentage": f"{duplicate_pct:.1f}%",
                "Description": f"{duplicate_rows_count} duplicate row(s) detected ({duplicate_pct:.1f}% of total rows).",
            }
        )

    # -------------------------------------------------------------
    # 2. Column-by-Column Quality Inspections
    # -------------------------------------------------------------
    for position, col_name in enumerate(dataframe.columns):
        series = dataframe.iloc[:, position]
        sem_type = semantic_map.get(str(col_name), "Unknown")

        non_null_series = series.dropna()
        non_null_count = len(non_null_series)
        missing_count = row_count - non_null_count
        missing_pct = (missing_count / row_count * 100) if row_count > 0 else 0.0

        # A. Missing Values & Empty Columns
        if missing_count == row_count:
            empty_cols += 1
            issues_list.append(
                {
                    "Issue Type": "Empty Column",
                    "Column": str(col_name),
                    "Severity": "High",
                    "Count": row_count,
                    "Percentage": "100.0%",
                    "Description": "Column is completely empty (100% missing values).",
                }
            )
            continue  # Skip further checks for 100% empty column

        if missing_count > 0:
            missing_value_cols += 1
            if missing_pct >= 50.0:
                m_severity = "High"
            elif missing_pct >= 20.0:
                m_severity = "Medium"
            else:
                m_severity = "Low"

            issues_list.append(
                {
                    "Issue Type": "Missing Values",
                    "Column": str(col_name),
                    "Severity": m_severity,
                    "Count": missing_count,
                    "Percentage": f"{missing_pct:.1f}%",
                    "Description": f"{missing_count} missing value(s) ({missing_pct:.1f}% of column).",
                }
            )

        unique_count = int(non_null_series.nunique())
        uniqueness_ratio = unique_count / non_null_count if non_null_count > 0 else 0.0

        # B. Constant & Near-Constant Columns
        if unique_count == 1:
            first_val = non_null_series.iloc[0]
            issues_list.append(
                {
                    "Issue Type": "Constant Column",
                    "Column": str(col_name),
                    "Severity": "Medium",
                    "Count": 1,
                    "Percentage": "100.0%",
                    "Description": f"Column contains only 1 unique value ('{first_val}') across all non-null rows.",
                }
            )
        elif unique_count > 1 and row_count >= 20:
            val_counts = non_null_series.value_counts()
            top_val = val_counts.index[0]
            top_count = int(val_counts.iloc[0])
            top_ratio = top_count / non_null_count
            if top_ratio >= 0.95:
                issues_list.append(
                    {
                        "Issue Type": "Near-Constant Column",
                        "Column": str(col_name),
                        "Severity": "Low",
                        "Count": top_count,
                        "Percentage": f"{top_ratio * 100:.1f}%",
                        "Description": f"Near-constant column: '{top_val}' accounts for {top_count} rows ({top_ratio * 100:.1f}%).",
                    }
                )

        # C. High Cardinality Detection
        if sem_type not in {"Identifier", "Text"} and pd.api.types.is_object_dtype(series):
            if unique_count >= 50 and uniqueness_ratio >= 0.8:
                high_cardinality_cols += 1
                issues_list.append(
                    {
                        "Issue Type": "High Cardinality",
                        "Column": str(col_name),
                        "Severity": "Info",
                        "Count": unique_count,
                        "Percentage": f"{uniqueness_ratio * 100:.1f}%",
                        "Description": f"Categorical/string column has high cardinality with {unique_count} distinct values.",
                    }
                )

        # D. Data-Type Inconsistencies & Invalid Domain Values
        # Check numerical column inconsistencies
        if sem_type == "Numerical" or (
            pd.api.types.is_numeric_dtype(series) and sem_type != "Identifier"
        ):
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                coerced = pd.to_numeric(non_null_series, errors="coerce")
                invalid_mask = coerced.isna()
                invalid_count = int(invalid_mask.sum())
                if invalid_count > 0:
                    inconsistent_cols.add(str(col_name))
                    invalid_pct = (invalid_count / non_null_count) * 100
                    sample_invalids = list(non_null_series[invalid_mask].unique()[:3])
                    sample_str = ", ".join([f"'{x}'" for x in sample_invalids])
                    issues_list.append(
                        {
                            "Issue Type": "Data-Type Inconsistency",
                            "Column": str(col_name),
                            "Severity": "High" if invalid_pct >= 10.0 else "Medium",
                            "Count": invalid_count,
                            "Percentage": f"{invalid_pct:.1f}%",
                            "Description": f"{invalid_count} non-numeric string value(s) (e.g., {sample_str}) found in numerical column.",
                        }
                    )
            elif pd.api.types.is_numeric_dtype(series):
                col_lower = str(col_name).lower()
                if "age" in col_lower:
                    neg_mask = non_null_series < 0
                    neg_count = int(neg_mask.sum())
                    if neg_count > 0:
                        inconsistent_cols.add(str(col_name))
                        neg_pct = (neg_count / non_null_count) * 100
                        neg_samples = list(non_null_series[neg_mask].unique()[:3])
                        issues_list.append(
                            {
                                "Issue Type": "Invalid Values",
                                "Column": str(col_name),
                                "Severity": "High",
                                "Count": neg_count,
                                "Percentage": f"{neg_pct:.1f}%",
                                "Description": f"{neg_count} negative age value(s) detected (e.g., {neg_samples}).",
                            }
                        )

        # Check Categorical casing/format inconsistencies
        if sem_type == "Categorical" or (
            pd.api.types.is_object_dtype(series) and sem_type not in {"Text", "Identifier"}
        ):
            raw_uniques = [str(val).strip() for val in non_null_series.unique()]
            norm_uniques = {val.lower() for val in raw_uniques}
            if len(norm_uniques) < len(raw_uniques):
                inconsistent_cols.add(str(col_name))
                diff_count = len(raw_uniques) - len(norm_uniques)
                issues_list.append(
                    {
                        "Issue Type": "Categorical Inconsistency",
                        "Column": str(col_name),
                        "Severity": "Low",
                        "Count": diff_count,
                        "Percentage": f"{(diff_count / len(raw_uniques) * 100):.1f}%",
                        "Description": f"Inconsistent capitalization/formatting detected among categories (e.g., {raw_uniques[:4]}).",
                    }
                )

        # Check Boolean inconsistent representations
        if sem_type == "Boolean":
            raw_str_vals = {str(v).strip() for v in non_null_series.unique()}
            norm_vals = {v.lower() for v in raw_str_vals}
            if len(raw_str_vals) > 2 or len(raw_str_vals) > len(norm_vals):
                inconsistent_cols.add(str(col_name))
                issues_list.append(
                    {
                        "Issue Type": "Inconsistent Boolean Formatting",
                        "Column": str(col_name),
                        "Severity": "Low",
                        "Count": len(raw_str_vals),
                        "Percentage": "-",
                        "Description": f"Mixed boolean representations found: {{{', '.join(sorted(list(raw_str_vals)))}}}.",
                    }
                )

        # Check Date parsing failures
        if sem_type == "Date/Datetime" and (
            pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        ):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coerced_dates = pd.to_datetime(
                    non_null_series, errors="coerce", format="mixed"
                )
            unparseable_mask = coerced_dates.isna()
            unparseable_count = int(unparseable_mask.sum())
            if unparseable_count > 0:
                inconsistent_cols.add(str(col_name))
                unparseable_pct = (unparseable_count / non_null_count) * 100
                unparseable_samples = list(non_null_series[unparseable_mask].unique()[:3])
                samples_str = ", ".join([f"'{x}'" for x in unparseable_samples])
                issues_list.append(
                    {
                        "Issue Type": "Date Parsing Failure",
                        "Column": str(col_name),
                        "Severity": "High",
                        "Count": unparseable_count,
                        "Percentage": f"{unparseable_pct:.1f}%",
                        "Description": f"{unparseable_count} value(s) failed to parse as valid dates (e.g., {samples_str}).",
                    }
                )

        # E. Numerical Outlier Detection (IQR Method)
        if (
            pd.api.types.is_numeric_dtype(series)
            and sem_type not in {"Identifier", "Boolean"}
            and unique_count > 2
        ):
            q1 = float(non_null_series.quantile(0.25))
            q3 = float(non_null_series.quantile(0.75))
            iqr = q3 - q1

            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = non_null_series[
                    (non_null_series < lower_bound) | (non_null_series > upper_bound)
                ]
                outlier_count = len(outliers)

                if outlier_count > 0:
                    outlier_cols += 1
                    outlier_pct = (outlier_count / non_null_count) * 100
                    o_severity = "Medium" if outlier_pct >= 10.0 else "Low"

                    issues_list.append(
                        {
                            "Issue Type": "Potential Outliers",
                            "Column": str(col_name),
                            "Severity": o_severity,
                            "Count": outlier_count,
                            "Percentage": f"{outlier_pct:.1f}%",
                            "Description": f"{outlier_count} potential outlier(s) outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}].",
                        }
                    )

                    outlier_list.append(
                        {
                            "Column": str(col_name),
                            "Outlier Count": outlier_count,
                            "Outlier %": f"{outlier_pct:.1f}%",
                            "Lower Bound": round(lower_bound, 2),
                            "Upper Bound": round(upper_bound, 2),
                            "Min Value": non_null_series.min(),
                            "Max Value": non_null_series.max(),
                        }
                    )

    # Summary metrics dictionary
    total_missing_cells = int(dataframe.isna().sum().sum())
    summary = {
        "total_issues_count": len(issues_list),
        "missing_value_columns_count": missing_value_cols,
        "total_missing_cells": total_missing_cells,
        "duplicate_rows_count": duplicate_rows_count,
        "outlier_columns_count": outlier_cols,
        "high_cardinality_columns_count": high_cardinality_cols,
        "empty_columns_count": empty_cols,
        "inconsistent_columns_count": len(inconsistent_cols),
    }

    issues_df = pd.DataFrame(
        issues_list,
        columns=[
            "Issue Type",
            "Column",
            "Severity",
            "Count",
            "Percentage",
            "Description",
        ],
    )

    outliers_df = pd.DataFrame(
        outlier_list,
        columns=[
            "Column",
            "Outlier Count",
            "Outlier %",
            "Lower Bound",
            "Upper Bound",
            "Min Value",
            "Max Value",
        ],
    )

    return {
        "summary": summary,
        "issues_df": issues_df,
        "outliers_df": outliers_df,
    }
