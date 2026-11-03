"""Automated Data Cleaning and Transformation Engine for AutoDataIQ.

Phase 11: Provides modular, transparent, and reproducible data cleaning operations
on an isolated copy of the uploaded DataFrame. The original DataFrame is NEVER modified.

Supported Operations:
1. Missing Value Handling:
   - Numerical: mean, median, constant replacement
   - Categorical: mode, constant replacement
   - Dropping empty columns (explicit user action)
   - Dropping missing value rows (explicit user action)
2. Duplicate Removal:
   - Detection and preview of duplicate rows
   - Removal preserving first occurrence (explicit user action)
3. Data-Type Conversion:
   - Safe conversion to numeric, date/datetime, boolean, string
   - Transparent tracking of unparseable/coerced values without silent loss
4. Text Standardization:
   - Whitespace trimming (leading/trailing)
   - Case transformation (lowercase, uppercase, titlecase)
5. Outlier Handling:
   - IQR-based detection (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
   - Optional clipping (winsorization) to IQR thresholds
6. Cleaning Log & Audit Trail:
   - Detailed record of every applied operation, affected rows/cells, and before/after stats
"""

from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd


def detect_cleaning_opportunities(
    dataframe: pd.DataFrame,
    semantic_types_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Inspect a DataFrame and identify available cleaning opportunities.

    Returns a structured dictionary of detected issues that can be addressed
    via cleaning operations (missing values, duplicates, outliers, text issues, conversions).
    """
    if dataframe is None or dataframe.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "empty_columns": [],
            "missing_value_columns": {},
            "duplicate_rows_count": 0,
            "text_standardization_candidates": {},
            "outlier_candidates": {},
            "type_conversion_candidates": {},
        }

    total_rows, total_cols = dataframe.shape

    # 1. Empty columns
    empty_cols = [
        col for col in dataframe.columns if dataframe[col].isna().all()
    ]

    # 2. Missing value columns
    missing_cols: dict[str, dict[str, Any]] = {}
    for col in dataframe.columns:
        null_count = int(dataframe[col].isna().sum())
        if null_count > 0:
            is_num = pd.api.types.is_numeric_dtype(dataframe[col])
            missing_cols[str(col)] = {
                "count": null_count,
                "percentage": round((null_count / total_rows * 100), 1),
                "is_numerical": is_num,
                "is_empty": null_count == total_rows,
            }

    # 3. Duplicate rows count
    dup_count = int(dataframe.duplicated().sum())

    # 4. Text standardization candidates
    text_candidates: dict[str, dict[str, Any]] = {}
    for col in dataframe.columns:
        if pd.api.types.is_object_dtype(dataframe[col]) or pd.api.types.is_string_dtype(dataframe[col]):
            non_null = dataframe[col].dropna()
            if not non_null.empty:
                str_series = non_null.astype(str)
                # Check for untrimmed whitespace
                untrimmed_count = int((str_series != str_series.str.strip()).sum())
                # Check for mixed casing
                has_mixed_case = bool(
                    (str_series.str.lower() != str_series).any()
                    and (str_series.str.upper() != str_series).any()
                )
                if untrimmed_count > 0 or has_mixed_case:
                    text_candidates[str(col)] = {
                        "untrimmed_count": untrimmed_count,
                        "has_mixed_case": has_mixed_case,
                        "sample_values": str_series.head(3).tolist(),
                    }

    # 5. Outlier candidates (IQR method)
    outlier_candidates: dict[str, dict[str, Any]] = {}
    for col in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            non_null = dataframe[col].dropna()
            if len(non_null) >= 4 and non_null.nunique() > 2:
                q1 = float(non_null.quantile(0.25))
                q3 = float(non_null.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lb = q1 - 1.5 * iqr
                    ub = q3 + 1.5 * iqr
                    outliers = non_null[(non_null < lb) | (non_null > ub)]
                    o_cnt = len(outliers)
                    if o_cnt > 0:
                        outlier_candidates[str(col)] = {
                            "outlier_count": o_cnt,
                            "percentage": round(o_cnt / len(non_null) * 100, 1),
                            "lower_bound": round(lb, 2),
                            "upper_bound": round(ub, 2),
                            "min_value": float(non_null.min()),
                            "max_value": float(non_null.max()),
                        }

    # 6. Type conversion candidates
    type_candidates: dict[str, dict[str, Any]] = {}
    for col in dataframe.columns:
        col_str = str(col)
        series = dataframe[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Test numeric conversion
            numeric_conv = pd.to_numeric(series, errors="coerce")
            num_valid = numeric_conv.notna().sum()
            if num_valid > 0 and (num_valid / len(series)) >= 0.8:
                type_candidates[col_str] = {
                    "current_type": str(dataframe[col].dtype),
                    "suggested_type": "numeric",
                    "parseable_count": int(num_valid),
                    "total_non_null": len(series),
                    "unparseable_count": int(len(series) - num_valid),
                }
                continue

            # Test date conversion
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                date_conv = pd.to_datetime(series, errors="coerce", format="mixed")
            date_valid = date_conv.notna().sum()
            if date_valid > 0 and (date_valid / len(series)) >= 0.8:
                type_candidates[col_str] = {
                    "current_type": str(dataframe[col].dtype),
                    "suggested_type": "datetime",
                    "parseable_count": int(date_valid),
                    "total_non_null": len(series),
                    "unparseable_count": int(len(series) - date_valid),
                }

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "empty_columns": empty_cols,
        "missing_value_columns": missing_cols,
        "duplicate_rows_count": dup_count,
        "text_standardization_candidates": text_candidates,
        "outlier_candidates": outlier_candidates,
        "type_conversion_candidates": type_candidates,
    }


def drop_empty_columns(
    dataframe: pd.DataFrame,
    columns_to_drop: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Drop columns that contain only missing values.

    Never modifies the original DataFrame.
    """
    df = dataframe.copy(deep=True)
    if columns_to_drop is None:
        # Detect all empty columns
        target_cols = [c for c in df.columns if df[c].isna().all()]
    else:
        # Verify specified columns are indeed in df and empty
        target_cols = [
            c for c in columns_to_drop if c in df.columns and df[c].isna().all()
        ]

    if not target_cols:
        return df, None

    df = df.drop(columns=target_cols)
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": "Drop Empty Columns",
        "columns": target_cols,
        "strategy": "drop_empty",
        "parameters": {},
        "affected_count": len(target_cols),
        "affected_type": "columns",
        "before_summary": f"{len(target_cols)} empty column(s)",
        "after_summary": f"Dropped {len(target_cols)} column(s)",
        "details": f"Removed {len(target_cols)} empty column(s): {', '.join(target_cols)}.",
        "warnings": [],
    }
    return df, log_entry


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Remove exact or subset-based duplicate rows, preserving first occurrence by default.

    Never modifies the original DataFrame.
    """
    df = dataframe.copy(deep=True)
    before_rows = len(df)
    valid_subset = [c for c in subset if c in df.columns] if subset else None

    dup_mask = df.duplicated(subset=valid_subset, keep=keep)
    dup_count = int(dup_mask.sum())

    if dup_count == 0:
        return df, None

    df = df.drop_duplicates(subset=valid_subset, keep=keep).reset_index(drop=True)
    after_rows = len(df)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": "Remove Duplicate Rows",
        "columns": valid_subset if valid_subset else ["All Columns"],
        "strategy": f"keep_{keep}",
        "parameters": {"keep": keep, "subset": valid_subset},
        "affected_count": dup_count,
        "affected_type": "rows",
        "before_summary": f"{before_rows:,} rows ({dup_count:,} duplicates)",
        "after_summary": f"{after_rows:,} rows (0 duplicates)",
        "details": f"Removed {dup_count:,} duplicate rows, preserving '{keep}' occurrence.",
        "warnings": [],
    }
    return df, log_entry


def drop_missing_rows(
    dataframe: pd.DataFrame,
    subset: list[str] | None = None,
    how: str = "any",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Drop rows containing missing values based on user selection.

    how: 'any' (drop if any selected column is NaN) or 'all' (drop if all selected columns are NaN).
    """
    df = dataframe.copy(deep=True)
    before_rows = len(df)
    valid_subset = [c for c in subset if c in df.columns] if subset else None

    target_df = df[valid_subset] if valid_subset else df
    if how == "all":
        drop_mask = target_df.isna().all(axis=1)
    else:
        drop_mask = target_df.isna().any(axis=1)

    dropped_count = int(drop_mask.sum())
    if dropped_count == 0:
        return df, None

    df = df[~drop_mask].reset_index(drop=True)
    after_rows = len(df)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": "Drop Missing Rows",
        "columns": valid_subset if valid_subset else ["All Columns"],
        "strategy": f"drop_{how}",
        "parameters": {"how": how, "subset": valid_subset},
        "affected_count": dropped_count,
        "affected_type": "rows",
        "before_summary": f"{before_rows:,} rows",
        "after_summary": f"{after_rows:,} rows ({dropped_count:,} rows dropped)",
        "details": f"Dropped {dropped_count:,} rows with missing values (how='{how}').",
        "warnings": [],
    }
    return df, log_entry


def impute_missing_values(
    dataframe: pd.DataFrame,
    column_strategies: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Impute missing values column by column according to user-selected strategies.

    Supported strategies:
    - Numerical: 'mean', 'median', 'constant' (with 'fill_value')
    - Categorical: 'mode', 'constant' (with 'fill_value')
    """
    df = dataframe.copy(deep=True)
    logs: list[dict[str, Any]] = []

    for col, config in column_strategies.items():
        if col not in df.columns:
            continue

        series = df[col]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        strategy = config.get("strategy", "constant")
        fill_val = None
        warning_msg = None

        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(series):
                fill_val = round(float(series.dropna().mean()), 4)
            else:
                warning_msg = f"Cannot calculate mean for non-numerical column '{col}'. Skipped."
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(series):
                fill_val = round(float(series.dropna().median()), 4)
            else:
                warning_msg = f"Cannot calculate median for non-numerical column '{col}'. Skipped."
        elif strategy == "mode":
            mode_vals = series.dropna().mode()
            if not mode_vals.empty:
                fill_val = mode_vals.iloc[0]
            else:
                warning_msg = f"No mode found for column '{col}'. Skipped."
        elif strategy == "constant":
            fill_val = config.get("fill_value")
            if fill_val is None:
                warning_msg = f"No constant value specified for column '{col}'. Skipped."

        if warning_msg:
            logs.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "operation": "Impute Missing Values (Skipped)",
                    "columns": [col],
                    "strategy": strategy,
                    "parameters": config,
                    "affected_count": 0,
                    "affected_type": "cells",
                    "before_summary": f"{missing_count:,} missing values",
                    "after_summary": f"{missing_count:,} missing values (unchanged)",
                    "details": warning_msg,
                    "warnings": [warning_msg],
                }
            )
            continue

        if fill_val is not None:
            df[col] = df[col].fillna(fill_val)
            after_missing = int(df[col].isna().sum())
            logs.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "operation": "Impute Missing Values",
                    "columns": [col],
                    "strategy": strategy,
                    "parameters": {"strategy": strategy, "fill_value": fill_val},
                    "affected_count": missing_count - after_missing,
                    "affected_type": "cells",
                    "before_summary": f"{missing_count:,} missing values",
                    "after_summary": f"{after_missing:,} missing values",
                    "details": f"Imputed {missing_count:,} missing values in '{col}' using {strategy} (value: {fill_val}).",
                    "warnings": [],
                }
            )

    return df, logs


def convert_column_types(
    dataframe: pd.DataFrame,
    type_conversions: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Safely convert selected columns to appropriate data types.

    Supported target types:
    - 'numeric' or 'float'
    - 'integer'
    - 'datetime'
    - 'boolean'
    - 'string'
    """
    df = dataframe.copy(deep=True)
    logs: list[dict[str, Any]] = []

    for col, target_type in type_conversions.items():
        if col not in df.columns:
            continue

        series = df[col]
        before_dtype = str(series.dtype)
        before_null = int(series.isna().sum())
        total_non_null = int(series.notna().sum())

        try:
            if target_type in {"numeric", "float"}:
                converted = pd.to_numeric(series, errors="coerce")
                after_null = int(converted.isna().sum())
                coerced_count = after_null - before_null

                df[col] = converted
                warnings_list = []
                if coerced_count > 0:
                    warnings_list.append(
                        f"{coerced_count} unparseable value(s) were coerced to null."
                    )

                logs.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "operation": "Convert Data Type",
                        "columns": [col],
                        "strategy": f"to_{target_type}",
                        "parameters": {"target_type": target_type},
                        "affected_count": total_non_null - coerced_count,
                        "affected_type": "cells",
                        "before_summary": f"dtype: {before_dtype}",
                        "after_summary": f"dtype: {df[col].dtype} ({coerced_count} coerced)",
                        "details": f"Converted '{col}' from {before_dtype} to numeric.",
                        "warnings": warnings_list,
                    }
                )

            elif target_type == "integer":
                # First convert to numeric
                converted = pd.to_numeric(series, errors="coerce")
                after_null = int(converted.isna().sum())
                coerced_count = after_null - before_null

                # Use Int64 nullable integer type to avoid crash on nulls
                df[col] = converted.round().astype("Int64")
                warnings_list = []
                if coerced_count > 0:
                    warnings_list.append(
                        f"{coerced_count} unparseable value(s) were coerced to null."
                    )

                logs.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "operation": "Convert Data Type",
                        "columns": [col],
                        "strategy": "to_integer",
                        "parameters": {"target_type": target_type},
                        "affected_count": total_non_null - coerced_count,
                        "affected_type": "cells",
                        "before_summary": f"dtype: {before_dtype}",
                        "after_summary": f"dtype: Int64",
                        "details": f"Converted '{col}' from {before_dtype} to nullable integer (Int64).",
                        "warnings": warnings_list,
                    }
                )

            elif target_type in {"datetime", "date"}:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(series, errors="coerce", format="mixed")
                after_null = int(converted.isna().sum())
                coerced_count = after_null - before_null

                df[col] = converted
                warnings_list = []
                if coerced_count > 0:
                    warnings_list.append(
                        f"{coerced_count} unparseable date value(s) were coerced to null."
                    )

                logs.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "operation": "Convert Data Type",
                        "columns": [col],
                        "strategy": "to_datetime",
                        "parameters": {"target_type": target_type},
                        "affected_count": total_non_null - coerced_count,
                        "affected_type": "cells",
                        "before_summary": f"dtype: {before_dtype}",
                        "after_summary": f"dtype: datetime64[ns]",
                        "details": f"Converted '{col}' from {before_dtype} to datetime.",
                        "warnings": warnings_list,
                    }
                )

            elif target_type == "boolean":
                truthy = {"true", "1", "yes", "y", "t"}
                falsy = {"false", "0", "no", "n", "f"}

                def map_bool(val: Any) -> Any:
                    if pd.isna(val):
                        return None
                    v_str = str(val).strip().lower()
                    if v_str in truthy:
                        return True
                    if v_str in falsy:
                        return False
                    return None

                converted = series.map(map_bool).astype("boolean")
                after_null = int(converted.isna().sum())
                coerced_count = after_null - before_null

                df[col] = converted
                warnings_list = []
                if coerced_count > 0:
                    warnings_list.append(
                        f"{coerced_count} unrecognised boolean values coerced to null."
                    )

                logs.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "operation": "Convert Data Type",
                        "columns": [col],
                        "strategy": "to_boolean",
                        "parameters": {"target_type": target_type},
                        "affected_count": total_non_null - coerced_count,
                        "affected_type": "cells",
                        "before_summary": f"dtype: {before_dtype}",
                        "after_summary": f"dtype: boolean",
                        "details": f"Converted '{col}' from {before_dtype} to boolean.",
                        "warnings": warnings_list,
                    }
                )

            elif target_type == "string":
                df[col] = series.astype(str).replace({"nan": None, "None": None, "<NA>": None})
                logs.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "operation": "Convert Data Type",
                        "columns": [col],
                        "strategy": "to_string",
                        "parameters": {"target_type": target_type},
                        "affected_count": total_non_null,
                        "affected_type": "cells",
                        "before_summary": f"dtype: {before_dtype}",
                        "after_summary": f"dtype: string",
                        "details": f"Converted '{col}' from {before_dtype} to string.",
                        "warnings": [],
                    }
                )

        except Exception as e:
            logs.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "operation": "Convert Data Type (Failed)",
                    "columns": [col],
                    "strategy": f"to_{target_type}",
                    "parameters": {"target_type": target_type},
                    "affected_count": 0,
                    "affected_type": "cells",
                    "before_summary": f"dtype: {before_dtype}",
                    "after_summary": f"dtype: {before_dtype} (unchanged)",
                    "details": f"Failed conversion for '{col}': {str(e)}",
                    "warnings": [f"Conversion failed: {str(e)}"],
                }
            )

    return df, logs


def standardize_text(
    dataframe: pd.DataFrame,
    text_operations: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Standardize text in selected columns: trim whitespace, transform casing.

    text_operations: dict of col_name -> {'trim': True/False, 'casing': 'none'/'lower'/'upper'/'title'}
    """
    df = dataframe.copy(deep=True)
    logs: list[dict[str, Any]] = []

    for col, ops in text_operations.items():
        if col not in df.columns:
            continue

        series = df[col]
        trim = ops.get("trim", False)
        casing = ops.get("casing", "none")

        if not trim and casing == "none":
            continue

        def clean_val(v: Any) -> Any:
            if pd.isna(v):
                return v
            s = str(v)
            if trim:
                s = s.strip()
            if casing == "lower":
                s = s.lower()
            elif casing == "upper":
                s = s.upper()
            elif casing == "title":
                s = s.title()
            return s

        before_vals = series.copy()
        df[col] = series.apply(clean_val)
        after_vals = df[col]

        # Calculate affected count safely
        changed_count = int((before_vals.dropna() != after_vals.dropna()).sum())

        actions_done = []
        if trim:
            actions_done.append("whitespace trimmed")
        if casing != "none":
            actions_done.append(f"cased to {casing}")

        logs.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operation": "Standardize Text",
                "columns": [col],
                "strategy": "text_cleaning",
                "parameters": ops,
                "affected_count": changed_count,
                "affected_type": "cells",
                "before_summary": f"Unstandardized text in '{col}'",
                "after_summary": f"{changed_count:,} cell(s) modified",
                "details": f"Applied {', '.join(actions_done)} on '{col}' ({changed_count:,} values changed).",
                "warnings": [],
            }
        )

    return df, logs


def clip_outliers(
    dataframe: pd.DataFrame,
    outlier_columns: list[str] | dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Clip numerical values to IQR bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

    Does NOT remove rows. Clips extreme values to boundary thresholds.
    """
    df = dataframe.copy(deep=True)
    logs: list[dict[str, Any]] = []

    # Normalize input
    targets: dict[str, dict[str, float] | None] = {}
    if isinstance(outlier_columns, list):
        targets = {col: None for col in outlier_columns}
    elif isinstance(outlier_columns, dict):
        targets = outlier_columns

    for col, custom_bounds in targets.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col].dropna()
        if len(series) < 4:
            continue

        if custom_bounds and "lower" in custom_bounds and "upper" in custom_bounds:
            lb = custom_bounds["lower"]
            ub = custom_bounds["upper"]
        else:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lb = q1 - 1.5 * iqr
            ub = q3 + 1.5 * iqr

        below_count = int((df[col] < lb).sum())
        above_count = int((df[col] > ub).sum())
        total_clipped = below_count + above_count

        if total_clipped == 0:
            continue

        df[col] = df[col].clip(lower=lb, upper=ub)

        logs.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operation": "Clip Outliers (IQR)",
                "columns": [col],
                "strategy": "iqr_clipping",
                "parameters": {"lower_bound": round(lb, 4), "upper_bound": round(ub, 4)},
                "affected_count": total_clipped,
                "affected_type": "cells",
                "before_summary": f"{total_clipped:,} values outside [{lb:.2f}, {ub:.2f}]",
                "after_summary": f"All values clamped to [{lb:.2f}, {ub:.2f}]",
                "details": f"Clamped {total_clipped:,} outlier(s) in '{col}' ({below_count} below {lb:.2f}, {above_count} above {ub:.2f}).",
                "warnings": [
                    "Outlier clipping modifies values to boundary limits. Verify domain validity."
                ],
            }
        )

    return df, logs


def apply_cleaning_pipeline(
    dataframe: pd.DataFrame,
    cleaning_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Execute the full data cleaning pipeline sequentially on a DataFrame copy.

    Order of operations:
    1. Drop Empty Columns
    2. Remove Duplicate Rows
    3. Drop Missing Rows
    4. Impute Missing Values
    5. Convert Data Types
    6. Standardize Text
    7. Clip Outliers

    Returns:
    - cleaned_df: Cleaned Pandas DataFrame
    - cleaning_log: List of all applied operation log entries
    - summary: Summary statistics comparing before vs after
    """
    if dataframe is None or dataframe.empty:
        return (
            dataframe.copy() if dataframe is not None else pd.DataFrame(),
            [],
            {
                "before_rows": 0,
                "after_rows": 0,
                "before_cols": 0,
                "after_cols": 0,
                "before_missing": 0,
                "after_missing": 0,
                "before_duplicates": 0,
                "after_duplicates": 0,
                "operations_applied_count": 0,
            },
        )

    # STRICT REQUIREMENT: Always start from a deep copy
    cleaned_df = dataframe.copy(deep=True)
    all_logs: list[dict[str, Any]] = []

    before_rows, before_cols = cleaned_df.shape
    before_missing = int(cleaned_df.isna().sum().sum())
    before_duplicates = int(cleaned_df.duplicated().sum())

    # 1. Drop Empty Columns
    if cleaning_config.get("drop_empty_columns", False):
        cols_to_drop = cleaning_config.get("empty_columns_to_drop")
        cleaned_df, log = drop_empty_columns(cleaned_df, cols_to_drop)
        if log:
            all_logs.append(log)

    # 2. Remove Duplicate Rows
    if cleaning_config.get("remove_duplicates", False):
        dup_subset = cleaning_config.get("duplicate_subset")
        dup_keep = cleaning_config.get("duplicate_keep", "first")
        cleaned_df, log = remove_duplicate_rows(
            cleaned_df, subset=dup_subset, keep=dup_keep
        )
        if log:
            all_logs.append(log)

    # 3. Drop Missing Rows
    if cleaning_config.get("drop_missing_rows", False):
        subset = cleaning_config.get("missing_rows_subset")
        how = cleaning_config.get("missing_rows_how", "any")
        cleaned_df, log = drop_missing_rows(cleaned_df, subset=subset, how=how)
        if log:
            all_logs.append(log)

    # 4. Impute Missing Values
    if "impute_missing" in cleaning_config and cleaning_config["impute_missing"]:
        impute_strategies = cleaning_config["impute_missing"]
        cleaned_df, logs = impute_missing_values(cleaned_df, impute_strategies)
        all_logs.extend(logs)

    # 5. Convert Data Types
    if "convert_types" in cleaning_config and cleaning_config["convert_types"]:
        conversions = cleaning_config["convert_types"]
        cleaned_df, logs = convert_column_types(cleaned_df, conversions)
        all_logs.extend(logs)

    # 6. Standardize Text
    if "standardize_text" in cleaning_config and cleaning_config["standardize_text"]:
        text_ops = cleaning_config["standardize_text"]
        cleaned_df, logs = standardize_text(cleaned_df, text_ops)
        all_logs.extend(logs)

    # 7. Clip Outliers
    if "clip_outliers" in cleaning_config and cleaning_config["clip_outliers"]:
        outlier_cols = cleaning_config["clip_outliers"]
        cleaned_df, logs = clip_outliers(cleaned_df, outlier_cols)
        all_logs.extend(logs)

    after_rows, after_cols = cleaned_df.shape
    after_missing = int(cleaned_df.isna().sum().sum())
    after_duplicates = int(cleaned_df.duplicated().sum())

    summary = {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "before_cols": before_cols,
        "after_cols": after_cols,
        "before_missing": before_missing,
        "after_missing": after_missing,
        "before_duplicates": before_duplicates,
        "after_duplicates": after_duplicates,
        "operations_applied_count": len(all_logs),
    }

    return cleaned_df, all_logs, summary
