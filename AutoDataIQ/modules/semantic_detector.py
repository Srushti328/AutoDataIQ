"""Semantic data type detector module for AutoDataIQ.

Determines what a column represents (Numerical, Categorical, Date/Datetime,
Boolean, Identifier, Text) using rule-based analysis on Pandas dtypes,
column names, uniqueness, sample values, and parsing patterns.
"""

import re
from typing import Any

import pandas as pd

# Keywords for column name heuristic matching
ID_KEYWORDS = {
    "id",
    "uuid",
    "guid",
    "code",
    "key",
    "identifier",
    "number",
    "no",
    "num",
    "pk",
    "fk",
}

DATE_KEYWORDS = {
    "date",
    "time",
    "dob",
    "created",
    "updated",
    "timestamp",
    "year",
    "month",
    "day",
    "birth",
    "expiry",
    "due",
}

TEXT_KEYWORDS = {
    "description",
    "comment",
    "comments",
    "review",
    "notes",
    "note",
    "address",
    "summary",
    "feedback",
    "text",
    "message",
    "bio",
    "details",
}

BOOLEAN_KEYWORDS = {
    "is",
    "has",
    "flag",
    "active",
    "enabled",
    "verified",
    "selected",
    "check",
    "valid",
}

BOOLEAN_VALUE_SET = {"true", "false", "t", "f", "1", "0", "1.0", "0.0", "yes", "no", "y", "n"}


def detect_column_semantic_type(series: pd.Series, column_name: str) -> dict[str, Any]:
    """Detect the semantic type of a single Pandas Series / column.

    Returns a dictionary with keys: Column, Pandas Type, Semantic Type, Confidence, Reason.
    """
    col_str_name = str(column_name).lower().strip()
    pandas_dtype_str = str(series.dtype)

    # Exclude missing values for analysis
    non_null_series = series.dropna()
    non_null_count = len(non_null_series)

    # Edge Case: Completely empty column
    if non_null_count == 0:
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Unknown / Empty",
            "Confidence": "Low",
            "Reason": "All values in this column are missing (null).",
        }

    unique_count = int(non_null_series.nunique())
    uniqueness_ratio = unique_count / non_null_count if non_null_count > 0 else 0.0

    # -------------------------------------------------------------
    # 1. Explicit Pandas Datetime or Date Name Pattern + Parseable
    # -------------------------------------------------------------
    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Date/Datetime",
            "Confidence": "High",
            "Reason": f"Native datetime data type ({pandas_dtype_str}).",
        }

    # -------------------------------------------------------------
    # 2. Boolean Detection
    # -------------------------------------------------------------
    if pd.api.types.is_bool_dtype(series):
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Boolean",
            "Confidence": "High",
            "Reason": "Native boolean data type.",
        }

    # Check value-based boolean for string / numeric columns
    str_unique_vals = {str(val).strip().lower() for val in non_null_series.unique()}
    if str_unique_vals.issubset(BOOLEAN_VALUE_SET) and len(str_unique_vals) <= 2:
        val_str = ", ".join(sorted(list(str_unique_vals)))
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Boolean",
            "Confidence": "High",
            "Reason": f"Contains boolean-like values: {{{val_str}}}.",
        }

    # -------------------------------------------------------------
    # 3. Identifier Detection
    # -------------------------------------------------------------
    # Check if column name suggests an identifier
    has_id_keyword = (
        any(
            re.search(r"(^|_|-)" + re.escape(kw) + r"($|_|-)", col_str_name)
            for kw in ID_KEYWORDS
        )
        or col_str_name.endswith("id")
        or col_str_name.startswith("id_")
        or col_str_name == "id"
    )

    # High uniqueness + ID name pattern
    if has_id_keyword and (uniqueness_ratio >= 0.7 or unique_count == non_null_count):
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Identifier",
            "Confidence": "High",
            "Reason": f"Column name contains ID pattern with high uniqueness ratio ({uniqueness_ratio:.1%}).",
        }

    if has_id_keyword and uniqueness_ratio >= 0.5:
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Identifier",
            "Confidence": "High" if uniqueness_ratio >= 0.8 else "Medium",
            "Reason": f"ID-like column name with {unique_count} distinct identifiers.",
        }

    # 100% unique values for string or integer column with non-null count >= 5
    if (
        uniqueness_ratio == 1.0
        and non_null_count >= 5
        and (pd.api.types.is_object_dtype(series) or pd.api.types.is_integer_dtype(series))
    ):
        if pd.api.types.is_object_dtype(series):
            avg_len = float(non_null_series.astype(str).str.len().mean())
            if avg_len <= 25 and not any(kw in col_str_name for kw in TEXT_KEYWORDS):
                return {
                    "Column": str(column_name),
                    "Pandas Type": pandas_dtype_str,
                    "Semantic Type": "Identifier",
                    "Confidence": "Medium",
                    "Reason": "All non-null values are unique short strings (potential key/code).",
                }
        elif pd.api.types.is_integer_dtype(series):
            if has_id_keyword:
                return {
                    "Column": str(column_name),
                    "Pandas Type": pandas_dtype_str,
                    "Semantic Type": "Identifier",
                    "Confidence": "High",
                    "Reason": "Unique integer identifier column.",
                }

    # -------------------------------------------------------------
    # 4. Date / Datetime Parsing (String/Object or Date Name)
    # -------------------------------------------------------------
    has_date_keyword = (
        any(
            re.search(r"(^|_|-)" + re.escape(kw) + r"($|_|-)", col_str_name)
            for kw in DATE_KEYWORDS
        )
        or col_str_name.endswith("date")
        or col_str_name.startswith("date_")
    )

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or has_date_keyword:
        if not pd.api.types.is_numeric_dtype(series) or has_date_keyword:
            sample = non_null_series.head(50)
            parsed_count = 0
            for val in sample:
                val_str = str(val).strip()
                if len(val_str) >= 4 and any(char.isdigit() for char in val_str):
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            parsed_val = pd.to_datetime(val_str, errors="coerce", format="mixed")
                        if pd.notna(parsed_val):
                            parsed_count += 1
                    except (ValueError, TypeError, OverflowError):
                        pass

            if len(sample) > 0 and (parsed_count / len(sample)) >= 0.7:
                confidence = "High" if has_date_keyword or (parsed_count / len(sample)) >= 0.9 else "Medium"
                return {
                    "Column": str(column_name),
                    "Pandas Type": pandas_dtype_str,
                    "Semantic Type": "Date/Datetime",
                    "Confidence": confidence,
                    "Reason": f"Values parse as dates/timestamps ({parsed_count}/{len(sample)} sample values parsed).",
                }

    # Special check for Numeric Year column (e.g., year=2024)
    if pd.api.types.is_numeric_dtype(series) and ("year" in col_str_name or "yr" in col_str_name):
        min_val = non_null_series.min()
        max_val = non_null_series.max()
        if 1800 <= min_val <= 2100 and 1800 <= max_val <= 2100:
            return {
                "Column": str(column_name),
                "Pandas Type": pandas_dtype_str,
                "Semantic Type": "Date/Datetime",
                "Confidence": "High",
                "Reason": f"Numeric column represents calendar years ({min_val} to {max_val}).",
            }

    # -------------------------------------------------------------
    # 5. Numerical Detection
    # -------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(series):
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Numerical",
            "Confidence": "High",
            "Reason": f"Numeric data type ({pandas_dtype_str}) with quantitative values.",
        }

    # String column that can be coerced to numeric (e.g. "$100", "25.5%")
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = non_null_series.head(50)
        cleaned_sample = sample.astype(str).str.replace(r"[\$,%]", "", regex=True).str.strip()
        numeric_converted = pd.to_numeric(cleaned_sample, errors="coerce")
        if len(sample) > 0 and (numeric_converted.notna().sum() / len(sample)) >= 0.8:
            return {
                "Column": str(column_name),
                "Pandas Type": pandas_dtype_str,
                "Semantic Type": "Numerical",
                "Confidence": "Medium",
                "Reason": "String values can be successfully parsed as numeric values.",
            }

    # -------------------------------------------------------------
    # 6. Text vs Categorical Detection (Object / String Dtypes)
    # -------------------------------------------------------------
    has_text_keyword = any(kw in col_str_name for kw in TEXT_KEYWORDS)
    avg_char_length = float(non_null_series.astype(str).str.len().mean()) if non_null_count > 0 else 0.0
    max_char_length = int(non_null_series.astype(str).str.len().max()) if non_null_count > 0 else 0

    # Free-form long text signals
    if (
        has_text_keyword
        or avg_char_length > 30
        or max_char_length > 100
        or (uniqueness_ratio > 0.6 and avg_char_length > 15)
    ):
        confidence = "High" if (avg_char_length > 30 or has_text_keyword) else "Medium"
        return {
            "Column": str(column_name),
            "Pandas Type": pandas_dtype_str,
            "Semantic Type": "Text",
            "Confidence": confidence,
            "Reason": f"Free-form text with average length of {avg_char_length:.1f} characters.",
        }

    # Categorical: Discrete set of values
    confidence = "High" if (unique_count <= 20 or uniqueness_ratio < 0.2) else "Medium"
    return {
        "Column": str(column_name),
        "Pandas Type": pandas_dtype_str,
        "Semantic Type": "Categorical",
        "Confidence": confidence,
        "Reason": f"String data type with {unique_count} distinct categories ({uniqueness_ratio:.1%} unique ratio).",
    }


def detect_semantic_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Analyze all columns in a DataFrame and return a structured DataFrame.

    Returns columns: Column, Pandas Type, Semantic Type, Confidence, Reason.
    """
    results: list[dict[str, Any]] = []

    for position, column_name in enumerate(dataframe.columns):
        series = dataframe.iloc[:, position]
        col_result = detect_column_semantic_type(series, column_name)
        results.append(col_result)

    return pd.DataFrame(
        results,
        columns=["Column", "Pandas Type", "Semantic Type", "Confidence", "Reason"],
    )
