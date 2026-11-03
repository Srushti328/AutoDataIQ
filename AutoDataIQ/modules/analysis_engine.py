"""Adaptive Data Analysis Engine module for AutoDataIQ.

Determines appropriate exploratory analysis modules dynamically based on the semantic
and structural characteristics of the uploaded dataset.

Modules implemented:
- Dataset Overview & Column Breakdown
- Numerical Analysis (Count, Mean, Median, Min, Max, Std Dev, IQR)
- Categorical Analysis (Uniques, Frequencies, Top Categories)
- Date / Datetime Analysis (Min/Max Date, Year/Month Trends)
- Boolean Analysis (True/False Distributions)
- Text Analysis (String Length Statistics)
- Identifier Analysis (Uniqueness Metrics)
- Pairwise Pearson Correlation Analysis
- Categorical Group Comparison (Categorical x Numerical)
- Time-based Trend Summary (Date x Numerical)

Strictly read-only: does not modify or transform the input DataFrame.
"""

from typing import Any
import pandas as pd

from modules.quality_engine import analyze_data_quality
from modules.semantic_detector import detect_semantic_types


def classify_columns_semantically(
    dataframe: pd.DataFrame,
    semantic_types_df: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Group columns into semantic lists (numerical, categorical, date, boolean, text, identifier)."""
    if semantic_types_df is None or semantic_types_df.empty:
        semantic_types_df = detect_semantic_types(dataframe)

    semantic_map = dict(
        zip(semantic_types_df["Column"], semantic_types_df["Semantic Type"])
    )

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    date_cols: list[str] = []
    boolean_cols: list[str] = []
    identifier_cols: list[str] = []
    text_cols: list[str] = []
    unknown_cols: list[str] = []

    for col in dataframe.columns:
        col_str = str(col)
        stype = semantic_map.get(col_str, "Unknown")
        series = dataframe[col]

        if stype == "Identifier":
            identifier_cols.append(col_str)
        elif stype == "Date/Datetime" or pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col_str)
        elif stype == "Boolean" or pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col_str)
        elif stype == "Numerical" or (
            pd.api.types.is_numeric_dtype(series) and stype not in {"Identifier", "Boolean"}
        ):
            numerical_cols.append(col_str)
        elif stype == "Text":
            text_cols.append(col_str)
        elif stype == "Categorical" or pd.api.types.is_object_dtype(series):
            categorical_cols.append(col_str)
        else:
            unknown_cols.append(col_str)

    return {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "date": date_cols,
        "boolean": boolean_cols,
        "identifier": identifier_cols,
        "text": text_cols,
        "unknown": unknown_cols,
    }


def get_analysis_plan(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Generate an adaptive analysis plan identifying available analysis modules."""
    if dataframe is None or dataframe.empty:
        return []

    row_count, col_count = dataframe.shape
    num_len = len(classified_cols["numerical"])
    cat_len = len(classified_cols["categorical"])
    date_len = len(classified_cols["date"])
    bool_len = len(classified_cols["boolean"])
    text_len = len(classified_cols["text"])
    id_len = len(classified_cols["identifier"])

    plan = [
        {
            "key": "dataset_summary",
            "name": "Dataset Overview & Structure",
            "available": True,
            "reason": f"Applicable for all datasets ({row_count} rows, {col_count} columns).",
        },
        {
            "key": "numerical_analysis",
            "name": "Numerical Univariate Analysis",
            "available": num_len >= 1,
            "reason": f"Dataset contains {num_len} suitable numerical column(s)."
            if num_len >= 1
            else "No suitable numerical columns available (Identifiers excluded).",
        },
        {
            "key": "categorical_analysis",
            "name": "Categorical Frequency Analysis",
            "available": cat_len >= 1,
            "reason": f"Dataset contains {cat_len} categorical column(s)."
            if cat_len >= 1
            else "No categorical columns found.",
        },
        {
            "key": "date_analysis",
            "name": "Date & Temporal Range Analysis",
            "available": date_len >= 1,
            "reason": f"Dataset contains {date_len} date/datetime column(s)."
            if date_len >= 1
            else "No date or timestamp columns found.",
        },
        {
            "key": "boolean_analysis",
            "name": "Boolean Flag Distribution",
            "available": bool_len >= 1,
            "reason": f"Dataset contains {bool_len} boolean column(s)."
            if bool_len >= 1
            else "No boolean columns found.",
        },
        {
            "key": "text_analysis",
            "name": "Free-Form Text Metrics",
            "available": text_len >= 1,
            "reason": f"Dataset contains {text_len} text column(s)."
            if text_len >= 1
            else "No free-form text columns found.",
        },
        {
            "key": "identifier_analysis",
            "name": "Identifier Uniqueness Analysis",
            "available": id_len >= 1,
            "reason": f"Dataset contains {id_len} identifier column(s)."
            if id_len >= 1
            else "No identifier columns found.",
        },
        {
            "key": "correlation_analysis",
            "name": "Pearson Correlation Matrix",
            "available": num_len >= 2,
            "reason": f"Dataset contains {num_len} suitable numerical columns (minimum 2 required)."
            if num_len >= 2
            else f"Requires at least 2 numerical columns (found {num_len}).",
        },
        {
            "key": "group_comparison",
            "name": "Categorical Group Comparison",
            "available": cat_len >= 1 and num_len >= 1,
            "reason": f"Found {cat_len} categorical and {num_len} numerical column(s)."
            if (cat_len >= 1 and num_len >= 1)
            else "Requires at least 1 categorical and 1 numerical column.",
        },
        {
            "key": "time_trend",
            "name": "Time-based Numerical Trend Summary",
            "available": date_len >= 1 and num_len >= 1,
            "reason": f"Found {date_len} date and {num_len} numerical column(s)."
            if (date_len >= 1 and num_len >= 1)
            else "Requires at least 1 date column and 1 numerical column.",
        },
    ]

    return plan


def analyze_numerical_columns(
    dataframe: pd.DataFrame, numerical_cols: list[str]
) -> pd.DataFrame:
    """Compute summary statistics (Count, Mean, Median, Std Dev, Min, Max, Q1, Q3, IQR) for numerical columns."""
    rows: list[dict[str, Any]] = []
    for col in numerical_cols:
        series = dataframe[col].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        rows.append(
            {
                "Column": col,
                "Count": len(series),
                "Mean": round(float(series.mean()), 2),
                "Median": round(float(series.median()), 2),
                "Std Dev": round(float(series.std()), 2) if len(series) > 1 else 0.0,
                "Min": series.min(),
                "Max": series.max(),
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(q3 - q1, 2),
            }
        )
    return pd.DataFrame(rows)


def analyze_categorical_columns(
    dataframe: pd.DataFrame, categorical_cols: list[str], top_n: int = 10
) -> dict[str, Any]:
    """Compute frequency breakdowns for categorical columns."""
    cat_summary: list[dict[str, Any]] = []
    breakdowns: dict[str, pd.DataFrame] = {}

    for col in categorical_cols:
        series = dataframe[col].dropna()
        total_non_null = len(series)
        unique_cnt = series.nunique()
        val_counts = series.value_counts()

        if val_counts.empty:
            top_val = None
            top_freq = 0
        else:
            top_val = val_counts.index[0]
            top_freq = int(val_counts.iloc[0])

        cat_summary.append(
            {
                "Column": col,
                "Unique Values": unique_cnt,
                "Most Frequent Value": top_val,
                "Frequency": top_freq,
                "Top Category %": round((top_freq / total_non_null * 100), 1)
                if total_non_null > 0
                else 0.0,
            }
        )

        # Top N Breakdown DataFrame
        top_df = val_counts.head(top_n).reset_index()
        top_df.columns = ["Category", "Count"]
        top_df["Percentage"] = (
            (top_df["Count"] / total_non_null * 100).round(1)
            if total_non_null > 0
            else 0.0
        )
        breakdowns[col] = top_df

    return {
        "summary_df": pd.DataFrame(cat_summary),
        "breakdowns": breakdowns,
    }


def analyze_date_columns(
    dataframe: pd.DataFrame, date_cols: list[str]
) -> dict[str, Any]:
    """Compute temporal range and time distribution for date columns."""
    date_summary: list[dict[str, Any]] = []
    distributions: dict[str, pd.DataFrame] = {}

    import warnings

    for col in date_cols:
        series = dataframe[col].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        valid_parsed = parsed.dropna()

        if valid_parsed.empty:
            continue

        min_date = valid_parsed.min()
        max_date = valid_parsed.max()

        date_summary.append(
            {
                "Column": col,
                "Valid Dates Count": len(valid_parsed),
                "Min Date": str(min_date.date()) if hasattr(min_date, "date") else str(min_date),
                "Max Date": str(max_date.date()) if hasattr(max_date, "date") else str(max_date),
            }
        )

        # Yearly breakdown
        years = valid_parsed.dt.year.value_counts().sort_index().reset_index()
        years.columns = ["Year", "Record Count"]
        distributions[col] = years

    return {
        "summary_df": pd.DataFrame(date_summary),
        "distributions": distributions,
    }


def analyze_boolean_columns(
    dataframe: pd.DataFrame, boolean_cols: list[str]
) -> pd.DataFrame:
    """Compute True/False distribution metrics for boolean columns."""
    rows: list[dict[str, Any]] = []
    truthy_vals = {"true", "1", "t", "yes", "y"}

    for col in boolean_cols:
        series = dataframe[col]
        non_null_series = series.dropna()
        total_valid = len(non_null_series)
        missing_cnt = len(series) - total_valid

        true_count = sum(
            1 for val in non_null_series if str(val).strip().lower() in truthy_vals
        )
        false_count = total_valid - true_count

        rows.append(
            {
                "Column": col,
                "True Count": true_count,
                "True %": round((true_count / total_valid * 100), 1)
                if total_valid > 0
                else 0.0,
                "False Count": false_count,
                "False %": round((false_count / total_valid * 100), 1)
                if total_valid > 0
                else 0.0,
                "Missing Count": missing_cnt,
            }
        )

    return pd.DataFrame(rows)


def analyze_text_columns(
    dataframe: pd.DataFrame, text_cols: list[str]
) -> pd.DataFrame:
    """Compute length statistics for free-form text columns."""
    rows: list[dict[str, Any]] = []
    for col in text_cols:
        series = dataframe[col].dropna().astype(str)
        if series.empty:
            continue
        lengths = series.str.len()
        rows.append(
            {
                "Column": col,
                "Non-Null Count": len(series),
                "Unique Count": series.nunique(),
                "Min Length": int(lengths.min()),
                "Max Length": int(lengths.max()),
                "Avg Length": round(float(lengths.mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def analyze_identifier_columns(
    dataframe: pd.DataFrame, identifier_cols: list[str]
) -> pd.DataFrame:
    """Compute uniqueness indicators for identifier columns."""
    rows: list[dict[str, Any]] = []
    for col in identifier_cols:
        series = dataframe[col].dropna()
        non_null_cnt = len(series)
        uniq_cnt = series.nunique()
        uniq_ratio = (uniq_cnt / non_null_cnt * 100) if non_null_cnt > 0 else 0.0

        rows.append(
            {
                "Column": col,
                "Non-Null Count": non_null_cnt,
                "Unique Count": uniq_cnt,
                "Uniqueness %": round(uniq_ratio, 1),
                "Status": "100% Unique Key"
                if uniq_ratio == 100.0
                else "High Uniqueness Identifier",
            }
        )
    return pd.DataFrame(rows)


def analyze_correlation(
    dataframe: pd.DataFrame, numerical_cols: list[str]
) -> pd.DataFrame:
    """Compute Pearson correlation matrix for suitable numerical columns."""
    if len(numerical_cols) < 2:
        return pd.DataFrame()
    corr_matrix = dataframe[numerical_cols].corr(method="pearson").round(3)
    return corr_matrix


def analyze_group_comparison(
    dataframe: pd.DataFrame, cat_col: str, num_col: str
) -> pd.DataFrame:
    """Compute group aggregations (Count, Mean, Median, Std Dev, Min, Max) for a categorical and numerical pair."""
    if cat_col not in dataframe.columns or num_col not in dataframe.columns:
        return pd.DataFrame()

    grouped = (
        dataframe.groupby(cat_col)[num_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    grouped.columns = [
        str(cat_col),
        "Count",
        "Mean",
        "Median",
        "Std Dev",
        "Min",
        "Max",
    ]

    grouped["Mean"] = grouped["Mean"].round(2)
    grouped["Median"] = grouped["Median"].round(2)
    grouped["Std Dev"] = grouped["Std Dev"].round(2).fillna(0.0)

    return grouped.sort_values(by="Count", ascending=False)


def analyze_time_trend(
    dataframe: pd.DataFrame, date_col: str, num_col: str | None = None
) -> pd.DataFrame:
    """Compute time-based monthly or yearly aggregations for a date column."""
    if date_col not in dataframe.columns:
        return pd.DataFrame()

    temp_df = pd.DataFrame()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_df["parsed_date"] = pd.to_datetime(
            dataframe[date_col], errors="coerce", format="mixed"
        )

    if num_col and num_col in dataframe.columns:
        temp_df[num_col] = pd.to_numeric(dataframe[num_col], errors="coerce")

    valid_mask = temp_df["parsed_date"].notna()
    valid_data = temp_df[valid_mask].copy()

    if valid_data.empty:
        return pd.DataFrame()

    valid_data["Year-Month"] = valid_data["parsed_date"].dt.to_period("M").astype(str)

    if num_col and num_col in valid_data.columns:
        agg_df = (
            valid_data.groupby("Year-Month")[num_col]
            .agg(["count", "mean", "median", "sum"])
            .reset_index()
        )
        agg_df.columns = ["Year-Month", "Record Count", "Mean", "Median", "Sum"]
        agg_df["Mean"] = agg_df["Mean"].round(2)
        agg_df["Median"] = agg_df["Median"].round(2)
        agg_df["Sum"] = agg_df["Sum"].round(2)
    else:
        agg_df = (
            valid_data.groupby("Year-Month")
            .size()
            .reset_index(name="Record Count")
        )

    return agg_df.sort_values(by="Year-Month")


def run_adaptive_analysis(
    dataframe: pd.DataFrame,
    semantic_types_df: pd.DataFrame | None = None,
    quality_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute complete adaptive exploratory analysis based on dataset structure.

    Returns:
    - plan: list of analysis availability dicts
    - classified_cols: dict of lists of column names
    - results: dict of calculated DataFrames / data structures
    - quality_warnings: dict mapping col_name -> warning message string
    """
    if dataframe is None or dataframe.empty:
        return {
            "plan": [],
            "classified_cols": {
                "numerical": [],
                "categorical": [],
                "date": [],
                "boolean": [],
                "identifier": [],
                "text": [],
                "unknown": [],
            },
            "results": {},
            "quality_warnings": {},
        }

    if semantic_types_df is None or semantic_types_df.empty:
        semantic_types_df = detect_semantic_types(dataframe)

    classified_cols = classify_columns_semantically(dataframe, semantic_types_df)
    plan = get_analysis_plan(dataframe, classified_cols)

    # Compile Quality Warnings from quality_report if available
    quality_warnings: dict[str, list[str]] = {}
    if quality_report and "issues_df" in quality_report and not quality_report["issues_df"].empty:
        for _, issue in quality_report["issues_df"].iterrows():
            col = str(issue["Column"])
            itype = str(issue["Issue Type"])
            desc = str(issue["Description"])
            if col != "Dataset-level":
                if col not in quality_warnings:
                    quality_warnings[col] = []
                quality_warnings[col].append(f"{itype}: {desc}")

    results: dict[str, Any] = {}

    # 1. Numerical Analysis
    if len(classified_cols["numerical"]) >= 1:
        results["numerical_analysis"] = analyze_numerical_columns(
            dataframe, classified_cols["numerical"]
        )

    # 2. Categorical Analysis
    if len(classified_cols["categorical"]) >= 1:
        results["categorical_analysis"] = analyze_categorical_columns(
            dataframe, classified_cols["categorical"]
        )

    # 3. Date Analysis
    if len(classified_cols["date"]) >= 1:
        results["date_analysis"] = analyze_date_columns(
            dataframe, classified_cols["date"]
        )

    # 4. Boolean Analysis
    if len(classified_cols["boolean"]) >= 1:
        results["boolean_analysis"] = analyze_boolean_columns(
            dataframe, classified_cols["boolean"]
        )

    # 5. Text Analysis
    if len(classified_cols["text"]) >= 1:
        results["text_analysis"] = analyze_text_columns(
            dataframe, classified_cols["text"]
        )

    # 6. Identifier Analysis
    if len(classified_cols["identifier"]) >= 1:
        results["identifier_analysis"] = analyze_identifier_columns(
            dataframe, classified_cols["identifier"]
        )

    # 7. Correlation Analysis
    if len(classified_cols["numerical"]) >= 2:
        results["correlation_analysis"] = analyze_correlation(
            dataframe, classified_cols["numerical"]
        )

    return {
        "plan": plan,
        "classified_cols": classified_cols,
        "results": results,
        "quality_warnings": quality_warnings,
    }
