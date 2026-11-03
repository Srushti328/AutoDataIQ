"""Intelligent Cleaning Recommendation Engine for AutoDataIQ.

Analyzes detected data quality issues from Phase 4 and generates structured,
actionable, read-only cleaning recommendations specifying:
- What should be cleaned
- Why it should be cleaned
- Recommended methods & proposed actions
- Expected effects if approved
- Priority level (High, Medium, Low, Info)

Strictly read-only: does not modify or transform the original DataFrame.
"""

from typing import Any
import pandas as pd

from modules.quality_engine import analyze_data_quality
from modules.semantic_detector import detect_semantic_types


def generate_cleaning_recommendations(
    dataframe: pd.DataFrame,
    quality_report: dict[str, Any] | None = None,
    semantic_types_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate intelligent cleaning recommendations based on dataset quality issues.

    Parameters:
    - dataframe: The input Pandas DataFrame (read-only).
    - quality_report: Output dict from quality_engine.analyze_data_quality().
    - semantic_types_df: Output DataFrame from semantic_detector.detect_semantic_types().

    Returns a dictionary containing:
    - summary: dict of recommendation counts (total, priorities, categories)
    - recommendations_df: pd.DataFrame of all recommendations
    - recommendations_list: list of dicts with detailed recommendation information
    """
    if dataframe is None or dataframe.empty:
        return {
            "summary": {
                "total_recommendations": 0,
                "priority_counts": {"High": 0, "Medium": 0, "Low": 0, "Info": 0},
                "category_counts": {
                    "Missing Values": 0,
                    "Duplicates": 0,
                    "Consistency": 0,
                    "Validity": 0,
                    "Structural": 0,
                    "Outliers": 0,
                },
            },
            "recommendations_df": pd.DataFrame(
                columns=[
                    "Column",
                    "Issue Type",
                    "Category",
                    "Priority",
                    "Recommended Action",
                    "Method",
                    "Reason",
                    "Current Situation",
                    "Proposed Action",
                    "Expected Effect",
                ]
            ),
            "recommendations_list": [],
        }

    # If quality report or semantic types are not provided, compute them
    if quality_report is None:
        quality_report = analyze_data_quality(dataframe, semantic_types_df)
    if semantic_types_df is None or semantic_types_df.empty:
        semantic_types_df = detect_semantic_types(dataframe)

    semantic_map = dict(
        zip(semantic_types_df["Column"], semantic_types_df["Semantic Type"])
    )

    issues_df: pd.DataFrame = quality_report.get("issues_df", pd.DataFrame())
    row_count, col_count = dataframe.shape

    recommendations: list[dict[str, Any]] = []

    # Iterate through all detected issues from Phase 4
    if not issues_df.empty:
        for _, issue in issues_df.iterrows():
            issue_type = issue["Issue Type"]
            col_name = str(issue["Column"])
            severity = issue["Severity"]
            count = issue["Count"]
            pct_str = issue["Percentage"]

            sem_type = semantic_map.get(col_name, "Unknown")
            series = dataframe[col_name] if col_name in dataframe.columns else None

            # -------------------------------------------------------------
            # 1. Missing Values & Empty Columns
            # -------------------------------------------------------------
            if issue_type == "Empty Column":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Missing Values",
                        "Priority": "High",
                        "Recommended Action": "Evaluate Column Removal",
                        "Method": "Drop Empty Column",
                        "Reason": "Column contains 100% missing values and provides no analytical data.",
                        "Current Situation": f"Column '{col_name}' has all {row_count} values missing (100.0% empty).",
                        "Proposed Action": f"Consider removing column '{col_name}' from downstream analysis after domain verification.",
                        "Expected Effect": f"Frees {row_count} empty cells and simplifies dataset schema without data loss.",
                    }
                )

            elif issue_type == "Missing Values":
                try:
                    missing_pct = float(str(pct_str).replace("%", "").strip())
                except ValueError:
                    missing_pct = 0.0

                if missing_pct > 50.0:
                    priority = "High"
                    rec_action = "Evaluate Column Retention / Imputation"
                    method = "Domain Review / Column Drop"
                    reason = f"High missingness ({missing_pct:.1f}%). Imputation may distort data distribution."
                    proposed = f"Review column '{col_name}' with domain experts to decide whether to drop or fill missing values."
                    effect = "Prevents bias introduced by imputing majority-missing values."
                elif missing_pct >= 20.0:
                    priority = "Medium"
                    rec_action = "Review & Impute Missing Values"
                    method = (
                        "Median Imputation"
                        if sem_type == "Numerical"
                        else (
                            "Mode / Unknown Category"
                            if sem_type == "Categorical"
                            else "Domain Imputation"
                        )
                    )
                    reason = f"Moderate missingness ({missing_pct:.1f}%). Imputation can preserve sample size."
                    proposed = f"Apply {method} to fill missing entries in '{col_name}'."
                    effect = f"Restores {count} missing value(s) to enable full row utilization in modeling."
                else:
                    priority = "Low" if missing_pct < 5.0 else "Medium"
                    is_numeric_col = (sem_type == "Numerical") or (
                        series is not None
                        and pd.api.types.is_numeric_dtype(series)
                        and sem_type != "Identifier"
                    )
                    if is_numeric_col:
                        rec_action = "Fill Missing Numerical Values"
                        method = "Median Imputation"
                        reason = f"Low missingness ({missing_pct:.1f}%). Median is robust against potential outliers."
                        proposed = f"Impute missing numerical values in '{col_name}' using column median."
                        effect = f"Replaces {count} missing value(s) with central median value."
                    elif sem_type == "Categorical":
                        rec_action = "Fill Missing Categorical Values"
                        method = "Mode Imputation or 'Unknown' Category"
                        reason = f"Low missingness ({missing_pct:.1f}%)."
                        proposed = f"Fill missing categorical entries in '{col_name}' using most frequent category or 'Unknown'."
                        effect = f"Standardizes {count} missing categorical record(s)."
                    elif sem_type == "Date/Datetime":
                        rec_action = "Review Invalid / Missing Dates"
                        method = "Date Validation & Source Review"
                        reason = "Date values should not be arbitrarily guessed or imputed with dummy dates."
                        proposed = f"Investigate source data for missing date entries in '{col_name}'."
                        effect = "Maintains temporal accuracy without introducing false timestamps."
                    elif sem_type == "Boolean":
                        rec_action = "Review Missing Boolean Values"
                        method = "Explicit Flag or Domain Rule"
                        reason = "Missing boolean values should not be assumed as False without domain context."
                        proposed = f"Review missing boolean records in '{col_name}'."
                        effect = "Prevents misinterpreting missing status as negative signal."
                    elif sem_type == "Text":
                        rec_action = "Insert Text Placeholder"
                        method = "Explicit Placeholder ('Not Specified')"
                        reason = "Free-form text fields benefit from explicit missingness labels."
                        proposed = f"Replace missing text entries in '{col_name}' with 'Not Specified'."
                        effect = f"Explicitly flags {count} unpopulated text field(s)."
                    else:
                        rec_action = "Impute Missing Values"
                        method = "Mode or Placeholder Imputation"
                        reason = f"Missingness of {missing_pct:.1f}% detected."
                        proposed = f"Review and fill missing entries in '{col_name}'."
                        effect = f"Fills {count} missing value(s)."

                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Missing Values",
                        "Priority": priority,
                        "Recommended Action": rec_action,
                        "Method": method,
                        "Reason": reason,
                        "Current Situation": f"Column '{col_name}' has {count} missing value(s) ({missing_pct:.1f}% of column).",
                        "Proposed Action": proposed,
                        "Expected Effect": effect,
                    }
                )

            # -------------------------------------------------------------
            # 2. Duplicate Rows
            # -------------------------------------------------------------
            elif issue_type == "Duplicate Rows":
                try:
                    dup_pct = float(str(pct_str).replace("%", "").strip())
                except ValueError:
                    dup_pct = 0.0

                priority = "High" if dup_pct >= 20.0 else ("Medium" if dup_pct >= 5.0 else "Low")
                recommendations.append(
                    {
                        "Column": "Dataset-level",
                        "Issue Type": issue_type,
                        "Category": "Duplicates",
                        "Priority": priority,
                        "Recommended Action": "Review & Remove Exact Duplicate Rows",
                        "Method": "Exact Deduplication (keep first)",
                        "Reason": "Exact duplicate rows distort statistical summaries and model training weights.",
                        "Current Situation": f"Found {count} exact duplicate row(s) ({dup_pct:.1f}% of total rows).",
                        "Proposed Action": "Inspect duplicate records and remove redundant exact copies if confirmed unintentional.",
                        "Expected Effect": f"Reduces dataset size by {count} row(s), leaving only distinct observations.",
                    }
                )

            # -------------------------------------------------------------
            # 3. Constant & Near-Constant Columns
            # -------------------------------------------------------------
            elif issue_type == "Constant Column":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Structural",
                        "Priority": "Medium",
                        "Recommended Action": "Consider Column Removal",
                        "Method": "Drop Zero-Variance Column",
                        "Reason": "Column has zero variance (only 1 unique value) and offers no predictive or analytical contrast.",
                        "Current Situation": f"Column '{col_name}' contains the exact same value across all non-null rows.",
                        "Proposed Action": f"Evaluate whether column '{col_name}' can be removed unless needed for auditing.",
                        "Expected Effect": "Eliminates uninformative feature and reduces dimensional complexity.",
                    }
                )

            elif issue_type == "Near-Constant Column":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Structural",
                        "Priority": "Low",
                        "Recommended Action": "Review Feature Variation",
                        "Method": "Analytical Review",
                        "Reason": "Near-constant feature (>=95% single value) has extremely low variance.",
                        "Current Situation": f"Dominant single value accounts for {pct_str} of rows in '{col_name}'.",
                        "Proposed Action": f"Review whether column '{col_name}' provides meaningful analytical differentiation.",
                        "Expected Effect": "Helps determine if low-variance feature adds value to downstream analysis.",
                    }
                )

            # -------------------------------------------------------------
            # 4. High Cardinality
            # -------------------------------------------------------------
            elif issue_type == "High Cardinality":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Structural",
                        "Priority": "Info",
                        "Recommended Action": "Reclassify or Exclude from Grouping",
                        "Method": "Semantic Reclassification / Identifier Handling",
                        "Reason": "High cardinality categorical fields can cause memory bloat and model overfitting.",
                        "Current Situation": f"Column '{col_name}' has {count} distinct values ({pct_str} unique ratio).",
                        "Proposed Action": f"Treat '{col_name}' as an Identifier or apply target encoding / frequency grouping.",
                        "Expected Effect": "Prevents high-cardinality explosion in categorical grouping and one-hot encoding.",
                    }
                )

            # -------------------------------------------------------------
            # 5. Data-Type Inconsistency & Invalid Values
            # -------------------------------------------------------------
            elif issue_type == "Data-Type Inconsistency":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Validity",
                        "Priority": "High",
                        "Recommended Action": "Coerce to Numeric & Handle Strings",
                        "Method": "Numeric Coercion with Invalid Value Isolation",
                        "Reason": "Non-numeric text entries (e.g. 'N/A', 'unknown') prevent quantitative calculations.",
                        "Current Situation": f"Column '{col_name}' contains {count} non-numeric string value(s) ({pct_str}).",
                        "Proposed Action": f"Convert valid string numbers in '{col_name}' to numeric type and cast non-numeric entries to NaN or explicit flag.",
                        "Expected Effect": f"Enables numerical operations and statistics on '{col_name}'.",
                    }
                )

            elif issue_type == "Invalid Values":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Validity",
                        "Priority": "High",
                        "Recommended Action": "Correct or Replace Invalid Domain Values",
                        "Method": "Domain Bounds Validation / Null Replacement",
                        "Reason": "Values violate basic domain logic (e.g. negative age or impossible percentages).",
                        "Current Situation": f"Found {count} invalid value(s) in '{col_name}' violating domain constraints.",
                        "Proposed Action": f"Correct invalid entries in '{col_name}' using source data or set to missing (NaN) for re-imputation.",
                        "Expected Effect": "Ensures domain integrity and prevents impossible values from distorting statistics.",
                    }
                )

            elif issue_type == "Categorical Inconsistency":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Consistency",
                        "Priority": "Medium",
                        "Recommended Action": "Standardize Category Capitalization & Formatting",
                        "Method": "Casing Normalization (e.g. Lowercase / Trim Whitespace)",
                        "Reason": "Mixed capitalization (e.g. 'Male', 'male', 'MALE') creates duplicate semantic categories.",
                        "Current Situation": f"Column '{col_name}' contains {count} redundant category variation(s) due to formatting.",
                        "Proposed Action": f"Standardize string values in '{col_name}' to uniform case (e.g. 'Male' -> 'male') and strip whitespace.",
                        "Expected Effect": f"Consolidates category counts and improves grouping accuracy for '{col_name}'.",
                    }
                )

            elif issue_type == "Inconsistent Boolean Formatting":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Consistency",
                        "Priority": "Medium",
                        "Recommended Action": "Standardize Boolean Representations",
                        "Method": "Boolean Casting (True/False or 1/0)",
                        "Reason": "Mixed boolean representations ('Yes/No', '1/0', 'True/False') fragment binary logic.",
                        "Current Situation": f"Column '{col_name}' contains {count} distinct boolean string formats.",
                        "Proposed Action": f"Convert all representations in '{col_name}' into a uniform True/False or 1/0 boolean type.",
                        "Expected Effect": "Standardizes binary column for clean logic and boolean modeling.",
                    }
                )

            elif issue_type == "Date Parsing Failure":
                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Validity",
                        "Priority": "High",
                        "Recommended Action": "Parse & Validate Date Formats",
                        "Method": "Standardized Datetime Conversion (ISO 8601)",
                        "Reason": "Unparseable date strings fail temporal indexing and datetime arithmetic.",
                        "Current Situation": f"Column '{col_name}' has {count} date string(s) ({pct_str}) failing standard parsing.",
                        "Proposed Action": f"Validate date formats in '{col_name}', parse valid dates to YYYY-MM-DD, and flag unparseable entries.",
                        "Expected Effect": "Enables native datetime features and temporal analysis.",
                    }
                )

            # -------------------------------------------------------------
            # 6. Potential Outliers
            # -------------------------------------------------------------
            elif issue_type == "Potential Outliers":
                try:
                    o_pct = float(str(pct_str).replace("%", "").strip())
                except ValueError:
                    o_pct = 0.0

                recommendations.append(
                    {
                        "Column": col_name,
                        "Issue Type": issue_type,
                        "Category": "Outliers",
                        "Priority": "Low",
                        "Recommended Action": "Investigate & Review Numerical Outliers",
                        "Method": "Outlier Investigation (Winsorize / Cap / Retain)",
                        "Reason": "Statistical outliers outside IQR bounds should be investigated before deciding to retain or cap.",
                        "Current Situation": f"Column '{col_name}' contains {count} potential outlier(s) ({o_pct:.1f}% of non-null values).",
                        "Proposed Action": f"Examine extreme values in '{col_name}'. Retain if valid extreme measurements, or cap at IQR boundaries if erroneous.",
                        "Expected Effect": "Protects model sensitivity while preserving genuine extreme data points.",
                    }
                )

    # Summary counting
    priority_counts = {"High": 0, "Medium": 0, "Low": 0, "Info": 0}
    category_counts = {
        "Missing Values": 0,
        "Duplicates": 0,
        "Consistency": 0,
        "Validity": 0,
        "Structural": 0,
        "Outliers": 0,
    }

    for rec in recommendations:
        p = rec["Priority"]
        c = rec["Category"]
        if p in priority_counts:
            priority_counts[p] += 1
        if c in category_counts:
            category_counts[c] += 1

    summary = {
        "total_recommendations": len(recommendations),
        "priority_counts": priority_counts,
        "category_counts": category_counts,
    }

    recommendations_df = pd.DataFrame(
        recommendations,
        columns=[
            "Column",
            "Issue Type",
            "Category",
            "Priority",
            "Recommended Action",
            "Method",
            "Reason",
            "Current Situation",
            "Proposed Action",
            "Expected Effect",
        ],
    )

    return {
        "summary": summary,
        "recommendations_df": recommendations_df,
        "recommendations_list": recommendations,
    }
