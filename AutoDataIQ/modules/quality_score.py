"""Data Quality Scoring module for AutoDataIQ.

Converts Phase 4 quality engine results into an overall Data Quality Score (0-100)
and dimension scores across 6 quality dimensions:
1. Completeness (25%)
2. Uniqueness (15%)
3. Consistency (20%)
4. Validity (20%)
5. Structural Quality (10%)
6. Numerical Reliability (10%)

Handles non-applicable dimensions dynamically and provides human-readable ratings.
"""

from typing import Any
import pandas as pd


def get_rating(score: float) -> str:
    """Return a human-readable quality rating based on score (0-100)."""
    if score >= 90.0:
        return "Excellent"
    if score >= 75.0:
        return "Good"
    if score >= 60.0:
        return "Fair"
    if score >= 40.0:
        return "Poor"
    return "Critical"


def calculate_quality_score(
    dataframe: pd.DataFrame,
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    """Calculate overall data quality score and dimension breakdown.

    Parameters:
    - dataframe: The Pandas DataFrame being evaluated.
    - quality_report: The output dictionary from quality_engine.analyze_data_quality().

    Returns a dictionary containing:
    - overall_score: float (0.0 to 100.0)
    - rating: str ("Excellent", "Good", "Fair", "Poor", "Critical")
    - dimensions: list of dicts with dimension scores and details
    - dimensions_df: pd.DataFrame summary table of dimensions
    """
    if dataframe is None or dataframe.empty:
        return {
            "overall_score": 0.0,
            "rating": "Critical",
            "dimensions": [],
            "dimensions_df": pd.DataFrame(
                columns=["Dimension", "Score", "Weight", "Contribution", "Reason"]
            ),
        }

    summary = quality_report.get("summary", {})
    issues_df = quality_report.get("issues_df", pd.DataFrame())
    outliers_df = quality_report.get("outliers_df", pd.DataFrame())

    row_count, col_count = dataframe.shape
    total_cells = row_count * col_count if row_count > 0 and col_count > 0 else 1

    # Base dimension weights
    base_weights = {
        "Completeness": 0.25,
        "Uniqueness": 0.15,
        "Consistency": 0.20,
        "Validity": 0.20,
        "Structural Quality": 0.10,
        "Numerical Reliability": 0.10,
    }

    dimension_results: dict[str, dict[str, Any]] = {}

    # -------------------------------------------------------------
    # 1. Completeness Score
    # -------------------------------------------------------------
    total_missing_cells = summary.get("total_missing_cells", 0)
    empty_cols_count = summary.get("empty_columns_count", 0)
    missing_cell_pct = (total_missing_cells / total_cells) * 100.0

    comp_penalty = (missing_cell_pct * 1.5) + ((empty_cols_count / col_count) * 50.0 if col_count > 0 else 0.0)
    completeness_score = max(0.0, min(100.0, 100.0 - comp_penalty))

    if missing_cell_pct == 0.0 and empty_cols_count == 0:
        comp_reason = "No missing values or empty columns detected."
    else:
        comp_reason = f"{total_missing_cells} missing cell(s) ({missing_cell_pct:.1f}% of total) across dataset."
        if empty_cols_count > 0:
            comp_reason += f" Includes {empty_cols_count} completely empty column(s)."

    dimension_results["Completeness"] = {
        "score": completeness_score,
        "applicable": True,
        "reason": comp_reason,
    }

    # -------------------------------------------------------------
    # 2. Uniqueness Score
    # -------------------------------------------------------------
    duplicate_rows = summary.get("duplicate_rows_count", 0)
    duplicate_pct = (duplicate_rows / row_count * 100.0) if row_count > 0 else 0.0

    uniq_penalty = duplicate_pct * 2.0
    uniqueness_score = max(0.0, min(100.0, 100.0 - uniq_penalty))

    if duplicate_rows == 0:
        uniq_reason = "No duplicate rows detected in dataset."
    else:
        uniq_reason = f"{duplicate_rows} duplicate row(s) detected ({duplicate_pct:.1f}% of total rows)."

    dimension_results["Uniqueness"] = {
        "score": uniqueness_score,
        "applicable": True,
        "reason": uniq_reason,
    }

    # -------------------------------------------------------------
    # 3. Consistency Score
    # -------------------------------------------------------------
    consistency_issue_types = {
        "Categorical Inconsistency",
        "Inconsistent Boolean Formatting",
        "Data-Type Inconsistency",
        "Date Parsing Failure",
    }

    if not issues_df.empty:
        c_issues = issues_df[issues_df["Issue Type"].isin(consistency_issue_types)]
        c_issue_count = len(c_issues)
        c_affected_counts = 0
        for _, row in c_issues.iterrows():
            cnt = row.get("Count", 0)
            if isinstance(cnt, (int, float)):
                c_affected_counts += int(cnt)
    else:
        c_issue_count = 0
        c_affected_counts = 0

    c_affected_pct = (c_affected_counts / total_cells) * 100.0 if total_cells > 0 else 0.0
    consistency_penalty = (c_affected_pct * 3.0) + (c_issue_count * 5.0)
    consistency_score = max(0.0, min(100.0, 100.0 - consistency_penalty))

    if c_issue_count == 0:
        c_reason = "No data-type or formatting inconsistencies detected."
    else:
        c_reason = f"{c_issue_count} inconsistency issue(s) affecting ~{c_affected_counts} cell(s)."

    dimension_results["Consistency"] = {
        "score": consistency_score,
        "applicable": True,
        "reason": c_reason,
    }

    # -------------------------------------------------------------
    # 4. Validity Score
    # -------------------------------------------------------------
    validity_issue_types = {"Invalid Values", "Data-Type Inconsistency", "Date Parsing Failure"}

    if not issues_df.empty:
        v_issues = issues_df[issues_df["Issue Type"].isin(validity_issue_types)]
        v_issue_count = len(v_issues)
        v_affected_counts = 0
        for _, row in v_issues.iterrows():
            cnt = row.get("Count", 0)
            if isinstance(cnt, (int, float)):
                v_affected_counts += int(cnt)
    else:
        v_issue_count = 0
        v_affected_counts = 0

    v_affected_pct = (v_affected_counts / total_cells) * 100.0 if total_cells > 0 else 0.0
    validity_penalty = (v_affected_pct * 4.0) + (v_issue_count * 6.0)
    validity_score = max(0.0, min(100.0, 100.0 - validity_penalty))

    if v_issue_count == 0:
        v_reason = "All non-null values comply with expected domain and type specifications."
    else:
        v_reason = f"{v_issue_count} invalid value issue(s) affecting ~{v_affected_counts} cell(s)."

    dimension_results["Validity"] = {
        "score": validity_score,
        "applicable": True,
        "reason": v_reason,
    }

    # -------------------------------------------------------------
    # 5. Structural Quality Score
    # -------------------------------------------------------------
    constant_cols = 0
    near_constant_cols = 0
    high_cardinality_cols = summary.get("high_cardinality_columns_count", 0)

    if not issues_df.empty:
        struct_issues = issues_df[
            issues_df["Issue Type"].isin(
                ["Constant Column", "Near-Constant Column", "High Cardinality"]
            )
        ]
        for _, row in struct_issues.iterrows():
            itype = row["Issue Type"]
            if itype == "Constant Column":
                constant_cols += 1
            elif itype == "Near-Constant Column":
                near_constant_cols += 1

    struct_penalty = (
        (constant_cols * 15.0)
        + (near_constant_cols * 8.0)
        + (high_cardinality_cols * 5.0)
    )
    structural_score = max(0.0, min(100.0, 100.0 - struct_penalty))

    if constant_cols == 0 and near_constant_cols == 0 and high_cardinality_cols == 0:
        struct_reason = "No structural issues (constant or high-cardinality fields) detected."
    else:
        struct_parts = []
        if constant_cols > 0:
            struct_parts.append(f"{constant_cols} constant col(s)")
        if near_constant_cols > 0:
            struct_parts.append(f"{near_constant_cols} near-constant col(s)")
        if high_cardinality_cols > 0:
            struct_parts.append(f"{high_cardinality_cols} high-cardinality col(s)")
        struct_reason = "Structural considerations: " + ", ".join(struct_parts) + "."

    dimension_results["Structural Quality"] = {
        "score": structural_score,
        "applicable": True,
        "reason": struct_reason,
    }

    # -------------------------------------------------------------
    # 6. Numerical Reliability Score (Applicable only if numerical cols exist)
    # -------------------------------------------------------------
    numerical_cols = dataframe.select_dtypes(include="number")
    num_col_count = len(numerical_cols.columns)

    if num_col_count == 0:
        dimension_results["Numerical Reliability"] = {
            "score": None,
            "applicable": False,
            "reason": "Not Applicable: Dataset contains no numerical columns.",
        }
    else:
        outlier_cols_count = summary.get("outlier_columns_count", 0)
        total_outliers = 0
        if not outliers_df.empty and "Outlier Count" in outliers_df.columns:
            total_outliers = int(outliers_df["Outlier Count"].sum())

        num_cells = int(numerical_cols.notna().sum().sum())
        outlier_pct = (total_outliers / num_cells * 100.0) if num_cells > 0 else 0.0

        num_penalty = (outlier_pct * 1.0) + (outlier_cols_count * 2.0)
        num_score = max(0.0, min(100.0, 100.0 - num_penalty))

        if total_outliers == 0:
            num_reason = f"No statistical outliers detected across {num_col_count} numerical column(s)."
        else:
            num_reason = f"{total_outliers} potential outlier(s) ({outlier_pct:.1f}%) detected across {outlier_cols_count} numerical column(s)."

        dimension_results["Numerical Reliability"] = {
            "score": num_score,
            "applicable": True,
            "reason": num_reason,
        }

    # -------------------------------------------------------------
    # 7. Normalize Weights & Compute Overall Weighted Score
    # -------------------------------------------------------------
    applicable_base_weight_sum = sum(
        base_weights[dim]
        for dim, info in dimension_results.items()
        if info["applicable"]
    )

    overall_weighted_score = 0.0
    dimension_rows: list[dict[str, Any]] = []

    for dim_name, base_w in base_weights.items():
        info = dimension_results[dim_name]
        if info["applicable"]:
            effective_weight = base_w / applicable_base_weight_sum
            score_val = info["score"]
            contribution = score_val * effective_weight
            overall_weighted_score += contribution

            dimension_rows.append(
                {
                    "Dimension": dim_name,
                    "Score": round(score_val, 1),
                    "Base Weight": f"{int(base_w * 100)}%",
                    "Effective Weight": f"{effective_weight * 100:.1f}%",
                    "Contribution": round(contribution, 1),
                    "Reason": info["reason"],
                    "Applicable": True,
                }
            )
        else:
            dimension_rows.append(
                {
                    "Dimension": dim_name,
                    "Score": "N/A",
                    "Base Weight": f"{int(base_w * 100)}%",
                    "Effective Weight": "0.0%",
                    "Contribution": 0.0,
                    "Reason": info["reason"],
                    "Applicable": False,
                }
            )

    final_score = round(max(0.0, min(100.0, overall_weighted_score)), 1)
    final_rating = get_rating(final_score)

    dimensions_df = pd.DataFrame(
        dimension_rows,
        columns=[
            "Dimension",
            "Score",
            "Base Weight",
            "Effective Weight",
            "Contribution",
            "Reason",
        ],
    )

    return {
        "overall_score": final_score,
        "rating": final_rating,
        "dimensions": dimension_rows,
        "dimensions_df": dimensions_df,
    }
