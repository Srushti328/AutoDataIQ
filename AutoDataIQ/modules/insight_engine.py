"""Insight Generation and Prioritization Engine for AutoDataIQ.

Phase 9: Automatically identifies meaningful patterns from the dataset,
profile metrics, quality assessments, and exploratory analysis results.
Explains findings in clear, plain language with supporting numerical evidence,
priority classifications (High, Medium, Low), and statistical caveats.

Read-only: Never modifies, imputes, or alters the input DataFrame.
"""

from typing import Any
import numpy as np
import pandas as pd

from modules.analysis_engine import classify_columns_semantically
from modules.quality_engine import analyze_data_quality
from modules.quality_score import calculate_quality_score
from modules.semantic_detector import detect_semantic_types


PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _format_num(val: float | int | Any) -> str:
    """Format numbers nicely for presentation."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        elif abs(val) >= 0.01:
            return f"{val:.2f}"
        elif val == 0:
            return "0.00"
        else:
            return f"{val:.4f}"
    return str(val)


def generate_insights(
    dataframe: pd.DataFrame,
    semantic_types_df: pd.DataFrame | None = None,
    quality_report: dict[str, Any] | None = None,
    quality_score_result: dict[str, Any] | None = None,
    analysis_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate prioritized, plain-language insights across all analytical dimensions.

    Parameters:
    - dataframe: The input dataset (strictly read-only).
    - semantic_types_df: Optional precomputed semantic types DataFrame.
    - quality_report: Optional precomputed quality report from analyze_data_quality().
    - quality_score_result: Optional precomputed quality score from calculate_quality_score().
    - analysis_results: Optional precomputed adaptive analysis results.

    Returns:
    dict containing:
    - summary: dict of counts (total, high, medium, low, category breakdown)
    - insights: list of individual insight dicts
    - insights_df: pd.DataFrame of insights for table display
    """
    if dataframe is None or dataframe.empty:
        empty_df = pd.DataFrame(
            columns=[
                "Title",
                "Category",
                "Priority",
                "Evidence",
                "Columns",
                "Description",
                "Reason",
                "Warnings",
            ]
        )
        return {
            "summary": {
                "total_insights": 0,
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0,
                "category_counts": {
                    "Data Quality": 0,
                    "Numerical": 0,
                    "Categorical": 0,
                    "Temporal": 0,
                    "General": 0,
                },
            },
            "insights": [],
            "insights_df": empty_df,
        }

    # 1. Resolve prerequisites safely
    if semantic_types_df is None or semantic_types_df.empty:
        semantic_types_df = detect_semantic_types(dataframe)

    classified_cols = classify_columns_semantically(dataframe, semantic_types_df)

    if quality_report is None:
        quality_report = analyze_data_quality(dataframe, semantic_types_df)

    if quality_score_result is None:
        quality_score_result = calculate_quality_score(dataframe, quality_report)

    insights_list: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def add_insight(
        title: str,
        description: str,
        category: str,
        priority: str,
        evidence: str,
        columns: list[str],
        reason: str,
        warnings: str = "Statistical finding based on observed dataset records. Correlation does not imply causation.",
    ) -> None:
        key = f"{category}::{title}::{'|'.join(sorted(columns))}"
        if key in seen_keys:
            return
        seen_keys.add(key)
        insights_list.append(
            {
                "Title": title,
                "Description": description,
                "Category": category,
                "Priority": priority,
                "Evidence": evidence,
                "Columns": columns,
                "Reason": reason,
                "Warnings": warnings,
            }
        )

    # -------------------------------------------------------------
    # 2. Data Quality Insights
    # -------------------------------------------------------------
    _extract_quality_insights(
        dataframe, quality_report, quality_score_result, add_insight
    )

    # -------------------------------------------------------------
    # 3. Numerical Insights (Skewness, Outliers, Correlation, Group Diff)
    # -------------------------------------------------------------
    _extract_numerical_insights(dataframe, classified_cols, add_insight)

    # -------------------------------------------------------------
    # 4. Categorical Insights (Dominance, Fragmentation, Uniformity)
    # -------------------------------------------------------------
    _extract_categorical_insights(dataframe, classified_cols, add_insight)

    # -------------------------------------------------------------
    # 5. Temporal Insights (Trends over time, Volume changes)
    # -------------------------------------------------------------
    _extract_temporal_insights(dataframe, classified_cols, add_insight)

    # -------------------------------------------------------------
    # 6. General Dataset Insights (Scale, Semantic Composition, IDs)
    # -------------------------------------------------------------
    _extract_general_insights(dataframe, classified_cols, add_insight)

    # -------------------------------------------------------------
    # 7. Sort Insights by Priority
    # -------------------------------------------------------------
    insights_list.sort(key=lambda x: PRIORITY_ORDER.get(x["Priority"], 99))

    # Calculate summary metrics
    high_count = sum(1 for i in insights_list if i["Priority"] == "High")
    med_count = sum(1 for i in insights_list if i["Priority"] == "Medium")
    low_count = sum(1 for i in insights_list if i["Priority"] == "Low")

    cat_counts: dict[str, int] = {
        "Data Quality": 0,
        "Numerical": 0,
        "Categorical": 0,
        "Temporal": 0,
        "General": 0,
    }
    for item in insights_list:
        cat = item["Category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    summary = {
        "total_insights": len(insights_list),
        "high_priority": high_count,
        "medium_priority": med_count,
        "low_priority": low_count,
        "category_counts": cat_counts,
    }

    # Format DataFrame for tabular display
    table_rows = []
    for item in insights_list:
        table_rows.append(
            {
                "Title": item["Title"],
                "Category": item["Category"],
                "Priority": item["Priority"],
                "Evidence": item["Evidence"],
                "Columns": ", ".join(item["Columns"]) if item["Columns"] else "Dataset-wide",
                "Description": item["Description"],
                "Reason": item["Reason"],
                "Warnings": item["Warnings"],
            }
        )

    insights_df = pd.DataFrame(
        table_rows,
        columns=[
            "Title",
            "Category",
            "Priority",
            "Evidence",
            "Columns",
            "Description",
            "Reason",
            "Warnings",
        ],
    )

    return {
        "summary": summary,
        "insights": insights_list,
        "insights_df": insights_df,
    }


def _extract_quality_insights(
    dataframe: pd.DataFrame,
    quality_report: dict[str, Any],
    quality_score_result: dict[str, Any],
    add_insight: Any,
) -> None:
    """Extract actionable insights from data quality scoring and issue detection."""
    total_rows = len(dataframe)
    overall_score = quality_score_result.get("overall_score", 100.0)
    rating = quality_score_result.get("rating", "Good")

    # Overall Score Insight
    if overall_score < 60.0:
        add_insight(
            title=f"Substantial Data Quality Risk ({overall_score:.1f}/100 - {rating})",
            description=f"The dataset scored an overall quality rating of {rating} ({overall_score:.1f}/100), indicating significant hygiene issues that may compromise analytical conclusions.",
            category="Data Quality",
            priority="High",
            evidence=f"Overall Score: {overall_score:.1f}/100, Rating: {rating}",
            columns=[],
            reason="Overall data quality score falls below 60.0, indicating widespread quality issues.",
            warnings="Cleaning recommendations should be reviewed before performing downstream statistical modeling.",
        )
    elif overall_score < 75.0:
        add_insight(
            title=f"Moderate Data Quality Concerns ({overall_score:.1f}/100 - {rating})",
            description=f"The dataset scored a quality rating of {rating} ({overall_score:.1f}/100). Notable anomalies or missingness exist but do not invalidate analysis.",
            category="Data Quality",
            priority="Medium",
            evidence=f"Overall Score: {overall_score:.1f}/100, Rating: {rating}",
            columns=[],
            reason="Quality score indicates moderate completeness, consistency, or validity defects.",
            warnings="Carefully verify affected fields before relying on automated aggregations.",
        )
    else:
        add_insight(
            title=f"High Overall Data Quality ({overall_score:.1f}/100 - {rating})",
            description=f"The dataset demonstrates high structural integrity with an overall quality score of {overall_score:.1f}/100 ({rating}).",
            category="Data Quality",
            priority="Low",
            evidence=f"Overall Score: {overall_score:.1f}/100, Rating: {rating}",
            columns=[],
            reason="High baseline quality indicates minimal systemic errors or missing records.",
            warnings="While overall quality is robust, localized column-specific anomalies may still exist.",
        )

    # Missing Values per Column
    for col in dataframe.columns:
        col_str = str(col)
        missing_count = int(dataframe[col].isna().sum())
        if missing_count > 0 and total_rows > 0:
            missing_pct = (missing_count / total_rows) * 100.0
            if missing_pct >= 30.0:
                add_insight(
                    title=f"Severe Missingness in '{col_str}' ({missing_pct:.1f}%)",
                    description=f"Column '{col_str}' is missing {missing_pct:.1f}% of its observations ({missing_count}/{total_rows} rows).",
                    category="Data Quality",
                    priority="High",
                    evidence=f"Missing Count: {missing_count:,} / {total_rows:,} ({missing_pct:.1f}%)",
                    columns=[col_str],
                    reason="Missingness exceeds 30.0%, potentially biasing statistics or imputations.",
                    warnings="Imputation on heavily missing columns may introduce artificial variance.",
                )
            elif missing_pct >= 10.0:
                add_insight(
                    title=f"Notable Missing Data in '{col_str}' ({missing_pct:.1f}%)",
                    description=f"Column '{col_str}' has {missing_count} missing values ({missing_pct:.1f}% of records).",
                    category="Data Quality",
                    priority="Medium",
                    evidence=f"Missing Count: {missing_count:,} / {total_rows:,} ({missing_pct:.1f}%)",
                    columns=[col_str],
                    reason="Missingness between 10.0% and 30.0% warrants attention during data preprocessing.",
                    warnings="Consider whether missing records occur completely at random or indicate systematic reporting omissions.",
                )

    # Duplicate Rows
    dup_count = int(dataframe.duplicated().sum())
    if dup_count > 0 and total_rows > 0:
        dup_pct = (dup_count / total_rows) * 100.0
        priority = "High" if dup_pct >= 5.0 else "Medium"
        add_insight(
            title=f"Duplicate Rows Detected ({dup_count:,} rows, {dup_pct:.1f}%)",
            description=f"The dataset contains {dup_count} exact duplicate rows ({dup_pct:.1f}% of total records), which can artificially inflate observation counts and metrics.",
            category="Data Quality",
            priority=priority,
            evidence=f"Duplicate Rows: {dup_count:,} out of {total_rows:,} ({dup_pct:.1f}%)",
            columns=[],
            reason="Duplicate records distort sample distributions and summary totals.",
            warnings="Verify whether duplicates represent true identical transactions or recording redundancies.",
        )

    # Constant Columns
    for col in dataframe.columns:
        col_str = str(col)
        non_null_series = dataframe[col].dropna()
        if len(non_null_series) > 0 and non_null_series.nunique() == 1:
            val = non_null_series.iloc[0]
            add_insight(
                title=f"Zero-Variance Constant Column: '{col_str}'",
                description=f"Column '{col_str}' contains only a single constant value ('{val}') across all non-null entries.",
                category="Data Quality",
                priority="High",
                evidence=f"Unique values: 1 (Value: '{val}', Total non-null records: {len(non_null_series):,})",
                columns=[col_str],
                reason="Constant columns carry zero statistical information and provide no predictive or analytical utility.",
                warnings="Exclusion from feature sets or predictive models is strongly recommended.",
            )


def _extract_numerical_insights(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_insight: Any,
) -> None:
    """Extract insights from numerical features: distributions, skewness, outliers, correlations, and group diffs."""
    num_cols = classified_cols.get("numerical", [])
    cat_cols = classified_cols.get("categorical", [])
    total_rows = len(dataframe)

    if total_rows < 3:
        return

    # 1. Distribution, Skewness, and Outlier Insights
    for col in num_cols:
        series = pd.to_numeric(dataframe[col], errors="coerce").dropna()
        n_count = len(series)
        if n_count < 4:
            continue

        mean_val = float(series.mean())
        median_val = float(series.median())
        std_val = float(series.std()) if n_count > 1 else 0.0
        q1_val = float(series.quantile(0.25))
        q3_val = float(series.quantile(0.75))
        iqr_val = q3_val - q1_val

        # Skewness
        if std_val > 0:
            skew_val = float(series.skew())
            if abs(skew_val) >= 2.0:
                direction = "right-skewed (positive skew)" if skew_val > 0 else "left-skewed (negative skew)"
                add_insight(
                    title=f"Significant Distribution Skewness in '{col}' (skew = {skew_val:+.2f})",
                    description=f"Column '{col}' exhibits heavy {direction}. The mean ({_format_num(mean_val)}) deviates notably from the median ({_format_num(median_val)}), driven by an asymmetrical tail.",
                    category="Numerical",
                    priority="High",
                    evidence=f"Skewness: {skew_val:+.2f}, Mean: {_format_num(mean_val)}, Median: {_format_num(median_val)}, Std Dev: {_format_num(std_val)}",
                    columns=[col],
                    reason="Skewness magnitude |skew| >= 2.0 indicates substantial departure from a symmetric distribution.",
                    warnings="Parametric statistical tests assuming normality may produce distorted results; consider median or non-parametric statistics.",
                )
            elif abs(skew_val) >= 1.0:
                direction = "moderately right-skewed" if skew_val > 0 else "moderately left-skewed"
                add_insight(
                    title=f"Moderate Asymmetry in '{col}' (skew = {skew_val:+.2f})",
                    description=f"Column '{col}' is {direction}. Mean ({_format_num(mean_val)}) versus median ({_format_num(median_val)}).",
                    category="Numerical",
                    priority="Medium",
                    evidence=f"Skewness: {skew_val:+.2f}, Mean: {_format_num(mean_val)}, Median: {_format_num(median_val)}",
                    columns=[col],
                    reason="Skewness magnitude 1.0 <= |skew| < 2.0 shows moderate asymmetry.",
                    warnings="Consider verifying if tail values represent valid observations or extreme outliers.",
                )

        # IQR Outliers
        if iqr_val > 0:
            lower_bound = q1_val - 1.5 * iqr_val
            upper_bound = q3_val + 1.5 * iqr_val
            outlier_series = series[(series < lower_bound) | (series > upper_bound)]
            outlier_count = len(outlier_series)

            if outlier_count > 0:
                outlier_pct = (outlier_count / n_count) * 100.0
                if outlier_pct >= 8.0:
                    add_insight(
                        title=f"High Outlier Concentration in '{col}' ({outlier_pct:.1f}%)",
                        description=f"Found {outlier_count} potential outlier values ({outlier_pct:.1f}% of non-null records) outside 1.5× IQR threshold [{_format_num(lower_bound)}, {_format_num(upper_bound)}].",
                        category="Numerical",
                        priority="High",
                        evidence=f"Outlier count: {outlier_count:,} / {n_count:,} ({outlier_pct:.1f}%), Bounds: [{_format_num(lower_bound)}, {_format_num(upper_bound)}]",
                        columns=[col],
                        reason="Outlier frequency >= 8.0% of valid observations indicates substantial spread in the extremes.",
                        warnings="Statistical outliers reflect mathematically distant values, not confirmed data entry errors. Domain context is required.",
                    )
                elif outlier_pct >= 3.0:
                    add_insight(
                        title=f"Notable Outliers in '{col}' ({outlier_count} records, {outlier_pct:.1f}%)",
                        description=f"Found {outlier_count} observations ({outlier_pct:.1f}%) falling outside 1.5× IQR bounds.",
                        category="Numerical",
                        priority="Medium",
                        evidence=f"Outliers: {outlier_count} ({outlier_pct:.1f}%), Q1: {_format_num(q1_val)}, Q3: {_format_num(q3_val)}, IQR: {_format_num(iqr_val)}",
                        columns=[col],
                        reason="Outliers represent between 3.0% and 8.0% of observations.",
                        warnings="Examine whether extreme observations represent genuine high-value events or measurement discrepancies.",
                    )

    # 2. Pairwise Pearson Correlation Insights
    if len(num_cols) >= 2:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1 = num_cols[i]
                col2 = num_cols[j]

                s1 = pd.to_numeric(dataframe[col1], errors="coerce")
                s2 = pd.to_numeric(dataframe[col2], errors="coerce")

                valid_df = pd.DataFrame({"x": s1, "y": s2}).dropna()
                valid_n = len(valid_df)

                if valid_n < 5:
                    continue

                std_x = float(valid_df["x"].std())
                std_y = float(valid_df["y"].std())
                if std_x == 0 or std_y == 0:
                    continue

                r_val = float(valid_df["x"].corr(valid_df["y"], method="pearson"))
                if np.isnan(r_val):
                    continue

                abs_r = abs(r_val)
                sign_text = "positive" if r_val > 0 else "negative"

                if abs_r >= 0.70:
                    add_insight(
                        title=f"Strong {sign_text.capitalize()} Correlation: '{col1}' & '{col2}' (r = {r_val:+.2f})",
                        description=f"A strong linear {sign_text} association (Pearson r = {r_val:+.2f}) was detected between '{col1}' and '{col2}'.",
                        category="Numerical",
                        priority="High",
                        evidence=f"Pearson r = {r_val:+.3f} (N = {valid_n:,} valid paired observations)",
                        columns=[col1, col2],
                        reason="Correlation magnitude |r| >= 0.70 signifies a strong linear relationship.",
                        warnings="Correlation indicates co-movement, not causality. Confounding variables or third-factor influences may be responsible.",
                    )
                elif abs_r >= 0.50:
                    add_insight(
                        title=f"Moderate {sign_text.capitalize()} Correlation: '{col1}' & '{col2}' (r = {r_val:+.2f})",
                        description=f"A moderate linear {sign_text} association (Pearson r = {r_val:+.2f}) was found between '{col1}' and '{col2}'.",
                        category="Numerical",
                        priority="Medium",
                        evidence=f"Pearson r = {r_val:+.3f} (N = {valid_n:,} valid paired observations)",
                        columns=[col1, col2],
                        reason="Correlation magnitude between 0.50 and 0.70 shows moderate linear relationship.",
                        warnings="Linear correlation only captures straight-line associations and does not imply direct causation.",
                    )
                elif abs_r >= 0.30:
                    add_insight(
                        title=f"Weak {sign_text.capitalize()} Association: '{col1}' & '{col2}' (r = {r_val:+.2f})",
                        description=f"A weak linear {sign_text} trend (Pearson r = {r_val:+.2f}) exists between '{col1}' and '{col2}'.",
                        category="Numerical",
                        priority="Low",
                        evidence=f"Pearson r = {r_val:+.3f} (N = {valid_n:,} observations)",
                        columns=[col1, col2],
                        reason="Correlation magnitude between 0.30 and 0.50 indicates mild association.",
                        warnings="Weak correlations may arise from random variation or indirect relationships.",
                    )

    # 3. Categorical Group Comparison Insights
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        for cat in cat_cols[:3]:  # Check top categorical columns to avoid combinatorial explosion
            for num in num_cols[:3]:
                temp_df = pd.DataFrame(
                    {
                        "cat": dataframe[cat].astype(str),
                        "num": pd.to_numeric(dataframe[num], errors="coerce"),
                    }
                ).dropna()

                if len(temp_df) < 6:
                    continue

                overall_num_mean = float(temp_df["num"].mean())
                if abs(overall_num_mean) < 1e-9:
                    continue

                # Group aggregations
                grouped = (
                    temp_df.groupby("cat")["num"]
                    .agg(["count", "mean"])
                    .reset_index()
                )
                # Keep groups with at least 3 records
                valid_groups = grouped[grouped["count"] >= 3]
                if len(valid_groups) < 2:
                    continue

                top_group = valid_groups.sort_values(by="mean", ascending=False).iloc[0]
                bottom_group = valid_groups.sort_values(by="mean", ascending=True).iloc[0]

                top_mean = float(top_group["mean"])
                bot_mean = float(bottom_group["mean"])
                spread = abs(top_mean - bot_mean)
                rel_spread = (spread / abs(overall_num_mean)) * 100.0

                top_name = str(top_group["cat"])
                bot_name = str(bottom_group["cat"])

                if rel_spread >= 50.0:
                    add_insight(
                        title=f"Pronounced Disparity in '{num}' across '{cat}' Groups",
                        description=f"Substantial differences observed in '{num}' by '{cat}': Group '{top_name}' has the highest mean ({_format_num(top_mean)}) whereas '{bot_name}' has the lowest ({_format_num(bot_mean)}), a spread of {rel_spread:.1f}% relative to overall mean.",
                        category="Numerical",
                        priority="High",
                        evidence=f"'{top_name}' mean: {_format_num(top_mean)} (N={int(top_group['count'])}), '{bot_name}' mean: {_format_num(bot_mean)} (N={int(bottom_group['count'])}), Dataset mean: {_format_num(overall_num_mean)}",
                        columns=[cat, num],
                        reason="Mean spread between extreme groups exceeds 50.0% of the overall dataset average.",
                        warnings="Group comparisons are descriptive. Sample sizes vary between subgroups and external drivers may exist.",
                    )
                elif rel_spread >= 25.0:
                    add_insight(
                        title=f"Notable Difference in '{num}' across '{cat}' Groups",
                        description=f"Notable contrast in '{num}' across categories of '{cat}': '{top_name}' average is {_format_num(top_mean)} vs '{bot_name}' average of {_format_num(bot_mean)}.",
                        category="Numerical",
                        priority="Medium",
                        evidence=f"'{top_name}' mean: {_format_num(top_mean)}, '{bot_name}' mean: {_format_num(bot_mean)}, Spread: {_format_num(spread)} ({rel_spread:.1f}%)",
                        columns=[cat, num],
                        reason="Mean spread between groups is between 25.0% and 50.0% of the overall mean.",
                        warnings="Subgroup disparity indicates statistical difference in sample averages; verify sample representativeness.",
                    )


def _extract_categorical_insights(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_insight: Any,
) -> None:
    """Extract insights from categorical columns: dominant categories, high cardinality, uniformity."""
    cat_cols = classified_cols.get("categorical", [])
    bool_cols = classified_cols.get("boolean", [])
    total_rows = len(dataframe)

    if total_rows == 0:
        return

    # Categorical Analysis
    for col in cat_cols:
        series = dataframe[col].dropna().astype(str)
        n_valid = len(series)
        if n_valid < 3:
            continue

        counts = series.value_counts()
        n_unique = len(counts)
        if counts.empty:
            continue

        top_val = str(counts.index[0])
        top_count = int(counts.iloc[0])
        top_pct = (top_count / n_valid) * 100.0

        # Dominant Category
        if top_pct >= 70.0:
            add_insight(
                title=f"Dominant Category in '{col}': '{top_val}' ({top_pct:.1f}%)",
                description=f"Category '{top_val}' represents an overwhelming majority ({top_pct:.1f}%) of non-null records in '{col}'.",
                category="Categorical",
                priority="High",
                evidence=f"Top category: '{top_val}' with {top_count:,} out of {n_valid:,} records ({top_pct:.1f}%) across {n_unique} unique categories",
                columns=[col],
                reason="Top category accounts for >= 70.0% of observations, showing severe class imbalance.",
                warnings="Heavy category concentration can lead to sample bias if this variable is used for segmentation.",
            )
        elif top_pct >= 50.0:
            add_insight(
                title=f"Majority Category in '{col}': '{top_val}' ({top_pct:.1f}%)",
                description=f"Category '{top_val}' represents more than half ({top_pct:.1f}%) of non-null observations in '{col}'.",
                category="Categorical",
                priority="Medium",
                evidence=f"Category '{top_val}': {top_count:,} / {n_valid:,} records ({top_pct:.1f}%)",
                columns=[col],
                reason="Top category constitutes between 50.0% and 70.0% of values.",
                warnings="Ensure secondary categories have sufficient observations for meaningful sub-group analysis.",
            )

        # High Cardinality / Fragmentation
        unique_ratio = (n_unique / n_valid) * 100.0
        if n_unique > 20 and unique_ratio >= 40.0:
            add_insight(
                title=f"High Category Diversity in '{col}' ({n_unique} categories)",
                description=f"Column '{col}' features high category dispersion with {n_unique} distinct classes across {n_valid} rows ({unique_ratio:.1f}% unique ratio).",
                category="Categorical",
                priority="Medium",
                evidence=f"Unique categories: {n_unique:,}, Total records: {n_valid:,}, Distinctness: {unique_ratio:.1f}%",
                columns=[col],
                reason="High cardinality with > 20 distinct values and >= 40.0% distinctness ratio.",
                warnings="High cardinality categorical fields may fragment group visualizations and increase dimensionality.",
            )
        elif 2 <= n_unique <= 4 and top_pct <= 35.0:
            # Balanced distribution
            add_insight(
                title=f"Balanced Category Distribution in '{col}'",
                description=f"Column '{col}' displays an evenly distributed categorical spread across {n_unique} classes, with no single value exceeding 35% of records.",
                category="Categorical",
                priority="Low",
                evidence=f"Classes: {n_unique}, Top class '{top_val}' has only {top_pct:.1f}% ({top_count}/{n_valid})",
                columns=[col],
                reason="Balanced distribution with low cardinality (<= 4 classes) and top class <= 35%.",
                warnings="Balanced categories provide stable sample sizes for comparative cross-tabulations.",
            )

    # Boolean Analysis
    for col in bool_cols:
        series = dataframe[col].dropna()
        n_valid = len(series)
        if n_valid < 3:
            continue

        truthy_vals = {"true", "1", "t", "yes", "y"}
        true_count = sum(1 for v in series if str(v).strip().lower() in truthy_vals)
        true_pct = (true_count / n_valid) * 100.0
        false_pct = 100.0 - true_pct

        if true_pct >= 85.0 or true_pct <= 15.0:
            dominant_flag = "True" if true_pct >= 85.0 else "False"
            dom_pct = max(true_pct, false_pct)
            add_insight(
                title=f"Heavily Skewed Boolean Flag in '{col}' ({dominant_flag}: {dom_pct:.1f}%)",
                description=f"Boolean column '{col}' is heavily skewed towards '{dominant_flag}' ({dom_pct:.1f}% of observations).",
                category="Categorical",
                priority="Medium",
                evidence=f"True: {true_count} ({true_pct:.1f}%), False: {n_valid - true_count} ({false_pct:.1f}%)",
                columns=[col],
                reason="Boolean indicator has >= 85.0% of records in one state.",
                warnings="Rare flag occurrences may make subgroup comparisons statistically fragile.",
            )
        else:
            add_insight(
                title=f"Active Distribution in Boolean Flag '{col}'",
                description=f"Boolean flag '{col}' has a balanced distribution between True ({true_pct:.1f}%) and False ({false_pct:.1f}%).",
                category="Categorical",
                priority="Low",
                evidence=f"True: {true_count} ({true_pct:.1f}%), False: {n_valid - true_count} ({false_pct:.1f}%)",
                columns=[col],
                reason="Boolean states are well-represented with neither state exceeding 85.0%.",
                warnings="Both True and False subsets have sufficient representations.",
            )


def _extract_temporal_insights(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_insight: Any,
) -> None:
    """Extract insights from date/datetime columns: trends over time and volume changes."""
    date_cols = classified_cols.get("date", [])
    num_cols = classified_cols.get("numerical", [])

    if not date_cols or len(dataframe) < 4:
        return

    import warnings

    for date_col in date_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_dates = pd.to_datetime(
                dataframe[date_col], errors="coerce", format="mixed"
            )

        valid_mask = parsed_dates.notna()
        if valid_mask.sum() < 4:
            continue

        temp_df = pd.DataFrame({"dt": parsed_dates[valid_mask]})
        temp_df["period"] = temp_df["dt"].dt.to_period("M").astype(str)

        periods = sorted(temp_df["period"].unique())
        if len(periods) < 3:
            # Insufficient periods to reliably identify a trend
            continue

        first_period = periods[0]
        last_period = periods[-1]

        # 1. Volume shift over time
        period_counts = temp_df["period"].value_counts().sort_index()
        first_count = int(period_counts.iloc[0])
        last_count = int(period_counts.iloc[-1])

        if first_count > 0:
            volume_delta_pct = ((last_count - first_count) / first_count) * 100.0
            if abs(volume_delta_pct) >= 40.0:
                direction = "increase" if volume_delta_pct > 0 else "decrease"
                add_insight(
                    title=f"Substantial Shift in Record Volume ({direction.capitalize()}) over '{date_col}'",
                    description=f"Observation volume underwent a {abs(volume_delta_pct):.1f}% {direction} from {first_period} ({first_count:,} records) to {last_period} ({last_count:,} records).",
                    category="Temporal",
                    priority="Medium",
                    evidence=f"Start ({first_period}): {first_count:,} records, End ({last_period}): {last_count:,} records, Change: {volume_delta_pct:+.1f}% across {len(periods)} monthly periods",
                    columns=[date_col],
                    reason="Record frequency change magnitude >= 40.0% between the first and last recorded time periods.",
                    warnings="Examine whether volume fluctuations represent business growth/decline or changes in data logging frequency.",
                )

        # 2. Numerical Trends over time
        for num_col in num_cols[:3]:  # Check top numerical columns
            num_series = pd.to_numeric(dataframe[num_col], errors="coerce")
            valid_combined = pd.DataFrame(
                {"dt": parsed_dates, "num": num_series}
            ).dropna()

            if len(valid_combined) < 6:
                continue

            valid_combined["period"] = valid_combined["dt"].dt.to_period("M").astype(str)
            period_aggs = (
                valid_combined.groupby("period")["num"]
                .agg(["count", "mean"])
                .sort_index()
            )

            # Keep periods with at least 2 records
            valid_period_aggs = period_aggs[period_aggs["count"] >= 2]
            if len(valid_period_aggs) < 3:
                continue

            first_p = valid_period_aggs.index[0]
            last_p = valid_period_aggs.index[-1]
            first_m = float(valid_period_aggs.iloc[0]["mean"])
            last_m = float(valid_period_aggs.iloc[-1]["mean"])

            if abs(first_m) > 1e-9:
                trend_delta_pct = ((last_m - first_m) / abs(first_m)) * 100.0
                abs_delta = abs(last_m - first_m)
                trend_dir = "Upward" if trend_delta_pct > 0 else "Downward"

                if abs(trend_delta_pct) >= 30.0:
                    add_insight(
                        title=f"{trend_dir} Trend in '{num_col}' over Time ({trend_delta_pct:+.1f}%)",
                        description=f"A marked {trend_dir.lower()} trajectory was observed in '{num_col}' across recorded periods: mean value shifted from {_format_num(first_m)} in {first_p} to {_format_num(last_m)} in {last_p}.",
                        category="Temporal",
                        priority="High",
                        evidence=f"Earliest ({first_p}): {_format_num(first_m)}, Latest ({last_p}): {_format_num(last_m)}, Delta: {trend_delta_pct:+.1f}% (abs change: {_format_num(abs_delta)}) across {len(valid_period_aggs)} periods",
                        columns=[date_col, num_col],
                        reason="Average value changed by >= 30.0% between earliest and latest temporal intervals with >= 3 distinct periods.",
                        warnings="Temporal trends represent observational historical progression; seasonality or unmeasured external macro factors may influence this path.",
                    )
                elif abs(trend_delta_pct) >= 10.0:
                    add_insight(
                        title=f"Moderate {trend_dir} Trend in '{num_col}' ({trend_delta_pct:+.1f}%)",
                        description=f"Mean '{num_col}' shifted by {trend_delta_pct:+.1f}% between {first_p} ({_format_num(first_m)}) and {last_p} ({_format_num(last_m)}).",
                        category="Temporal",
                        priority="Medium",
                        evidence=f"Start ({first_p}): {_format_num(first_m)}, End ({last_p}): {_format_num(last_m)}, Change: {trend_delta_pct:+.1f}% across {len(valid_period_aggs)} periods",
                        columns=[date_col, num_col],
                        reason="Temporal change is between 10.0% and 30.0% across >= 3 periods.",
                        warnings="Examine intermediate time periods to distinguish smooth secular trends from erratic monthly noise.",
                    )
                else:
                    add_insight(
                        title=f"Stable Temporal Trajectory in '{num_col}'",
                        description=f"Mean '{num_col}' remained relatively consistent across the timeline ({trend_delta_pct:+.1f}% change from {first_p} to {last_p}).",
                        category="Temporal",
                        priority="Low",
                        evidence=f"Start ({first_p}): {_format_num(first_m)}, End ({last_p}): {_format_num(last_m)}, Delta: {trend_delta_pct:+.1f}%",
                        columns=[date_col, num_col],
                        reason="Magnitude of change across recorded periods is under 10.0%.",
                        warnings="Metric demonstrated minimal long-term drift over observed intervals.",
                    )


def _extract_general_insights(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_insight: Any,
) -> None:
    """Extract general structural and semantic composition insights."""
    total_rows, total_cols = dataframe.shape

    num_len = len(classified_cols.get("numerical", []))
    cat_len = len(classified_cols.get("categorical", []))
    date_len = len(classified_cols.get("date", []))
    bool_len = len(classified_cols.get("boolean", []))
    id_len = len(classified_cols.get("identifier", []))
    text_len = len(classified_cols.get("text", []))

    # Dataset Scale & Composition
    add_insight(
        title=f"Dataset Structure & Semantic Composition ({total_rows:,} rows, {total_cols} columns)",
        description=f"The dataset consists of {total_rows:,} records and {total_cols} attributes spanning {num_len} Numerical, {cat_len} Categorical, {date_len} Date/Datetime, {bool_len} Boolean, {id_len} Identifier, and {text_len} Text features.",
        category="General",
        priority="Low",
        evidence=f"Total cells: {total_rows * total_cols:,} | Numerical: {num_len}, Categorical: {cat_len}, Date: {date_len}, Boolean: {bool_len}, Identifier: {id_len}, Text: {text_len}",
        columns=[],
        reason="Baseline structural summary describing the feature space and dimensional scope.",
        warnings="Structural composition guides exploratory methods and analytical suitability.",
    )

    # Identifier Key Exclusions
    for id_col in classified_cols.get("identifier", []):
        series = dataframe[id_col].dropna()
        n_valid = len(series)
        n_unique = series.nunique()
        uniq_pct = (n_unique / n_valid * 100.0) if n_valid > 0 else 0.0

        add_insight(
            title=f"Entity Key Identified: '{id_col}' ({uniq_pct:.1f}% Unique)",
            description=f"Column '{id_col}' is recognized as an entity identifier ({n_unique:,} unique keys across {n_valid:,} records) and is safely quarantined from numerical statistical calculations.",
            category="General",
            priority="Low",
            evidence=f"Unique Count: {n_unique:,} / {n_valid:,} ({uniq_pct:.1f}% uniqueness)",
            columns=[id_col],
            reason="Identifier columns must be excluded from arithmetic aggregations to prevent nonsensical averages.",
            warnings="Identifiers serve entity linkage purposes rather than quantitative evaluation.",
        )
