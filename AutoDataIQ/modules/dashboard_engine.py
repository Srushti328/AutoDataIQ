"""Automated Dashboard Generation Engine for AutoDataIQ.

Phase 12: Aggregates and curates dataset-tailored KPIs, recommended visualizations,
prioritized analytical insights, and data quality summaries into an executive,
unified dashboard presentation.

Strictly read-only: Reuses existing calculation engines and never modifies the input DataFrame.
"""

from typing import Any
import pandas as pd
import plotly.graph_objects as go

from modules.analysis_engine import classify_columns_semantically
from modules.insight_engine import generate_insights
from modules.kpi_engine import detect_kpis
from modules.quality_engine import analyze_data_quality
from modules.quality_score import calculate_quality_score
from modules.semantic_detector import detect_semantic_types
from modules.visualization_engine import (
    generate_boolean_barchart,
    generate_boxplot,
    generate_categorical_barchart,
    generate_correlation_heatmap,
    generate_grouped_barchart,
    generate_histogram,
    generate_line_chart,
    generate_scatterplot,
    get_visualization_plan,
)


def generate_dashboard_data(
    dataframe: pd.DataFrame,
    dataset_name: str = "Uploaded Dataset",
    is_cleaned: bool = False,
    max_kpis: int = 6,
    max_charts: int = 4,
    max_insights: int = 5,
) -> dict[str, Any]:
    """Compile structured, curated dashboard elements tailored to the active dataset.

    Parameters:
    - dataframe: The input dataset (strictly read-only).
    - dataset_name: Human-readable label for the dataset.
    - is_cleaned: Boolean flag indicating if this is the cleaned version.
    - max_kpis: Maximum number of summary KPI cards to curate.
    - max_charts: Maximum number of diverse charts to recommend.
    - max_insights: Maximum number of highlighted insights to curate.

    Returns:
    A structured dictionary containing:
    - header: metadata, row/col counts, active choice
    - kpis: curated summary cards and total count
    - visualizations: list of curated Plotly figure dicts
    - insights: highlighted prioritized findings and summary
    - quality_summary: overall score, rating, missing/dup counts
    - is_empty: boolean flag for empty/insufficient data
    - limitations: list of data limitation caveats
    """
    if dataframe is None or dataframe.empty:
        return {
            "header": {
                "dataset_name": dataset_name,
                "rows": 0,
                "columns": 0,
                "is_cleaned": is_cleaned,
                "active_dataset_label": "Cleaned Dataset" if is_cleaned else "Original Dataset",
                "description": "No data available to display in dashboard.",
            },
            "kpis": {
                "summary_cards": [],
                "total_kpis_detected": 0,
                "all_kpis": [],
            },
            "visualizations": [],
            "insights": {
                "summary": {
                    "total_insights": 0,
                    "high_priority": 0,
                    "medium_priority": 0,
                    "low_priority": 0,
                },
                "highlighted_insights": [],
                "all_insights": [],
            },
            "quality_summary": {
                "overall_score": 0.0,
                "rating": "Critical",
                "missing_cells": 0,
                "missing_percentage": 0.0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
                "total_issues": 0,
                "top_warnings": ["Dataset contains no rows."],
            },
            "is_empty": True,
            "limitations": ["Dataset is empty. Upload a valid dataset to generate dashboard content."],
        }

    row_count, col_count = dataframe.shape
    limitations: list[str] = []

    if row_count < 5:
        limitations.append("Small sample size (< 5 rows); statistical aggregations and charts may be sensitive to sample noise.")

    # -------------------------------------------------------------
    # 1. Semantic Column Classification
    # -------------------------------------------------------------
    semantic_types_df = detect_semantic_types(dataframe)
    classified_cols = classify_columns_semantically(dataframe, semantic_types_df)

    # -------------------------------------------------------------
    # 2. Data Quality & Scoring
    # -------------------------------------------------------------
    quality_report = analyze_data_quality(dataframe, semantic_types_df)
    quality_score_result = calculate_quality_score(dataframe, quality_report)

    overall_score = quality_score_result.get("overall_score", 100.0)
    rating = quality_score_result.get("rating", "Good")
    q_summary = quality_report.get("summary", {})

    total_cells = row_count * col_count
    missing_cells = q_summary.get("total_missing_cells", int(dataframe.isna().sum().sum()))
    missing_pct = round((missing_cells / total_cells * 100.0), 2) if total_cells > 0 else 0.0

    duplicate_rows = q_summary.get("duplicate_rows_count", int(dataframe.duplicated().sum()))
    duplicate_pct = round((duplicate_rows / row_count * 100.0), 2) if row_count > 0 else 0.0

    # Extract top quality warnings
    top_warnings: list[str] = []
    issues_df = quality_report.get("issues_df", pd.DataFrame())
    if not issues_df.empty:
        high_issues = issues_df[issues_df["Severity"] == "High"]
        target_issues = high_issues if not high_issues.empty else issues_df
        for _, iss in target_issues.head(3).iterrows():
            top_warnings.append(f"{iss['Issue Type']} in '{iss['Column']}': {iss['Description']}")

    quality_summary_data = {
        "overall_score": round(overall_score, 1),
        "rating": rating,
        "missing_cells": missing_cells,
        "missing_percentage": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_pct,
        "total_issues": len(issues_df),
        "top_warnings": top_warnings,
    }

    # -------------------------------------------------------------
    # 3. KPI Extraction & Curated Summary Cards
    # -------------------------------------------------------------
    kpi_results = detect_kpis(dataframe, semantic_types_df)
    all_kpis = kpi_results.get("kpis", [])
    total_kpis_count = kpi_results.get("summary", {}).get("total_kpis", len(all_kpis))

    # Curate top KPIs:
    # 1st priority: Potential Business KPIs (Revenue, Profit, Sales, Volume, Salary)
    # 2nd priority: Overview KPIs (Total Records, Unique Entities)
    # 3rd priority: Significant Statistical KPIs (Dominant category, key averages)
    curated_kpis: list[dict[str, Any]] = []

    business_kpis = [k for k in all_kpis if k["Category"] == "Potential Business"]
    overview_kpis = [k for k in all_kpis if k["Category"] == "Overview" and "Total Records" in k["KPI Name"]]
    entity_kpis = [k for k in all_kpis if k["Category"] == "Overview" and "Distinct Entities" in k["KPI Name"]]
    other_high_kpis = [
        k for k in all_kpis
        if k["Priority"] == "High" and k not in business_kpis and k not in overview_kpis and k not in entity_kpis
    ]
    other_kpis = [
        k for k in all_kpis
        if k not in business_kpis and k not in overview_kpis and k not in entity_kpis and k not in other_high_kpis
    ]

    candidate_kpis = overview_kpis + business_kpis + entity_kpis + other_high_kpis + other_kpis

    seen_kpi_names: set[str] = set()
    for k in candidate_kpis:
        k_name = k["KPI Name"]
        if k_name not in seen_kpi_names:
            seen_kpi_names.add(k_name)
            curated_kpis.append(k)
            if len(curated_kpis) >= max_kpis:
                break

    # -------------------------------------------------------------
    # 4. Curated Recommended Visualizations
    # -------------------------------------------------------------
    curated_charts: list[dict[str, Any]] = []
    used_chart_types: set[str] = set()
    used_columns: set[str] = set()

    num_cols = classified_cols.get("numerical", [])
    cat_cols = classified_cols.get("categorical", [])
    date_cols = classified_cols.get("date", [])
    bool_cols = classified_cols.get("boolean", [])

    # Strategy 1: Temporal Trend (Date x Num or Date Count)
    if date_cols and len(curated_charts) < max_charts:
        d_col = date_cols[0]
        n_col = num_cols[0] if num_cols else None
        fig_line = generate_line_chart(dataframe, d_col, n_col)
        if fig_line is not None:
            curated_charts.append({
                "title": f"Time-Series Trend: {n_col if n_col else 'Record Count'} over '{d_col}'",
                "chart_type": "Line Chart",
                "figure": fig_line,
                "description": f"Tracks monthly progression of {n_col if n_col else 'record counts'} across time periods.",
                "category": "Temporal",
                "columns": [d_col, n_col] if n_col else [d_col],
            })
            used_chart_types.add("Line Chart")
            used_columns.add(d_col)
            if n_col:
                used_columns.add(n_col)

    # Strategy 2: Categorical x Numerical Group Comparison
    if cat_cols and num_cols and len(curated_charts) < max_charts:
        c_col = cat_cols[0]
        # Choose numerical col not already overused
        n_candidates = [n for n in num_cols if n not in used_columns]
        n_col = n_candidates[0] if n_candidates else num_cols[0]

        fig_grp = generate_grouped_barchart(dataframe, c_col, n_col, top_n=8)
        if fig_grp is not None:
            curated_charts.append({
                "title": f"Average '{n_col}' by '{c_col}' (Group Comparison)",
                "chart_type": "Grouped Bar Chart",
                "figure": fig_grp,
                "description": f"Compares average {n_col} values across leading categories of {c_col}.",
                "category": "Comparison",
                "columns": [c_col, n_col],
            })
            used_chart_types.add("Grouped Bar Chart")
            used_columns.add(c_col)
            used_columns.add(n_col)

    # Strategy 3: Correlation Heatmap or Scatter Plot (if >= 2 numerical columns)
    if len(num_cols) >= 2 and len(curated_charts) < max_charts:
        if len(num_cols) >= 3:
            fig_hm = generate_correlation_heatmap(dataframe, num_cols[:6])
            if fig_hm is not None:
                curated_charts.append({
                    "title": "Pearson Correlation Heatmap (Numerical Relationships)",
                    "chart_type": "Correlation Heatmap",
                    "figure": fig_hm,
                    "description": "Displays pairwise linear association strengths across numerical attributes.",
                    "category": "Relationship",
                    "columns": num_cols[:6],
                })
                used_chart_types.add("Correlation Heatmap")
        else:
            fig_scatter = generate_scatterplot(dataframe, num_cols[0], num_cols[1])
            if fig_scatter is not None:
                curated_charts.append({
                    "title": f"Relationship: '{num_cols[0]}' vs '{num_cols[1]}'",
                    "chart_type": "Scatter Plot",
                    "figure": fig_scatter,
                    "description": f"Examines bivariate relationship and point distribution between {num_cols[0]} and {num_cols[1]}.",
                    "category": "Relationship",
                    "columns": [num_cols[0], num_cols[1]],
                })
                used_chart_types.add("Scatter Plot")

    # Strategy 4: Numerical Distribution (Histogram)
    if num_cols and len(curated_charts) < max_charts:
        n_candidates = [n for n in num_cols if n not in used_columns]
        n_col = n_candidates[0] if n_candidates else num_cols[0]
        fig_hist = generate_histogram(dataframe, n_col)
        if fig_hist is not None:
            curated_charts.append({
                "title": f"Distribution of '{n_col}' (Histogram)",
                "chart_type": "Histogram",
                "figure": fig_hist,
                "description": f"Shows distribution shape, frequency density, and spread for {n_col}.",
                "category": "Distribution",
                "columns": [n_col],
            })
            used_chart_types.add("Histogram")
            used_columns.add(n_col)

    # Strategy 5: Categorical Distribution (Bar Chart)
    if cat_cols and len(curated_charts) < max_charts:
        c_candidates = [c for c in cat_cols if c not in used_columns]
        c_col = c_candidates[0] if c_candidates else cat_cols[0]
        fig_cat = generate_categorical_barchart(dataframe, c_col, top_n=10)
        if fig_cat is not None:
            curated_charts.append({
                "title": f"Category Frequencies: '{c_col}' (Top Categories)",
                "chart_type": "Bar Chart",
                "figure": fig_cat,
                "description": f"Frequency breakdown across the most common classes in {c_col}.",
                "category": "Distribution",
                "columns": [c_col],
            })
            used_chart_types.add("Bar Chart")
            used_columns.add(c_col)

    # Strategy 6: Boolean Flag Distribution
    if bool_cols and len(curated_charts) < max_charts:
        b_col = bool_cols[0]
        fig_bool = generate_boolean_barchart(dataframe, b_col)
        if fig_bool is not None:
            curated_charts.append({
                "title": f"Boolean Flag Breakdown: '{b_col}'",
                "chart_type": "Boolean Bar Chart",
                "figure": fig_bool,
                "description": f"Proportion of True vs False states for flag {b_col}.",
                "category": "Distribution",
                "columns": [b_col],
            })
            used_chart_types.add("Boolean Bar Chart")

    if not curated_charts:
        limitations.append("Dataset features do not meet the minimum requirements for quantitative visualization.")

    # -------------------------------------------------------------
    # 5. Prioritized Key Insights
    # -------------------------------------------------------------
    insights_result = generate_insights(
        dataframe,
        semantic_types_df=semantic_types_df,
        quality_report=quality_report,
        quality_score_result=quality_score_result,
    )
    all_insights = insights_result.get("insights", [])
    insight_summary = insights_result.get("summary", {})

    # Prioritize: High priority insights first, followed by Medium
    high_insights = [i for i in all_insights if i["Priority"] == "High"]
    med_insights = [i for i in all_insights if i["Priority"] == "Medium"]
    low_insights = [i for i in all_insights if i["Priority"] == "Low"]

    ordered_insights = high_insights + med_insights + low_insights
    curated_insights = ordered_insights[:max_insights]

    if not curated_insights:
        limitations.append("Insufficient variation to produce statistical insights.")

    # -------------------------------------------------------------
    # 6. Assemble Final Dashboard Object
    # -------------------------------------------------------------
    return {
        "header": {
            "dataset_name": dataset_name,
            "rows": row_count,
            "columns": col_count,
            "is_cleaned": is_cleaned,
            "active_dataset_label": "Cleaned Dataset" if is_cleaned else "Original Dataset",
            "description": "Curated executive overview synthesizing automatic KPIs, recommended visual patterns, prioritized statistical insights, and data quality metrics.",
        },
        "kpis": {
            "summary_cards": curated_kpis,
            "total_kpis_detected": total_kpis_count,
            "all_kpis": all_kpis,
        },
        "visualizations": curated_charts,
        "insights": {
            "summary": {
                "total_insights": insight_summary.get("total_insights", len(all_insights)),
                "high_priority": insight_summary.get("high_priority", 0),
                "medium_priority": insight_summary.get("medium_priority", 0),
                "low_priority": insight_summary.get("low_priority", 0),
            },
            "highlighted_insights": curated_insights,
            "all_insights": all_insights,
        },
        "quality_summary": quality_summary_data,
        "is_empty": False,
        "limitations": limitations,
    }


DEFAULT_SECTION_ORDER: list[str] = [
    "KPI Summary",
    "Visualizations",
    "Key Insights",
    "Data Quality Summary",
]


def get_default_customization_config(dashboard_data: dict[str, Any]) -> dict[str, Any]:
    """Extract default customization configuration matching the automated dashboard layout.

    Parameters:
    - dashboard_data: Output dictionary from generate_dashboard_data().

    Returns:
    Dictionary containing default lists of selected KPI names, chart titles,
    insight titles, section order, and customization flag.
    """
    curated_kpi_names = [
        k.get("KPI Name")
        for k in dashboard_data.get("kpis", {}).get("summary_cards", [])
        if "KPI Name" in k
    ]
    curated_chart_titles = [
        c.get("title")
        for c in dashboard_data.get("visualizations", [])
        if "title" in c
    ]
    curated_insight_titles = [
        i.get("Title")
        for i in dashboard_data.get("insights", {}).get("highlighted_insights", [])
        if "Title" in i
    ]

    return {
        "selected_kpi_names": list(curated_kpi_names),
        "selected_chart_titles": list(curated_chart_titles),
        "selected_insight_titles": list(curated_insight_titles),
        "section_order": list(DEFAULT_SECTION_ORDER),
        "is_customized": False,
    }


def filter_customization_for_dataset(
    config: dict[str, Any],
    dashboard_data: dict[str, Any],
) -> dict[str, Any]:
    """Sanitize and update customization config when applied to a (possibly switched) dataset.

    Removes any selected KPI, visualization, or insight that is not available in the target dataset,
    preventing misleading placeholders, stale widget states, or broken keys.

    Parameters:
    - config: Current customization configuration dict.
    - dashboard_data: Dashboard data dictionary for the active dataset.

    Returns:
    Sanitized customization configuration dict.
    """
    if not config:
        return get_default_customization_config(dashboard_data)

    # Valid available items in the target dataset
    available_kpis = {
        k.get("KPI Name")
        for k in dashboard_data.get("kpis", {}).get("all_kpis", [])
        if "KPI Name" in k
    }
    for k in dashboard_data.get("kpis", {}).get("summary_cards", []):
        if "KPI Name" in k:
            available_kpis.add(k["KPI Name"])

    available_charts = {
        c.get("title")
        for c in dashboard_data.get("visualizations", [])
        if "title" in c
    }

    available_insights = {
        i.get("Title")
        for i in dashboard_data.get("insights", {}).get("all_insights", [])
        if "Title" in i
    }
    for i in dashboard_data.get("insights", {}).get("highlighted_insights", []):
        if "Title" in i:
            available_insights.add(i["Title"])

    # Filter selections to only what exists
    clean_kpis = [k for k in config.get("selected_kpi_names", []) if k in available_kpis]
    clean_charts = [c for c in config.get("selected_chart_titles", []) if c in available_charts]
    clean_insights = [i for i in config.get("selected_insight_titles", []) if i in available_insights]

    # Validate section order
    raw_sections = config.get("section_order", DEFAULT_SECTION_ORDER)
    clean_sections = [s for s in raw_sections if s in DEFAULT_SECTION_ORDER]
    if not clean_sections:
        clean_sections = list(DEFAULT_SECTION_ORDER)

    return {
        "selected_kpi_names": clean_kpis,
        "selected_chart_titles": clean_charts,
        "selected_insight_titles": clean_insights,
        "section_order": clean_sections,
        "is_customized": config.get("is_customized", False),
    }


def apply_dashboard_customization(
    dashboard_data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply user customization choices to filter and reorder dashboard elements.

    Non-mutating: Operates on a copy and does not alter the input dashboard_data or DataFrame.

    Parameters:
    - dashboard_data: The output dict from generate_dashboard_data().
    - config: Optional customization dict with selected_kpi_names, selected_chart_titles,
              selected_insight_titles, section_order, and is_customized.

    Returns:
    A new dashboard data dictionary reflecting user selections, ordering, and customization metadata.
    """
    if dashboard_data is None:
        return {}

    if dashboard_data.get("is_empty", False):
        return dict(dashboard_data)

    if config is None:
        config = get_default_customization_config(dashboard_data)

    # Sanitize config against available elements
    sanitized_config = filter_customization_for_dataset(config, dashboard_data)

    # 1. Filter and order KPI cards
    all_kpis_dict = {
        k["KPI Name"]: k
        for k in dashboard_data.get("kpis", {}).get("all_kpis", [])
        if "KPI Name" in k
    }
    for k in dashboard_data.get("kpis", {}).get("summary_cards", []):
        if "KPI Name" in k and k["KPI Name"] not in all_kpis_dict:
            all_kpis_dict[k["KPI Name"]] = k

    selected_kpis: list[dict[str, Any]] = []
    for k_name in sanitized_config["selected_kpi_names"]:
        if k_name in all_kpis_dict:
            selected_kpis.append(all_kpis_dict[k_name])

    # 2. Filter and order Visualizations
    all_charts_dict = {
        c["title"]: c
        for c in dashboard_data.get("visualizations", [])
        if "title" in c
    }
    selected_charts: list[dict[str, Any]] = []
    for c_title in sanitized_config["selected_chart_titles"]:
        if c_title in all_charts_dict:
            selected_charts.append(all_charts_dict[c_title])

    # 3. Filter and order Insights
    all_insights_dict = {
        i["Title"]: i
        for i in dashboard_data.get("insights", {}).get("all_insights", [])
        if "Title" in i
    }
    for i in dashboard_data.get("insights", {}).get("highlighted_insights", []):
        if "Title" in i and i["Title"] not in all_insights_dict:
            all_insights_dict[i["Title"]] = i

    selected_insights: list[dict[str, Any]] = []
    for i_title in sanitized_config["selected_insight_titles"]:
        if i_title in all_insights_dict:
            selected_insights.append(all_insights_dict[i_title])

    # 4. Construct customized dashboard dictionary
    customized_result = {
        "header": dict(dashboard_data.get("header", {})),
        "kpis": {
            "summary_cards": selected_kpis,
            "total_kpis_detected": dashboard_data.get("kpis", {}).get("total_kpis_detected", 0),
            "all_kpis": dashboard_data.get("kpis", {}).get("all_kpis", []),
        },
        "visualizations": selected_charts,
        "insights": {
            "summary": dict(dashboard_data.get("insights", {}).get("summary", {})),
            "highlighted_insights": selected_insights,
            "all_insights": dashboard_data.get("insights", {}).get("all_insights", []),
        },
        "quality_summary": dict(dashboard_data.get("quality_summary", {})),
        "is_empty": False,
        "limitations": list(dashboard_data.get("limitations", [])),
        "customization": {
            "is_customized": sanitized_config.get("is_customized", False),
            "section_order": sanitized_config.get("section_order", list(DEFAULT_SECTION_ORDER)),
            "selected_kpi_names": sanitized_config.get("selected_kpi_names", []),
            "selected_chart_titles": sanitized_config.get("selected_chart_titles", []),
            "selected_insight_titles": sanitized_config.get("selected_insight_titles", []),
            "displayed_kpis_count": len(selected_kpis),
            "displayed_charts_count": len(selected_charts),
            "displayed_insights_count": len(selected_insights),
        },
    }

    return customized_result
