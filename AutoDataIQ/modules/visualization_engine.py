"""Automated Visualization Selection & Visualization Engine module for AutoDataIQ.

Provides automated visualization selection rules and interactive Plotly chart generation for:
- Numerical Distributions (Histogram, Box Plot)
- Categorical Distributions (Top-N Bar Chart)
- Numerical Relationships (Scatter Plot)
- Pairwise Correlation Heatmaps (Plotly Heatmap)
- Time-Series Trends (Line Chart for Date x Numerical or Date Record Count)
- Categorical Group Comparisons (Aggregated Bar Chart for Categorical x Numerical)
- Boolean Flag Distributions (True/False Bar Chart)

Key Features:
- Automatic rule-based visualization recommendations with priorities (High, Medium, Low).
- Identifier column protection (excludes ID keys from quantitative charts).
- High-cardinality control (auto Top-N capping for clean readability).
- Quality awareness (warns if plotted columns contain quality issues).
- Strictly read-only: does not modify or transform the input DataFrame.
"""

from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.analysis_engine import (
    analyze_correlation,
    classify_columns_semantically,
)
from modules.semantic_detector import detect_semantic_types


def get_visualization_plan(
    dataframe: pd.DataFrame,
    semantic_types_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Determine recommended Plotly visualization types based on column classifications."""
    if dataframe is None or dataframe.empty:
        return []

    if semantic_types_df is None or semantic_types_df.empty:
        semantic_types_df = detect_semantic_types(dataframe)

    classified = classify_columns_semantically(dataframe, semantic_types_df)
    num_cols = classified["numerical"]
    cat_cols = classified["categorical"]
    date_cols = classified["date"]
    bool_cols = classified["boolean"]
    text_cols = classified["text"]
    id_cols = classified["identifier"]

    plan: list[dict[str, Any]] = []

    # 1. Numerical Histograms & Box Plots
    for col in num_cols:
        plan.append(
            {
                "chart_type": "Histogram",
                "columns": col,
                "category": "Distribution",
                "priority": "High",
                "reason": f"Shows numerical distribution and skewness for quantitative column '{col}'.",
            }
        )
        plan.append(
            {
                "chart_type": "Box Plot",
                "columns": col,
                "category": "Distribution",
                "priority": "Medium",
                "reason": f"Displays quartiles, median, and potential statistical outliers for '{col}'.",
            }
        )

    # 2. Categorical Distribution Bar Charts
    for col in cat_cols:
        uniq_cnt = dataframe[col].nunique()
        if uniq_cnt > 10:
            reason = f"Top-10 frequency bar chart recommended to manage high cardinality ({uniq_cnt} unique categories)."
        else:
            reason = f"Bar chart suitable for comparing category frequencies across {uniq_cnt} categories."

        plan.append(
            {
                "chart_type": "Bar Chart (Top-N)" if uniq_cnt > 10 else "Bar Chart",
                "columns": col,
                "category": "Categories",
                "priority": "High",
                "reason": reason,
            }
        )

    # 3. Boolean Flag Bar Charts
    for col in bool_cols:
        plan.append(
            {
                "chart_type": "Bar Chart",
                "columns": col,
                "category": "Categories",
                "priority": "High",
                "reason": f"Shows True vs False binary flag distribution for '{col}'.",
            }
        )

    # 4. Numerical Pairwise Scatter Plots
    if len(num_cols) >= 2:
        col_pair = f"{num_cols[0]} vs {num_cols[1]}"
        plan.append(
            {
                "chart_type": "Scatter Plot",
                "columns": col_pair,
                "category": "Relationships",
                "priority": "Medium",
                "reason": f"Visualizes bivariate numerical relationship between '{num_cols[0]}' and '{num_cols[1]}'.",
            }
        )

    # 5. Correlation Heatmap
    if len(num_cols) >= 2:
        plan.append(
            {
                "chart_type": "Correlation Heatmap",
                "columns": f"{len(num_cols)} Numerical Columns",
                "category": "Relationships",
                "priority": "Medium",
                "reason": f"Displays Pearson correlation matrix heatmap across all {len(num_cols)} numerical columns.",
            }
        )

    # 6. Time-Series Line Charts
    for col in date_cols:
        if len(num_cols) >= 1:
            target_num = num_cols[0]
            col_pair = f"{col} -> {target_num}"
            reason = f"Line chart shows time-series trend of '{target_num}' over date column '{col}'."
        else:
            col_pair = col
            reason = f"Line chart shows record count volume trend over date column '{col}'."

        plan.append(
            {
                "chart_type": "Line Chart",
                "columns": col_pair,
                "category": "Trends",
                "priority": "High",
                "reason": reason,
            }
        )

    # 7. Categorical + Numerical Grouped Bar Charts
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        col_pair = f"{cat_cols[0]} x {num_cols[0]}"
        plan.append(
            {
                "chart_type": "Grouped Bar Chart",
                "columns": col_pair,
                "category": "Group Comparison",
                "priority": "High",
                "reason": f"Compares average '{num_cols[0]}' grouped across '{cat_cols[0]}' categories.",
            }
        )

    # 8. Identifier Exclusions Notice
    for col in id_cols:
        plan.append(
            {
                "chart_type": "Excluded",
                "columns": col,
                "category": "Identifier",
                "priority": "Info",
                "reason": f"Identifier key column '{col}' excluded from automatic analytical charts.",
            }
        )

    return plan


# =============================================================================
# Plotly Chart Generation Helpers
# =============================================================================


def generate_histogram(
    dataframe: pd.DataFrame, column: str, nbins: int = 30
) -> go.Figure:
    """Generate an interactive Plotly Histogram for a numerical column."""
    series = dataframe[column].dropna()
    fig = px.histogram(
        dataframe,
        x=column,
        nbins=nbins,
        title=f"Distribution of {column}",
        labels={column: column, "count": "Frequency"},
        color_discrete_sequence=["#2b5c8f"],
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=column,
        yaxis_title="Frequency / Count",
        bargap=0.05,
    )
    return fig


def generate_boxplot(dataframe: pd.DataFrame, column: str) -> go.Figure:
    """Generate an interactive Plotly Box Plot for a numerical column."""
    fig = px.box(
        dataframe,
        y=column,
        title=f"Box Plot & Potential Outliers of {column}",
        points="outliers",
        color_discrete_sequence=["#d9534f"],
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_title=column,
    )
    return fig


def generate_categorical_barchart(
    dataframe: pd.DataFrame, column: str, top_n: int = 10
) -> go.Figure:
    """Generate an interactive Plotly Bar Chart for a categorical column (capped to top_n)."""
    val_counts = dataframe[column].value_counts().head(top_n).reset_index()
    val_counts.columns = [column, "Count"]

    total_cnt = len(dataframe[column].dropna())
    subtitle_note = (
        f" (Top {top_n} Categories)" if dataframe[column].nunique() > top_n else ""
    )

    fig = px.bar(
        val_counts,
        x=column,
        y="Count",
        text="Count",
        title=f"Category Distribution of {column}{subtitle_note}",
        color_discrete_sequence=["#428bca"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        xaxis_title=column,
        yaxis_title="Count",
    )
    return fig


def generate_scatterplot(
    dataframe: pd.DataFrame, x_col: str, y_col: str
) -> go.Figure:
    """Generate an interactive Plotly Scatter Plot for two numerical columns."""
    fig = px.scatter(
        dataframe,
        x=x_col,
        y=y_col,
        title=f"{y_col} vs {x_col}",
        color_discrete_sequence=["#2e7d32"],
        opacity=0.7,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return fig


def generate_correlation_heatmap(
    dataframe: pd.DataFrame, numerical_cols: list[str]
) -> go.Figure:
    """Generate an interactive Plotly Correlation Heatmap for numerical columns."""
    if len(numerical_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Minimum 2 numerical columns required for correlation heatmap.",
            showarrow=False,
        )
        return fig

    corr_df = analyze_correlation(dataframe, numerical_cols)

    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Pearson Correlation Matrix Heatmap",
    )
    fig.update_layout(template="plotly_white")
    return fig


def generate_line_chart(
    dataframe: pd.DataFrame,
    date_col: str,
    num_col: str | None = None,
    agg_func: str = "Mean",
) -> go.Figure:
    """Generate an interactive Plotly Line Chart for time-series trends."""
    temp_df = pd.DataFrame()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_df["parsed_date"] = pd.to_datetime(
            dataframe[date_col], errors="coerce", format="mixed"
        )

    if num_col and num_col in dataframe.columns:
        temp_df[num_col] = pd.to_numeric(dataframe[num_col], errors="coerce")

    valid_data = temp_df[temp_df["parsed_date"].notna()].copy()
    if valid_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid parsed dates available for line chart.", showarrow=False
        )
        return fig

    valid_data["Year-Month"] = (
        valid_data["parsed_date"].dt.to_period("M").astype(str)
    )

    if num_col and num_col in valid_data.columns:
        agg_key = agg_func.lower()
        if agg_key == "sum":
            grouped = valid_data.groupby("Year-Month")[num_col].sum().reset_index()
            y_title = f"Sum of {num_col}"
        elif agg_key == "median":
            grouped = valid_data.groupby("Year-Month")[num_col].median().reset_index()
            y_title = f"Median of {num_col}"
        elif agg_key == "count":
            grouped = valid_data.groupby("Year-Month")[num_col].count().reset_index()
            y_title = f"Count of {num_col}"
        else:  # default Mean
            grouped = valid_data.groupby("Year-Month")[num_col].mean().reset_index()
            y_title = f"Average {num_col}"

        chart_title = f"Monthly {y_title} Trend over {date_col}"
        y_col_name = num_col
    else:
        grouped = (
            valid_data.groupby("Year-Month")
            .size()
            .reset_index(name="Record Count")
        )
        chart_title = f"Monthly Record Count Volume Trend over {date_col}"
        y_col_name = "Record Count"
        y_title = "Record Count"

    grouped = grouped.sort_values(by="Year-Month")

    fig = px.line(
        grouped,
        x="Year-Month",
        y=y_col_name,
        markers=True,
        title=chart_title,
        color_discrete_sequence=["#e65100"],
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Year-Month",
        yaxis_title=y_title,
    )
    return fig


def generate_grouped_barchart(
    dataframe: pd.DataFrame,
    cat_col: str,
    num_col: str,
    agg_func: str = "Mean",
    top_n: int = 10,
) -> go.Figure:
    """Generate an interactive Plotly Bar Chart comparing numerical aggregation across categories."""
    temp_df = pd.DataFrame()
    temp_df[cat_col] = dataframe[cat_col].astype(str)
    temp_df[num_col] = pd.to_numeric(dataframe[num_col], errors="coerce")
    valid_data = temp_df.dropna()

    if valid_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid data for group comparison.", showarrow=False)
        return fig

    top_cats = valid_data[cat_col].value_counts().head(top_n).index
    valid_data = valid_data[valid_data[cat_col].isin(top_cats)]

    agg_key = agg_func.lower()
    if agg_key == "sum":
        grouped = valid_data.groupby(cat_col)[num_col].sum().reset_index()
        y_title = f"Sum of {num_col}"
    elif agg_key == "median":
        grouped = valid_data.groupby(cat_col)[num_col].median().reset_index()
        y_title = f"Median of {num_col}"
    elif agg_key == "count":
        grouped = valid_data.groupby(cat_col)[num_col].count().reset_index()
        y_title = f"Count of {num_col}"
    else:  # default Mean
        grouped = valid_data.groupby(cat_col)[num_col].mean().reset_index()
        y_title = f"Average {num_col}"

    grouped = grouped.sort_values(by=num_col, ascending=False)
    grouped[num_col] = grouped[num_col].round(2)

    fig = px.bar(
        grouped,
        x=cat_col,
        y=num_col,
        text=num_col,
        title=f"{agg_func.capitalize()} {num_col} by {cat_col} (Top {top_n} Categories)",
        color_discrete_sequence=["#6a1b9a"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        xaxis_title=cat_col,
        yaxis_title=y_title,
    )
    return fig


def generate_boolean_barchart(dataframe: pd.DataFrame, column: str) -> go.Figure:
    """Generate an interactive Plotly Bar Chart for a boolean column."""
    series = dataframe[column].dropna()
    truthy_vals = {"true", "1", "t", "yes", "y"}

    true_count = sum(
        1 for val in series if str(val).strip().lower() in truthy_vals
    )
    false_count = len(series) - true_count

    df_bool = pd.DataFrame(
        {
            "Status": ["True / Positive", "False / Negative"],
            "Count": [true_count, false_count],
        }
    )

    fig = px.bar(
        df_bool,
        x="Status",
        y="Count",
        text="Count",
        title=f"Boolean Distribution of {column}",
        color="Status",
        color_discrete_map={"True / Positive": "#2e7d32", "False / Negative": "#c62828"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Boolean Status",
        yaxis_title="Count",
        showlegend=False,
    )
    return fig
