"""Streamlit page for Phase 4 & Phase 5 Data Quality Intelligence and Scoring."""

import pandas as pd
import streamlit as st

from modules.quality_engine import analyze_data_quality
from modules.quality_score import calculate_quality_score

st.set_page_config(page_title="Quality Report - AutoDataIQ", page_icon="🔍", layout="wide")

st.title("🔍 Data Quality Intelligence & Scoring")
st.write(
    "Automated detection of data quality issues and overall Data Quality Scoring across 6 core dimensions."
)

if "data" not in st.session_state or st.session_state["data"] is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Upload page first.")
    st.info("💡 Navigate to **Upload** using the sidebar on the left.")
else:
    dataframe: pd.DataFrame = st.session_state["data"]
    filename = st.session_state.get("filename", "Uploaded Dataset")

    st.write(
        f"**Active Dataset:** `{filename}` | **Rows:** {len(dataframe)} | **Columns:** {len(dataframe.columns)}"
    )

    # -------------------------------------------------------------
    # Run Quality Analysis & Calculate Score
    # -------------------------------------------------------------
    quality_report = analyze_data_quality(dataframe)
    score_report = calculate_quality_score(dataframe, quality_report)

    overall_score = score_report["overall_score"]
    rating = score_report["rating"]
    dimensions_df = score_report["dimensions_df"]

    summary = quality_report["summary"]
    issues_df: pd.DataFrame = quality_report["issues_df"]
    outliers_df: pd.DataFrame = quality_report["outliers_df"]

    # -------------------------------------------------------------
    # 1. Phase 5: Overall Data Quality Score Section
    # -------------------------------------------------------------
    st.subheader("📊 Overall Data Quality")

    # Rating badge styling helper
    rating_emojis = {
        "Excellent": "🌟",
        "Good": "✅",
        "Fair": "⚠️",
        "Poor": "🟠",
        "Critical": "🚨",
    }
    emoji = rating_emojis.get(rating, "📊")

    score_col1, score_col2 = st.columns([1, 2])
    with score_col1:
        st.metric(label="Data Quality Score", value=f"{overall_score} / 100")
        st.subheader(f"Rating: {emoji} {rating}")

    with score_col2:
        st.info(
            "💡 **Score Interpretation:** The Data Quality Score is a project-defined indicator "
            "calculated from detected data-quality issues. It evaluates Completeness (25%), "
            "Uniqueness (15%), Consistency (20%), Validity (20%), Structural Quality (10%), and "
            "Numerical Reliability (10%). A higher score indicates fewer detected issues."
        )

    st.markdown("#### Dimension Breakdown")

    # Render metric cards for dimensions
    dims_list = score_report["dimensions"]
    dim_cols = st.columns(len(dims_list) if len(dims_list) <= 6 else 6)
    for idx, dim in enumerate(dims_list):
        c_idx = idx % len(dim_cols)
        score_display = f"{dim['Score']}/100" if dim["Applicable"] else "N/A"
        dim_cols[c_idx].metric(
            label=dim["Dimension"],
            value=score_display,
            delta=f"Weight: {dim['Effective Weight']}",
            delta_color="off",
        )

    with st.expander("View Detailed Dimension Calculation Table", expanded=False):
        st.dataframe(dimensions_df, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. Phase 4: Data Quality Overview Metrics
    # -------------------------------------------------------------
    st.subheader("Data Quality Issues Overview")
    m_cols = st.columns(6)
    m_cols[0].metric("Total Issues", summary["total_issues_count"])
    m_cols[1].metric("Missing Value Cols", summary["missing_value_columns_count"])
    m_cols[2].metric("Duplicate Rows", summary["duplicate_rows_count"])
    m_cols[3].metric("Outlier Cols (IQR)", summary["outlier_columns_count"])
    m_cols[4].metric("High Cardinality", summary["high_cardinality_columns_count"])
    m_cols[5].metric("Inconsistent Cols", summary["inconsistent_columns_count"])

    # -------------------------------------------------------------
    # 3. All Detected Quality Issues Table
    # -------------------------------------------------------------
    st.subheader("Detected Quality Issues")

    if issues_df.empty:
        st.success("🎉 No significant quality issues detected in this dataset!")
    else:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            severity_options = ["All"] + sorted(list(issues_df["Severity"].unique()))
            selected_severity = st.selectbox("Filter by Severity:", severity_options)
        with filter_col2:
            type_options = ["All"] + sorted(list(issues_df["Issue Type"].unique()))
            selected_type = st.selectbox("Filter by Issue Type:", type_options)

        filtered_issues = issues_df.copy()
        if selected_severity != "All":
            filtered_issues = filtered_issues[filtered_issues["Severity"] == selected_severity]
        if selected_type != "All":
            filtered_issues = filtered_issues[filtered_issues["Issue Type"] == selected_type]

        st.dataframe(filtered_issues, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. Detailed Categorized Quality Reports
    # -------------------------------------------------------------
    st.subheader("Categorized Issue Breakdown")
    tab_missing, tab_dups, tab_outliers, tab_inconsistent, tab_cardinality = st.tabs(
        [
            "Missing & Empty",
            "Duplicate Records",
            "Potential Outliers (IQR)",
            "Data-Type & Inconsistencies",
            "Structural & Cardinality",
        ]
    )

    # Tab 1: Missing & Empty Columns
    with tab_missing:
        st.write("#### Missing Values & Empty Columns")
        missing_issues = issues_df[issues_df["Issue Type"].isin(["Missing Values", "Empty Column"])]
        if missing_issues.empty:
            st.info("No missing values or empty columns detected.")
        else:
            st.dataframe(missing_issues, use_container_width=True)

    # Tab 2: Duplicate Records
    with tab_dups:
        st.write("#### Duplicate Records Analysis")
        dup_issues = issues_df[issues_df["Issue Type"] == "Duplicate Rows"]
        if dup_issues.empty:
            st.info("No duplicate rows found in the dataset.")
        else:
            st.dataframe(dup_issues, use_container_width=True)

    # Tab 3: Potential Outliers (IQR)
    with tab_outliers:
        st.write("#### Statistical Numerical Outliers (IQR Method)")
        st.caption(
            "Outliers are defined as values outside $[Q1 - 1.5 \\times IQR, Q3 + 1.5 \\times IQR]$. "
            "Note: Outliers represent statistical anomalies and are not automatically errors."
        )
        if outliers_df.empty:
            st.info("No numerical outliers detected outside IQR bounds.")
        else:
            st.dataframe(outliers_df, use_container_width=True)

    # Tab 4: Data-Type & Inconsistencies
    with tab_inconsistent:
        st.write("#### Data-Type & Formatting Inconsistencies")
        inconsistent_issues = issues_df[
            issues_df["Issue Type"].isin(
                [
                    "Data-Type Inconsistency",
                    "Invalid Values",
                    "Categorical Inconsistency",
                    "Inconsistent Boolean Formatting",
                    "Date Parsing Failure",
                ]
            )
        ]
        if inconsistent_issues.empty:
            st.info("No data-type or formatting inconsistencies detected.")
        else:
            st.dataframe(inconsistent_issues, use_container_width=True)

    # Tab 5: Structural & Cardinality
    with tab_cardinality:
        st.write("#### High-Cardinality & Constant Columns")
        struct_issues = issues_df[
            issues_df["Issue Type"].isin(
                ["High Cardinality", "Constant Column", "Near-Constant Column"]
            )
        ]
        if struct_issues.empty:
            st.info("No high-cardinality or constant column issues detected.")
        else:
            st.dataframe(struct_issues, use_container_width=True)
