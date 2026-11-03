"""Streamlit page for Phase 7 Adaptive Data Analysis."""

import pandas as pd
import streamlit as st

from modules.analysis_engine import (
    analyze_group_comparison,
    analyze_time_trend,
    run_adaptive_analysis,
)
from modules.quality_engine import analyze_data_quality

st.set_page_config(
    page_title="Adaptive Analysis - AutoDataIQ",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Adaptive Data Analysis")
st.write(
    "AutoDataIQ automatically selects appropriate analysis techniques based on the "
    "detected structural and semantic characteristics of your uploaded dataset."
)

st.info(
    "⚠️ **Read-Only Analysis:** All exploratory calculations are performed in-memory "
    "and do not alter your raw dataset."
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
    # Run Quality Report & Adaptive Analysis Engine
    # -------------------------------------------------------------
    quality_report = analyze_data_quality(dataframe)
    analysis_data = run_adaptive_analysis(dataframe, quality_report=quality_report)

    plan = analysis_data["plan"]
    classified_cols = analysis_data["classified_cols"]
    results = analysis_data["results"]
    quality_warnings = analysis_data["quality_warnings"]

    num_cols = classified_cols["numerical"]
    cat_cols = classified_cols["categorical"]
    date_cols = classified_cols["date"]
    bool_cols = classified_cols["boolean"]
    text_cols = classified_cols["text"]
    id_cols = classified_cols["identifier"]

    # -------------------------------------------------------------
    # 1. Dataset Summary & Column Classification Metrics
    # -------------------------------------------------------------
    st.subheader("Dataset Structure Summary")
    sum_cols = st.columns(6)
    sum_cols[0].metric("Total Rows", len(dataframe))
    sum_cols[1].metric("Numerical Cols", len(num_cols))
    sum_cols[2].metric("Categorical Cols", len(cat_cols))
    sum_cols[3].metric("Date Cols", len(date_cols))
    sum_cols[4].metric("Boolean Cols", len(bool_cols))
    sum_cols[5].metric("Identifier Cols", len(id_cols))

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. Adaptive Analysis Plan Overview
    # -------------------------------------------------------------
    st.subheader("Adaptive Analysis Plan")
    st.caption("AutoDataIQ evaluates available features to determine applicable exploratory analysis modules.")

    plan_rows = []
    for item in plan:
        status_str = "✓ Available" if item["available"] else "✗ Not Applicable"
        plan_rows.append(
            {
                "Analysis Module": item["name"],
                "Status": status_str,
                "Applicability Reason": item["reason"],
            }
        )

    plan_df = pd.DataFrame(plan_rows)
    st.dataframe(plan_df, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. Dynamic Tabs for Available Analyses
    # -------------------------------------------------------------
    st.subheader("Exploratory Analysis Results")

    # Filter available plan items
    available_items = [p for p in plan if p["available"]]

    if not available_items:
        st.warning("No analysis modules are applicable for the current dataset structure.")
    else:
        tab_titles = [p["name"] for p in available_items]
        tabs = st.tabs(tab_titles)

        for idx, p in enumerate(available_items):
            key = p["key"]
            tab = tabs[idx]

            with tab:
                # A. Dataset Overview & Structure
                if key == "dataset_summary":
                    st.write("#### Dataset Structure & Types Overview")
                    summary_df = pd.DataFrame(
                        {
                            "Column": list(dataframe.columns),
                            "Pandas Dtype": [str(dataframe[c].dtype) for c in dataframe.columns],
                            "Non-Null Count": [dataframe[c].notna().sum() for c in dataframe.columns],
                            "Missing %": [
                                f"{(dataframe[c].isna().sum() / len(dataframe) * 100):.1f}%"
                                for c in dataframe.columns
                            ],
                            "Semantic Type": [
                                "Identifier"
                                if c in id_cols
                                else (
                                    "Numerical"
                                    if c in num_cols
                                    else (
                                        "Categorical"
                                        if c in cat_cols
                                        else (
                                            "Date/Datetime"
                                            if c in date_cols
                                            else (
                                                "Boolean"
                                                if c in bool_cols
                                                else ("Text" if c in text_cols else "Unknown")
                                            )
                                        )
                                    )
                                )
                                for c in dataframe.columns
                            ],
                        }
                    )
                    st.dataframe(summary_df, use_container_width=True)

                # B. Numerical Univariate Analysis
                elif key == "numerical_analysis":
                    st.write("#### Numerical Univariate Statistics")
                    st.caption(
                        "Descriptive statistics for quantitative numerical variables. "
                        "Note: Identifier columns are excluded."
                    )
                    num_res_df = results.get("numerical_analysis", pd.DataFrame())
                    if not num_res_df.empty:
                        st.dataframe(num_res_df, use_container_width=True)
                        
                        # Show quality warnings if any exist for numerical columns
                        for col in num_cols:
                            if col in quality_warnings:
                                with st.expander(f"⚠️ Quality Warning for '{col}'", expanded=False):
                                    for warn in quality_warnings[col]:
                                        st.warning(warn)

                # C. Categorical Frequency Analysis
                elif key == "categorical_analysis":
                    st.write("#### Categorical Frequency Distributions")
                    cat_data = results.get("categorical_analysis", {})
                    cat_summary_df = cat_data.get("summary_df", pd.DataFrame())
                    breakdowns = cat_data.get("breakdowns", {})

                    if not cat_summary_df.empty:
                        st.dataframe(cat_summary_df, use_container_width=True)
                        st.markdown("##### Top Category Frequency Breakdowns")
                        for col, b_df in breakdowns.items():
                            with st.expander(f"Frequency Breakdown: **{col}**", expanded=False):
                                if col in quality_warnings:
                                    for warn in quality_warnings[col]:
                                        st.warning(warn)
                                st.dataframe(b_df, use_container_width=True)

                # D. Date & Temporal Range Analysis
                elif key == "date_analysis":
                    st.write("#### Date & Temporal Range Analysis")
                    date_data = results.get("date_analysis", {})
                    d_summary_df = date_data.get("summary_df", pd.DataFrame())
                    distributions = date_data.get("distributions", {})

                    if not d_summary_df.empty:
                        st.dataframe(d_summary_df, use_container_width=True)
                        st.markdown("##### Yearly Record Distributions")
                        for col, y_df in distributions.items():
                            with st.expander(f"Yearly Breakdown: **{col}**", expanded=False):
                                st.dataframe(y_df, use_container_width=True)

                # E. Boolean Flag Distribution
                elif key == "boolean_analysis":
                    st.write("#### Boolean Flag Distributions")
                    bool_res_df = results.get("boolean_analysis", pd.DataFrame())
                    if not bool_res_df.empty:
                        st.dataframe(bool_res_df, use_container_width=True)

                # F. Free-Form Text Metrics
                elif key == "text_analysis":
                    st.write("#### Free-Form Text Statistics")
                    text_res_df = results.get("text_analysis", pd.DataFrame())
                    if not text_res_df.empty:
                        st.dataframe(text_res_df, use_container_width=True)

                # G. Identifier Uniqueness Analysis
                elif key == "identifier_analysis":
                    st.write("#### Identifier Uniqueness & Key Analysis")
                    st.caption(
                        "Identifier columns contain primary keys or transaction codes. "
                        "They are excluded from mathematical mean/correlation calculations."
                    )
                    id_res_df = results.get("identifier_analysis", pd.DataFrame())
                    if not id_res_df.empty:
                        st.dataframe(id_res_df, use_container_width=True)

                # H. Pearson Correlation Matrix
                elif key == "correlation_analysis":
                    st.write("#### Pearson Pairwise Correlation Matrix")
                    st.caption(
                        "Values range from -1.0 (strong negative correlation) to +1.0 (strong positive correlation). "
                        "Note: Correlation indicates linear association, not causation."
                    )
                    corr_df = results.get("correlation_analysis", pd.DataFrame())
                    if not corr_df.empty:
                        st.dataframe(corr_df, use_container_width=True)

                # I. Categorical Group Comparison
                elif key == "group_comparison":
                    st.write("#### Categorical Group Comparison")
                    st.caption("Compare numerical metrics aggregated across categories.")

                    sel_cat_col = st.selectbox(
                        "Select Categorical Column:",
                        options=cat_cols,
                        key="group_comp_cat_sel",
                    )
                    sel_num_col = st.selectbox(
                        "Select Numerical Column:",
                        options=num_cols,
                        key="group_comp_num_sel",
                    )

                    if sel_cat_col and sel_num_col:
                        group_res_df = analyze_group_comparison(
                            dataframe, sel_cat_col, sel_num_col
                        )
                        if not group_res_df.empty:
                            st.write(
                                f"**Grouped Summary:** `{sel_num_col}` by `{sel_cat_col}`"
                            )
                            st.dataframe(group_res_df, use_container_width=True)
                        else:
                            st.info("No group comparison statistics could be calculated.")

                # J. Time-based Numerical Trend Summary
                elif key == "time_trend":
                    st.write("#### Time-Based Monthly Trend Aggregation")
                    st.caption("Summarize record counts or numerical aggregations over monthly time periods.")

                    sel_date_col = st.selectbox(
                        "Select Date Column:",
                        options=date_cols,
                        key="time_trend_date_sel",
                    )
                    sel_num_trend_col = st.selectbox(
                        "Select Numerical Measure (Optional):",
                        options=["None (Record Count Only)"] + num_cols,
                        key="time_trend_num_sel",
                    )

                    target_num = (
                        None
                        if sel_num_trend_col == "None (Record Count Only)"
                        else sel_num_trend_col
                    )

                    if sel_date_col:
                        trend_df = analyze_time_trend(
                            dataframe, sel_date_col, target_num
                        )
                        if not trend_df.empty:
                            measure_label = (
                                f"`{target_num}`" if target_num else "Record Count"
                            )
                            st.write(
                                f"**Monthly Trend Summary:** {measure_label} by `{sel_date_col}`"
                            )
                            st.dataframe(trend_df, use_container_width=True)
                        else:
                            st.info("No valid parsed dates available for monthly trend analysis.")
