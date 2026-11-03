"""Streamlit page for Phase 8 Automated Visualization Selection & Visualization Engine."""

import pandas as pd
import streamlit as st

from modules.analysis_engine import classify_columns_semantically
from modules.quality_engine import analyze_data_quality
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

st.set_page_config(
    page_title="Automated Visualizations - AutoDataIQ",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Automated Visualizations")
st.write(
    "AutoDataIQ automatically recommends and renders suitable interactive Plotly "
    "visualizations based on the structural and semantic characteristics of your dataset."
)

st.info(
    "⚠️ **Read-Only Visualizations:** All charts are rendered in-memory using temporary "
    "aggregations and do not alter your raw dataset."
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
    # Run Semantic Detector, Quality Report & Visualization Plan
    # -------------------------------------------------------------
    semantic_types_df = detect_semantic_types(dataframe)
    classified_cols = classify_columns_semantically(dataframe, semantic_types_df)
    quality_report = analyze_data_quality(dataframe, semantic_types_df)
    viz_plan = get_visualization_plan(dataframe, semantic_types_df)

    num_cols = classified_cols["numerical"]
    cat_cols = classified_cols["categorical"]
    date_cols = classified_cols["date"]
    bool_cols = classified_cols["boolean"]
    id_cols = classified_cols["identifier"]

    # Quality warnings dictionary
    quality_warnings: dict[str, list[str]] = {}
    if "issues_df" in quality_report and not quality_report["issues_df"].empty:
        for _, issue in quality_report["issues_df"].iterrows():
            col = str(issue["Column"])
            if col != "Dataset-level":
                if col not in quality_warnings:
                    quality_warnings[col] = []
                quality_warnings[col].append(f"{issue['Issue Type']}: {issue['Description']}")

    # -------------------------------------------------------------
    # 1. Visualization Recommendations Plan Overview
    # -------------------------------------------------------------
    st.subheader("Visualization Recommendations Plan")
    st.caption("AutoDataIQ evaluates column semantics to determine recommended Plotly charts.")

    plan_df = pd.DataFrame(viz_plan)
    if not plan_df.empty:
        filter_priority = st.selectbox(
            "Filter Recommendations by Priority:",
            options=["All"] + sorted(list(plan_df["priority"].unique())),
        )
        filtered_plan = plan_df.copy()
        if filter_priority != "All":
            filtered_plan = filtered_plan[filtered_plan["priority"] == filter_priority]

        st.dataframe(
            filtered_plan[["chart_type", "category", "columns", "priority", "reason"]],
            use_container_width=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. Interactive Charts Section
    # -------------------------------------------------------------
    st.subheader("Generated Interactive Visualizations")

    tab_dist, tab_cat, tab_rel, tab_trend, tab_group = st.tabs(
        [
            "Distributions",
            "Categories & Flags",
            "Relationships & Correlation",
            "Temporal Trends",
            "Group Comparisons",
        ]
    )

    # Tab 1: Numerical Distributions (Histogram & Box Plot)
    with tab_dist:
        st.write("#### Numerical Distribution Charts")
        if not num_cols:
            st.info("No suitable numerical columns available for distribution charts (Identifiers excluded).")
        else:
            sel_num_col = st.selectbox("Select Numerical Column:", options=num_cols, key="viz_num_sel")

            if sel_num_col:
                if sel_num_col in quality_warnings:
                    for warn in quality_warnings[sel_num_col]:
                        st.warning(f"⚠️ Quality Warning for '{sel_num_col}': {warn}")

                col_h, col_b = st.columns(2)
                with col_h:
                    fig_hist = generate_histogram(dataframe, sel_num_col)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_b:
                    fig_box = generate_boxplot(dataframe, sel_num_col)
                    st.plotly_chart(fig_box, use_container_width=True)

    # Tab 2: Categories & Boolean Flags
    with tab_cat:
        st.write("#### Category & Boolean Flag Distributions")
        available_cat_bool = cat_cols + bool_cols
        if not available_cat_bool:
            st.info("No categorical or boolean columns available for frequency charts.")
        else:
            sel_cat_col = st.selectbox("Select Column:", options=available_cat_bool, key="viz_cat_sel")

            if sel_cat_col:
                if sel_cat_col in quality_warnings:
                    for warn in quality_warnings[sel_cat_col]:
                        st.warning(f"⚠️ Quality Warning for '{sel_cat_col}': {warn}")

                if sel_cat_col in bool_cols:
                    fig_bool = generate_boolean_barchart(dataframe, sel_cat_col)
                    st.plotly_chart(fig_bool, use_container_width=True)
                else:
                    fig_cat = generate_categorical_barchart(dataframe, sel_cat_col, top_n=10)
                    st.plotly_chart(fig_cat, use_container_width=True)

    # Tab 3: Relationships & Correlation
    with tab_rel:
        st.write("#### Bivariate Relationships & Correlation")
        if len(num_cols) < 2:
            st.info("At least 2 numerical columns are required for scatter plots and correlation heatmaps.")
        else:
            col_rel1, col_rel2 = st.columns(2)
            with col_rel1:
                st.write("##### Pearson Correlation Heatmap")
                fig_corr = generate_correlation_heatmap(dataframe, num_cols)
                st.plotly_chart(fig_corr, use_container_width=True)

            with col_rel2:
                st.write("##### Bivariate Scatter Plot")
                scat_x = st.selectbox("Select X-Axis Column:", options=num_cols, index=0, key="viz_scat_x")
                scat_y = st.selectbox(
                    "Select Y-Axis Column:",
                    options=num_cols,
                    index=1 if len(num_cols) > 1 else 0,
                    key="viz_scat_y",
                )
                if scat_x and scat_y:
                    fig_scat = generate_scatterplot(dataframe, scat_x, scat_y)
                    st.plotly_chart(fig_scat, use_container_width=True)

    # Tab 4: Temporal Trends
    with tab_trend:
        st.write("#### Time-Series Trend Charts")
        if not date_cols:
            st.info("No date/datetime columns available for time-series trend charts.")
        else:
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                sel_date_col = st.selectbox("Select Date Column:", options=date_cols, key="viz_date_sel")
            with t_col2:
                sel_num_trend = st.selectbox(
                    "Select Numerical Measure (Optional):",
                    options=["None (Record Count Only)"] + num_cols,
                    key="viz_date_num_sel",
                )
            with t_col3:
                sel_agg = st.selectbox(
                    "Aggregation:",
                    options=["Mean", "Sum", "Median", "Count"],
                    key="viz_date_agg_sel",
                )

            target_num_trend = None if sel_num_trend == "None (Record Count Only)" else sel_num_trend

            if sel_date_col:
                fig_line = generate_line_chart(dataframe, sel_date_col, target_num_trend, sel_agg)
                st.plotly_chart(fig_line, use_container_width=True)

    # Tab 5: Group Comparisons
    with tab_group:
        st.write("#### Categorical Group Comparisons")
        if not cat_cols or not num_cols:
            st.info("Requires at least 1 categorical column and 1 numerical column for group comparisons.")
        else:
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                grp_cat = st.selectbox("Select Category Column:", options=cat_cols, key="viz_grp_cat")
            with g_col2:
                grp_num = st.selectbox("Select Numerical Measure:", options=num_cols, key="viz_grp_num")
            with g_col3:
                grp_agg = st.selectbox(
                    "Aggregation Method:",
                    options=["Mean", "Sum", "Median", "Count"],
                    key="viz_grp_agg",
                )

            if grp_cat and grp_num:
                fig_grp = generate_grouped_barchart(dataframe, grp_cat, grp_num, grp_agg, top_n=10)
                st.plotly_chart(fig_grp, use_container_width=True)
