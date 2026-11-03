"""Streamlit page for Phase 9 Automated Insight Generation and Prioritization."""

import pandas as pd
import streamlit as st

from modules.insight_engine import generate_insights
from modules.quality_engine import analyze_data_quality
from modules.quality_score import calculate_quality_score
from modules.semantic_detector import detect_semantic_types

st.set_page_config(
    page_title="Automated Insights - AutoDataIQ",
    page_icon="💡",
    layout="wide",
)

st.title("💡 Automated Insight Generation & Prioritization")
st.write(
    "AutoDataIQ automatically extracts meaningful, prioritized patterns across numerical, "
    "categorical, temporal, and quality dimensions, explaining them in plain language with "
    "concrete statistical evidence."
)

st.info(
    "⚠️ **Observational Findings & Methodology Disclaimer:** Generated insights represent "
    "statistical observations computed from the uploaded sample. Statistical association or correlation "
    "does not imply causation, and unusual observations do not necessarily constitute errors."
)

if "data" not in st.session_state or st.session_state["data"] is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Upload page first.")
    st.info("💡 Navigate to **Upload** using the sidebar on the left.")
else:
    dataframe: pd.DataFrame = st.session_state["data"]
    filename = st.session_state.get("filename", "Uploaded Dataset")

    st.write(
        f"**Active Dataset:** `{filename}` | **Rows:** {len(dataframe):,} | **Columns:** {len(dataframe.columns):,}"
    )

    # -------------------------------------------------------------
    # 1. Run Semantic, Quality & Insight Generation Engines
    # -------------------------------------------------------------
    semantic_types_df = detect_semantic_types(dataframe)
    quality_report = analyze_data_quality(dataframe, semantic_types_df)
    quality_score_result = calculate_quality_score(dataframe, quality_report)

    insights_result = generate_insights(
        dataframe,
        semantic_types_df=semantic_types_df,
        quality_report=quality_report,
        quality_score_result=quality_score_result,
    )

    summary = insights_result["summary"]
    insights = insights_result["insights"]
    insights_df = insights_result["insights_df"]

    # -------------------------------------------------------------
    # 2. Executive Summary Metrics
    # -------------------------------------------------------------
    st.subheader("📊 Insight Summary Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Insights", summary["total_insights"])
    m2.metric("🔴 High Priority", summary["high_priority"])
    m3.metric("🟠 Medium Priority", summary["medium_priority"])
    m4.metric("🟡 Low Priority", summary["low_priority"])

    # Category pills summary
    cat_counts = summary["category_counts"]
    cat_summary_text = " | ".join(
        [f"**{cat}:** {count}" for cat, count in cat_counts.items() if count > 0]
    )
    if cat_summary_text:
        st.caption(f"Categories Detected: {cat_summary_text}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. Filtering & Search Controls
    # -------------------------------------------------------------
    st.subheader("🔍 Filter & Explore Insights")
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])

    with f_col1:
        priority_options = ["All Priorities", "High", "Medium", "Low"]
        selected_priority = st.selectbox(
            "Priority Level:",
            options=priority_options,
            index=0,
            help="Filter insights by prioritized impact and effect size.",
        )

    with f_col2:
        available_categories = ["All Categories"] + sorted(
            [cat for cat, cnt in cat_counts.items() if cnt > 0]
        )
        selected_category = st.selectbox(
            "Insight Category:",
            options=available_categories,
            index=0,
            help="Filter by analytical domain (Numerical, Categorical, Temporal, Data Quality, General).",
        )

    with f_col3:
        search_term = st.text_input(
            "Search keywords or column name:",
            value="",
            placeholder="e.g. salary, correlation, missing, age...",
        )

    # Apply filters
    filtered_insights = insights

    if selected_priority != "All Priorities":
        filtered_insights = [
            i for i in filtered_insights if i["Priority"] == selected_priority
        ]

    if selected_category != "All Categories":
        filtered_insights = [
            i for i in filtered_insights if i["Category"] == selected_category
        ]

    if search_term.strip():
        term = search_term.strip().lower()
        filtered_insights = [
            i
            for i in filtered_insights
            if term in i["Title"].lower()
            or term in i["Description"].lower()
            or term in i["Evidence"].lower()
            or any(term in str(c).lower() for c in i["Columns"])
        ]

    # -------------------------------------------------------------
    # 4. Display Findings (Cards & Table Tabs)
    # -------------------------------------------------------------
    st.write(
        f"Showing **{len(filtered_insights)}** of **{len(insights)}** insights matching criteria."
    )

    if not filtered_insights:
        st.info("💡 No insights match the specified filter criteria. Try adjusting the filters above.")
    else:
        tab_cards, tab_table = st.tabs(["🗂️ Detailed Insight Cards", "📋 Tabular Summary"])

        with tab_cards:
            priority_emojis = {
                "High": "🔴",
                "Medium": "🟠",
                "Low": "🟡",
            }
            category_badges = {
                "Data Quality": "🛡️ Data Quality",
                "Numerical": "📈 Numerical",
                "Categorical": "🏷️ Categorical",
                "Temporal": "📅 Temporal",
                "General": "🌐 General",
            }

            for idx, item in enumerate(filtered_insights):
                p_emoji = priority_emojis.get(item["Priority"], "⚪")
                c_badge = category_badges.get(item["Category"], item["Category"])
                col_display = ", ".join(item["Columns"]) if item["Columns"] else "Dataset-wide"

                with st.expander(
                    f"{p_emoji} [{item['Priority']}] {item['Title']} ({c_badge})",
                    expanded=(item["Priority"] == "High" or idx < 3),
                ):
                    st.markdown(f"**Explanation:** {item['Description']}")

                    ev_col, det_col = st.columns([1, 1])
                    with ev_col:
                        st.markdown(f"📊 **Supporting Evidence:**")
                        st.code(item["Evidence"], language="text")

                    with det_col:
                        st.markdown(f"📌 **Relevant Column(s):** `{col_display}`")
                        st.markdown(f"⚖️ **Priority Rationale:** {item['Reason']}")

                    if item.get("Warnings"):
                        st.warning(f"⚠️ **Analytical Caveat:** {item['Warnings']}")

        with tab_table:
            # Prepare filtered table
            filtered_rows = []
            for item in filtered_insights:
                filtered_rows.append(
                    {
                        "Priority": f"{priority_emojis.get(item['Priority'], '')} {item['Priority']}",
                        "Category": item["Category"],
                        "Title": item["Title"],
                        "Evidence": item["Evidence"],
                        "Columns": ", ".join(item["Columns"]) if item["Columns"] else "Dataset-wide",
                        "Reason": item["Reason"],
                        "Caveats": item["Warnings"],
                    }
                )
            f_table_df = pd.DataFrame(filtered_rows)
            st.dataframe(f_table_df, use_container_width=True)

            csv_data = f_table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Filtered Insights (CSV)",
                data=csv_data,
                file_name="autodataiq_insights.csv",
                mime="text/csv",
            )
