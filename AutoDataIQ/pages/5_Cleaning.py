"""Streamlit page for Phase 6 Intelligent Cleaning Recommendations."""

import pandas as pd
import streamlit as st

from modules.cleaning_engine import generate_cleaning_recommendations
from modules.quality_engine import analyze_data_quality

st.set_page_config(
    page_title="Cleaning Recommendations - AutoDataIQ",
    page_icon="💡",
    layout="wide",
)

st.title("💡 Intelligent Cleaning Recommendations")
st.write(
    "Automated, rule-based recommendations specifying **what** should be cleaned, "
    "**why** it should be cleaned, and **what methods** can be used."
)

st.info(
    "⚠️ **Read-Only Guidance:** AutoDataIQ does not silently modify your raw dataset. "
    "All recommendations presented here serve as analytical guidance for user review."
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
    # Run Recommendation Engine
    # -------------------------------------------------------------
    quality_report = analyze_data_quality(dataframe)
    rec_report = generate_cleaning_recommendations(dataframe, quality_report)

    summary = rec_report["summary"]
    recs_df: pd.DataFrame = rec_report["recommendations_df"]
    recs_list: list = rec_report["recommendations_list"]

    # -------------------------------------------------------------
    # 1. Overview Priority Cards
    # -------------------------------------------------------------
    st.subheader("Recommendations Summary")
    p_counts = summary["priority_counts"]
    m_cols = st.columns(5)
    m_cols[0].metric("Total Recommendations", summary["total_recommendations"])
    m_cols[1].metric("🔴 High Priority", p_counts["High"])
    m_cols[2].metric("🟠 Medium Priority", p_counts["Medium"])
    m_cols[3].metric("🟡 Low Priority", p_counts["Low"])
    m_cols[4].metric("🔵 Info Priority", p_counts["Info"])

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. Filterable Recommendations Table
    # -------------------------------------------------------------
    st.subheader("Recommended Cleaning Actions")

    if recs_df.empty:
        st.success("🎉 No data cleaning issues or recommendations found for this dataset!")
    else:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            p_options = ["All"] + sorted(list(recs_df["Priority"].unique()))
            selected_priority = st.selectbox("Filter by Priority:", p_options)
        with filter_col2:
            c_options = ["All"] + sorted(list(recs_df["Category"].unique()))
            selected_category = st.selectbox("Filter by Category:", c_options)

        filtered_df = recs_df.copy()
        if selected_priority != "All":
            filtered_df = filtered_df[filtered_df["Priority"] == selected_priority]
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["Category"] == selected_category]

        st.dataframe(
            filtered_df[
                ["Column", "Issue Type", "Category", "Priority", "Recommended Action", "Method", "Reason"]
            ],
            use_container_width=True,
        )

        st.markdown("---")

        # -------------------------------------------------------------
        # 3. Detailed Proposed Action Breakdown ("What Will Change")
        # -------------------------------------------------------------
        st.subheader("Detailed Action Proposals & Expected Effects")
        st.caption("Expand any recommendation to view the current situation, proposed method, and expected impact.")

        for idx, rec in enumerate(recs_list):
            # Apply filter matching to cards
            if selected_priority != "All" and rec["Priority"] != selected_priority:
                continue
            if selected_category != "All" and rec["Category"] != selected_category:
                continue

            p_badge = {
                "High": "🔴 High",
                "Medium": "🟠 Medium",
                "Low": "🟡 Low",
                "Info": "🔵 Info",
            }.get(rec["Priority"], rec["Priority"])

            expander_title = f"{p_badge} | Col: **{rec['Column']}** — {rec['Recommended Action']} ({rec['Issue Type']})"
            
            with st.expander(expander_title, expanded=False):
                card_col1, card_col2, card_col3 = st.columns(3)
                with card_col1:
                    st.markdown("**📍 Current Situation**")
                    st.info(rec["Current Situation"])
                with card_col2:
                    st.markdown("**🛠️ Proposed Action & Method**")
                    st.success(f"**Method:** {rec['Method']}\n\n{rec['Proposed Action']}")
                with card_col3:
                    st.markdown("**📈 Expected Effect**")
                    st.warning(rec["Expected Effect"])

                # Review Checkbox for analyst tracking (Pure UI state)
                st.checkbox("Mark as reviewed", key=f"rec_review_{idx}")
