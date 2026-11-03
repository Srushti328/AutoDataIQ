"""Streamlit page for Phase 10 Automatic KPI Detection & Metrics."""

import pandas as pd
import streamlit as st
from typing import Any

from modules.kpi_engine import detect_kpis
from modules.semantic_detector import detect_semantic_types

st.set_page_config(
    page_title="Automatic KPI Detection - AutoDataIQ",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Automatic KPI Detection & Metrics")
st.write(
    "AutoDataIQ automatically identifies measurable quantitative indicators across dataset "
    "overview, potential business domains, numerical distributions, categorical proportions, "
    "and temporal spans with transparent formulas and metadata."
)

st.info(
    "⚠️ **KPI Transparency & Domain Boundaries:** Indicators labeled *Potential Business* are "
    "inferred from column nomenclature and quantitative traits. They do not presume unverified "
    "currencies, targets, or business benchmarks. All metrics are computed strictly read-only."
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
    # 1. Run Semantic Detector & KPI Detection Engine
    # -------------------------------------------------------------
    semantic_types_df = detect_semantic_types(dataframe)
    kpi_results = detect_kpis(dataframe, semantic_types_df=semantic_types_df)

    summary = kpi_results["summary"]
    overview_metrics = kpi_results["overview_metrics"]
    kpis = kpi_results["kpis"]
    kpis_df = kpi_results["kpis_df"]

    # -------------------------------------------------------------
    # 2. Executive Overview Metrics Cards
    # -------------------------------------------------------------
    st.subheader("📌 Dataset-Level Overview Metrics")
    o_cols = st.columns(4)
    o_cols[0].metric("Total Records", f"{overview_metrics['total_records']:,}")
    o_cols[1].metric("Total Attributes", f"{overview_metrics['total_columns']:,}")
    o_cols[2].metric(
        "Missing Cells",
        f"{overview_metrics['missing_cells']:,} ({overview_metrics['missing_percentage']}%)",
    )
    o_cols[3].metric(
        "Duplicate Rows",
        f"{overview_metrics['duplicate_rows']:,} ({overview_metrics['duplicate_percentage']}%)",
    )

    # Show unique entity cards if present
    unique_entities = overview_metrics.get("unique_entities", {})
    if unique_entities:
        id_cards = st.columns(min(len(unique_entities), 4))
        for idx, (id_col, count) in enumerate(unique_entities.items()):
            col_target = id_cards[idx % len(id_cards)]
            col_target.metric(f"Distinct '{id_col}'", f"{count:,}")

    # Summary metrics breakdown
    st.subheader("📊 KPI Detection Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total KPIs Detected", summary["total_kpis"])
    m2.metric("🔴 High Priority", summary["high_priority"])
    m3.metric("🟠 Medium Priority", summary["medium_priority"])
    m4.metric("🟡 Low Priority", summary["low_priority"])

    cat_counts = summary["category_counts"]
    cat_summary_text = " | ".join(
        [f"**{cat}:** {count}" for cat, count in cat_counts.items() if count > 0]
    )
    if cat_summary_text:
        st.caption(f"Detected Categories: {cat_summary_text}")

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. Filtering Controls
    # -------------------------------------------------------------
    st.subheader("🔍 Filter & Explore KPIs")
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])

    with f_col1:
        priority_options = ["All Priorities", "High", "Medium", "Low"]
        selected_priority = st.selectbox(
            "Priority Filter:",
            options=priority_options,
            index=0,
            help="Filter KPIs by priority/relevance level.",
        )

    with f_col2:
        available_categories = ["All Categories"] + sorted(
            [cat for cat, cnt in cat_counts.items() if cnt > 0]
        )
        selected_category = st.selectbox(
            "Category Filter:",
            options=available_categories,
            index=0,
            help="Filter by indicator domain.",
        )

    with f_col3:
        search_query = st.text_input(
            "Search KPI name, column, or formula:",
            value="",
            placeholder="e.g. revenue, average, date, records...",
        )

    # Apply filters
    filtered_kpis = kpis

    if selected_priority != "All Priorities":
        filtered_kpis = [k for k in filtered_kpis if k["Priority"] == selected_priority]

    if selected_category != "All Categories":
        filtered_kpis = [k for k in filtered_kpis if k["Category"] == selected_category]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_kpis = [
            k
            for k in filtered_kpis
            if q in k["KPI Name"].lower()
            or q in k["Description"].lower()
            or q in k["Formula"].lower()
            or any(q in str(c).lower() for c in k["Source Column(s)"])
        ]

    st.write(f"Displaying **{len(filtered_kpis)}** of **{len(kpis)}** detected KPIs.")

    # -------------------------------------------------------------
    # 4. Tabbed KPI Presentations
    # -------------------------------------------------------------
    priority_emojis = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
    category_badges = {
        "Overview": "🌐 Overview",
        "Potential Business": "💼 Potential Business",
        "Statistical Numerical": "📈 Numerical",
        "Statistical Categorical": "🏷️ Categorical",
        "Temporal": "📅 Temporal",
    }

    tab_business_num, tab_cat_date, tab_all_table = st.tabs(
        [
            "💼 Business & Numerical KPIs",
            "🏷️ Categorical & Temporal KPIs",
            "📋 Complete KPI Master Table",
        ]
    )

    # Helper function to render KPI cards
    def render_kpi_cards(kpi_subset: list[dict[str, Any]]) -> None:
        if not kpi_subset:
            st.info("💡 No indicators available in this section matching your filters.")
            return

        for idx, item in enumerate(kpi_subset):
            p_emoji = priority_emojis.get(item["Priority"], "⚪")
            c_badge = category_badges.get(item["Category"], item["Category"])
            src_cols = (
                ", ".join(item["Source Column(s)"])
                if item["Source Column(s)"]
                else "Dataset-wide"
            )

            with st.expander(
                f"{p_emoji} [{item['Priority']}] {item['KPI Name']} : {item['Value']} ({c_badge})",
                expanded=(item["Priority"] == "High" or idx < 2),
            ):
                st.markdown(f"**Description:** {item['Description']}")

                v_col, f_col = st.columns([1, 1])
                with v_col:
                    st.metric(label=item["KPI Name"], value=item["Value"])
                    st.markdown(f"📌 **Source Column(s):** `{src_cols}`")

                with f_col:
                    st.markdown("📐 **Calculation Formula / Logic:**")
                    st.code(item["Formula"], language="sql")
                    if item.get("Unit"):
                        st.markdown(f"📏 **Unit:** `{item['Unit']}`")

                if item.get("Warnings"):
                    st.warning(f"⚠️ **Caveat / Context:** {item['Warnings']}")

    with tab_business_num:
        st.subheader("💼 Potential Business & Numerical Indicators")
        biz_num_subset = [
            k
            for k in filtered_kpis
            if k["Category"] in {"Potential Business", "Statistical Numerical"}
        ]
        render_kpi_cards(biz_num_subset)

    with tab_cat_date:
        st.subheader("🏷️ Categorical & Temporal Indicators")
        cat_date_subset = [
            k
            for k in filtered_kpis
            if k["Category"] in {"Statistical Categorical", "Temporal", "Overview"}
        ]
        render_kpi_cards(cat_date_subset)

    with tab_all_table:
        st.subheader("📋 Detected KPIs Master Table")
        if filtered_kpis:
            table_display_rows = []
            for k in filtered_kpis:
                table_display_rows.append(
                    {
                        "Priority": f"{priority_emojis.get(k['Priority'], '')} {k['Priority']}",
                        "Category": k["Category"],
                        "KPI Name": k["KPI Name"],
                        "Value": str(k["Value"]),
                        "Source Column(s)": ", ".join(k["Source Column(s)"])
                        if k["Source Column(s)"]
                        else "Dataset-wide",
                        "Formula": k["Formula"],
                        "Description": k["Description"],
                        "Warnings": k["Warnings"] if k["Warnings"] else "None",
                    }
                )
            tbl_df = pd.DataFrame(table_display_rows)
            st.dataframe(tbl_df, use_container_width=True)

            csv_export = tbl_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Detected KPIs (CSV)",
                data=csv_export,
                file_name="autodataiq_detected_kpis.csv",
                mime="text/csv",
            )
        else:
            st.info("💡 No KPIs to display in table.")
