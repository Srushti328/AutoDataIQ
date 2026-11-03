"""Streamlit page for Phase 12 & Phase 13: Interactive Dashboard Customization."""

import re
from typing import Any
import pandas as pd
import streamlit as st

from modules.dashboard_engine import (
    DEFAULT_SECTION_ORDER,
    apply_dashboard_customization,
    filter_customization_for_dataset,
    generate_dashboard_data,
    get_default_customization_config,
)

st.set_page_config(
    page_title="Executive Dashboard - AutoDataIQ",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Automated Executive Dashboard")
st.write(
    "A unified executive overview dynamically synthesizing automated KPI detection, "
    "recommended visualization patterns, prioritized analytical insights, and data quality health. "
    "Customize card selections, visual choices, and section arrangements to tailor the presentation."
)

st.info(
    "💡 **Interactive Customization:** AutoDataIQ curates high-impact metrics by default. "
    "Expand the **Customization Controls** below to choose which KPIs, charts, and insights are shown, "
    "reorder dashboard sections, or reset back to the default layout anytime."
)

# -------------------------------------------------------------
# Verify Session State Dataset
# -------------------------------------------------------------
if "data" not in st.session_state or st.session_state["data"] is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Upload page first.")
    st.info("💡 Navigate to **Upload** using the sidebar on the left.")
    st.stop()

dataframe: pd.DataFrame = st.session_state["data"]
filename = st.session_state.get("filename", "Uploaded Dataset")
raw_df = st.session_state.get("raw_data")
cleaned_df = st.session_state.get("cleaned_data")
active_dataset_name = st.session_state.get("active_dataset_name", "Original Dataset")

# -------------------------------------------------------------
# Dataset Status Bar & Active Dataset Switcher
# -------------------------------------------------------------
st.subheader("📌 Dataset Status")
header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1, 1, 1])

header_col1.metric("Active Dataset", filename)
header_col2.metric("Total Records", f"{len(dataframe):,}")
header_col3.metric("Total Columns", f"{len(dataframe.columns):,}")
has_cleaned = cleaned_df is not None
header_col4.metric("Cleaned Copy", "Available" if has_cleaned else "Not Created")

# Dataset Switcher if cleaned data exists
if has_cleaned and raw_df is not None:
    switcher_col, refresh_col = st.columns([3, 1])
    with switcher_col:
        current_idx = 1 if active_dataset_name == "Cleaned Dataset" else 0
        chosen_dataset = st.radio(
            "Select active dataset for dashboard:",
            options=["Original Dataset", "Cleaned Dataset"],
            index=current_idx,
            horizontal=True,
            help="Switch between the original raw dataset and the cleaned transformation.",
        )
        if chosen_dataset != active_dataset_name:
            if chosen_dataset == "Cleaned Dataset":
                st.session_state["data"] = cleaned_df
                st.session_state["filename"] = st.session_state.get(
                    "cleaned_filename", f"cleaned_{st.session_state.get('raw_filename', filename)}"
                )
                st.session_state["active_dataset_name"] = "Cleaned Dataset"
            else:
                st.session_state["data"] = raw_df
                st.session_state["filename"] = st.session_state.get("raw_filename", filename)
                st.session_state["active_dataset_name"] = "Original Dataset"
            st.rerun()

    with refresh_col:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            if "cached_dashboard_data" in st.session_state:
                del st.session_state["cached_dashboard_data"]
            st.rerun()
else:
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            if "cached_dashboard_data" in st.session_state:
                del st.session_state["cached_dashboard_data"]
            st.rerun()

# -------------------------------------------------------------
# Generate / Retrieve Cached Dashboard Data (Strictly Read-Only)
# -------------------------------------------------------------
is_cleaned_flag = active_dataset_name == "Cleaned Dataset"
current_dataset_signature = (active_dataset_name, id(dataframe), len(dataframe), len(dataframe.columns))

if (
    "cached_dashboard_data" not in st.session_state
    or st.session_state.get("cached_dataset_signature") != current_dataset_signature
):
    st.session_state["cached_dashboard_data"] = generate_dashboard_data(
        dataframe=dataframe,
        dataset_name=filename,
        is_cleaned=is_cleaned_flag,
        max_kpis=6,
        max_charts=4,
        max_insights=5,
    )
    st.session_state["cached_dataset_signature"] = current_dataset_signature

dashboard_data = st.session_state["cached_dashboard_data"]

if dashboard_data.get("is_empty", False):
    st.warning("⚠️ The loaded dataset contains no rows or columns to construct dashboard content.")
    st.stop()

# -------------------------------------------------------------
# Phase 13: Dashboard Customization State Management
# -------------------------------------------------------------
custom_version = st.session_state.get("custom_version", 0)

if "dashboard_custom_config" not in st.session_state:
    st.session_state["dashboard_custom_config"] = get_default_customization_config(dashboard_data)
else:
    # Safely filter config against the active dataset to remove any invalid/unavailable keys
    st.session_state["dashboard_custom_config"] = filter_customization_for_dataset(
        st.session_state["dashboard_custom_config"], dashboard_data
    )

current_config = st.session_state["dashboard_custom_config"]

# Available candidates in the current dataset
all_kpis_list = dashboard_data.get("kpis", {}).get("all_kpis", [])
all_kpi_names = [k["KPI Name"] for k in all_kpis_list if "KPI Name" in k]
# Also ensure summary cards are present
for k in dashboard_data.get("kpis", {}).get("summary_cards", []):
    if "KPI Name" in k and k["KPI Name"] not in all_kpi_names:
        all_kpi_names.append(k["KPI Name"])

all_charts_list = dashboard_data.get("visualizations", [])
all_chart_titles = [c["title"] for c in all_charts_list if "title" in c]

all_insights_list = dashboard_data.get("insights", {}).get("all_insights", [])
all_insight_titles = [i["Title"] for i in all_insights_list if "Title" in i]
for i in dashboard_data.get("insights", {}).get("highlighted_insights", []):
    if "Title" in i and i["Title"] not in all_insight_titles:
        all_insight_titles.append(i["Title"])

# -------------------------------------------------------------
# Phase 13: Interactive Customization Controls Panel
# -------------------------------------------------------------
with st.expander("🛠️ Dashboard Customization Controls & Section Ordering", expanded=False):
    st.write(
        "Fine-tune the dashboard presentation by selecting which metric cards, visual charts, "
        "and analytical findings to display, and customize the order of major sections."
    )

    tab_order, tab_kpi, tab_viz, tab_ins = st.tabs(
        [
            "📑 Section Arrangement",
            "🎯 KPI Cards",
            "📈 Visualizations",
            "🧠 Key Insights",
        ]
    )

    # Tab 1: Section Ordering & Arrangement
    with tab_order:
        st.markdown("#### Dashboard Section Presentation Order")
        preset_options = [
            "Default (KPIs → Visualizations → Insights → Quality)",
            "Executive Focus (KPIs → Insights → Visualizations → Quality)",
            "Visuals First (Visualizations → KPIs → Insights → Quality)",
            "Quality & Health First (Data Quality Summary → KPIs → Visualizations → Insights)",
            "Custom Section Selection & Order",
        ]

        preset_choice = st.selectbox(
            "Select Section Order Preset:",
            options=preset_options,
            index=0,
            key=f"preset_order_select_{custom_version}",
            help="Choose an executive arrangement preset or specify a custom section order.",
        )

        preset_mapping = {
            "Default (KPIs → Visualizations → Insights → Quality)": [
                "KPI Summary",
                "Visualizations",
                "Key Insights",
                "Data Quality Summary",
            ],
            "Executive Focus (KPIs → Insights → Visualizations → Quality)": [
                "KPI Summary",
                "Key Insights",
                "Visualizations",
                "Data Quality Summary",
            ],
            "Visuals First (Visualizations → KPIs → Insights → Quality)": [
                "Visualizations",
                "KPI Summary",
                "Key Insights",
                "Data Quality Summary",
            ],
            "Quality & Health First (Data Quality Summary → KPIs → Visualizations → Insights)": [
                "Data Quality Summary",
                "KPI Summary",
                "Visualizations",
                "Key Insights",
            ],
        }

        if preset_choice in preset_mapping:
            chosen_section_order = preset_mapping[preset_choice]
            st.caption(f"Active arrangement: **{' → '.join(chosen_section_order)}**")
        else:
            chosen_section_order = st.multiselect(
                "Choose Active Sections in Display Order:",
                options=DEFAULT_SECTION_ORDER,
                default=current_config.get("section_order", DEFAULT_SECTION_ORDER),
                key=f"custom_section_multiselect_{custom_version}",
                help="Select and re-order the sections you want displayed.",
            )
            if not chosen_section_order:
                st.warning("⚠️ At least one section should be selected to render dashboard content.")
                chosen_section_order = list(DEFAULT_SECTION_ORDER)

    # Tab 2: KPI Card Selection
    with tab_kpi:
        st.markdown("#### Select KPI Cards to Display")
        default_kpis = [k for k in current_config.get("selected_kpi_names", []) if k in all_kpi_names]
        selected_kpis_input = st.multiselect(
            "Available Quantitative KPIs:",
            options=all_kpi_names,
            default=default_kpis,
            key=f"kpi_select_{custom_version}",
            help="Select one or more metric cards to feature in the executive summary.",
        )
        st.caption(f"Selected **{len(selected_kpis_input)}** of **{len(all_kpi_names)}** available indicators.")

    # Tab 3: Visualization Selection
    with tab_viz:
        st.markdown("#### Select Visualizations to Display")
        default_charts = [c for c in current_config.get("selected_chart_titles", []) if c in all_chart_titles]
        if all_chart_titles:
            selected_charts_input = st.multiselect(
                "Recommended Interactive Visualizations:",
                options=all_chart_titles,
                default=default_charts,
                key=f"chart_select_{custom_version}",
                help="Select which recommended visual charts to render in the dashboard grid.",
            )
            st.caption(f"Selected **{len(selected_charts_input)}** of **{len(all_chart_titles)}** available charts.")
        else:
            selected_charts_input = []
            st.info("💡 No visualizations supported by the column types in this dataset.")

    # Tab 4: Insight Selection
    with tab_ins:
        st.markdown("#### Select Analytical Insights to Highlight")
        default_insights = [i for i in current_config.get("selected_insight_titles", []) if i in all_insight_titles]
        if all_insight_titles:
            selected_insights_input = st.multiselect(
                "Prioritized Statistical Findings:",
                options=all_insight_titles,
                default=default_insights,
                key=f"insight_select_{custom_version}",
                help="Choose which analytical insights and anomaly notes to feature.",
            )
            st.caption(f"Selected **{len(selected_insights_input)}** of **{len(all_insight_titles)}** available insights.")
        else:
            selected_insights_input = []
            st.info("💡 No analytical insights generated for this dataset.")

    st.markdown("---")
    ctrl_col1, ctrl_col2 = st.columns([1, 2])

    with ctrl_col1:
        if st.button("🔄 Reset to Default Layout", use_container_width=True, help="Restore the automatically generated default layout."):
            st.session_state["dashboard_custom_config"] = get_default_customization_config(dashboard_data)
            st.session_state["custom_version"] = custom_version + 1
            st.success("✅ Dashboard layout reset to default!")
            st.rerun()

    # Determine if current configuration differs from default
    default_config = get_default_customization_config(dashboard_data)
    is_diff = (
        selected_kpis_input != default_config["selected_kpi_names"]
        or selected_charts_input != default_config["selected_chart_titles"]
        or selected_insights_input != default_config["selected_insight_titles"]
        or chosen_section_order != default_config["section_order"]
    )

    with ctrl_col2:
        if is_diff:
            st.info(
                f"🎨 **Active Status: Customized Layout** "
                f"({len(selected_kpis_input)} KPIs, {len(selected_charts_input)} Charts, "
                f"{len(selected_insights_input)} Insights, {len(chosen_section_order)} Sections)"
            )
        else:
            st.caption("✨ **Active Status: Default Automatic Layout** (Curated by AutoDataIQ)")

    # Update active config in session state
    st.session_state["dashboard_custom_config"] = {
        "selected_kpi_names": selected_kpis_input,
        "selected_chart_titles": selected_charts_input,
        "selected_insight_titles": selected_insights_input,
        "section_order": chosen_section_order,
        "is_customized": is_diff,
    }

# -------------------------------------------------------------
# Apply Customization to Dashboard Data (Read-Only)
# -------------------------------------------------------------
customized_dashboard = apply_dashboard_customization(
    dashboard_data, st.session_state["dashboard_custom_config"]
)

priority_emojis = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
category_emojis = {
    "Numerical": "📈",
    "Categorical": "🏷️",
    "Temporal": "📅",
    "Data Quality": "🛡️",
    "General": "📌",
}


# -------------------------------------------------------------
# Section Renderers
# -------------------------------------------------------------
def render_kpis_section() -> None:
    st.markdown("---")
    st.subheader("🎯 Key Performance Indicators (KPIs)")
    curated_kpis = customized_dashboard.get("kpis", {}).get("summary_cards", [])
    total_kpis_count = customized_dashboard.get("kpis", {}).get("total_kpis_detected", len(curated_kpis))

    if curated_kpis:
        for row_start in range(0, len(curated_kpis), 3):
            kpi_row = curated_kpis[row_start : row_start + 3]
            cols = st.columns(len(kpi_row))
            for idx, kpi in enumerate(kpi_row):
                with cols[idx]:
                    p_emoji = priority_emojis.get(kpi.get("Priority", "Medium"), "⚪")
                    st.metric(
                        label=f"{p_emoji} {kpi['KPI Name']}",
                        value=str(kpi["Value"]),
                    )
                    with st.expander("Formula & Details", expanded=False):
                        st.caption(f"**Category:** {kpi.get('Category', 'General')}")
                        st.caption(f"**Formula:** `{kpi.get('Formula', 'N/A')}`")
                        src = (
                            ", ".join(kpi.get("Source Column(s)", []))
                            if kpi.get("Source Column(s)")
                            else "Dataset-wide"
                        )
                        st.caption(f"**Source Column(s):** `{src}`")
                        st.write(kpi.get("Description", ""))
                        if kpi.get("Warnings"):
                            st.warning(f"⚠️ {kpi['Warnings']}")
        st.caption(
            f"Displaying **{len(curated_kpis)}** selected indicators out of **{total_kpis_count}** detected. "
            "Explore all metrics, formulas, and filters in **10_KPIs**."
        )
    else:
        st.info("💡 No KPI cards currently selected for display. Expand **Customization Controls** above to select cards.")


def render_visualizations_section() -> None:
    st.markdown("---")
    st.subheader("📈 Recommended Visualizations")
    curated_charts = customized_dashboard.get("visualizations", [])

    if curated_charts:
        for i in range(0, len(curated_charts), 2):
            chart_cols = st.columns(2)
            for j in range(2):
                if i + j < len(curated_charts):
                    chart_meta = curated_charts[i + j]
                    with chart_cols[j]:
                        st.markdown(f"#### {chart_meta['title']}")
                        st.caption(
                            f"🏷️ Category: **{chart_meta.get('category', 'Overview')}** | "
                            f"Type: **{chart_meta['chart_type']}**"
                        )
                        st.plotly_chart(chart_meta["figure"], use_container_width=True)
                        st.caption(f"💡 {chart_meta['description']}")

                        # Individual interactive chart HTML download
                        try:
                            chart_html = chart_meta["figure"].to_html(include_plotlyjs="cdn")
                            slug = re.sub(r"[^\w\-_\. ]", "_", chart_meta["title"]).strip().replace(" ", "_").lower()
                            st.download_button(
                                label=f"📥 Download Chart (HTML)",
                                data=chart_html,
                                file_name=f"{slug}.html",
                                mime="text/html",
                                key=f"dl_chart_{i + j}_{custom_version}",
                                help="Download an interactive, standalone HTML version of this Plotly figure.",
                            )
                        except Exception as e:
                            st.caption(f"HTML export unavailable: {e}")
        st.caption(
            f"Displaying **{len(curated_charts)}** selected charts. Explore custom charts and configurations in **8_Visualization**."
        )
    else:
        st.info("💡 No visualizations currently selected for display. Expand **Customization Controls** above to select charts.")


def render_insights_section() -> None:
    st.markdown("---")
    st.subheader("🧠 Prioritized Key Insights")
    highlighted_insights = customized_dashboard.get("insights", {}).get("highlighted_insights", [])
    insight_summary = customized_dashboard.get("insights", {}).get("summary", {})
    total_insights = insight_summary.get("total_insights", len(highlighted_insights))

    if highlighted_insights:
        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        i_col1.metric("Selected Findings", len(highlighted_insights))
        i_col2.metric("Total Generated", total_insights)
        i_col3.metric("🔴 High Priority", insight_summary.get("high_priority", 0))
        i_col4.metric("🟠 Medium Priority", insight_summary.get("medium_priority", 0))

        for idx, item in enumerate(highlighted_insights):
            p_badge = priority_emojis.get(item.get("Priority", "Medium"), "⚪")
            c_badge = category_emojis.get(item.get("Category", "General"), "📌")
            title_text = f"{p_badge} [{item.get('Priority', 'Medium')}] {item.get('Title', 'Analytical Finding')} ({c_badge} {item.get('Category', 'General')})"

            with st.expander(title_text, expanded=(idx < 2)):
                st.markdown(f"**Explanation:** {item.get('Description', '')}")

                evidence_str = item.get("Evidence")
                if evidence_str:
                    st.markdown(f"📊 **Supporting Evidence:** `{evidence_str}`")

                if item.get("Reason"):
                    st.info(f"💡 **Analytical Context:** {item['Reason']}")
                if item.get("Warnings"):
                    st.caption(f"⚠️ *Caveat:* {item['Warnings']}")

        st.caption(
            f"Displaying **{len(highlighted_insights)}** selected insights out of **{total_insights}** findings. "
            "Review full analytical details and category filters in **9_Insights**."
        )
    else:
        st.info("💡 No insights currently selected for display. Expand **Customization Controls** above to select insights.")


def render_quality_section() -> None:
    st.markdown("---")
    st.subheader("🛡️ Data Quality & Health Summary")
    quality_summary = customized_dashboard.get("quality_summary", {})
    overall_score = quality_summary.get("overall_score", 100.0)
    rating = quality_summary.get("rating", "Good")
    missing_cells = quality_summary.get("missing_cells", 0)
    missing_pct = quality_summary.get("missing_percentage", 0.0)
    dup_rows = quality_summary.get("duplicate_rows", 0)
    dup_pct = quality_summary.get("duplicate_percentage", 0.0)
    total_issues = quality_summary.get("total_issues", 0)

    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    q_col1.metric("Quality Score", f"{overall_score:.1f} / 100", f"Rating: {rating}")
    q_col2.metric("Missing Values", f"{missing_cells:,}", f"{missing_pct:.2f}% of cells")
    q_col3.metric("Duplicate Rows", f"{dup_rows:,}", f"{dup_pct:.2f}% of rows")
    q_col4.metric("Detected Issues", f"{total_issues:,}")

    top_warnings = quality_summary.get("top_warnings", [])
    if top_warnings:
        with st.expander("⚠️ Active Data Quality Warnings", expanded=True):
            for w in top_warnings:
                st.warning(f"• {w}")
    else:
        st.success("✅ Excellent data hygiene. No critical data quality issues detected.")

    st.caption("Complete 6-dimension quality evaluation available in **4_Quality**.")


# -------------------------------------------------------------
# Execute Customized Section Rendering Order
# -------------------------------------------------------------
section_dispatch = {
    "KPI Summary": render_kpis_section,
    "Visualizations": render_visualizations_section,
    "Key Insights": render_insights_section,
    "Data Quality Summary": render_quality_section,
}

active_sections = customized_dashboard.get("customization", {}).get("section_order", DEFAULT_SECTION_ORDER)
for section_name in active_sections:
    renderer = section_dispatch.get(section_name)
    if renderer:
        renderer()

# -------------------------------------------------------------
# Section 5: Data Limitations & Guidance
# -------------------------------------------------------------
limitations = customized_dashboard.get("limitations", [])
if limitations:
    st.markdown("---")
    st.subheader("ℹ️ Dataset Limitations & Analysis Guidance")
    for lim in limitations:
        st.info(f"📌 {lim}")

# -------------------------------------------------------------
# Section 6: Export Customized Dashboard Artifacts
# -------------------------------------------------------------
st.markdown("---")
st.subheader("📥 Export Dashboard Artifacts")
st.write(
    f"Download tabular summaries and chart assets curated from **{filename}** "
    f"(`{active_dataset_name}`)."
)

exp_col1, exp_col2 = st.columns(2)

# Export Displayed KPIs
displayed_kpis = customized_dashboard.get("kpis", {}).get("summary_cards", [])
with exp_col1:
    if displayed_kpis:
        kpi_rows = []
        for k in displayed_kpis:
            kpi_rows.append(
                {
                    "Dataset": f"{filename} ({active_dataset_name})",
                    "Priority": k.get("Priority", ""),
                    "Category": k.get("Category", ""),
                    "KPI Name": k.get("KPI Name", ""),
                    "Value": str(k.get("Value", "")),
                    "Source Columns": ", ".join(k.get("Source Column(s)", []))
                    if k.get("Source Column(s)")
                    else "Dataset-wide",
                    "Formula": k.get("Formula", ""),
                    "Description": k.get("Description", ""),
                    "Warnings": k.get("Warnings", ""),
                }
            )
        kpi_export_df = pd.DataFrame(kpi_rows)
        kpi_csv_bytes = kpi_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Displayed KPIs (CSV)",
            data=kpi_csv_bytes,
            file_name=f"{filename.rsplit('.', 1)[0]}_displayed_kpis.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("No KPIs selected to export.")

# Export Displayed Insights
displayed_insights = customized_dashboard.get("insights", {}).get("highlighted_insights", [])
with exp_col2:
    if displayed_insights:
        insight_rows = []
        for i in displayed_insights:
            insight_rows.append(
                {
                    "Dataset": f"{filename} ({active_dataset_name})",
                    "Priority": i.get("Priority", ""),
                    "Category": i.get("Category", ""),
                    "Title": i.get("Title", ""),
                    "Description": i.get("Description", ""),
                    "Evidence": i.get("Evidence", ""),
                    "Reason": i.get("Reason", ""),
                    "Warnings": i.get("Warnings", ""),
                }
            )
        insight_export_df = pd.DataFrame(insight_rows)
        insight_csv_bytes = insight_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Displayed Insights (CSV)",
            data=insight_csv_bytes,
            file_name=f"{filename.rsplit('.', 1)[0]}_displayed_insights.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("No insights selected to export.")

st.markdown("---")
st.caption(
    "🧭 **Deep-Dive Navigation:** "
    "[4_Quality] Quality Report | [5_Cleaning] Cleaning Recommendations | [6_Clean_Data] Clean Data | "
    "[7_Analysis] Adaptive Analysis | [8_Visualization] Automated Visualizations | [9_Insights] Insights | [10_KPIs] Automatic KPIs"
)
