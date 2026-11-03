"""Streamlit page for Phase 3 semantic data-type detection."""

import pandas as pd
import streamlit as st

from modules.semantic_detector import detect_semantic_types

st.set_page_config(page_title="Data Types - AutoDataIQ", page_icon="🏷️")

st.title("🏷️ Semantic Data-Type Detection")
st.write(
    "Automated detection of domain-level semantic data types (Numerical, Categorical, "
    "Date/Datetime, Boolean, Identifier, Text) using multi-signal rule-based analysis."
)

if "data" not in st.session_state or st.session_state["data"] is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Upload page first.")
    st.info("💡 Navigate to **Upload** using the sidebar on the left.")
else:
    dataframe: pd.DataFrame = st.session_state["data"]
    filename = st.session_state.get("filename", "Uploaded Dataset")

    st.write(f"**Active Dataset:** `{filename}` | **Total Columns:** {len(dataframe.columns)}")

    # Run semantic detection module
    semantic_results = detect_semantic_types(dataframe)

    # Summary metrics for semantic types
    st.subheader("Semantic Type Breakdown")
    type_counts = semantic_results["Semantic Type"].value_counts()
    
    # Display top semantic categories as metrics
    count_cols = st.columns(len(type_counts) if len(type_counts) <= 6 else 6)
    for idx, (stype, count) in enumerate(type_counts.items()):
        col_idx = idx % len(count_cols)
        count_cols[col_idx].metric(label=stype, value=count)

    st.subheader("Detected Semantic Data Types")
    st.dataframe(semantic_results, use_container_width=True)

    # Filtering option for detailed exploration
    st.subheader("Filter by Semantic Type")
    selected_type = st.selectbox(
        "Select a semantic type to filter columns:",
        options=["All"] + sorted(list(semantic_results["Semantic Type"].unique())),
    )

    if selected_type != "All":
        filtered_df = semantic_results[semantic_results["Semantic Type"] == selected_type]
        st.dataframe(filtered_df, use_container_width=True)
