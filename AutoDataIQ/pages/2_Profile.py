"""Streamlit page for Phase 2 dataset profiling."""

import pandas as pd
import streamlit as st

from modules.profiler import (
    profile_categorical_columns,
    profile_columns,
    profile_dataset,
    profile_numerical_columns,
)

st.set_page_config(page_title="Dataset Profile - AutoDataIQ", page_icon="📈")

st.title("📈 Dataset Profile")

if "data" not in st.session_state or st.session_state["data"] is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Upload page first.")
    st.info("💡 Navigate to **Upload** using the sidebar on the left.")
else:
    dataframe: pd.DataFrame = st.session_state["data"]
    filename = st.session_state.get("filename", "Uploaded Dataset")

    dataset_profile = profile_dataset(dataframe)
    column_profile = profile_columns(dataframe)
    numerical_profile = profile_numerical_columns(dataframe)
    categorical_profile = profile_categorical_columns(dataframe)

    st.subheader("Dataset Overview")
    st.write(f"**File name:** `{filename}`")
    overview_columns = st.columns(3)
    overview_columns[0].metric("Rows", dataset_profile["rows"])
    overview_columns[1].metric("Columns", dataset_profile["columns"])
    overview_columns[2].metric("Total Cells", dataset_profile["total_cells"])

    metric_cols = st.columns(2)
    metric_cols[0].write(f"**Total missing values:** {dataset_profile['total_missing_values']}")
    metric_cols[1].write(f"**Duplicate rows:** {dataset_profile['duplicate_rows']}")

    with st.expander("Show Column Names", expanded=False):
        st.write(list(dataframe.columns))

    st.subheader("Column Profile")
    st.dataframe(column_profile, use_container_width=True)

    st.subheader("Numerical Statistics")
    if numerical_profile.empty:
        st.info("This dataset has no numerical columns to summarize.")
    else:
        st.dataframe(numerical_profile, use_container_width=True)

    st.subheader("Categorical Column Summary")
    if categorical_profile.empty:
        st.info("This dataset has no object or string columns to summarize.")
    else:
        st.dataframe(categorical_profile, use_container_width=True)
