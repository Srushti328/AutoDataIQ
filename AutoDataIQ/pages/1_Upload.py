"""Streamlit page for dataset upload and initialization."""

import pandas as pd
import streamlit as st

from modules.data_loader import DatasetLoadError, load_dataset

st.set_page_config(page_title="Upload Dataset - AutoDataIQ", page_icon="📁")

st.title("📁 Dataset Upload")
st.write(
    "Upload a structured dataset in CSV, XLSX, or XLS format to begin automated analysis."
)

# Display current dataset status if present in session state
if "data" in st.session_state and st.session_state["data"] is not None:
    current_df: pd.DataFrame = st.session_state["data"]
    current_filename = st.session_state.get("filename", "Uploaded Dataset")
    st.info(
        f"**Active Dataset:** `{current_filename}` | "
        f"**Rows:** {len(current_df)} | **Columns:** {len(current_df.columns)}"
    )

uploaded_file = st.file_uploader(
    "Choose a dataset file",
    type=["csv", "xlsx", "xls"],
    help="Supported file formats: CSV (.csv), Excel (.xlsx, .xls)",
)

if uploaded_file is not None:
    try:
        dataframe = load_dataset(uploaded_file)
    except DatasetLoadError as error:
        st.error(f"❌ {error}")
    else:
        # Save dataset to session state
        st.session_state["data"] = dataframe
        st.session_state["raw_data"] = dataframe.copy(deep=True)
        st.session_state["filename"] = uploaded_file.name
        st.session_state["raw_filename"] = uploaded_file.name
        st.session_state["cleaned_data"] = None
        st.session_state["cleaning_log"] = None
        st.session_state["cleaning_summary"] = None
        st.session_state["active_dataset_name"] = "Original Dataset"
        st.success(f"✅ Successfully loaded `{uploaded_file.name}`!")

        st.subheader("Dataset Preview")
        st.dataframe(dataframe.head(10), use_container_width=True)

        st.markdown("---")
        st.info(
            "👉 Next Step: Use the sidebar navigation under **Understanding** to explore **Profile** or **Data Types**."
        )
elif "data" in st.session_state and st.session_state["data"] is not None:
    st.subheader("Current Dataset Preview")
    st.dataframe(st.session_state["data"].head(10), use_container_width=True)
