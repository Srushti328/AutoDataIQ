"""AutoDataIQ – Main Application Landing Page & Navigation Router.

Phase 18: Organizes multi-page application into four structured navigation sections:
1. Home (Overview)
2. Structured Data (Upload, Profile, Data Types, Quality, Cleaning, Clean Data, Analysis, Visualization, Insights, KPIs, Dashboard)
3. Unstructured Documents (Document Processing, Text Analysis)
4. Reports (PDF Report Generation)
"""

import streamlit as st


def show_overview() -> None:
    """Render the AutoDataIQ overview and getting started guidance."""
    st.title("🧠 AutoDataIQ")
    st.subheader(
        "An Intelligent Framework for Automated Data Quality Assessment, Adaptive Data Analysis & Executive Reporting"
    )

    st.markdown(
        """
> **Core Architecture:** End-to-end local data intelligence supporting two dedicated, isolated workflows:
> 1. **Structured Tabular Data Workflow** (CSV / Excel): Ingestion → Profiling → Semantic Typing → Quality Scoring → Cleaning → Adaptive Analysis → Visualizations → Insights → KPIs → Executive Dashboard.
> 2. **Unstructured Document Analytics Workflow** (TXT / PDF / Word): Text Extraction → Document Profiling → Formatting Hygiene → Non-destructive Cleaning → Stopword-filtered Keywords → Rule-based Sentiment → Topic Mapping → Actionable Insights.
>
> Both workflows culminate in publication-ready, downloadable **Consolidated PDF Reports**.

---

### 📌 Application Workflow Sections

AutoDataIQ organizes its capabilities into four dedicated sidebar sections:

#### 1. 🏠 Home
* **Overview:** High-level project architecture, feature index, and dual-workflow execution guidance.

#### 2. 📊 Structured Tabular Data Workflow
* **Upload** (`pages/1_Upload.py`): In-memory ingestion of CSV, XLSX, and XLS spreadsheets with encoding recovery.
* **Profile** (`pages/2_Profile.py`): Dataset-level and column-level metrics, descriptive statistics, and memory usage.
* **Data Types** (`pages/3_Data_Types.py`): Multi-signal semantic detection (Numerical, Categorical, Date, Boolean, ID, Text).
* **Quality** (`pages/4_Quality.py`): 11 quality issue detectors and 6-dimension Quality Score (0–100) with formal hygiene ratings.
* **Cleaning** (`pages/5_Cleaning.py`): Rule-based, actionable recommendations answering what, why, and how to sanitize data.
* **Clean Data** (`pages/6_Clean_Data.py`): Interactive transformation execution on an isolated copy; guarantees 100% raw data immutability.
* **Analysis** (`pages/7_Analysis.py`): Dynamic adaptive exploration: distributions, correlations, group comparisons, time trends.
* **Visualization** (`pages/8_Visualization.py`): Automated recommendation and rendering of interactive 2D Plotly charts.
* **Insights** (`pages/9_Insights.py`): Prioritized findings (High, Medium, Low) with concrete empirical evidence and caveats.
* **KPIs** (`pages/10_KPIs.py`): Domain indicator derivation with explicit mathematical formulas and non-summable safeguards.
* **Dashboard** (`pages/11_Dashboard.py`): Executive dashboard synthesis with active dataset switching, section reordering, and element toggles.

#### 3. 📄 Unstructured Document Analytics Workflow
* **Document Processing** (`pages/12_Document_Processing.py`): Selectable text extraction from TXT, PDF, and DOCX files, deep profiling metrics, scanned PDF detection, and non-destructive text cleaning previews.
* **Text Analysis** (`pages/13_Text_Analysis.py`): Lexical statistics, stopword-filtered keyword frequencies, explainable rule-based sentiment classification, domain topic identification, and prioritized textual insights.

#### 4. 📑 Consolidated Reports
* **PDF Report Generation** (`pages/14_PDF_Reports.py`): Downloadable, publication-ready PDF reports for either Structured Data or Unstructured Documents.

---

### 🚀 Getting Started Guide

#### A. Structured Tabular Data Walkthrough
1. Navigate to **Upload** under *Structured Data* in the sidebar and select a CSV or Excel file.
2. Review dimensional health in **Profile** and verify classification accuracy in **Data Types**.
3. Inspect anomalies and composite score in **Quality**, then consult **Cleaning** for prescriptions.
4. Execute user-directed sanitization in **Clean Data** without mutating raw data.
5. Explore dynamic statistics in **Analysis** and interactive figures in **Visualization**.
6. Review prioritized patterns in **Insights** and quantitative targets in **KPIs**.
7. Tailor layout and inspect executive summaries in **Dashboard**.
8. Compile an executive artifact in **PDF Report Generation**.

#### B. Unstructured Document Analytics Walkthrough
1. Navigate to **Document Processing** under *Unstructured Documents* in the sidebar.
2. Upload a TXT, text-based PDF, or Word DOCX document, or select a bundled sample.
3. Review character, word, line, and page/paragraph metrics; inspect quality flags and optionally apply non-destructive text cleaning.
4. Navigate to **Text Analysis** to review keyword frequencies, sentiment distributions, domain topic alignments, and rule-based insights.
5. Navigate to **PDF Report Generation** under *Reports* to export a comprehensive PDF report.
"""
    )

    # Active status badge
    if "data" in st.session_state and st.session_state["data"] is not None:
        df = st.session_state["data"]
        filename = st.session_state.get("filename", "Uploaded Dataset")
        is_clean = st.session_state.get("active_dataset_type", "original") == "cleaned"
        st.success(
            f"✅ **Active Tabular Dataset Loaded:** `{filename}` ({len(df):,} rows, {len(df.columns)} columns) | "
            f"Mode: `{'Cleaned' if is_clean else 'Original Raw'}`"
        )
    else:
        st.info("💡 **Tabular Data:** No structured dataset loaded yet. Head over to **Structured Data -> Upload** to start.")

    if "doc_file_bytes" in st.session_state and st.session_state["doc_file_bytes"] is not None:
        doc_name = st.session_state.get("doc_filename", "uploaded_document")
        st.success(
            f"✅ **Active Unstructured Document Loaded:** `{doc_name}` | "
            "Available for Text Analysis & PDF Report Generation."
        )
    else:
        st.info("💡 **Document Data:** No document loaded yet. Head over to **Unstructured Documents -> Document Processing** to test.")


# =====================================================================
# Sidebar Navigation Router (Streamlit Supported Section Navigation)
# =====================================================================

st.set_page_config(
    page_title="AutoDataIQ - Intelligent Data Quality & Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Home": [
        st.Page(show_overview, title="Overview", icon="🏠", default=True),
    ],
    "Structured Data": [
        st.Page("pages/1_Upload.py", title="Upload", icon="📤"),
        st.Page("pages/2_Profile.py", title="Profile", icon="📋"),
        st.Page("pages/3_Data_Types.py", title="Data Types", icon="🏷️"),
        st.Page("pages/4_Quality.py", title="Quality", icon="🛡️"),
        st.Page("pages/5_Cleaning.py", title="Cleaning", icon="🧹"),
        st.Page("pages/6_Clean_Data.py", title="Clean Data", icon="✨"),
        st.Page("pages/7_Analysis.py", title="Analysis", icon="🔬"),
        st.Page("pages/8_Visualization.py", title="Visualization", icon="📊"),
        st.Page("pages/9_Insights.py", title="Insights", icon="💡"),
        st.Page("pages/10_KPIs.py", title="KPIs", icon="🎯"),
        st.Page("pages/11_Dashboard.py", title="Dashboard", icon="📈"),
    ],
    "Unstructured Documents": [
        st.Page("pages/12_Document_Processing.py", title="Document Processing", icon="📄"),
        st.Page("pages/13_Text_Analysis.py", title="Text Analysis", icon="🔍"),
    ],
    "Reports": [
        st.Page("pages/14_PDF_Reports.py", title="PDF Report Generation", icon="📑"),
    ],
}

nav = st.navigation(pages)
nav.run()
