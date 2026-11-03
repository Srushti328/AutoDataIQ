# AutoDataIQ 🧠
### An Intelligent Framework for Automated Data Quality Assessment and Adaptive Data Analysis
**MCA Project — Final Submission Package (Phases 1–18)**

---

## 📌 Project Overview

**AutoDataIQ** is a comprehensive, modular Python and Streamlit-based data intelligence application designed for automated data understanding, quality assessment, transformation, adaptive analysis, visualization, executive dashboard generation, unstructured document text analytics, and consolidated publication-ready PDF reporting.

Developed as a Master of Computer Applications (MCA) project, AutoDataIQ bridges the gap between raw data ingestion and analytical readiness. It operates transparently without unverified assumptions, opaque "black-box" models, or external commercial AI APIs—relying entirely on mathematical rigor, statistical heuristics, deterministic data-type profiling, explainable rule-based text intelligence, and professional ReportLab PDF compilation.

---

## 🚀 Key Features by Phase

1. **Phase 1: Dataset Upload & Multi-Format Ingestion** (`1_Upload.py`)
   - Secure in-memory ingestion supporting **CSV**, **XLSX**, and **XLS** formats.
   - Robust error handling for unreadable, corrupted, or empty files.

2. **Phase 2: Automated Structural Profiling** (`2_Profile.py`)
   - Dataset-level metrics: total rows, columns, memory usage, missing cells, and duplicate rows.
   - Column-level statistical profiles: five-number summaries, uniqueness, cardinality, and sample distributions.

3. **Phase 3: Multi-Signal Semantic Data-Type Detection** (`3_Data_Types.py`)
   - Rule-based inference beyond primitive Pandas types: **Numerical**, **Categorical**, **Date/Datetime**, **Boolean**, **Identifier**, and **Text**.
   - Confidence levels (*High / Medium / Low*) with transparent diagnostic rationale.

4. **Phase 4: Data Quality Intelligence** (`4_Quality.py`)
   - Read-only detection of 11 distinct quality issues: missing values, empty columns, duplicates, constants, near-constants, high-cardinality flags, type inconsistencies, invalid values, casing inconsistencies, boolean formatting variants, and IQR statistical outliers.

5. **Phase 5: Six-Dimension Data Quality Scoring** (`4_Quality.py`)
   - Quantitative Data Quality Score (0–100) and Rating (*Excellent, Good, Fair, Poor, Critical*) evaluated across:
     - *Completeness* (25%)
     - *Uniqueness* (15%)
     - *Consistency* (20%)
     - *Validity* (20%)
     - *Structural Quality* (10%)
     - *Numerical Reliability* (10%)

6. **Phase 6: Intelligent Cleaning Recommendations** (`5_Cleaning.py`)
   - Rule-based, actionable guidance explaining *what* to clean, *why*, *which method* to employ, and *expected impact*.
   - Structured priority classification (*High / Medium / Low / Info*).

7. **Phase 7: Adaptive Exploratory Data Analysis** (`7_Analysis.py`)
   - Dynamic activation of analytical modules tailored to detected column semantics.
   - Includes numerical summary distributions, categorical frequencies, temporal trends, Pearson correlation matrices, and group comparisons.

8. **Phase 8: Automated Visualization Engine** (`8_Visualization.py`)
   - Automated recommendation and rendering of interactive 2D Plotly figures:
     - Histograms & Box Plots
     - Top-N Categorical Bar Charts
     - Bivariate Scatter Plots
     - Pearson Correlation Heatmaps
     - Time-Series Line Charts
     - Grouped Comparison Bar Charts
     - Boolean Proportion Charts

9. **Phase 9: Automated Insight Generation & Prioritization** (`9_Insights.py`)
   - Automated extraction of statistically significant findings across Numerical, Categorical, Temporal, and Quality dimensions.
   - Priority ranking (*High / Medium / Low*) accompanied by concrete numerical evidence and observational caveats.

10. **Phase 10: Automatic KPI Detection & Metrics** (`10_KPIs.py`)
    - Identifies domain-relevant indicators: Overview Metrics, Potential Business KPIs (Revenue, Volume, Profit, Salary), Statistical Averages, Categorical Dominance, and Temporal Spans.
    - Full transparency: formulas, source columns, and identifier safeguards (e.g. no meaningless arithmetic on IDs or non-summable fields like Age).

11. **Phase 11: Automated Data Cleaning & Transformation Execution** (`6_Clean_Data.py`)
    - User-directed transformation pipeline executing on an isolated copy of the dataset:
      - Empty column dropping
      - Duplicate row removal (preserves first occurrence)
      - Imputation: Mean, Median, Mode, Constant
      - Safe data-type conversions with coercion tracking
      - Text whitespace trimming and case standardization
      - IQR outlier clipping (winsorization preserving row integrity)
    - Full operation audit log, CSV/XLSX export, and active dataset switcher.
    - **Strict Immutability Guarantee:** Original raw dataset is never mutated.

12. **Phase 12: Automated Executive Dashboard Generation** (`11_Dashboard.py`)
    - Consolidated executive overview synthesizing curated top KPIs, recommended diverse visualizations, prioritized analytical findings, and data quality health metrics.
    - Dynamic active dataset switcher (Original vs. Cleaned Dataset).

13. **Phase 13: Interactive Dashboard Customization** (`11_Dashboard.py`)
    - Customization drawer with tabs for Section Arrangement, KPI Cards, Visualizations, and Key Insights.
    - Section arrangement presets (*Default*, *Executive Focus*, *Visuals First*, *Quality First*) or custom ordered multiselect.
    - Instant "Reset to Default Layout" mechanism.
    - Export options: customized KPIs (CSV), customized Insights (CSV), and standalone interactive Plotly charts (HTML).

14. **Phase 14: Final Testing, Documentation & Project Readiness**
    - Comprehensive multi-phase integration test suite (`test_suite.py`).
    - Standardized phase numbering, sample datasets in `datasets/`, and academic project documentation.

15. **Phase 15: Unstructured Document Processing** (`12_Document_Processing.py`)
    - Dedicated, isolated document-processing workflow ingesting **TXT**, text-based **PDF**, and Microsoft Word (**DOCX**) files.
    - Deep document profiling: character counts, word counts, line counts, PDF page counts, DOCX paragraph counts, average word length, and readable previews.
    - Scanned/image-only PDF detection with clear user notices explaining OCR scope.
    - Strict isolation from structured CSV/Excel session state.

16. **Phase 16: Document Quality Assessment & Non-Destructive Cleaning** (`12_Document_Processing.py`)
    - Rule-based document quality checks: empty content, whitespace anomalies, excessive blank lines, repeated lines/paragraphs, and non-printable characters.
    - Interactive cleaning preview with selective operations: trimming, line ending normalization, blank-line collapse, and repeated line deduplication without modifying raw extracted text.

17. **Phase 17: Text Analysis & Rule-Based Insights** (`13_Text_Analysis.py`)
    - Comprehensive text statistics: total characters, words, sentences, paragraphs, average word and sentence lengths, and frequent raw words.
    - Meaningful keyword frequency analysis with English stop-word removal and interactive Plotly bar charts.
    - Transparent rule-based sentiment classification (Positive, Negative, Neutral) with sentence-by-sentence evidence, score formulas, and limitation disclaimers.
    - Multi-topic identification across Delivery, Billing, Quality, Support, Mobile App, Website, and General categories with matched keyword evidence.
    - Prioritized rule-based insight generation (High, Medium, Low) synthesizing customer sentiment, complaints, praise, and key theme concentrations.
    - Cleaned vs. original text analysis toggle with complete isolation from tabular data.

18. **Phase 18: Consolidated PDF Report Generation & Sidebar Organization** (`14_PDF_Reports.py`)
    - Reorganized Streamlit sidebar navigation into 4 clean sections: **Home**, **Structured Data**, **Unstructured Documents**, and **Reports**.
    - Publication-ready, downloadable PDF reports for both workflows using ReportLab and dynamic two-pass page numbering (`Page X of Y`).
    - **Structured Data Report**: Dataset profiling, column profiles, semantic typing, data quality score, dimension weights, anomaly logs, cleaning audit, KPIs, and analytical insights.
    - **Unstructured Document Report**: Document metadata, text statistics, quality findings, cleaning operations applied, top keywords, sentiment distribution, topic mappings, prioritized insights, and explainability notices.
    - Strict non-destructive reporting operating on in-memory findings without mutating raw datasets.

---

## 📂 Project Directory Structure

```text
E:\AutoDataIQ\
│
├── app.py                      # Main application entry point & section navigation router
├── requirements.txt            # Python dependencies
├── test_suite.py               # Comprehensive 18-phase integration test suite
├── README.md                   # Project overview & quick start guide
├── PROJECT_DOCUMENTATION.md    # Full academic MCA project documentation report
│
├── datasets/                   # Bundled sample datasets for testing
│   ├── sample_sales.csv        # Retail sales transaction dataset (CSV)
│   ├── sample_employees.xlsx   # Human resources & salary dataset (XLSX)
│   ├── sample_edge_cases.csv   # Messy dataset with missingness, duplicates, outliers
│   ├── sample_document.txt     # Unstructured plain text sample (TXT)
│   ├── sample_customer_feedback.txt # Multi-topic customer reviews sample (TXT)
│   ├── sample_report.docx      # Unstructured Microsoft Word sample (DOCX)
│   ├── sample_document.pdf     # Unstructured selectable text PDF sample (PDF)
│   └── sample_scanned_mock.pdf # Scanned mock PDF for zero-text detection (PDF)
│
├── modules/                    # Pure Python core logic & analytical engines
│   ├── __init__.py
│   ├── data_loader.py          # Phase 1: File ingestion & validation
│   ├── profiler.py             # Phase 2: Structural & statistical profiling
│   ├── semantic_detector.py    # Phase 3: Rule-based semantic typing
│   ├── quality_engine.py       # Phase 4: Data hygiene & anomaly detection
│   ├── quality_score.py        # Phase 5: 6-dimension scoring algorithm
│   ├── cleaning_engine.py      # Phase 6: Rule-based cleaning recommendations
│   ├── cleaner.py              # Phase 11: Interactive transformation execution
│   ├── analysis_engine.py      # Phase 7: Adaptive exploratory analysis
│   ├── visualization_engine.py # Phase 8: Plotly visualization recommendations
│   ├── insight_engine.py       # Phase 9: Statistical insight generation
│   ├── kpi_engine.py           # Phase 10: Automatic KPI & formula derivation
│   ├── dashboard_engine.py     # Phases 12 & 13: Dashboard compilation & customization
│   ├── document_processor.py   # Phase 15: TXT, PDF & DOCX extraction and profiling
│   ├── document_quality.py     # Phase 16: Document quality checks & non-destructive cleaning
│   ├── text_analysis_engine.py # Phase 17: Text statistics, sentiment, topics & insights
│   └── report_generator.py     # Phase 18: Consolidated PDF report compilation (ReportLab)
│
├── pages/                      # Streamlit multi-page UI frontend
│   ├── 1_Upload.py             # [Structured] Dataset Upload & initialization
│   ├── 2_Profile.py            # [Structured] Dataset & Column Profiling
│   ├── 3_Data_Types.py         # [Structured] Semantic Type Classifications
│   ├── 4_Quality.py            # [Structured] Quality Report & Scoring Dimensions
│   ├── 5_Cleaning.py           # [Structured] Cleaning Recommendations
│   ├── 6_Clean_Data.py         # [Structured] Automated Data Cleaning Pipeline
│   ├── 7_Analysis.py           # [Structured] Adaptive Data Analysis
│   ├── 8_Visualization.py      # [Structured] Automated Interactive Visualizations
│   ├── 9_Insights.py           # [Structured] Prioritized Analytical Insights
│   ├── 10_KPIs.py              # [Structured] Automatic KPI Metrics & Formulas
│   ├── 11_Dashboard.py         # [Structured] Executive Dashboard & Customization
│   ├── 12_Document_Processing.py # [Documents] Document Processing & Quality
│   ├── 13_Text_Analysis.py     # [Documents] Text Analysis & Rule-Based Insights
│   └── 14_PDF_Reports.py       # [Reports] Consolidated PDF Report Generation
│
└── tests/                      # Unit test suites
    ├── test_phase16_document_quality.py # Phase 16 focused test suite
    ├── test_phase17_text_analysis.py    # Phase 17 focused test suite
    └── test_phase18_report_generation.py # Phase 18 focused test suite
```

---

## 🛠️ Technology Stack & Dependencies

- **Language:** Python 3.10+
- **Frontend / Framework:** Streamlit (v1.30.0+)
- **Data Manipulation & Computation:** Pandas (v2.0.0+), NumPy (v1.24.0+)
- **Interactive Visualizations:** Plotly (v5.15.0+)
- **Spreadsheet Ingestion & Export:** OpenPyXL (v3.1.0+), XLRD (v2.0.1+)
- **Document Processing:** PyPDF (v5.0.0+), Python-Docx (v1.0.0+)
- **Publication PDF Report Generation:** ReportLab (v4.0.0+)

---

## 💻 Installation & Execution Instructions

### Prerequisites
Ensure Python (version 3.10 or higher) is installed on your system.

### Step 1: Clone / Open Project Directory
```powershell
cd E:\AutoDataIQ
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run the Integration Test Suite
Verify that all 18 phases pass cleanly:
```powershell
python test_suite.py
```
*Expected Output: `All Phase 1 - Phase 18 Integration Tests Passed!`*

You can also run the focused unit test suite for Phase 18:
```powershell
python tests/test_phase18_report_generation.py
```

### Step 4: Launch the AutoDataIQ Application
```powershell
python -m streamlit run app.py
```

The Streamlit web application will launch in your default web browser at `http://localhost:8501`.

---

## 🧪 Testing Summary

The project includes an automated regression test suite in [`test_suite.py`](file:///e:/AutoDataIQ/test_suite.py) covering all 18 phases:
- **Test 1:** File upload validation, corrupted files, and unsupported formats.
- **Test 2:** Profiling metrics, dimensions, and numerical/categorical breakdowns.
- **Test 3:** Multi-signal semantic detection on various column patterns.
- **Test 4:** Data quality anomaly detection (missing, duplicate, casing, outliers).
- **Test 5:** 6-dimension quality scoring mathematics and rating assignments.
- **Test 6:** Cleaning recommendation generation and priority ratings.
- **Test 7:** Adaptive analysis module dispatch and group comparisons.
- **Test 8:** Plotly visualization recommendations, chart types, and immutability.
- **Test 9:** Insight generation, deduplication, priority ranking, and evidence.
- **Test 10:** KPI formula derivation, identifier protection, and non-summable rules.
- **Test 11:** Cleaning pipeline execution, imputation, outlier clipping, and raw DataFrame preservation.
- **Test 12:** Dashboard data aggregation, KPI curation, chart variety, and sidebar sequence.
- **Test 13:** Dashboard customization, element toggles, section reordering, reset to default, and exports.
- **Test 14:** Final end-to-end workflow execution on retail CSV, employee XLSX, and edge-case datasets with strict immutability.
- **Test 15:** Unstructured document processing on TXT, PDF, and DOCX files, page/paragraph counting, scanned PDF detection, and state isolation.
- **Test 16:** Rule-based document quality checks (empty, whitespace, excessive breaks, repeats) and non-destructive cleaning previews.
- **Test 17:** Text analysis, keyword frequencies with stop-word removal, rule-based sentiment, topic classification, and prioritized insights.
- **Test 18:** Consolidated PDF report compilation for structured datasets and unstructured documents, multi-page pagination, and 4-section sidebar navigation.

---

## 🔒 Data Protection & Immutability Guarantee

All analyses, transformations, visualizations, and dashboard customizations in AutoDataIQ strictly respect raw data integrity:
1. **Isolated Transformation:** Transformations in Phase 11 operate exclusively on a separate copy stored in `st.session_state["cleaned_data"]`. The raw uploaded dataset in `st.session_state["raw_data"]` is never mutated.
2. **Document Pipeline Isolation:** Unstructured document processing in Phase 15 operates in strict isolation with dedicated session-state keys. Ingesting, parsing, or profiling documents never couples with or mutates the tabular DataFrame environment.
3. **Read-Only Intelligence:** Profiling, semantic detection, quality scoring, KPI derivation, and insight extraction are 100% read-only operations.
4. **Session-Level Separation:** Customization settings do not alter DataFrame values or schemas.
5. **Local Processing:** All calculations are performed in-memory on the local machine without external network transfers.

---

## 🎓 Academic Submission Note
This project was designed, developed, and tested in accordance with MCA curriculum guidelines for practical, scalable, and reproducible data engineering software.
