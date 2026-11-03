"""Streamlit page for Phase 18 Consolidated PDF Report Generation."""

from datetime import datetime
from pathlib import Path
import streamlit as st

from modules.analysis_engine import classify_columns_semantically
from modules.cleaning_engine import generate_cleaning_recommendations
from modules.document_processor import extract_document_text, profile_document
from modules.document_quality import assess_document_quality
from modules.insight_engine import generate_insights
from modules.kpi_engine import detect_kpis
from modules.profiler import profile_columns, profile_dataset
from modules.quality_engine import analyze_data_quality
from modules.quality_score import calculate_quality_score
from modules.report_generator import (
    generate_structured_data_pdf,
    generate_unstructured_document_pdf,
)
from modules.semantic_detector import detect_semantic_types
from modules.text_analysis_engine import get_analyzable_text, run_text_analysis

st.set_page_config(
    page_title="PDF Reports - AutoDataIQ",
    page_icon="📑",
    layout="wide",
)

st.title("📑 Consolidated PDF Report Generation")
st.write(
    "Generate comprehensive, publication-ready PDF reports summarizing automated analytical findings, "
    "quality health benchmarking, and domain indicators for both structured datasets and unstructured documents."
)

st.info(
    "🔒 **Read-Only Reporting:** Report generation strictly inspects existing findings and does not mutate "
    "your raw datasets, cleaned datasets, or document extracts."
)

report_workflow = st.radio(
    "Select Workflow for Report Generation:",
    options=["Structured Tabular Data", "Unstructured Document Analytics"],
    horizontal=True,
    help="Generate reports tailored to either tabular CSV/Excel datasets or unstructured TXT/PDF/DOCX documents.",
)

# =====================================================================
# Workflow 1: Structured Tabular Data Report
# =====================================================================
if report_workflow == "Structured Tabular Data":
    st.markdown("---")
    st.subheader("📊 Structured Data Intelligence Report")

    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("⚠️ No tabular dataset is currently loaded in session state.")
        st.info("💡 Navigate to **Upload** in the sidebar to load a CSV or Excel file first.")
    else:
        df = st.session_state["data"]
        dataset_name = st.session_state.get("filename", "Uploaded_Dataset.csv")
        is_cleaned = st.session_state.get("active_dataset_type", "original") == "cleaned"
        pipeline_label = "Cleaned Dataset" if is_cleaned else "Original Raw Dataset"

        st.write(
            f"**Target Dataset:** `{dataset_name}` | **Rows:** {len(df):,} | "
            f"**Columns:** {len(df.columns)} | **Pipeline Mode:** `{pipeline_label}`"
        )

        st.markdown("##### 📋 Included Report Sections Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("• ✅ **Dataset Profiling Overview:** Dimensions, missing cell counts, duplicate records, memory footprint.")
            st.markdown("• ✅ **Column-Level Profiles:** Storage types, null counts, null percentages, unique cardinalities.")
            st.markdown("• ✅ **Semantic Data-Type Detection:** Multi-signal inference with confidence ratings and reasoning.")
        with col2:
            st.markdown("• ✅ **Data Quality Scoring:** 6-dimension score (0–100), hygiene rating, and anomaly detections.")
            st.markdown("• ✅ **Cleaning & Transformations:** Prescriptions or applied operation audit log.")
            st.markdown("• ✅ **Domain KPIs & Prioritized Insights:** Key performance indicators and empirical findings.")

        st.markdown("")
        if st.button("📄 Generate Structured Data PDF Report", type="primary", key="btn_gen_struct_pdf"):
            with st.spinner("Compiling structured intelligence report into PDF..."):
                try:
                    # 1. Profiling
                    prof = profile_dataset(df)
                    col_prof_df = profile_columns(df)
                    col_prof = col_prof_df.to_dict(orient="records") if col_prof_df is not None else []

                    # 2. Semantics
                    sem_df = detect_semantic_types(df)
                    sem_types = sem_df.to_dict(orient="records") if sem_df is not None else []

                    # 3. Quality
                    q_report = analyze_data_quality(df, sem_df)
                    q_score = calculate_quality_score(q_report, len(df), len(df.columns))
                    findings = q_report.get("quality_issues", [])

                    # 4. Cleaning
                    clean_recs = generate_cleaning_recommendations(df, sem_df, q_report)
                    clean_logs = st.session_state.get("cleaning_logs", [])

                    # 5. KPIs
                    kpi_res = detect_kpis(df, sem_df)
                    all_kpis = kpi_res.get("all_kpis", [])

                    # 6. Insights
                    ins_res = generate_insights(df, sem_df, q_report, q_score)
                    all_insights = ins_res.get("all_insights", [])

                    pdf_bytes = generate_structured_data_pdf(
                        dataset_name,
                        profile_summary=prof,
                        column_profile=col_prof,
                        semantic_types=sem_types,
                        quality_score=q_score,
                        quality_findings=findings,
                        cleaning_recommendations=clean_recs,
                        applied_cleaning_log=clean_logs if is_cleaned else None,
                        insights=all_insights,
                        kpis=all_kpis,
                        is_cleaned=is_cleaned,
                    )
                    st.session_state["generated_struct_pdf"] = pdf_bytes
                    st.session_state["struct_pdf_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("✅ Structured Data PDF Report successfully generated!")
                except Exception as e:
                    st.error(f"Failed to generate structured PDF report: {e}")

        if "generated_struct_pdf" in st.session_state and st.session_state["generated_struct_pdf"]:
            base_name = dataset_name.rsplit(".", 1)[0]
            st.download_button(
                label=f"⬇️ Download Structured Report ({base_name}_AutoDataIQ.pdf)",
                data=st.session_state["generated_struct_pdf"],
                file_name=f"{base_name}_AutoDataIQ_Report.pdf",
                mime="application/pdf",
                key="btn_download_struct_pdf",
            )
            st.caption(f"Generated at: {st.session_state.get('struct_pdf_timestamp', '')}")

# =====================================================================
# Workflow 2: Unstructured Document Report
# =====================================================================
else:
    st.markdown("---")
    st.subheader("📄 Unstructured Document Analytics Report")

    if "doc_file_bytes" not in st.session_state or st.session_state["doc_file_bytes"] is None:
        st.warning("⚠️ No document is currently loaded in session state.")
        st.info("💡 Navigate to **Document Processing** or **Text Analysis** in the sidebar to upload a document first.")
    else:
        doc_name = st.session_state.get("doc_filename", "uploaded_document.txt")
        doc_profile = st.session_state.get("doc_profile", {})
        file_type = doc_profile.get("file_type", "TXT")

        # Select Text Source (Original vs Cleaned)
        has_cleaned = (
            "doc_cleaned_text" in st.session_state
            and st.session_state["doc_cleaned_text"] is not None
            and len(st.session_state["doc_cleaned_text"].strip()) > 0
        )
        source_options = ["Original Extracted Text"]
        if has_cleaned:
            source_options.insert(0, "Cleaned Text (Phase 16)")

        selected_source = st.radio(
            "Select text version to compile into report:",
            options=source_options,
            index=0,
            horizontal=True,
            key="pdf_doc_text_source",
        )
        text_mode = "cleaned" if "Cleaned" in selected_source else "original"
        doc_text, source_title, _ = get_analyzable_text(st.session_state, text_source=text_mode)

        st.write(
            f"**Target Document:** `{doc_name}` | **Format:** {file_type} | "
            f"**Size:** {doc_profile.get('file_size_formatted', 'N/A')} | **Source:** `{source_title}`"
        )

        if not doc_text or not doc_text.strip():
            st.warning(
                "⚠️ The selected document contains no extractable text. "
                "Reports cannot be generated for empty documents or scanned PDFs without OCR."
            )
        else:
            st.markdown("##### 📋 Included Report Sections Preview")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("• ✅ **Document Profile & Statistics:** File details, characters, words, sentences, paragraphs.")
                st.markdown("• ✅ **Document Quality Findings:** Rule-based hygiene evaluation and non-destructive cleaning audit.")
                st.markdown("• ✅ **Keyword Frequency Summary:** Top 12 meaningful keywords with stop-words removed.")
            with c2:
                st.markdown("• ✅ **Rule-Based Sentiment:** Sentence-level classification, net score, and positive/negative triggers.")
                st.markdown("• ✅ **Domain Topic Mapping:** Keyword matches across Delivery, Billing, Quality, Support, App, Web.")
                st.markdown("• ✅ **Prioritized Insights & Limitations:** Key findings with supporting evidence and scope notes.")

            st.markdown("")
            if st.button("📄 Generate Document Analytics PDF Report", type="primary", key="btn_gen_doc_pdf"):
                with st.spinner("Compiling document analytics report into PDF..."):
                    try:
                        # Quality findings
                        q_findings = assess_document_quality(doc_text)

                        # Run text analysis
                        analysis_res = run_text_analysis(doc_text)
                        t_stats = analysis_res.get("statistics", {})
                        kws = analysis_res.get("keywords", [])
                        sent = analysis_res.get("sentiment", {})
                        tops = analysis_res.get("topics", {}).get("topics", [])
                        ins = analysis_res.get("insights", [])

                        # Cleaning preview if cleaned
                        cleaning_applied = None
                        if text_mode == "cleaned":
                            ops = []
                            if st.session_state.get("doc_clean_trim"):
                                ops.append("Trimmed leading and trailing whitespace")
                            if st.session_state.get("doc_clean_normalize_endings"):
                                ops.append("Normalized line endings to Unix standard")
                            if st.session_state.get("doc_clean_collapse_blank_lines"):
                                ops.append("Collapsed excessive consecutive blank lines")
                            if st.session_state.get("doc_clean_remove_repeated_lines"):
                                ops.append("Removed repeated identical lines")
                            cleaning_applied = {"selected_operations": ops}

                        pdf_bytes = generate_unstructured_document_pdf(
                            doc_name,
                            file_type=file_type,
                            text_source=source_title,
                            doc_profile=doc_profile,
                            quality_findings=q_findings,
                            applied_cleaning=cleaning_applied,
                            text_statistics=t_stats,
                            keywords=kws,
                            sentiment=sent,
                            topics=tops,
                            insights=ins,
                        )
                        st.session_state["generated_doc_pdf"] = pdf_bytes
                        st.session_state["doc_pdf_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success("✅ Document Analytics PDF Report successfully generated!")
                    except Exception as e:
                        st.error(f"Failed to generate document PDF report: {e}")

            if "generated_doc_pdf" in st.session_state and st.session_state["generated_doc_pdf"]:
                base_doc = doc_name.rsplit(".", 1)[0]
                st.download_button(
                    label=f"⬇️ Download Document Report ({base_doc}_AutoDataIQ.pdf)",
                    data=st.session_state["generated_doc_pdf"],
                    file_name=f"{base_doc}_AutoDataIQ_Report.pdf",
                    mime="application/pdf",
                    key="btn_download_doc_pdf",
                )
                st.caption(f"Generated at: {st.session_state.get('doc_pdf_timestamp', '')}")
