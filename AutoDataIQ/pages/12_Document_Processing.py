"""Streamlit page for Phase 15-16 document processing and text quality review."""

import hashlib
from pathlib import Path
import streamlit as st

from modules.document_processor import (
    DocumentProcessingError,
    detect_document_type,
    extract_document_text,
    profile_document,
)
from modules.document_quality import (
    assess_document_quality,
    build_cleaning_preview,
    initialize_document_cleaning_state,
    reset_document_cleaning_state,
)

st.set_page_config(
    page_title="Document Processing - AutoDataIQ",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Unstructured Document Processing")
st.write(
    "Upload unstructured documents in **TXT**, **PDF**, or **DOCX** formats to extract text, "
    "inspect document structural traits, and review comprehensive profiling metrics."
)

st.info(
    "🔒 **Workflow Isolation Guarantee:** This document processing workflow operates completely "
    "independently from the structured CSV/Excel analysis pipeline. Uploading or profiling documents "
    "does not mutate your loaded tabular datasets (`raw_data`, `cleaned_data`) or dashboard state."
)

# -------------------------------------------------------------
# Document Upload Area
# -------------------------------------------------------------
st.subheader("📤 Document Upload")

uploaded_file = st.file_uploader(
    "Choose a document to process:",
    type=["txt", "pdf", "docx"],
    help="Supported document formats: TXT (Plain text), PDF (Text-based PDF), and DOCX (Microsoft Word).",
)

# Quick sample document selector if user has not uploaded a file
sample_docs = {
    "None": None,
    "Sample Plain Text (.txt)": "datasets/sample_document.txt",
    "Sample Word Document (.docx)": "datasets/sample_report.docx",
    "Sample Text-based PDF (.pdf)": "datasets/sample_document.pdf",
    "Sample Scanned PDF (.pdf) [0 text OCR check]": "datasets/sample_scanned_mock.pdf",
}

with st.expander("📁 Or Load a Bundled Sample Document", expanded=False):
    selected_sample = st.selectbox(
        "Select a sample document from the repository:",
        options=list(sample_docs.keys()),
        index=0,
        help="Test unstructured document processing immediately with pre-bundled samples.",
    )
    if selected_sample and sample_docs[selected_sample]:
        sample_path = sample_docs[selected_sample]
        if Path(sample_path).exists():
            with open(sample_path, "rb") as sf:
                st.session_state["doc_file_bytes"] = sf.read()
                st.session_state["doc_filename"] = Path(sample_path).name

if uploaded_file is not None:
    st.session_state["doc_file_bytes"] = uploaded_file.getvalue()
    st.session_state["doc_filename"] = uploaded_file.name

# -------------------------------------------------------------
# Process Uploaded Document
# -------------------------------------------------------------
if "doc_file_bytes" in st.session_state and st.session_state["doc_file_bytes"] is not None:
    doc_bytes = st.session_state["doc_file_bytes"]
    doc_name = st.session_state.get("doc_filename", "uploaded_document")

    try:
        extraction_result = extract_document_text(doc_bytes, doc_name)
        doc_profile = profile_document(extraction_result)
        st.session_state["doc_extraction_result"] = extraction_result
        st.session_state["doc_profile"] = doc_profile
        raw_text = extraction_result.get("text", "")
        document_source_id = f"{doc_name}:{hashlib.sha256(doc_bytes).hexdigest()}"
        initialize_document_cleaning_state(
            st.session_state, document_source_id, raw_text
        )

        # ---------------------------------------------------------
        # Status Messages & Diagnostics
        # ---------------------------------------------------------
        status = doc_profile["status"]
        if doc_profile.get("is_scanned"):
            st.warning(
                "⚠️ **Scanned or Image-Only PDF Detected:** "
                f"{doc_profile['message']}"
            )
        elif status == "NO_TEXT":
            st.warning(
                f"⚠️ **Notice:** {doc_profile['message']}"
            )
        else:
            st.success(
                f"✅ **Extraction Successful:** {doc_profile['message']}"
            )

        # ---------------------------------------------------------
        # Document Profiling Metric Cards
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Document Profiling Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("File Name", doc_profile["filename"])
        col2.metric("Detected Format", doc_profile["file_type"])
        col3.metric("File Size", doc_profile["file_size_formatted"])
        col4.metric("Extraction Status", doc_profile["status"])

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total Characters", f"{doc_profile['character_count']:,}")
        col6.metric("Non-Whitespace Chars", f"{doc_profile['non_whitespace_characters']:,}")
        col7.metric("Word Count", f"{doc_profile['word_count']:,}")
        col8.metric("Line Count", f"{doc_profile['line_count']:,}")

        col9, col10, col11, col12 = st.columns(4)
        # Format-specific metrics
        if doc_profile["file_type"] == "PDF":
            col9.metric("Page Count", f"{doc_profile['page_count']:,}")
        else:
            col9.metric("Page Count", "N/A", help=f"Page count is not applicable for {doc_profile['file_type']} files.")

        if doc_profile["file_type"] == "DOCX":
            col10.metric("Paragraph Count", f"{doc_profile['paragraph_count']:,}")
        else:
            col10.metric("Paragraph Count", "N/A", help=f"Paragraph count is not applicable for {doc_profile['file_type']} files.")

        col11.metric("Avg. Word Length", f"{doc_profile['average_word_length']:.1f} chars")
        col12.metric("Scanned/Image PDF?", "Yes (No OCR)" if doc_profile.get("is_scanned") else "No")

        # ---------------------------------------------------------
        # Phase 16: Rule-Based Document Quality Assessment
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Document Quality Assessment")
        quality_findings = assess_document_quality(raw_text)
        st.session_state["doc_quality_findings"] = quality_findings
        if quality_findings:
            st.dataframe(quality_findings, use_container_width=True, hide_index=True)
        else:
            st.success("No rule-based text quality concerns were found.")

        # ---------------------------------------------------------
        # Phase 16: Optional, Non-Destructive Text Cleaning
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Optional Text Cleaning")
        st.caption(
            "Cleaning is previewed before it is applied. The original extracted text is kept unchanged."
        )

        setting_columns = st.columns(2)
        with setting_columns[0]:
            trim_whitespace = st.checkbox(
                "Trim leading and trailing whitespace",
                key="doc_clean_trim",
            )
            normalize_endings = st.checkbox(
                "Normalize line endings",
                key="doc_clean_normalize_endings",
            )
        with setting_columns[1]:
            collapse_blank_lines = st.checkbox(
                "Collapse excessive blank lines",
                key="doc_clean_collapse_blank_lines",
            )
            remove_repeated_lines = st.checkbox(
                "Remove repeated lines (keeps the first occurrence)",
                key="doc_clean_remove_repeated_lines",
                help="This is optional because repeated lines can be meaningful in some documents.",
            )

        cleaning_preview = build_cleaning_preview(
            st.session_state["doc_clean_original_text"],
            trim_whitespace=trim_whitespace,
            normalize_endings=normalize_endings,
            collapse_blank_lines=collapse_blank_lines,
            remove_repeated_lines=remove_repeated_lines,
        )
        st.session_state["doc_clean_preview_text"] = cleaning_preview["cleaned_text"]

        if cleaning_preview["selected_operations"]:
            st.caption(
                "Selected operations: "
                + "; ".join(cleaning_preview["selected_operations"])
            )
        else:
            st.info("Select one or more cleaning operations to create a preview.")

        original_column, preview_column = st.columns(2)
        with original_column:
            st.text_area(
                "Original extracted text",
                value=cleaning_preview["original_text"],
                height=240,
                disabled=True,
            )
        with preview_column:
            st.text_area(
                "Cleaned text preview",
                value=cleaning_preview["cleaned_text"],
                height=240,
                disabled=True,
            )

        action_column, reset_column = st.columns(2)
        with action_column:
            if st.button(
                "Apply selected cleaning",
                key="doc_clean_apply",
                disabled=not cleaning_preview["changed"],
            ):
                st.session_state["doc_cleaned_text"] = cleaning_preview["cleaned_text"]
                st.success("The cleaned text was saved separately from the original extraction.")
        with reset_column:
            if st.button("Reset to original text", key="doc_clean_reset"):
                reset_document_cleaning_state(st.session_state)
                st.rerun()

        cleaned_text = st.session_state.get("doc_cleaned_text")
        if cleaned_text is not None:
            st.text_area(
                "Applied cleaned text",
                value=cleaned_text,
                height=180,
                disabled=True,
            )
            base_filename = doc_name.rsplit(".", 1)[0]
            st.download_button(
                label="Download applied cleaned text (.txt)",
                data=cleaned_text.encode("utf-8"),
                file_name=f"{base_filename}_cleaned.txt",
                mime="text/plain",
            )

        # ---------------------------------------------------------
        # Text Content Preview
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📖 Extracted Text Content")

        if raw_text.strip():
            with st.expander("🔍 View Extracted Text Preview", expanded=True):
                st.text_area(
                    label="Extracted Text (first 1,500 characters):",
                    value=doc_profile["preview_text"],
                    height=280,
                    disabled=True,
                )
                st.caption(
                    f"Showing preview of {len(raw_text):,} extracted characters across {doc_profile['word_count']:,} words."
                )

            # -----------------------------------------------------
            # Export Extracted Text
            # -----------------------------------------------------
            st.markdown("---")
            st.subheader("📥 Export Extracted Text")
            base_filename = doc_name.rsplit(".", 1)[0]
            st.download_button(
                label="📥 Download Raw Extracted Text (.txt)",
                data=raw_text.encode("utf-8"),
                file_name=f"{base_filename}_extracted.txt",
                mime="text/plain",
                help="Download the complete extracted raw text as a UTF-8 encoded text file.",
            )
        else:
            st.info(
                "💡 No textual characters were extracted from this document. "
                "For scanned PDFs or image-based files, Optical Character Recognition (OCR) is required, "
                "which is not included in this phase."
            )

    except DocumentProcessingError as dpe:
        st.error(f"❌ **Document Processing Error:** {dpe}")
    except Exception as e:
        st.error(
            f"❌ **Unexpected Error:** An unexpected error occurred while processing '{doc_name}'. "
            f"Details: {e}"
        )

else:
    st.info("💡 Please upload a TXT, PDF, or DOCX document to begin unstructured document profiling.")

st.markdown("---")
st.caption(
    "🧭 **AutoDataIQ Architecture:** Document processing in Phase 15 establishes text ingestion and profiling. "
    "Text sanitization, normalization, and advanced linguistic intelligence will be introduced in subsequent phases."
)
