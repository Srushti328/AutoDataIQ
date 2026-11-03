"""Streamlit page for Phase 17 Rule-Based Text Analysis and Insights."""

import hashlib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.document_processor import (
    DocumentProcessingError,
    detect_document_type,
    extract_document_text,
    profile_document,
)
from modules.document_quality import initialize_document_cleaning_state
from modules.text_analysis_engine import (
    DOCUMENT_ANALYSIS_STATE_KEYS,
    clear_document_analysis_state,
    get_analyzable_text,
    run_text_analysis,
)

st.set_page_config(
    page_title="Text Analysis & Insights - AutoDataIQ",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Unstructured Text Analysis & Rule-Based Insights")
st.write(
    "Perform automated, explainable text analytics on unstructured documents. Inspect lexical statistics, "
    "identify frequent keywords, evaluate sentiment using transparent rule-based dictionaries, discover "
    "discussion topics, and generate prioritized findings."
)

st.info(
    "🔒 **Workflow Isolation Guarantee:** This text analysis module operates exclusively on unstructured "
    "text extracts. It does not access, modify, or interfere with tabular structured datasets "
    "(`raw_data`, `cleaned_data`), KPI definitions, or executive dashboard state."
)

# ---------------------------------------------------------------------
# Document Selection / Loader Area
# ---------------------------------------------------------------------
st.subheader("📄 Document Selection")

sample_docs = {
    "None": None,
    "Customer Feedback (Reviews) [Phase 17 Sample]": "datasets/sample_customer_feedback.txt",
    "Sample Plain Text (.txt)": "datasets/sample_document.txt",
    "Sample Word Document (.docx)": "datasets/sample_report.docx",
    "Sample Text-based PDF (.pdf)": "datasets/sample_document.pdf",
    "Sample Scanned PDF (.pdf) [No Text]": "datasets/sample_scanned_mock.pdf",
}

with st.expander("📁 Upload or Select a Document", expanded="doc_file_bytes" not in st.session_state):
    doc_cols = st.columns([2, 2])
    with doc_cols[0]:
        uploaded_doc = st.file_uploader(
            "Upload a document (TXT, PDF, DOCX):",
            type=["txt", "pdf", "docx"],
            key="text_analysis_uploader",
            help="Upload a new document or use one previously loaded on the Document Processing page.",
        )
    with doc_cols[1]:
        selected_sample = st.selectbox(
            "Or load a bundled repository sample:",
            options=list(sample_docs.keys()),
            index=0,
            key="text_analysis_sample_selector",
            help="Choose from pre-bundled documents to test sentiment and topic analysis immediately.",
        )

    if uploaded_doc is not None:
        file_bytes = uploaded_doc.getvalue()
        file_name = uploaded_doc.name
        if st.session_state.get("doc_filename") != file_name or st.session_state.get("doc_file_bytes") != file_bytes:
            st.session_state["doc_file_bytes"] = file_bytes
            st.session_state["doc_filename"] = file_name
            clear_document_analysis_state(st.session_state)
            # Extract and initialize document cleaning state
            try:
                ext_res = extract_document_text(file_bytes, file_name)
                st.session_state["doc_extraction_result"] = ext_res
                st.session_state["doc_profile"] = profile_document(ext_res)
                doc_source_id = f"{file_name}:{hashlib.sha256(file_bytes).hexdigest()}"
                initialize_document_cleaning_state(st.session_state, doc_source_id, ext_res.get("text", ""))
            except Exception as e:
                st.error(f"Failed to extract uploaded document: {e}")

    elif selected_sample and sample_docs[selected_sample]:
        sample_path = sample_docs[selected_sample]
        sample_file_name = Path(sample_path).name
        if Path(sample_path).exists():
            if st.session_state.get("doc_filename") != sample_file_name:
                with open(sample_path, "rb") as sf:
                    sample_bytes = sf.read()
                st.session_state["doc_file_bytes"] = sample_bytes
                st.session_state["doc_filename"] = sample_file_name
                clear_document_analysis_state(st.session_state)
                try:
                    ext_res = extract_document_text(sample_bytes, sample_file_name)
                    st.session_state["doc_extraction_result"] = ext_res
                    st.session_state["doc_profile"] = profile_document(ext_res)
                    doc_source_id = f"{sample_file_name}:{hashlib.sha256(sample_bytes).hexdigest()}"
                    initialize_document_cleaning_state(st.session_state, doc_source_id, ext_res.get("text", ""))
                except Exception as e:
                    st.error(f"Failed to extract sample document: {e}")

# ---------------------------------------------------------------------
# Check Document Availability
# ---------------------------------------------------------------------
if "doc_file_bytes" not in st.session_state or st.session_state["doc_file_bytes"] is None:
    st.warning("⚠️ No document loaded. Please upload a document or choose a sample from above.")
    st.stop()

doc_name = st.session_state.get("doc_filename", "uploaded_document")
doc_profile = st.session_state.get("doc_profile", {})
file_type = doc_profile.get("file_type", "Unknown")

st.write(
    f"**Active Document:** `{doc_name}` | **Format:** {file_type} | "
    f"**Size:** {doc_profile.get('file_size_formatted', 'N/A')}"
)

# ---------------------------------------------------------------------
# Text Source Selection (Original Extracted vs. Cleaned)
# ---------------------------------------------------------------------
has_cleaned = (
    "doc_cleaned_text" in st.session_state
    and st.session_state["doc_cleaned_text"] is not None
    and len(st.session_state["doc_cleaned_text"].strip()) > 0
)

st.markdown("---")
st.subheader("⚙️ Analysis Source Configuration")

source_col1, source_col2 = st.columns([2, 3])

with source_col1:
    source_options = ["Original Extracted Text"]
    if has_cleaned:
        source_options.insert(0, "Cleaned Text (Phase 16)")

    selected_source_label = st.radio(
        "Select text version to analyze:",
        options=source_options,
        index=0,
        help="Analyze the raw text or the sanitized version after Phase 16 non-destructive cleaning.",
    )

text_mode = "cleaned" if "Cleaned" in selected_source_label else "original"
analysis_text, source_title, source_desc = get_analyzable_text(st.session_state, text_source=text_mode)

with source_col2:
    st.caption(f"**Current Source:** {source_title}")
    st.caption(source_desc)
    if not has_cleaned:
        st.info(
            "💡 **Tip:** You can generate and preview a cleaned version (whitespace trimming, "
            "line ending normalization, duplicate line removal) on the **12_Document_Processing** page."
        )

# ---------------------------------------------------------------------
# Empty / Unusable Text Guard
# ---------------------------------------------------------------------
if not analysis_text or not analysis_text.strip():
    st.warning(
        f"⚠️ **Notice: No extractable text available in `{doc_name}`.**\n\n"
        "The selected document contains 0 non-whitespace characters. "
        "If this is a scanned or image-only PDF, optical character recognition (OCR) is required, "
        "which is outside the scope of Phase 17."
    )
    st.stop()

# ---------------------------------------------------------------------
# Run Text Analysis Engine
# ---------------------------------------------------------------------
analysis_result = run_text_analysis(analysis_text)
stats = analysis_result["statistics"]
keywords = analysis_result["keywords"]
sentiment = analysis_result["sentiment"]
topics = analysis_result["topics"]
insights = analysis_result["insights"]

# Save to isolated session-state
st.session_state["doc_analysis_result"] = analysis_result
st.session_state["doc_analysis_text_source"] = text_mode

# ---------------------------------------------------------------------
# 1. Text Statistics Overview
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Text Statistics Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Characters", f"{stats['total_characters']:,}")
c2.metric("Total Words", f"{stats['total_words']:,}")
c3.metric("Sentences", f"{stats['total_sentences']:,}")
c4.metric("Paragraphs", f"{stats['total_paragraphs']:,}")
c5.metric("Avg Word Length", f"{stats['average_word_length']:.1f} chars")
c6.metric("Avg Sentence Length", f"{stats['average_sentence_length']:.1f} words")

# ---------------------------------------------------------------------
# 2. Visual Analysis Tabs
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Visual Exploration & Distributions")

tab_kw, tab_sent, tab_top, tab_text = st.tabs([
    "🔑 Keyword Frequencies",
    "😊 Sentiment Analysis",
    "🎯 Topic Identification",
    "📜 Analyzed Text Preview",
])

# -------------------------------------------------------------
# Tab 1: Keyword Frequencies
# -------------------------------------------------------------
with tab_kw:
    st.markdown("#### Meaningful Keyword Frequency (Stop-Words Removed)")
    st.caption(
        "Keywords are extracted by removing common English stop-words (e.g. *the, and, was, is*), "
        "normalizing casing, and ignoring isolated punctuation."
    )

    if keywords:
        kw_df_data = [
            {"Keyword": k["keyword"], "Frequency": k["frequency"], "Percentage (%)": k["percentage"]}
            for k in keywords
        ]
        # Sort ascending for horizontal bar chart
        chart_data = sorted(kw_df_data, key=lambda x: x["Frequency"])

        fig_kw = px.bar(
            chart_data,
            x="Frequency",
            y="Keyword",
            orientation="h",
            text="Frequency",
            title=f"Top {len(keywords)} Meaningful Keywords in {doc_name}",
            labels={"Frequency": "Occurrences", "Keyword": "Keyword"},
            color="Frequency",
            color_continuous_scale="Blues",
        )
        fig_kw.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_kw, use_container_width=True)

        with st.expander("📋 View Keyword Frequency Table", expanded=False):
            st.dataframe(kw_df_data, use_container_width=True, hide_index=True)
    else:
        st.info("No meaningful keywords found after stop-word filtering.")

# -------------------------------------------------------------
# Tab 2: Sentiment Analysis
# -------------------------------------------------------------
with tab_sent:
    st.markdown("#### Rule-Based Sentiment Analysis")
    st.caption(
        "Sentences are classified using transparent positive and negative keyword lexicons. "
        "Net sentence score determines classification: Positive (score > 0), Negative (score < 0), "
        "or Neutral (score = 0 or balanced)."
    )

    sent_counts = sentiment.get("sentence_counts", {})
    sent_pcts = sentiment.get("sentence_percentages", {})
    overall_sentiment = sentiment.get("overall_sentiment", "Neutral")

    # Overview Metrics
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Overall Document Tone", overall_sentiment)
    s_col2.metric("Positive Sentences", f"{sent_counts.get('Positive', 0)} ({sent_pcts.get('Positive', 0.0)}%)")
    s_col3.metric("Negative Sentences", f"{sent_counts.get('Negative', 0)} ({sent_pcts.get('Negative', 0.0)}%)")
    s_col4.metric("Neutral Sentences", f"{sent_counts.get('Neutral', 0)} ({sent_pcts.get('Neutral', 0.0)}%)")

    # Visual Sentiment Charts
    sent_chart_cols = st.columns([1, 1])

    with sent_chart_cols[0]:
        # Donut Chart for Sentiment Share
        sent_labels = ["Positive", "Neutral", "Negative"]
        sent_values = [
            sent_counts.get("Positive", 0),
            sent_counts.get("Neutral", 0),
            sent_counts.get("Negative", 0),
        ]
        color_map = {
            "Positive": "#2ECC71",
            "Neutral": "#95A5A6",
            "Negative": "#E74C3C",
        }

        fig_sent = go.Figure(
            data=[
                go.Pie(
                    labels=sent_labels,
                    values=sent_values,
                    hole=0.45,
                    marker=dict(colors=[color_map[l] for l in sent_labels]),
                    textinfo="label+percent+value",
                )
            ]
        )
        fig_sent.update_layout(
            title="Sentiment Distribution by Sentence Count",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with sent_chart_cols[1]:
        # Top Matched Keyword Triggers
        pos_kws = sentiment.get("positive_keywords_found", [])
        neg_kws = sentiment.get("negative_keywords_found", [])

        st.markdown("**Top Positive Keyword Triggers:**")
        if pos_kws:
            st.write(", ".join(f"`{k['word']}` ({k['count']}x)" for k in pos_kws[:6]))
        else:
            st.write("None detected.")

        st.markdown("**Top Negative Keyword Triggers:**")
        if neg_kws:
            st.write(", ".join(f"`{k['word']}` ({k['count']}x)" for k in neg_kws[:6]))
        else:
            st.write("None detected.")

        st.caption(sentiment.get("explanation", ""))

    with st.expander("🔍 View Sentence-by-Sentence Sentiment Classification", expanded=False):
        classified_sentences = sentiment.get("classified_sentences", [])
        if classified_sentences:
            table_records = []
            for item in classified_sentences:
                table_records.append({
                    "#": item["index"],
                    "Sentence": item["sentence"][:120] + ("..." if len(item["sentence"]) > 120 else ""),
                    "Classification": item["sentiment"],
                    "Pos Words": ", ".join(item["pos_words"]) or "-",
                    "Neg Words": ", ".join(item["neg_words"]) or "-",
                })
            st.dataframe(table_records, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# Tab 3: Topic Identification
# -------------------------------------------------------------
with tab_top:
    st.markdown("#### Rule-Based Topic Identification")
    st.caption(
        "Sentences are mapped to business domains using keyword taxonomies. "
        "A single sentence may match multiple topics when relevant."
    )

    ranked_topics = topics.get("topics", [])
    if ranked_topics:
        topic_chart_data = [
            {"Topic": t["topic"], "Keyword Matches": t["match_count"], "Sentences": t["sentence_count"]}
            for t in sorted(ranked_topics, key=lambda x: x["match_count"])
        ]

        fig_top = px.bar(
            topic_chart_data,
            x="Keyword Matches",
            y="Topic",
            orientation="h",
            text="Keyword Matches",
            title="Discussion Topics Ranked by Keyword Matches",
            labels={"Keyword Matches": "Keyword Matches", "Topic": "Domain Topic"},
            color="Keyword Matches",
            color_continuous_scale="Purples",
        )
        fig_top.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("##### Detailed Topic Evidence & Sample Sentences")
        for topic in ranked_topics:
            with st.expander(
                f"📌 {topic['topic']} — {topic['match_count']} keyword triggers across {topic['sentence_count']} sentence(s)",
                expanded=False,
            ):
                kw_str = ", ".join(f"`{k['keyword']}` ({k['count']}x)" for k in topic["matched_keywords"])
                st.write(f"**Matched Keywords:** {kw_str}")
                if topic.get("sample_sentences"):
                    st.write("**Sample Excerpts:**")
                    for s in topic["sample_sentences"]:
                        st.markdown(f"> *\"{s}\"*")
    else:
        st.info("No predefined topics matched the vocabulary in this document.")

# -------------------------------------------------------------
# Tab 4: Analyzed Text Preview
# -------------------------------------------------------------
with tab_text:
    st.markdown("#### Active Text Content")
    st.caption(f"Displaying content currently evaluated ({source_title}).")
    st.text_area(
        label="Document Text:",
        value=analysis_text,
        height=320,
        disabled=True,
    )

# ---------------------------------------------------------------------
# 3. Prioritized Rule-Based Insights
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("💡 Rule-Based Analytical Insights")
st.caption(
    "Automated findings derived from lexical distributions, sentiment classifications, and topic taxonomy matching. "
    "Every insight includes supporting evidence and priority weighting."
)

if insights:
    priority_order = {"High": 1, "Medium": 2, "Low": 3}
    sorted_insights = sorted(insights, key=lambda x: priority_order.get(x.get("priority", "Low"), 4))

    for idx, insight in enumerate(sorted_insights, 1):
        prio = insight.get("priority", "Low")
        prio_color = {
            "High": "🔴 **High Priority**",
            "Medium": "🟡 **Medium Priority**",
            "Low": "🟢 **Low Priority**",
        }.get(prio, prio)

        with st.expander(f"{idx}. {insight['title']} — [{prio}]", expanded=(prio == "High" or idx == 1)):
            c_prio, c_cat = st.columns([1, 1])
            with c_prio:
                st.markdown(f"**Priority:** {prio_color}")
            with c_cat:
                st.markdown(f"**Category:** `{insight.get('category', 'General')}`")

            st.markdown(f"**Finding:** {insight.get('explanation', '')}")
            st.info(f"📊 **Supporting Evidence:** {insight.get('evidence', 'N/A')}")
else:
    st.info("No specific insights were identified for this document.")

# ---------------------------------------------------------------------
# 4. Methodological Transparency & Limitations Note
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("ℹ️ Methodology & Limitations of Rule-Based Text Analytics")

with st.expander("📖 Read Algorithmic Logic & Boundary Disclaimers", expanded=False):
    st.markdown(
        """
        ### Transparency & Explainability Disclaimers:
        1. **Lexicon-Driven Sentiment Analysis:**
           Sentiment is calculated using deterministic keyword lookups against curated positive and negative word lists.
           It does **not** employ opaque black-box machine learning models or commercial third-party cloud APIs.
        2. **Absence of Syntactic Context:**
           Keyword matching cannot evaluate complex grammatical structures, double negations (e.g. *"not bad"*, *"hardly great"*),
           or conversational nuance.
        3. **Irony and Sarcasm:**
           Phrases like *"What a wonderful disaster"* will register as positive due to the term *"wonderful"*.
        4. **Topic Taxonomies:**
           Topic detection evaluates domain-specific keyword clusters. Feedback containing colloquialisms or words outside
           the predefined vocabulary may be assigned to *General Feedback* or remain unclassified.
        5. **Strict Data Protection:**
           Processing occurs entirely locally in memory. No customer data or document text is transmitted externally.
        """
    )
