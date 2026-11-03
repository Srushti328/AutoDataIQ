"""Consolidated PDF Report Generation Engine for AutoDataIQ.

Phase 18 module generating professional, publication-ready PDF reports for:
1. Structured Tabular Data Workflows (Profiling, Semantics, Quality, Cleaning, Analysis, KPIs, Dashboard)
2. Unstructured Document Analytics Workflows (Text Statistics, Keywords, Sentiment, Topics, Insights)

Built with ReportLab using deterministic flowables and custom NumberedCanvas for dynamic pagination.
"""

from datetime import datetime
import io
from typing import Any, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# =====================================================================
# Numbered Canvas for Dynamic Page Numbering & Running Headers/Footers
# =====================================================================

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that dynamically calculates and renders 'Page X of Y' footers and headers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, total_pages: int) -> None:
        self.saveState()
        page_width, page_height = letter

        # Running Header (pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#2C3E50"))
            self.drawString(54, page_height - 36, "AutoDataIQ: Intelligent Data Quality Assessment & Adaptive Analytics")
            self.setStrokeColor(colors.HexColor("#BDC3C7"))
            self.setLineWidth(0.5)
            self.line(54, page_height - 40, page_width - 54, page_height - 40)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#BDC3C7"))
        self.setLineWidth(0.5)
        self.line(54, 42, page_width - 54, 42)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        self.drawString(54, 30, "AutoDataIQ Framework — Academic Project Report")
        self.drawRightString(page_width - 54, 30, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# =====================================================================
# Style Palette & Helper Builders
# =====================================================================

def _build_report_styles() -> dict[str, ParagraphStyle]:
    """Construct a clean, cohesive typographical hierarchy."""
    base_styles = getSampleStyleSheet()

    custom_styles: dict[str, ParagraphStyle] = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1B365D"),
            alignment=0,
            spaceAfter=4,
        ),
        "ReportSubtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=10,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1B365D"),
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "SubSectionHeading": ParagraphStyle(
            "SubSectionHeading",
            parent=base_styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#2C3E50"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "ReportBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#2D3748"),
            spaceAfter=4,
        ),
        "BodyBold": ParagraphStyle(
            "ReportBodyBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#1A202C"),
            spaceAfter=4,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=0,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#2D3748"),
            alignment=0,
        ),
        "TableCellBold": ParagraphStyle(
            "TableCellBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#1A202C"),
            alignment=0,
        ),
        "MetaLabel": ParagraphStyle(
            "MetaLabel",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#4A5568"),
        ),
        "MetaValue": ParagraphStyle(
            "MetaValue",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1A202C"),
        ),
        "Notice": ParagraphStyle(
            "Notice",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#718096"),
            spaceAfter=6,
        ),
    }
    return custom_styles


def _create_metadata_box(
    metadata_pairs: list[tuple[str, str]],
    styles: dict[str, ParagraphStyle],
    available_width: float = 504.0,
) -> Table:
    """Format key-value metadata into a styled two-column card."""
    half_width = available_width / 2.0
    label_w = 90.0
    val_w = half_width - label_w

    table_data = []
    # Pair items into rows of 2 pairs
    for i in range(0, len(metadata_pairs), 2):
        row = []
        k1, v1 = metadata_pairs[i]
        row.extend([Paragraph(k1, styles["MetaLabel"]), Paragraph(v1, styles["MetaValue"])])
        if i + 1 < len(metadata_pairs):
            k2, v2 = metadata_pairs[i + 1]
            row.extend([Paragraph(k2, styles["MetaLabel"]), Paragraph(v2, styles["MetaValue"])])
        else:
            row.extend(["", ""])
        table_data.append(row)

    col_widths = [label_w, val_w, label_w, val_w]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#EDF2F7")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    return t


def _create_standard_table(
    headers: list[str],
    rows: list[list[Any]],
    col_widths: list[float],
    styles: dict[str, ParagraphStyle],
    header_color: str = "#1B365D",
) -> Table:
    """Build a wrapped, clean table with alternating row colors."""
    wrapped_header = [Paragraph(h, styles["TableHeader"]) for h in headers]
    table_data = [wrapped_header]

    for row_idx, row in enumerate(rows):
        wrapped_row = []
        for cell in row:
            if isinstance(cell, Paragraph):
                wrapped_row.append(cell)
            else:
                text_val = str(cell) if cell is not None else "—"
                wrapped_row.append(Paragraph(text_val, styles["TableCell"]))
        table_data.append(wrapped_row)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
    ]

    for i in range(1, len(table_data)):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")))

    t.setStyle(TableStyle(ts))
    return t


# =====================================================================
# 1. Structured Data PDF Report Generator
# =====================================================================

def generate_structured_data_pdf(
    dataset_name: str,
    *,
    profile_summary: Optional[dict[str, Any]] = None,
    column_profile: Optional[list[dict[str, Any]]] = None,
    semantic_types: Optional[list[dict[str, Any]]] = None,
    quality_score: Optional[dict[str, Any]] = None,
    quality_findings: Optional[list[dict[str, Any]]] = None,
    cleaning_recommendations: Optional[list[dict[str, Any]]] = None,
    applied_cleaning_log: Optional[list[str]] = None,
    insights: Optional[list[dict[str, Any]]] = None,
    kpis: Optional[list[dict[str, Any]]] = None,
    is_cleaned: bool = False,
) -> bytes:
    """Generate a downloadable PDF report for the structured-data workflow.

    Only sections with actual provided data are rendered.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = _build_report_styles()
    story = []
    width = 504.0  # 8.5 * 72 - 108

    # Title & Subtitle
    story.append(Paragraph("AutoDataIQ — Structured Data Intelligence Report", styles["ReportTitle"]))
    mode_label = "Cleaned Dataset" if is_cleaned else "Original Raw Dataset"
    story.append(
        Paragraph(
            f"Comprehensive Data Quality Assessment, Profiling, Semantic Typing, KPIs, and Insights | {mode_label}",
            styles["ReportSubtitle"],
        )
    )

    # Metadata Block
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = [
        ("Dataset Name:", dataset_name),
        ("Generation Date:", now_str),
        ("Active Pipeline:", mode_label),
        ("Framework Version:", "AutoDataIQ 2.0 (MCA Project)"),
    ]
    if profile_summary:
        meta.append(("Total Records:", f"{profile_summary.get('row_count', 'N/A'):,}"))
        meta.append(("Total Columns:", f"{profile_summary.get('col_count', 'N/A'):,}"))

    story.append(_create_metadata_box(meta, styles, available_width=width))
    story.append(Spacer(1, 10))

    # 1. Dataset Profiling Overview
    if profile_summary:
        story.append(Paragraph("1. Dataset Profiling Summary", styles["SectionHeading"]))
        missing_cells = profile_summary.get("missing_cells", 0)
        missing_pct = profile_summary.get("missing_percentage", 0.0)
        dup_rows = profile_summary.get("duplicate_rows", 0)
        dup_pct = profile_summary.get("duplicate_percentage", 0.0)

        summary_rows = [
            ["Row Count", f"{profile_summary.get('row_count', 0):,}"],
            ["Column Count", f"{profile_summary.get('col_count', 0):,}"],
            ["Total Data Cells", f"{profile_summary.get('total_cells', 0):,}"],
            ["Missing Cells", f"{missing_cells:,} ({missing_pct:.1f}%)"],
            ["Duplicate Rows", f"{dup_rows:,} ({dup_pct:.1f}%)"],
            ["Memory Usage", f"{profile_summary.get('memory_kb', 0):.1f} KB"],
        ]
        story.append(
            _create_standard_table(
                ["Profiling Dimension", "Observed Measurement"],
                summary_rows,
                [200.0, 304.0],
                styles,
                header_color="#2C3E50",
            )
        )
        story.append(Spacer(1, 8))

    # Column Level Profiles (top columns if available)
    if column_profile:
        story.append(Paragraph("Column-Level Structural Profiles", styles["SubSectionHeading"]))
        col_rows = []
        for c in column_profile[:35]:
            col_rows.append([
                c.get("Column", "—"),
                c.get("Data Type", "—"),
                f"{c.get('Missing Count', 0)} ({c.get('Missing %', 0.0):.1f}%)",
                f"{c.get('Unique Values', '—')}",
            ])
        story.append(
            _create_standard_table(
                ["Column Name", "Storage Dtype", "Missing Values", "Unique Count"],
                col_rows,
                [140.0, 110.0, 130.0, 124.0],
                styles,
                header_color="#34495E",
            )
        )
        if len(column_profile) > 35:
            story.append(Paragraph(f"*Note: Showing first 35 of {len(column_profile)} columns.*", styles["Notice"]))
        story.append(Spacer(1, 10))

    # 2. Semantic Data-Type Detection
    if semantic_types:
        story.append(Paragraph("2. Semantic Data-Type Inference", styles["SectionHeading"]))
        sem_rows = []
        for s in semantic_types[:30]:
            sem_rows.append([
                s.get("Column", s.get("column", "—")),
                s.get("Semantic Type", s.get("semantic_type", "—")),
                s.get("Confidence", s.get("confidence", "—")),
                s.get("Reasoning", s.get("reasoning", "—")),
            ])
        story.append(
            _create_standard_table(
                ["Column Name", "Inferred Semantic Type", "Confidence", "Diagnostic Rationale"],
                sem_rows,
                [120.0, 100.0, 70.0, 214.0],
                styles,
                header_color="#1E3A8A",
            )
        )
        story.append(Spacer(1, 10))

    # 3. Data Quality Intelligence & Six-Dimension Score
    if quality_score:
        story.append(Paragraph("3. Data Quality Scoring & Health", styles["SectionHeading"]))
        overall_score = quality_score.get("overall_score", 0.0)
        rating = quality_score.get("rating", "N/A")

        score_intro = (
            f"<b>Overall Quality Score:</b> {overall_score:.1f} / 100 &nbsp;|&nbsp; "
            f"<b>Hygiene Rating:</b> {rating} &nbsp;|&nbsp; "
            f"Evaluated across six standardized dimensions."
        )
        story.append(Paragraph(score_intro, styles["BodyBold"]))
        story.append(Spacer(1, 4))

        dimension_scores = quality_score.get("dimension_scores", {})
        if dimension_scores:
            dim_rows = []
            for dim, d_data in dimension_scores.items():
                w = d_data.get("weight", 0) * 100
                s = d_data.get("score", 0.0)
                pen = d_data.get("penalty", 0.0)
                dim_rows.append([
                    dim,
                    f"{w:.0f}%",
                    f"{pen:.1f} pts",
                    f"{s:.1f} / 100",
                ])
            story.append(
                _create_standard_table(
                    ["Quality Dimension", "Dimension Weight", "Penalties Deducted", "Dimension Score"],
                    dim_rows,
                    [160.0, 100.0, 120.0, 124.0],
                    styles,
                    header_color="#0F766E",
                )
            )
        story.append(Spacer(1, 8))

    # Quality Findings Table
    if quality_findings:
        story.append(Paragraph("Identified Data Quality Anomalies", styles["SubSectionHeading"]))
        findings_rows = []
        for f in quality_findings[:10]:
            findings_rows.append([
                f.get("Issue Type", f.get("issue", "—")),
                f.get("Affected Column", f.get("column", "Dataset-level")),
                f.get("Severity", "Medium"),
                f.get("Description", f.get("evidence", "—")),
            ])
        story.append(
            _create_standard_table(
                ["Issue Type", "Scope / Column", "Severity", "Empirical Evidence & Details"],
                findings_rows,
                [110.0, 110.0, 60.0, 224.0],
                styles,
                header_color="#B91C1C",
            )
        )
        story.append(Spacer(1, 10))

    # 4. Cleaning Recommendations & Transformation Audit
    if cleaning_recommendations or applied_cleaning_log:
        story.append(Paragraph("4. Cleaning Prescriptions & Transformations", styles["SectionHeading"]))
        if applied_cleaning_log:
            story.append(Paragraph("<b>Applied Transformation Pipeline Audit:</b>", styles["BodyBold"]))
            for log_entry in applied_cleaning_log[:8]:
                story.append(Paragraph(f"• {log_entry}", styles["Body"]))
            story.append(Spacer(1, 4))

        if cleaning_recommendations and not applied_cleaning_log:
            rec_rows = []
            for r in cleaning_recommendations[:8]:
                rec_rows.append([
                    r.get("Priority", "Medium"),
                    r.get("Column", "—"),
                    r.get("Recommendation", "—"),
                    r.get("Suggested Method", "—"),
                ])
            story.append(
                _create_standard_table(
                    ["Priority", "Target", "Recommended Action", "Methodology"],
                    rec_rows,
                    [64.0, 110.0, 170.0, 160.0],
                    styles,
                    header_color="#D97706",
                )
            )
        story.append(Spacer(1, 10))

    # 5. Key Business & Operational KPIs
    if kpis:
        story.append(Paragraph("5. Domain KPI Detection Summary", styles["SectionHeading"]))
        kpi_rows = []
        for k in kpis[:12]:
            kpi_rows.append([
                k.get("KPI Name", k.get("name", "—")),
                str(k.get("Formatted Value", k.get("value", "—"))),
                k.get("Category", "General"),
                k.get("Formula", k.get("calculation_formula", "—")),
            ])
        story.append(
            _create_standard_table(
                ["KPI Indicator", "Computed Value", "Category", "Calculation Formula"],
                kpi_rows,
                [130.0, 94.0, 90.0, 190.0],
                styles,
                header_color="#4338CA",
            )
        )
        story.append(Spacer(1, 10))

    # 6. Prioritized Analytical Insights
    if insights:
        story.append(Paragraph("6. Prioritized Analytical Insights", styles["SectionHeading"]))
        ins_rows = []
        for ins in insights[:20]:
            prio = ins.get("priority", "Medium")
            ins_rows.append([
                prio,
                ins.get("title", "—"),
                ins.get("category", "Analysis"),
                f"{ins.get('explanation', '')} <i>[Evidence: {ins.get('evidence', 'N/A')}]</i>",
            ])
        story.append(
            _create_standard_table(
                ["Priority", "Insight Title", "Category", "Analytical Findings & Supporting Evidence"],
                ins_rows,
                [54.0, 120.0, 80.0, 250.0],
                styles,
                header_color="#047857",
            )
        )
        story.append(Spacer(1, 10))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


# =====================================================================
# 2. Unstructured Document PDF Report Generator
# =====================================================================

def generate_unstructured_document_pdf(
    document_name: str,
    *,
    file_type: str = "TXT",
    text_source: str = "Original Extracted Text",
    doc_profile: Optional[dict[str, Any]] = None,
    quality_findings: Optional[list[dict[str, Any]]] = None,
    applied_cleaning: Optional[dict[str, Any]] = None,
    text_statistics: Optional[dict[str, Any]] = None,
    keywords: Optional[list[dict[str, Any]]] = None,
    sentiment: Optional[dict[str, Any]] = None,
    topics: Optional[list[dict[str, Any]]] = None,
    insights: Optional[list[dict[str, Any]]] = None,
) -> bytes:
    """Generate a downloadable PDF report for the unstructured-document workflow.

    Includes document profiling, text statistics, quality findings, keyword analysis,
    sentiment distributions, topic identification, and rule-based insights.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = _build_report_styles()
    story = []
    width = 504.0

    # Title & Subtitle
    story.append(Paragraph("AutoDataIQ — Unstructured Document Analytics Report", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Text Extraction, Structural Profiling, Rule-Based Sentiment, Topic Mapping & Prioritized Insights",
            styles["ReportSubtitle"],
        )
    )

    # Metadata Block
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_size_str = doc_profile.get("file_size_formatted", "N/A") if doc_profile else "N/A"
    meta = [
        ("Source Document:", document_name),
        ("Document Format:", file_type),
        ("Analysis Date:", now_str),
        ("Text Version Analyzed:", text_source),
        ("File Size:", file_size_str),
        ("Framework Version:", "AutoDataIQ 2.0 (MCA Project)"),
    ]
    story.append(_create_metadata_box(meta, styles, available_width=width))
    story.append(Spacer(1, 10))

    # 1. Document Profiling & Text Statistics
    if text_statistics or doc_profile:
        story.append(Paragraph("1. Document Profiling & Text Statistics", styles["SectionHeading"]))
        stat_rows = []
        if doc_profile:
            stat_rows.append(["File Size", doc_profile.get("file_size_formatted", "—")])
            if doc_profile.get("file_type") == "PDF":
                stat_rows.append(["Page Count", f"{doc_profile.get('page_count', 'N/A')}"])
            if doc_profile.get("file_type") == "DOCX":
                stat_rows.append(["Paragraph Count", f"{doc_profile.get('paragraph_count', 'N/A')}"])

        if text_statistics:
            stat_rows.extend([
                ["Total Characters", f"{text_statistics.get('total_characters', 0):,}"],
                ["Non-Whitespace Characters", f"{text_statistics.get('non_whitespace_characters', 0):,}"],
                ["Total Words", f"{text_statistics.get('total_words', 0):,}"],
                ["Total Sentences", f"{text_statistics.get('total_sentences', 0):,}"],
                ["Total Paragraphs", f"{text_statistics.get('total_paragraphs', 0):,}"],
                ["Average Word Length", f"{text_statistics.get('average_word_length', 0.0):.1f} characters"],
                ["Average Sentence Length", f"{text_statistics.get('average_sentence_length', 0.0):.1f} words"],
            ])

        story.append(
            _create_standard_table(
                ["Metric Dimension", "Value"],
                stat_rows,
                [200.0, 304.0],
                styles,
                header_color="#1E293B",
            )
        )
        story.append(Spacer(1, 10))

    # 2. Document Quality Assessment
    if quality_findings:
        story.append(Paragraph("2. Document Quality & Hygiene Assessment", styles["SectionHeading"]))
        q_rows = []
        for q in quality_findings:
            q_rows.append([
                q.get("Issue", "—"),
                q.get("Severity", "Low"),
                q.get("Evidence", "—"),
                q.get("Suggested Action", "—"),
            ])
        story.append(
            _create_standard_table(
                ["Quality Finding", "Severity", "Observed Evidence", "Suggested Action"],
                q_rows,
                [110.0, 54.0, 180.0, 160.0],
                styles,
                header_color="#DC2626",
            )
        )
        story.append(Spacer(1, 10))

    # Applied Cleaning Summary (if any)
    if applied_cleaning and applied_cleaning.get("selected_operations"):
        story.append(Paragraph("Applied Text Sanitization Operations", styles["SubSectionHeading"]))
        ops = applied_cleaning.get("selected_operations", [])
        for op in ops:
            story.append(Paragraph(f"• {op}", styles["Body"]))
        story.append(Spacer(1, 8))

    # 3. Meaningful Keyword Frequency Analysis
    if keywords:
        story.append(Paragraph("3. Meaningful Keyword Frequency (Stop-Words Excluded)", styles["SectionHeading"]))
        kw_rows = []
        for idx, k in enumerate(keywords[:12], 1):
            kw_rows.append([
                str(idx),
                k.get("keyword", "—"),
                f"{k.get('frequency', 0):,}",
                f"{k.get('percentage', 0.0):.1f}%",
            ])
        story.append(
            _create_standard_table(
                ["Rank", "Keyword", "Occurrences", "Vocabulary Share (%)"],
                kw_rows,
                [50.0, 170.0, 130.0, 154.0],
                styles,
                header_color="#2563EB",
            )
        )
        story.append(Spacer(1, 10))

    # 4. Rule-Based Sentiment Analysis
    if sentiment:
        story.append(Paragraph("4. Rule-Based Sentiment Classification", styles["SectionHeading"]))
        overall_tone = sentiment.get("overall_sentiment", "Neutral")
        score = sentiment.get("score", 0)
        sent_counts = sentiment.get("sentence_counts", {})
        sent_pcts = sentiment.get("sentence_percentages", {})

        sent_summary = (
            f"<b>Overall Document Sentiment:</b> {overall_tone} &nbsp;|&nbsp; "
            f"<b>Net Keyword Score:</b> {score:+d} &nbsp;|&nbsp; "
            f"Classified {sentiment.get('total_sentences', 0)} sentence(s)."
        )
        story.append(Paragraph(sent_summary, styles["BodyBold"]))
        story.append(Spacer(1, 4))

        sent_rows = [
            ["Positive Sentences", f"{sent_counts.get('Positive', 0)} ({sent_pcts.get('Positive', 0.0):.1f}%)", "Sentence pos_score > neg_score"],
            ["Neutral Sentences", f"{sent_counts.get('Neutral', 0)} ({sent_pcts.get('Neutral', 0.0):.1f}%)", "Balanced or zero sentiment words"],
            ["Negative Sentences", f"{sent_counts.get('Negative', 0)} ({sent_pcts.get('Negative', 0.0):.1f}%)", "Sentence neg_score > pos_score"],
        ]
        story.append(
            _create_standard_table(
                ["Sentiment Classification", "Sentence Count & Share", "Classification Criterion"],
                sent_rows,
                [150.0, 160.0, 194.0],
                styles,
                header_color="#059669",
            )
        )
        story.append(Spacer(1, 6))

        pos_kws = sentiment.get("positive_keywords_found", [])
        neg_kws = sentiment.get("negative_keywords_found", [])
        if pos_kws or neg_kws:
            pos_str = ", ".join(f"{item['word']} ({item['count']})" for item in pos_kws[:6]) or "None"
            neg_str = ", ".join(f"{item['word']} ({item['count']})" for item in neg_kws[:6]) or "None"
            story.append(Paragraph(f"<b>Top Positive Triggers:</b> {pos_str}", styles["Body"]))
            story.append(Paragraph(f"<b>Top Negative Triggers:</b> {neg_str}", styles["Body"]))
            story.append(Spacer(1, 4))

        story.append(Spacer(1, 8))

    # 5. Domain Topic Identification
    if topics:
        story.append(Paragraph("5. Domain Topic Identification", styles["SectionHeading"]))
        top_rows = []
        for t in topics[:7]:
            top_kw_str = ", ".join(f"{k['keyword']} ({k['count']})" for k in t.get("matched_keywords", [])[:4])
            top_rows.append([
                t.get("topic", "—"),
                str(t.get("match_count", 0)),
                str(t.get("sentence_count", 0)),
                top_kw_str or "—",
            ])
        story.append(
            _create_standard_table(
                ["Domain Topic", "Keyword Matches", "Sentences", "Key Trigger Terms"],
                top_rows,
                [140.0, 90.0, 74.0, 200.0],
                styles,
                header_color="#7C3AED",
            )
        )
        story.append(Spacer(1, 10))

    # 6. Prioritized Rule-Based Insights
    if insights:
        story.append(Paragraph("6. Prioritized Actionable Insights", styles["SectionHeading"]))
        ins_rows = []
        for ins in insights[:8]:
            prio = ins.get("priority", "Medium")
            ins_rows.append([
                prio,
                ins.get("title", "—"),
                ins.get("category", "Feedback"),
                f"{ins.get('explanation', '')} <i>[Evidence: {ins.get('evidence', 'N/A')}]</i>",
            ])
        story.append(
            _create_standard_table(
                ["Priority", "Insight Title", "Category", "Findings & Empirical Evidence"],
                ins_rows,
                [54.0, 120.0, 80.0, 250.0],
                styles,
                header_color="#D97706",
            )
        )
        story.append(Spacer(1, 10))

    # 7. Methodological Transparency & Limitations Note
    story.append(Paragraph("7. Methodological Scope & Explainability Note", styles["SectionHeading"]))
    disclaimer = (
        "<b>Rule-Based Analytic Limitations:</b> All document metrics and sentiment classifications in this report "
        "were produced using transparent, deterministic lexicon matching and structural tokenization. "
        "The system does not deploy black-box machine learning models or external cloud APIs. "
        "Keyword matching cannot resolve conversational sarcasm, irony, or complex double-negation structures "
        "(e.g., 'not great' or 'hardly defective'). All processing was executed entirely in-memory on the local workstation."
    )
    story.append(Paragraph(disclaimer, styles["Notice"]))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
