"""Rule-based document quality checks and non-destructive text cleaning."""

from collections import Counter
import re
from typing import Any, MutableMapping


SHORT_TEXT_THRESHOLD = 40
NON_PRINTABLE_RATIO_THRESHOLD = 0.02
DOCUMENT_CLEANING_STATE_KEYS = (
    "doc_clean_source_id",
    "doc_clean_original_text",
    "doc_clean_preview_text",
    "doc_cleaned_text",
    "doc_clean_trim",
    "doc_clean_normalize_endings",
    "doc_clean_collapse_blank_lines",
    "doc_clean_remove_repeated_lines",
    "doc_quality_findings",
)


def assess_document_quality(text: str) -> list[dict[str, str]]:
    """Return explainable, rule-based findings for extracted document text.

    The checks identify formatting and extraction signals only. They do not
    judge writing style, meaning, or whether a short document is incorrect.
    """
    text = text or ""
    findings: list[dict[str, str]] = []

    if not text.strip():
        findings.append(
            _finding(
                "Empty or whitespace-only content",
                "The extracted text contains 0 non-whitespace characters.",
                "High",
                "Review the original file. Image-only PDFs may require OCR, which is not part of this workflow.",
            )
        )
        return findings

    if text != text.lstrip():
        findings.append(
            _finding(
                "Leading whitespace",
                "The extracted text starts with whitespace characters.",
                "Low",
                "Use the optional trim operation if this whitespace is not meaningful.",
            )
        )

    if text != text.rstrip():
        findings.append(
            _finding(
                "Trailing whitespace",
                "The extracted text ends with whitespace characters.",
                "Low",
                "Use the optional trim operation if this whitespace is not meaningful.",
            )
        )

    normalized_endings = normalize_line_endings(text)
    blank_runs = re.findall(r"(?:\n[ \t]*){3,}", normalized_endings)
    if blank_runs:
        longest_run = max(run.count("\n") for run in blank_runs)
        findings.append(
            _finding(
                "Excessive blank lines",
                f"Found {len(blank_runs)} run(s) with at least 3 consecutive line breaks; the longest has {longest_run}.",
                "Low",
                "Use the optional blank-line collapse operation to retain one blank line between paragraphs.",
            )
        )

    repeated_lines = _repeated_entries(
        [line.strip() for line in normalized_endings.split("\n") if line.strip()]
    )
    if repeated_lines:
        findings.append(
            _finding(
                "Repeated lines",
                _repeat_evidence(repeated_lines, "line"),
                "Low",
                "Review the repeated lines. Remove them only with the optional repeated-line operation if the duplicates are unintended.",
            )
        )

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n[ \t]*\n+", normalized_endings)
        if paragraph.strip()
    ]
    repeated_paragraphs = _repeated_entries(paragraphs)
    if repeated_paragraphs:
        findings.append(
            _finding(
                "Repeated paragraphs",
                _repeat_evidence(repeated_paragraphs, "paragraph"),
                "Low",
                "Review the repeated paragraphs before changing the document. The repeated-line operation removes repeated lines, not whole paragraphs.",
            )
        )

    non_printable_count = sum(
        1 for character in text if not character.isprintable() and character not in "\r\n\t"
    )
    non_printable_ratio = non_printable_count / len(text)
    if (
        non_printable_count >= 3
        and non_printable_ratio >= NON_PRINTABLE_RATIO_THRESHOLD
    ):
        findings.append(
            _finding(
                "High proportion of non-printable characters",
                f"Found {non_printable_count} non-printable characters in {len(text)} characters ({non_printable_ratio:.1%}).",
                "Medium",
                "Review the source file and extraction result for encoding or extraction problems.",
            )
        )

    non_whitespace_count = len(re.sub(r"\s+", "", text))
    if non_whitespace_count < SHORT_TEXT_THRESHOLD:
        findings.append(
            _finding(
                "Very little text was extracted",
                f"Only {non_whitespace_count} non-whitespace characters were extracted (threshold: fewer than {SHORT_TEXT_THRESHOLD}). This alone does not prove an extraction failure.",
                "Info",
                "Compare the preview with the original document. A short document may be valid; image-only PDFs may need OCR.",
            )
        )

    return findings


def normalize_line_endings(text: str) -> str:
    """Convert Windows and classic Mac line endings to Unix-style line endings."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_document_text(
    text: str,
    *,
    trim_whitespace: bool = False,
    normalize_endings: bool = False,
    collapse_blank_lines: bool = False,
    remove_repeated_lines: bool = False,
) -> str:
    """Return cleaned text without changing the original extracted text."""
    cleaned_text = text or ""

    if trim_whitespace:
        cleaned_text = cleaned_text.strip()
    if normalize_endings:
        cleaned_text = normalize_line_endings(cleaned_text)
    if collapse_blank_lines:
        cleaned_text = _collapse_excessive_blank_lines(cleaned_text)
    if remove_repeated_lines:
        cleaned_text = _remove_repeated_nonempty_lines(cleaned_text)

    return cleaned_text


def build_cleaning_preview(text: str, **cleaning_options: bool) -> dict[str, Any]:
    """Create a before/after preview for the selected cleaning operations."""
    cleaned_text = clean_document_text(text, **cleaning_options)
    selected_operations = [
        label
        for option, label in {
            "trim_whitespace": "Trim leading and trailing whitespace",
            "normalize_endings": "Normalize line endings",
            "collapse_blank_lines": "Collapse excessive blank lines",
            "remove_repeated_lines": "Remove repeated lines",
        }.items()
        if cleaning_options.get(option, False)
    ]
    return {
        "original_text": text or "",
        "cleaned_text": cleaned_text,
        "changed": cleaned_text != (text or ""),
        "selected_operations": selected_operations,
    }


def initialize_document_cleaning_state(
    state: MutableMapping[str, Any], source_id: str, original_text: str
) -> bool:
    """Initialize document-only cleaning state and reset it for a new document.

    Returns True when a document change required stale preview/results to reset.
    """
    if state.get("doc_clean_source_id") == source_id:
        return False

    for key in DOCUMENT_CLEANING_STATE_KEYS:
        state.pop(key, None)

    state.update(
        {
            "doc_clean_source_id": source_id,
            "doc_clean_original_text": original_text,
            "doc_clean_preview_text": original_text,
            "doc_cleaned_text": None,
            "doc_clean_trim": False,
            "doc_clean_normalize_endings": False,
            "doc_clean_collapse_blank_lines": False,
            "doc_clean_remove_repeated_lines": False,
        }
    )
    return True


def reset_document_cleaning_state(state: MutableMapping[str, Any]) -> None:
    """Reset document cleaning results and options back to the extracted text."""
    source_id = state.get("doc_clean_source_id", "")
    original_text = state.get("doc_clean_original_text", "")
    for key in DOCUMENT_CLEANING_STATE_KEYS:
        state.pop(key, None)
    initialize_document_cleaning_state(state, source_id, original_text)


def _finding(issue: str, evidence: str, severity: str, suggested_action: str) -> dict[str, str]:
    return {
        "Issue": issue,
        "Evidence": evidence,
        "Severity": severity,
        "Suggested Action": suggested_action,
    }


def _repeated_entries(entries: list[str]) -> dict[str, int]:
    """Return exact normalized entries that occur more than once."""
    return {entry: count for entry, count in Counter(entries).items() if count > 1}


def _repeat_evidence(repeated_entries: dict[str, int], entry_type: str) -> str:
    """Build concise evidence without treating repeated wording as an error."""
    examples = list(repeated_entries.items())[:2]
    example_text = "; ".join(f"{value[:80]!r} ({count} times)" for value, count in examples)
    return (
        f"{len(repeated_entries)} distinct {entry_type}(s) occur more than once "
        f"when compared as complete normalized {entry_type}s. Examples: {example_text}"
    )


def _remove_repeated_nonempty_lines(text: str) -> str:
    """Keep the first non-empty line; preserve blank lines and first-line formatting."""
    seen_lines: set[str] = set()
    retained_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        line_content = line.rstrip("\r\n")
        normalized_line = line_content.strip()
        if normalized_line and normalized_line in seen_lines:
            continue
        if normalized_line:
            seen_lines.add(normalized_line)
        retained_lines.append(line)

    return "".join(retained_lines)


def _collapse_excessive_blank_lines(text: str) -> str:
    """Keep one blank line while retaining the document's current line-ending style."""
    line_break_pattern = r"(?:\r\n|\r|\n)[ \t]*"

    def replacement(match: re.Match[str]) -> str:
        matched_text = match.group(0)
        if "\r\n" in matched_text:
            line_ending = "\r\n"
        elif "\r" in matched_text:
            line_ending = "\r"
        else:
            line_ending = "\n"
        return line_ending * 2

    return re.sub(f"(?:{line_break_pattern}){{3,}}", replacement, text)
