"""Unstructured Document Processing and Profiling Engine for AutoDataIQ.

Phase 15: Provides robust, isolated text extraction and document profiling
for TXT, PDF, and DOCX files without modifying or coupling with structured CSV/Excel state.
"""

import io
import os
from pathlib import Path
import re
from typing import Any

try:
    import pypdf
    from pypdf.errors import PdfReadError
except ImportError:
    pypdf = None
    PdfReadError = Exception

try:
    import docx
    from docx.opc.exceptions import PackageNotFoundError
except ImportError:
    docx = None
    PackageNotFoundError = Exception


class DocumentProcessingError(Exception):
    """Custom exception raised for unrecoverable document parsing and extraction errors."""
    pass


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def detect_document_type(filename: str) -> str:
    """Identify the document format from its filename extension.

    Parameters:
    - filename: The name or path of the uploaded file.

    Returns:
    One of 'TXT', 'PDF', or 'DOCX'.

    Raises:
    DocumentProcessingError if the extension is unsupported.
    """
    if not filename or "." not in filename:
        raise DocumentProcessingError(
            "Unsupported file type. AutoDataIQ Document Processing supports TXT, PDF, and DOCX files."
        )

    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return "TXT"
    elif ext == ".pdf":
        return "PDF"
    elif ext == ".docx":
        return "DOCX"
    else:
        raise DocumentProcessingError(
            f"Unsupported file extension '{ext}'. AutoDataIQ Document Processing supports TXT, PDF, and DOCX files."
        )


def format_file_size(size_bytes: int) -> str:
    """Format raw byte counts into human-readable strings."""
    if size_bytes < 0:
        return "0 B"
    if size_bytes < 1024:
        return f"{size_bytes:,} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB ({size_bytes:,} bytes)"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB ({size_bytes:,} bytes)"


def _read_file_bytes(file_input: Any) -> bytes:
    """Safely extract bytes from a file-like object, Streamlit UploadedFile, or file path."""
    if isinstance(file_input, (str, Path)):
        with open(file_input, "rb") as f:
            return f.read()
    elif hasattr(file_input, "read"):
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        data = file_input.read()
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        return data if isinstance(data, bytes) else str(data).encode("utf-8")
    elif isinstance(file_input, bytes):
        return file_input
    else:
        raise DocumentProcessingError("Invalid file input: unable to read binary document data.")


def extract_document_text(file_input: Any, filename: str) -> dict[str, Any]:
    """Extract raw text and structural metadata from a TXT, PDF, or DOCX document.

    Strictly isolated: Does not mutate any DataFrame or structured session-state.

    Parameters:
    - file_input: File path, bytes, or Streamlit UploadedFile object.
    - filename: Name of the file including extension.

    Returns:
    Dictionary containing:
    - filename: Original file name.
    - file_type: 'TXT', 'PDF', or 'DOCX'.
    - file_size_bytes: Raw file size in bytes.
    - file_size_formatted: Human-readable file size.
    - status: 'SUCCESS', 'NO_TEXT', or 'EMPTY'.
    - text: Full extracted text content.
    - page_count: Integer page count for PDF, or None for TXT and DOCX.
    - paragraph_count: Integer paragraph count for DOCX, or None for TXT and PDF.
    - is_scanned: Boolean flag for PDFs that contain no selectable text.
    - message: Descriptive status message or diagnostic notes.
    """
    file_type = detect_document_type(filename)
    raw_bytes = _read_file_bytes(file_input)
    size_bytes = len(raw_bytes)

    # 1. Handle empty file check
    if size_bytes == 0:
        raise DocumentProcessingError(
            f"The uploaded document '{filename}' is completely empty (0 bytes). "
            "Please upload a valid document containing content."
        )

    # 2. Format-specific extraction
    if file_type == "TXT":
        return _extract_txt(raw_bytes, filename, size_bytes)
    elif file_type == "PDF":
        return _extract_pdf(raw_bytes, filename, size_bytes)
    elif file_type == "DOCX":
        return _extract_docx(raw_bytes, filename, size_bytes)
    else:
        raise DocumentProcessingError(f"Unsupported file type '{file_type}'.")


def _extract_txt(raw_bytes: bytes, filename: str, size_bytes: int) -> dict[str, Any]:
    """Extract text from TXT files with multi-encoding fallback."""
    encodings_to_try = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    decoded_text = None
    used_encoding = None

    for enc in encodings_to_try:
        try:
            decoded_text = raw_bytes.decode(enc)
            used_encoding = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if decoded_text is None:
        raise DocumentProcessingError(
            f"Failed to decode text file '{filename}'. The file uses an unrecognized or unsupported text encoding."
        )

    is_empty_text = len(decoded_text.strip()) == 0
    status = "NO_TEXT" if is_empty_text else "SUCCESS"
    message = (
        "Text file decoded successfully."
        if not is_empty_text
        else "The text file was read successfully but contains only whitespace."
    )

    return {
        "filename": filename,
        "file_type": "TXT",
        "encoding_used": used_encoding,
        "file_size_bytes": size_bytes,
        "file_size_formatted": format_file_size(size_bytes),
        "status": status,
        "text": decoded_text,
        "page_count": None,  # Explicitly not invented for TXT
        "paragraph_count": None,
        "is_scanned": False,
        "message": message,
    }


def _extract_pdf(raw_bytes: bytes, filename: str, size_bytes: int) -> dict[str, Any]:
    """Extract selectable text and page count from text-based PDF documents."""
    if pypdf is None:
        raise DocumentProcessingError(
            "The 'pypdf' library is not available in the Python environment. "
            "Please install it using 'pip install pypdf'."
        )

    try:
        pdf_stream = io.BytesIO(raw_bytes)
        reader = pypdf.PdfReader(pdf_stream)
    except Exception as e:
        raise DocumentProcessingError(
            f"Unable to read PDF file '{filename}'. The file may be corrupted, password-protected, "
            f"or not a valid PDF document. Diagnostic detail: {e}"
        )

    if reader.is_encrypted:
        try:
            # Attempt empty-password decrypt
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise DocumentProcessingError(
                    f"The PDF file '{filename}' is encrypted and password-protected. "
                    "Please provide an unencrypted PDF document."
                )
        except Exception:
            raise DocumentProcessingError(
                f"The PDF file '{filename}' is password-protected and cannot be extracted."
            )

    page_count = len(reader.pages)
    if page_count == 0:
        return {
            "filename": filename,
            "file_type": "PDF",
            "file_size_bytes": size_bytes,
            "file_size_formatted": format_file_size(size_bytes),
            "status": "NO_TEXT",
            "text": "",
            "page_count": 0,
            "paragraph_count": None,
            "is_scanned": False,
            "message": "The PDF document contains 0 pages.",
        }

    extracted_pages: list[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            p_text = page.extract_text() or ""
            extracted_pages.append(p_text.strip())
        except Exception as pe:
            extracted_pages.append(f"[Page {idx + 1} extraction error: {pe}]")

    full_text = "\n\n".join([p for p in extracted_pages if p])
    is_empty_text = len(full_text.strip()) == 0

    if is_empty_text:
        status = "NO_TEXT"
        is_scanned = True
        message = (
            "This PDF document contains 0 selectable characters. It appears to be scanned or contains only "
            "raster images. Note that Optical Character Recognition (OCR) is not included in this phase."
        )
    else:
        status = "SUCCESS"
        is_scanned = False
        message = f"Selectable text successfully extracted from {page_count} PDF page(s)."

    return {
        "filename": filename,
        "file_type": "PDF",
        "file_size_bytes": size_bytes,
        "file_size_formatted": format_file_size(size_bytes),
        "status": status,
        "text": full_text,
        "page_count": page_count,
        "paragraph_count": None,
        "is_scanned": is_scanned,
        "message": message,
    }


def _extract_docx(raw_bytes: bytes, filename: str, size_bytes: int) -> dict[str, Any]:
    """Extract paragraphs and text from Microsoft Word (DOCX) documents."""
    if docx is None:
        raise DocumentProcessingError(
            "The 'python-docx' library is not available in the Python environment. "
            "Please install it using 'pip install python-docx'."
        )

    try:
        docx_stream = io.BytesIO(raw_bytes)
        doc = docx.Document(docx_stream)
    except Exception as e:
        raise DocumentProcessingError(
            f"Unable to read DOCX file '{filename}'. The file may be corrupted, an older binary .doc format, "
            f"or not a valid Word package. Diagnostic detail: {e}"
        )

    # Extract paragraphs
    paragraphs = [p.text for p in doc.paragraphs]
    paragraph_count = len(doc.paragraphs)

    # Also capture table cell text if present
    table_texts: list[str] = []
    if doc.tables:
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    table_texts.append(" | ".join(row_cells))

    all_blocks = [p for p in paragraphs if p.strip()] + table_texts
    full_text = "\n\n".join(all_blocks)
    is_empty_text = len(full_text.strip()) == 0

    if is_empty_text:
        status = "NO_TEXT"
        message = "The Word document was read successfully but contains no textual paragraphs or table content."
    else:
        status = "SUCCESS"
        message = f"Text successfully extracted from {paragraph_count} paragraph(s) in DOCX file."

    return {
        "filename": filename,
        "file_type": "DOCX",
        "file_size_bytes": size_bytes,
        "file_size_formatted": format_file_size(size_bytes),
        "status": status,
        "text": full_text,
        "page_count": None,  # Word documents do not have static page counts without rendering
        "paragraph_count": paragraph_count,
        "is_scanned": False,
        "message": message,
    }


def profile_document(extraction_result: dict[str, Any]) -> dict[str, Any]:
    """Compute detailed document profiling statistics from an extraction result.

    Parameters:
    - extraction_result: Output dictionary from extract_document_text().

    Returns:
    Dictionary containing document profiling metrics:
    - filename, file_type, file_size_bytes, file_size_formatted
    - character_count (total)
    - non_whitespace_character_count
    - word_count
    - line_count
    - paragraph_count (DOCX only, or None)
    - page_count (PDF only, or None)
    - average_word_length
    - preview_text (first 1,500 characters)
    - extraction_status ('SUCCESS', 'NO_TEXT', etc.)
    - status_message
    - is_scanned_notice (True if scanned PDF detected)
    """
    text = extraction_result.get("text", "")
    char_count = len(text)
    clean_chars = len(re.sub(r"\s+", "", text))
    words = text.split()
    word_count = len(words)
    lines = text.splitlines() if text else []
    line_count = len([l for l in lines if l.strip()])

    avg_word_length = round(clean_chars / word_count, 1) if word_count > 0 else 0.0

    preview_limit = 1500
    preview_text = text[:preview_limit]
    if len(text) > preview_limit:
        preview_text += f"\n\n... [Truncated: showing first {preview_limit:,} of {char_count:,} characters]"

    return {
        "filename": extraction_result.get("filename", "Unknown Document"),
        "file_type": extraction_result.get("file_type", "Unknown"),
        "file_size_bytes": extraction_result.get("file_size_bytes", 0),
        "file_size_formatted": extraction_result.get("file_size_formatted", "0 B"),
        "character_count": char_count,
        "non_whitespace_characters": clean_chars,
        "word_count": word_count,
        "line_count": line_count,
        "page_count": extraction_result.get("page_count"),
        "paragraph_count": extraction_result.get("paragraph_count"),
        "average_word_length": avg_word_length,
        "status": extraction_result.get("status", "SUCCESS"),
        "is_scanned": extraction_result.get("is_scanned", False),
        "message": extraction_result.get("message", ""),
        "preview_text": preview_text,
    }
