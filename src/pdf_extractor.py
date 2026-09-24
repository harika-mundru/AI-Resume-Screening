"""
pdf_extractor.py

Extracts raw text from uploaded resume PDFs using PyMuPDF (fitz).

Design notes:
- Every PDF is opened and read independently. If one resume is corrupted,
  password-protected, or has no extractable text, this module reports that
  clearly through the ExtractionResult object instead of raising an
  exception that would crash the rest of the screening run.
- app.py is expected to loop over multiple uploaded files and call
  extract_text_from_pdf() for each one, collecting warnings for any file
  that failed so the user can see which resumes were skipped and why.
"""

from dataclasses import dataclass
from typing import Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class ExtractionResult:
    filename: str
    text: str
    success: bool
    page_count: int = 0
    warning: Optional[str] = None


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> ExtractionResult:
    """
    Extract text from a single PDF file given as raw bytes.

    Returns an ExtractionResult. success=False means this resume should be
    skipped by the caller, with `warning` explaining why, so the rest of
    the batch can keep processing.
    """
    if not PYMUPDF_AVAILABLE:
        return ExtractionResult(
            filename=filename, text="", success=False, page_count=0,
            warning=(
                "PyMuPDF (fitz) is not installed. Run "
                "'pip install PyMuPDF' and try again."
            ),
        )

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        return ExtractionResult(
            filename=filename, text="", success=False, page_count=0,
            warning=f"Could not open '{filename}' as a PDF ({exc}). File skipped.",
        )

    if doc.is_encrypted:
        # Try an empty-password unlock (common for "restricted but not
        # really password protected" PDFs); if that fails, skip the file.
        try:
            unlocked = doc.authenticate("")
        except Exception:
            unlocked = False
        if not unlocked:
            doc.close()
            return ExtractionResult(
                filename=filename, text="", success=False, page_count=0,
                warning=f"'{filename}' is password-protected and could not be opened. File skipped.",
            )

    page_count = doc.page_count
    text_parts = []
    for page_index in range(page_count):
        try:
            page = doc.load_page(page_index)
            text_parts.append(page.get_text("text") or "")
        except Exception:
            # Skip a single bad page rather than the whole document.
            continue

    doc.close()
    raw_text = "\n".join(text_parts).strip()

    if not raw_text:
        return ExtractionResult(
            filename=filename, text="", success=False, page_count=page_count,
            warning=(
                f"No extractable text was found in '{filename}'. It may be a "
                "scanned/image-only PDF, which this tool cannot read without OCR."
            ),
        )

    return ExtractionResult(
        filename=filename, text=raw_text, success=True, page_count=page_count, warning=None,
    )
