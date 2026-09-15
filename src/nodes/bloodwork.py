from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

import pymupdf
from pydantic import ValidationError

from src.config import (
    EXTRACTION_RETRIES,
    MAX_LAB_PDF_PAGES,
    MIN_PDF_TEXT_CHARS,
    OCR_DPI,
    UNMAPPED,
    settings,
)
from src.llm import build_llm
from src.schemas import BloodPanel, ExtractionCheck, PipelineState
from src.validation import normalize_blood_panel

EXTRACTION_SYSTEM = f"""You transcribe laboratory reports into structured data.

Rules:
- Transcribe only what is printed. Never infer, convert, correct, or complete a value.
- Copy analyte names and units exactly as they appear, including capitalisation.
- Set canonical_name to "{UNMAPPED}"; canonical names are assigned downstream, not by you.
- Leave reference_range_low or reference_range_high null when the report does not print one.
- Put any line that looks like a result but cannot be transcribed confidently into
  unparsed_fields, verbatim. A surfaced unparsed line is always better than a guess.
- Do not comment on whether a value is normal, high, low, or concerning.
"""

RETRY_SUFFIX = """

Your previous response did not satisfy the schema. The validation error was:
{error}

Return a corrected response. Do not invent values to fill required fields."""


class PdfText(NamedTuple):
    text: str
    used_ocr: bool
    unreadable_pages: int


def extract_pdf_text(path: Path) -> PdfText:
    if not path.exists():
        raise FileNotFoundError(f"No lab report at {path}.")

    with pymupdf.open(path) as document:
        if document.page_count > MAX_LAB_PDF_PAGES:
            raise ValueError(
                f"{path.name} has {document.page_count} pages; the limit is {MAX_LAB_PDF_PAGES}."
            )
        pages: list[str] = []
        used_ocr = False
        unreadable_pages = 0
        for page in document:
            text = page.get_text("text")
            if len(text.strip()) >= MIN_PDF_TEXT_CHARS:
                pages.append(text)
                continue
            ocr_text = _ocr_page(page)
            if ocr_text.strip():
                used_ocr = True
                pages.append(ocr_text)
                continue
            if not text.strip():
                unreadable_pages += 1
            pages.append(text)
    return PdfText("\n".join(pages), used_ocr, unreadable_pages)


def _ocr_page(page: Any) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    pixmap = page.get_pixmap(dpi=OCR_DPI)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        return str(pytesseract.image_to_string(image))
    except pytesseract.TesseractNotFoundError:
        return ""


def transcribe_blood_panel(document_text: str) -> BloodPanel:
    llm = build_llm()
    prompt = f"Transcribe every laboratory result in this report.\n\n{document_text}"
    system = EXTRACTION_SYSTEM
    retries_left = EXTRACTION_RETRIES

    while True:
        try:
            panel = llm.generate(system=system, user=prompt, schema=BloodPanel)
        except ValidationError as exc:
            if retries_left == 0:
                raise
            retries_left -= 1
            system = EXTRACTION_SYSTEM + RETRY_SUFFIX.format(error=exc)
            continue
        return panel


def values_not_in_document(panel: BloodPanel, document_text: str) -> list[str]:
    printed_numbers: set[float] = set()
    for number in re.findall(r"\d+(?:[.,]\d+)*", document_text):
        # "1,000" is a thousands separator on some reports and "5,4" a decimal on others.
        printed_numbers.add(float(number.replace(",", "")))
        if number.count(",") == 1 and "." not in number:
            printed_numbers.add(float(number.replace(",", ".")))
    return [
        f"{analyte.reported_name}: {analyte.value}"
        for analyte in panel.analytes
        if abs(analyte.value) not in printed_numbers
    ]


def bloodwork_node(state: PipelineState) -> dict[str, Any]:
    pdf_path = state.get("lab_pdf_path")
    if not pdf_path:
        return {}

    try:
        pdf = extract_pdf_text(Path(pdf_path))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return {"warnings": [f"Lab report not read: {exc}"]}

    if not pdf.text.strip():
        return {"warnings": ["The lab report contained no readable text, even after OCR."]}

    warnings: list[str] = []
    if pdf.used_ocr:
        warnings.append("Values were read by OCR and may contain transcription errors.")
    if pdf.unreadable_pages:
        warnings.append(
            f"{pdf.unreadable_pages} lab report page(s) had no text and could not be OCR'd; "
            "any results on them are missing."
        )

    try:
        transcribed = transcribe_blood_panel(pdf.text)
    except ValidationError as exc:
        return {"warnings": [f"Lab extraction failed validation: {exc.error_count()} field(s)."]}
    except Exception as exc:  # noqa: BLE001
        # Provider SDKs raise their own timeout, auth, and rate-limit types. Any of them
        # should skip the lab branch, not abort a run that can still report on the intake.
        return {"warnings": [f"Lab extraction failed: {type(exc).__name__}: {exc}"]}

    panel = normalize_blood_panel(transcribed).model_copy(update={"used_ocr": pdf.used_ocr})
    check = ExtractionCheck(
        analytes_extracted=len(panel.analytes),
        analytes_mapped=sum(analyte.canonical_name != UNMAPPED for analyte in panel.analytes),
        values_not_in_document=values_not_in_document(transcribed, pdf.text),
        unparsed_lines=len(panel.unparsed_fields),
        used_ocr=pdf.used_ocr,
        unreadable_pages=pdf.unreadable_pages,
    )
    return {"blood_panel": panel, "extraction_check": check, "warnings": warnings}
