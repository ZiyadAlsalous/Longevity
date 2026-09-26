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
from src.schemas import BloodAnalyte, BloodPanel, ExtractionCheck, PipelineState
from src.validation import normalize_blood_panel

EXTRACTION_SYSTEM = """You transcribe laboratory reports into structured data.

Rules:
- Transcribe only what is printed. Never infer, convert, correct, or complete a value.
- Each result is one row: test name, numeric result, unit, then the reference range. A letter
  next to the result such as H or L is a flag, not part of the value.
- Copy analyte names and units exactly as they appear, including capitalisation.
- Transcribe every numeric result, including ones reported both as a percentage and as an
  absolute count, as separate analytes. A reference range belongs only to the result it is
  printed beside; when a row has two results and one range, the other result has no range.
- Fill reference_range_low and reference_range_high only from a printed normal range, such as
  "a - b", "up to b" or "above a". Risk tiers or categories are not a normal range.
- Ignore billing, prices, receipts, patient details and dates that are not results. A page
  with no laboratory results returns an empty analytes list.
- If a value cannot be read, leave that analyte out. Never write 0 as a placeholder.
- Copy any comment or recommendation the laboratory wrote about the results into lab_comments.
  Signatures, job titles and headings are not comments.
- lab_name is the name of the laboratory, not a doctor.
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
    pages: list[str]
    used_ocr: bool
    unreadable_pages: int


def extract_pdf_text(path: Path) -> PdfText:
    """Extract text from the lab PDF, using OCR for scanned pages."""
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
            # sort=True keeps each table row on one line, in reading order.
            text = page.get_text("text", sort=True)
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
    return PdfText("\n".join(pages), pages, used_ocr, unreadable_pages)


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


def transcribe_blood_panel(pages: list[str]) -> BloodPanel:
    """Transcribe each page on its own, then merge; one page at a time keeps the model accurate."""
    read = [_transcribe_page(page) for page in pages if page.strip()]
    panels = [panel for panel in read if panel.contains_lab_results]
    if not panels:
        return BloodPanel()
    seen: set[tuple[str, float, str]] = set()
    analytes: list[BloodAnalyte] = []
    for analyte in (a for panel in panels for a in panel.analytes):
        key = (analyte.reported_name.lower(), analyte.value, analyte.unit.lower())
        if key not in seen:
            seen.add(key)
            analytes.append(analyte)
    return panels[0].model_copy(
        update={
            "analytes": analytes,
            "lab_name": next((p.lab_name for p in panels if p.lab_name), None),
            "collected_on": next((p.collected_on for p in panels if p.collected_on), None),
            "unparsed_fields": [line for p in panels for line in p.unparsed_fields],
            "lab_comments": list(dict.fromkeys(c for p in panels for c in p.lab_comments)),
        }
    )


def _transcribe_page(page_text: str) -> BloodPanel:
    llm = build_llm()
    prompt = f"Transcribe every laboratory result on this page of a lab report.\n\n{page_text}"
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


def _numbers_at(text: str) -> list[tuple[int, float]]:
    """Every number printed in a piece of text, with where it starts."""
    found: list[tuple[int, float]] = []
    for match in re.finditer(r"\d+(?:[.,]\d+)*", text):
        number = match.group()
        # Accept both "1,000" and "5,4" number styles.
        found.append((match.start(), float(number.replace(",", ""))))
        if number.count(",") == 1 and "." not in number:
            found.append((match.start(), float(number.replace(",", "."))))
    return found


def printed_numbers(text: str) -> set[float]:
    """Every number printed in a piece of text."""
    return {value for _, value in _numbers_at(text)}


def row_of(analyte: BloodAnalyte, lines: list[str]) -> int | None:
    """The line where the value is printed together with the start of the test's name."""
    words = re.findall(r"[a-z0-9]+", analyte.reported_name.lower())
    for index, line in enumerate(lines):
        if (
            words
            and words[0] in re.findall(r"[a-z0-9]+", line.lower())
            and abs(analyte.value) in printed_numbers(line)
        ):
            return index
    return None


def check_ranges(analytes: list[BloodAnalyte], lines: list[str]) -> list[BloodAnalyte]:
    """Keep a reference range only if it is printed on the result's own line, beside it."""
    rows = [row_of(a, lines) for a in analytes]
    checked: list[BloodAnalyte] = []
    for analyte, row in zip(analytes, rows, strict=True):
        bounds = [
            b for b in (analyte.reference_range_low, analyte.reference_range_high) if b is not None
        ]
        on_line = row is not None and _range_printed_beside(analyte, lines[row])
        checked.append(analyte if on_line else _without_range(analyte))

    # Two results on one line with one printed range: the range belongs to the result printed
    # just before it, whichever result the model gave it to.
    for row in {r for r in rows if r is not None}:
        members = [i for i, r in enumerate(rows) if r == row]
        if len(members) < 2:
            continue
        numbers = _numbers_at(lines[row])
        value_at = {
            j: min((pos for pos, v in numbers if v == abs(checked[j].value)), default=-1)
            for j in members
        }
        for i in members:
            bounds = _bounds(checked[i])
            if not bounds:
                continue
            range_at = max((pos for pos, v in numbers if v == bounds[0]), default=-1)
            before = [j for j in members if value_at[j] < range_at]
            owner = max(before, key=lambda j: value_at[j], default=i)
            if owner != i:
                if not _bounds(checked[owner]):
                    checked[owner] = checked[owner].model_copy(
                        update={
                            "reference_range_low": checked[i].reference_range_low,
                            "reference_range_high": checked[i].reference_range_high,
                        }
                    )
                checked[i] = _without_range(checked[i])
    return checked


def _range_printed_beside(analyte: BloodAnalyte, line: str) -> bool:
    """Each bound is printed on the line, and apart from the value: a number printed once cannot
    be both the result and its own range."""
    numbers = [value for _, value in _numbers_at(line)]
    for bound in _bounds(analyte):
        needed = 2 if abs(bound) == abs(analyte.value) else 1
        if numbers.count(abs(bound)) < needed:
            return False
    return True


def check_units(analytes: list[BloodAnalyte], lines: list[str]) -> list[BloodAnalyte]:
    """Keep a unit only if it is printed on the result's own line."""
    checked: list[BloodAnalyte] = []
    for analyte in analytes:
        row = row_of(analyte, lines)
        unit = analyte.unit.strip()
        words = re.findall(r"[a-z0-9]+", unit.lower())
        printed = row is not None and (
            words[0] in re.findall(r"[a-z0-9]+", lines[row].lower())
            if words
            else unit in lines[row]
        )
        checked.append(analyte if not unit or printed else analyte.model_copy(update={"unit": ""}))
    return checked


def _bounds(analyte: BloodAnalyte) -> list[float]:
    return [b for b in (analyte.reference_range_low, analyte.reference_range_high) if b is not None]


def _without_range(analyte: BloodAnalyte) -> BloodAnalyte:
    return analyte.model_copy(update={"reference_range_low": None, "reference_range_high": None})


def printed_in(text: str, document_text: str) -> bool:
    """Is this text printed in the document, ignoring spacing and case?"""
    return bool(text.strip()) and _squash(text) in _squash(document_text)


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def bloodwork_node(state: PipelineState) -> dict[str, Any]:
    """Read, transcribe and normalise the lab report."""
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
        transcribed = transcribe_blood_panel(pdf.pages)
    except ValidationError as exc:
        return {"warnings": [f"Lab extraction failed validation: {exc.error_count()} field(s)."]}
    except Exception as exc:  # noqa: BLE001
        # Any model failure skips the lab branch instead of failing the run.
        return {"warnings": [f"Lab extraction failed: {type(exc).__name__}: {exc}"]}

    # A value that is not printed next to its own test name is never used, and neither is a
    # number with no unit and no range, because it cannot be interpreted.
    lines = pdf.text.splitlines()
    on_row = check_ranges([a for a in transcribed.analytes if row_of(a, lines) is not None], lines)
    verified = [a for a in check_units(on_row, lines) if a.unit.strip() or _bounds(a)]
    kept = {(a.reported_name, a.value) for a in verified}
    rejected = [a for a in transcribed.analytes if (a.reported_name, a.value) not in kept]
    transcribed = transcribed.model_copy(
        update={
            "analytes": verified,
            # Comments and the lab name are shown to the user, so they must be printed too.
            "lab_comments": [c for c in transcribed.lab_comments if printed_in(c, pdf.text)],
            "lab_name": transcribed.lab_name
            if transcribed.lab_name and printed_in(transcribed.lab_name, pdf.text)
            else None,
            "unparsed_fields": [
                *transcribed.unparsed_fields,
                *(
                    f"{a.reported_name}: {a.value} could not be verified on the report; not used."
                    for a in rejected
                ),
            ],
        }
    )
    panel = normalize_blood_panel(transcribed)
    check = ExtractionCheck(
        analytes_extracted=len(panel.analytes),
        analytes_mapped=sum(analyte.canonical_name != UNMAPPED for analyte in panel.analytes),
        values_not_in_document=[f"{a.reported_name}: {a.value}" for a in rejected],
        unparsed_lines=len(panel.unparsed_fields),
        used_ocr=pdf.used_ocr,
        unreadable_pages=pdf.unreadable_pages,
    )
    return {"blood_panel": panel, "extraction_check": check, "warnings": warnings}
