from __future__ import annotations

import re
from typing import Any

from src.config import UNSAFE_OUTPUT_PATTERNS
from src.schemas import AgingReport, PipelineState, RunEvaluation, SafetyCheck


def evaluation_node(state: PipelineState) -> dict[str, Any]:
    """Score this run's output and record any failures."""
    safety = SafetyCheck(unsafe_phrases=find_unsafe_phrases(state["report"]))
    extraction = state.get("extraction_check")
    vision = state.get("vision_check")
    grounding = state["grounding_check"]

    failures: list[str] = []
    if safety.unsafe_phrases:
        failures.append(f"Report contains dosing or diagnostic language: {safety.unsafe_phrases}.")
    # Values that could not be verified on the report are rejected before use, so they are
    # recorded in the extraction check rather than failing the run.
    if grounding.factors_dropped_for_invented_biomarkers:
        failures.append(
            "The model cited biomarkers that were not supplied: "
            f"{grounding.factors_dropped_for_invented_biomarkers}."
        )

    evaluation = RunEvaluation(
        extraction=extraction,
        vision=vision,
        grounding=grounding,
        safety=safety,
        passed=not failures,
        failures=failures,
    )
    return {"evaluation": evaluation}


def find_unsafe_phrases(report: AgingReport) -> list[str]:
    """Find dose, medication or diagnosis language in the report."""
    body = _report_text(report)
    return [
        match.group(0)
        for pattern in UNSAFE_OUTPUT_PATTERNS
        for match in re.finditer(pattern, body, flags=re.IGNORECASE)
    ]


def unsafe_sentences(report: AgingReport) -> list[str]:
    """The sentences that contain dose, medication or diagnosis language."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", _report_text(report))
    return [sentence.strip() for sentence in sentences if _is_unsafe(sentence)]


def remove_unsafe_sentences(report: AgingReport) -> tuple[AgingReport, list[str]]:
    """Drop every sentence that still has dose, medication or diagnosis language."""
    removed: list[str] = []

    def clean(text: str) -> str:
        # Find the unsafe wording in the whole text first, then drop every sentence it touches,
        # so an abbreviation such as "Dr." splitting a sentence cannot hide a match.
        hits = [
            match.span()
            for pattern in UNSAFE_OUTPUT_PATTERNS
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        ]
        kept, position = [], 0
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            start = text.find(sentence, position)
            end = position = start + len(sentence)
            if any(hit_start < end and hit_end > start for hit_start, hit_end in hits):
                removed.append(sentence.strip())
            else:
                kept.append(sentence)
        return " ".join(kept).strip()

    # A factor whose title is itself unsafe goes, with its recommendations.
    unsafe_titles = {f.title for f in report.factors if _is_unsafe(f.title)}
    removed += sorted(unsafe_titles)
    factors = [
        f.model_copy(update={"explanation": clean(f.explanation)})
        for f in report.factors
        if f.title not in unsafe_titles
    ]
    recommendations = []
    for rec in report.recommendations:
        if rec.linked_factor in unsafe_titles:
            continue
        fields = ("action", "target", "how_to_track", "recheck", "rationale")
        cleaned = {name: clean(getattr(rec, name)) for name in fields}
        if cleaned["action"]:
            recommendations.append(rec.model_copy(update=cleaned))
    gaps = [clean(gap) for gap in report.insufficient_data]
    cleaned_report = report.model_copy(
        update={
            "summary": clean(report.summary),
            "apparent_age_note": clean(report.apparent_age_note)
            if report.apparent_age_note
            else None,
            "factors": factors,
            "recommendations": recommendations,
            "insufficient_data": [gap for gap in gaps if gap],
        }
    )
    return cleaned_report, removed


def _is_unsafe(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in UNSAFE_OUTPUT_PATTERNS)


def _report_text(report: AgingReport) -> str:
    return "\n".join(
        [
            report.summary,
            report.apparent_age_note or "",
            *(f"{factor.title}. {factor.explanation}" for factor in report.factors),
            *(
                f"{rec.action} {rec.target} {rec.how_to_track} {rec.recheck} {rec.rationale}"
                for rec in report.recommendations
            ),
            *report.insufficient_data,
        ]
    )
