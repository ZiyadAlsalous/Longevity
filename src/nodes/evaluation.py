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
    return [
        sentence.strip()
        for sentence in sentences
        if any(re.search(p, sentence, flags=re.IGNORECASE) for p in UNSAFE_OUTPUT_PATTERNS)
    ]


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
