from __future__ import annotations

import re
from typing import Any

from src.config import UNSAFE_OUTPUT_PATTERNS
from src.schemas import AgingReport, PipelineState, RunEvaluation, SafetyCheck


def evaluation_node(state: PipelineState) -> dict[str, Any]:
    safety = SafetyCheck(unsafe_phrases=find_unsafe_phrases(state["report"]))
    extraction = state.get("extraction_check")
    vision = state.get("vision_check")
    grounding = state["grounding_check"]

    failures: list[str] = []
    if safety.unsafe_phrases:
        failures.append(f"Report contains dosing or diagnostic language: {safety.unsafe_phrases}.")
    if extraction is not None and extraction.values_not_in_document:
        failures.append(
            f"Extracted lab values not found in the document: {extraction.values_not_in_document}."
        )
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
    body = "\n".join(
        [
            report.summary,
            report.apparent_age_note or "",
            *(f"{factor.title}. {factor.explanation}" for factor in report.factors),
            *(f"{rec.action} {rec.rationale}" for rec in report.recommendations),
            *report.insufficient_data,
        ]
    )
    return [
        match.group(0)
        for pattern in UNSAFE_OUTPUT_PATTERNS
        for match in re.finditer(pattern, body, flags=re.IGNORECASE)
    ]
