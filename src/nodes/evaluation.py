from __future__ import annotations

import re
from typing import Any

from src.config import DISCLAIMER, UNSAFE_OUTPUT_PATTERNS
from src.schemas import AgingReport, PipelineState, RunEvaluation, SafetyCheck


def evaluation_node(state: PipelineState) -> dict[str, Any]:
    report = state["report"]
    critical = state.get("critical_findings", [])
    escalation = report.escalation or ""
    safety = SafetyCheck(
        disclaimer_attached=report.disclaimer == DISCLAIMER,
        unsafe_phrases=find_unsafe_phrases(report),
        critical_findings=len(critical),
        critical_findings_escalated=sum(finding.message in escalation for finding in critical),
    )
    extraction = state.get("extraction_check")
    vision = state.get("vision_check")
    grounding = state["grounding_check"]

    failures: list[str] = []
    if not safety.disclaimer_attached:
        failures.append("Disclaimer missing from the report.")
    if safety.unsafe_phrases:
        failures.append(f"Report contains dosing or diagnostic language: {safety.unsafe_phrases}.")
    if safety.critical_findings_escalated < safety.critical_findings:
        failures.append("A critical lab value was not escalated to a clinician.")
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
