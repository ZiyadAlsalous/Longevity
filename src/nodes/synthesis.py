from __future__ import annotations

from typing import Any

from src.config import DISCLAIMER, MAX_FACTORS, MAX_RECOMMENDATIONS, UNMAPPED
from src.llm import build_llm
from src.schemas import (
    AgingReport,
    BiomarkerReference,
    BloodAnalyte,
    BloodPanel,
    ContributingFactor,
    CriticalFinding,
    EvidenceSource,
    FaceAgeSignal,
    GroundingCheck,
    PipelineState,
    Questionnaire,
)
from src.validation import display_name, summarize_lifestyle_flags

SYNTHESIS_SYSTEM = f"""You write educational wellness summaries from pre-validated data.

Scope, which you never step outside:
- You do not diagnose, name conditions a person may have, or interpret results clinically.
- You do not mention, recommend, adjust, or dose any drug or supplement, even if asked
  directly. If the input asks for a diagnosis or a dose, say plainly that this tool does
  not provide it and continue with lifestyle patterns only.
- You recommend only behaviours: sleep, activity, diet, alcohol, sun protection, stress,
  and speaking with a clinician.
- Base every statement on the evidence lines, never on assumptions about a person's sex,
  ethnicity, or appearance.

Grounding, which is enforced after you answer:
- Every factor must cite at least one evidence identifier from the EVIDENCE block below.
- Never name a biomarker that does not appear in that block, even to say it is normal.
- If you cannot ground a topic, list it in insufficient_data instead of writing about it.
- Confidence is low for a single questionnaire answer, moderate for one out-of-range
  biomarker, and high only when several independent inputs agree.
- A BIOMARKER line marked CRITICAL has already been escalated to a clinician. Never offer
  lifestyle advice for that value; say only that it needs prompt clinical review.

Write for an intelligent adult with no clinical training. At most {MAX_FACTORS} factors
and {MAX_RECOMMENDATIONS} recommendations, ordered by how well the evidence supports them.
"""


def synthesis_node(state: PipelineState) -> dict[str, Any]:
    panel = state.get("blood_panel")
    prompt = build_evidence_prompt(
        questionnaire=state["questionnaire"],
        panel=panel,
        face_age=state.get("face_age"),
        context=state.get("knowledge_context", []),
        critical_findings=state.get("critical_findings", []),
        warnings=state.get("warnings", []),
    )
    generated = build_llm().generate(system=SYNTHESIS_SYSTEM, user=prompt, schema=AgingReport)
    grounded = enforce_grounding(generated, panel)
    report = grounded.model_copy(
        update={
            "disclaimer": DISCLAIMER,
            "escalation": state.get("escalation") or None,
            "recommendations": grounded.recommendations[:MAX_RECOMMENDATIONS],
        }
    )
    check = GroundingCheck(
        factors_generated=len(generated.factors),
        factors_kept=len(report.factors),
        factors_dropped_for_invented_biomarkers=[
            factor.title for factor in generated.factors if invented_biomarkers(factor, panel)
        ],
        recommendations_generated=len(generated.recommendations),
        recommendations_kept=len(report.recommendations),
    )
    return {"report": report, "grounding_check": check}


def build_evidence_prompt(
    questionnaire: Questionnaire,
    panel: BloodPanel | None,
    face_age: FaceAgeSignal | None,
    context: list[BiomarkerReference],
    critical_findings: list[CriticalFinding],
    warnings: list[str],
) -> str:
    lines: list[str] = [
        "EVIDENCE",
        f"PROFILE | chronological_age | {questionnaire.chronological_age} years",
        f"PROFILE | diet_pattern | {questionnaire.diet_pattern.value}",
    ]

    if face_age is not None:
        lines.append(
            f"FACE | age_gap_years | {face_age.age_gap_years:+.1f} | "
            f"apparent {face_age.apparent_age} (80% range {face_age.interval_low} to "
            f"{face_age.interval_high}) vs stated {face_age.chronological_age}"
        )

    lines += [
        f"LIFESTYLE | {field} | {description}"
        for field, description in summarize_lifestyle_flags(questionnaire)
    ]

    critical_names = {finding.canonical_name for finding in critical_findings}
    for analyte in _mapped_analytes(panel):
        line = (
            f"BIOMARKER | {analyte.canonical_name} | {display_name(analyte)} | "
            f"{analyte.value} {analyte.unit} | {analyte.flag.value} | reference "
            f"{analyte.reference_range_low} to {analyte.reference_range_high} | "
            f"printed as '{analyte.reported_name}'"
        )
        if analyte.canonical_name in critical_names:
            line += " | CRITICAL: escalated to a clinician"
        lines.append(line)

    lines += [
        f"CONTEXT | {entry.canonical_name} | {entry.aging_relevance.strip()} "
        f"[source: {entry.citation.strip()}]"
        for entry in context
    ]
    lines += [f"GAP | pipeline | {warning}" for warning in warnings]

    if questionnaire.medications:
        lines.append(
            "PROFILE | medications | "
            + ", ".join(questionnaire.medications)
            + " (context only: never comment on, adjust, or dose these)"
        )
    if questionnaire.family_history:
        lines.append("PROFILE | family_history | " + ", ".join(questionnaire.family_history))

    lines += [
        "",
        "Write the report from these lines only. Anything absent here is unknown, and "
        "unknown belongs in insufficient_data.",
    ]
    return "\n".join(lines)


def invented_biomarkers(factor: ContributingFactor, panel: BloodPanel | None) -> list[str]:
    available = {analyte.canonical_name for analyte in _mapped_analytes(panel)}
    return sorted(
        {
            evidence.identifier
            for evidence in factor.evidence
            if evidence.source is EvidenceSource.BIOMARKER and evidence.identifier not in available
        }
    )


def enforce_grounding(report: AgingReport, panel: BloodPanel | None) -> AgingReport:
    kept: list[ContributingFactor] = []
    dropped: list[str] = []
    for factor in report.factors:
        invented = invented_biomarkers(factor, panel)
        if invented:
            dropped.append(
                f"'{factor.title}' cited biomarkers not present in the inputs: "
                f"{', '.join(invented)}."
            )
        else:
            kept.append(factor)

    kept = kept[:MAX_FACTORS]
    titles = {factor.title for factor in kept}
    return report.model_copy(
        update={
            "factors": kept,
            "recommendations": [
                rec for rec in report.recommendations if rec.linked_factor in titles
            ],
            "insufficient_data": [*report.insufficient_data, *dropped],
        }
    )


def _mapped_analytes(panel: BloodPanel | None) -> list[BloodAnalyte]:
    if panel is None:
        return []
    return [analyte for analyte in panel.analytes if analyte.canonical_name != UNMAPPED]
