from __future__ import annotations

from typing import Any

from src.config import DISCLAIMER, MAX_FACTORS, MAX_RECOMMENDATIONS, UNMAPPED
from src.llm import build_llm
from src.schemas import (
    AgingReport,
    BiomarkerReference,
    BloodAnalyte,
    BloodPanel,
    Confidence,
    ContributingFactor,
    CriticalFinding,
    Evidence,
    EvidenceSource,
    FaceAgeSignal,
    GroundingCheck,
    PipelineState,
    Questionnaire,
)
from src.validation import display_name, load_biomarker_reference, summarize_lifestyle_flags

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
Be concise. The summary is two to four complete sentences. Put the apparent-age reading in
apparent_age_note only, not as a factor. A LIFESTYLE line marked "not flagged" was answered
and is not a concern; it is not missing data. Each insufficient_data entry is a full
sentence, never an identifier such as a field name.
"""

CONFIDENCE_ORDER = [Confidence.LOW, Confidence.MODERATE, Confidence.HIGH]


def synthesis_node(state: PipelineState) -> dict[str, Any]:
    """Write the report from the evidence, then enforce grounding."""
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
    grounded, dropped = enforce_grounding(generated, panel, state.get("knowledge_context", []))
    report = grounded.model_copy(
        update={"disclaimer": DISCLAIMER, "escalation": state.get("escalation") or None}
    )
    check = GroundingCheck(
        factors_generated=len(generated.factors),
        factors_kept=len(report.factors),
        factors_dropped_for_invented_biomarkers=dropped,
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
    """Turn validated inputs into the evidence block."""
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

    # Every answer is listed, so an unflagged answer is not mistaken for missing data.
    flagged = dict(summarize_lifestyle_flags(questionnaire))
    for field, answer in _lifestyle_answers(questionnaire):
        lines.append(f"LIFESTYLE | {field} | {flagged.get(field, f'{answer}, not flagged')}")

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


def invented_biomarkers(
    factor: ContributingFactor, panel: BloodPanel | None, context: list[BiomarkerReference]
) -> list[str]:
    """Biomarkers a factor cites that were never supplied, whatever source it claims."""
    measured = {analyte.canonical_name for analyte in _mapped_analytes(panel)}
    supplied = measured | {entry.canonical_name for entry in context}
    known = set(load_biomarker_reference())
    return sorted(
        {
            evidence.identifier
            for evidence in factor.evidence
            if evidence.identifier not in supplied
            and (evidence.identifier in known or evidence.source is EvidenceSource.BIOMARKER)
        }
    )


def correct_factor(
    factor: ContributingFactor, panel: BloodPanel | None, context: list[BiomarkerReference]
) -> ContributingFactor:
    """Set each evidence source from its identifier, then cap confidence by the evidence."""
    measured = {analyte.canonical_name for analyte in _mapped_analytes(panel)}
    context_names = {entry.canonical_name for entry in context}
    evidence: list[Evidence] = []
    for item in factor.evidence:
        identifier = item.identifier.split("|")[-1].strip().lower()
        source = item.source
        if identifier in measured:
            source = EvidenceSource.BIOMARKER
        elif identifier in context_names:
            source = EvidenceSource.KNOWLEDGE_BASE
        elif identifier in Questionnaire.model_fields:
            source = EvidenceSource.QUESTIONNAIRE
        elif identifier == "age_gap_years":
            source = EvidenceSource.FACE_MODEL
        corrected = item.model_copy(update={"identifier": identifier, "source": source})
        if all((e.source, e.identifier) != (source, identifier) for e in evidence):
            evidence.append(corrected)

    # Low for one questionnaire answer, moderate for one kind of input, high only when
    # independent kinds of input agree.
    sources = {item.source for item in evidence}
    if sources == {EvidenceSource.QUESTIONNAIRE} and len({e.identifier for e in evidence}) == 1:
        cap = Confidence.LOW
    elif len(sources) < 2:
        cap = Confidence.MODERATE
    else:
        cap = Confidence.HIGH
    confidence = min(factor.confidence, cap, key=CONFIDENCE_ORDER.index)
    return factor.model_copy(update={"evidence": evidence, "confidence": confidence})


def enforce_grounding(
    report: AgingReport, panel: BloodPanel | None, context: list[BiomarkerReference]
) -> tuple[AgingReport, list[str]]:
    """Correct sources and confidence, then drop ungrounded factors and their recommendations."""
    kept: list[ContributingFactor] = []
    dropped: list[str] = []
    notes: list[str] = []
    for factor in (correct_factor(f, panel, context) for f in report.factors):
        invented = invented_biomarkers(factor, panel, context)
        if invented:
            dropped.append(factor.title)
            notes.append(
                f"'{factor.title}' cited biomarkers not present in the inputs: "
                f"{', '.join(invented)}."
            )
        else:
            kept.append(factor)

    kept = kept[:MAX_FACTORS]
    # A recommendation may link to a factor by title or by one of its evidence ids.
    links = {factor.title for factor in kept}
    links |= {evidence.identifier for factor in kept for evidence in factor.evidence}
    recommendations = [rec for rec in report.recommendations if rec.linked_factor in links]
    grounded = report.model_copy(
        update={
            "factors": kept,
            "recommendations": recommendations[:MAX_RECOMMENDATIONS],
            "insufficient_data": [*report.insufficient_data, *notes],
        }
    )
    return grounded, dropped


def _lifestyle_answers(questionnaire: Questionnaire) -> list[tuple[str, str]]:
    return [
        ("sleep_hours", f"{questionnaire.sleep_hours} hours per night"),
        ("exercise_minutes_per_week", f"{questionnaire.exercise_minutes_per_week} minutes"),
        ("alcohol_units_per_week", f"{questionnaire.alcohol_units_per_week} units"),
        ("smoking_status", questionnaire.smoking_status.value),
        ("perceived_stress", f"{questionnaire.perceived_stress} of 10"),
        ("sun_exposure", questionnaire.sun_exposure.value),
    ]


def _mapped_analytes(panel: BloodPanel | None) -> list[BloodAnalyte]:
    if panel is None:
        return []
    return [analyte for analyte in panel.analytes if analyte.canonical_name != UNMAPPED]
