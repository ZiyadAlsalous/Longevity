from __future__ import annotations

import re
from typing import Any

from src.config import DISCLAIMER, MAX_FACTORS, MAX_RECOMMENDATIONS, UNMAPPED
from src.llm import build_llm
from src.nodes.evaluation import unsafe_sentences
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
from src.validation import (
    display_name,
    load_biomarker_reference,
    summarize_lifestyle_flags,
    untested_biomarkers,
)

SYNTHESIS_SYSTEM = f"""You are a health coach writing a personal improvement plan from
pre-validated data. Your job is to help the person improve their health and slow aging, not
to restate what they entered. Explain why each priority matters for them, connecting their
inputs, and give concrete next steps.

Scope, which you never step outside:
- You do not diagnose, name conditions a person may have, or interpret results clinically.
- You do not mention, recommend, adjust, or dose any drug or supplement, even if asked
  directly. If the input asks for a diagnosis or a dose, say plainly that this tool does
  not provide it and continue with lifestyle patterns only.
- You recommend only behaviours: sleep, activity, diet, alcohol, sun protection, stress,
  and speaking with a clinician.
- For an out-of-range lab result, recommend discussing it with a clinician. Never name a
  condition it could mean.
- Base every statement on the evidence lines, never on assumptions about a person's sex,
  ethnicity, or appearance.

Grounding, which is enforced after you answer:
- Every factor cites at least one evidence identifier: the second field of an EVIDENCE line.
- Never name a biomarker that does not appear in the block, even to say it is normal.
- NOT_TESTED lists markers that were not measured. Never guess their values. You may suggest
  asking a clinician about testing one when it is relevant to a priority.
- If you cannot ground a topic, list it in insufficient_data instead of writing about it.
- Confidence is low for a single questionnaire answer, moderate for one kind of input, and
  high only when independent inputs agree.
- A line marked CRITICAL has already been escalated to a clinician. Never offer lifestyle
  advice for that value; say only that it needs prompt clinical review.

The plan:
- Factors are the person's priorities, most impactful first, at most {MAX_FACTORS}. Favour
  out-of-range results and flagged answers. When any lab result is out of range, one priority
  is reviewing those results with a clinician. For a lab result, explain what the test
  measures in plain words and that the laboratory flagged it, never what it could mean. An answer marked "not flagged" is healthy and is
  not missing data; mention it only as something to keep doing.
- Each recommendation belongs to one factor, named by its exact title in linked_factor, and
  gives a first step for this week, a measurable target, how to track progress, and when to
  recheck. At most {MAX_RECOMMENDATIONS}.
- Be concise. The summary is two to four complete sentences about what to focus on. Put the
  apparent-age reading in apparent_age_note only, not as a factor. Each insufficient_data
  entry is a full sentence, never an identifier. NOT_TESTED markers are listed separately,
  so do not repeat them in insufficient_data.
"""

REWRITE = """

These sentences in your previous answer used diagnostic or dosing language, which this tool
never provides:
{sentences}
Write the plan again without them. For a lab result, say what the test measures in plain words
and that the laboratory flagged it, never what it could mean."""

MAX_REWRITES = 2

LAB_COMMENT_ID = "lab_comment"
CONFIDENCE_ORDER = [Confidence.LOW, Confidence.MODERATE, Confidence.HIGH]


def synthesis_node(state: PipelineState) -> dict[str, Any]:
    """Write the plan from the evidence, then enforce grounding."""
    panel = state.get("blood_panel")
    context = state.get("knowledge_context", [])
    prompt = build_evidence_prompt(
        questionnaire=state["questionnaire"],
        panel=panel,
        face_age=state.get("face_age"),
        context=context,
        critical_findings=state.get("critical_findings", []),
        warnings=state.get("warnings", []),
    )
    llm = build_llm()
    generated = llm.generate(system=SYNTHESIS_SYSTEM, user=prompt, schema=AgingReport)
    for _ in range(MAX_REWRITES):
        sentences = unsafe_sentences(generated)
        if not sentences:
            break
        # Rewrite with the offending sentences pointed out; the evaluation records any left.
        feedback = "\n".join(f"- {sentence}" for sentence in sentences)
        generated = llm.generate(
            system=SYNTHESIS_SYSTEM + REWRITE.format(sentences=feedback),
            user=prompt,
            schema=AgingReport,
        )
    grounded, dropped = enforce_grounding(generated, panel, context)
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
    for analyte in panel.analytes if panel else []:
        line = (
            f"BIOMARKER | {analyte_id(analyte)} | {display_name(analyte)} | "
            f"{analyte.value} {analyte.unit} | {analyte.flag.value} | reference "
            f"{analyte.reference_range_low} to {analyte.reference_range_high}"
        )
        if analyte.canonical_name in critical_names:
            line += " | CRITICAL: escalated to a clinician"
        lines.append(line)

    lines += [
        f"LAB_COMMENT | {LAB_COMMENT_ID} | {comment}"
        for comment in (panel.lab_comments if panel else [])
    ]
    lines += [
        f"CONTEXT | {entry.canonical_name} | {entry.aging_relevance.strip()}" for entry in context
    ]
    if panel is not None:
        lines += [
            f"NOT_TESTED | {entry.canonical_name} | {entry.display_name}"
            for entry in untested_biomarkers(panel)
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
        "Write the plan from these lines only. Anything absent here is unknown, and "
        "unknown belongs in insufficient_data.",
    ]
    return "\n".join(lines)


def analyte_id(analyte: BloodAnalyte) -> str:
    """The evidence identifier for a lab result: its canonical name, or its printed name."""
    if analyte.canonical_name != UNMAPPED:
        return analyte.canonical_name
    return normalize_id(analyte.reported_name)


def normalize_id(text: str) -> str:
    """Lower-case snake_case, so 'LDL Cholesterol' and 'ldl_cholesterol' compare equal."""
    return "_".join(re.findall(r"[a-z0-9]+", text.split("|")[-1].lower()))


def invented_biomarkers(
    factor: ContributingFactor, panel: BloodPanel | None, context: list[BiomarkerReference]
) -> list[str]:
    """Biomarkers a factor cites that were never supplied, whatever source it claims."""
    supplied = _supplied_ids(panel, context)
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
    measured = {analyte_id(analyte) for analyte in panel.analytes} if panel else set()
    reference_names = _supplied_ids(panel, context) - measured
    evidence: list[Evidence] = []
    for item in factor.evidence:
        identifier = normalize_id(item.identifier)
        source = item.source
        if identifier in measured:
            source = EvidenceSource.BIOMARKER
        elif identifier in reference_names:
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
    # Untested markers have their own section, so entries that only restate one are dropped.
    untested = {e.display_name.lower() for e in untested_biomarkers(panel)} if panel else set()
    gaps = [g for g in report.insufficient_data if not any(name in g.lower() for name in untested)]
    # A recommendation may name its factor by title or by one of its evidence ids; either way
    # it is relinked to the title, so the plan groups it under the right priority.
    owner: dict[str, str] = {}
    for factor in kept:
        owner[normalize_id(factor.title)] = factor.title
        for evidence in factor.evidence:
            owner.setdefault(evidence.identifier, factor.title)
    recommendations = [
        rec.model_copy(update={"linked_factor": owner[normalize_id(rec.linked_factor)]})
        for rec in report.recommendations
        if normalize_id(rec.linked_factor) in owner
    ]
    grounded = report.model_copy(
        update={
            "factors": kept,
            "recommendations": recommendations[:MAX_RECOMMENDATIONS],
            "insufficient_data": [*gaps, *notes],
        }
    )
    return grounded, dropped


def _supplied_ids(panel: BloodPanel | None, context: list[BiomarkerReference]) -> set[str]:
    supplied = {entry.canonical_name for entry in context}
    if panel is not None:
        supplied |= {analyte_id(analyte) for analyte in panel.analytes}
        if panel.lab_comments:
            supplied.add(LAB_COMMENT_ID)
        supplied |= {entry.canonical_name for entry in untested_biomarkers(panel)}
    return supplied


def _lifestyle_answers(questionnaire: Questionnaire) -> list[tuple[str, str]]:
    return [
        ("sleep_hours", f"{questionnaire.sleep_hours} hours per night"),
        ("exercise_minutes_per_week", f"{questionnaire.exercise_minutes_per_week} minutes"),
        ("alcohol_units_per_week", f"{questionnaire.alcohol_units_per_week} units"),
        ("smoking_status", questionnaire.smoking_status.value),
        ("perceived_stress", f"{questionnaire.perceived_stress} of 10"),
        ("sun_exposure", questionnaire.sun_exposure.value),
    ]
