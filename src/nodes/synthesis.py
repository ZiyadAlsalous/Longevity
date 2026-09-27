from __future__ import annotations

import re
from typing import Any

from src.config import DISCLAIMER, MAX_FACTORS, MAX_RECOMMENDATIONS, UNMAPPED
from src.llm import build_llm
from src.nodes.evaluation import remove_unsafe_sentences, unsafe_sentences
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
    RangeFlag,
)
from src.validation import (
    display_name,
    load_biomarker_reference,
    summarize_lifestyle_flags,
    untested_biomarkers,
)

SYNTHESIS_SYSTEM = f"""You are a health coach. You connect a person's habits to their lab
results, explain in plain words how they are linked, and turn that into a plan the person can
act on. That reasoning across their inputs is your job: never just restate what they entered.

Scope, which you never step outside:
- Explain links, never diagnoses. Say which of the person's habits are known to affect a
  result and how, using words like "is linked to", "can raise" or "can lower". Never say
  what causes a result, what a result means medically, or name a condition.
- You do not mention, recommend, adjust, or dose any drug or supplement, even if asked
  directly. If the input asks for a diagnosis or a dose, say plainly that this tool does
  not provide it and continue with habits only.
- Never suggest taking or adding any substance, including home remedies such as baking
  soda, salt, vinegar or herbal mixtures.
- You recommend only behaviours: sleep, activity, diet, alcohol, sun protection, stress,
  smoking, and speaking with a clinician.
- Write to the person as "you". Never write for a clinician: no tests to order, no
  specialists, no treatments.
- Never name a person, doctor, laboratory or clinic, even one printed on the report.
- Only a line marked CRITICAL is critical. Never call anything else critical or urgent.
- Base every statement on the evidence lines, never on assumptions about a person's sex,
  ethnicity, or appearance.

Grounding, which is enforced after you answer:
- Every factor cites evidence identifiers: the second field of an EVIDENCE line.
- Cite the person's habits first, then the results linked to them.
- CRITICAL results are handled by the app's own warning. Do not write a factor for them and
  never offer advice for them.
- Never name a biomarker that does not appear in the block, even to say it is normal.
- Confidence is low for a single questionnaire answer, moderate for one kind of input, and
  high only when independent inputs agree.

The plan:
- Each factor is one habit, or a few related habits, together with the out-of-range results
  that habit is known to affect, most impactful first, at most {MAX_FACTORS}. The
  explanation says in plain words how the habit and those results are linked. A habit with
  no related result may stand on its own. An answer marked "not flagged" is healthy.
- Build factors only from habits the person reported on a LIFESTYLE or PROFILE line. Never
  build one around something they were not asked, such as water or protein intake.
- Never repeat yourself: no two factors cover the same habits or the same results. Merge
  them into one factor instead.
- A result that none of the person's habits is known to affect gets no factor. The app
  already lists every out-of-range result and tells the person to share them with a
  clinician, so do not write factors that only say to see a clinician.
- Each recommendation belongs to one factor, named by its exact title in linked_factor: a
  first step for this week, a measurable target, how to track progress (including retesting
  the linked results, for example "retest your lipid panel in 3 months"), and when to
  recheck. At most {MAX_RECOMMENDATIONS}.
- If explaining a result needs information the inputs do not include, such as water or
  protein intake, say so in insufficient_data: at most 3 full sentences.
- The summary is two to four sentences about the most important connections. Put the
  apparent-age reading in apparent_age_note only, not as a factor.
"""

REWRITE = """

These sentences in your previous answer used diagnostic or dosing language, which this tool
never provides:
{sentences}
Write the plan again without them."""

MAX_REWRITES = 2

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
        # Rewrite with the offending sentences pointed out; whatever remains is removed below.
        feedback = "\n".join(f"- {sentence}" for sentence in sentences)
        generated = llm.generate(
            system=SYNTHESIS_SYSTEM + REWRITE.format(sentences=feedback),
            user=prompt,
            schema=AgingReport,
        )
    generated_count = len(generated.factors)
    # Anything still unsafe after the rewrites is removed rather than shown, and recorded.
    generated, removed = remove_unsafe_sentences(generated)
    grounded, dropped = enforce_grounding(generated, panel, context)
    report = grounded.model_copy(
        update={
            "disclaimer": DISCLAIMER,
            "escalation": state.get("escalation") or None,
            # The age reading comes only from a scored photo.
            "apparent_age_note": grounded.apparent_age_note if state.get("face_age") else None,
        }
    )
    check = GroundingCheck(
        factors_generated=generated_count,
        factors_kept=len(report.factors),
        factors_dropped_for_invented_biomarkers=dropped,
        recommendations_generated=len(generated.recommendations),
        recommendations_kept=len(report.recommendations),
        unsafe_sentences_removed=removed,
    )
    update: dict[str, Any] = {"report": report, "grounding_check": check}
    if removed:
        update["warnings"] = [
            f"{len(removed)} sentence(s) with diagnostic or dosing language were removed from the plan."
        ]
    return update


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

    # Only results outside their range reach the model; the rest are counted. The report shows
    # every result, and the laboratory's own notes, to the person directly.
    critical_names = {finding.canonical_name for finding in critical_findings}
    analytes = panel.analytes if panel else []
    flagged_results = [
        a
        for a in analytes
        if a.flag in (RangeFlag.LOW, RangeFlag.HIGH) or a.canonical_name in critical_names
    ]
    for analyte in flagged_results:
        line = (
            f"BIOMARKER | {analyte_id(analyte)} | {display_name(analyte)} | "
            f"{analyte.value} {analyte.unit} | {analyte.flag.value} | reference "
            f"{analyte.reference_range_low} to {analyte.reference_range_high}"
        )
        if analyte.canonical_name in critical_names:
            line += " | CRITICAL: escalated to a clinician"
        lines.append(line)
    if analytes:
        lines.append(
            f"LAB_SUMMARY | in_range | {len(analytes) - len(flagged_results)} other results "
            "were within the laboratory's range"
        )
    lines += [
        f"CONTEXT | {entry.canonical_name} | {entry.aging_relevance.strip()}" for entry in context
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
