from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache

import yaml
from pydantic import ValidationError

from src.config import (
    BIOMARKERS_PATH,
    ESCALATION_TEMPLATE,
    HEAVY_DRINKING_UNITS_PER_WEEK,
    HIGH_SLEEP_HOURS,
    HIGH_STRESS_SCORE,
    LOW_SLEEP_HOURS,
    LOW_SUN_BIOMARKER,
    STRESS_SCALE_MAX,
    UNMAPPED,
    WEEKLY_EXERCISE_TARGET_MINUTES,
)
from src.schemas import (
    BiomarkerReference,
    BloodAnalyte,
    BloodPanel,
    CriticalFinding,
    Questionnaire,
    RangeFlag,
    SmokingStatus,
    SunExposure,
)


class QuestionnaireError(ValueError):
    pass


def validate_questionnaire(raw: object) -> Questionnaire:
    """Validate the intake answers."""
    if not isinstance(raw, Mapping):
        raise QuestionnaireError(
            f"Invalid intake answers: expected an object of answers, got {type(raw).__name__}."
        )
    try:
        return Questionnaire.model_validate(dict(raw))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise QuestionnaireError(f"Invalid intake answers: {problems}") from exc


def summarize_lifestyle_flags(questionnaire: Questionnaire) -> list[tuple[str, str]]:
    """Flag lifestyle answers outside healthy ranges."""
    flags: list[tuple[str, str]] = []
    if questionnaire.sleep_hours < LOW_SLEEP_HOURS:
        flags.append(("sleep_hours", f"Short sleep: {questionnaire.sleep_hours} hours per night."))
    if questionnaire.sleep_hours > HIGH_SLEEP_HOURS:
        flags.append(("sleep_hours", f"Long sleep: {questionnaire.sleep_hours} hours per night."))
    if questionnaire.exercise_minutes_per_week < WEEKLY_EXERCISE_TARGET_MINUTES:
        flags.append(
            (
                "exercise_minutes_per_week",
                f"Activity below the {WEEKLY_EXERCISE_TARGET_MINUTES} minute weekly guideline: "
                f"{questionnaire.exercise_minutes_per_week} minutes.",
            )
        )
    if questionnaire.alcohol_units_per_week > HEAVY_DRINKING_UNITS_PER_WEEK:
        flags.append(
            (
                "alcohol_units_per_week",
                f"Alcohol intake: {questionnaire.alcohol_units_per_week} units per week.",
            )
        )
    if questionnaire.smoking_status is SmokingStatus.CURRENT:
        flags.append(("smoking_status", "Current smoking or vaping."))
    if questionnaire.perceived_stress >= HIGH_STRESS_SCORE:
        flags.append(
            (
                "perceived_stress",
                f"High perceived stress: {questionnaire.perceived_stress} of {STRESS_SCALE_MAX}.",
            )
        )
    if questionnaire.sun_exposure is SunExposure.HIGH:
        flags.append(("sun_exposure", "High unprotected sun exposure."))
    if questionnaire.sun_exposure is SunExposure.MINIMAL:
        flags.append(("sun_exposure", "Minimal sun exposure, relevant to vitamin D status."))
    return flags


@lru_cache(maxsize=1)
def load_biomarker_reference() -> dict[str, BiomarkerReference]:
    entries = yaml.safe_load(BIOMARKERS_PATH.read_text(encoding="utf-8"))
    references = [BiomarkerReference.model_validate(entry) for entry in entries]
    return {reference.canonical_name: reference for reference in references}


def canonical_analyte_name(reported_name: str) -> str | None:
    """Match the printed name, then the name outside brackets, then each bracketed name."""
    index = _synonym_index()
    outside = re.sub(r"\([^)]*\)", " ", reported_name)
    for candidate in (reported_name, outside, *re.findall(r"\(([^)]*)\)", reported_name)):
        label = _normalize_label(candidate)
        if label in index:
            return index[label]
    return None


def display_name(analyte: BloodAnalyte) -> str:
    reference = load_biomarker_reference().get(analyte.canonical_name)
    return reference.display_name if reference else analyte.reported_name


@lru_cache(maxsize=1)
def _synonym_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for reference in load_biomarker_reference().values():
        for name in [reference.canonical_name, reference.display_name, *reference.synonyms]:
            index[_normalize_label(name)] = reference.canonical_name
    return index


def _normalize_label(label: str) -> str:
    # Letters and digits only, so "Vitamin B-12" and "vitamin b12" match.
    return "".join(char for char in label.lower() if char.isalnum())


def canonical_unit_factor(unit: str, reference: BiomarkerReference) -> float | None:
    printed = _normalize_unit_label(unit)
    if printed == _normalize_unit_label(reference.canonical_unit):
        return 1.0
    return reference.unit_conversions.get(printed)


def _normalize_unit_label(unit: str) -> str:
    return unit.strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def normalize_blood_panel(panel: BloodPanel) -> BloodPanel:
    """Map names, convert units and flag each lab value."""
    references = load_biomarker_reference()
    normalized: list[BloodAnalyte] = []
    unparsed = list(panel.unparsed_fields)
    seen: set[str] = set()

    for analyte in panel.analytes:
        canonical = canonical_analyte_name(analyte.reported_name)
        reference = references.get(canonical) if canonical else None
        factor = canonical_unit_factor(analyte.unit, reference) if reference else None
        if canonical is None or reference is None or factor is None or canonical in seen:
            # Not in the knowledge base, a repeat, or an unknown unit: judged only against
            # the range the laboratory printed, in the laboratory's own units.
            normalized.append(_on_printed_range(analyte))
            continue

        seen.add(canonical)
        value = analyte.value * factor
        low, high = _effective_range(analyte, reference, factor)
        normalized.append(
            analyte.model_copy(
                update={
                    "canonical_name": canonical,
                    "value": round(value, 2),
                    "unit": reference.canonical_unit,
                    "reference_range_low": low,
                    "reference_range_high": high,
                    "flag": _flag_for(value, low, high),
                }
            )
        )

    return panel.model_copy(update={"analytes": normalized, "unparsed_fields": unparsed})


def _on_printed_range(analyte: BloodAnalyte) -> BloodAnalyte:
    low, high = analyte.reference_range_low, analyte.reference_range_high
    return analyte.model_copy(
        update={"canonical_name": UNMAPPED, "flag": _flag_for(analyte.value, low, high)}
    )


def _effective_range(
    analyte: BloodAnalyte, reference: BiomarkerReference, factor: float
) -> tuple[float | None, float | None]:
    low, high = analyte.reference_range_low, analyte.reference_range_high
    if low is None and high is None:
        return reference.optimal_low, reference.optimal_high
    return _scale_bound(low, factor), _scale_bound(high, factor)


def _scale_bound(bound: float | None, factor: float) -> float | None:
    if bound is None:
        return None
    return round(bound * factor, 2)


def _flag_for(value: float, low: float | None, high: float | None) -> RangeFlag:
    if low is None and high is None:
        return RangeFlag.UNKNOWN
    if low is not None and value < low:
        return RangeFlag.LOW
    if high is not None and value > high:
        return RangeFlag.HIGH
    return RangeFlag.NORMAL


def out_of_range_analytes(panel: BloodPanel) -> list[BloodAnalyte]:
    flagged = [a for a in panel.analytes if a.flag in (RangeFlag.LOW, RangeFlag.HIGH)]
    return sorted(flagged, key=lambda a: abs(deviation_percent(a)), reverse=True)


def deviation_percent(analyte: BloodAnalyte) -> float:
    high, low = analyte.reference_range_high, analyte.reference_range_low
    if analyte.flag is RangeFlag.HIGH and high:
        return (analyte.value - high) / abs(high) * 100
    if analyte.flag is RangeFlag.LOW and low:
        return (analyte.value - low) / abs(low) * 100
    return 0.0


def find_critical_findings(panel: BloodPanel) -> list[CriticalFinding]:
    """Find lab values past a critical threshold."""
    references = load_biomarker_reference()
    findings: list[CriticalFinding] = []

    for analyte in panel.analytes:
        reference = references.get(analyte.canonical_name)
        if reference is None:
            continue
        if reference.critical_high is not None and analyte.value >= reference.critical_high:
            findings.append(
                _critical_finding(analyte, reference, reference.critical_high, RangeFlag.HIGH)
            )
        elif reference.critical_low is not None and analyte.value <= reference.critical_low:
            findings.append(
                _critical_finding(analyte, reference, reference.critical_low, RangeFlag.LOW)
            )
    return findings


def _critical_finding(
    analyte: BloodAnalyte, reference: BiomarkerReference, threshold: float, direction: RangeFlag
) -> CriticalFinding:
    return CriticalFinding(
        canonical_name=analyte.canonical_name,
        value=analyte.value,
        unit=analyte.unit,
        threshold=threshold,
        direction=direction,
        message=ESCALATION_TEMPLATE.format(
            display_name=reference.display_name, value=analyte.value, unit=analyte.unit
        ),
    )


def select_biomarker_context(
    panel: BloodPanel | None, questionnaire: Questionnaire
) -> list[BiomarkerReference]:
    """Pick the knowledge base entries relevant to this run."""
    references = load_biomarker_reference()
    selected = {a.canonical_name for a in panel.analytes} if panel else set()
    if questionnaire.sun_exposure is SunExposure.MINIMAL:
        selected.add(LOW_SUN_BIOMARKER)
    return [reference for name, reference in references.items() if name in selected]


def untested_biomarkers(panel: BloodPanel) -> list[BiomarkerReference]:
    """Knowledge-base markers the lab report did not include."""
    measured = {analyte.canonical_name for analyte in panel.analytes}
    return [entry for name, entry in load_biomarker_reference().items() if name not in measured]
