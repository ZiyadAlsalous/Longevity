from __future__ import annotations

import operator
from datetime import date
from enum import Enum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import (
    MAX_ALCOHOL_UNITS_PER_WEEK,
    MAX_CHRONOLOGICAL_AGE,
    MAX_EXERCISE_MINUTES_PER_WEEK,
    MAX_SLEEP_HOURS,
    MIN_CHRONOLOGICAL_AGE,
    STRESS_SCALE_MAX,
    STRESS_SCALE_MIN,
    UNMAPPED,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Sex(str, Enum):
    FEMALE = "female"
    MALE = "male"
    INTERSEX_OR_OTHER = "intersex_or_other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class SmokingStatus(str, Enum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class DietPattern(str, Enum):
    MIXED_WESTERN = "mixed_western"
    MEDITERRANEAN = "mediterranean"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    LOW_CARB = "low_carb"
    HIGHLY_PROCESSED = "highly_processed"


class SunExposure(str, Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HIGH = "high"


class Questionnaire(StrictModel):
    chronological_age: int = Field(ge=MIN_CHRONOLOGICAL_AGE, le=MAX_CHRONOLOGICAL_AGE)
    sex: Sex
    sleep_hours: float = Field(ge=0.0, le=MAX_SLEEP_HOURS)
    alcohol_units_per_week: float = Field(ge=0.0, le=MAX_ALCOHOL_UNITS_PER_WEEK)
    smoking_status: SmokingStatus
    exercise_minutes_per_week: int = Field(ge=0, le=MAX_EXERCISE_MINUTES_PER_WEEK)
    diet_pattern: DietPattern
    perceived_stress: int = Field(ge=STRESS_SCALE_MIN, le=STRESS_SCALE_MAX)
    sun_exposure: SunExposure
    medications: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)


class ApparentAgeEstimate(StrictModel):
    apparent_age: float
    interval_low: float
    interval_high: float
    face_confidence: float = Field(ge=0.0, le=1.0)


class FaceAgeSignal(StrictModel):
    apparent_age: float
    chronological_age: int
    age_gap_years: float
    interval_low: float
    interval_high: float
    face_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_interval(self) -> FaceAgeSignal:
        if not self.interval_low <= self.apparent_age <= self.interval_high:
            raise ValueError("apparent_age must fall inside [interval_low, interval_high]")
        return self


class RangeFlag(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class BloodAnalyte(StrictModel):
    canonical_name: str = Field(
        description=f"Canonical name from the knowledge base, or '{UNMAPPED}'. Assigned "
        f"downstream, so always return '{UNMAPPED}'."
    )
    reported_name: str = Field(description="Analyte name exactly as printed.")
    value: float = Field(description="Numeric result, as printed.")
    unit: str = Field(description="Unit exactly as printed.")
    reference_range_low: float | None = Field(
        default=None, description="Lower bound printed on the report, null if absent."
    )
    reference_range_high: float | None = Field(
        default=None, description="Upper bound printed on the report, null if absent."
    )
    flag: RangeFlag = Field(
        default=RangeFlag.UNKNOWN, description="Computed downstream, not by the model."
    )


class BloodPanel(StrictModel):
    analytes: list[BloodAnalyte] = Field(default_factory=list)
    collected_on: date | None = Field(default=None, description="Specimen collection date.")
    lab_name: str | None = Field(default=None, description="Issuing laboratory, if printed.")
    unparsed_fields: list[str] = Field(
        default_factory=list,
        description="Result-looking lines that could not be transcribed confidently. "
        "Surfaced verbatim rather than guessed at.",
    )
    used_ocr: bool = False


class BiomarkerReference(StrictModel):
    canonical_name: str
    display_name: str
    synonyms: list[str] = Field(default_factory=list)
    canonical_unit: str
    unit_conversions: dict[str, float] = Field(default_factory=dict)
    optimal_low: float | None = None
    optimal_high: float | None = None
    critical_low: float | None = None
    critical_high: float | None = None
    aging_relevance: str
    citation: str


class CriticalFinding(StrictModel):
    canonical_name: str
    value: float
    unit: str
    threshold: float
    direction: RangeFlag
    message: str


class EvidenceSource(str, Enum):
    BIOMARKER = "biomarker"
    QUESTIONNAIRE = "questionnaire"
    FACE_MODEL = "face_model"
    KNOWLEDGE_BASE = "knowledge_base"


class Evidence(StrictModel):
    source: EvidenceSource
    identifier: str = Field(description="Canonical analyte name or questionnaire field name.")
    observed: str = Field(description="The observed value, rendered for the reader.")

    def render(self) -> str:
        return f"{self.source.value}: {self.identifier} = {self.observed.rstrip('.')}"


class Confidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ContributingFactor(StrictModel):
    title: str = Field(description="Short name, for example 'Elevated fasting glucose'.")
    explanation: str = Field(description="Plain-language, non-diagnostic explanation.")
    evidence: list[Evidence] = Field(
        min_length=1, description="At least one identifier from the evidence block."
    )
    confidence: Confidence


class Recommendation(StrictModel):
    priority: int = Field(ge=1, description="1 is highest priority.")
    action: str = Field(description="A behaviour. Never a drug, dose, or supplement regimen.")
    rationale: str
    linked_factor: str = Field(description="Title of the factor this addresses.")
    citations: list[str] = Field(default_factory=list)


class AgingReport(StrictModel):
    summary: str = Field(description="Two to four sentences on the aging signal.")
    apparent_age_note: str | None = Field(
        default=None, description="How the perceived-age gap was read, if a photo was scored."
    )
    factors: list[ContributingFactor] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    insufficient_data: list[str] = Field(
        default_factory=list, description="What could not be assessed. Preferred to a guess."
    )
    escalation: str | None = Field(default=None, description="Attached by code, not by the model.")
    disclaimer: str = Field(default="", description="Attached by code, not by the model.")

    def ranked_recommendations(self) -> list[Recommendation]:
        return sorted(self.recommendations, key=lambda item: item.priority)


class ExtractionCheck(StrictModel):
    analytes_extracted: int
    analytes_mapped: int
    values_not_in_document: list[str]
    unparsed_lines: int
    used_ocr: bool
    unreadable_pages: int


class VisionCheck(StrictModel):
    scored: bool
    face_confidence: float | None = None
    reason_not_scored: str | None = None


class GroundingCheck(StrictModel):
    factors_generated: int
    factors_kept: int
    factors_dropped_for_invented_biomarkers: list[str]
    recommendations_generated: int
    recommendations_kept: int


class SafetyCheck(StrictModel):
    disclaimer_attached: bool
    unsafe_phrases: list[str]
    critical_findings: int
    critical_findings_escalated: int


class RunEvaluation(StrictModel):
    extraction: ExtractionCheck | None
    vision: VisionCheck | None
    grounding: GroundingCheck
    safety: SafetyCheck
    passed: bool
    failures: list[str]


class PipelineState(TypedDict, total=False):
    image_path: str | None
    lab_pdf_path: str | None
    raw_questionnaire: dict[str, Any]

    questionnaire: Questionnaire
    apparent_age: ApparentAgeEstimate | None
    face_age: FaceAgeSignal | None
    blood_panel: BloodPanel | None
    knowledge_context: list[BiomarkerReference]
    critical_findings: list[CriticalFinding]
    escalation: str | None

    report: AgingReport
    extraction_check: ExtractionCheck | None
    vision_check: VisionCheck | None
    grounding_check: GroundingCheck
    evaluation: RunEvaluation
    report_pdf_path: str | None

    warnings: Annotated[list[str], operator.add]
