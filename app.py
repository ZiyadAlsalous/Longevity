from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    DISCLAIMER,
    MAX_ALCOHOL_UNITS_PER_WEEK,
    MAX_CHRONOLOGICAL_AGE,
    MAX_EXERCISE_MINUTES_PER_WEEK,
    MAX_SLEEP_HOURS,
    MIN_SLEEP_HOURS,
    MIN_CHRONOLOGICAL_AGE,
    STRESS_SCALE_MAX,
    STRESS_SCALE_MIN,
    settings,
)
from src.graph import run_pipeline
from src.llm import LLMConfigError
from src.nodes.report import render_report_pdf
from src.schemas import DietPattern, PipelineState, Sex, SmokingStatus, SunExposure
from src.validation import QuestionnaireError

st.set_page_config(page_title="Longevity Insights", page_icon="🧬", layout="centered")

RESULT_KEY = "pipeline_state"


def collect_questionnaire() -> dict[str, Any]:
    """Render the intake form and return the answers."""
    left, right = st.columns(2)
    with left:
        age = st.number_input(
            "Age", min_value=MIN_CHRONOLOGICAL_AGE, max_value=MAX_CHRONOLOGICAL_AGE, value=45
        )
        sleep = st.slider(
            "Average sleep (hours per night)", MIN_SLEEP_HOURS, MAX_SLEEP_HOURS, 7.0, 0.5
        )
        exercise = st.slider(
            "Exercise (minutes per week)", 0, MAX_EXERCISE_MINUTES_PER_WEEK, 120, 15
        )
        alcohol = st.slider("Alcohol (units per week)", 0.0, MAX_ALCOHOL_UNITS_PER_WEEK, 4.0, 1.0)
        stress = st.slider(
            f"Perceived stress ({STRESS_SCALE_MIN} low, {STRESS_SCALE_MAX} high)",
            STRESS_SCALE_MIN,
            STRESS_SCALE_MAX,
            5,
        )
    with right:
        sex = st.selectbox(
            "Sex (recorded only; not used in the analysis)", [item.value for item in Sex]
        )
        smoking = st.selectbox("Smoking or vaping", [item.value for item in SmokingStatus])
        diet = st.selectbox("Dietary pattern", [item.value for item in DietPattern])
        sun = st.selectbox("Unprotected sun exposure", [item.value for item in SunExposure])
        medications = st.text_input("Medications or supplements (comma separated)", "")
        family = st.text_input("Family history (comma separated)", "")

    return {
        "chronological_age": int(age),
        "sex": sex,
        "sleep_hours": float(sleep),
        "alcohol_units_per_week": float(alcohol),
        "smoking_status": smoking,
        "exercise_minutes_per_week": int(exercise),
        "diet_pattern": diet,
        "perceived_stress": int(stress),
        "sun_exposure": sun,
        "medications": _split_comma_list(medications),
        "family_history": _split_comma_list(family),
    }


def render_report(state: PipelineState) -> None:
    """Show the finished report on the page."""
    report = state["report"]

    if report.escalation:
        st.error(report.escalation)

    st.subheader("Summary")
    st.write(report.summary)

    if report.apparent_age_note:
        st.subheader("Perceived age signal")
        st.write(report.apparent_age_note)

    if report.factors:
        st.subheader("Contributing factors")
        for index, factor in enumerate(report.factors, start=1):
            with st.expander(f"{index}. {factor.title}  ({factor.confidence.value} confidence)"):
                st.write(factor.explanation)
                for evidence in factor.evidence:
                    st.caption(evidence.render())

    if report.recommendations:
        st.subheader("What to consider next")
        for rec in report.ranked_recommendations():
            st.markdown(f"**{rec.priority}. {rec.action}**")
            st.caption(rec.rationale)

    if report.insufficient_data:
        st.subheader("Not assessed")
        for item in report.insufficient_data:
            st.write(f"- {item}")

    for warning in state.get("warnings", []):
        st.warning(warning)

    st.download_button(
        "Download PDF report",
        data=render_report_pdf(state),
        file_name="longevity_insights_report.pdf",
        mime="application/pdf",
    )


def main() -> None:
    """Run the Streamlit app."""
    st.title("Longevity Insights")
    st.caption(
        "Local demo of an evidence-grounded, non-diagnostic aging-signal pipeline. "
        f"Model: {settings.llm_model}, running locally."
    )
    st.info(DISCLAIMER, icon="⚠️")

    questionnaire = collect_questionnaire()
    photo = st.file_uploader("Face photo (optional)", type=["jpg", "jpeg", "png"])
    labs = st.file_uploader("Lab report PDF (optional)", type=["pdf"])

    if st.button("Run pipeline", type="primary"):
        with tempfile.TemporaryDirectory() as workspace:
            paths = {
                "image_path": _save_upload(photo, Path(workspace)),
                "lab_pdf_path": _save_upload(labs, Path(workspace)),
            }
            with st.spinner("Running the pipeline..."):
                try:
                    st.session_state[RESULT_KEY] = run_pipeline(
                        questionnaire=questionnaire, **paths
                    )
                except (QuestionnaireError, LLMConfigError) as exc:
                    st.session_state.pop(RESULT_KEY, None)
                    st.error(str(exc))
                    return

    if RESULT_KEY in st.session_state:
        render_report(st.session_state[RESULT_KEY])


def _split_comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _save_upload(upload: Any, workspace: Path) -> Path | None:
    if upload is None:
        return None
    path = workspace / Path(str(upload.name)).name
    path.write_bytes(upload.getbuffer())
    return path


main()
