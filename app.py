from __future__ import annotations

import tempfile
import threading
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
from src.nodes.bloodwork import merge_pdfs
from src.nodes.report import render_report_pdf
from src.schemas import DietPattern, PipelineState, Sex, SmokingStatus, SunExposure
from src.validation import (
    QuestionnaireError,
    display_name,
    out_of_range_analytes,
    untested_biomarkers,
)

st.set_page_config(page_title="Longevity Insights", page_icon="🧬", layout="centered")

RESULT_KEY = "pipeline_state"


@st.cache_resource
def _run_lock() -> threading.Lock:
    """One pipeline run at a time, across every tab and rerun."""
    return threading.Lock()


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

    panel = state.get("blood_panel")
    if panel is not None:
        st.subheader("Blood test results")
        st.dataframe(
            [
                {
                    "Test": display_name(a),
                    "Result": f"{a.value} {a.unit}",
                    "Range": f"{a.reference_range_low} to {a.reference_range_high}",
                    "Flag": a.flag.value,
                }
                for a in panel.analytes
            ],
            hide_index=True,
        )
        flagged = [display_name(a) for a in out_of_range_analytes(panel)]
        if flagged:
            st.warning(
                f"Outside the laboratory's range: {', '.join(flagged)}. "
                "Share these results with your clinician."
            )
        for comment in panel.lab_comments:
            st.caption(f"The laboratory noted: {comment}")
        untested = untested_biomarkers(panel)
        if untested:
            st.caption(
                "Not included in this blood test: "
                + ", ".join(entry.display_name for entry in untested)
                + ". Ask your clinician whether any are worth testing."
            )

    if report.factors:
        st.subheader("Your priorities")
        for index, factor in enumerate(report.factors, start=1):
            with st.expander(
                f"{index}. {factor.title}  ({factor.confidence.value} confidence)", expanded=True
            ):
                st.write(factor.explanation)
                for rec in report.ranked_recommendations():
                    if rec.linked_factor == factor.title:
                        st.markdown(
                            f"**First step:** {rec.action}  \n**Target:** {rec.target}  \n"
                            f"**How to track it:** {rec.how_to_track}  \n"
                            f"**Check again:** {rec.recheck}"
                        )

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
    labs = st.file_uploader(
        "Lab report PDFs from one visit (optional)", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Run pipeline", type="primary"):
        lock = _run_lock()
        if not lock.acquire(blocking=False):
            st.warning(
                "A report is already being generated. Wait for it to finish, then run again."
            )
            return
        try:
            _run(questionnaire, photo, labs)
        finally:
            lock.release()

    if RESULT_KEY in st.session_state:
        render_report(st.session_state[RESULT_KEY])


def _run(questionnaire: dict[str, Any], photo: Any, labs: list[Any]) -> None:
    with tempfile.TemporaryDirectory() as workspace:
        folder = Path(workspace)
        reports = [_save_upload(upload, folder, prefix=f"{i}_") for i, upload in enumerate(labs)]
        try:
            lab_pdf = (
                merge_pdfs([p for p in reports if p], folder / "lab_reports.pdf") if labs else None
            )
        except RuntimeError as exc:
            st.error(f"A lab report could not be opened: {exc}")
            return
        paths = {"image_path": _save_upload(photo, folder), "lab_pdf_path": lab_pdf}
        with st.spinner("Running the pipeline..."):
            try:
                st.session_state[RESULT_KEY] = run_pipeline(questionnaire=questionnaire, **paths)
            except (QuestionnaireError, LLMConfigError) as exc:
                st.session_state.pop(RESULT_KEY, None)
                st.error(str(exc))


def _split_comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _save_upload(upload: Any, workspace: Path, prefix: str = "") -> Path | None:
    if upload is None:
        return None
    path = workspace / f"{prefix}{Path(str(upload.name)).name}"
    path.write_bytes(upload.getbuffer())
    return path


main()
