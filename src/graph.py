from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Final, Literal, cast

from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.llm import LLMConfigError
from src.nodes.bloodwork import bloodwork_node, merge_pdfs
from src.nodes.evaluation import evaluation_node
from src.nodes.face_age import compare_with_stated_age, face_age_node
from src.nodes.report import report_node
from src.nodes.synthesis import synthesis_node
from src.schemas import PipelineState
from src.validation import (
    QuestionnaireError,
    find_critical_findings,
    select_biomarker_context,
    validate_questionnaire,
)

VALIDATE_INTAKE: Final = "validate_intake"
FACE_AGE: Final = "face_age"
BLOODWORK: Final = "bloodwork"
GATHER_CONTEXT: Final = "gather_context"
CRITICAL_WARNING: Final = "critical_warning"
SYNTHESIS: Final = "synthesis"
EVALUATION: Final = "evaluation"
REPORT: Final = "report"


def validate_intake_node(state: PipelineState) -> dict[str, Any]:
    """Validate the questionnaire; the only node that can stop a run."""
    return {"questionnaire": validate_questionnaire(state.get("raw_questionnaire", {}))}


def gather_context_node(state: PipelineState) -> dict[str, Any]:
    """Merge both branches: age gap, knowledge lookup, critical screen."""
    panel = state.get("blood_panel")
    questionnaire = state["questionnaire"]
    warnings: list[str] = []
    update: dict[str, Any] = {
        "knowledge_context": select_biomarker_context(panel, questionnaire),
        "critical_findings": find_critical_findings(panel) if panel else [],
    }

    estimate = state.get("apparent_age")
    if estimate is not None:
        update["face_age"], gap_warnings = compare_with_stated_age(estimate, questionnaire)
        warnings += gap_warnings

    if panel and panel.unparsed_fields:
        warnings.append(
            f"{len(panel.unparsed_fields)} lab line(s) could not be parsed and were excluded."
        )
    if warnings:
        update["warnings"] = warnings
    return update


def critical_warning_node(state: PipelineState) -> dict[str, Any]:
    """Build the clinician warning from the critical findings."""
    return {"escalation": " ".join(f.message for f in state.get("critical_findings", []))}


def route_ingestion(state: PipelineState) -> list[str]:
    """Run the photo and lab branches that have an input."""
    branches: list[str] = []
    if state.get("image_path"):
        branches.append(FACE_AGE)
    if state.get("lab_pdf_path"):
        branches.append(BLOODWORK)
    return branches or [GATHER_CONTEXT]


def route_after_context(state: PipelineState) -> Literal["critical_warning", "synthesis"]:
    """Escalate first if any lab value is critical."""
    return CRITICAL_WARNING if state.get("critical_findings") else SYNTHESIS


def build_graph() -> Any:
    """Wire the nodes into the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node(VALIDATE_INTAKE, validate_intake_node)
    graph.add_node(FACE_AGE, face_age_node)
    graph.add_node(BLOODWORK, bloodwork_node)
    graph.add_node(GATHER_CONTEXT, gather_context_node)
    graph.add_node(CRITICAL_WARNING, critical_warning_node)
    graph.add_node(SYNTHESIS, synthesis_node)
    graph.add_node(EVALUATION, evaluation_node)
    graph.add_node(REPORT, report_node)

    graph.add_edge(START, VALIDATE_INTAKE)
    graph.add_conditional_edges(
        VALIDATE_INTAKE, route_ingestion, [FACE_AGE, BLOODWORK, GATHER_CONTEXT]
    )
    graph.add_edge(FACE_AGE, GATHER_CONTEXT)
    graph.add_edge(BLOODWORK, GATHER_CONTEXT)
    graph.add_conditional_edges(GATHER_CONTEXT, route_after_context, [CRITICAL_WARNING, SYNTHESIS])
    graph.add_edge(CRITICAL_WARNING, SYNTHESIS)
    graph.add_edge(SYNTHESIS, EVALUATION)
    graph.add_edge(EVALUATION, REPORT)
    graph.add_edge(REPORT, END)

    return graph.compile()


def run_pipeline(
    questionnaire: dict[str, Any],
    image_path: str | Path | None = None,
    lab_pdf_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> PipelineState:
    """Run the whole pipeline once and return the final state."""
    initial: PipelineState = {
        "raw_questionnaire": questionnaire,
        "image_path": str(image_path) if image_path else None,
        "lab_pdf_path": str(lab_pdf_path) if lab_pdf_path else None,
        "report_pdf_path": str(report_path) if report_path else None,
        "warnings": [],
    }
    return cast(PipelineState, build_graph().invoke(initial))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Run the Longevity Insights pipeline.")
    parser.add_argument("--intake", required=True, type=Path, help="JSON file of intake answers.")
    parser.add_argument("--image", type=Path, default=None, help="Optional face photo.")
    parser.add_argument(
        "--labs",
        type=Path,
        nargs="+",
        default=None,
        help="Optional lab report PDFs from one visit.",
    )
    parser.add_argument(
        "--out", type=Path, default=settings.default_report_path, help="PDF output path."
    )
    args = parser.parse_args()

    try:
        questionnaire = json.loads(args.intake.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read intake file {args.intake}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        with tempfile.TemporaryDirectory() as workspace:
            labs = merge_pdfs(args.labs, Path(workspace) / "lab_reports.pdf") if args.labs else None
            state = run_pipeline(
                questionnaire=questionnaire,
                image_path=args.image,
                lab_pdf_path=labs,
                report_path=args.out,
            )
    except (QuestionnaireError, LLMConfigError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    except RuntimeError as exc:
        print(f"A lab report could not be opened: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    for warning in state.get("warnings", []):
        print(f"[warning] {warning}")
    print(state["report"].summary)
    evaluation = state["evaluation"]
    print(f"Quality checks: {'passed' if evaluation.passed else 'FAILED'}")
    for failure in evaluation.failures:
        print(f"[check failed] {failure}")
    print(f"Report written to {state.get('report_pdf_path')}")


if __name__ == "__main__":
    main()
