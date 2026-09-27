from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.schemas import (
    AgingReport,
    BloodAnalyte,
    BloodPanel,
    PipelineState,
    RangeFlag,
    Recommendation,
)
from src.validation import (
    deviation_percent,
    display_name,
    load_biomarker_reference,
    out_of_range_analytes,
    untested_biomarkers,
)

ACCENT = colors.HexColor("#1F4E79")
WARNING = colors.HexColor("#B00020")
MUTED = colors.HexColor("#5A5A5A")
MAX_CHART_BARS = 6


def report_node(state: PipelineState) -> dict[str, Any]:
    """Write the report PDF to disk."""
    output_path = state.get("report_pdf_path")
    if not output_path:
        return {"report_pdf_path": None}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_report_pdf(state))
    return {"report_pdf_path": str(path)}


def render_report_pdf(state: PipelineState) -> bytes:
    """Build the report PDF in memory."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        title="Longevity Insights report",
        author="Longevity Insights",
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    document.build(_build_story(state))
    return buffer.getvalue()


def _build_story(state: PipelineState) -> list[Flowable]:
    report = state["report"]
    panel = state.get("blood_panel")
    styles = _styles()

    generated_on = datetime.now(UTC).astimezone().date().isoformat()
    story: list[Flowable] = [
        Paragraph("Longevity Insights", styles["title"]),
        Paragraph(
            f"Educational wellness summary generated on {generated_on}",
            styles["muted"],
        ),
        Spacer(1, 14),
    ]

    if report.escalation:
        story += [_callout(report.escalation, styles["callout"], WARNING), Spacer(1, 12)]

    story += [
        Paragraph("Summary", styles["heading"]),
        Paragraph(report.summary, styles["body"]),
        Spacer(1, 10),
    ]

    if report.apparent_age_note:
        story += [
            Paragraph("Perceived age signal", styles["heading"]),
            Paragraph(report.apparent_age_note, styles["body"]),
            Spacer(1, 10),
        ]

    story += _biomarker_section(panel, styles)
    story += _plan_section(report, styles)

    if report.insufficient_data:
        story += [
            Spacer(1, 6),
            Paragraph("Not assessed", styles["heading"]),
            Paragraph(
                "The pipeline reports gaps rather than filling them. It could not assess:",
                styles["body"],
            ),
            *[Paragraph(f"- {item}", styles["body"]) for item in report.insufficient_data],
        ]

    story += [PageBreak(), Paragraph("Limitations and run notes", styles["heading"])]
    story += _limitations_section(state, styles)
    story += [Spacer(1, 12), _callout(report.disclaimer, styles["callout"], MUTED)]
    return story


def _biomarker_section(
    panel: BloodPanel | None, styles: dict[str, ParagraphStyle]
) -> list[Flowable]:
    if panel is None:
        return []

    section: list[Flowable] = [Paragraph("Lab results", styles["heading"])]
    flagged = out_of_range_analytes(panel)
    if flagged:
        section.append(_biomarker_chart(flagged))
    if panel.analytes:
        section.append(_biomarker_table(panel.analytes, styles["cell"]))
    else:
        section.append(Paragraph("No results could be read from the lab report.", styles["body"]))

    if panel.collected_on:
        section.append(Paragraph(f"Collected {panel.collected_on.isoformat()}.", styles["muted"]))
    flagged_names = [display_name(a) for a in flagged]
    if flagged_names:
        section.append(
            Paragraph(
                f"<b>Outside the laboratory's range:</b> {', '.join(flagged_names)}. "
                "Share these results with your clinician.",
                styles["body"],
            )
        )
    if panel.lab_comments:
        section += [
            Spacer(1, 6),
            Paragraph("The laboratory noted", styles["subheading"]),
            *[Paragraph(f"- {comment}", styles["body"]) for comment in panel.lab_comments],
        ]
    untested = untested_biomarkers(panel)
    if untested:
        section += [
            Spacer(1, 6),
            Paragraph("Not included in this lab report", styles["subheading"]),
            Paragraph(
                ", ".join(entry.display_name for entry in untested)
                + ". Ask your clinician whether any of these are worth testing.",
                styles["body"],
            ),
        ]
    section.append(Spacer(1, 12))
    return section


def _plan_section(report: AgingReport, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    if not report.factors:
        return []

    section: list[Flowable] = [Paragraph("Your priorities", styles["heading"])]
    for index, factor in enumerate(report.factors, start=1):
        block: list[Flowable] = [
            Paragraph(
                f"{index}. {factor.title} ({factor.confidence.value} confidence)",
                styles["subheading"],
            ),
            Paragraph(factor.explanation, styles["body"]),
        ]
        for rec in report.ranked_recommendations():
            if rec.linked_factor != factor.title:
                continue
            block.append(Spacer(1, 3))
            for label, text in plan_lines(rec):
                block.append(Paragraph(f"<b>{label}:</b> {text}", styles["body"]))
        block.append(Spacer(1, 10))
        section.append(KeepTogether(block))
    return section


def plan_lines(rec: Recommendation) -> list[tuple[str, str]]:
    """The labelled lines of one step, leaving out any that are empty."""
    lines = [
        ("First step", rec.action),
        ("Target", rec.target),
        ("How to track it", rec.how_to_track),
        ("Check again", rec.recheck),
    ]
    return [(label, text) for label, text in lines if text.strip()]


def _limitations_section(state: PipelineState, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    items: list[Flowable] = [
        Paragraph("Known limitations", styles["subheading"]),
        Paragraph(
            "The perceived-age model is a pretrained estimator whose error grows with age "
            "and is not uniform across groups. Biomarker context is a curated summary of "
            f"{len(load_biomarker_reference())} analytes, not a clinical reference. Results "
            "outside that summary are judged only against the range the laboratory printed. "
            "Lifestyle answers are self-reported and unverified.",
            styles["body"],
        ),
    ]

    warnings = state.get("warnings", [])
    if warnings:
        items += [
            Spacer(1, 8),
            Paragraph("Run notes", styles["subheading"]),
            *[Paragraph(f"- {warning}", styles["muted"]) for warning in warnings],
        ]
    return items


def _biomarker_chart(analytes: list[BloodAnalyte]) -> Drawing:
    shown = list(reversed(analytes[:MAX_CHART_BARS]))
    deviations = [deviation_percent(analyte) for analyte in shown]

    drawing = Drawing(430, 28 * len(shown) + 46)
    drawing.add(
        String(
            0,
            drawing.height - 12,
            "Deviation from reference range (%)",
            fontSize=8,
            fillColor=MUTED,
        )
    )

    chart = HorizontalBarChart()
    chart.x, chart.y = 110, 12
    chart.width, chart.height = 300, 28 * len(shown)
    chart.data = [deviations]
    chart.categoryAxis.categoryNames = [display_name(analyte) for analyte in shown]
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = min(0, min(deviations) * 1.35)
    chart.valueAxis.valueMax = max(10, max(deviations) * 1.35)
    chart.valueAxis.labels.fontSize = 7
    chart.barLabels.fontSize = 7
    chart.barLabelFormat = "%.0f%%"
    chart.barLabels.dx = 8
    chart.barLabels.boxAnchor = "w"
    chart.bars[0].fillColor = ACCENT
    drawing.add(chart)
    return drawing


def _biomarker_table(analytes: list[BloodAnalyte], cell: ParagraphStyle) -> Table:
    rows = [
        [
            Paragraph(display_name(analyte), cell),
            Paragraph(f"{analyte.value} {analyte.unit}", cell),
            Paragraph(_range_text(analyte), cell),
            Paragraph(analyte.flag.value, cell),
        ]
        for analyte in analytes
    ]
    out_of_range = [
        ("TEXTCOLOR", (0, row), (-1, row), WARNING)
        for row, analyte in enumerate(analytes, start=1)
        if analyte.flag in (RangeFlag.LOW, RangeFlag.HIGH)
    ]

    table = Table(
        [["Biomarker", "Result", "Reference range", "Flag"], *rows],
        colWidths=[150, 100, 130, 50],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                *out_of_range,
            ]
        )
    )
    return table


def _callout(text: str, style: ParagraphStyle, color: colors.Color) -> Table:
    table = Table([[Paragraph(text, style)]], colWidths=[6.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, color),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _range_text(analyte: BloodAnalyte) -> str:
    low, high = analyte.reference_range_low, analyte.reference_range_high
    if low is not None and high is not None:
        return f"{low} to {high} {analyte.unit}"
    if high is not None:
        return f"below {high} {analyte.unit}"
    if low is not None:
        return f"above {low} {analyte.unit}"
    return "not printed"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "title": ParagraphStyle("title", parent=base, fontSize=20, leading=24, textColor=ACCENT),
        "heading": ParagraphStyle(
            "heading",
            parent=base,
            fontSize=12,
            leading=15,
            spaceAfter=4,
            textColor=ACCENT,
            fontName="Helvetica-Bold",
        ),
        "subheading": ParagraphStyle(
            "subheading", parent=base, fontSize=10, leading=13, fontName="Helvetica-Bold"
        ),
        "body": ParagraphStyle("body", parent=base, fontSize=9.5, leading=13, alignment=TA_LEFT),
        "muted": ParagraphStyle("muted", parent=base, fontSize=8, leading=11, textColor=MUTED),
        "callout": ParagraphStyle("callout", parent=base, fontSize=8.5, leading=11.5),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8, leading=10),
    }
