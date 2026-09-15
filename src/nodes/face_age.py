from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from src.config import (
    AGE_MODEL_ID,
    AGE_MODEL_REVISION,
    APPARENT_AGE_RANGE_BY_PREDICTION,
    MAX_PREDICTED_AGE,
    MIN_PREDICTED_AGE,
    settings,
)
from src.image_preprocessing import ImageQualityError, prepare_face
from src.schemas import (
    ApparentAgeEstimate,
    FaceAgeSignal,
    PipelineState,
    Questionnaire,
    VisionCheck,
)


def face_age_node(state: PipelineState) -> dict[str, Any]:
    image_path = state.get("image_path")
    if not image_path:
        return {}
    return estimate_apparent_age(image_path)


def estimate_apparent_age(image_path: str) -> dict[str, Any]:
    # Takes only the photo path, so the estimate cannot be influenced by any intake answer.
    # The stated age is compared against it later, in compare_with_stated_age.
    try:
        crop, confidence = prepare_face(image_path)
    except ImageQualityError as exc:
        return {
            "vision_check": VisionCheck(scored=False, reason_not_scored=str(exc)),
            "warnings": [f"Photo not scored: {exc}"],
        }

    try:
        (apparent_age,) = predict_apparent_ages([crop])
    except Exception as exc:  # noqa: BLE001
        # The vision extra is optional and runs remote model code, so import, download,
        # and torch device errors all mean "no vision signal", never a failed run.
        return {
            "vision_check": VisionCheck(
                scored=False, reason_not_scored=f"Vision model unavailable: {exc}"
            ),
            "warnings": [f"Vision model unavailable: {exc}"],
        }

    estimate = ApparentAgeEstimate(
        apparent_age=round(apparent_age, 1),
        interval_low=round(apparent_age - range_half_width(apparent_age), 1),
        interval_high=round(apparent_age + range_half_width(apparent_age), 1),
        face_confidence=round(confidence, 3),
    )
    check = VisionCheck(scored=True, face_confidence=estimate.face_confidence)
    return {"apparent_age": estimate, "vision_check": check}


def range_half_width(apparent_age: float) -> float:
    for upper, half_width in APPARENT_AGE_RANGE_BY_PREDICTION:
        if apparent_age < upper:
            return half_width
    return APPARENT_AGE_RANGE_BY_PREDICTION[-1][1]


def compare_with_stated_age(
    estimate: ApparentAgeEstimate, questionnaire: Questionnaire
) -> tuple[FaceAgeSignal, list[str]]:
    signal = FaceAgeSignal(
        apparent_age=estimate.apparent_age,
        chronological_age=questionnaire.chronological_age,
        age_gap_years=round(estimate.apparent_age - questionnaire.chronological_age, 1),
        interval_low=estimate.interval_low,
        interval_high=estimate.interval_high,
        face_confidence=estimate.face_confidence,
    )
    warnings: list[str] = []
    if abs(signal.age_gap_years) < range_half_width(signal.apparent_age):
        warnings.append(
            "Apparent age is within the model's usual error of the stated age; "
            "the gap is not notable."
        )
    return signal, warnings


def predict_apparent_ages(crops: list[np.ndarray]) -> list[float]:
    import torch

    model, processor = _load_age_model()
    faces = processor(images=crops)["pixel_values"].to(model.device)
    bodies = processor(images=[None] * len(crops))["pixel_values"].to(model.device)

    with torch.no_grad():
        ages = model(faces_input=faces, body_input=bodies).age_output.reshape(-1).cpu().tolist()
    return [float(np.clip(age, MIN_PREDICTED_AGE, MAX_PREDICTED_AGE)) for age in ages]


@lru_cache(maxsize=1)
def _load_age_model() -> tuple[Any, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(
        AGE_MODEL_ID,
        revision=AGE_MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    processor = AutoImageProcessor.from_pretrained(
        AGE_MODEL_ID, revision=AGE_MODEL_REVISION, trust_remote_code=True
    )
    return model.eval().to(_resolve_device()), processor


def _resolve_device() -> str:
    import torch

    if settings.torch_device != "auto":
        return settings.torch_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
