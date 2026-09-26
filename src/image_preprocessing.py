from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

from src.config import (
    CONTRAST_NORMALIZED_STD,
    DETECTION_PAD_FRACTION,
    FACE_CROP_MARGIN,
    FACE_CROP_SIZE,
    FACE_DETECTOR_MIN_SCORE,
    FACE_DETECTOR_PATH,
    MAX_MEAN_BRIGHTNESS,
    MIN_MEAN_BRIGHTNESS,
    MIN_NORMALIZED_SHARPNESS,
)


class ImageQualityError(ValueError):
    pass


class FaceBox(NamedTuple):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]


def prepare_face(path: str | Path) -> tuple[np.ndarray, float]:
    """Detect, align, crop and quality-check the face in a photo."""
    padded = pad_for_detection(load_image(path))
    box = detect_face(padded)
    crop = align_and_crop(padded, box)
    assess_quality(crop)
    return crop, box.confidence


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageQualityError(f"Could not decode an image at {path}.")
    return image


def pad_for_detection(image: np.ndarray) -> np.ndarray:
    margin = int(min(image.shape[:2]) * DETECTION_PAD_FRACTION)
    return cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_REPLICATE)


def detect_face(image: np.ndarray) -> FaceBox:
    """Find exactly one face, and its eyes, with YuNet."""
    detector = _detector()
    with _DETECTOR_LOCK:
        detector.setInputSize((image.shape[1], image.shape[0]))
        _, found = detector.detect(image)
    faces = [] if found is None else [f for f in found if f[-1] >= FACE_DETECTOR_MIN_SCORE]
    if not faces:
        raise ImageQualityError("No face was detected. Use a clear, front-facing, well-lit photo.")
    if len(faces) > 1:
        raise ImageQualityError(f"{len(faces)} faces were detected. Submit a photo of one face.")

    face = faces[0]
    x, y, width, height = (float(value) for value in face[:4])
    # YuNet returns five landmarks; the first two are the eyes. Left and right are as seen.
    left_eye, right_eye = sorted(
        [(float(face[4]), float(face[5])), (float(face[6]), float(face[7]))]
    )
    return FaceBox(x, y, width, height, float(face[-1]), left_eye, right_eye)


def align_and_crop(image: np.ndarray, box: FaceBox) -> np.ndarray:
    """Level the eyes and crop the face to the model's input size."""
    (left_x, left_y), (right_x, right_y) = box.left_eye, box.right_eye
    angle = float(np.degrees(np.arctan2(right_y - left_y, right_x - left_x)))
    center = (box.x + box.width / 2.0, box.y + box.height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    margin_x = int(box.width * FACE_CROP_MARGIN)
    margin_y = int(box.height * FACE_CROP_MARGIN)
    left = max(int(box.x) - margin_x, 0)
    top = max(int(box.y) - margin_y, 0)
    right = min(int(box.x + box.width) + margin_x, image.shape[1])
    bottom = min(int(box.y + box.height) + margin_y, image.shape[0])

    crop = image[top:bottom, left:right]
    return cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=cv2.INTER_AREA)


def assess_quality(crop: np.ndarray) -> None:
    """Reject blurry, dark or overexposed crops."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = normalized_sharpness(gray)
    if sharpness < MIN_NORMALIZED_SHARPNESS:
        raise ImageQualityError(f"The photo is too blurry to score (sharpness {sharpness:.0f}).")

    brightness = float(gray.mean())
    if brightness < MIN_MEAN_BRIGHTNESS:
        raise ImageQualityError("The photo is too dark to score.")
    if brightness > MAX_MEAN_BRIGHTNESS:
        raise ImageQualityError("The photo is overexposed.")


def normalized_sharpness(gray: np.ndarray) -> float:
    """Measure blur after equalising contrast."""
    pixels = gray.astype(np.float64)
    spread = pixels.std()
    if spread == 0:
        return 0.0
    scaled = (pixels - pixels.mean()) / spread * CONTRAST_NORMALIZED_STD
    return float(cv2.Laplacian(scaled, cv2.CV_64F).var())


_DETECTOR_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _detector() -> Any:
    if not FACE_DETECTOR_PATH.exists():
        raise ImageQualityError(f"Face detector model missing at {FACE_DETECTOR_PATH}.")
    return cv2.FaceDetectorYN.create(str(FACE_DETECTOR_PATH), "", (320, 320))
