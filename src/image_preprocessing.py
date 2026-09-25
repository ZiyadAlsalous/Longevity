from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from src.config import (
    CONTRAST_NORMALIZED_STD,
    DETECTION_PAD_FRACTION,
    FACE_CROP_MARGIN,
    FACE_CROP_SIZE,
    HAAR_EYE_MIN_NEIGHBORS,
    HAAR_MIN_NEIGHBORS,
    HAAR_SCALE_FACTOR,
    MAX_MEAN_BRIGHTNESS,
    MIN_FACE_PIXELS,
    MIN_MEAN_BRIGHTNESS,
    MIN_NORMALIZED_SHARPNESS,
)


class ImageQualityError(ValueError):
    pass


class FaceBox(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    confidence: float


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
    """Find exactly one face with the Haar cascade."""
    cascade = _cascade("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections, _, weights = cascade.detectMultiScale3(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(MIN_FACE_PIXELS, MIN_FACE_PIXELS),
        outputRejectLevels=True,
    )
    if len(detections) == 0:
        raise ImageQualityError("No face was detected. Use a clear, front-facing, well-lit photo.")
    if len(detections) > 1:
        raise ImageQualityError(
            f"{len(detections)} faces were detected. Submit a photo of one face."
        )

    x, y, width, height = (int(value) for value in detections[0])
    score = float(weights[0]) if len(weights) else 0.0
    return FaceBox(x, y, width, height, float(1.0 / (1.0 + np.exp(-score))))


def align_and_crop(image: np.ndarray, box: FaceBox) -> np.ndarray:
    """Level the eyes and crop the face to the model's input size."""
    angle = _eye_angle(image, box)
    if angle is not None:
        center = (box.x + box.width / 2.0, box.y + box.height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
        )

    margin_x = int(box.width * FACE_CROP_MARGIN)
    margin_y = int(box.height * FACE_CROP_MARGIN)
    left = max(box.x - margin_x, 0)
    top = max(box.y - margin_y, 0)
    right = min(box.x + box.width + margin_x, image.shape[1])
    bottom = min(box.y + box.height + margin_y, image.shape[0])

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


def _cascade(filename: str) -> cv2.CascadeClassifier:
    cascade_dir = Path(str(cv2.data.haarcascades))  # type: ignore[attr-defined]
    classifier = cv2.CascadeClassifier(str(cascade_dir / filename))
    if classifier.empty():
        raise ImageQualityError(f"OpenCV cascade {filename} could not be loaded.")
    return classifier


def _eye_angle(image: np.ndarray, box: FaceBox) -> float | None:
    region = cv2.cvtColor(
        image[box.y : box.y + box.height, box.x : box.x + box.width], cv2.COLOR_BGR2GRAY
    )
    eyes = _cascade("haarcascade_eye.xml").detectMultiScale(
        region, HAAR_SCALE_FACTOR, HAAR_EYE_MIN_NEIGHBORS
    )
    if len(eyes) < 2:
        return None

    largest = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]
    centers = sorted(
        ((int(x + w / 2), int(y + h / 2)) for x, y, w, h in largest), key=lambda point: point[0]
    )
    (left_x, left_y), (right_x, right_y) = centers
    return float(np.degrees(np.arctan2(right_y - left_y, right_x - left_x)))
