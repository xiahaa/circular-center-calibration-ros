# SPDX-License-Identifier: Apache-2.0
"""OpenCV-only ellipse detection followed by the released center estimator."""

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from .models import CameraModel


@dataclass(frozen=True)
class ImageDetectorConfig:
    marker_diameter: float = 0.24
    expected_count: int = 4
    grid_columns: int = 2
    threshold_mode: str = "otsu_dark"
    adaptive_block_size: int = 31
    adaptive_constant: float = 5.0
    min_contour_area: float = 100.0
    max_contour_area: float = 100000.0
    min_axis_ratio: float = 0.20
    duplicate_center_distance: float = 8.0

    def __post_init__(self):
        if self.marker_diameter <= 0.0:
            raise ValueError("marker_diameter must be positive")
        if self.expected_count <= 0 or self.grid_columns <= 0:
            raise ValueError("expected_count and grid_columns must be positive")
        if self.threshold_mode not in ("otsu_dark", "adaptive_dark"):
            raise ValueError("unsupported threshold_mode")
        if self.adaptive_block_size < 3:
            raise ValueError("adaptive_block_size must be at least three")


@dataclass(frozen=True)
class EllipseDetection:
    contour: np.ndarray
    ellipse_cv: tuple
    fit_error: float

    @property
    def center(self):
        return np.asarray(self.ellipse_cv[0], dtype=float)

    @property
    def semi_axes(self):
        return 0.5 * np.asarray(self.ellipse_cv[1], dtype=float)


@dataclass(frozen=True)
class ProjectedCenterDetection:
    id: int
    primary: np.ndarray
    alternative: np.ndarray
    scores: np.ndarray
    confidence: float
    ellipse: np.ndarray


def _rectify_image(image: np.ndarray, camera: CameraModel) -> np.ndarray:
    if camera.distortion.size == 0 or np.allclose(camera.distortion, 0.0):
        return image.copy()
    size = (image.shape[1], image.shape[0])
    if camera.distortion_model == "pinhole":
        return cv2.undistort(
            image,
            camera.matrix,
            camera.distortion,
            None,
            camera.matrix,
        )
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        camera.matrix,
        camera.distortion.reshape(4, 1),
        np.eye(3),
        camera.matrix,
        size,
        cv2.CV_32FC1,
    )
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def _threshold(gray: np.ndarray, config: ImageDetectorConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
    if config.threshold_mode == "otsu_dark":
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        return binary
    block_size = config.adaptive_block_size
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        config.adaptive_constant,
    )


def _ellipse_fit_error(contour: np.ndarray, ellipse_cv: tuple) -> float:
    center = np.asarray(ellipse_cv[0], dtype=float)
    semi_axes = np.maximum(0.5 * np.asarray(ellipse_cv[1], dtype=float), 1e-9)
    angle = np.deg2rad(float(ellipse_cv[2]))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    local = (contour.reshape(-1, 2) - center) @ rotation
    radial = np.sum(np.square(local / semi_axes), axis=1)
    return float(np.sqrt(np.mean(np.square(radial - 1.0))))


def find_ellipse_contours(
    image: np.ndarray, config: ImageDetectorConfig
) -> List[EllipseDetection]:
    if image is None or image.ndim not in (2, 3):
        raise ValueError("image must be a non-empty grayscale or color array")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = _threshold(gray, config)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    detections = []
    for contour in contours:
        if len(contour) < 20:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if not config.min_contour_area <= area <= config.max_contour_area:
            continue
        ellipse_cv = cv2.fitEllipse(contour)
        axes = np.asarray(ellipse_cv[1], dtype=float)
        if np.min(axes) <= 0.0 or np.min(axes) / np.max(axes) < config.min_axis_ratio:
            continue
        detections.append(
            EllipseDetection(
                contour=contour.reshape(-1, 2).astype(float),
                ellipse_cv=ellipse_cv,
                fit_error=_ellipse_fit_error(contour, ellipse_cv),
            )
        )

    detections.sort(key=lambda item: (item.fit_error, -len(item.contour)))
    unique = []
    for detection in detections:
        if all(
            np.linalg.norm(detection.center - previous.center)
            >= config.duplicate_center_distance
            for previous in unique
        ):
            unique.append(detection)
    return unique[: config.expected_count]


def _row_major_order(centers: Sequence[np.ndarray], columns: int) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    order_y = np.argsort(centers[:, 1], kind="stable")
    ordered = []
    for offset in range(0, len(order_y), columns):
        row = order_y[offset : offset + columns]
        ordered.extend(row[np.argsort(centers[row, 0], kind="stable")].tolist())
    return np.asarray(ordered, dtype=int)


def detect_projected_centers(
    image: np.ndarray,
    camera: CameraModel,
    config: ImageDetectorConfig,
):
    """Return center candidates and the rectified image used to obtain them."""
    try:
        from circular_center.center2d import (
            get_ellipse_polynomial_coeff,
            refine_projected_center,
        )
    except ImportError as error:
        raise RuntimeError(
            "install circular-center-calibration from requirements.txt"
        ) from error

    rectified = _rectify_image(image, camera)
    ellipses = find_ellipse_contours(rectified, config)
    if len(ellipses) != config.expected_count:
        raise RuntimeError(
            f"expected {config.expected_count} circular contours, found {len(ellipses)}"
        )
    order = _row_major_order([ellipse.center for ellipse in ellipses], config.grid_columns)
    results = []
    for circle_id, detection_index in enumerate(order):
        detection = ellipses[int(detection_index)]
        (cx, cy), (width, height), angle_degrees = detection.ellipse_cv
        ellipse = np.array(
            [cx, cy, 0.5 * width, 0.5 * height, np.deg2rad(angle_degrees)],
            dtype=float,
        )
        polynomial = get_ellipse_polynomial_coeff(detection.ellipse_cv)
        refinement = refine_projected_center(
            ellipse,
            polynomial,
            camera.matrix,
            config.marker_diameter,
            input_is_rectified=True,
        )
        results.append(
            ProjectedCenterDetection(
                id=circle_id,
                primary=refinement.candidates[0],
                alternative=refinement.candidates[1],
                scores=refinement.scores,
                confidence=refinement.confidence,
                ellipse=ellipse,
            )
        )
    return results, rectified


__all__ = [
    "EllipseDetection",
    "ImageDetectorConfig",
    "ProjectedCenterDetection",
    "detect_projected_centers",
    "find_ellipse_contours",
]
