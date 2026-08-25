# SPDX-License-Identifier: Apache-2.0
"""Validated data structures shared by online and offline calibration."""

from dataclasses import dataclass, field
from typing import List

import numpy as np


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


@dataclass(frozen=True)
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    distortion_model: str = "pinhole"

    def __post_init__(self):
        matrix = _finite_array(self.matrix, (3, 3), "camera matrix")
        distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if not np.isfinite(distortion).all():
            raise ValueError("distortion coefficients must be finite")
        if self.distortion_model not in ("pinhole", "fisheye"):
            raise ValueError("distortion_model must be 'pinhole' or 'fisheye'")
        if self.distortion_model == "fisheye" and distortion.size != 4:
            raise ValueError("fisheye distortion requires four coefficients")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "distortion", distortion)


@dataclass(frozen=True)
class CalibrationProblem:
    points_3d: np.ndarray
    candidates_2d: np.ndarray
    camera: CameraModel
    labels: List[str] = field(default_factory=list)
    frame_sizes: List[int] = field(default_factory=list)

    def __post_init__(self):
        points = np.asarray(self.points_3d, dtype=float)
        candidates = np.asarray(self.candidates_2d, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_3d must have shape (N, 3)")
        if candidates.shape != (points.shape[0], 2, 2):
            raise ValueError("candidates_2d must have shape (N, 2, 2)")
        if points.shape[0] < 4:
            raise ValueError("at least four correspondences are required")
        if not np.isfinite(points).all() or not np.isfinite(candidates).all():
            raise ValueError("calibration correspondences must be finite")
        labels = list(self.labels)
        if labels and len(labels) != points.shape[0]:
            raise ValueError("labels must be empty or match the correspondence count")
        frame_sizes = list(self.frame_sizes) if self.frame_sizes else [points.shape[0]]
        if any(int(size) < 4 for size in frame_sizes):
            raise ValueError("every calibration frame must contain at least four circles")
        if sum(int(size) for size in frame_sizes) != points.shape[0]:
            raise ValueError("frame_sizes must sum to the correspondence count")
        object.__setattr__(self, "points_3d", points)
        object.__setattr__(self, "candidates_2d", candidates)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "frame_sizes", [int(size) for size in frame_sizes])


@dataclass(frozen=True)
class CalibrationResult:
    success: bool
    status: str
    transform_camera_lidar: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    reprojection_rmse_pixels: float
    selected_candidates: np.ndarray
    correspondence_indices: np.ndarray
    inlier_mask: np.ndarray
    projected_points: np.ndarray
    iterations: int

    def __post_init__(self):
        transform = _finite_array(
            self.transform_camera_lidar, (4, 4), "transform_camera_lidar"
        )
        rotation = _finite_array(
            np.asarray(self.rotation_vector).reshape(3), (3,), "rotation_vector"
        )
        translation = _finite_array(
            np.asarray(self.translation_vector).reshape(3), (3,), "translation_vector"
        )
        selection = np.asarray(self.selected_candidates, dtype=np.uint8).reshape(-1)
        correspondence = np.asarray(
            self.correspondence_indices, dtype=np.uint32
        ).reshape(-1)
        inliers = np.asarray(self.inlier_mask, dtype=bool).reshape(-1)
        projected = np.asarray(self.projected_points, dtype=float)
        if projected.shape != (selection.size, 2):
            raise ValueError("projected_points must have shape (N, 2)")
        if inliers.size != selection.size:
            raise ValueError("inlier_mask must match selected_candidates")
        if correspondence.size != selection.size:
            raise ValueError("correspondence_indices must match selected_candidates")
        if not np.isfinite(projected).all():
            raise ValueError("projected_points must be finite")
        object.__setattr__(self, "transform_camera_lidar", transform)
        object.__setattr__(self, "rotation_vector", rotation)
        object.__setattr__(self, "translation_vector", translation)
        object.__setattr__(self, "selected_candidates", selection)
        object.__setattr__(self, "correspondence_indices", correspondence)
        object.__setattr__(self, "inlier_mask", inliers)
        object.__setattr__(self, "projected_points", projected)
