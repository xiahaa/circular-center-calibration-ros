# SPDX-License-Identifier: Apache-2.0
"""Planar-board circular-hole detection without FAST-Calib source code."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LidarDetectorConfig:
    roi_min: np.ndarray
    roi_max: np.ndarray
    expected_count: int = 4
    grid_columns: int = 2
    plane_distance_threshold: float = 0.015
    plane_ransac_iterations: int = 500
    occupancy_resolution: float = 0.01
    occupancy_dilation_cells: int = 1
    minimum_hole_radius: float = 0.06
    maximum_hole_radius: float = 0.18
    annulus_half_width: float = 0.025
    circle_residual_threshold: float = 0.015
    random_seed: int = 2025

    def __post_init__(self):
        roi_min = np.asarray(self.roi_min, dtype=float).reshape(3)
        roi_max = np.asarray(self.roi_max, dtype=float).reshape(3)
        if not np.isfinite(roi_min).all() or not np.isfinite(roi_max).all():
            raise ValueError("ROI bounds must be finite")
        if np.any(roi_min >= roi_max):
            raise ValueError("each ROI minimum must be smaller than its maximum")
        if self.expected_count <= 0 or self.grid_columns <= 0:
            raise ValueError("expected_count and grid_columns must be positive")
        positive = (
            self.plane_distance_threshold,
            self.occupancy_resolution,
            self.minimum_hole_radius,
            self.maximum_hole_radius,
            self.annulus_half_width,
            self.circle_residual_threshold,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("detector distances and radii must be positive")
        if self.minimum_hole_radius >= self.maximum_hole_radius:
            raise ValueError("minimum_hole_radius must be smaller than maximum_hole_radius")
        object.__setattr__(self, "roi_min", roi_min)
        object.__setattr__(self, "roi_max", roi_max)


@dataclass(frozen=True)
class PlaneModel:
    origin: np.ndarray
    normal: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    inlier_mask: np.ndarray


@dataclass(frozen=True)
class DetectedCircle3D:
    id: int
    center: np.ndarray
    normal: np.ndarray
    radius: float
    rmse: float
    inlier_count: int


def _canonical_normal(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal)
    index = int(np.argmax(np.abs(normal)))
    return -normal if normal[index] < 0.0 else normal


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ normal)))]
    basis_u = np.cross(normal, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    return basis_u, basis_v


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> PlaneModel:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must have shape (N, 3), N >= 3")
    generator = np.random.default_rng(seed)
    best_mask = None
    best_score = None
    for _ in range(iterations):
        sample = points[generator.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length <= 1e-10:
            continue
        normal /= length
        distances = np.abs((points - sample[0]) @ normal)
        mask = distances <= threshold
        count = int(np.count_nonzero(mask))
        if count < 3:
            continue
        score = (count, -float(np.median(distances[mask])))
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None:
        raise RuntimeError("plane RANSAC found no consensus set")

    inlier_points = points[best_mask]
    origin = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - origin, full_matrices=False)
    normal = _canonical_normal(vh[-1])
    distances = np.abs((points - origin) @ normal)
    mask = distances <= threshold
    origin = np.mean(points[mask], axis=0)
    basis_u, basis_v = _plane_basis(normal)
    return PlaneModel(origin, normal, basis_u, basis_v, mask)


def _row_major_order(centers_2d: np.ndarray, columns: int) -> np.ndarray:
    order_y = np.argsort(centers_2d[:, 1], kind="stable")
    ordered = []
    for offset in range(0, len(order_y), columns):
        row = order_y[offset : offset + columns]
        ordered.extend(row[np.argsort(centers_2d[row, 0], kind="stable")].tolist())
    return np.asarray(ordered, dtype=int)


def _hole_seeds(
    coordinates: np.ndarray,
    config: LidarDetectorConfig,
):
    resolution = config.occupancy_resolution
    lower = np.min(coordinates, axis=0) - resolution
    upper = np.max(coordinates, axis=0) + resolution
    shape_xy = np.ceil((upper - lower) / resolution).astype(int) + 1
    if np.any(shape_xy > 6000) or int(np.prod(shape_xy)) > 20_000_000:
        raise RuntimeError("occupancy grid is too large; tighten the ROI or increase resolution")
    cells = np.floor((coordinates - lower) / resolution).astype(int)
    cells = np.clip(cells, 0, shape_xy - 1)
    occupancy = np.zeros((shape_xy[1], shape_xy[0]), dtype=np.uint8)
    occupancy[cells[:, 1], cells[:, 0]] = 255
    dilation = int(config.occupancy_dilation_cells)
    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
        )
        occupancy = cv2.dilate(occupancy, kernel)
    empty = cv2.bitwise_not(occupancy)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(empty, 8)
    candidates = []
    height, width = empty.shape
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if x == 0 or y == 0 or x + component_width >= width or y + component_height >= height:
            continue
        radius = np.sqrt(float(area) / np.pi) * resolution + dilation * resolution
        if config.minimum_hole_radius <= radius <= config.maximum_hole_radius:
            uv = lower + resolution * centroids[label]
            target_radius = 0.5 * (
                config.minimum_hole_radius + config.maximum_hole_radius
            )
            candidates.append((abs(radius - target_radius), uv, radius, int(area)))
    candidates.sort(key=lambda value: (value[0], -value[3]))
    return candidates[: config.expected_count]


def detect_circular_holes(
    points: np.ndarray, config: LidarDetectorConfig
) -> Tuple[List[DetectedCircle3D], PlaneModel]:
    try:
        from circular_center.center3d import fit_circle_ransac
    except ImportError as error:
        raise RuntimeError(
            "install circular-center-calibration from requirements.txt"
        ) from error

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    roi_mask = np.all((points >= config.roi_min) & (points <= config.roi_max), axis=1)
    filtered = points[roi_mask]
    if len(filtered) < 50:
        raise RuntimeError("fewer than 50 points remain after ROI filtering")
    plane = fit_plane_ransac(
        filtered,
        config.plane_distance_threshold,
        config.plane_ransac_iterations,
        config.random_seed,
    )
    plane_points = filtered[plane.inlier_mask]
    offsets = plane_points - plane.origin
    coordinates = np.column_stack((offsets @ plane.basis_u, offsets @ plane.basis_v))
    seeds = _hole_seeds(coordinates, config)
    if len(seeds) != config.expected_count:
        raise RuntimeError(
            f"expected {config.expected_count} circular holes, found {len(seeds)}"
        )

    raw_results = []
    for _, seed, estimated_radius, _ in seeds:
        radial = np.linalg.norm(coordinates - seed, axis=1)
        annulus = np.abs(radial - estimated_radius) <= config.annulus_half_width
        boundary = plane_points[annulus]
        if len(boundary) < 8:
            raise RuntimeError("not enough points around a detected hole boundary")
        fit = fit_circle_ransac(
            boundary,
            residual_threshold=config.circle_residual_threshold,
            max_iterations=500,
            sample_size=5,
            minimum_inliers=max(5, int(0.35 * len(boundary))),
            seed=config.random_seed,
        )
        rmse = float(np.sqrt(np.mean(np.square(fit.residuals[fit.inlier_mask]))))
        center_offset = fit.center - plane.origin
        center_2d = np.array(
            [center_offset @ plane.basis_u, center_offset @ plane.basis_v]
        )
        raw_results.append((fit, rmse, center_2d))

    order = _row_major_order(
        np.asarray([result[2] for result in raw_results]), config.grid_columns
    )
    detections = []
    for circle_id, index in enumerate(order):
        fit, rmse, _ = raw_results[int(index)]
        detections.append(
            DetectedCircle3D(
                id=circle_id,
                center=fit.center,
                normal=fit.normal,
                radius=float(fit.radius),
                rmse=rmse,
                inlier_count=int(np.count_nonzero(fit.inlier_mask)),
            )
        )
    return detections, plane


__all__ = [
    "DetectedCircle3D",
    "LidarDetectorConfig",
    "PlaneModel",
    "detect_circular_holes",
    "fit_plane_ransac",
]
