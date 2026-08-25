# SPDX-License-Identifier: Apache-2.0
"""Deterministic PnP solver for two projected-center candidates per circle."""

from dataclasses import dataclass
from itertools import permutations, product
from typing import Optional, Tuple

import cv2
import numpy as np

from .models import CalibrationProblem, CalibrationResult


@dataclass(frozen=True)
class SolverConfig:
    exhaustive_candidate_limit: int = 12
    quasi_ransac_iterations: int = 1000
    reprojection_inlier_threshold: float = 3.0
    minimum_inliers: int = 4
    random_seed: int = 2025
    resolve_correspondences: bool = False
    maximum_permutation_size: int = 6

    def __post_init__(self):
        if self.exhaustive_candidate_limit < 4:
            raise ValueError("exhaustive_candidate_limit must be at least four")
        if self.quasi_ransac_iterations <= 0:
            raise ValueError("quasi_ransac_iterations must be positive")
        if self.reprojection_inlier_threshold <= 0.0:
            raise ValueError("reprojection_inlier_threshold must be positive")
        if self.minimum_inliers < 4:
            raise ValueError("minimum_inliers must be at least four")
        if self.maximum_permutation_size < 4:
            raise ValueError("maximum_permutation_size must be at least four")


class CandidatePnPSolver:
    """Estimate ``T_camera_lidar`` while resolving each binary 2D ambiguity."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

    @staticmethod
    def _is_planar(points_3d: np.ndarray) -> bool:
        centered = points_3d - np.mean(points_3d, axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values[0] <= np.finfo(float).eps:
            return True
        return bool(singular_values[-1] / singular_values[0] < 1e-5)

    @classmethod
    def _solve_pose(
        cls,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        initial: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(points_3d) < 4:
            return None
        object_points = np.ascontiguousarray(points_3d, dtype=np.float64)
        image_points = np.ascontiguousarray(points_2d, dtype=np.float64)
        distortion = np.ascontiguousarray(distortion, dtype=np.float64)
        try:
            if initial is None:
                flag = (
                    cv2.SOLVEPNP_IPPE
                    if cls._is_planar(object_points) and hasattr(cv2, "SOLVEPNP_IPPE")
                    else cv2.SOLVEPNP_EPNP
                )
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=flag,
                )
            else:
                rvec, tvec = initial
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                    tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
        except cv2.error:
            return None
        if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        depths = (rotation @ object_points.T + tvec.reshape(3, 1))[2]
        if np.count_nonzero(depths > 0.0) < max(4, int(0.75 * len(depths))):
            return None
        return rvec.reshape(3), tvec.reshape(3)

    @staticmethod
    def _project(
        points_3d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> np.ndarray:
        projected, _ = cv2.projectPoints(
            np.ascontiguousarray(points_3d, dtype=np.float64),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            camera_matrix,
            distortion,
        )
        return projected.reshape(-1, 2)

    def _nearest_candidate_score(
        self,
        problem: CalibrationProblem,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ):
        projected = self._project(
            problem.points_3d,
            rvec,
            tvec,
            problem.camera.matrix,
            problem.camera.distortion,
        )
        selection = np.empty(len(projected), dtype=np.uint8)
        correspondence = np.empty(len(projected), dtype=np.uint32)
        errors = np.empty(len(projected), dtype=float)
        start = 0
        for size in problem.frame_sizes:
            stop = start + size
            frame_projected = projected[start:stop]
            frame_candidates = problem.candidates_2d[start:stop]
            pair_distances = np.linalg.norm(
                frame_projected[:, None, None, :] - frame_candidates[None, :, :, :],
                axis=3,
            )
            if self.config.resolve_correspondences:
                if size > self.config.maximum_permutation_size:
                    raise RuntimeError(
                        "a frame exceeds maximum_permutation_size; provide explicit IDs "
                        "or increase the configured limit"
                    )
                best_mapping = min(
                    permutations(range(size)),
                    key=lambda mapping: float(
                        np.sum(
                            np.min(
                                pair_distances[np.arange(size), np.asarray(mapping)],
                                axis=1,
                            )
                        )
                    ),
                )
                mapping = np.asarray(best_mapping, dtype=int)
            else:
                mapping = np.arange(size)
            matched = pair_distances[np.arange(size), mapping]
            frame_selection = np.argmin(matched, axis=1).astype(np.uint8)
            selection[start:stop] = frame_selection
            correspondence[start:stop] = mapping.astype(np.uint32)
            errors[start:stop] = matched[np.arange(size), frame_selection]
            start = stop
        inliers = errors <= self.config.reprojection_inlier_threshold
        inlier_count = int(np.count_nonzero(inliers))
        rmse = (
            float(np.sqrt(np.mean(np.square(errors[inliers]))))
            if inlier_count
            else float("inf")
        )
        return inlier_count, rmse, selection, correspondence, inliers, projected

    @staticmethod
    def _selected_image_points(problem, selection, correspondence):
        selected = np.empty((len(selection), 2), dtype=float)
        start = 0
        for size in problem.frame_sizes:
            stop = start + size
            rows = start + np.asarray(correspondence[start:stop], dtype=int)
            selected[start:stop] = problem.candidates_2d[
                rows, selection[start:stop]
            ]
            start = stop
        return selected

    def _refine_hypothesis(
        self,
        problem: CalibrationProblem,
        pose: Tuple[np.ndarray, np.ndarray],
    ):
        rvec, tvec = pose
        count, _, selection, correspondence, inliers, _ = self._nearest_candidate_score(
            problem, rvec, tvec
        )
        if count < self.config.minimum_inliers:
            return None
        selected = self._selected_image_points(problem, selection, correspondence)
        refined = self._solve_pose(
            problem.points_3d[inliers],
            selected[inliers],
            problem.camera.matrix,
            problem.camera.distortion,
            initial=(rvec, tvec),
        )
        if refined is None:
            refined = pose
        return (*refined, *self._nearest_candidate_score(problem, *refined))

    def _solve_exhaustive(self, problem: CalibrationProblem):
        best = None
        iterations = 0
        indices = np.arange(len(problem.points_3d))
        mappings = (
            permutations(range(len(indices)))
            if self.config.resolve_correspondences
            else [tuple(range(len(indices)))]
        )
        for mapping_tuple in mappings:
            mapping = np.asarray(mapping_tuple, dtype=int)
            for bits in product((0, 1), repeat=len(indices)):
                iterations += 1
                selection = np.fromiter(bits, dtype=np.uint8)
                image_points = problem.candidates_2d[mapping, selection]
                pose = self._solve_pose(
                    problem.points_3d,
                    image_points,
                    problem.camera.matrix,
                    problem.camera.distortion,
                )
                if pose is None:
                    continue
                refined = self._refine_hypothesis(problem, pose)
                if refined is None:
                    continue
                (
                    rvec,
                    tvec,
                    count,
                    rmse,
                    chosen,
                    correspondence,
                    inliers,
                    projected,
                ) = refined
                score = (count, -rmse)
                if best is None or score > best[0]:
                    best = (
                        score,
                        rvec,
                        tvec,
                        rmse,
                        chosen,
                        correspondence,
                        inliers,
                        projected,
                    )
        return best, iterations

    def _solve_quasi_ransac(self, problem: CalibrationProblem):
        generator = np.random.default_rng(self.config.random_seed)
        best = None
        sample_size = 4
        all_indices = np.arange(len(problem.points_3d))
        frame_offsets = np.cumsum([0] + problem.frame_sizes)
        for _ in range(self.config.quasi_ransac_iterations):
            if self.config.resolve_correspondences:
                valid_frames = [
                    index
                    for index, size in enumerate(problem.frame_sizes)
                    if size >= sample_size
                ]
                frame_index = int(generator.choice(valid_frames))
                start, stop = frame_offsets[frame_index : frame_index + 2]
                sample = generator.choice(np.arange(start, stop), sample_size, replace=False)
                candidate_rows = generator.choice(
                    np.arange(start, stop), sample_size, replace=False
                )
            else:
                sample = generator.choice(all_indices, sample_size, replace=False)
                candidate_rows = sample
            choices = generator.integers(0, 2, size=sample_size, dtype=np.uint8)
            image_points = problem.candidates_2d[candidate_rows, choices]
            pose = self._solve_pose(
                problem.points_3d[sample],
                image_points,
                problem.camera.matrix,
                problem.camera.distortion,
            )
            if pose is None:
                continue
            refined = self._refine_hypothesis(problem, pose)
            if refined is None:
                continue
            (
                rvec,
                tvec,
                count,
                rmse,
                chosen,
                correspondence,
                inliers,
                projected,
            ) = refined
            score = (count, -rmse)
            if best is None or score > best[0]:
                best = (
                    score,
                    rvec,
                    tvec,
                    rmse,
                    chosen,
                    correspondence,
                    inliers,
                    projected,
                )
                if count == len(all_indices) and rmse < 0.05:
                    break
        return best, self.config.quasi_ransac_iterations

    def solve(self, problem: CalibrationProblem) -> CalibrationResult:
        can_exhaust = (
            len(problem.frame_sizes) == 1
            and len(problem.points_3d) <= self.config.exhaustive_candidate_limit
            and (
                not self.config.resolve_correspondences
                or len(problem.points_3d) <= self.config.maximum_permutation_size
            )
        )
        if can_exhaust:
            best, iterations = self._solve_exhaustive(problem)
            method = "exhaustive"
        else:
            best, iterations = self._solve_quasi_ransac(problem)
            method = "quasi_ransac"
        if best is None:
            raise RuntimeError("no valid positive-depth PnP hypothesis was found")

        _, rvec, tvec, rmse, selection, correspondence, inliers, projected = best
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec
        return CalibrationResult(
            success=True,
            status=method,
            transform_camera_lidar=transform,
            rotation_vector=rvec,
            translation_vector=tvec,
            reprojection_rmse_pixels=rmse,
            selected_candidates=selection,
            correspondence_indices=correspondence,
            inlier_mask=inliers,
            projected_points=projected,
            iterations=iterations,
        )


__all__ = ["CandidatePnPSolver", "SolverConfig"]
