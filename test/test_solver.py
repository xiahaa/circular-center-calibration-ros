# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

from circular_center_calibration_ros.models import CalibrationProblem, CameraModel
from circular_center_calibration_ros.solver import CandidatePnPSolver, SolverConfig


def _problem(point_count=6, seed=9):
    generator = np.random.default_rng(seed)
    if point_count == 6:
        points = np.array(
            [
                [-0.35, -0.20, 0.0],
                [0.00, -0.20, 0.0],
                [0.35, -0.20, 0.0],
                [-0.35, 0.20, 0.0],
                [0.00, 0.20, 0.0],
                [0.35, 0.20, 0.0],
            ],
            dtype=float,
        )
    else:
        points = generator.uniform([-0.5, -0.4, -0.2], [0.5, 0.4, 0.2], (point_count, 3))
    camera_matrix = np.array(
        [[820.0, 0.0, 640.0], [0.0, 815.0, 360.0], [0.0, 0.0, 1.0]]
    )
    true_rvec = np.array([0.18, -0.11, 0.07])
    true_tvec = np.array([0.10, -0.04, 2.2])
    projected, _ = cv2.projectPoints(
        points, true_rvec, true_tvec, camera_matrix, np.zeros(5)
    )
    projected = projected.reshape(-1, 2)
    alternatives = projected + generator.normal([26.0, -19.0], [2.0, 2.0], projected.shape)
    correct_side = np.arange(point_count) % 2
    candidates = np.empty((point_count, 2, 2), dtype=float)
    for index, side in enumerate(correct_side):
        candidates[index, side] = projected[index]
        candidates[index, 1 - side] = alternatives[index]
    return (
        CalibrationProblem(points, candidates, CameraModel(camera_matrix, np.zeros(5))),
        true_rvec,
        true_tvec,
        correct_side,
    )


def _rotation_error(first_rvec, second_rvec):
    first, _ = cv2.Rodrigues(first_rvec)
    second, _ = cv2.Rodrigues(second_rvec)
    relative = first @ second.T
    return np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))


def test_exhaustive_solver_recovers_planar_pose_and_candidate_assignment():
    problem, true_rvec, true_tvec, correct_side = _problem()
    result = CandidatePnPSolver().solve(problem)
    assert result.status == "exhaustive"
    assert result.reprojection_rmse_pixels < 1e-4
    assert np.array_equal(result.selected_candidates, correct_side)
    assert _rotation_error(result.rotation_vector, true_rvec) < 1e-5
    assert np.linalg.norm(result.translation_vector - true_tvec) < 1e-5


def test_quasi_ransac_solver_is_deterministic_for_many_candidates():
    problem, true_rvec, true_tvec, correct_side = _problem(point_count=18)
    config = SolverConfig(
        exhaustive_candidate_limit=8,
        quasi_ransac_iterations=600,
        reprojection_inlier_threshold=2.0,
        minimum_inliers=8,
        random_seed=2025,
    )
    first = CandidatePnPSolver(config).solve(problem)
    second = CandidatePnPSolver(config).solve(problem)
    assert first.status == "quasi_ransac"
    assert np.array_equal(first.selected_candidates, second.selected_candidates)
    assert np.array_equal(first.selected_candidates, correct_side)
    assert first.reprojection_rmse_pixels < 1e-4
    assert _rotation_error(first.rotation_vector, true_rvec) < 1e-5
    assert np.linalg.norm(first.translation_vector - true_tvec) < 1e-5


def test_solver_resolves_unlabelled_circle_order_within_a_frame():
    problem, true_rvec, true_tvec, _ = _problem()
    keep = np.array([0, 1, 2, 3, 4])
    permutation = np.array([3, 0, 4, 1, 2])
    unordered = CalibrationProblem(
        problem.points_3d[keep],
        problem.candidates_2d[keep][permutation],
        problem.camera,
        frame_sizes=[5],
    )
    config = SolverConfig(
        exhaustive_candidate_limit=5,
        resolve_correspondences=True,
        maximum_permutation_size=5,
    )

    result = CandidatePnPSolver(config).solve(unordered)

    expected_mapping = np.argsort(permutation)
    assert np.array_equal(result.correspondence_indices, expected_mapping)
    assert result.reprojection_rmse_pixels < 1e-4
    assert _rotation_error(result.rotation_vector, true_rvec) < 1e-5
    assert np.linalg.norm(result.translation_vector - true_tvec) < 1e-5
