# SPDX-License-Identifier: Apache-2.0

import numpy as np

from circular_center_calibration_ros.lidar_detector import (
    LidarDetectorConfig,
    detect_circular_holes,
)


def _synthetic_board(seed=4):
    generator = np.random.default_rng(seed)
    step = 0.01
    x, y = np.meshgrid(
        np.arange(-0.60, 0.601, step),
        np.arange(-0.45, 0.451, step),
        indexing="xy",
    )
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    centers = np.array(
        [[-0.30, -0.20], [0.30, -0.20], [-0.30, 0.20], [0.30, 0.20]]
    )
    keep = np.ones(len(coordinates), dtype=bool)
    for center in centers:
        keep &= np.linalg.norm(coordinates - center, axis=1) >= 0.10
    board = np.column_stack((coordinates[keep], np.zeros(np.count_nonzero(keep))))
    board[:, 2] += generator.normal(0.0, 0.0005, len(board))

    true_rvec = np.array([0.30, -0.18, 0.12])
    angle = np.linalg.norm(true_rvec)
    axis = true_rvec / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    translation = np.array([1.2, -0.3, 0.5])
    return board @ rotation.T + translation, centers, rotation, translation


def test_planar_occupancy_detector_recovers_four_hole_centers():
    points, centers_2d, rotation, translation = _synthetic_board()
    config = LidarDetectorConfig(
        roi_min=[0.0, -2.0, -1.0],
        roi_max=[3.0, 2.0, 2.0],
        expected_count=4,
        grid_columns=2,
        plane_distance_threshold=0.004,
        plane_ransac_iterations=100,
        occupancy_resolution=0.011,
        occupancy_dilation_cells=0,
        minimum_hole_radius=0.07,
        maximum_hole_radius=0.13,
        annulus_half_width=0.025,
        circle_residual_threshold=0.02,
        random_seed=2025,
    )
    detections, _ = detect_circular_holes(points, config)
    expected_3d = np.column_stack((centers_2d, np.zeros(4))) @ rotation.T + translation
    actual = np.asarray([item.center for item in detections])
    assert len(detections) == 4
    set_errors = np.array(
        [[np.linalg.norm(point - expected) for expected in expected_3d] for point in actual]
    )
    assert np.max(np.min(set_errors, axis=1)) < 0.035
    assert max(abs(item.radius - 0.10) for item in detections) < 0.035
