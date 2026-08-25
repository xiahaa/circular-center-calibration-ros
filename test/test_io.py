# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import numpy as np
import yaml

from circular_center_calibration_ros.io import load_problem, result_to_document
from circular_center_calibration_ros.models import CalibrationResult


def test_yaml_problem_flattens_multiple_frames():
    document = {
        "camera": {
            "matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0],
        },
        "frames": [
            {
                "id": "a",
                "correspondences": [
                    {
                        "id": index,
                        "point_3d": [index * 0.1, 0, 0],
                        "candidates_2d": [[100 + index, 200], [110 + index, 210]],
                    }
                    for index in range(4)
                ],
            },
            {
                "id": "b",
                "correspondences": [
                    {
                        "id": index,
                        "point_3d": [index * 0.1, 0.2, 0.1],
                        "candidates_2d": [[120 + index, 220], [130 + index, 230]],
                    }
                    for index in range(4)
                ],
            },
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "problem.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        problem = load_problem(path)
    assert problem.points_3d.shape == (8, 3)
    assert problem.candidates_2d.shape == (8, 2, 2)
    assert problem.labels[0] == "a:0"
    assert problem.labels[-1] == "b:3"


def test_result_document_states_transform_convention():
    result = CalibrationResult(
        success=True,
        status="test",
        transform_camera_lidar=np.eye(4),
        rotation_vector=np.zeros(3),
        translation_vector=np.zeros(3),
        reprojection_rmse_pixels=0.25,
        selected_candidates=np.array([0, 1, 0, 1]),
        correspondence_indices=np.arange(4),
        inlier_mask=np.ones(4, dtype=bool),
        projected_points=np.zeros((4, 2)),
        iterations=16,
    )
    document = result_to_document(result, "camera", "lidar")
    assert document["parent_frame"] == "camera"
    assert document["child_frame"] == "lidar"
    assert document["quaternion_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert "LiDAR frame to the camera frame" in document["convention"]
