# SPDX-License-Identifier: Apache-2.0
"""Portable YAML input/output for repeatable offline calibration."""

from pathlib import Path

import numpy as np
import yaml

from .models import CalibrationProblem, CalibrationResult, CameraModel


def load_problem(path) -> CalibrationProblem:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("calibration input must be a YAML mapping")

    camera_data = document.get("camera", {})
    camera = CameraModel(
        matrix=np.asarray(camera_data["matrix"], dtype=float).reshape(3, 3),
        distortion=np.asarray(camera_data.get("distortion", []), dtype=float),
        distortion_model=camera_data.get("distortion_model", "pinhole"),
    )
    points = []
    candidates = []
    labels = []
    frame_sizes = []
    for frame_index, frame in enumerate(document.get("frames", [])):
        frame_name = str(frame.get("id", frame_index))
        correspondences = frame.get("correspondences", [])
        frame_sizes.append(len(correspondences))
        for item in correspondences:
            circle_id = str(item["id"])
            points.append(item["point_3d"])
            candidates.append(item["candidates_2d"])
            labels.append(f"{frame_name}:{circle_id}")
    return CalibrationProblem(points, candidates, camera, labels, frame_sizes)


def rotation_matrix_to_quaternion(rotation):
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def result_to_document(result: CalibrationResult, parent_frame: str, child_frame: str):
    quaternion = rotation_matrix_to_quaternion(result.transform_camera_lidar[:3, :3])
    return {
        "convention": "T_camera_lidar transforms points from the LiDAR frame to the camera frame",
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "status": result.status,
        "transform_camera_lidar": result.transform_camera_lidar.tolist(),
        "translation": result.transform_camera_lidar[:3, 3].tolist(),
        "quaternion_xyzw": quaternion.tolist(),
        "reprojection_rmse_pixels": float(result.reprojection_rmse_pixels),
        "correspondence_count": int(result.selected_candidates.size),
        "inlier_count": int(np.count_nonzero(result.inlier_mask)),
        "selected_candidates": result.selected_candidates.astype(int).tolist(),
        "correspondence_indices": result.correspondence_indices.astype(int).tolist(),
        "inlier_mask": result.inlier_mask.astype(bool).tolist(),
        "iterations": int(result.iterations),
    }


def save_result(path, result: CalibrationResult, parent_frame: str, child_frame: str):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            result_to_document(result, parent_frame, child_frame),
            stream,
            sort_keys=False,
        )


__all__ = [
    "load_problem",
    "result_to_document",
    "rotation_matrix_to_quaternion",
    "save_result",
]
