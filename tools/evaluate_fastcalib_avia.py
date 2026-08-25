#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the independent pipeline on the five FAST-Calib Avia bags."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from circular_center_calibration_ros.image_detector import (
    ImageDetectorConfig,
    detect_projected_centers,
)
from circular_center_calibration_ros.io import result_to_document
from circular_center_calibration_ros.lidar_detector import (
    LidarDetectorConfig,
    detect_circular_holes,
)
from circular_center_calibration_ros.models import CalibrationProblem, CameraModel
from circular_center_calibration_ros.solver import CandidatePnPSolver, SolverConfig


def _point_cloud_array(message):
    offsets = {field.name: field.offset for field in message.fields}
    byte_order = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [byte_order + "f4"] * 3,
            "offsets": [offsets[name] for name in ("x", "y", "z")],
            "itemsize": message.point_step,
        }
    )
    structured = np.frombuffer(
        message.data,
        dtype=dtype,
        count=message.width * message.height,
    )
    return np.column_stack((structured["x"], structured["y"], structured["z"])).astype(float)


def _read_observation(path, accumulation_frames, typestore):
    try:
        from rosbags.rosbag1 import Reader
    except ImportError as error:
        raise RuntimeError("install requirements-evaluation.txt") from error

    image = None
    clouds = []
    with Reader(path) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in ("/left_camera/image/compressed", "/livox/lidar")
        ]
        for connection, _, raw_data in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(raw_data, connection.msgtype)
            if connection.topic.endswith("compressed") and image is None:
                image = cv2.imdecode(np.asarray(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            elif connection.topic.endswith("lidar") and len(clouds) < accumulation_frames:
                clouds.append(_point_cloud_array(message))
            if image is not None and len(clouds) == accumulation_frames:
                break
    if image is None or len(clouds) != accumulation_frames:
        raise RuntimeError(f"{path.name} does not contain a complete observation")
    return image, np.vstack(clouds)


def _make_configs(document):
    camera_data = document["camera"]
    image_data = document["image_detector"]
    lidar_data = document["lidar_detector"]
    target = document["target"]
    solver_data = document["solver"]
    camera = CameraModel(
        camera_data["matrix"],
        camera_data["distortion"],
        image_data["distortion_model"],
    )
    image = ImageDetectorConfig(
        marker_diameter=target["circle_diameter"],
        expected_count=target["circle_count"],
        grid_columns=target["grid_columns"],
        min_contour_area=image_data["min_contour_area"],
        max_contour_area=image_data["max_contour_area"],
        aruco_dictionary=image_data["aruco_dictionary"],
        aruco_corner_ids=image_data["aruco_corner_ids"],
        aruco_hole_centers=image_data["aruco_hole_centers"],
        aruco_warp_size=image_data["aruco_warp_size"],
        aruco_radius_fraction_min=image_data["aruco_radius_fraction_min"],
        aruco_radius_fraction_max=image_data["aruco_radius_fraction_max"],
    )
    lidar = LidarDetectorConfig(
        roi_min=lidar_data["roi_min"],
        roi_max=lidar_data["roi_max"],
        expected_count=target["circle_count"],
        grid_columns=target["grid_columns"],
        plane_distance_threshold=lidar_data["plane_distance_threshold"],
        plane_ransac_iterations=lidar_data["plane_ransac_iterations"],
        maximum_plane_candidates=lidar_data["maximum_plane_candidates"],
        occupancy_resolution=lidar_data["occupancy_resolution"],
        occupancy_dilation_cells=lidar_data["occupancy_dilation_cells"],
        minimum_hole_radius=lidar_data["minimum_hole_radius"],
        maximum_hole_radius=lidar_data["maximum_hole_radius"],
        annulus_half_width=lidar_data["annulus_half_width"],
        circle_residual_threshold=lidar_data["circle_residual_threshold"],
        random_seed=lidar_data["random_seed"],
    )
    solver = SolverConfig(
        exhaustive_candidate_limit=solver_data["exhaustive_candidate_limit"],
        quasi_ransac_iterations=solver_data["quasi_ransac_iterations"],
        reprojection_inlier_threshold=solver_data["reprojection_inlier_threshold"],
        minimum_inliers=solver_data["minimum_inliers"],
        random_seed=solver_data["random_seed"],
        resolve_correspondences=solver_data["resolve_correspondences"],
        maximum_permutation_size=solver_data["maximum_permutation_size"],
    )
    return camera, image, lidar, solver


def _draw_debug(image, detections):
    output = image.copy()
    for detection in detections:
        center = tuple(np.round(detection.ellipse[:2]).astype(int))
        axes = tuple(np.round(detection.ellipse[2:4]).astype(int))
        angle = float(np.rad2deg(detection.ellipse[4]))
        cv2.ellipse(output, center, axes, angle, 0.0, 360.0, (255, 0, 255), 3)
        cv2.circle(
            output,
            tuple(np.round(detection.primary).astype(int)),
            5,
            (0, 255, 0),
            -1,
        )
        cv2.circle(
            output,
            tuple(np.round(detection.alternative).astype(int)),
            5,
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            output,
            str(detection.id),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )
    return output


def _reference_errors(result, reference):
    reference_rotation = np.asarray(reference["rotation_camera_lidar"], dtype=float)
    reference_translation = np.asarray(reference["translation_camera_lidar"], dtype=float)
    relative = result.transform_camera_lidar[:3, :3] @ reference_rotation.T
    rotation_error = np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    translation_error = np.linalg.norm(result.translation_vector - reference_translation)
    return float(rotation_error), float(translation_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_directory", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "config" / "fastcalib_avia.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path("avia_result.yaml"))
    parser.add_argument("--debug-directory", type=Path)
    arguments = parser.parse_args()

    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError("install requirements-evaluation.txt") from error

    document = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    camera, image_config, lidar_config, solver_config = _make_configs(document)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    bag_paths = sorted(arguments.bag_directory.glob("*.bag"), key=lambda path: int(path.stem))
    if not bag_paths:
        raise RuntimeError("no numbered ROS1 bags were found")
    if arguments.debug_directory:
        arguments.debug_directory.mkdir(parents=True, exist_ok=True)

    points = []
    candidates = []
    frame_sizes = []
    observations = []
    accumulation_frames = int(document["lidar_detector"]["accumulation_frames"])
    for bag_path in bag_paths:
        raw_image, cloud = _read_observation(bag_path, accumulation_frames, typestore)
        image_detections, rectified = detect_projected_centers(raw_image, camera, image_config)
        lidar_detections, _ = detect_circular_holes(cloud, lidar_config)
        points.extend(detection.center for detection in lidar_detections)
        candidates.extend(
            [detection.primary, detection.alternative] for detection in image_detections
        )
        frame_sizes.append(len(lidar_detections))
        observations.append(
            {
                "bag": bag_path.name,
                "image_circle_count": len(image_detections),
                "lidar_circle_count": len(lidar_detections),
                "lidar_radii": [float(item.radius) for item in lidar_detections],
                "lidar_circle_rmse": [float(item.rmse) for item in lidar_detections],
            }
        )
        if arguments.debug_directory:
            cv2.imwrite(
                str(arguments.debug_directory / f"{bag_path.stem}_detections.png"),
                _draw_debug(rectified, image_detections),
            )

    problem = CalibrationProblem(points, candidates, camera, frame_sizes=frame_sizes)
    result = CandidatePnPSolver(solver_config).solve(problem)
    output = result_to_document(result, document["frames"]["camera"], document["frames"]["lidar"])
    output["observations"] = observations
    if "reference" in document:
        rotation_error, translation_error = _reference_errors(result, document["reference"])
        output["reference_rotation_error_degrees"] = rotation_error
        output["reference_translation_error_meters"] = translation_error
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
