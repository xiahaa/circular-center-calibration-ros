# SPDX-License-Identifier: Apache-2.0
"""Conversions between pure calibration results and ROS messages."""

import numpy as np

from .io import rotation_matrix_to_quaternion


def make_transform_stamped(result, stamp, camera_frame, lidar_frame):
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = camera_frame
    transform.child_frame_id = lidar_frame
    translation = result.transform_camera_lidar[:3, 3]
    quaternion = rotation_matrix_to_quaternion(result.transform_camera_lidar[:3, :3])
    transform.transform.translation.x = float(translation[0])
    transform.transform.translation.y = float(translation[1])
    transform.transform.translation.z = float(translation[2])
    transform.transform.rotation.x = float(quaternion[0])
    transform.transform.rotation.y = float(quaternion[1])
    transform.transform.rotation.z = float(quaternion[2])
    transform.transform.rotation.w = float(quaternion[3])
    return transform


def make_result_message(result, stamp, camera_frame, lidar_frame):
    from circular_center_calibration_ros.msg import CalibrationResult as ResultMessage

    message = ResultMessage()
    message.header.stamp = stamp
    message.header.frame_id = camera_frame
    message.success = bool(result.success)
    message.status = result.status
    stamped = make_transform_stamped(result, stamp, camera_frame, lidar_frame)
    message.transform_lidar_to_camera = stamped.transform
    message.reprojection_rmse_pixels = float(result.reprojection_rmse_pixels)
    message.correspondence_count = int(result.selected_candidates.size)
    message.inlier_count = int(np.count_nonzero(result.inlier_mask))
    message.selected_candidates = result.selected_candidates.astype(np.uint8).tolist()
    message.correspondence_indices = result.correspondence_indices.astype(np.uint32).tolist()
    return message


__all__ = ["make_result_message", "make_transform_stamped"]
