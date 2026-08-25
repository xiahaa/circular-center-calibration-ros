#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from collections import deque

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

from circular_center_calibration_ros.lidar_detector import (
    LidarDetectorConfig,
    detect_circular_holes,
)
from circular_center_calibration_ros.msg import CircleCenter3D, CircleCenterArray


class LidarCircleDetectorNode:
    def __init__(self):
        parameters = rospy.get_param("/lidar_detector", {})
        target = rospy.get_param("/target", {})
        self.config = LidarDetectorConfig(
            roi_min=parameters.get("roi_min", [-5.0, -5.0, -3.0]),
            roi_max=parameters.get("roi_max", [5.0, 5.0, 3.0]),
            expected_count=int(target.get("circle_count", 4)),
            grid_columns=int(target.get("grid_columns", 2)),
            plane_distance_threshold=float(parameters.get("plane_distance_threshold", 0.015)),
            plane_ransac_iterations=int(parameters.get("plane_ransac_iterations", 500)),
            maximum_plane_candidates=int(parameters.get("maximum_plane_candidates", 1)),
            occupancy_resolution=float(parameters.get("occupancy_resolution", 0.01)),
            occupancy_dilation_cells=int(parameters.get("occupancy_dilation_cells", 1)),
            minimum_hole_radius=float(parameters.get("minimum_hole_radius", 0.06)),
            maximum_hole_radius=float(parameters.get("maximum_hole_radius", 0.18)),
            annulus_half_width=float(parameters.get("annulus_half_width", 0.025)),
            circle_residual_threshold=float(parameters.get("circle_residual_threshold", 0.015)),
            random_seed=int(parameters.get("random_seed", 2025)),
        )
        output_topic = parameters.get("output_topic", "lidar_circle_centers")
        self.accumulation_frames = int(parameters.get("accumulation_frames", 1))
        if self.accumulation_frames <= 0:
            raise ValueError("accumulation_frames must be positive")
        self.cloud_buffer = deque(maxlen=self.accumulation_frames)
        self.publisher = rospy.Publisher(output_topic, CircleCenterArray, queue_size=2)
        self.subscriber = rospy.Subscriber("points", PointCloud2, self.callback, queue_size=1)

    def callback(self, cloud_message):
        points = np.asarray(
            list(
                point_cloud2.read_points(cloud_message, field_names=("x", "y", "z"), skip_nans=True)
            ),
            dtype=float,
        )
        self.cloud_buffer.append(points)
        if len(self.cloud_buffer) < self.accumulation_frames:
            return
        accumulated_points = np.vstack(self.cloud_buffer)
        try:
            detections, _ = detect_circular_holes(accumulated_points, self.config)
        except Exception as error:
            rospy.logwarn_throttle(2.0, "LiDAR circle detection failed: %s", error)
            return

        output = CircleCenterArray()
        output.header = cloud_message.header
        for detection in detections:
            message = CircleCenter3D()
            message.id = detection.id
            message.center.x, message.center.y, message.center.z = map(float, detection.center)
            message.normal.x, message.normal.y, message.normal.z = map(float, detection.normal)
            message.radius = detection.radius
            message.rmse = detection.rmse
            message.inlier_count = detection.inlier_count
            output.centers.append(message)
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("lidar_circle_detector")
    LidarCircleDetectorNode()
    rospy.spin()
