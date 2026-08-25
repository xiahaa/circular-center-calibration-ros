#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from circular_center_calibration_ros.image_detector import (
    ImageDetectorConfig,
    detect_projected_centers,
)
from circular_center_calibration_ros.models import CameraModel
from circular_center_calibration_ros.msg import (
    ProjectedCenter,
    ProjectedCenterArray,
)


class ImageCircleDetectorNode:
    def __init__(self):
        parameters = rospy.get_param("/image_detector", {})
        target = rospy.get_param("/target", {})
        self.config = ImageDetectorConfig(
            marker_diameter=float(target.get("circle_diameter", 0.24)),
            expected_count=int(target.get("circle_count", 4)),
            grid_columns=int(target.get("grid_columns", 2)),
            threshold_mode=parameters.get("threshold_mode", "otsu_dark"),
            adaptive_block_size=int(parameters.get("adaptive_block_size", 31)),
            adaptive_constant=float(parameters.get("adaptive_constant", 5.0)),
            min_contour_area=float(parameters.get("min_contour_area", 100.0)),
            max_contour_area=float(parameters.get("max_contour_area", 100000.0)),
            min_axis_ratio=float(parameters.get("min_axis_ratio", 0.20)),
            duplicate_center_distance=float(
                parameters.get("duplicate_center_distance", 8.0)
            ),
        )
        self.distortion_model = parameters.get("distortion_model", "pinhole")
        output_topic = parameters.get("output_topic", "projected_circle_centers")
        debug_topic = parameters.get(
            "debug_image_topic", "projected_circle_centers/debug"
        )
        self.publisher = rospy.Publisher(output_topic, ProjectedCenterArray, queue_size=2)
        self.debug_publisher = rospy.Publisher(debug_topic, Image, queue_size=1)
        self.bridge = CvBridge()

        image_subscriber = message_filters.Subscriber("image", Image)
        info_subscriber = message_filters.Subscriber("camera_info", CameraInfo)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_subscriber, info_subscriber], queue_size=10, slop=0.05
        )
        self.synchronizer.registerCallback(self.callback)

    def callback(self, image_message, camera_info):
        try:
            image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
            camera = CameraModel(
                matrix=np.asarray(camera_info.K, dtype=float).reshape(3, 3),
                distortion=np.asarray(camera_info.D, dtype=float),
                distortion_model=self.distortion_model,
            )
            detections, rectified = detect_projected_centers(
                image, camera, self.config
            )
        except Exception as error:
            rospy.logwarn_throttle(2.0, "image circle detection failed: %s", error)
            return

        output = ProjectedCenterArray()
        output.header = image_message.header
        output.camera_info = camera_info
        output.camera_info.K = camera.matrix.reshape(-1).tolist()
        output.camera_info.D = [0.0] * len(camera_info.D)
        output.marker_diameter = self.config.marker_diameter
        debug = rectified.copy()
        for detection in detections:
            message = ProjectedCenter()
            message.id = detection.id
            message.primary.x = float(detection.primary[0])
            message.primary.y = float(detection.primary[1])
            message.alternative.x = float(detection.alternative[0])
            message.alternative.y = float(detection.alternative[1])
            message.primary_score = float(detection.scores[0])
            message.alternative_score = float(detection.scores[1])
            message.confidence = float(detection.confidence)
            message.ellipse_center_u = float(detection.ellipse[0])
            message.ellipse_center_v = float(detection.ellipse[1])
            message.ellipse_semi_major = float(detection.ellipse[2])
            message.ellipse_semi_minor = float(detection.ellipse[3])
            message.ellipse_angle = float(detection.ellipse[4])
            output.centers.append(message)
            cv2.circle(debug, tuple(np.round(detection.primary).astype(int)), 4, (0, 255, 0), -1)
            cv2.circle(debug, tuple(np.round(detection.alternative).astype(int)), 4, (0, 0, 255), -1)
            cv2.putText(
                debug,
                str(detection.id),
                tuple(np.round(detection.ellipse[:2]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )
        self.publisher.publish(output)
        debug_message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_message.header = image_message.header
        self.debug_publisher.publish(debug_message)


if __name__ == "__main__":
    rospy.init_node("image_circle_detector")
    ImageCircleDetectorNode()
    rospy.spin()
