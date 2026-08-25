#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import threading

import message_filters
import numpy as np
import rospy
import tf2_ros
from std_srvs.srv import Trigger, TriggerResponse

from circular_center_calibration_ros.io import save_result
from circular_center_calibration_ros.models import CalibrationProblem, CameraModel
from circular_center_calibration_ros.msg import (
    CalibrationResult,
    CircleCenterArray,
    ProjectedCenterArray,
)
from circular_center_calibration_ros.ros_utils import (
    make_result_message,
    make_transform_stamped,
)
from circular_center_calibration_ros.solver import CandidatePnPSolver, SolverConfig


class CalibrationServer:
    def __init__(self):
        parameters = rospy.get_param("/solver", {})
        frames = rospy.get_param("/frames", {})
        self.camera_frame = frames.get("camera", "camera")
        self.lidar_frame = frames.get("lidar", "lidar")
        self.minimum_frames = int(parameters.get("minimum_frames", 1))
        self.auto_solve = bool(parameters.get("auto_solve", True))
        self.output_path = parameters.get("output_path", "")
        self.solver = CandidatePnPSolver(
            SolverConfig(
                exhaustive_candidate_limit=int(
                    parameters.get("exhaustive_candidate_limit", 12)
                ),
                quasi_ransac_iterations=int(
                    parameters.get("quasi_ransac_iterations", 1000)
                ),
                reprojection_inlier_threshold=float(
                    parameters.get("reprojection_inlier_threshold", 3.0)
                ),
                minimum_inliers=int(parameters.get("minimum_inliers", 4)),
                random_seed=int(parameters.get("random_seed", 2025)),
                resolve_correspondences=bool(
                    parameters.get("resolve_correspondences", True)
                ),
                maximum_permutation_size=int(
                    parameters.get("maximum_permutation_size", 6)
                ),
            )
        )
        self.lock = threading.Lock()
        self.frames = []
        self.camera = None
        self.result_publisher = rospy.Publisher(
            "calibration_result", CalibrationResult, queue_size=1, latch=True
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.solve_service = rospy.Service("~solve", Trigger, self.solve_service_callback)
        self.reset_service = rospy.Service("~reset", Trigger, self.reset_service_callback)

        image_subscriber = message_filters.Subscriber(
            "projected_circle_centers", ProjectedCenterArray
        )
        lidar_subscriber = message_filters.Subscriber(
            "lidar_circle_centers", CircleCenterArray
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_subscriber, lidar_subscriber], queue_size=10, slop=0.10
        )
        self.synchronizer.registerCallback(self.observation_callback)

    @staticmethod
    def _extract_frame(image_message, lidar_message):
        image_by_id = {int(item.id): item for item in image_message.centers}
        lidar_by_id = {int(item.id): item for item in lidar_message.centers}
        common_ids = sorted(set(image_by_id) & set(lidar_by_id))
        if len(common_ids) < 4:
            raise ValueError("a synchronized frame needs at least four matching circle IDs")
        points = []
        candidates = []
        labels = []
        for circle_id in common_ids:
            image = image_by_id[circle_id]
            lidar = lidar_by_id[circle_id]
            points.append([lidar.center.x, lidar.center.y, lidar.center.z])
            candidates.append(
                [
                    [image.primary.x, image.primary.y],
                    [image.alternative.x, image.alternative.y],
                ]
            )
            labels.append(str(circle_id))
        return np.asarray(points), np.asarray(candidates), labels

    def observation_callback(self, image_message, lidar_message):
        try:
            points, candidates, labels = self._extract_frame(
                image_message, lidar_message
            )
            camera = CameraModel(
                np.asarray(image_message.camera_info.K, dtype=float).reshape(3, 3),
                np.asarray(image_message.camera_info.D, dtype=float),
            )
        except Exception as error:
            rospy.logwarn("rejected calibration observation: %s", error)
            return
        with self.lock:
            if self.camera is not None and not np.allclose(
                self.camera.matrix, camera.matrix, rtol=1e-6, atol=1e-6
            ):
                rospy.logwarn("camera intrinsics changed; reset the calibration server")
                return
            self.camera = camera
            frame_index = len(self.frames)
            self.frames.append(
                (points, candidates, [f"{frame_index}:{label}" for label in labels])
            )
            frame_count = len(self.frames)
        rospy.loginfo("accepted calibration frame %d", frame_count)
        if self.auto_solve and frame_count >= self.minimum_frames:
            self.solve_and_publish(image_message.header.stamp)

    def _problem(self):
        with self.lock:
            if self.camera is None or not self.frames:
                raise RuntimeError("no synchronized observations have been received")
            points = np.vstack([frame[0] for frame in self.frames])
            candidates = np.vstack([frame[1] for frame in self.frames])
            labels = [label for frame in self.frames for label in frame[2]]
            return CalibrationProblem(
                points,
                candidates,
                self.camera,
                labels,
                [len(frame[0]) for frame in self.frames],
            )

    def solve_and_publish(self, stamp):
        try:
            result = self.solver.solve(self._problem())
        except Exception as error:
            rospy.logerr("calibration failed: %s", error)
            return None
        result_message = make_result_message(
            result, stamp, self.camera_frame, self.lidar_frame
        )
        self.result_publisher.publish(result_message)
        self.tf_broadcaster.sendTransform(
            make_transform_stamped(result, stamp, self.camera_frame, self.lidar_frame)
        )
        if self.output_path:
            save_result(
                self.output_path, result, self.camera_frame, self.lidar_frame
            )
        rospy.loginfo(
            "calibration succeeded with %d/%d inliers, RMSE %.4f px",
            result_message.inlier_count,
            result_message.correspondence_count,
            result_message.reprojection_rmse_pixels,
        )
        return result

    def solve_service_callback(self, _request):
        result = self.solve_and_publish(rospy.Time.now())
        if result is None:
            return TriggerResponse(False, "calibration failed; inspect ROS logs")
        return TriggerResponse(
            True, f"calibration RMSE {result.reprojection_rmse_pixels:.4f} px"
        )

    def reset_service_callback(self, _request):
        with self.lock:
            self.frames = []
            self.camera = None
        return TriggerResponse(True, "calibration observations cleared")


if __name__ == "__main__":
    rospy.init_node("circular_center_calibration")
    CalibrationServer()
    rospy.spin()
