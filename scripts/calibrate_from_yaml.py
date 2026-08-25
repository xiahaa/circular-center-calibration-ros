#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import rospy
import tf2_ros

from circular_center_calibration_ros.io import load_problem, save_result
from circular_center_calibration_ros.msg import CalibrationResult
from circular_center_calibration_ros.ros_utils import (
    make_result_message,
    make_transform_stamped,
)
from circular_center_calibration_ros.solver import CandidatePnPSolver, SolverConfig


def main():
    input_path = rospy.get_param("~input")
    output_path = rospy.get_param("~output", "result.yaml")
    camera_frame = rospy.get_param("~camera_frame", "camera")
    lidar_frame = rospy.get_param("~lidar_frame", "lidar")
    publish_tf = bool(rospy.get_param("~publish_tf", True))
    solver = CandidatePnPSolver(
        SolverConfig(
            exhaustive_candidate_limit=int(
                rospy.get_param("~exhaustive_candidate_limit", 12)
            ),
            quasi_ransac_iterations=int(
                rospy.get_param("~quasi_ransac_iterations", 1000)
            ),
            reprojection_inlier_threshold=float(
                rospy.get_param("~reprojection_inlier_threshold", 3.0)
            ),
            minimum_inliers=int(rospy.get_param("~minimum_inliers", 4)),
            random_seed=int(rospy.get_param("~random_seed", 2025)),
            resolve_correspondences=bool(
                rospy.get_param("~resolve_correspondences", False)
            ),
            maximum_permutation_size=int(
                rospy.get_param("~maximum_permutation_size", 6)
            ),
        )
    )
    problem = load_problem(input_path)
    result = solver.solve(problem)
    save_result(output_path, result, camera_frame, lidar_frame)

    stamp = rospy.Time.now()
    publisher = rospy.Publisher(
        "calibration_result", CalibrationResult, queue_size=1, latch=True
    )
    publisher.publish(make_result_message(result, stamp, camera_frame, lidar_frame))
    if publish_tf:
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        broadcaster.sendTransform(
            make_transform_stamped(result, stamp, camera_frame, lidar_frame)
        )
    rospy.loginfo(
        "saved %s; %d/%d inliers, RMSE %.4f px",
        output_path,
        int(result.inlier_mask.sum()),
        len(result.inlier_mask),
        result.reprojection_rmse_pixels,
    )
    rospy.sleep(1.0)


if __name__ == "__main__":
    rospy.init_node("circular_center_offline_calibration")
    main()
