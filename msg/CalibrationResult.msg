# SPDX-License-Identifier: Apache-2.0
std_msgs/Header header
bool success
string status
geometry_msgs/Transform transform_lidar_to_camera
float64 reprojection_rmse_pixels
uint32 correspondence_count
uint32 inlier_count
uint8[] selected_candidates
uint32[] correspondence_indices
