cmake_minimum_required(VERSION 3.0.2)
project(circular_center_calibration_ros)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_filters
  message_generation
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
)

catkin_python_setup()

add_message_files(
  FILES
  CalibrationResult.msg
  CircleCenter3D.msg
  CircleCenterArray.msg
  ProjectedCenter.msg
  ProjectedCenterArray.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  cv_bridge
  geometry_msgs
  message_filters
  message_runtime
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
)

catkin_install_python(
  PROGRAMS
  scripts/calibrate_from_yaml.py
  scripts/calibration_server.py
  scripts/image_circle_detector.py
  scripts/lidar_circle_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY config examples launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()
