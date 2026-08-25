# Third-party notices

This repository contains original ROS integration code and does not vendor
third-party source.

Runtime dependencies are installed separately:

- `circular-center-calibration`, Apache-2.0, provides the released 2D and 3D
  circular-center estimators.
- ROS 1 packages, under their respective licenses, provide transport, messages,
  synchronization, image conversion, services, and TF.
- NumPy, OpenCV, and PyYAML are used under their respective licenses.
- `rosbags` is an optional dependency of the cross-platform evaluation tool and is
  used under its respective license.

FAST-Calib was used as prior experimental context but is not a build/runtime
dependency, and no FAST-Calib code is redistributed. AAMED is not included or
required; ellipse detection in this repository uses OpenCV APIs.
