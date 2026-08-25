# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

from circular_center_calibration_ros.image_detector import (
    ImageDetectorConfig,
    find_ellipse_contours,
)


def test_contour_detector_finds_four_dark_ellipses_without_duplicate_edges():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    centers = [(210, 160), (430, 160), (210, 330), (430, 330)]
    for center in centers:
        cv2.ellipse(image, center, (45, 30), 18.0, 0.0, 360.0, (0, 0, 0), -1)
    config = ImageDetectorConfig(
        min_contour_area=1000.0,
        max_contour_area=10000.0,
        duplicate_center_distance=20.0,
    )
    detections = find_ellipse_contours(image, config)
    assert len(detections) == 4
    actual = sorted(tuple(np.round(item.center).astype(int)) for item in detections)
    assert actual == sorted(centers)
    assert max(item.fit_error for item in detections) < 0.08
