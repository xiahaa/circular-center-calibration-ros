# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

from circular_center_calibration_ros.image_detector import (
    ImageDetectorConfig,
    find_aruco_guided_ellipses,
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


def _draw_marker(dictionary, marker_id, size):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    marker = np.zeros((size, size), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size, marker, 1)
    return marker


def test_aruco_guidance_recovers_hole_hidden_from_global_thresholding():
    dictionary_id = cv2.aruco.DICT_6X6_50
    dictionary = (
        cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "getPredefinedDictionary")
        else cv2.aruco.Dictionary_get(dictionary_id)
    )
    image = np.full((900, 1200, 3), 255, dtype=np.uint8)
    marker_centers = [(100, 100), (1100, 100), (100, 800), (1100, 800)]
    for marker_id, center in zip((1, 2, 3, 4), marker_centers):
        marker = _draw_marker(dictionary, marker_id, 80)
        x, y = center
        image[y - 40 : y + 40, x - 40 : x + 40] = marker[:, :, None]

    normalized = ((0.273, 0.210), (0.727, 0.210), (0.273, 0.785), (0.727, 0.785))
    expected = []
    for index, (x_value, y_value) in enumerate(normalized):
        center = (round(100 + 1000 * x_value), round(100 + 700 * y_value))
        expected.append(center)
        color = (0, 0, 0) if index != 2 else (45, 45, 45)
        cv2.circle(image, center, 112, color, -1)
    config = ImageDetectorConfig(
        aruco_dictionary="DICT_6X6_50",
        aruco_corner_ids=(1, 2, 3, 4),
        aruco_hole_centers=normalized,
    )

    detections = find_aruco_guided_ellipses(image, config)

    assert len(detections) == 4
    actual = np.asarray([detection.center for detection in detections])
    assert np.max(np.linalg.norm(actual - np.asarray(expected), axis=1)) < 2.0
    assert np.max(np.abs(np.asarray([item.semi_axes for item in detections]) - 112.0)) < 3.0
