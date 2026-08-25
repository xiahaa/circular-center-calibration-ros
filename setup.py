#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup_args = generate_distutils_setup(
    packages=["circular_center_calibration_ros"],
    package_dir={"": "src"},
)

setup(**setup_args)
