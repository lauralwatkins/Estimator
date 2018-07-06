#!/usr/bin/env python

from setuptools import setup

setup(
    name="kinematics",
    version="0.0",
    description="Maximum likelihood estimator for kinematics.",
    author="Laura L Watkins",
    author_email="lauralwatkins@gmail.com",
    url="https://github.com/lauralwatkins/kinematics",
    package_dir = {
        "kinematics": "kinematics",
    },
    packages=["kinematics"],
)
