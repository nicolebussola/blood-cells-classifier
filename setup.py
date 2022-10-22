#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Blood cell classification",
    author="Nicole Bussola",
    author_email="nicole.bussolaceradini@gmail.com",
    url="https://github.com/nicolebussola/blood-cells-classifier",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
