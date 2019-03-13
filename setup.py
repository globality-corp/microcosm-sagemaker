#!/usr/bin/env python
from setuptools import find_packages, setup

project = "microcosm-sagemaker"
version = "1.0.0"

setup(
    name=project,
    version=version,
    description="Opinionated machine learning organization and configuration",
    author="Globality Engineering",
    author_email="engineering@globality.com",
    url="https://github.com/globality-corp/microcosm-sagemaker",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6",
    keywords="microcosm",
    install_requires=[
        "microcosm>=2.0.0",
        "click>=7.0",
        "boto3>=1.9.90",
    ],
    setup_requires=[
        "nose>=1.3.6",
    ],
    dependency_links=[
    ],
    entry_points={
        "console_scripts": [
            "put-sagemaker-config = microcosm_sagemaker.commands.config:put_sagemaker_config",
            "list-sagemaker-configs = microcosm_sagemaker.commands.config:list_sagemaker_configs",
            "get-sagemaker-config = microcosm_sagemaker.commands.config:get_sagemaker_config",
        ],
        "microcosm.factories": [
            "active_bundle = microcosm_sagemaker.factories:configure_active_bundle",
            "active_evaluation = microcosm_sagemaker.factories:configure_active_evaluation",
        ],
    },
    tests_require=[
        "coverage>=3.7.1",
        "PyHamcrest>=1.9.0",
    ],
)
