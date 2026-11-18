# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared test setup.

Tests marked `gpu` skip when no HIP device is visible. With HIPBLASLT_PY_REQUIRE_GPU=1 (set on the
GPU CI lanes) they fail instead, so a lane that loses its device can't go green by skipping.
"""

import os

import pytest

import hipblaslt


def _require_gpu() -> bool:
    return os.environ.get("HIPBLASLT_PY_REQUIRE_GPU", "") not in ("", "0")


@pytest.fixture(scope="session")
def hip_available() -> bool:
    return hipblaslt._core.hip_available()


def pytest_collection_modifyitems(config, items):
    if hipblaslt._core.hip_available():
        return
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if not gpu_items:
        return
    if _require_gpu():
        fail = pytest.mark.usefixtures("_no_device_but_required")
        for item in gpu_items:
            item.add_marker(fail)
    else:
        skip = pytest.mark.skip(reason="no HIP device visible")
        for item in gpu_items:
            item.add_marker(skip)


@pytest.fixture
def _no_device_but_required():
    pytest.fail("HIPBLASLT_PY_REQUIRE_GPU=1, but no HIP device is visible")
