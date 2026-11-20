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


def pytest_runtest_setup(item):
    if "gpu" not in item.keywords or hipblaslt._core.hip_available():
        return
    if _require_gpu():
        pytest.fail(
            "HIPBLASLT_PY_REQUIRE_GPU=1, but no HIP device is visible", pytrace=False
        )
    pytest.skip("no HIP device visible")
