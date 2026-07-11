# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt

c = hipblaslt._core


@pytest.fixture(scope="session")
def hip_available():
    """Return True if a HIP device is reachable, False otherwise."""
    return c.hip_available()
