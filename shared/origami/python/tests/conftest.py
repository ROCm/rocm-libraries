# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for origami tests."""

import pytest

# torch is optional, tests that need it will skip appropriately
try:
    import torch
except ImportError:
    pass
import origami

from helpers import HARDWARE


def pytest_addoption(parser):
    """Add command line options for origami tests."""
    parser.addoption(
        "--generate-baseline",
        action="store_true",
        default=False,
        help="Generate new baseline files for ranking regression tests",
    )


@pytest.fixture(scope="session")
def generate_baseline(request):
    """Fixture to check if we should generate baselines."""
    return request.config.getoption("--generate-baseline")


@pytest.fixture
def hardware():
    """Get hardware for device 0, or create a mock hardware object for testing.

    Returns:
        origami.hardware_t: Hardware object for testing. Uses real device if available,
                           otherwise creates a mock MI300X (gfx942) configuration.
    """
    try:
        # Try to get real hardware from device 0
        return origami.get_hardware_for_device(0)
    except RuntimeError:
        # No ROCm device available, use mock gfx942 hardware for testing
        return HARDWARE["gfx942"]
