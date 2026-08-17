# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests to verify profile subset relationships.

These tests ensure that profile hierarchies are maintained:
- smoke <= precheckin (smoke tests are a subset of precheckin)
- codecov <= precheckin (codecov tests are a subset of precheckin)
"""

from pathlib import Path

import pytest
from rrtest import list_tests


@pytest.fixture
def build_dir():
    """Fixture to provide build directory."""
    import os

    if "ROCROLLER_BUILD_DIR" in os.environ:
        return Path(os.environ["ROCROLLER_BUILD_DIR"])

    return Path("build")


def test_smoke_subset_of_precheckin(build_dir):
    """Test that smoke profile tests are a subset of precheckin."""
    smoke = list_tests("smoke", build_dir)
    precheckin = list_tests("precheckin", build_dir)

    assert len(smoke) > 0, "smoke profile should have tests"
    assert len(precheckin) > 0, "precheckin profile should have tests"

    assert smoke.issubset(precheckin), (
        f"smoke tests should be a subset of precheckin. "
        f"Found {len(smoke - precheckin)} tests in smoke but not in precheckin"
    )


def test_codecov_subset_of_precheckin_mci(build_dir):
    """Test that codecov profile tests are a subset of precheckin."""
    codecov = list_tests("codecov-mci", build_dir)
    precheckin = list_tests("precheckin-mci", build_dir)

    assert len(codecov) > 0, "codecov profile should have tests"
    assert len(precheckin) > 0, "precheckin profile should have tests"

    assert codecov.issubset(precheckin), (
        f"codecov tests should be a subset of precheckin. "
        f"Found {len(codecov - precheckin)} tests in codecov but not in precheckin"
    )
