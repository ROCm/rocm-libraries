# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package builds, imports, and reports hipBLASLt's own version."""

import importlib.metadata

import pytest

import hipblaslt


def _dotted(version_int: int) -> str:
    major, rest = divmod(version_int, 100000)
    minor, patch = divmod(rest, 100)
    return f"{major}.{minor}.{patch}"


def test_version_is_the_built_against_hipblaslt():
    built_version, built_rev = hipblaslt._core._built_library_version()
    assert hipblaslt.__version__ == _dotted(built_version)
    assert built_rev.strip()


def test_package_metadata_matches_the_header():
    # pyproject's version provider and the compiled-in header must agree.
    assert importlib.metadata.version("hipblaslt") == hipblaslt.__version__


def test_hip_available_is_bool():
    assert isinstance(hipblaslt._core.hip_available(), bool)


def test_library_records_where_it_came_from():
    assert hipblaslt.library.source in {
        "HIPBLASLT_PY_LIBRARY",
        "rocm-sdk",
        "ROCM_PATH",
        "ROCM_HOME",
        "build",
        "already loaded",
    }
    assert hipblaslt.library.path


@pytest.mark.gpu
def test_loaded_version_is_reported_with_a_device():
    loaded = hipblaslt._core._loaded_library_version()
    assert loaded is not None
    version, rev = loaded
    assert version > 0
    assert rev.strip()
