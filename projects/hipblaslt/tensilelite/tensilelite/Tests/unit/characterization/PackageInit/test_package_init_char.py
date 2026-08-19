################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the ROCm-coupled package boundary."""

from importlib.metadata import version

import pytest

import tensilelite

pytestmark = pytest.mark.unit


def test_distribution_version():
    assert tensilelite.__version__ == version("tensilelite")
    assert "+rocm" in tensilelite.__version__


def test_generator_version_is_independent():
    assert tensilelite.GENERATOR_VERSION == "5.0.0"
    assert tensilelite.__version__.startswith(tensilelite.GENERATOR_VERSION + "+rocm")


def test_runtime_details_are_not_public():
    assert not hasattr(tensilelite, "RUNTIME")
    assert not hasattr(tensilelite, "RuntimeInfo")
    assert not hasattr(tensilelite, "TENSILELITE_CLIENT_PATH")
