################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the ROCm-coupled package boundary."""

from _tensilelite_client_binding import current_installation
import re

import pytest

import tensilelite

pytestmark = pytest.mark.unit


# Development builds are labelled with their commit; only TheRock builds carry a
# ROCm identity, as +rocm<identity> or +devrocm<identity>.
_LABEL = r"(g[0-9a-f]{40}(\.dirty)?|(dev)?rocm[0-9a-z.]+)"


def test_distribution_version():
    assert tensilelite.__version__ == current_installation().version


def test_generator_version_is_independent():
    assert tensilelite.GENERATOR_VERSION == "5.0.0"
    assert re.fullmatch(
        re.escape(tensilelite.GENERATOR_VERSION) + r"(\+" + _LABEL + r")?", tensilelite.__version__
    ), tensilelite.__version__


def test_runtime_details_are_not_public():
    assert not hasattr(tensilelite, "RUNTIME")
    assert not hasattr(tensilelite, "RuntimeInfo")
    assert not hasattr(tensilelite, "TENSILELITE_CLIENT_PATH")
