# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixtures shared by the tiling package's tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def requires_matplotlib() -> None:
    """Skip the test unless matplotlib is installed.

    matplotlib is an optional dependency, needed only to draw the visualization figures. A test that
    draws one asks for this fixture (``@pytest.mark.usefixtures("requires_matplotlib")``), so it
    skips where matplotlib is absent while the tests of the same module that draw nothing still run.
    """
    pytest.importorskip(
        "matplotlib", reason="matplotlib is not installed; it is needed to draw figures"
    )
