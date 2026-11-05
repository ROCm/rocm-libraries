# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixtures shared by the tiling package's tests."""

from __future__ import annotations

import pytest

from rocke.tests.gpu import current_device_arch_in


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


@pytest.fixture
def gfx90a_gpu(request: pytest.FixtureRequest) -> str:
    """Skip the test unless the current GPU is a gfx90a, and return its arch.

    A numeric test that launches its kernels on gfx90a asks for this fixture
    (``@pytest.mark.usefixtures("gfx90a_gpu")``). Under ``ROCKE_REQUIRE_GFX90A=1`` the test fails
    instead of skipping when there is no gfx90a GPU (``rocke.tests.gpu.current_device_arch_in``).
    """
    return current_device_arch_in(("gfx90a",), subject=request.node.name)
