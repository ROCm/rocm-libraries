# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixtures shared by the tiling package's tests.

A test that launches a kernel asks for one of the GPU fixtures below (``any_gpu`` or
``gfx90a_gpu``), and this file marks it ``gpu``. Installed CTest runs the tree as two entries
selected by that marker (``-m gpu`` and ``-m "not gpu"``), so a new GPU test that asks for a GPU
fixture runs in the GPU entry with no other edit. The launch helpers of the tree take the arch
the fixture returns, so a launch cannot be written without one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from rocke.tests.gpu import current_device_arch, current_device_arch_in

_Fixture = TypeVar("_Fixture", bound=Callable[..., object])

# The names of the fixtures defined with ``_gpu_fixture``; a test that asks for one is marked gpu.
_GPU_FIXTURES: set[str] = set()


def _gpu_fixture(function: _Fixture) -> _Fixture:
    """Define a pytest fixture whose tests are marked ``gpu``."""
    _GPU_FIXTURES.add(function.__name__)
    return pytest.fixture(function)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: launches a kernel on the current GPU (asks for a GPU fixture)"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # A conftest's collection hook sees every item of the session; only the tiling tree can ask
    # for these fixtures, which are defined here. tryfirst: the mark must be on the item before
    # -m deselects by it.
    for item in items:
        if _GPU_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.gpu)


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


@_gpu_fixture
def any_gpu() -> str:
    """Skip the test unless there is a GPU, and return its arch.

    A test that launches its kernels on whatever GPU is current takes this fixture as an argument
    and passes the arch to the launch helper. One that narrows the arch further, through
    ``rocke.tests.gpu.run_on_device``, which finds the GPU itself, asks for it with
    ``@pytest.mark.usefixtures("any_gpu")``.
    """
    return current_device_arch()


@_gpu_fixture
def gfx90a_gpu(request: pytest.FixtureRequest) -> str:
    """Skip the test unless the current GPU is a gfx90a, and return its arch.

    A numeric test that launches its kernels on gfx90a asks for this fixture
    (``@pytest.mark.usefixtures("gfx90a_gpu")``). Under ``ROCKE_REQUIRE_GFX90A=1`` the test fails
    instead of skipping when there is no gfx90a GPU (``rocke.tests.gpu.current_device_arch_in``).
    """
    return current_device_arch_in(("gfx90a",), subject=request.node.name)
