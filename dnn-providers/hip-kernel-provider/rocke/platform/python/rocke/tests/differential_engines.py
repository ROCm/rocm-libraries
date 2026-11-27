# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytest plugin: lower every kernel of the session with both engines.

Loaded as ``-p rocke.tests.differential_engines`` by the installed GPU CTest
entries of a build that installs the shared C++ engine beside the ``rocke``
package (``_ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY`` in
``platform/CMakeLists.txt``). The provider packages its shipped kernels with
the Python engine, so a GPU check that lowered with the C++ engine alone would
not test what ships. Before any test runs, for this process and every process
it starts, the plugin sets ``ROCKE_BACKEND=both``: each lowering runs both
engines, raises when their output differs, and returns the Python engine's
(``rocke.core.backend``). The GPU checks then run the kernels that ship, and
check the engines' byte identity on each kernel they build. ``both`` never
falls back to the Python engine, so a missing C++ engine fails the lowering.

A test that chooses its engine itself (a ``backend=`` argument, or its own
``ROCKE_BACKEND``) keeps its choice.
"""

from __future__ import annotations

import os

import pytest

from ..core.backend import BACKEND_BOTH, BACKEND_ENV

__all__ = ["PLUGIN_NAME", "pytest_configure"]

PLUGIN_NAME = __name__


def pytest_configure(config: pytest.Config) -> None:
    """Lower every kernel of the run with both engines."""
    os.environ[BACKEND_ENV] = BACKEND_BOTH
