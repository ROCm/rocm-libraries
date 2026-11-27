# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytest plugin: run the session on the Python engine, as a declared lane.

Loaded as ``-p rocke.tests.python_engine_only`` by the installed CTest entries
of a provider build that installs no C++ engine beside the ``rocke`` package
(the Windows provider build). Without it, every lowering there asks for the
default cpp backend, finds no engine, warns and falls back to the Python
engine, so a test that compares the two engines compares the Python engine
with itself. Before any test runs, for this process and every process it
starts, the plugin sets:

- ``ROCKE_BACKEND=python``: lowerings that name no backend run the Python
  engine, without a fallback;
- ``ROCKE_CPP_STRICT=1``: a lowering that asks for the cpp backend raises
  instead of falling back (``rocke.core.backend.cpp_strict``), so a test that
  needs the C++ engine skips or fails rather than passing on the Python one.

``rocke.tests.require_cpp_engine`` is its counterpart for the builds that
install the shared engine.
"""

from __future__ import annotations

import os

import pytest

from ..core.backend import BACKEND_ENV, BACKEND_PYTHON, CPP_STRICT_ENV

__all__ = ["pytest_configure"]


def pytest_configure(config: pytest.Config) -> None:
    """Make the Python engine the run's backend, and forbid the cpp backend's
    fallback to it."""
    os.environ[BACKEND_ENV] = BACKEND_PYTHON
    os.environ[CPP_STRICT_ENV] = "1"
