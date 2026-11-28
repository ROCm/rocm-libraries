# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytest plugin: run the session with the C++ engine required.

Loaded as ``-p rocke.tests.require_cpp_engine`` by the installed CTest entries
of a build that installs the shared C++ engine beside the ``rocke`` package.
Those entries count as covering the C++ engine, so before any test runs, for
this process and every process it starts, the plugin sets:

- ``ROCKE_REQUIRE_CPP_ENGINE=1``: a C++ case whose shared engine is missing
  fails rather than skips (``rocke.tests.lanes.skip_until_shared_engine``,
  which the ``rocke.tests.engines`` helpers use);
- ``ROCKE_CPP_STRICT=1``: the cpp backend raises instead of falling back to the
  Python engine (``rocke.core.backend.cpp_strict``).

The two are separate settings because a test may set ``ROCKE_CPP_STRICT`` for
its own lowerings, which says nothing about whether the run requires the
engine. A test that runs the Python engine alone says so itself
(``backend="python"``, ``ROCKE_BACKEND=python``, or the Python lowerer).
"""

from __future__ import annotations

import os

import pytest

from ..core.backend import CPP_STRICT_ENV
from .lanes import REQUIRE_CPP_ENGINE_ENV

__all__ = ["pytest_configure"]


def pytest_configure(config: pytest.Config) -> None:
    """Require the C++ engine for the whole run, and forbid the cpp backend's
    fallback to the Python engine."""
    os.environ[REQUIRE_CPP_ENGINE_ENV] = "1"
    os.environ[CPP_STRICT_ENV] = "1"
