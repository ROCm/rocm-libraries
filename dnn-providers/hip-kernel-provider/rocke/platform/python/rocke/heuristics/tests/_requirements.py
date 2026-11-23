# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The heuristics tests' own requirements: each test module skips unless the
packages ``requirements.txt`` lists are installed."""

from __future__ import annotations

from pathlib import Path

from rocke.tests.lanes import CI_TEST_TOOLS, skip_until_installed

REQUIREMENTS = Path(__file__).with_name("requirements.txt")


def skip_without_requirements() -> None:
    """Skip the calling test module, as waiting for the optional packages of the
    CI test environment, unless every package of ``REQUIREMENTS`` is installed."""
    __tracebackhide__ = True  # pytest reports the skip at the caller
    skip_until_installed(REQUIREMENTS, CI_TEST_TOOLS)
