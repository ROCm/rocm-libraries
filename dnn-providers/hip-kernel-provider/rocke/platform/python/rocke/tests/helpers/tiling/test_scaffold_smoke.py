# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scaffold smoke test — every subpackage imports cleanly.

A failsafe: if the package tree or an ``__init__`` breaks, this fails first with a
clear signal before any component test runs.
"""

from __future__ import annotations

import importlib

import pytest

_SUBPACKAGES = [
    "rocke.helpers.tiling",
    "rocke.helpers.tiling.traits",
    "rocke.helpers.tiling.layouts",
    "rocke.helpers.tiling.mma",
    "rocke.helpers.tiling.visualization",
]


@pytest.mark.parametrize("module_name", _SUBPACKAGES)
def test_subpackage_imports(module_name: str) -> None:
    importlib.import_module(module_name)
