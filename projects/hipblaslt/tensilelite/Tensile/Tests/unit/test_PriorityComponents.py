# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from rocisa.code import Module

from Tensile.Components.Priority import AggressivePriority


def test_unchanged_priority_returns_empty_module():
    priority = AggressivePriority(currentPrio=1)

    result = priority(None, 1)

    assert isinstance(result, Module)
    assert not list(result.items())
