# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tier algebra over a small in-memory table."""

from pathlib import Path

import pytest

from codegen import generic_targets as gt

_TABLE = gt.GenericTargets(
    Path("table.json"),
    {
        "gfx11-generic": ("gfx1100", "gfx1151"),
        "gfx12-generic": ("gfx1200", "gfx1201"),
    },
)
_G11 = "gfx11-generic"
_G12 = "gfx12-generic"
_UNKNOWN = "gfx9-4-generic"

_UNRESTRICTED = gt.TIER_UNRESTRICTED
_EXPLICIT = gt.TIER_EXPLICIT
_GENERIC = gt.TIER_GENERIC


@pytest.mark.parametrize(
    ("arch", "device", "expect"),
    [
        ([], "gfx1151", _UNRESTRICTED),
        (["gfx1151"], "gfx1151", _EXPLICIT),
        ([_G11], "gfx1100", _GENERIC),
        ([_G11], "gfx1151:sramecc+", _GENERIC),
        (["gfx1151", _G11], "gfx1151", _EXPLICIT),
        (["gfx1151", _G11], "gfx1100", _GENERIC),
        ([_G11], "gfx1154", None),
        ([_G12], "gfx1100", None),
        (["gfx942"], "gfx950", None),
        (["gfx94"], "gfx942", None),
        (["gfx1250"], "gfx1250-strict", None),
        (["gfx1250-strict"], "gfx1250-strict", _EXPLICIT),
        ([_UNKNOWN], "gfx942", None),
    ],
)
def test_list_tier(arch, device, expect):
    assert gt.list_tier(arch, device, _TABLE) == expect


@pytest.mark.parametrize(
    ("outer", "inner", "expect"),
    [
        ([], ["gfx942"], True),
        (["gfx942"], [], True),
        (["gfx942", "gfx950"], ["gfx942"], True),
        (["gfx942"], ["gfx942", "gfx950"], False),
        ([_G11], ["gfx1151"], True),
        ([_G11], ["gfx1154"], False),
        (["gfx1151"], [_G11], False),
        (["gfx1100", "gfx1151"], [_G11], True),
        ([_G11], [_G11], True),
        ([_G11], [_G12], False),
        (["gfx942", _G11], ["gfx942", "gfx1100"], True),
        (["gfx942", _G11], ["gfx950", "gfx1100"], False),
    ],
)
def test_covers(outer, inner, expect):
    assert gt.covers(outer, inner, _TABLE) is expect


@pytest.mark.parametrize(
    ("a", "b", "expect"),
    [
        ([], [], True),
        ([], ["gfx942"], False),
        (["gfx942"], ["gfx942"], True),
        (["gfx942"], ["gfx950"], False),
        ([_G11], ["gfx1151"], False),
        ([_G11], [_G11], True),
        ([_G11], [_G12], False),
        (["gfx1151", _G11], [_G11], True),
        ([_G11], [_G11, "gfx942"], True),
    ],
)
def test_compete(a, b, expect):
    assert gt.compete(a, b, _TABLE) is expect
    assert gt.compete(b, a, _TABLE) is expect


def test_the_shipped_table_loads_and_lists_gfx11_generic_members():
    table = gt.default_table()
    assert table.has(_G11)
    assert "gfx1151" in table.members(_G11)
    assert gt.list_tier([_G11], "gfx1151", table) == _GENERIC
