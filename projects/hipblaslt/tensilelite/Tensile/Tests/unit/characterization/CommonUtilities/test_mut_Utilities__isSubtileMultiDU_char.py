################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.isSubtileMultiDU``.

The function is a single expression::

    du = kernel["DepthU"]
    return kernel.get("_DepthUA", du) < du or kernel.get("_DepthUB", du) < du

It returns True when either per-uid DepthU (``_DepthUA``/``_DepthUB``) is
strictly smaller than the loop ``DepthU``. A missing ``_DepthUA``/``_DepthUB``
key defaults to ``du`` (so ``du < du`` is False -- an absent tensor cannot
trigger multi-DU on its own).

These tests pin the CURRENT behavior exactly:
  - the literal ``"DepthU"`` subscript (a missing key raises ``KeyError``),
  - the two literal ``.get`` keys ``"_DepthUA"`` / ``"_DepthUB"``,
  - each ``.get`` default being ``du`` (NOT ``None``): an absent side compares
    ``du < du`` and yields False rather than raising ``TypeError``,
  - the strict ``<`` on each side (equal DepthU is NOT multi-DU),
  - the ``or`` (either side alone is sufficient).

Every listed surviving mutant is distinguished by at least one assertion whose
value (or raised/not-raised status) differs from the clean source.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
isSubtileMultiDU = U.isSubtileMultiDU

pytestmark = pytest.mark.unit


def test_left_side_smaller_returns_true():
    # kernel where _DepthUA (present) < du drives the result True, while
    # _DepthUB == du (right side False). Original: True or False == True.
    #
    # Kills:
    #   mutmut_1  (du = None): kernel.get("_DepthUA", du) == 8; 8 < None -> TypeError.
    #   mutmut_2/3/4 (kernel["XXDepthUXX"]/["depthu"]/["DEPTHU"]): KeyError on lookup.
    #   mutmut_5  (or -> and): True and False == False (differs from True).
    #   mutmut_6  (get(None, du) first key): du < du == False; then du<du False -> False.
    #   mutmut_8  (get(du) first, no default -> None): None < du -> TypeError.
    #   mutmut_10/11/12 (first key -> "XX_DepthUAXX"/"_depthua"/"_DEPTHUA"): default du,
    #                    so du<du False; right du<du False -> False.
    kernel = {"DepthU": 16, "_DepthUA": 8, "_DepthUB": 16}
    assert isSubtileMultiDU(kernel) is True


def test_right_side_smaller_returns_true():
    # Left side False (_DepthUA == du), right side (_DepthUB present) < du drives
    # the result True. Original: False or True == True.
    #
    # Kills:
    #   mutmut_14 (second get(None, du)): du < du == False -> overall False.
    #   mutmut_16 (second get(du), no default -> None): None < du -> TypeError.
    #   mutmut_18/19/20 (second key -> "XX_DepthUBXX"/"_depthub"/"_DEPTHUB"): default du,
    #                    so du<du False -> overall False.
    #   (also re-kills mutmut_5: False and True == False.)
    kernel = {"DepthU": 16, "_DepthUA": 16, "_DepthUB": 8}
    assert isSubtileMultiDU(kernel) is True


def test_both_equal_to_depthu_returns_false():
    # Both per-uid DepthU present and EQUAL to du: strict-< means NOT multi-DU.
    # Original: (16<16=False) or (16<16=False) == False.
    #
    # Kills:
    #   mutmut_13 (first < -> <=): (16<=16=True) or ... == True (differs from False).
    #   mutmut_21 (second < -> <=): False or (16<=16=True) == True (differs from False).
    kernel = {"DepthU": 16, "_DepthUA": 16, "_DepthUB": 16}
    assert isSubtileMultiDU(kernel) is False


def test_both_larger_than_depthu_returns_false():
    # Both per-uid DepthU strictly larger than du: not multi-DU.
    # Pins the plain False happy path and the strict comparison direction.
    kernel = {"DepthU": 16, "_DepthUA": 32, "_DepthUB": 32}
    assert isSubtileMultiDU(kernel) is False


def test_missing_depthua_defaults_to_du_not_none():
    # _DepthUA absent: original defaults the .get to du, so left compares
    # du < du == False (no error). Right (_DepthUB == du) also False -> False.
    #
    # Kills:
    #   mutmut_7 (get("_DepthUA", None)): None < du -> TypeError (differs from False).
    #   mutmut_9 (get("_DepthUA", ) no default -> None): None < du -> TypeError.
    kernel = {"DepthU": 16, "_DepthUB": 16}
    assert isSubtileMultiDU(kernel) is False


def test_missing_depthub_defaults_to_du_not_none():
    # _DepthUB absent AND left side False (so the right side is evaluated, not
    # short-circuited). Original defaults the .get to du: du < du == False.
    #
    # Kills:
    #   mutmut_15 (get("_DepthUB", None)): None < du -> TypeError (differs from False).
    #   mutmut_17 (get("_DepthUB", ) no default -> None): None < du -> TypeError.
    #   (also re-kills mutmut_16: second get(du) -> None < du -> TypeError.)
    kernel = {"DepthU": 16, "_DepthUA": 16}
    assert isSubtileMultiDU(kernel) is False


def test_missing_depthu_key_raises_keyerror():
    # du = kernel["DepthU"] is a subscript, not a .get: absence raises KeyError.
    # Pins the literal top-level key and the raising (non-defaulting) access.
    kernel = {"_DepthUA": 8, "_DepthUB": 8}
    with pytest.raises(KeyError):
        isSubtileMultiDU(kernel)


def test_no_per_uid_keys_returns_false():
    # Only DepthU present: both sides default to du, du<du == False on both.
    kernel = {"DepthU": 16}
    assert isSubtileMultiDU(kernel) is False
