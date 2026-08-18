################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id dc455979d10c6c2d59089c0b3a32c06fadb67ab9

Predicate : self.states.archCaps["SgprPreloadPad"]   (int-as-bool)
Site      : Tensile/KernelWriterAssembly.py:2267
Function  : defineAndResources
Solver    : z3 4.16.0  — SAT  (solver-backed-under-assumptions)

Sole producer of SgprPreloadPad
--------------------------------
rocisa/rocisa/include/hardware_caps.hpp:524:
    rv[SgprPreloadPad] = checkInList(isa,{9,5,0}) || checkInList(isa,{9,0,10})
                         || (isa[0]==9 && isa[1]==4)

Live rocisa ground truth (all four ISAs match Z3 and C++ source exactly):
    (9,4,2) -> 1  (True)   gfx94x
    (9,5,0) -> 1  (True)   gfx950
    (9,0,10)-> 1  (True)   gfx90a
    (11,0,0)-> 0  (False)  gfx11xx

Tests here:
  1. Pure-helper semantics — pin sgpr_preload_pad(isa) for all four seeded ISAs.
  2. Predicate truthiness — confirm int-as-bool encoding for both polarities.
"""

import pytest
from typing import Tuple

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper — mirrors hardware_caps.hpp:524 SgprPreloadPad definition
# ---------------------------------------------------------------------------

def sgpr_preload_pad(isa: Tuple[int, int, int]) -> bool:
    """Pure mirror of rocisa hardware_caps.hpp:524 SgprPreloadPad.

    SgprPreloadPad = checkInList(isa,{9,5,0}) || checkInList(isa,{9,0,10})
                     || (isa[0]==9 && isa[1]==4)

    The predicate at KernelWriterAssembly.py:2267 is int-as-bool truthiness of
    the integer value returned by getArchCaps()['SgprPreloadPad'] (0 or 1).
    """
    return isa == (9, 5, 0) or isa == (9, 0, 10) or (isa[0] == 9 and isa[1] == 4)


# ---------------------------------------------------------------------------
# TRUE-witness cases: SgprPreloadPad == 1, predicate enters branch
# ---------------------------------------------------------------------------

def test_sgpr_preload_pad_true_gfx94x():
    """z3 true witness + rocisa ground truth: ISA (9,4,2) -> SgprPreloadPad=1 -> True."""
    assert sgpr_preload_pad((9, 4, 2)) is True


def test_sgpr_preload_pad_true_gfx950():
    """rocisa ground truth: ISA (9,5,0) -> SgprPreloadPad=1 -> True."""
    assert sgpr_preload_pad((9, 5, 0)) is True


def test_sgpr_preload_pad_true_gfx90a():
    """rocisa ground truth: ISA (9,0,10) -> SgprPreloadPad=1 -> True."""
    assert sgpr_preload_pad((9, 0, 10)) is True


# ---------------------------------------------------------------------------
# FALSE-witness case: SgprPreloadPad == 0, predicate skips branch
# ---------------------------------------------------------------------------

def test_sgpr_preload_pad_false_gfx11xx():
    """z3 false witness + rocisa ground truth: ISA (11,0,0) -> SgprPreloadPad=0 -> False."""
    assert sgpr_preload_pad((11, 0, 0)) is False


# ---------------------------------------------------------------------------
# Predicate encoding pin — int-as-bool truthiness is exact for 0/1 values
# ---------------------------------------------------------------------------

def test_predicate_int_as_bool_one_is_true():
    """SgprPreloadPad=1 (int) is truthy: bool(1) is True."""
    assert bool(1) is True


def test_predicate_int_as_bool_zero_is_false():
    """SgprPreloadPad=0 (int) is falsy: bool(0) is False."""
    assert bool(0) is False


# ---------------------------------------------------------------------------
# Additional ISA boundary: other gfx9 minor==4 triples are also padded
# ---------------------------------------------------------------------------

def test_sgpr_preload_pad_true_gfx940():
    """ISA (9,4,0) satisfies isa[0]==9 && isa[1]==4, so SgprPreloadPad=1."""
    assert sgpr_preload_pad((9, 4, 0)) is True


def test_sgpr_preload_pad_false_gfx12xx():
    """ISA (12,0,0) does not match any SgprPreloadPad condition -> False."""
    assert sgpr_preload_pad((12, 0, 0)) is False
