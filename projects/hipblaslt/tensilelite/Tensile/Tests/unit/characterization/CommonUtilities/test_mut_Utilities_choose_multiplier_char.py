################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.choose_multiplier``.

``choose_multiplier(d, N, p)`` computes the magic-division constants used by the
assembly kernel writer:

  l      = ceil(log2(d))
  shPost = l
  mlow   = 2**(N+l) // d
  mhigh  = (2**(N+l) + 2**(N+l-p)) // d
  while (mlow // 2) < (mhigh // 2) and shPost > 0:
      mlow  //= 2
      mhigh //= 2
      shPost -= 1
  return mhigh, shPost, l

These tests pin the ACTUAL current return tuple ``(mhigh, shPost, l)`` at inputs
chosen so that each surviving mutant produces a different tuple. They pass on
clean source and fail once the corresponding mutant is applied.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
choose_multiplier = U.choose_multiplier

pytestmark = pytest.mark.unit


def test_l_uses_base_2_log():
    # Kills mutmut_7: `math.log(d, 2)` -> `math.log(d)` (base e).
    # For d=5: ceil(log2(5)) == 3 (original) but ceil(ln(5)) == 2 (mutant),
    # which changes the returned l (and the shift count derived from it).
    assert choose_multiplier(5, 4, 4) == (13, 2, 3)


def test_mlow_initial_value_and_loop_condition():
    # Kills mutmut_11 (mlow = 2**(N+l) / d, float),
    #       mutmut_12 (mlow = 2 * (N+l) // d),
    #       mutmut_14 (mlow = 2**(N-l) // d),
    #       mutmut_27 ((mlow // 3) < (mhigh // 2)),
    #       mutmut_28 (< replaced by <=),
    #       mutmut_29 ((mlow // 2) < (mhigh / 2)).
    # At d=2,N=4,p=4 the original loop iterates once: (mhigh,shPost,l)=(17,1,1).
    # Each of these mutants changes how many times the loop runs (or never enters
    # it for the integer-precision cases), yielding (8,0,1) instead.
    assert choose_multiplier(2, 4, 4) == (17, 1, 1)


def test_mlow_halving_inside_loop():
    # Kills mutmut_33 (mlow //= 2 -> mlow = 2),
    #       mutmut_35 (mlow //= 2 -> mlow //= 3).
    # At d=3,N=4,p=4 the original performs exactly one halving step and stops,
    # returning (11,1,2). Both mutants change the per-iteration update of mlow so
    # the loop continues an extra step, returning (5,0,2).
    assert choose_multiplier(3, 4, 4) == (11, 1, 2)


def test_float_division_loses_precision_for_large_operands():
    # Kills mutmut_11 (mlow = 2**(N+l) / d),
    #       mutmut_26 ((mlow / 2) < (mhigh // 2)),
    #       mutmut_34 (mlow /= 2).
    # These turn mlow / the comparison into float arithmetic. For small operands
    # the float result is exact, but at d=3,N=64,p=64 the magnitudes exceed
    # 2**53, so float rounding changes the loop's first comparison and the
    # returned tuple from (12297829382473034411, 1, 2) to a halved value.
    assert choose_multiplier(3, 64, 64) == (12297829382473034411, 1, 2)
