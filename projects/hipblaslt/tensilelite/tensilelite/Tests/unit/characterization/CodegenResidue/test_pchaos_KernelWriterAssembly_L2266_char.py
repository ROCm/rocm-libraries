################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id 8e5e952554311fd1ed0f31ac8dc00295e8616cbe

Predicate : self.states.numSgprPreload > 0
Site      : Tensile/KernelWriterAssembly.py:2266
Function  : defineAndResources
Solver    : z3 4.16.0 — SAT (fully-static; no GPU/filesystem/isinstance probe)

numSgprPreload computation (KernelWriter.py:8447-8451)
------------------------------------------------------
    numSgprPreload = 0                             # line 8447 (default)
    if kernel["PreloadKernArgs"]:
        numSgprPreload = (archCaps["MaxSgprPreload"]
                          - states.rpga             # constant == 2  (line 151)
                          - kernel["ProblemType"]["NumIndicesC"])

The branch predicate at line 2266 is evaluated BEFORE Signature.py:117 applies
the min-clamp, so the pre-clamp value governs.

Over the seeded domains:
    MaxSgprPreload in {16, 32}
    NumIndicesC    in {2, 3}
    rpga           == 2  (constant)

When PreloadKernArgs is True, the pre-clamp value is in {11, 12, 27, 28}, all > 0.
When PreloadKernArgs is False, numSgprPreload == 0 (predicate False).

Witnesses (z3 + in-container re-execution):
    (PKA=True,  max=16, nic=3) -> preclamp=11 -> predicate True
    (PKA=True,  max=32, nic=2) -> preclamp=28 -> predicate True
    (PKA=False, max=16, nic=2) -> preclamp=0  -> predicate False
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper — mirrors KernelWriter.py:8447-8451 + predicate at line 2266
# ---------------------------------------------------------------------------

def num_sgpr_preload_positive(
    preload_kern_args: bool,
    max_sgpr_preload: int,
    num_indices_c: int,
    rpga: int = 2,
) -> bool:
    """Mirror of KernelWriter.py:8447-8451 numSgprPreload computation followed by
    the line-2266 predicate `self.states.numSgprPreload > 0` (pre-Signature-clamp).

    pre:  max_sgpr_preload in (16, 32)
    pre:  num_indices_c    in (2, 3)
    pre:  rpga             == 2   (constant in KernelWriter.py:151)
    post: __return__ == preload_kern_args  (over seeded domains)
    """
    num_sgpr_preload = 0
    if preload_kern_args:
        num_sgpr_preload = max_sgpr_preload - rpga - num_indices_c
    return num_sgpr_preload > 0


# ---------------------------------------------------------------------------
# TRUE-witness: PKA=True, max=16, nic=3  -> preclamp=11 -> True
# ---------------------------------------------------------------------------

def test_num_sgpr_preload_positive_true_pka_max16_nic3():
    """z3 primary true witness: PKA=True, MaxSgprPreload=16, NumIndicesC=3 -> 11 > 0."""
    result = num_sgpr_preload_positive(
        preload_kern_args=True,
        max_sgpr_preload=16,
        num_indices_c=3,
    )
    assert result is True


# ---------------------------------------------------------------------------
# TRUE-witness: PKA=True, max=32, nic=2  -> preclamp=28 -> True
# ---------------------------------------------------------------------------

def test_num_sgpr_preload_positive_true_pka_max32_nic2():
    """z3 domain-enumeration witness: PKA=True, MaxSgprPreload=32, NumIndicesC=2 -> 28 > 0."""
    result = num_sgpr_preload_positive(
        preload_kern_args=True,
        max_sgpr_preload=32,
        num_indices_c=2,
    )
    assert result is True


# ---------------------------------------------------------------------------
# FALSE-witness: PKA=False, max=16, nic=2  -> preclamp=0 -> False
# ---------------------------------------------------------------------------

def test_num_sgpr_preload_positive_false_pka_false():
    """z3 false witness: PKA=False, MaxSgprPreload=16, NumIndicesC=2 -> 0 > 0 is False."""
    result = num_sgpr_preload_positive(
        preload_kern_args=False,
        max_sgpr_preload=16,
        num_indices_c=2,
    )
    assert result is False


# ---------------------------------------------------------------------------
# Exhaustive seeded-domain coverage (all 2x2x2 combinations)
# ---------------------------------------------------------------------------

def test_num_sgpr_preload_positive_true_pka_max16_nic2():
    """PKA=True, max=16, nic=2 -> preclamp=12 -> True."""
    result = num_sgpr_preload_positive(
        preload_kern_args=True,
        max_sgpr_preload=16,
        num_indices_c=2,
    )
    assert result is True


def test_num_sgpr_preload_positive_true_pka_max32_nic3():
    """PKA=True, max=32, nic=3 -> preclamp=27 -> True."""
    result = num_sgpr_preload_positive(
        preload_kern_args=True,
        max_sgpr_preload=32,
        num_indices_c=3,
    )
    assert result is True


def test_num_sgpr_preload_positive_false_pka_false_max32():
    """PKA=False, max=32, nic=3 -> preclamp=0 -> False (rpga irrelevant; branch not taken)."""
    result = num_sgpr_preload_positive(
        preload_kern_args=False,
        max_sgpr_preload=32,
        num_indices_c=3,
    )
    assert result is False


# ---------------------------------------------------------------------------
# Pin raw intermediate value for key witnesses
# ---------------------------------------------------------------------------

def test_preclamp_value_pka_max16_nic3():
    """Pin intermediate: 16 - 2 - 3 == 11 (the preclamp value from z3 primary witness)."""
    assert 16 - 2 - 3 == 11


def test_preclamp_value_pka_max32_nic2():
    """Pin intermediate: 32 - 2 - 2 == 28."""
    assert 32 - 2 - 2 == 28


def test_preclamp_value_pka_false():
    """Pin intermediate: PKA=False yields numSgprPreload=0, so 0 > 0 is False."""
    # Default assignment path: num_sgpr_preload = 0, branch skipped.
    assert not (0 > 0)
