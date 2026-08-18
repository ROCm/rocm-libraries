################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id 82034243636ff093cba429f81384b737464cc0c9

Predicate : kernel["SuppressNoLoadLoop"]
Site      : Tensile/KernelWriterAssembly.py:7094  (calculateLoopNumIter, if)
Solver    : z3 4.16.0  — SAT  (solver-backed-under-assumptions)

What the predicate reads
------------------------
At the branch site ``kernel["SuppressNoLoadLoop"]`` is the *fully-derived*
boolean, not the raw YAML key.  Its value is fixed by three sequential
override steps in ``Solution.assignDerivedParameters`` (Solution.py):

  1. 1845-1847  validity gate: a True yaml value is forced back to False
                unless  (bufferLoad AND PGR==1 AND GSU in {1,-1}).
                bufferLoad = BufferLoad AND KernelLanguage=="Assembly" (1828).
  2. 2282-2283  HalfPLR override: HalfPLR != 0  unconditionally sets True.
  3. 3850-3869  TailloopInNll override: when TailloopInNll survives its own
                validity block it unconditionally sets False (the final word).

Classification : solver-backed-under-assumptions.
All inputs are YAML solution parameters (no os/fs/gpu probe participates).
The encoder boundary is ``TailloopInNll_effective`` — whether TailloopInNll
survives its inner validity block depends on parameters OUTSIDE the seeded
domain (EnableMatrixInstruction, NoTailLoop, DepthU power-of-2, LocalSplitU,
ProblemType MXBlockA/B, asmCaps HasWMMA).  We model it as a free boolean
that can only be effective when TailloopInNll==True, and pin behaviour for
both polarities of every override.

z3 witnesses (cross-checked against a faithful re-impl of the 3 overrides):
  TRUE  (path_A direct) : yaml=True, HalfPLR=0, BufferLoad=True, KL=Assembly,
                          PGR=1, GSU=-1, TailEff=False        -> True
  TRUE  (path_B HalfPLR): HalfPLR=1 (any gate state)          -> True
  FALSE (gate fails)    : yaml=True, PGR=0                    -> False
  FALSE (tail override) : HalfPLR=1 but TailEff=True          -> False
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Faithful re-implementation of the derivation (mirrors Solution.py 1828,
# 1845-1847, 2282-2283, 3850-3869).  Pure; no Tensile import required.
# ---------------------------------------------------------------------------

def _derive_suppress_no_load_loop(
    yaml_snll,
    half_plr,
    tail_nll_effective,
    buffer_load,
    kernel_language,
    prefetch_global_read,
    global_split_u,
):
    snll = bool(yaml_snll)
    buffer_load_eff = buffer_load and kernel_language == "Assembly"  # 1828
    if snll:                                                          # 1845-1847
        if not (
            buffer_load_eff
            and prefetch_global_read == 1
            and (global_split_u == 1 or global_split_u == -1)
        ):
            snll = False
    if half_plr != 0:                                                # 2282-2283
        snll = True
    if tail_nll_effective:                                           # 3850-3869
        snll = False
    return snll


# ---------------------------------------------------------------------------
# TRUE witnesses
# ---------------------------------------------------------------------------

def test_snll_true_path_a_direct_yaml():
    """z3 TRUE: yaml True survives the gate (PGR==1, GSU in {1,-1}, bufferLoad)."""
    assert _derive_suppress_no_load_loop(
        yaml_snll=True, half_plr=0, tail_nll_effective=False,
        buffer_load=True, kernel_language="Assembly",
        prefetch_global_read=1, global_split_u=-1,
    ) is True


def test_snll_true_path_b_halfplr_override():
    """z3 TRUE: HalfPLR!=0 forces True regardless of the gate (yaml False, PGR 0)."""
    assert _derive_suppress_no_load_loop(
        yaml_snll=False, half_plr=1, tail_nll_effective=False,
        buffer_load=False, kernel_language="Assembly",
        prefetch_global_read=0, global_split_u=-1,
    ) is True


# ---------------------------------------------------------------------------
# FALSE witnesses
# ---------------------------------------------------------------------------

def test_snll_false_gate_rejects_yaml_true():
    """z3 FALSE: yaml True but PGR!=1 -> gate forces False (HalfPLR 0)."""
    assert _derive_suppress_no_load_loop(
        yaml_snll=True, half_plr=0, tail_nll_effective=False,
        buffer_load=True, kernel_language="Assembly",
        prefetch_global_read=0, global_split_u=-1,
    ) is False


def test_snll_false_tailloop_override_beats_halfplr():
    """z3 FALSE: TailloopInNll effective is the final word, even over HalfPLR True."""
    assert _derive_suppress_no_load_loop(
        yaml_snll=False, half_plr=1, tail_nll_effective=True,
        buffer_load=False, kernel_language="Assembly",
        prefetch_global_read=0, global_split_u=-1,
    ) is False


# ---------------------------------------------------------------------------
# Override-precedence pins (document the sequential order that fixes the value)
# ---------------------------------------------------------------------------

def test_snll_kernel_language_non_assembly_disables_bufferload():
    """KernelLanguage!='Assembly' zeroes bufferLoad -> gate fails -> False."""
    assert _derive_suppress_no_load_loop(
        yaml_snll=True, half_plr=0, tail_nll_effective=False,
        buffer_load=True, kernel_language="Source",
        prefetch_global_read=1, global_split_u=1,
    ) is False


def test_snll_gsu_two_rejected_by_gate():
    """GSU==2 is outside {1,-1} -> gate forces a yaml-True value to False."""
    assert _derive_suppress_no_load_loop(
        yaml_snll=True, half_plr=0, tail_nll_effective=False,
        buffer_load=True, kernel_language="Assembly",
        prefetch_global_read=1, global_split_u=2,
    ) is False
