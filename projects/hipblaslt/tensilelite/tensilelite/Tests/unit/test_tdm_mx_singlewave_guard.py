################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Solution acceptance for single-wave MX scale TDM (gfx1250).

At one wave a single descriptor covers the whole scale tile, and it is only
correct for one scale K group per fetch: past that the scales landing in LDS are
not the ones the MFMA reads. Wave-separated kernels split the K groups across
wave components instead, so they are unaffected.

The reject lives in ``depthUIteration`` rather than ``assignDerivedParameters``
for two reasons, both pinned here: ``DepthU`` may still be -1 (auto) in the
earlier pass, and rejecting there would shadow the TDMFuse/PAP arms, whose own
diagnostics ``test_TDMFuse`` and ``test_r4_tdmsplit_pap_mx_char`` assert on.
"""

import pytest

# Reuse the gfx1250 solution-derivation harness; _derive returns the derived
# Solution plus whatever rejection reasons it printed. The toolchain fixtures
# come along so pytest resolves them by name here (they auto-skip when
# amdclang++ cannot target gfx1250).
from test_TDMFuse import (  # noqa: F401  (fixtures used by name)
    _ONE_WAVE_MI,
    _ONE_WAVE_WG,
    _derive,
    _gp_gfx1250,
    assembler,
    gfx1250_iim,
)

pytestmark = pytest.mark.unit

_CLAUSE = "single-wave MX scale TDM requires DepthU <= MatrixInstK"
# _ONE_WAVE_MI is a MatrixInstK=128 instruction.
_MATRIX_INST_K = 128
_NO_MX = {"MacDataTypeA": "F8", "MacDataTypeB": "F8",
          "MXBlockA": 0, "MXBlockB": 0}


def _single_wave(gfx1250_iim, assembler, capsys, **overrides):
    return _derive(gfx1250_iim, assembler, capsys, TDMFuse=0,
                   MatrixInstruction=_ONE_WAVE_MI, WorkGroup=_ONE_WAVE_WG,
                   **overrides)


@pytest.mark.parametrize("depthU", [2 * _MATRIX_INST_K, 4 * _MATRIX_INST_K])
def test_single_wave_mx_rejects_multiple_k_groups(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, depthU):
    """More than one scale K group per fetch is refused, with the reason why."""
    sol, out = _single_wave(gfx1250_iim, assembler, capsys, DepthU=depthU)
    assert sol.get("Valid") is False
    assert _CLAUSE in out, f"expected the scale K group reason, got: {out}"


def test_single_wave_mx_accepts_one_k_group(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """DepthU == MatrixInstK is the accepted boundary, not a rejected one."""
    sol, out = _single_wave(gfx1250_iim, assembler, capsys, DepthU=_MATRIX_INST_K)
    assert sol.get("Valid") is True, out
    assert _CLAUSE not in out


def test_single_wave_mx_auto_depthu_falls_through_to_one_k_group(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """DepthU=-1 keeps searching instead of losing the kernel.

    The reject marks the candidate invalid rather than the solution, so the
    auto-search walks its list down to one that holds a single scale K group.
    """
    sol, out = _single_wave(gfx1250_iim, assembler, capsys, DepthU=-1)
    assert sol.get("Valid") is True, out
    assert sol["DepthU"] == _MATRIX_INST_K, (
        f"auto-search settled on DepthU={sol['DepthU']}, not {_MATRIX_INST_K}"
    )


def test_wave_separated_mx_keeps_multiple_k_groups(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """The wave-separated path divides the groups, so it is left alone."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0,
                       DepthU=2 * _MATRIX_INST_K)
    assert sol.get("Valid") is True, out
    assert _CLAUSE not in out


def test_single_wave_without_mx_is_unaffected(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """No scales, no scale descriptor: plain TDM keeps its DepthU range."""
    sol, out = _single_wave(gfx1250_iim, assembler, capsys,
                            DepthU=2 * _MATRIX_INST_K, ProblemType=_NO_MX)
    assert sol.get("Valid") is True, out
    assert _CLAUSE not in out, f"the guard fired without MX scales: {out}"


def test_single_wave_mx_noreject_still_derives(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """NoReject keeps the candidate and finishes derivation.

    reject() leaves Valid set in that mode. Returning from depthUIteration
    anyway would stop the search before LoopIters exists, on a solution that
    still claims to be valid.
    """
    sol, out = _single_wave(gfx1250_iim, assembler, capsys, NoReject=True,
                            DepthU=2 * _MATRIX_INST_K)
    assert sol.get("Valid") is True, out
    assert sol["DepthU"] == 2 * _MATRIX_INST_K
    assert sol["LoopIters"] >= 1
    assert _CLAUSE not in out
