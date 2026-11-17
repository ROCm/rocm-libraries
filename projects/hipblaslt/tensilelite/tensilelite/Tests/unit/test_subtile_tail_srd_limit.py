# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit tests for the subtile tail-loop buffer limit on A, B and their MX scales.

The main loop bounds a TLU=1 operand's buffer to a window of DepthU K rows.
When the tail loop has fewer than DepthU rows left, that window runs past the
end of the tensor, so the limit has to shrink by the missing rows times the
K-row stride, not by the missing elements as for a K-contiguous operand.
A swizzled MX scale is block-linear in K whatever the operand layout, so its
limit shrinks the same way for TLU=0 and TLU=1.
"""

import contextlib

import pytest

from rocisa.container import sgpr

from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

_UNROLL_IDX = 3


class _States:
    groOffsetInMacroTile = 1
    unrollIdx = 0


class _TmpSgpr:
    idx = 100


class _Writer:
    """Stub with only what computeTailLoopSrdLimit reads."""

    isConstUnitStride = KernelWriterAssembly.isConstUnitStride

    def __init__(self, unitStrideTcs):
        self.states = _States()
        self._unitStrideTcs = unitStrideTcs

    def strideRef(self, tc, dim):
        if dim != _UNROLL_IDX and tc in self._unitStrideTcs:
            return "constStride%sI" % tc
        return sgpr("Strides%s+0" % tc)

    def loopCounterName(self, kernel, loopIdx):
        return "LoopCounterL"

    @contextlib.contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield _TmpSgpr()


def _emit(unitStrideTcs, tPs=None, depthU=256, mxScaleFormat=None):
    kernel = {
        "UseSubtileImpl": True,
        "DepthU": depthU,
        "ProblemType": {"IndexUnroll": _UNROLL_IDX, "MXBlockA": 32, "MXBlockB": 32},
    }
    if mxScaleFormat is not None:
        kernel["MXScaleFormat"] = mxScaleFormat
    if tPs is None:
        tPs = [{"tensorChar": "A", "tileIdx": 0, "bpeGR": 0.5},
               {"tensorChar": "B", "tileIdx": 1, "bpeGR": 0.5}]
    module = KernelWriterAssembly.computeTailLoopSrdLimit(_Writer(unitStrideTcs), kernel, tPs)
    return [line.strip() for line in str(module).splitlines() if line.strip()]


def _instructions(lines):
    return [line.split("//")[0].strip() for line in lines
            if not line.startswith("/*") and not line.startswith("//")]


def test_tlu1_operand_scales_the_tail_delta_by_its_k_row_stride():
    insts = _instructions(_emit(unitStrideTcs={"A"}))
    muls = [i for i in insts if i.startswith("s_mul_i32")]
    assert len(muls) == 1
    assert "StridesA" in muls[0]
    subs = [i for i in insts if i.startswith("s_sub_u32") and "+2]" in i]
    assert len(subs) == 2
    assert any("SrdA+2" in i for i in subs) and any("SrdB+2" in i for i in subs)


def test_k_contiguous_operands_keep_the_element_delta():
    insts = _instructions(_emit(unitStrideTcs=set()))
    assert not any(i.startswith("s_mul_i32") for i in insts)
    subs = [i for i in insts if i.startswith("s_sub_u32") and "+2]" in i]
    assert len(subs) == 2


_SCALE_TPS = [{"tensorChar": "MXSA", "tileIdx": 0, "bpeGR": 1},
              {"tensorChar": "MXSB", "tileIdx": 1, "bpeGR": 1}]


def test_tlu1_swizzled_scale_is_tightened_like_tlu0():
    tlu1 = _instructions(_emit(unitStrideTcs={"MXSA"}, tPs=_SCALE_TPS, depthU=512,
                               mxScaleFormat="HostPreSwizzle"))
    tlu0 = _instructions(_emit(unitStrideTcs=set(), tPs=_SCALE_TPS, depthU=512,
                               mxScaleFormat="HostPreSwizzle"))
    assert tlu1 == tlu0
    subs = [i for i in tlu1 if i.startswith("s_sub_u32") and "+2]" in i]
    assert any("SrdMXSA+2" in i for i in subs) and any("SrdMXSB+2" in i for i in subs)


def test_tlu1_unswizzled_scale_is_left_alone():
    insts = _instructions(_emit(unitStrideTcs={"MXSA"}, tPs=_SCALE_TPS, depthU=512,
                                mxScaleFormat="NoSwizzle"))
    assert not any(i.startswith("s_sub_u32") for i in insts)


def _advance(tightenLimit):
    from rocisa.code import Module
    from tensilelite.Components.Subtile.SubtileGREmit import emitSrdAdvance

    module = Module()
    emitSrdAdvance(module, "A", 128, writer=None, kernel=None, tightenLimit=tightenLimit)
    return _instructions([line.strip() for line in str(module).splitlines() if line.strip()])


def test_mainloop_advance_leaves_the_limit_alone():
    """Srd+2 is a generous tile-boundary window there; every K-block in it is live."""
    assert _advance(tightenLimit=False) == [
        "s_add_u32 s[sgprSrdA], s[sgprSrdA], 128",
        "s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0",
    ]


def test_tail_uid_advance_carries_the_tightened_limit_along():
    """A multi-unroll tail advances the base after computeTailLoopSrdLimit has
    already ended the window at K_rem, so the window has to shrink with it or the
    next unroll reads live memory past the data instead of clamping to zero."""
    insts = _advance(tightenLimit=True)
    assert insts[2:] == [
        "s_sub_u32 s[sgprSrdA+2], s[sgprSrdA+2], 128",
        "s_cselect_b32 s[sgprSrdA+2], 0, s[sgprSrdA+2]",
    ]


def test_tail_preloop_marks_only_the_uid_advances():
    from tensilelite.Components.Subtile.LogicalScheduler import GRIncOp

    assert GRIncOp(tensor="A", unrollId=0).tightenLimit is False
    assert GRIncOp(tensor="A", unrollId=1, tightenLimit=True).tightenLimit is True
