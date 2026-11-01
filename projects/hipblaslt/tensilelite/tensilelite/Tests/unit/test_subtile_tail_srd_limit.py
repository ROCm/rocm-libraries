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
