# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Characterization tests for the LoopModel register-sizing helpers.

Two on ``KernelWriterAssembly``:

``tdmTpByChar``      resolves a tensorChar to its tensor-parameter dict, looking
                     through each operand's ``MX`` scale companion.
``loopModelTailValuRegs``
                     sizes the tail loop's ValuA/ValuB window: one MatrixInstK
                     slab plus whatever the ring depth carries ahead of it.

and the module-level ``operandNumVgprBuffer`` in
``Components.LoopModel.Registers`` that the second one asks for the ring depth.

All three read state off the writer rather than emitting, so they are driven
through a stub carrying only the fields they touch -- the same shape
``test_custom_kernel_staggeru_metadata`` uses, and the reason these are unit
assertions rather than snapshots.
"""

import pytest

from tensilelite.Components.LoopModel.Registers import operandNumVgprBuffer
from tensilelite.KernelWriterAssembly import KernelWriterAssembly


class _Bytes:
    """Minimal stand-in for a DataType: only ``numBytes`` is consulted here."""

    def __init__(self, n):
        self._n = n

    def numBytes(self):
        return self._n


class _OperandState:
    def __init__(self, perBlock):
        self.numVgprValuPerBlock = perBlock


class _States:
    def __init__(self, perBlockA=8, perBlockB=8, buffers=None, shared=1):
        self.bpr = 4
        self.a = _OperandState(perBlockA)
        self.b = _OperandState(perBlockB)
        buffers = buffers or {}
        self.numVgprBuffer = shared
        self.numVgprBufferA = buffers.get("A", 0)
        self.numVgprBufferB = buffers.get("B", 0)
        self.numVgprBufferMXSA = buffers.get("MXSA", 0)
        self.numVgprBufferMXSB = buffers.get("MXSB", 0)


class _Writer:
    """Enough of KernelWriterAssembly for the register-sizing helpers."""

    def __init__(self, tPA=None, tPB=None, states=None):
        self.tPA = tPA
        self.tPB = tPB
        self.states = states or _States()

    def tdmTpByChar(self, tc):
        return KernelWriterAssembly.tdmTpByChar(self, tc)

    def loopModelTailValuRegs(self, kernel, tc):
        return KernelWriterAssembly.loopModelTailValuRegs(self, kernel, tc)


def _kernel(**over):
    kernel = {
        "EnableMatrixInstruction": True,
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
        "MIInputPerThreadA": 8,
        "MIInputPerThreadB": 8,
        "DepthU": 64,
        "MatrixInstK": 32,
        "LocalSplitU": 1,
        "LoopIters": 2,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        # Read off the kernel, not off ProblemType, unlike its Sparse companion.
        "DirectToVgprSparseMetadata": False,
        "ProblemType": {
            "DataTypeA": _Bytes(2),
            "DataTypeB": _Bytes(2),
            "Sparse": 0,
        },
    }
    problem = over.pop("ProblemType", None)
    kernel.update(over)
    if problem:
        kernel["ProblemType"].update(problem)
    return kernel


# --- tdmTpByChar ----------------------------------------------------------


def test_tdm_tp_by_char_finds_each_data_operand():
    tPA = {"tensorChar": "A", "MX": None}
    tPB = {"tensorChar": "B", "MX": None}
    writer = _Writer(tPA, tPB)
    assert writer.tdmTpByChar("A") is tPA
    assert writer.tdmTpByChar("B") is tPB


def test_tdm_tp_by_char_looks_through_the_scale_companion():
    mxsa = {"tensorChar": "MXSA"}
    tPA = {"tensorChar": "A", "MX": mxsa}
    writer = _Writer(tPA, {"tensorChar": "B", "MX": None})
    assert writer.tdmTpByChar("MXSA") is mxsa


def test_tdm_tp_by_char_skips_an_operand_that_is_not_set_up():
    tPB = {"tensorChar": "B", "MX": None}
    assert _Writer(None, tPB).tdmTpByChar("B") is tPB


def test_tdm_tp_by_char_raises_for_an_unknown_tensor_char():
    writer = _Writer({"tensorChar": "A", "MX": None}, {"tensorChar": "B", "MX": None})
    with pytest.raises(KeyError, match="no tensor parameters for Z"):
        writer.tdmTpByChar("Z")


# --- operandNumVgprBuffer -------------------------------------------------


@pytest.mark.parametrize("tc", ["A", "B", "MXSA", "MXSB"])
def test_operand_num_vgpr_buffer_prefers_the_per_operand_count(tc):
    writer = _Writer(states=_States(buffers={tc: 3}, shared=1))
    assert operandNumVgprBuffer(writer, tc) == 3


@pytest.mark.parametrize("tc", ["A", "B", "MXSA", "MXSB"])
def test_operand_num_vgpr_buffer_falls_back_to_the_shared_count(tc):
    """A zero per-operand field means allocation has not run yet."""
    writer = _Writer(states=_States(shared=2))
    assert operandNumVgprBuffer(writer, tc) == 2


# --- loopModelTailValuRegs ------------------------------------------------


def test_tail_valu_regs_is_zero_without_matrix_instructions():
    writer = _Writer(states=_States())
    assert writer.loopModelTailValuRegs(_kernel(EnableMatrixInstruction=False), "A") == 0


def test_tail_valu_regs_is_one_slab_at_ring_depth_one():
    """2 tiles * 8 inputs * 2 bytes = 32 bytes = 8 registers, and no ring ahead."""
    writer = _Writer(states=_States(shared=1))
    assert writer.loopModelTailValuRegs(_kernel(), "A") == 8


def test_tail_valu_regs_ignores_ring_depth_without_a_buffered_source():
    """The ring depth only caps the buffer count; nothing here raises it above 1."""
    writer = _Writer(states=_States(perBlockA=8, buffers={"A": 3}))
    assert writer.loopModelTailValuRegs(_kernel(), "A") == 8


def test_tail_valu_regs_rounds_a_partial_register_up():
    """1 tile * 1 input * 1 byte is a quarter of a register, and still costs one."""
    writer = _Writer(states=_States(shared=1))
    kernel = _kernel(MIWaveTileA=1, MIInputPerThreadA=1,
                     ProblemType={"DataTypeA": _Bytes(1)})
    assert writer.loopModelTailValuRegs(kernel, "A") == 1


def test_tail_valu_regs_unrolls_over_depthu_when_direct_to_vgpr_is_on():
    """DepthU/(MatrixInstK*LocalSplitU) = 2 buffers, capped by the ring depth."""
    writer = _Writer(states=_States(perBlockA=8, buffers={"A": 4}))
    assert writer.loopModelTailValuRegs(_kernel(DirectToVgprA=True), "A") == (2 - 1) * 8 + 8


def test_tail_valu_regs_uses_loop_iters_for_sparse_metadata_in_vgprs():
    writer = _Writer(states=_States(perBlockB=8, buffers={"B": 4}))
    kernel = _kernel(LoopIters=3, DirectToVgprSparseMetadata=True,
                     ProblemType={"Sparse": 1})
    assert writer.loopModelTailValuRegs(kernel, "B") == (3 - 1) * 8 + 8


def test_tail_valu_regs_caps_the_buffer_count_at_the_ring_depth():
    """DirectToVgpr asks for 2; a ring of 1 is what the allocation actually has."""
    writer = _Writer(states=_States(perBlockA=8, buffers={"A": 1}))
    assert writer.loopModelTailValuRegs(_kernel(DirectToVgprA=True), "A") == 8
