# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The WMMA mnemonic support query and the naming it depends on.

The check rejects a MatrixInstruction whose shape/data-type pair has no opcode on
the target, so the two ways it can be wrong are both covered here: naming an
instruction the emitter would not emit (false reject) and failing to name one at
all (silent no-op).
"""

import pytest
import rocisa

from Tensile.Common.DataType import DataType
from Tensile.Common.MatrixInstructionNaming import matrixInstructionMnemonic
from Tensile.SolutionStructs.Validators.MatrixInstruction import (
    unsupportedMatrixInstructionMnemonic,
)

GFX1250 = (12, 5, 0)

pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(GFX1250),
    reason="needs a rocisa built with the StinkyTofu gfx1250 backend",
)


def mnemonic(mi4, dtype, **kwargs):
    dt = DataType(dtype)
    return matrixInstructionMnemonic(
        GFX1250, 32, mi4, dt, dt, DataType(kwargs.pop("compute", "float")), **kwargs
    )


def solutionFor(dtype, compute="float", **kwargs):
    """A solution shaped like library-logic YAML: data types as raw enum ints."""
    problemType = {
        "MacDataTypeA": DataType(dtype).value,
        "MacDataTypeB": DataType(dtype).value,
        "ComputeDataType": DataType(compute).value,
        "DataType": DataType(dtype).value,
        "F32XdlMathOp": DataType("float").value,
        "Sparse": 0,
    }
    problemType.update(kwargs.pop("problemType", {}))
    solution = {"WavefrontSize": 32, "MFMA_BF16_1K": 0, "ProblemType": problemType}
    solution.update(kwargs)
    return solution


def unsupported(solution, mi4):
    ptype = solution["ProblemType"]
    return unsupportedMatrixInstructionMnemonic(
        solution,
        GFX1250,
        mi4,
        DataType(ptype["MacDataTypeA"]),
        DataType(ptype["MacDataTypeB"]),
        DataType(ptype["ComputeDataType"]),
        ptype["Sparse"],
        False,
    )


@pytest.mark.parametrize(
    "mi4,dtype,expected",
    [
        ([16, 16, 32, 1], "bfloat16", "v_wmma_f32_16x16x32_bf16"),
        ([16, 16, 32, 1], "half", "v_wmma_f32_16x16x32_f16"),
        ([16, 16, 4, 1], "float", "v_wmma_f32_16x16x4_f32"),
    ],
)
def test_names_the_supported_instruction(mi4, dtype, expected):
    assert mnemonic(mi4, dtype) == expected
    assert unsupported(solutionFor(dtype), mi4) is None


def test_wmma_spells_int8_as_iu8():
    # WMMA has v_wmma_i32_*_iu8, not _i8; naming it i8 would reject every int8 kernel.
    assert mnemonic([16, 16, 64, 1], "int8", compute="int32") == "v_wmma_i32_16x16x64_iu8"
    assert unsupported(solutionFor("int8", compute="int32"), [16, 16, 64, 1]) is None


def test_f32_emulation_is_named_as_the_bf16_it_emits():
    # UseF32XEmulation multiplies bf16 halves, so xf32 never reaches the mnemonic.
    assert (
        mnemonic([16, 16, 32, 1], "xfloat32", useF32XEmulation=True)
        == "v_wmma_f32_16x16x32_bf16"
    )
    solution = solutionFor("float", UseF32XEmulation=True, EnableF32XdlMathOp=True)
    assert unsupported(solution, [16, 16, 32, 1]) is None


@pytest.mark.parametrize(
    "mxBlock,expected",
    [
        (32, "v_wmma_scale_f32_16x16x128_f8f6f4"),
        (16, "v_wmma_scale16_f32_16x16x128_f8f6f4"),
    ],
)
def test_mx_block_selects_the_scale_encoding(mxBlock, expected):
    # MX kernels are emitted as MXMFMAInstruction; block 16 is a different opcode.
    assert mnemonic([16, 16, 128, 1], "float8", mxBlock=mxBlock) == expected


@pytest.mark.parametrize(
    "mi4,dtype",
    [
        ([16, 16, 4, 1], "half"),  # shape exists, but only as v_wmma_f32_16x16x4_f32
        ([16, 16, 64, 1], "bfloat16"),
        ([16, 16, 8, 1], "half"),
    ],
)
def test_rejects_shape_and_type_pairs_with_no_opcode(mi4, dtype):
    rejected = unsupported(solutionFor(dtype), mi4)
    assert rejected is not None
    assert not rocisa.isMnemonicSupportedByStinkyTofu(rejected, GFX1250)


def test_reads_raw_enum_int_problem_types():
    # Library-logic YAML stores data types as ints; the check must not raise on them.
    solution = solutionFor("half")
    assert all(isinstance(v, int) for v in solution["ProblemType"].values())
    assert unsupported(solution, [16, 16, 32, 1]) is None


def test_leaves_thread_vgpr_state_alone():
    # setKernel clears the thread's VGPR index map, so the query has to put it back.
    ti = rocisa.rocIsa.getInstance()
    ti.setVgprIdx("vgprAlpha", 7)
    ti.setVgprMsb(1)

    mnemonic([16, 16, 32, 1], "bfloat16")

    assert ti.getVgprIdx().get("vgprAlpha") == 7
    assert ti.getVgprMsb() == 1


def test_restores_nothing_when_no_kernel_was_pinned(monkeypatch):
    """The stinkytofu adaptor starts with KernelInfo.isa None, which setKernel
    cannot express; restoring it blindly would raise TypeError on the first call."""
    import Tensile.Common.MatrixInstructionNaming as naming

    real = naming.rocIsa.getInstance()

    class UnpinnedKernelInfo:
        isa = None
        wavefrontSize = 0

    class FakeTi:
        def __init__(self):
            self.setKernelCalls = []

        def getKernel(self):
            return UnpinnedKernelInfo()

        def getVgprIdx(self):
            return {}

        def getVgprMsb(self):
            return 0

        def setKernel(self, isa, wavefrontSize):
            self.setKernelCalls.append((isa, wavefrontSize))
            real.setKernel(isa, wavefrontSize)

        def setVgprIdx(self, name, idx):
            real.setVgprIdx(name, idx)

        def setVgprMsb(self, msb):
            real.setVgprMsb(msb)

    fake = FakeTi()
    monkeypatch.setattr(naming.rocIsa, "getInstance", staticmethod(lambda: fake))

    assert mnemonic([16, 16, 32, 1], "bfloat16") == "v_wmma_f32_16x16x32_bf16"
    assert fake.setKernelCalls == [(GFX1250, 32)]  # pinned once, no restore attempted


def test_absorbs_only_the_unnameable_type():
    # Complex is emulated with real matrix ops and has no matrix InstType, so the
    # backend raises and the check declines rather than rejecting the solution.
    assert unsupported(solutionFor("complexFloat", compute="complexFloat"), [16, 16, 4, 1]) is None

    # A wrong argument must still surface instead of being swallowed.
    with pytest.raises(Exception):
        matrixInstructionMnemonic(
            GFX1250, 32, [16, 16, 32, 1], DataType("half"), DataType("half"), None
        )
