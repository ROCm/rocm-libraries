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
from Tensile.Common.MatrixInstructionNaming import (
    backendCapsLoaded,
    matrixInstructionMnemonic,
    pinnedIsa,
)
from Tensile.SolutionStructs.Validators.MatrixInstruction import (
    unsupportedMatrixInstructionMnemonic,
    useF32XEmulationFor,
)

GFX1250 = (12, 5, 0)

pytestmark = pytest.mark.skipif(
    not rocisa.isSupportedByStinkyTofu(GFX1250),
    reason="needs a rocisa built with the StinkyTofu gfx1250 backend",
)


@pytest.fixture(scope="module", autouse=True)
def capsLoaded():
    """Load the gfx1250 capabilities the mnemonic depends on.

    The spelling of an f8f6f4 or scaled-WMMA instruction is chosen from the ISA's
    assembler capabilities, which rocisa only has after ``rocIsa.init``. A bare
    pytest process has never run one, so without this the rejection cases below
    would be asserting against capability defaults rather than gfx1250.
    """
    if backendCapsLoaded(GFX1250):
        return
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    try:
        makeIsaInfoMap([GFX1250], validateToolchain("amdclang++"))
    except Exception as e:  # no toolchain here; nothing to compare a mnemonic to
        pytest.skip(f"cannot load gfx1250 capabilities: {e}")
    if not backendCapsLoaded(GFX1250):
        pytest.skip("gfx1250 capabilities unavailable")


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


def unsupported(solution, mi4, **kwargs):
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
        **kwargs,
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


@pytest.mark.parametrize(
    "mi4,dtype,mxBlock,expected",
    [
        # The f8/bf8 solution that failed the gfx1250 build. With capabilities the
        # f8f6f4 encoding starts at K=128, so K=64 has an opcode of its own; without
        # them it starts at 64 and this is named v_wmma_scale_f32_16x16x64_f8f6f4.
        ([16, 16, 64, 1], "float8bfloat8", 0, "v_wmma_f32_16x16x64_fp8_bf8"),
        # The f4 shapes behind the datamover characterization tests. Without
        # capabilities the f4 tile threshold reads 0 and this becomes
        # v_wmma_scale_f32_16x16x128_f4, which gfx1250 only has at 32x16.
        ([16, 16, 128, 1], "float4", 0, "v_wmma_scale_f32_16x16x128_f8f6f4"),
        ([16, 16, 128, 1], "float4", 32, "v_wmma_scale_f32_16x16x128_f8f6f4"),
    ],
)
def test_names_the_shapes_a_capability_less_process_got_wrong(mi4, dtype, mxBlock, expected):
    """Every spelling here flips if the ISA's capabilities are missing.

    Each of these is a shipping gfx1250 solution that the check rejected when it ran
    in a joblib worker, so they are the cases that prove the capabilities reached it.
    """
    assert rocisa.isMnemonicSupportedByStinkyTofu(expected, GFX1250)
    assert mnemonic(mi4, dtype, mxBlock=mxBlock) == expected

    problemType = {"MXBlockA": mxBlock, "MXBlockB": mxBlock} if mxBlock else {}
    assert unsupported(solutionFor(dtype, problemType=problemType), mi4) is None


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
    assert unsupported(solution, [16, 16, 32, 1], useF32XEmulation=True) is None


@pytest.mark.parametrize(
    "solution,archCaps,expected",
    [
        ({"UseF32XEmulation": True}, {"HasF32XEmulation": 0}, True),
        ({"UseF32XEmulation": False}, {"HasF32XEmulation": 1}, False),
        # BenchmarkProblems validates before Solution.assignDerivedParameters runs and
        # matrixInstructionToMIParameters derives only EnableF32XdlMathOp, so the key
        # is missing rather than False on that path.
        ({"EnableF32XdlMathOp": True}, {"HasF32XEmulation": 1}, True),
        ({"EnableF32XdlMathOp": True}, {"HasF32XEmulation": 0}, False),
        ({}, {"HasF32XEmulation": 1}, False),
    ],
)
def test_derives_f32_emulation_when_the_solution_has_not_got_it(solution, archCaps, expected):
    assert useF32XEmulationFor(solution, archCaps) is expected


def test_xf32_survives_validation_before_derived_parameters_exist():
    """The generation path: EnableF32XdlMathOp is set, UseF32XEmulation is not yet.

    Reading the absent key as False names v_wmma_f32_16x16x32_xf32, which gfx1250 has
    no opcode for, so every xf32 solution would be rejected before it can be emitted.
    """
    solution = solutionFor("xfloat32", EnableF32XdlMathOp=True)
    assert "UseF32XEmulation" not in solution

    assert unsupported(solution, [16, 16, 32, 1], useF32XEmulation=False) is not None

    emulated = useF32XEmulationFor(solution, {"HasF32XEmulation": 1})
    assert emulated is True
    assert unsupported(solution, [16, 16, 32, 1], useF32XEmulation=emulated) is None


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


@pytest.mark.parametrize(
    "isa,expected",
    [
        (None, None),  # how the stinkytofu adaptor spells "never pinned"
        ((0, 0, 0), None),  # how rocisa spells it: a value-initialised KernelInfo
        (GFX1250, GFX1250),
    ],
)
def test_pinned_isa_reads_an_unpinned_thread_as_none(isa, expected):
    class KernelInfo:
        pass

    info = KernelInfo()
    info.isa = isa
    assert pinnedIsa(info) == expected


@pytest.mark.parametrize("unpinned", [None, (0, 0, 0)])
def test_restores_nothing_when_no_kernel_was_pinned(monkeypatch, unpinned):
    """Neither spelling of "no kernel" may be handed back to setKernel.

    The adaptor's None cannot be expressed at all, and rocisa's (0, 0, 0) would pin
    the thread to an ISA with an empty capability map, so every kernel the thread
    generated afterwards would read capability defaults instead of its target's.
    """
    import Tensile.Common.MatrixInstructionNaming as naming

    real = naming.rocIsa.getInstance()

    class UnpinnedKernelInfo:
        isa = unpinned
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


def test_declines_when_the_backend_has_no_capabilities_for_the_isa(monkeypatch):
    """A process that never ran rocIsa.init reads an empty capability map.

    rocisa indexes its per-ISA map with operator[], so the miss is silent and the
    query names an instruction from capability defaults -- v_wmma_scale_* for an
    f8 shape the emitter spells v_wmma_f32_16x16x64_fp8_bf8, say. TensileLogic runs
    its checks in joblib workers, so that miss reached a shipping solution and
    failed the build; declining is the only safe answer without capabilities.
    """
    import Tensile.SolutionStructs.Validators.MatrixInstruction as validator

    # A pair the suite above proves is rejected once capabilities are loaded.
    assert unsupported(solutionFor("half"), [16, 16, 4, 1]) is not None

    monkeypatch.setattr(validator, "backendCapsLoaded", lambda isa: False)
    assert unsupported(solutionFor("half"), [16, 16, 4, 1]) is None


def test_absorbs_only_the_unnameable_type():
    # Complex is emulated with real matrix ops and has no matrix InstType, so the
    # backend raises and the check declines rather than rejecting the solution.
    assert unsupported(solutionFor("complexFloat", compute="complexFloat"), [16, 16, 4, 1]) is None

    # A wrong argument must still surface instead of being swallowed.
    with pytest.raises(Exception):
        matrixInstructionMnemonic(
            GFX1250, 32, [16, 16, 32, 1], DataType("half"), DataType("half"), None
        )
