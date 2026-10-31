# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# f16 accumulation on gfx1250: HighPrecisionAccumulate=False with fp8, bf8 or fp4
# inputs and ComputeDataType=H runs the f16-accumulate 8-bit-float WMMA, whose 16x16
# output tile holds two elements per VGPR; D is f16, fp8 or bf16. fp4 operands are converted to fp8
# after the LDS read. Covers the problem type, the solution rules on both
# gfx1250 steppings, the WMMA spelling, and the emitted assembly.
################################################################################

import collections
import copy
import re
import shutil
import struct
from pathlib import Path

import pytest

from tensilelite import run
from tensilelite.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
from tensilelite.Common.DataType import DataType
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Common.Types import IsaVersion
from tensilelite.KernelHelperNaming import conversionKernelNames
from tensilelite.SolutionStructs.Problem import ProblemType, isF16Accumulation
from tensilelite.SolutionStructs.Solution import Solution
from tensilelite.SolutionStructs.segment_interleave import mn_major_b_requested as segMnMajorBRequested

pytestmark = pytest.mark.unit

ISA_GFX1250 = IsaVersion(12, 5, 0)
_GFX12_CONFIGS = (Path(__file__).parent / ".." / "common" / "gemm" / "gfx12").resolve()
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


def _problem_config(**overrides):
    config = {
        "OperationType": "GEMM",
        "DataType": "F8",
        "DestDataType": "h",
        "ComputeDataType": "h",
        "HighPrecisionAccumulate": False,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    config.update(overrides)
    return config


# =========================================================================== #
# Problem type. No toolchain needed.
# =========================================================================== #
@pytest.mark.parametrize("dest, name", [("h", "_F8HH_"), ("F8", "_F8F8H_"), ("B", "_F8BH_")])
def test_fp8_inputs_with_half_compute_accumulate_in_f16(dest, name):
    problem = ProblemType(_problem_config(DestDataType=dest), False)
    assert isF16Accumulation(problem)
    assert name in str(problem)


@pytest.mark.parametrize("operand", ["A", "B"])
def test_fp4_input_keeps_fp8_as_the_wmma_type(operand):
    problem = ProblemType(_problem_config(**{"DataType%s" % operand: "F4"}), False)
    assert isF16Accumulation(problem)
    assert problem["MacDataType%s" % operand].is8bitFloat()
    assert problem["DataType%s" % operand].isFloat4()


def test_hpa_with_half_compute_falls_back_to_f32_accumulation():
    problem = ProblemType(_problem_config(HighPrecisionAccumulate=True), False)
    assert problem["ComputeDataType"].isSingle()
    assert not isF16Accumulation(problem)


def test_half_inputs_are_not_f16_accumulation():
    problem = ProblemType(_problem_config(DataType="h"), False)
    assert not isF16Accumulation(problem)


@pytest.mark.parametrize("types, macA, macB", [
    ({"DataType": "B8"}, "B8", "B8"),
    ({"DataType": "F8B8"}, "F8", "B8"),
    ({"DataType": "B8F8"}, "B8", "F8"),
    ({"DataType": "F8B8", "DataTypeA": "F4"}, "F8", "B8"),
    ({"DataType": "B8F8", "DataTypeB": "F4"}, "B8", "F8"),
], ids=["b8", "f8b8", "b8f8", "f4b8", "b8f4"])
@pytest.mark.parametrize("dest", ["h", "F8", "B"])
def test_bf8_inputs_accumulate_in_f16(types, macA, macB, dest):
    problem = ProblemType(_problem_config(DestDataType=dest, **types), False)
    assert isF16Accumulation(problem)
    assert problem["MacDataTypeA"] == DataType(macA) and problem["MacDataTypeB"] == DataType(macB)


_STORAGE = "f16 accumulation stores each input as its MAC type, or as F4 with an F8 MAC type"
_TABLE = "is not supported yet"


@pytest.mark.parametrize("types, reason", [
    ({"DataType": "F8N"}, _TABLE),
    ({"DataType": "B8N"}, _TABLE),
    ({"DataTypeA": "F8", "DataTypeB": "B8"}, _STORAGE),
    ({"DataTypeA": "B8", "DataTypeB": "F8"}, _STORAGE),
    ({"DataTypeA": "F4", "DataTypeB": "B8"}, _STORAGE),
    ({"DataTypeA": "F8", "DataTypeB": "F8N"}, _STORAGE),
    ({"DataTypeA": "F8N", "DataTypeB": "F8"}, _STORAGE),
    ({"DataTypeA": "F4", "DataTypeB": "F8N"}, _STORAGE),
    ({"DataTypeA": "B8", "DataTypeB": "F4"}, _STORAGE),
    ({"DataTypeA": "F8N", "DataTypeB": "F4"}, _STORAGE),
    ({"DataType": "F8B8", "DataTypeB": "F4"}, _STORAGE),
], ids=["f8n", "b8n", "f8b8", "b8f8", "f4b8", "f8f8n", "f8nf8", "f4f8n", "b8f4", "f8nf4", "f8b8_f4b"])
def test_storage_types_other_than_the_mac_type_and_fnuz_inputs_are_rejected(types, reason):
    with pytest.raises(Exception, match=re.escape(reason)):
        ProblemType(_problem_config(**types), False)


@pytest.mark.parametrize("types, storage", [
    ({}, "F4F4"), ({"MacDataTypeA": "F8", "MacDataTypeB": "F8"}, "F4F4"),
    ({"DataTypeA": "F8"}, "F8F4"), ({"DataTypeB": "F8"}, "F4F8")], ids=["f4", "f4_f8mac", "f8f4", "f4f8"])
def test_data_type_f4_takes_the_data_type_f8_spelling_for_f16_accumulation(types, storage):
    problem = ProblemType(_problem_config(DataType="F4", **types), False)
    spelled = ProblemType(_problem_config(DataTypeA=storage[:2], DataTypeB=storage[2:]), False)
    for k in ("DataType", "DataTypeA", "DataTypeB", "MacDataTypeA", "MacDataTypeB", "DestDataType", "ComputeDataType"):
        assert problem[k].toChar() == spelled[k].toChar(), k
    assert isF16Accumulation(problem)


def test_bf8_inputs_with_f32_accumulation_are_not_rejected():
    problem = ProblemType(_problem_config(DataType="B8", HighPrecisionAccumulate=True), False)
    assert not isF16Accumulation(problem)


# =========================================================================== #
# Solution rules on real gfx1250 capabilities.
# =========================================================================== #
@pytest.fixture(scope="module")
def gfx1250_cxx():
    from tensilelite.Toolchain.Validators import validateToolchain

    try:
        return validateToolchain("amdclang++")
    except (ValueError, FileNotFoundError) as e:
        pytest.skip(f"amdclang++ is unavailable: {e}")


@pytest.fixture(scope="module")
def gfx1250_iim(gfx1250_cxx):
    iim = makeIsaInfoMap([ISA_GFX1250], gfx1250_cxx)
    if not iim[ISA_GFX1250].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    applyArchCapOverrides(iim, ["gfx1250"])
    return iim


@pytest.fixture(scope="module")
def assembler(gfx1250_cxx):
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(gfx1250_cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from tensilelite.Common.GlobalParameters import assignGlobalParameters, globalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _derive(iim, assembler, capsys, problem=None, mi=(16, 16, 64, 1, 1, 1, 1, 1, 1), **overrides):
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

    problemType = _problem_config(**(problem or {}))
    params = {
        "ProblemType": problemType,
        "ISA": ISA_GFX1250,
        "MatrixInstruction": list(mi),
        "WavefrontSize": 32,
        "DepthU": 128,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4,
        "TDMInst": 3,
        "TransposeLDS": 1,
        "LocalReadVectorWidth": 16,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "SourceSwap": False,
        "GlobalSplitU": 1,
        "AssertFree0ElementMultiple": 2,
        "AssertSummationElementMultiple": 64,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(list(mi), ISA_GFX1250, params["WavefrontSize"], problemType,
                                                  params.get("WorkGroup"), iim))
    sol = Solution(params, False, True, False, assembler, iim)
    return sol, capsys.readouterr().out


def test_fp8_solution_holds_two_outputs_per_vgpr(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, out
    assert sol["MIRegPerOut"] == 0.5
    assert sol["InternalSupportParams"]["SupportUserGSU"] is True
    # The split-K reduction kernels read an f16 workspace.
    names = conversionKernelNames(sol)
    assert names and all(name.startswith("Cijk_HH_") for name in names)


@pytest.mark.parametrize("overrides", [{}, {"StaggerU": 16, "StaggerUMapping": 3, "StaggerUStride": 512}],
                         ids=["default", "custom"])
def test_stagger_u_is_kept(_gp_gfx1250, gfx1250_iim, assembler, capsys, overrides):
    reference, _ = _derive(gfx1250_iim, assembler, capsys, problem={"HighPrecisionAccumulate": True}, **overrides)
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is True, out
    for key in ("StaggerU", "StaggerUMapping", "StaggerUStride", "_staggerStrideShift"):
        assert sol.get(key) == reference.get(key), key
    assert sol["StaggerU"] > 0
    assert sol["InternalSupportParams"]["SupportCustomStaggerU"] is True


@pytest.mark.parametrize("problem, overrides", [
    ({}, {}),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True}),
], ids=["f8f8", "f8f4"])
@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("stepping, accepted", [("gfx1250", True), ("gfx1250:xnack-", True), ("gfx1250-strict", False),
                                                ("gfx1250-strict:xnack-", False)])
def test_f16_accumulation_is_refused_on_gfx1250_strict(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                       stepping, accepted, k, problem, overrides):
    iim = copy.deepcopy(gfx1250_iim)
    applyArchCapOverrides(iim, [stepping])
    sol, out = _derive(iim, assembler, capsys, problem=problem, mi=(16, 16, k, 1, 1, 2, 2, 1, 1), **overrides)
    assert sol.get("Valid") is accepted, out
    assert ("needs the HasF16Accumulation capability" in out) is not accepted, out


@pytest.mark.parametrize("sourceSwap", [False, True])
@pytest.mark.parametrize("types", [{"DataTypeA": "F4"}, {"DataTypeB": "F4"}, {"DataTypeA": "F4", "DataTypeB": "F4"}],
                         ids=["f4f8", "f8f4", "f4f4"])
@pytest.mark.parametrize("dest", ["h", "F8", "B"])
def test_fp4_split_k_reduction_kernels_read_an_f16_workspace(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest, types,
                                                             sourceSwap):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=dict(types, DestDataType=dest), SourceSwap=sourceSwap,
                       ConvertAfterDS=True, GlobalSplitU=2, GlobalSplitUAlgorithm="MultipleBuffer")
    assert sol.get("Valid") is True, out
    names = conversionKernelNames(sol)
    assert names and all(name.startswith("Cijk_H") for name in names), names


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("gsu, coalesced", [(2, False), (3, False), (4, True)])
def test_split_k_multiple_buffer_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, gsu, coalesced, dest):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, GlobalSplitU=gsu,
                       GlobalSplitUAlgorithm="MultipleBuffer", GlobalSplitUCoalesced=coalesced)
    assert sol.get("Valid") is True, out
    assert sol["_GlobalAccumulation"] == "MultipleBuffer"
    assert sol["_WorkspaceSizePerElemC"] == 2


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("srvw", [4, 8])
@pytest.mark.parametrize("gsu", [1, 3])
def test_store_remap_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest, srvw, gsu):
    # The existing StoreRemap rules apply unchanged: GSU > 1 only through MultipleBuffer, no SourceSwap.
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, StoreRemapVectorWidth=srvw,
                       GlobalSplitU=gsu, GlobalSplitUAlgorithm="MultipleBuffer")
    assert sol.get("Valid") is True, out
    assert sol["StoreRemapVectorWidth"] == srvw


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
def test_output_amax_d_is_valid_and_kept_in_f32(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest):
    # The existing amax rules apply; the amax is taken from the f32 epilogue values and kept in f32, as for HPA.
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest, "OutputAmaxD": True})
    assert sol.get("Valid") is True, out
    assert sol["ProblemType"]["DataTypeAmaxD"].isSingle()


_SCALE_BIAS_ACT = {"UseBias": 1, "Activation": True, "ActivationType": "hipblaslt_all", "UseScaleAB": "Vector",
                   "UseScaleCD": True, "UseScaleAlphaVec": 1}


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("gsu", [1, 3])
def test_scale_bias_and_activation_are_valid_in_f32(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest, gsu):
    # The epilogue computes in f32, so the activation runs in f32 and an f32 bias is among the default bias types.
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest, **_SCALE_BIAS_ACT},
                       GlobalSplitU=gsu, GlobalSplitUAlgorithm="MultipleBuffer")
    assert sol.get("Valid") is True, out
    assert sol["ProblemType"]["ActivationComputeDataType"].isSingle()
    assert any(t.isSingle() for t in sol["ProblemType"]["BiasDataTypeList"])


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
def test_split_k_reduction_applies_scales_bias_and_activation_in_f32(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest):
    from tensilelite.KernelHelperNaming import initConversionKernelObjects

    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest, **_SCALE_BIAS_ACT},
                       GlobalSplitU=2, GlobalSplitUAlgorithm="MultipleBuffer")
    assert sol.get("Valid") is True, out
    for k in initConversionKernelObjects(sol, gfx1250_iim):
        header = k.getHeaderFileString()
        _, source = k.getSourceFileString()
        # The scales are f32 in memory; the reduction kernel's f32 epilogue applies everything to the widened sum.
        for name in ("ScaleA", "ScaleB", "ScaleC", "ScaleD", "ScaleAlphaVec"):
            assert "  float * %s;" % name in header, name
        assert "    accum[0] *= (float)arg.ScaleA[id0+0];" in source
        assert "    accum[0] *= (float)arg.ScaleAlphaVec[id0+0];" in source
        assert "    accum[0] += (float)arg.Bias[idxBias+0];" in source
        assert "  accum[0] = activation((float)accum[0]" in source
        assert "  accum[0] *= scaleD_data;" in source
        assert "  if(arg.alpha != 0.0f)" in source or "  if((arg.deviceScalarAlpha ? arg.ScaleAlphaVec[0] : arg.alpha) != 0.0f)" in source


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("gsu, method", [(2, 0), (3, 0), (4, 1), (8, 1), (17, 0)])
def test_split_k_mbsk_is_valid_without_a_reduction_kernel(_gp_gfx1250, gfx1250_iim, assembler, capsys, gsu, method,
                                                           dest):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, GlobalSplitU=gsu,
                       GlobalSplitUAlgorithm="MultipleBufferSingleKernel", MbskPrefetchMethod=method)
    assert sol.get("Valid") is True, out
    assert sol["_GlobalAccumulation"] == "MultipleBufferSingleKernel"
    assert sol["_WorkspaceSizePerElemC"] == 2 and sol["MbskPrefetchMethod"] == method
    assert not conversionKernelNames(sol)


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
def test_mbsk_prefetch_batch_is_a_whole_number(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest):
    # A packed f16 accumulator is half a VGPR, and the store code sizes VGPR blocks with the prefetch batch.
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, mi=(16, 16, 128, 1, 1, 2, 2, 1, 1),
                       AssertFree0ElementMultiple=1, VectorWidthA=2, GlobalSplitU=17,
                       GlobalSplitUAlgorithm="MultipleBufferSingleKernel", MbskPrefetchMethod=1)
    assert sol.get("Valid") is True, out
    assert type(sol["NumMbskPrefetchElements"]) is int and type(sol["NumElementsPerBatchStore"]) is int, \
        (sol["StoreVectorWidth"], sol["NumMbskPrefetchElements"], sol["NumElementsPerBatchStore"])
    assert sol["NumMbskPrefetchElements"] * sol["StoreVectorWidth"] // 2 <= 64


@pytest.mark.parametrize("algorithm", ["MultipleBuffer", "MultipleBufferSingleKernel"])
def test_adaptive_gsua_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, algorithm):
    # The host picks MultipleBuffer or MultipleBufferSingleKernel per launch; both leave the same f16 sums.
    sol, out = _derive(gfx1250_iim, assembler, capsys, GlobalSplitU=4, GlobalSplitUAlgorithm=algorithm,
                       AdaptiveGemmGSUA=1)
    assert sol.get("Valid") is True, out
    assert sol["AdaptiveGemmGSUA"] == 1


@pytest.mark.parametrize("algorithm, adaptive", [("MultipleBuffer", 0), ("MultipleBufferSingleKernel", 0),
                                                 ("MultipleBuffer", 1)], ids=["mb", "mbsk", "mb_gsua"])
def test_auto_gsu_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, algorithm, adaptive):
    sol, out = _derive(gfx1250_iim, assembler, capsys, GlobalSplitU=-1, GlobalSplitUAlgorithm=algorithm,
                       AdaptiveGemmGSUA=adaptive)
    assert sol.get("Valid") is True, out
    assert sol["InternalSupportParams"]["SupportUserGSU"] is True


_PERSISTENT = ("supports persistent kernels without DebugStreamK, PrefetchAcrossPersistent, ReuseAcrossPersistent, "
               "work stealing or clusters")
_ATOMICS = "splits K only with GlobalSplitUAlgorithm MultipleBuffer or MultipleBufferSingleKernel"
_ATOMIC_DEST = "AtomicDest currently only supports a BF16 D"


@pytest.mark.parametrize("strategy, assignment, tree", [
    ("DataParallel", "StaticGrid", 0), ("StreamK", "StaticGrid", 0), ("StreamK", "StaticGrid", 1),
    ("StreamK", "DynamicWorkQueue", 0), ("StreamK", "DynamicWorkQueue", 1), ("StreamK", "Hybrid", 1)])
def test_persistent_and_streamk_are_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, strategy, assignment, tree):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TileProcessingStrategy=strategy, WorkAssignment=assignment,
                       StreamKFixupTreeReduction=tree)
    assert sol.get("Valid") is True, out
    # The tree fixup slots hold f32 sums; Hybrid takes the tree fixup in its static sub-mode, DynamicWorkQueue never.
    assert sol["_WorkspaceSizePerElemC"] == (0 if strategy == "DataParallel" else
                                             4 if tree and assignment != "DynamicWorkQueue" else 2)


@pytest.mark.parametrize("assignment", ["StaticGrid", "DynamicWorkQueue", "Hybrid"])
def test_space_filling_streamk_needs_a_static_grid(_gp_gfx1250, gfx1250_iim, assembler, capsys, assignment):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TileProcessingStrategy="StreamK", WorkAssignment=assignment,
                       SpaceFillingAlgo=[2], SFCWGM=[[2, 2]])
    assert sol.get("Valid") is (assignment == "StaticGrid"), out
    assert ("SpaceFillingAlgo requires WorkAssignment=StaticGrid" in out) is (assignment != "StaticGrid"), out


@pytest.mark.parametrize("assignment", ["StaticGrid", "DynamicWorkQueue", "Hybrid"])
def test_space_filling_streamk_needs_a_static_grid_with_f32_accumulation(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                                         assignment):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"ComputeDataType": "s", "HighPrecisionAccumulate": True},
                       TileProcessingStrategy="StreamK", WorkAssignment=assignment, SpaceFillingAlgo=[2], SFCWGM=[[2, 2]])
    assert ("SpaceFillingAlgo requires WorkAssignment=StaticGrid" in out) is (assignment != "StaticGrid"), out


@pytest.mark.parametrize("overrides, reason", [
    ({"GlobalSplitU": 2, "GlobalSplitUAlgorithm": "SingleBuffer"}, _ATOMICS),
    # A user GSU can split a GSU 1 or auto-GSU solution at run time.
    ({"GlobalSplitU": 1, "GlobalSplitUAlgorithm": "SingleBuffer"}, _ATOMICS),
    ({"GlobalSplitU": -1, "GlobalSplitUAlgorithm": "SingleBuffer"}, _ATOMICS),
    # The existing AtomicDest rules refuse an f16 D before the f16 accumulation rules.
    ({"GlobalSplitU": 1, "GlobalSplitUAlgorithm": "AtomicDest"}, _ATOMIC_DEST),
    ({"GlobalSplitU": -1, "GlobalSplitUAlgorithm": "AtomicDest"}, _ATOMIC_DEST),
    # The existing SGEMM-only rule refuses StreamKAtomic before the f16 accumulation rules.
    ({"TileProcessingStrategy": "StreamK", "StreamKAtomic": 1}, "Atomic Stream-K currently only tested for SGEMM"),
], ids=["single_buffer", "single_buffer_gsu1", "single_buffer_gsu_auto", "atomic_dest_gsu1", "atomic_dest_gsu_auto",
        "stream_k_atomic"])
def test_split_k_paths_with_a_run_dependent_order_are_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                               overrides, reason):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is False
    assert reason in out


def test_wave64_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, WavefrontSize=64)
    assert sol.get("Valid") is False
    assert "f16 accumulation supports only wave32" in out


_LSU = "does not support LocalSplitU"
_STORE = "does not support CompactLoopStore or UseSubtileImpl"
_EPILOGUE = "does not support UseE, Gradient, UseGateResidual, FusedGemmA2A, Sparse, MXBlock or StochasticRounding"
_F4A = "reads F4 A K-contiguous from unroll-major LDS only"
_F4B = "reads F4 B K-contiguous from unroll-major LDS only"
_F4A_MN = "reads MN-contiguous F4 A with ds_load_tr4_b64 (LDSTrInst) from MN-major LDS only"
_F4B_MN = "reads MN-contiguous F4 B with ds_load_tr4_b64 (LDSTrInst) from MN-major LDS only"
_CAD = "converts F4 to F8 after the LDS read and needs ConvertAfterDS"


# Each case changes one condition of a valid configuration.
@pytest.mark.parametrize("problem, overrides, reason", [
    ({}, {"mi": (16, 16, 32, 1, 1, 1, 1, 1, 1)}, "supports only the 16x16x64 and 16x16x128 WMMA"),
    ({}, {"WorkGroup": [16, 16, 2]}, _LSU),
    ({}, {"TileProcessingStrategy": "StreamK", "DebugStreamK": 1}, _PERSISTENT),
    ({}, {"TileProcessingStrategy": "StreamK", "PrefetchAcrossPersistent": 1}, _PERSISTENT),
    ({}, {"TileProcessingStrategy": "StreamK", "ReuseAcrossPersistent": 1}, _PERSISTENT),
    ({}, {"TileProcessingStrategy": "StreamK", "ClusterDim": [2, 1]}, _PERSISTENT),
    ({"GroupedGemm": True}, {}, "does not support grouped GEMM"),
    ({}, {"CompactLoopStore": True}, _STORE),
    ({}, {"UseSubtileImpl": True}, _STORE),
    ({"Gradient": True}, {}, _EPILOGUE),
    ({"UseGateResidual": True}, {}, _EPILOGUE),
    ({"Sparse": 1}, {}, _EPILOGUE),
    ({"MXBlockA": 32}, {}, _EPILOGUE),
    ({"MXBlockB": 32}, {}, _EPILOGUE),
    ({}, {"ConvertAfterDS": True}, "uses ConvertAfterDS only for F4 inputs"),
    ({"DataTypeA": "F4"}, {}, _CAD),
    ({"DataTypeB": "F4"}, {}, _CAD),
    ({"DataTypeA": "F4", "TransposeA": False}, {"ConvertAfterDS": True}, _F4A_MN),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True, "TransposeLDS": 0}, _F4A),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True, "DirectToVgprA": True}, _F4A),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True, "DirectToLds": 1}, _F4A),
    ({"DataTypeB": "F4", "TransposeB": True}, {"ConvertAfterDS": True}, _F4B_MN),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True, "TransposeLDS": 0}, _F4B),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True, "DirectToVgprB": True}, _F4B),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True, "DirectToLds": 1}, _F4B),
], ids=["mi_16x16x32", "lsu2", "streamk", "prefetch_across_persistent", "reuse_across_persistent", "cluster_dim",
        "grouped_gemm", "compact_loop_store", "subtile", "gradient", "gate_residual", "sparse",
        "mx_block_a", "mx_block_b", "convert_after_ds_without_f4", "f4_a_without_convert_after_ds",
        "f4_b_without_convert_after_ds", "f4_a_tlu", "f4_a_not_unroll_major", "f4_a_direct_to_vgpr",
        "f4_a_direct_to_lds", "f4_b_tlu", "f4_b_not_unroll_major", "f4_b_direct_to_vgpr", "f4_b_direct_to_lds"])
def test_each_rule_rejects_on_its_own(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem, overrides, reason):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem, **overrides)
    assert sol.get("Valid") is False
    assert reason in out


_MNF4 = {"ConvertAfterDS": True, "LDSTrInst": True, "LocalReadVectorWidth": -1, "AssertSummationElementMultiple": 8,
         "TransposeLDS": -1}
# An MN-contiguous fp4 global read spans 32 elements, so the macro tile must be at least that wide.
_MNF4_MI = (16, 16, 64, 1, 1, 2, 2, 1, 1)


@pytest.mark.parametrize("problem", [
    {"DataTypeA": "F4", "TransposeA": False},
    {"DataTypeB": "F4", "TransposeB": True},
    {"DataTypeA": "F4", "DataTypeB": "F4", "TransposeA": False, "TransposeB": True},
], ids=["f4_a", "f4_b", "f4_ab"])
def test_mn_contiguous_f4_is_read_with_the_fp4_transpose_load(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem, mi=_MNF4_MI, **_MNF4)
    assert sol.get("Valid") is True, out
    for tc, free in (("A", "AssertFree0ElementMultiple"), ("B", "AssertFree1ElementMultiple")):
        if problem.get("DataType%s" % tc) == "F4":
            assert sol["enableLDSTr%s" % tc]
            assert sol["LocalReadVectorWidth%s" % tc] == 32
            assert sol[free] == 2


@pytest.mark.parametrize("problem, overrides, reason", [
    ({"DataTypeA": "F4", "TransposeA": False}, {"LDSTrInst": False}, _F4A_MN),
    ({"DataTypeA": "F4", "TransposeA": False}, {"DirectToVgprA": True}, _F4A_MN),
    ({"DataTypeA": "F4", "TransposeA": False}, {"DirectToLds": 1}, _F4A_MN),
    ({"DataTypeA": "F4", "TransposeA": False}, {"LocalReadVectorWidth": 16}, "MN-contiguous F4 A with a LocalReadVectorWidth"),
    ({"DataTypeA": "F4", "TransposeA": False}, {"HalfPLR": 1}, "HalfPLR does not support ConvertAfterDS"),
    ({"DataTypeB": "F4", "TransposeB": True}, {"LDSTrInst": False}, _F4B_MN),
    ({"DataTypeB": "F4", "TransposeB": True}, {"DirectToVgprB": True}, _F4B_MN),
    ({"DataTypeB": "F4", "TransposeB": True}, {"DirectToLds": 1}, _F4B_MN),
    ({"DataTypeB": "F4", "TransposeB": True}, {"LocalReadVectorWidth": 16}, "MN-contiguous F4 B with a LocalReadVectorWidth"),
    ({"DataTypeB": "F4", "TransposeB": True}, {"HalfPLR": 2}, "HalfPLR does not support ConvertAfterDS"),
], ids=["f4_a_no_ldstr", "f4_a_direct_to_vgpr", "f4_a_direct_to_lds", "f4_a_lrvw16", "f4_a_half_plr",
        "f4_b_no_ldstr", "f4_b_direct_to_vgpr", "f4_b_direct_to_lds", "f4_b_lrvw16", "f4_b_half_plr"])
def test_each_mn_contiguous_f4_rule_rejects_on_its_own(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem,
                                                       overrides, reason):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem, mi=_MNF4_MI, **{**_MNF4, **overrides})
    assert sol.get("Valid") is False
    assert reason in out


# The fp4 tail loop needs K % 8 == 0 for a K- or MN-contiguous fp4 operand; fp8 operands keep any multiple.
@pytest.mark.parametrize("asem", [1, 64])
@pytest.mark.parametrize("problem, overrides, multiple", [
    ({}, {}, 1),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True}, 8),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True}, 8),
    ({"DataTypeA": "F4", "DataTypeB": "F4"}, {"ConvertAfterDS": True}, 8),
    ({"DataTypeA": "F4", "TransposeA": False}, {**_MNF4, "mi": _MNF4_MI}, 8),
    ({"DataTypeB": "F4", "TransposeB": True}, {**_MNF4, "mi": _MNF4_MI}, 8),
], ids=["f8f8", "f4_a", "f4_b", "f4_ab", "f4_a_mn", "f4_b_mn"])
def test_summation_multiple_is_at_least_8_for_every_fp4_operand(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem,
                                                                overrides, multiple, asem):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem,
                       **{**overrides, "AssertSummationElementMultiple": asem})
    assert sol.get("Valid") is True, out
    assert sol["AssertSummationElementMultiple"] == max(asem, multiple)


# StoreVectorWidth -1 resolves to VectorWidthA under SourceSwap; 3 or 6 only come from a VectorWidthA of 3 or 6.
@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("sourceSwap, svw", [(True, -1), (True, 3), (False, 3), (False, 6)],
                         ids=["ss_auto", "ss_svw3", "svw3", "svw6"])
def test_vector_width_that_is_not_a_power_of_two_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest,
                                                             sourceSwap, svw):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, mi=(16, 16, 64, 1, 1, 3, 1, 1, 1),
                       SourceSwap=sourceSwap, VectorWidthA=3, StoreVectorWidth=svw)
    assert sol.get("Valid") is False
    assert "f16 accumulation needs a power-of-two VectorWidthA" in out


# MIWaveTile 1x3 with VectorWidthB 3, with and without SourceSwap; with f32 accumulation the same solution is valid.
@pytest.mark.parametrize("sourceSwap", [True, False], ids=["ss", "noss"])
def test_vector_width_b_that_is_not_a_power_of_two_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys, sourceSwap):
    overrides = dict(mi=(16, 16, 64, 1, 1, 1, 3, 1, 1), SourceSwap=sourceSwap, VectorWidthB=3)
    reference, refOut = _derive(gfx1250_iim, assembler, capsys, problem={"HighPrecisionAccumulate": True}, **overrides)
    assert reference.get("Valid") is True, refOut
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is False
    assert "f16 accumulation needs a power-of-two VectorWidthB" in out


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("problem, overrides", [
    ({}, {"SourceSwap": True}),
    ({"TransposeA": False, "TransposeB": False}, {"SourceSwap": True, "LDSTrInst": True, "DepthU": 64,
                                                  "AssertSummationElementMultiple": 32}),
    ({"DataTypeB": "F4"}, {"SourceSwap": True, "ConvertAfterDS": True}),
    ({"DataTypeA": "F4"}, {"SourceSwap": True, "ConvertAfterDS": True}),
], ids=["tn", "nn", "f8f4", "f4f8"])
def test_source_swap_with_vector_width_one_stores_single_elements(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                                   problem, overrides, dest):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={**problem, "DestDataType": dest},
                       mi=(16, 16, 64, 1, 1, 2, 2, 1, 1), VectorWidthA=1, **overrides)
    assert sol.get("Valid") is True, out
    assert sol["StoreVectorWidth"] == 1


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("sourceSwap", [False, True])
def test_odd_free0_multiple_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, sourceSwap, dest):
    # The HasEccHalf AF0EM%2 rule covers DataType half only; these inputs are fp8.
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, mi=(16, 16, 64, 1, 1, 2, 2, 1, 1),
                       AssertFree0ElementMultiple=1, SourceSwap=sourceSwap, VectorWidthA=2)
    assert sol.get("Valid") is True, out


@pytest.mark.parametrize("af0em, sourceSwap, widths", [
    (2, False, [8, 4, 2]),
    (1, False, [8, 4, 2, 1]),
    (2, True, [1]),
    (1, True, [1]),
])
def test_edge_store_narrows_to_one_element_only_when_needed(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                            af0em, sourceSwap, widths):
    from tensilelite.Components.NotLocalFullTileElements import NotLocalFullTileElementsMFMA

    class _Writer:
        def maxGwvw(self, kernel):
            return 8

    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=(16, 16, 64, 1, 1, 2, 2, 1, 1),
                       AssertFree0ElementMultiple=af0em, SourceSwap=sourceSwap,
                       VectorWidthA=1 if sourceSwap else 2)
    assert sol.get("Valid") is True, out
    vectorWidths, _ = NotLocalFullTileElementsMFMA().getElements(_Writer(), sol)
    assert vectorWidths == widths


def test_width_one_templates_move_each_element_to_a_low_half(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                             _rocisa_gfx1250):
    from tensilelite.KernelWriterModules import accToArchMapper, packedAccSingles

    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=(16, 16, 64, 1, 1, 2, 2, 1, 1),
                       SourceSwap=True, VectorWidthA=1)
    assert sol.get("Valid") is True, out
    _, arch2acc = accToArchMapper(sol)
    moves = [str(i).split("//")[0].split() for i in packedAccSingles(sol).items()]
    assert len(moves) == len(arch2acc)
    for d, move in enumerate(moves):
        s = arch2acc[d]
        src = "v[vgprValuC+%u]" % (s // 2)
        # Element s is in half s%2 of its MI output VGPR; the temp gets it in its low half.
        if s % 2:
            assert move[0] == "v_lshrrev_b32" and move[-2].rstrip(",") in ("16", "0x10") and move[-1] == src, move
        else:
            assert move[0] == "v_mov_b32" and move[-1] == src, move
    assert sorted(arch2acc.values()) == list(range(len(arch2acc)))


@pytest.mark.parametrize("sourceSwap", [False, True])
def test_epilogue_unpacks_each_accumulator_to_f32(_gp_gfx1250, gfx1250_iim, assembler, capsys, _rocisa_gfx1250,
                                                  sourceSwap):
    from tensilelite.KernelWriterModules import accToArchMapper, mulMIoutAlphaToArch, packedAccToF32

    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=(16, 16, 64, 1, 1, 2, 2, 1, 1), SourceSwap=sourceSwap,
                       VectorWidthA=1 if sourceSwap else 2)
    assert sol.get("Valid") is True, out
    _, arch2acc = accToArchMapper(sol)
    reads = [_instructions(i) for i in packedAccToF32(sol, mulAlpha=False).items()]
    alphas = [_instructions(i) for i in mulMIoutAlphaToArch(sol, 0).items()]
    assert len(reads) == len(alphas) == len(arch2acc)
    for d, (read, alpha) in enumerate(zip(reads, alphas)):
        s = arch2acc[d]
        # Element s is in half s%2 of its MI output VGPR; its temp gets it widened to f32, then scaled in f32.
        assert len(read) == 1 and read[0].split()[0] == "v_cvt_f32_f16", read
        src = read[0].split(",")[-1].strip()
        assert src.startswith("v[vgprValuC+%u]" % (s // 2)) and src.endswith(".h") == bool(s % 2), read
        assert alpha[0] == read[0] and len(alpha) == 2, alpha
        assert alpha[1].split()[0] == "v_mul_f32" and "s[sgprAlpha]" in alpha[1], alpha


@pytest.mark.parametrize("sourceSwap", [False, True])
def test_partial_templates_move_pairs(_gp_gfx1250, gfx1250_iim, assembler, capsys, _rocisa_gfx1250, sourceSwap):
    from tensilelite.KernelWriterModules import accToArchMapper, mapAcctoArchRegs, moveMIoutToArch

    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=(16, 16, 64, 1, 1, 2, 2, 1, 1), SourceSwap=sourceSwap,
                       VectorWidthA=1 if sourceSwap else 2)
    assert sol.get("Valid") is True, out
    _, arch2acc = accToArchMapper(sol)
    pairs = [(arch2acc[d], arch2acc[d + 1]) for d in range(0, len(arch2acc), 2)]
    reads = [str(i).split("//")[0] for i in mapAcctoArchRegs(sol).items()]
    assert reads == [str(i).split("//")[0] for i in moveMIoutToArch(sol, 0).items()]
    assert len(reads) == len(pairs)
    contiguous = [s1 == s0 + 1 and s0 % 2 == 0 for s0, s1 in pairs]
    for (s0, s1), whole, read in zip(pairs, contiguous, reads):
        if whole:
            assert read.split()[0] == "v_mov_b32" and "v[vgprValuC+%u]" % (s0 // 2) in read
        else:
            assert read.split()[0] == "v_pack_b32_f16"
            assert "v[vgprValuC+%u]" % (s0 // 2) in read and "v[vgprValuC+%u]" % (s1 // 2) in read


def _run_packed_templates(lines, regs):
    # Interprets the moves, shifts and packs of the packed-accumulator templates on ValuC offsets.
    def reg(token):
        return int(re.fullmatch(r"v\[vgprValuC(?:\+(\d+))?\]", token).group(1) or 0)

    for line in lines:
        op, rest = line.split(None, 1)
        opSel = re.search(r"op_sel:\[(\d),(\d)", rest)
        ops = [t.strip() for t in re.sub(r"\s+op_sel.*", "", rest).split(",")]
        half = lambda x, h: (x >> (16 * h)) & 0xffff
        if op == "v_mov_b32":
            regs[reg(ops[0])] = regs[reg(ops[1])]
        elif op == "v_lshrrev_b32":
            regs[reg(ops[0])] = regs[reg(ops[2])] >> int(ops[1], 0)
        elif op == "v_pack_b32_f16":
            lo, hi = (int(opSel.group(1)), int(opSel.group(2))) if opSel else (0, 0)
            regs[reg(ops[0])] = half(regs[reg(ops[1])], lo) | half(regs[reg(ops[2])], hi) << 16
        else:
            raise AssertionError(line)


@pytest.mark.parametrize("sourceSwap, vw", [(False, 2), (True, 1), (True, 2)])
def test_packed_write_back_returns_every_half_to_its_accumulator(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                                 _rocisa_gfx1250, sourceSwap, vw):
    from rocisa.container import replaceHolder
    from tensilelite.KernelWriterModules import accToArchMapper, mapAcctoArchRegs, packedAccSingles, packedArchSinglesToAcc

    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=(16, 16, 64, 1, 1, 2, 2, 1, 1), SourceSwap=sourceSwap,
                       VectorWidthA=vw)
    assert sol.get("Valid") is True, out
    _, arch2acc = accToArchMapper(sol)
    element = lambda regs, s: (regs[s // 2] >> (16 * (s % 2))) & 0xffff
    initial = {i: (0x100 + 2 * i + 1) << 16 | (0x100 + 2 * i) for i in range(len(arch2acc) // 2)}
    # Temps stay below VGPR 256, which needs no s_set_vgpr_msb.
    for read, write, perTemp, base in ((mapAcctoArchRegs(sol), mapAcctoArchRegs(sol, write=True), 2, 40),
                                       (packedAccSingles(sol), packedArchSinglesToAcc(sol), 1, 80)):
        regs = dict(initial)
        for t, item in enumerate(read.items()):
            _run_packed_templates(_instructions(replaceHolder(item, base + t)), regs)
        for t in range(len(arch2acc) // perTemp):
            for h in range(perTemp):
                assert (regs[base + t] >> (16 * h)) & 0xffff == element(initial, arch2acc[perTemp * t + h])
        regs.update({i: 0xdeadbeef for i in initial})
        for t, item in enumerate(write.items()):
            _run_packed_templates(_instructions(replaceHolder(item, base + t)), regs)
        assert {i: regs[i] for i in initial} == initial


def _instructions(module):
    return [line.split("//")[0].strip() for line in str(module).splitlines() if line.split("//")[0].strip()]


@pytest.mark.parametrize("dest, destDevice", [("h", "tensile_half"), ("F8", "tensile_float8"), ("B", "tensile_bfloat16")])
def test_split_k_reduction_adds_f16_partials_in_f32(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest, destDevice):
    from tensilelite.KernelHelperNaming import conversionKernelNames, initConversionKernelObjects

    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, GlobalSplitU=2,
                       GlobalSplitUAlgorithm="MultipleBuffer")
    assert sol.get("Valid") is True, out
    kernels = initConversionKernelObjects(sol, gfx1250_iim)
    # The host picks VW8 for f16 accumulation when M % 8 == 0.
    assert sorted({k.num_elements_load for k in kernels}) == [1, 2, 4, 8]
    assert sorted({n[n.rindex("_VW"):] for n in conversionKernelNames(sol)}) == ["_VW1", "_VW2", "_VW4", "_VW8"]
    for k in kernels:
        assert k.kernelName.startswith("Cijk_H")
        header = k.getHeaderFileString()
        _, source = k.getSourceFileString()
        # The host passes a Half alpha/beta as f32.
        assert "  float alpha;" in header and "  float beta;" in header
        # The first slice converts exactly; each later one is one f32 add (x * 1.0 + sum, one rounding).
        assert "accum[e] = (float)__builtin_bit_cast(tensile_half, (uint16_t)((e & 1) ? (w >> 16) : w));" in source
        assert "v_fma_mix_f32 %0, %1, 1.0, %2 op_sel_hi:[1,1,0]" in source
        assert "(arg.W + idxW + (g0 + u) * strideW);" in source
        # The sum is rounded once to f16 and widened; the f32 epilogue then runs: alpha, then beta, then D.
        assert "  float accum[NUM_ELEMENT_LOAD] = {0};" in source
        assert "accum[e] = (float)(tensile_half)accum[e];" in source
        assert "  accum[0] *= (float)arg.alpha;" in source
        assert "    accum[0] += arg.beta * (float)arg.C[idxC+0];" in source
        assert "fmaf" not in source
        # alpha == 0 never reads a slice that overflowed to inf.
        assert "  if(arg.alpha != 0.0f)" in source
        # An accumulator that overflowed to inf saturates an fp8 D like a finite overflow.
        assert ("__builtin_copysignf(448.0f, accum[0])" in source) is (dest == "F8")
        assert "  result[0] = (%s)accum[0];" % destDevice in source


@pytest.mark.parametrize("dest", ["F8", "B"])
@pytest.mark.parametrize("problem, overrides", [
    ({}, {}),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True}),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True}),
    ({"DataTypeA": "F4", "DataTypeB": "F4"}, {"ConvertAfterDS": True}),
], ids=["f8f8", "f8f4", "f4f8", "f4f4"])
def test_fp8_and_bf16_outputs_are_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem, overrides, dest):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={**problem, "DestDataType": dest}, **overrides)
    assert sol.get("Valid") is True, out
    assert sol["MIRegPerOut"] == 0.5


@pytest.mark.parametrize("problem, overrides", [
    ({}, {}),
    ({"DestDataType": "F8"}, {"GlobalSplitU": 2, "GlobalSplitUAlgorithm": "MultipleBuffer"}),
    ({"DestDataType": "B"}, {"StaggerU": 0}),
    ({"DataTypeA": "F4", "DataTypeB": "F4"}, {"ConvertAfterDS": True, "TransposeLDS": 1, "LocalReadVectorWidth": 16}),
], ids=["h", "f8_gsu2", "bf16_no_stagger", "f4f4"])
def test_lds_segment_b_rotate_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem, overrides):
    params = {"DepthU": 256, "HalfPLR": 0, "TDMSplit": False, "LDSSegmentInterleave": 1, "LDSSegmentBRotate": 1,
              "TransposeLDS": -1, "LocalReadVectorWidth": -1, "VectorWidthA": -1, "SourceSwap": True,
              "ExpandPointerSwap": False, "AssertSummationElementMultiple": 128, **overrides}
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem, mi=(16, 16, 128, 1, 1, 4, 4, 2, 2), **params)
    assert sol.get("Valid") is True, out
    assert sol["LDSSegmentBRotate"] == 1 and sol["MIRegPerOut"] == 0.5


_MN_MAJOR_B_INTERLEAVE = {"DepthU": 256, "HalfPLR": 0, "TDMSplit": False, "LDSSegmentInterleave": 1, "TransposeLDS": -1,
                          "LocalReadVectorWidth": -1, "VectorWidthA": -1, "VectorWidthB": 1, "SourceSwap": True,
                          "ExpandPointerSwap": False, "AssertSummationElementMultiple": 128,
                          "AssertFree1ElementMultiple": 2, "ConvertAfterDS": True, "LDSTrInst": True}


@pytest.mark.parametrize("problem", [{"DataTypeB": "F4", "TransposeB": True},
                                     {"DataTypeA": "F4", "DataTypeB": "F4", "TransposeB": True}], ids=["f8f4", "f4f4"])
@pytest.mark.parametrize("brotate", [0, 1])
def test_lds_segment_interleave_with_mn_major_fp4_b_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem,
                                                             brotate):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem, mi=(16, 16, 128, 1, 1, 4, 4, 2, 2),
                       LDSSegmentBRotate=brotate, **_MN_MAJOR_B_INTERLEAVE)
    assert sol.get("Valid") is True, out
    assert sol["LDSSegmentBRotate"] == brotate and sol["LDSSegInterleaveOffsets"]["bMnMajor"]


# Each component holds half of B's tiles, so an odd MIWaveTile[1] has no split.
@pytest.mark.parametrize("waveTile1", [1, 3])
def test_lds_segment_interleave_with_mn_major_fp4_b_needs_an_even_wave_tile(_gp_gfx1250, gfx1250_iim, assembler,
                                                                            capsys, waveTile1):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DataTypeB": "F4", "TransposeB": True},
                       mi=(16, 16, 128, 1, 1, 4, waveTile1, 2, 2), LDSSegmentBRotate=0, **_MN_MAJOR_B_INTERLEAVE)
    assert sol.get("Valid") is False
    assert "MN-major B: needs an even MIWaveTile[1]" in out


def test_segment_interleave_sizes_an_fp4_operand_by_the_fp4_it_holds_in_lds():
    from tensilelite.SolutionStructs.segment_interleave import _bpe

    problem = ProblemType(_problem_config(DataTypeB="F4", TransposeB=True), False)
    state = {"ProblemType": problem, "ConvertAfterDS": True}
    # B is widened to its F8 MAC type after the LDS read, so LDS holds 0.5 B per element.
    assert problem["MacDataTypeB"].numBytes() == 1
    assert _bpe(state, "B") == 0.5 and _bpe(state, "A") == 1


# The TDMSplit term cannot be set alone: an earlier rule refuses TDMSplit first.
def test_lds_segment_interleave_with_mn_major_fp4_b_needs_wave_group_2x2(_gp_gfx1250, gfx1250_iim, assembler,
                                                                         capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DataTypeB": "F4", "TransposeB": True},
                       mi=(16, 16, 128, 1, 1, 2, 4, 4, 1), LDSSegmentBRotate=0, **_MN_MAJOR_B_INTERLEAVE)
    assert sol.get("Valid") is False
    assert "MN-major B: needs MIWaveGroup [2,2], no TDMSplit, and B reads within a component" in out


def _mn_major_b_state(**overrides):
    state = {"LDSSegmentInterleave": 1, "UnrollMajorLDSB": False, "enableLDSTrB": True, "VectorWidthB": 1,
             "MIWaveTile": [4, 4],
             "ProblemType": ProblemType(_problem_config(DataTypeB="F4", TransposeB=True), False)}
    state.update(overrides)
    return state


@pytest.mark.parametrize("overrides", [
    {"LDSSegmentInterleave": -1}, {"UnrollMajorLDSB": True}, {"enableLDSTrB": False}, {"VectorWidthB": 2},
    {"MIWaveTile": [4, 1]}, {"MIWaveTile": [4, 3]},
    {"ProblemType": ProblemType(_problem_config(TransposeB=True), False)},
    {"ProblemType": ProblemType(_problem_config(DataTypeB="F4", TransposeB=True, HighPrecisionAccumulate=True), False)},
], ids=["auto", "unroll_major", "no_transpose_read", "vector_width_2", "odd_wave_tile_1",
        "odd_wave_tile_3", "fp8_b", "f32_accumulation"])
def test_mn_major_b_interleave_needs_a_request_and_an_f16_accumulated_fp4_b(overrides):
    assert segMnMajorBRequested(_mn_major_b_state())
    assert not segMnMajorBRequested(_mn_major_b_state(**overrides))


def test_fp8_output_stochastic_rounding_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": "F8", "StochasticRounding": True})
    assert sol.get("Valid") is False
    assert _EPILOGUE in out


# Large-K F8 TN tuning configurations with only HighPrecisionAccumulate=False and
# ComputeDataType=h changed; AssertFree0ElementMultiple stays at its default. The last one writes bf16.
_TUNING_FORK = {
    "UseSgprForGRO": 0, "ForceDisableShadowInit": True, "DepthU": 256, "TDMInst": 3, "HalfPLR": 1,
    "TDMIterateMode": -1, "TransposeLDS": -1, "LDSTrInst": False, "StaggerU": 0, "ScheduleIterAlg": 4,
    "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "StoreRemapVectorWidth": 0, "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer", "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1,
    "VectorWidthA": -1, "VectorWidthB": -1, "LDSSegmentInterleave": 1, "LocalReadVectorWidth": -1,
    "1LDSBuffer": 0, "SourceSwap": True, "WorkGroupMapping": 1, "StoreVectorWidth": -1, "WavefrontSize": 32,
    "ExpandPointerSwap": False, "CompactLoopStore": False, "PreloadKernArgs": True,
    "AssertFree0ElementMultiple": 1, "AssertSummationElementMultiple": 1,
}
_TUNING_GROUP_STORE = {"NumElementsPerBatchStore": 16, "GroupLoadStore": True, "InterleaveAlpha": 2,
                       "InitCIterWmma": 0, "SwInstructionPrefetch": 0}


@pytest.mark.parametrize("dest, overrides", [
    ("F8", {"TDMPlusLdsBuf": -1, "PrefetchGL2": 2, "ClusterDim": [4, 4]}),
    ("F8", {**_TUNING_GROUP_STORE, "ClusterDim": [2, 4], "PrefetchGL2": 2}),
    ("F8", {**_TUNING_GROUP_STORE, "ClusterDim": [4, 4], "PrefetchGL2": 1}),
    ("B", {"InitCIterWmma": 0, "SwInstructionPrefetch": 0, "ClusterDim": [2, 4], "PrefetchGL2": 1}),
], ids=["fp8_out", "fp8_out_group_store", "fp8_out_group_store_cluster4x4", "bf16_out"])
def test_fp8_tuning_configs_derive_with_f16_accumulation(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest, overrides):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest},
                       mi=(16, 16, 128, 1, 1, 8, 8, 2, 2), **{**_TUNING_FORK, **overrides})
    assert sol.get("Valid") is True, out
    assert sol["MIRegPerOut"] == 0.5
    svw = sol["StoreVectorWidth"]
    assert svw & (svw - 1) == 0, svw


def test_partial_global_read_vector_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"TransposeA": False}, TDMInst=0,
                       GlobalReadVectorWidthA=8)
    assert sol.get("Valid") is False
    assert "cannot be shifted for a partial global-read vector" in out


def test_fp4_operand_converted_after_ds(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DataTypeB": "F4"}, ConvertAfterDS=True)
    assert sol.get("Valid") is True, out


_F16_ACCUMULATION_CAPS = [("archCaps", "HasF16Accumulation")]


@pytest.mark.parametrize("table, key", _F16_ACCUMULATION_CAPS)
@pytest.mark.parametrize("stepping, expected", [("gfx1250", True), ("gfx1250:xnack-", True), ("gfx1250-strict", False),
                                                ("gfx1250-strict:xnack-", False)])
def test_f16_accumulation_caps_are_declared_for_every_stepping(gfx1250_iim, stepping, expected, table, key):
    iim = copy.deepcopy(gfx1250_iim)
    applyArchCapOverrides(iim, [stepping])
    caps = getattr(iim[ISA_GFX1250], table)
    assert key in caps and bool(caps[key]) is expected, caps.get(key)


def test_fp4_widening_emits_the_instructions_the_schedule_budgets(_rocisa_gfx1250):
    from types import SimpleNamespace

    from tensilelite.Components.LocalRead import widenFp4ToFp8
    from tensilelite.KernelWriter import KernelWriter

    insts = []
    widenFp4ToFp8(SimpleNamespace(add=insts.append), "ValuA_X0_I0+%u", 0, 1)
    names = [str(inst).split()[0] for inst in insts]
    assert "v_cvt_scalef32_pk8_fp8_f16" in names, names
    assert len(names) == KernelWriter.numInstPerFp4Vgpr(None)


# =========================================================================== #
# rocisa spelling of the f16-accumulate WMMA.
# =========================================================================== #
@pytest.fixture
def _rocisa_gfx1250(gfx1250_iim):
    from rocisa import rocIsa

    ri = rocIsa.getInstance()
    ri.init(ISA_GFX1250, shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
    ri.setKernel(ISA_GFX1250, 32)


@pytest.mark.parametrize("strict", [False, True], ids=["gfx1250", "strict"])
@pytest.mark.parametrize("k, accType, instType", [
    (64, "INST_F16", "INST_F8"), (128, "INST_F16", "INST_F8"), (128, "INST_F32", "INST_F8"),
    (64, "INST_F16", "INST_F8_BF8"), (128, "INST_F16", "INST_F8_BF8"), (64, "INST_F16", "INST_BF8_F8"),
    (128, "INST_F16", "INST_BF8_F8"), (64, "INST_F16", "INST_BF8"), (128, "INST_F16", "INST_BF8")])
def test_wmma_spelling(_rocisa_gfx1250, k, accType, instType, strict):
    # As rocisa's MFMAInstruction, in the instruction text and in the StinkyTofu lowering: the zero scales and the
    # matrix formats belong to the scaled f8f6f4 form only. strict is the gfx1250-strict toggle.
    import rocisa
    from rocisa import rocIsa
    from rocisa.code import Module, SignatureBase
    from rocisa.container import vgpr
    from rocisa.enum import InstType
    from rocisa.instruction import MFMAInstruction
    from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

    accRegs = 4 if accType == "INST_F16" else 8
    numRegs = k // 8
    operands = "v[0:%u], v[16:%u], v[32:%u], v[0:%u]" % (accRegs - 1, 15 + numRegs, 31 + numRegs, accRegs - 1)
    if accType == "INST_F32" and strict:
        want = "v_wmma_scale_f32_16x16x128_f8f6f4 %s, 0, 0 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP8" \
            % operands
    else:
        pair = {"INST_F8": "fp8_fp8", "INST_F8_BF8": "fp8_bf8", "INST_BF8_F8": "bf8_fp8", "INST_BF8": "bf8_bf8"}[instType]
        want = "v_wmma_%s_16x16x%u_%s %s" % (accType[5:].lower(), k, pair, operands)
    with preserve_rocisa_kernel_state():
        rocIsa.getInstance().setForceScaledWMMA(strict)
        inst = MFMAInstruction(instType=getattr(InstType, instType), accType=getattr(InstType, accType),
                               variant=[16, 16, k, 1],
                               mfma1k=False, acc=vgpr(0, accRegs), a=vgpr(16, numRegs), b=vgpr(32, numRegs),
                               acc2=vgpr(0, accRegs))
        assert str(inst).split("//")[0].strip() == want
        module = Module("k")
        module.add(inst)
        module.setParent()
        signature = SignatureBase(kernelName="k", kernArgsVersion=1, codeObjectVersion="4", groupSegmentSize=0,
                                  sgprWorkGroup=(1, 1, 0), vgprWorkItem=0, flatWorkGroupSize=64, numSgprPreload=0)
        lowered = rocisa.toStinkyTofuModule(module, (12, 5, 0), "k", signature=signature,
                                            options={"OptLevel": 0}).emitAssembly()
    assert re.findall(r"^\s*(v_wmma\S* [^/\n]*?)\s*(?://.*)?$", lowered, re.M) == [want]


# =========================================================================== #
# Emitted assembly (--build-only; no GPU needed).
# =========================================================================== #
def _assembly_files(config, tensile_args, tmp_path, target="gfx1250"):
    output_dir = tmp_path / "output"
    run.main([str(_GFX12_CONFIGS / config), str(output_dir), "--build-only",
              "--global-parameters", "KeepBuildTmp=True", *tensile_args, "--gpu-targets", target])
    files = {p.name: p.read_text() for p in output_dir.rglob("*.s")}
    assert files, "no assembly generated"
    return files


def _build_assembly(config, tensile_args, tmp_path, target="gfx1250"):
    return "\n".join(_assembly_files(config, tensile_args, tmp_path, target).values())


_REG = r"(v\[vgprValuC(?:\+\d+)?\]|v\d+)"


@pytest.mark.parametrize("config, store", [
    ("f8f8hh_gfx1250.yaml", "buffer_store_b16"),
    ("f8f4hh_gfx1250.yaml", "buffer_store_b16"),
    ("f8f8f8h_gfx1250.yaml", "buffer_store_b8"),
    ("f8f8bh_gfx1250.yaml", "buffer_store_b16"),
])
def test_width_one_stores_widen_each_element_from_its_half(gfx1250_iim, tensile_args, tmp_path, config, store):
    files = _assembly_files(config, tensile_args, tmp_path)
    widthOne = {name: text for name, text in files.items() if re.search(r"^%s .*// store D" % store, text, re.M)}
    assert widthOne, "no kernel stores a single element"
    for name, text in widthOne.items():
        halves = collections.Counter((int(v), h == ".h") for v, h in re.findall(
            r"^v_cvt_f32_f16 \S+, v\[vgprValuC\+(\d+)\](\.[hl])?\s+// MI out element", text, re.M))
        vgprs = {v for v, _ in halves}
        # Each MI output VGPR is widened from its low half and from its high half equally often.
        assert vgprs and all(halves[(v, False)] == halves[(v, True)] > 0 for v in vgprs), (name, halves)


def _global_write_sections(text, bytesPerElement):
    # Per GSU1 or MB (MultipleBuffer) store section: D elements stored, single-element stores, alpha multiplies
    # and f16/f32 multiplies or FMAs of any kind.
    sections, key = {}, None
    for line in text.splitlines():
        label = re.match(r"label_(\w+):", line)
        if label:
            mode = re.fullmatch(r"GW_B\d+_FD\d+_VW\d+_(GSU1|MB|MBSK)_(?:NonEdge|Then|Else)(?:_\d+)?", label.group(1))
            key = (mode.group(1), label.group(1)) if mode else None
            if key:
                sections[key] = collections.Counter()
            continue
        if key is None:
            continue
        # MBSK stores D through its SrdTD ("store TD").
        store = re.match(r"buffer_store_(d16_hi\S*|b(\d+)) .*// store T?D\b", line)
        if store:
            elements = (16 if store.group(1).startswith("d16_hi") else int(store.group(2))) // 8 // bytesPerElement
            sections[key]["elements"] += elements
            sections[key]["singles"] += elements == 1
        alpha = re.match(r"v_(pk_)?mul_f(?:16|32) \S+, s\[sgprAlpha", line)
        sections[key]["alpha"] += (2 if alpha.group(1) else 1) if alpha else 0
        sections[key]["arithmetic"] += bool(re.match(r"v_(?:pk_)?(?:mul|fma)_f(?:16|32) ", line))
    return sections


@pytest.mark.parametrize("config", ["f16acc_gsu_edge_beta0_gfx1250.yaml", "f16acc_overflow_gfx1250.yaml",
                                    "f16acc_adaptive_gsua_gfx1250.yaml"])
def test_multiple_buffer_partials_leave_unscaled_and_gsu1_scales_once(gfx1250_iim, tensile_args, tmp_path, config):
    # The reduction kernel applies alpha and beta to MultipleBuffer partials, so their stores only move the MI outputs.
    files = _assembly_files(config, tensile_args, tmp_path)
    seen = collections.Counter()
    for name, text in files.items():
        dest = re.search(r"_F8(F8|H|B)H_", name).group(1)
        for (mode, label), c in _global_write_sections(text, 1 if dest == "F8" else 2).items():
            if mode == "MB":
                assert c["alpha"] == 0 and c["arithmetic"] == 0, (name, label, c)
                seen["MB singles"] += c["singles"] > 0
            else:
                # Alpha scales every stored element once, in f32.
                assert c["alpha"] == c["elements"], (name, label, c)
                seen["GSU1 singles"] += c["singles"] > 0
    assert seen["MB singles"] and seen["GSU1 singles"], seen


def _mbsk_regions(text):
    # MBSK partial writes and synchronization (reduction start to the reduction body) and the reduction (to its end).
    regions, state = collections.defaultdict(list), None
    for line in text.splitlines():
        label = re.match(r"label_(Reduction_Start|reduction_body|Reduction_End)(?:_\d+)?:", line)
        if label:
            state = {"Reduction_Start": "partial", "reduction_body": "reduction"}.get(label.group(1))
        elif state and line.strip() and not line.startswith(("label_", "/", ".")):
            regions[state].append(line.split("//")[0].strip())
    return regions


def _streamk_f16_partial_loops(text):
    # The f16 fixup's loops over partials, one per block of accumulator dwords.
    loops, inside = [], False
    for line in text.splitlines():
        if re.match(r"label_SK_F16_AddPartial(?:_\d+)?:", line):
            loops.append([])
            inside = True
        elif re.match(r"label_SK_F16_AddPartialEnd(?:_\d+)?:", line):
            inside = False
        elif inside and line.strip() and not line.startswith(("label_", "/", ".")):
            loops[-1].append(line.split("//")[0].strip())
    return loops


def test_streamk_partials_are_added_in_f32_and_rounded_once(gfx1250_iim, tensile_args, tmp_path, capfd):
    # Linear slots hold raw f16 partials, added with v_fma_mix_f32; tree slots hold f32 sums, added with v_add_f32.
    # Either way the sums are rounded to f16 once and the final store scales them like a GSU1 store. The
    # space-filling walk's s_cselect_b64 and s_flbit_i32_b32 go through StinkyTofu on gfx1250 as well; only the
    # walk emits s_flbit_i32_b32.
    files = _assembly_files("f16acc_streamk_gfx1250.yaml", tensile_args, tmp_path)
    # A kernel whose generation raises is dropped with a warning while the build goes on.
    assert "Failed to generate assembly source code" not in "".join(capfd.readouterr())
    seen = collections.Counter()
    walk = set()
    for name, text in files.items():
        if re.search(r"^\s*s_flbit_i32_b32 ", text, re.M):
            seen["space filling"] += 1
            walk.update(re.findall(r"^\s*(s_cselect_b64|s_flbit_i32_b32) ", text, re.M))
        loops = _streamk_f16_partial_loops(text)
        if not loops:
            assert "label_SK_F16_" not in text, name
            seen["whole tiles"] += 1
            continue
        kinds = set()
        for loop in loops:
            assert not [l for l in loop if re.match(r"v_pk_(?:add|mul|fma)_f16 ", l)], name
            adds = [l for l in loop if re.match(r"v_(?:add_f32|fma_mix_f32) ", l)]
            tree = bool(adds) and adds[0].startswith("v_add_f32 ")
            assert adds and all(l.startswith("v_add_f32 ") if tree else
                                (l.startswith("v_fma_mix_f32 ") and ", 1.0, " in l and "op_sel_hi:[1,1,0]" in l)
                                for l in adds), name
            kinds.add("tree" if tree else "linear")
        assert re.search(r"^\s*v_cvt_(?:pk_)?f16_f32 ", text, re.M), name
        assert ("label_SK_F16_StoreSums:" in text) == ("tree" in kinds), name
        seen.update(kinds)
    assert seen["linear"] and seen["tree"] and seen["whole tiles"] and seen["space filling"], seen
    assert walk == {"s_cselect_b64", "s_flbit_i32_b32"}, walk


def test_mbsk_partials_leave_unscaled_and_the_sum_is_added_in_f32_then_scaled_once(gfx1250_iim, tensile_args, tmp_path):
    # MBSK slices store raw f16 partials; the reduction adds them as the reduction kernel does and the final store
    # scales the sum like a GSU1 store.
    files = _assembly_files("f16acc_mbsk_beta0_gfx1250.yaml", tensile_args, tmp_path)
    seen = collections.Counter()
    for name, text in files.items():
        regions = _mbsk_regions(text)
        assert any("store WS" in l for l in text.splitlines()) and regions["reduction"], name
        assert not [l for l in regions["partial"] if re.match(r"v_(?:pk_)?(?:mul|fma)_f(?:16|32) ", l)], name
        assert not [l for l in regions["reduction"] if re.match(r"v_(?:pk_)?add_f(?:16|32) ", l)], name
        mix = [l for l in regions["reduction"] if l.startswith("v_fma_mix_f32 ")]
        assert mix and all(", 1.0, " in l and "op_sel_hi:[1,1,0]" in l for l in mix), name
        assert any(l.startswith("v_cvt_f16_f32 ") for l in regions["reduction"]), name
        dest = re.search(r"_F8(F8|H|B)H_", name).group(1)
        for (mode, label), c in _global_write_sections(text, 1 if dest == "F8" else 2).items():
            if mode != "MBSK":
                continue
            seen["MBSK"] += 1
            assert c["alpha"] == c["elements"], (name, label, c)
            seen["MBSK singles"] += c["singles"] > 0
    assert seen["MBSK"] and seen["MBSK singles"], seen


def test_fp8_output_saturates_and_converts_from_f32(gfx1250_iim, tensile_args, tmp_path):
    files = _assembly_files("f8f8f8h_gfx1250.yaml", tensile_args, tmp_path)
    text = "\n".join(files.values())
    assert "v_wmma_f16_16x16x64_fp8_fp8" in text
    assert "v_wmma_f16_16x16x128_fp8_fp8" in text
    assert "v_wmma_f32_" not in text
    # Each accumulator is widened to f32, scaled by alpha and added to C*beta in f32 (C widened exactly from fp8),
    # then saturated and converted to fp8.
    assert "v_pk_mul_f16" not in text and "v_pk_fma_f16" not in text
    assert re.search(r"^v_cvt_pk_f32_fp8 ", text, re.M)
    assert re.search(r"^v_(?:fmac|mac)_f32 .*s\[sgprBeta\]", text, re.M)
    assert re.search(r"^v_cvt_pk_fp8_f32 ", text, re.M)
    # The class mask is NaN only (0x3): inf from an f16 overflow is clamped to +/-448 like a finite overflow.
    assert re.search(r"^v_mov_b32 \S+, 0x3 +// Nan$", text, re.M) and "0x207" not in text
    assert re.search(r"0x43[eE]00000", text) and re.search(r"0xc3[eE]00000", text)
    assert re.search(r"^buffer_store_b8 .*// store D", text, re.M)
    assert re.search(r"^buffer_store_b(32|64) .*// store D", text, re.M)
    # inf is never turned into NaN before the clamp (the eight-wide conversion's x*0+x step).
    assert "x*0+x" not in text
    for name, body in files.items():
        widened = len(re.findall(r"^v_cvt_f32_f16 .*// MI out element", body, re.M))
        med3 = len(re.findall(r"^v_med3_\w*f32 ", body, re.M))
        maxmin = len(re.findall(r"^v_maximum\w*minimum\w*_f32 ", body, re.M))
        # Every widened value is clamped to [-448, 448] once, unless it is NaN, before its conversion to fp8: one at
        # a time (v_cmp_class_f32 with the NaN-only mask, then v_med3_f32) or eight at a time (NaN-propagating clamp).
        assert widened and widened == med3 + maxmin, (name, widened, med3, maxmin)
        assert med3 == len(re.findall(r"^v_cmp_class_f32\w* ", body, re.M)), name


def _epilogue_stores_of_unwritten_registers(text):
    # The D stores of the GSU1 / MBSK / OptNLL epilogue sections whose first data VGPR nothing earlier in the
    # section wrote: such a store writes an accumulator or a stale register instead of the converted result.
    base = int(re.search(r"^\.set vgprValuC, (\d+)", text, re.M).group(1))

    def first(op):
        m = re.match(r"v\[vgprValuC\+(\d+)", op)
        if m:
            return base + int(m.group(1))
        m = re.match(r"v\[?(\d+)", op)
        return int(m.group(1)) if m else None

    bad, written, live = [], set(), False
    for line in text.splitlines():
        x = line.split("//")[0].strip()
        if x.endswith(":"):
            live, written = bool(re.match(r"label_GW_B\d+_FD\d+_VW\d+_(GSU1|MBSK|OptNLL)", x)), set()
            continue
        if not live or not x:
            continue
        op, _, args = x.partition(" ")
        dst = first(args.split(",")[0].strip())
        if op.startswith("buffer_store") and "store D" in line:
            if dst is not None and dst not in written:
                bad.append(line.strip())
        elif dst is not None:
            # Every VGPR of a destination range, e.g. the v[30:31] a StoreRemap read-back loads.
            written.update(_vgprs(args.split(",")[0].strip().split()[0], base) or [dst])
    return bad


def _beta_reads_of_unloaded_c_halves(text):
    # The beta FMAs (v_fma_mix_f32 with s[sgprBeta]) of the GSU1 / MBSK / OptNLL epilogue sections that read a half of
    # a C register no earlier C load of the section wrote: d16 loads fill the low half, d16_hi the high half, wider
    # loads every half of their registers.
    loaded, live, bad = {}, False, []
    for line in text.splitlines():
        x = line.split("//")[0].strip()
        if x.endswith(":"):
            live, loaded = bool(re.match(r"label_GW_B\d+_FD\d+_VW\d+_(GSU1|MBSK|OptNLL)", x)), {}
            continue
        if not live or not x:
            continue
        load = re.match(r"buffer_load_(d16_hi_b16|d16_b16|b32|b64|b96|b128) v\[?(\d+)(?::(\d+))?", x)
        if load and "load C" in line:
            first, last = int(load.group(2)), int(load.group(3) or load.group(2))
            for r in range(first, last + 1):
                loaded.setdefault(r, set()).update({"d16_b16": {0}, "d16_hi_b16": {1}}.get(load.group(1), {0, 1}))
            continue
        beta = re.match(r"v_fma_mix_f32 \S+, s\[sgprBeta\], v\[?(\d+)\]?, \S+ op_sel:\[0,(\d),0\]", x)
        if beta and int(beta.group(2)) not in loaded.get(int(beta.group(1)), set()):
            bad.append(line.strip())
    return bad


def test_fp8_kernels_accumulate_in_f16_and_widen_for_the_epilogue(gfx1250_iim, tensile_args, tmp_path):
    text = _build_assembly("f8f8hh_gfx1250.yaml", tensile_args, tmp_path)
    assert not _epilogue_stores_of_unwritten_registers(text)
    assert not _beta_reads_of_unloaded_c_halves(text)
    assert "v_wmma_f16_16x16x64_fp8_fp8" in text
    assert "v_wmma_f16_16x16x128_fp8_fp8" in text
    assert "v_wmma_f32_" not in text
    # alpha and beta*C in f32 on accumulators widened from f16; no f16 epilogue arithmetic.
    assert "v_pk_mul_f16" not in text and "v_pk_fma_f16" not in text
    assert re.search(r"^v_cvt_f32_f16 \S+, v\[vgprValuC\+\d+\]", text, re.M)
    assert re.search(r"^v_mul_f32 \S+, s\[sgprAlpha\]", text, re.M)
    # The K loop starts at the launched StaggerU offset.
    assert "StaggerUIter" in text


@pytest.mark.parametrize("config", ["f16acc_brotate_gfx1250.yaml", "f16acc_brotate_mnf4_gfx1250.yaml"])
def test_b_rotate_swaps_whole_packed_accumulator_tiles(gfx1250_iim, tensile_args, tmp_path, config):
    files = _assembly_files(config, tensile_args, tmp_path)
    rotated = 0
    for name, body in files.items():
        if "B-rotate: SIMD pair 1" not in body:
            continue
        rotated += 1
        # B-rotate runs on MIWaveGroup [2, 2] with the 16x16 WMMA, so the wave tile is MT / 32, and a packed 16x16 tile
        # is four VGPRs: the first B half (wt1 / 2 tile columns) is wt1 / 2 * wt0 * 4 VGPRs.
        mt0, mt1 = map(int, re.search(r"Cijk_\w*?_MT(\d+)x(\d+)x\d+", body).groups())
        half = mt1 // 32 // 2 * (mt0 // 32) * 4
        swaps = [(int(a), int(b)) for a, b in
                 re.findall(r"^v_swap_b32 v\[vgprValuC\+(\d+)\], v\[vgprValuC\+(\d+)\]", body, re.M)]
        assert sorted(swaps) == [(r, r + half) for r in range(half)], name
    assert rotated


def test_bf16_output_does_alpha_and_beta_in_f32(gfx1250_iim, tensile_args, tmp_path):
    files = _assembly_files("f8f8bh_gfx1250.yaml", tensile_args, tmp_path)
    text = "\n".join(files.values())
    assert "v_wmma_f16_16x16x64_fp8_fp8" in text
    assert "v_wmma_f16_16x16x128_fp8_fp8" in text
    assert "v_wmma_f32_" not in text
    # Never alpha or beta in f16: each value is widened to f32, scaled by alpha and added to C*beta (C widened
    # exactly from bf16), then rounded to bf16 in place; an f16 overflow stays inf.
    assert "v_pk_mul_f16" not in text and "v_pk_fma_f16" not in text
    assert re.search(r"^v_cvt_f32_f16 \S+, \S+\.h +// MI out element", text, re.M)
    assert re.search(r"^buffer_store_b16 .*// store D", text, re.M)
    assert re.search(r"^buffer_store_b(32|64|128) .*// store D", text, re.M)
    for name, body in files.items():
        widened = len(re.findall(r"^v_cvt_f32_f16 .*// MI out element", body, re.M))
        alphas = len(re.findall(r"^v_mul_f32 \S+, s\[sgprAlpha\]", body, re.M))
        betas = len(re.findall(r"^v_(?:fmac|mac)_f32 .*s\[sgprBeta\]", body, re.M))
        packed = len(re.findall(r"^v_cvt_pk_bf16_f32 ", body, re.M))
        rounded = len(re.findall(r"^v_add3_u32 .*increment for rounding", body, re.M))
        assert widened and alphas <= widened and betas <= widened, name
        # One conversion per pair (or per single element), or the integer rounding per value.
        assert widened / 2 <= packed <= widened or rounded == widened, (name, widened, packed, rounded)


def _fp4_operands(config):
    return {tc for tc, t in zip("AB", (config[:2], config[2:4])) if t == "f4"}


def _converted_operands(text, write):
    # Found from each operand's own registers, so that one fp4 operand's conversion cannot stand in for the other's.
    return {tc for tc in "AB" if re.search(r"^%s v\[vgprValu%s_" % (write, tc), text, re.M)}


@pytest.mark.parametrize("config", ["f8f4hh_gfx1250.yaml", "f8f4f8h_gfx1250.yaml", "f8f4bh_gfx1250.yaml",
                                    "f4f8hh_gfx1250.yaml", "f4f8f8h_gfx1250.yaml", "f4f8bh_gfx1250.yaml",
                                    "f4f4hh_gfx1250.yaml", "f4f4f8h_gfx1250.yaml", "f4f4bh_gfx1250.yaml"])
def test_fp4_operand_is_converted_to_fp8(gfx1250_iim, tensile_args, tmp_path, config):
    text = _build_assembly(config, tensile_args, tmp_path)
    # Both half-waves need a 1.0 scale byte.
    assert set(re.findall(r"v_cvt_scale_pk8_f16_fp4 \S+, \S+, (\S+)", text)) == {str(0x7f7f7f7f)}
    assert _converted_operands(text, "v_cvt_scalef32_pk8_fp8_f16") == _fp4_operands(config)
    assert "v_wmma_f16_16x16x64_fp8_fp8" in text


def test_fp8_operand_beside_fp4_reads_whole_vectors(gfx1250_iim, tensile_args, tmp_path):
    text = _build_assembly("f8f4hh_gfx1250.yaml", tensile_args, tmp_path)
    # The second 16-element read of the unconverted fp8 operand starts 32 K elements in.
    assert re.search(r"ds_load_b128 v\[vgprValuA_X0_I0\+4:vgprValuA_X0_I0\+4\+3\], "
                     r"v\[vgprLocalReadAddrA\+0\] offset:32\b", text)


def test_mn_contiguous_fp4_is_transposed_on_read_and_converted(gfx1250_iim, tensile_args, tmp_path):
    text = _build_assembly("f16acc_mnf4_gfx1250.yaml", tensile_args, tmp_path)
    for tc in "AB":
        assert re.search(r"ds_load_tr4_b64 v\[vgprValu%s_" % tc, text), tc
    assert _converted_operands(text, "v_cvt_scalef32_pk8_fp8_f16") == {"A", "B"}
    # The MN-contiguous fp8 operand beside an fp4 one keeps the fp8 transpose read.
    assert re.search(r"ds_load_tr8_b64 v\[vgprValuA_", text)
    assert "v_wmma_f16_16x16x128_fp8_fp8" in text


_LANE_OPS = {"v_and_b32": lambda a, b: a & b, "v_lshrrev_b32": lambda a, b: b >> a, "v_lshlrev_b32": lambda a, b: b << a,
             "v_add_nc_u32": lambda a, b: a + b, "v_lshl_add_u32": lambda a, b, c: (a << b) + c,
             "v_mul_u32_u24": lambda a, b: a * b}


def _wave_lane_offsets(section, wave=0):
    # Runs one operand's local read offset code for lanes 0..31 of a wave; returns the last register written.
    regs, dst = {}, None
    for op, args in re.findall(r"^(v_\w+) ([^/\n]*)", section, re.M):
        assert op in _LANE_OPS, op
        dst, *srcs = [a.strip() for a in args.split(",")]
        vals = [[32 * wave + lane for lane in range(32)] if re.fullmatch(r"v\[vgprSerial(-\d+)?\]", s) else regs[s]
                if s in regs else [int(s, 0)] for s in srcs]
        regs[dst] = [_LANE_OPS[op](*(v[lane % len(v)] for v in vals)) for lane in range(32)]
    return regs[dst]


def test_mn_contiguous_fp4_lanes_read_in_fp8_wmma_k_order(gfx1250_iim, tensile_args, tmp_path):
    # Widened to fp8, half-wave h takes K 32g + 16h + 0..15 in registers 4g..4g+3, and ds_load_tr4_b64 gives lane t
    # nibble j from the row addressed by lane 8*(t // 16) + j % 8 + 16*(j // 8). So lane L must address K row
    # L % 8 + 8*(L // 16) + 16*((L % 16) // 8) of the operand's MN block, and the loads of one block step 32 rows.
    rows = [lane % 8 + 8 * (lane // 16) + 16 * ((lane % 16) // 8) for lane in range(32)]
    checked = set()
    for name, text in _assembly_files("f16acc_mnf4_gfx1250.yaml", tensile_args, tmp_path).items():
        for tc, tile in (("A", "lr0I"), ("B", "lr1J")):
            loads = re.findall(r"^ds_load_tr4_b64 v\[vgprValu%s_X0_I0\+0\+(\d+):.* offset:(\d+)" % tc, text, re.M)
            if not loads:
                continue
            section = re.search(r"/\* %s \*/\n(.*?)(?=/\* lr|/\* local read addresses: final)" % tile, text, re.S).group(1)
            stride = int(re.search(r"t\*=strideUnroll\((\d+)\)", section).group(1))
            offsets = _wave_lane_offsets(section)
            assert [o // stride for o in offsets] == rows and all(o % stride == 0 for o in offsets), (name, tc, offsets)
            final = re.search(r"final offsets %s \*/\n(.*?)\n\n" % tc.lower(), text, re.S).group(1)
            padding = re.search(r"padding (\d+) per block (\d+)", final)
            pad, block = map(int, padding.groups()) if padding else (0, 1)
            step = 32 * stride // 2
            byReg = {}
            for r, o in loads:
                byReg.setdefault(int(r), int(o))
            assert byReg[4] - byReg[0] == step + step // block * pad, (name, tc, loads[:2])
            checked.add(tc)
    assert checked == {"A", "B"}


def test_mn_major_fp4_b_rotate_reads_each_tile_from_its_component(gfx1250_iim, tensile_args, tmp_path):
    # Interleaved, an MN-major B is two [DepthU][MT1/2] blocks S bytes apart from ldsBaseB, each padded from its own
    # start, so B(k, n) is at ldsBaseB + c*S + pad((k*MT1/2 + n - c*MT1/2) / 2) with c = n // (MT1/2). TDM writes
    # component c (one N half) at ldsBaseB + c*S. SIMD pair 1 (odd waves) reads tile slot t from tile (t + nT/2) % nT.
    # Lane L of wave w reads K row L % 8 + 8*(L // 16) + 16*((L % 16) // 8), 16 columns from tile u's u*32 + (w // 2)*16.
    rows = [lane % 8 + 8 * (lane // 16) + 16 * ((lane % 16) // 8) for lane in range(32)]
    checked = 0
    for name, text in _assembly_files("f16acc_brotate_mnf4_gfx1250.yaml", tensile_args, tmp_path).items():
        if "B-rotate: pair * S" not in text:
            continue
        half = int(re.search(r"_MT\d+x(\d+)x\d+", name).group(1)) // 2
        base = int(re.search(r"0x([0-9a-f]+), v\[vgprLocalReadAddrB\+0(?:-\d+)?\]\s*//\s*\+= LdsOffsetB", text).group(1), 16)
        stride = int(re.search(r"(\d+)\s*// B-rotate: pair \* S", text).group(1))
        assert str(stride) in re.findall(r"s_mul_i32 \w+, \w+, (\d+)\s*// woffset = wId \* \(mt // numComp", text), name
        assert str(base) in re.findall(r"s_add_u32 \w+, \w+, (\d+)\s*// ldsOffset = woffset \+ ldsConstOffset", text), name
        assert len(re.findall(r"one N half per component", text)) >= 2, name
        section = re.search(r"/\* lr1J \*/\n(.*?)(?=/\* lr|/\* local read addresses: final)", text, re.S).group(1)
        padding = re.search(r"padding (\d+) per block (\d+)", re.search(r"final offsets b \*/\n(.*?)\n\n", text, re.S).group(1))
        pad, block = map(int, padding.groups()) if padding else (0, 1)
        first = {}
        for slot, regOffset, reg, imm in re.findall(r"^ds_load_tr4_b64 v\[vgprValuB_X0_I0\+(\d+)\+(\d+):[^\]]*\], "
                                                     r"v\[vgprLocalReadAddrB\+(\d+)(?:-\d+)?\](?: offset:(\d+))?", text, re.M):
            first.setdefault((int(slot), int(regOffset)), (int(reg), int(imm or 0)))
        slots = sorted({s for s, _ in first})
        nT = len(slots)
        for wave in range(4):
            pair = wave % 2
            addr0 = [base + pair * stride + o // 2 + o // 2 // block * pad for o in _wave_lane_offsets(section, wave)]
            addr = (addr0, [a + 65536 - 2 * pair * stride for a in addr0])
            for (slot, regOffset), (reg, imm) in first.items():
                comp, col = divmod((slots.index(slot) + pair * nT // 2) % nT * 32, half)
                for lane in range(32):
                    elem = ((regOffset // 4 * 32 + rows[lane]) * half + col + wave // 2 * 16) // 2
                    want = base + comp * stride + elem + elem // block * pad
                    assert addr[reg][lane] + imm == want, (name, wave, slot, regOffset, lane)
        checked += 1
    assert checked


# =========================================================================== #
# Static checks of the epilogue on every f16-accumulation yaml (kernel sources only).
# =========================================================================== #
_F16_YAMLS = sorted(p.name for p in _GFX12_CONFIGS.glob("*_gfx1250.yaml")
                    if re.fullmatch(r"(f16acc_.*|[fb][48][fb][48](hh|f8h|bh))_gfx1250\.yaml", p.name))
_EPILOGUE_SECTION = re.compile(r"label_GW_B\d+_FD\d+_VW\d+_(GSU1|MBSK|OptNLL)")
# The conversions of an f32 epilogue value to the stored D type.
_CONVERSIONS = {"v_cvt_pk_f16_f32", "v_cvt_f16_f32", "v_cvt_pk_bf16_f32", "v_and_or_b32", "v_perm_b32",
                "v_cvt_pk_fp8_f32", "v_cvt_scalef32_pk8_fp8_f32"}
# x*0+x, which turns +/-inf into NaN before the eight-wide fp8 clamp.
_X0X = re.compile(r"^v_pk_fma_f32\s+(\S+), \1, 0, \1\s*(?://.*)?$", re.M)
_C_LOAD_BYTES = {"u8": {0}, "d16_u8": {0}, "d16_hi_u8": {2}, "u16": {0, 1}, "b16": {0, 1}, "d16_b16": {0, 1},
                 "d16_hi_b16": {2, 3}}
_HALVES = {"l": {0, 1}, "h": {2, 3}}


def _generated_assembly(config, tensile_args, tmp_path):
    # The .s of every kernel of the yaml, without building the code objects.
    output_dir = tmp_path / "output"
    run.main([str(_GFX12_CONFIGS / config), str(output_dir), "--build-only", "--global-parameters", "KeepBuildTmp=True",
              "GenerateSourcesAndExit=True", *tensile_args, "--gpu-targets", "gfx1250"])
    files = {p.name: p.read_text() for p in output_dir.rglob("*.s")}
    assert files, "no assembly generated"
    return files


def _vgprs(op, base):
    # The VGPRs of operand op, v[vgprValuC+k] counted from base; a true16 half (.l/.h) names its whole register.
    m = re.fullmatch(r"v\[vgprValuC\+(\d+)(?::vgprValuC\+(\d+)(?:\+(\d+))?)?\](?:\.[lh])?", op)
    if m:
        last = int(m.group(2)) + int(m.group(3) or 0) if m.group(2) else int(m.group(1))
        return range(base + int(m.group(1)), base + last + 1)
    m = re.fullmatch(r"v\[(\d+)(?::(\d+))?\](?:\.[lh])?|v(\d+)(?:\.[lh])?", op)
    if not m:
        return range(0)
    first = int(m.group(1) or m.group(3))
    return range(first, int(m.group(2) or first) + 1)


def _epilogue_lines(text):
    # (instruction without its comment, line) for the lines of the GSU1 / MBSK / OptNLL epilogue sections; (None, label)
    # where each section starts.
    live = False
    for line in text.splitlines():
        x = line.split("//")[0].strip()
        if x.endswith(":"):
            live = bool(_EPILOGUE_SECTION.match(x))
            if live:
                yield None, line
        elif live and x:
            yield x, line


def _c_reads_of_unloaded_bytes(text):
    # The reads of C in the epilogue sections that take bytes no earlier C load of the section wrote: the f16 beta
    # (v_fma_mix_f32 op_sel), the bf16 widening (v_cvt_f32_bf16 .l/.h, or v_lshlrev_b32 16 and v_and_b32 with the
    # high-half mask) and the fp8 widening (v_cvt_pk_f32_fp8 .l/.h for bytes 0-1 and 2-3, v_cvt_f32_fp8 byte_sel).
    loaded, masks, bad = {}, set(), []
    for x, line in _epilogue_lines(text):
        if x is None:
            loaded = {}
            continue
        mask = re.match(r"v_mov_b32 v(\d+), 0xffff0000$", x)
        if mask:
            masks.add(int(mask.group(1)))
        load = re.match(r"buffer_load_(\w+) v\[?(\d+)(?::(\d+))?\]?,", x)
        if load and "load C" in line:
            for r in range(int(load.group(2)), int(load.group(3) or load.group(2)) + 1):
                loaded.setdefault(r, set()).update(_C_LOAD_BYTES.get(load.group(1), {0, 1, 2, 3}))
            continue
        read = None
        m = re.match(r"v_fma_mix_f32 \S+, s\[sgprBeta\], v\[?(\d+)\]?, \S+ op_sel:\[0,(\d),0\]", x)
        if m:
            read = int(m.group(1)), _HALVES["h" if m.group(2) == "1" else "l"]
        m = re.match(r"v_cvt_(?:pk_f32_fp8|f32_bf16) \S+, v(\d+)\.([lh])$", x)
        if m:
            read = int(m.group(1)), _HALVES[m.group(2)]
        m = re.match(r"v_cvt_f32_fp8 \S+, v(\d+) byte_sel:\[?(\d)\]?$", x)
        if m:
            read = int(m.group(1)), {int(m.group(2))}
        m = re.match(r"v_lshlrev_b32 \S+, 16, v(\d+)$", x)
        if m and int(m.group(1)) in loaded:
            read = int(m.group(1)), _HALVES["l"]
        m = re.match(r"v_and_b32 \S+, v(\d+), v(\d+)$", x)
        if m and int(m.group(2)) in masks:
            read = int(m.group(1)), _HALVES["h"]
        if read and not read[1] <= loaded.get(read[0], set()):
            bad.append(line.strip())
    return bad


def _stores_not_last_written_by_a_conversion(text):
    # The D stores of the epilogue sections whose data VGPRs a conversion to the D type did not write last (a
    # v_mov_b32 copy counts as its source's writer): such a store writes a widened, scaled or loaded value, or a stale
    # register, instead of the converted result.
    base = int(re.search(r"^\.set vgprValuC, (\d+)", text, re.M).group(1))
    last, bad = {}, []
    for x, line in _epilogue_lines(text):
        if x is None:
            last = {}
            continue
        op, _, args = x.partition(" ")
        operands = [a.strip().split()[0] for a in args.split(",") if a.strip()]
        if "storeRemap lw" in line:
            # StoreRemap stages the converted values in LDS; the D store stores what "storeRemap lr" reads back.
            writers = {last.get(r) for r in _vgprs(operands[1], base)}
            if not writers or not writers <= _CONVERSIONS:
                bad.append("%s (last writers %s)" % (line.strip(), sorted(map(str, writers))))
            continue
        if op.startswith(("buffer_store", "global_store", "ds_store", "ds_write", "buffer_atomic", "global_atomic")):
            if op.startswith("buffer_store") and re.search(r"// store T?D\b", line):
                writers = {last.get(r) for r in _vgprs(operands[0], base)}
                if not writers <= _CONVERSIONS | {"storeRemap lr"}:
                    bad.append("%s (last writers %s)" % (line.strip(), sorted(map(str, writers))))
            continue
        # A copy, or a shift that exposes the next byte of a stored register, keeps its source's writer.
        moves = op == "v_mov_b32" or (op in ("v_ashrrev_i32", "v_lshrrev_b32") and len(operands) > 2)
        src = _vgprs(operands[-1], base) if moves and len(operands) > 1 else range(0)
        for r in _vgprs(operands[0], base):
            last[r] = "storeRemap lr" if "storeRemap lr" in line else (last.get(src[0]) if len(src) == 1 else op)
    return bad


def test_epilogue_checks_report_synthetic_violations():
    text = "\n".join([
        ".set vgprValuC, 0",
        "label_GW_B0_FD0_VW1_GSU1_NonEdge:",
        "buffer_load_d16_u8 v40, v1, s[sgprSrdC:sgprSrdC+3], null offen offset:0 // load C",
        "v_cvt_f32_fp8 v41, v40 byte_sel:[1]",
        "v_cvt_pk_f32_fp8 v[42:43], v40.h",
        "v_cvt_f32_fp8 v44, v40 byte_sel:[0]",
        "v_fma_mix_f32 v[vgprValuC+0], s[sgprBeta], v40, v[vgprValuC+0] op_sel:[0,1,0] op_sel_hi:[0,1,0]",
        "v_cvt_f32_f16 v45, v[vgprValuC+4] // MI out element",
        "buffer_store_b16 v45, v2, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D",
        "v_cvt_pk_fp8_f32 v46, v41, v44 op_sel:[0,0,0]",
        "buffer_store_b16 v46, v2, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D",
        "v_mov_b32 v47, v46",
        "buffer_store_b16 v47, v2, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D",
        "buffer_store_b16 v48, v2, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D",
    ])
    # Byte 1, bytes 2-3 and the high half of a C register that holds byte 0 only.
    assert len(_c_reads_of_unloaded_bytes(text)) == 3
    assert len(_beta_reads_of_unloaded_c_halves(text)) == 1
    # The widened accumulator and the register nothing wrote; the conversion and its copy pass.
    assert len(_stores_not_last_written_by_a_conversion(text)) == 2
    assert len(_epilogue_stores_of_unwritten_registers(text)) == 1


def test_last_writer_check_follows_the_store_remap_round_trip():
    text = "\n".join([
        ".set vgprValuC, 0",
        "label_GW_B0_FD0_VW1_GSU1_NonEdge:",
        "v_cvt_pk_f16_f32 v10, v[vgprValuC+0], v[vgprValuC+1]",
        "v_cvt_pk_f16_f32 v11, v[vgprValuC+2], v[vgprValuC+3]",
        "ds_store_b64 v5, v[10:11] offset:0 // storeRemap lw",
        "ds_store_b32 v5, v[vgprValuC+4] offset:8 // storeRemap lw",
        "ds_load_b64 v[20:21], v6 offset:0 // storeRemap lr",
        "buffer_store_b64 v[20:21], v7, s[sgprSrdD:sgprSrdD+3], null offen offset:0 // store D StoreRemapVectorWidth",
    ])
    # The second local write stages an accumulator that was never converted; the read-back store passes.
    assert _stores_not_last_written_by_a_conversion(text) == [
        "ds_store_b32 v5, v[vgprValuC+4] offset:8 // storeRemap lw (last writers ['None'])"]


# The bias and scale vectors are staged in LDS as f32 (one dword per element) and sized as f32 in memory.
_VECTOR_LDS_SHIFT = re.compile(r"v_lshlrev_b32 \S+, (0x[0-9a-f]+), \S+\s+// (?:Bias|ScaleAlpha|ScaleAVec|ScaleBVec|Local) address scaled by BPE")
_VECTOR_SRD_BYTES = re.compile(r"s_mul_i32 s\[sgprSrd(?:ScaleAlphaVec|ScaleA|ScaleB)\+2\], (0x[0-9a-f]+),")


def test_vector_lds_stride_check_reports_a_half_stride():
    assert _VECTOR_LDS_SHIFT.findall("v_lshlrev_b32 v38, 0x1, v38                        // Bias address scaled by BPE") == ["0x1"]
    assert _VECTOR_SRD_BYTES.findall("s_mul_i32 s[sgprSrdScaleA+2], 0x2, s[sgprSrdScaleA+2]  // ScaleAVec scaled by BPE") == ["0x2"]


@pytest.mark.parametrize("config", ["f16acc_bias_act_gfx1250.yaml", "f16acc_scale_gfx1250.yaml"])
def test_epilogue_vectors_use_f32_strides_on_every_store_path(gfx1250_iim, tensile_args, tmp_path, config):
    for name, text in _generated_assembly(config, tensile_args, tmp_path).items():
        shifts = _VECTOR_LDS_SHIFT.findall(text)
        assert shifts and set(shifts) == {"0x2"}, (name, sorted(set(shifts)))
        assert set(_VECTOR_SRD_BYTES.findall(text)) <= {"0x4"}, name


@pytest.mark.parametrize("config", _F16_YAMLS)
def test_every_f16_yaml_stores_converted_values_and_reads_loaded_c(gfx1250_iim, tensile_args, tmp_path, config):
    for name, text in _generated_assembly(config, tensile_args, tmp_path).items():
        assert not _epilogue_stores_of_unwritten_registers(text), name
        assert not _beta_reads_of_unloaded_c_halves(text), name
        assert not _c_reads_of_unloaded_bytes(text), name
        assert not _stores_not_last_written_by_a_conversion(text), name


def test_eight_wide_fp8_pack_drops_x0x_only_for_f16_accumulation(_rocisa_gfx1250):
    from types import SimpleNamespace

    from tensilelite.Components.PackData import PackData_FLOAT8, PackData_FLOAT8_F16Accumulation

    struct = SimpleNamespace(vgprFp8NanInf=20, vgprFp8Temp=21, vgprFp8Min=22, vgprFp8Max=23)
    develop, f16acc = (str(cls()(gwvw=8, destIdx=0, elementSumIdx=0, fp8CVTVgprStruct=struct, tmpS01=30, laneSGPRC=2))
                       for cls in (PackData_FLOAT8, PackData_FLOAT8_F16Accumulation))
    # PackData_FLOAT8 turns +/-inf into NaN with x*0+x, one per pair; f16 accumulation saturates +/-inf with the clamp.
    assert len(_X0X.findall(develop)) == 4 and not _X0X.findall(f16acc)
    assert develop.count("v_maximumminimum_f32") == f16acc.count("v_maximumminimum_f32") == 8


def test_fp8_output_takes_both_saturation_paths_without_x0x(gfx1250_iim, tensile_args, tmp_path):
    paths = collections.Counter()
    for name, body in _generated_assembly("f8f8f8h_gfx1250.yaml", tensile_args, tmp_path).items():
        assert not _X0X.search(body), name
        paths["one at a time"] += bool(re.search(r"^v_med3_f32 ", body, re.M))
        paths["eight at a time"] += bool(re.search(r"^v_maximumminimum_f32 ", body, re.M))
    assert paths["one at a time"] and paths["eight at a time"], paths
