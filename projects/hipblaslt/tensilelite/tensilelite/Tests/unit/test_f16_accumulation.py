# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# f16 accumulation on gfx1250: HighPrecisionAccumulate=False with fp8 or fp4
# inputs and ComputeDataType=H runs the f16-accumulate fp8 WMMA, whose 16x16
# output tile holds two elements per VGPR; D is f16, fp8 or bf16. fp4 operands are converted to fp8
# after the LDS read. Covers the problem type, the solution rules on both
# gfx1250 steppings, the WMMA spelling, and the emitted assembly.
################################################################################

import collections
import copy
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tensilelite import Tensile
from tensilelite.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
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
def test_fp8_inputs_with_half_compute_accumulate_in_f16():
    problem = ProblemType(_problem_config(), False)
    assert isF16Accumulation(problem)
    assert "_F8HH_" in str(problem)


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


_TYPES = "supports only F8 (OCP e4m3) and F4 (e2m1) inputs, not BF8 or the fnuz types"


@pytest.mark.parametrize("types", [
    {"DataType": "B8"},
    {"DataType": "F8N"},
    {"DataType": "B8N"},
    {"DataTypeA": "F8", "DataTypeB": "B8"},
    {"DataTypeA": "B8", "DataTypeB": "F8"},
    {"DataTypeA": "F4", "DataTypeB": "B8"},
    {"DataTypeA": "F8", "DataTypeB": "F8N"},
    {"DataTypeA": "F8N", "DataTypeB": "F8"},
    {"DataTypeA": "F4", "DataTypeB": "F8N"},
    {"DataTypeA": "B8", "DataTypeB": "F4"},
    {"DataTypeA": "F8N", "DataTypeB": "F4"},
], ids=["b8", "f8n", "b8n", "f8b8", "b8f8", "f4b8", "f8f8n", "f8nf8", "f4f8n", "b8f4", "f8nf4"])
def test_bf8_and_fnuz_inputs_are_rejected_for_f16_accumulation(types):
    with pytest.raises(Exception, match=re.escape(_TYPES)):
        ProblemType(_problem_config(**types), False)


_FP4_DATA_TYPE = "takes DataType F8 with DataTypeA/DataTypeB F4, not DataType F4"


@pytest.mark.parametrize("types", [{}, {"DataTypeA": "F8"}, {"DataTypeB": "F8"}], ids=["f4f4", "f8f4", "f4f8"])
def test_fp4_data_type_with_fp8_mac_types_is_rejected_for_f16_accumulation(types):
    with pytest.raises(Exception, match=re.escape(_FP4_DATA_TYPE)):
        ProblemType(_problem_config(DataType="F4", MacDataTypeA="F8", MacDataTypeB="F8", **types), False)


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
def gfx1250_strict_iim(gfx1250_iim):
    iim = copy.deepcopy(gfx1250_iim)
    applyArchCapOverrides(iim, ["gfx1250-strict"])
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
    assert sol["InternalSupportParams"]["SupportUserGSU"] is False
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
def test_valid_on_gfx1250_strict(_gp_gfx1250, gfx1250_strict_iim, assembler, capsys, problem, overrides):
    sol, out = _derive(gfx1250_strict_iim, assembler, capsys, problem=problem, **overrides)
    assert sol.get("Valid") is True, out
    assert sol["MIRegPerOut"] == 0.5


_SOURCES = Path(__file__).parents[3]
_CLIENT_TYPES = {"Float8": "Float8", "Float4x2": "Float4", "Half": "Half", "BFloat16": "BFloat16"}


def _f16_accumulation_reference_cases():
    aliases = dict(re.findall(r"using (TypedGemm_\w+)\s*=\s*TypedGemm<([^>]*)>;",
                              (_SOURCES / "client" / "include" / "TypedId.hpp").read_text()))
    cases = re.findall(r"case (TypedGemm_\w+)::TypeId\(\):", (_SOURCES / "client" / "src" / "Reference.cpp").read_text())
    types = [tuple(t.strip() for t in aliases[name].split(",")) for name in cases if name in aliases]
    return {tuple(_CLIENT_TYPES[t] for t in ts) for ts in types
            if len(ts) == 8 and ts[4] == "Half" and all(t in _CLIENT_TYPES for t in ts)}


def _f16_accumulation_mac_inputs(source, function):
    text = (_SOURCES / source).read_text()
    start = text.index(function)
    return set(re.findall(r"rocisa::DataType::(\w+)", text[start:text.index("\n        }\n", start)])) - {"Half"}


@pytest.mark.parametrize("sourceSwap", [False, True])
@pytest.mark.parametrize("types", [{"DataTypeA": "F4"}, {"DataTypeB": "F4"}, {"DataTypeA": "F4", "DataTypeB": "F4"}],
                         ids=["f4f8", "f8f4", "f4f4"])
@pytest.mark.parametrize("dest", ["h", "F8", "B"])
def test_fp4_split_k_problem_is_f16_accumulation_on_the_host_and_in_the_reference(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, dest, types, sourceSwap):
    from tensilelite.Contractions import ProblemType as ContractionProblemType

    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=dict(types, DestDataType=dest), SourceSwap=sourceSwap,
                       ConvertAfterDS=True, GlobalSplitU=2, GlobalSplitUAlgorithm="MultipleBuffer")
    assert sol.get("Valid") is True, out
    problem = ContractionProblemType.FromOriginalState(sol["ProblemType"])
    macInputs = {problem.computeInputTypeA.toName(), problem.computeInputTypeB.toName()}
    # The reduction kernels are generated for an f16 workspace (Cijk_H...); the host builds the name of the one it
    # launches from the library's problem type, with H only when isF16AccumulationType accepts its compute inputs.
    names = conversionKernelNames(sol)
    assert names and all(name.startswith("Cijk_H") for name in names), names
    assert macInputs <= _f16_accumulation_mac_inputs("src/ContractionSolution.cpp", "bool isF16AccumulationType("), macInputs
    assert macInputs <= _f16_accumulation_mac_inputs("client/src/Reference.cpp", "bool isF16AccumulationProblem("), macInputs
    key = tuple(t.toName() for t in (problem.aType, problem.bType, problem.cType, problem.dType, problem.alphaType,
                                     problem.betaType, problem.computeInputTypeA, problem.computeInputTypeB))
    assert key in _f16_accumulation_reference_cases(), key


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
@pytest.mark.parametrize("gsu, coalesced", [(2, False), (3, False), (4, True)])
def test_split_k_multiple_buffer_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, gsu, coalesced, dest):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, GlobalSplitU=gsu,
                       GlobalSplitUAlgorithm="MultipleBuffer", GlobalSplitUCoalesced=coalesced)
    assert sol.get("Valid") is True, out
    assert sol["_GlobalAccumulation"] == "MultipleBuffer"
    assert sol["_WorkspaceSizePerElemC"] == 2


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


_PERSISTENT = "supports persistent kernels without DebugStreamK"
_GSU_MIN = "needs GlobalSplitU >= 1"
_ATOMICS = "splits K only with GlobalSplitUAlgorithm MultipleBuffer or MultipleBufferSingleKernel"


@pytest.mark.parametrize("strategy, assignment, tree", [
    ("DataParallel", "StaticGrid", 0), ("StreamK", "StaticGrid", 0), ("StreamK", "StaticGrid", 1),
    ("StreamK", "DynamicWorkQueue", 0), ("StreamK", "Hybrid", 1)])
def test_persistent_and_streamk_are_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys, strategy, assignment, tree):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TileProcessingStrategy=strategy, WorkAssignment=assignment,
                       StreamKFixupTreeReduction=tree)
    assert sol.get("Valid") is True, out
    # The tree fixup slots hold f32 sums.
    assert sol["_WorkspaceSizePerElemC"] == (0 if strategy == "DataParallel" else 4 if tree else 2)


@pytest.mark.parametrize("assignment", ["StaticGrid", "DynamicWorkQueue", "Hybrid"])
def test_space_filling_streamk_needs_a_static_grid(_gp_gfx1250, gfx1250_iim, assembler, capsys, assignment):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TileProcessingStrategy="StreamK", WorkAssignment=assignment,
                       SpaceFillingAlgo=[2], SFCWGM=[[2, 2]])
    assert sol.get("Valid") is (assignment == "StaticGrid"), out
    assert ("SpaceFillingAlgo only with StaticGrid" in out) is (assignment != "StaticGrid"), out


@pytest.mark.parametrize("overrides, reason", [
    ({"GlobalSplitU": 2, "GlobalSplitUAlgorithm": "SingleBuffer"}, _ATOMICS),
    ({"GlobalSplitU": -1}, _GSU_MIN),
    ({"TileProcessingStrategy": "StreamK", "StreamKAtomic": 1}, "Atomic Stream-K"),
], ids=["single_buffer", "gsu_auto", "stream_k_atomic"])
def test_split_k_paths_with_a_run_dependent_order_are_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                               overrides, reason):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is False
    assert reason in out


def test_wave64_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, WavefrontSize=64)
    assert sol.get("Valid") is False
    assert "f16 accumulation supports only wave32" in out


def test_no_f16_accumulate_wmma_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    iim = copy.deepcopy(gfx1250_iim)
    iim[ISA_GFX1250].asmCaps["HasWMMA_V3"] = False
    sol, out = _derive(iim, assembler, capsys)
    assert sol.get("Valid") is False
    assert "requires the gfx1250 f16-accumulate fp8 WMMA" in out


_LSU = "does not support LocalSplitU"
_STORE = "does not support StoreRemapVectorWidth, CompactLoopStore or UseSubtileImpl"
_EPILOGUE = "supports only alpha/beta in the epilogue"
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
    ({"GroupedGemm": True}, {}, "does not support grouped GEMM"),
    ({}, {"StoreRemapVectorWidth": 2}, _STORE),
    ({}, {"CompactLoopStore": True}, _STORE),
    ({}, {"UseSubtileImpl": True}, _STORE),
    ({"UseBias": 1}, {}, _EPILOGUE),
    ({"Activation": True, "ActivationType": "all"}, {}, _EPILOGUE),
    ({"Gradient": True}, {}, _EPILOGUE),
    ({"UseScaleAB": "Scalar"}, {}, _EPILOGUE),
    ({"UseScaleCD": True}, {}, _EPILOGUE),
    ({"UseScaleAlphaVec": 1}, {}, _EPILOGUE),
    ({"OutputAmaxD": True}, {}, _EPILOGUE),
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
], ids=["mi_16x16x32", "lsu2", "streamk", "grouped_gemm", "store_remap", "compact_loop_store", "subtile", "bias",
        "activation", "gradient", "scale_ab", "scale_cd", "scale_alpha_vec", "amax_d", "gate_residual", "sparse",
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
    ({"DataTypeA": "F4", "TransposeA": False}, {"LocalReadVectorWidth": 16}, "MN-contiguous F4 A with a LocalReadVectorWidth"),
    ({"DataTypeA": "F4", "TransposeA": False}, {"HalfPLR": 1}, "HalfPLR with MN-contiguous F4 A"),
    ({"DataTypeB": "F4", "TransposeB": True}, {"LDSTrInst": False}, _F4B_MN),
    ({"DataTypeB": "F4", "TransposeB": True}, {"DirectToVgprB": True}, _F4B_MN),
    ({"DataTypeB": "F4", "TransposeB": True}, {"HalfPLR": 2}, "HalfPLR with MN-contiguous F4 B"),
], ids=["f4_a_no_ldstr", "f4_a_direct_to_vgpr", "f4_a_lrvw16", "f4_a_half_plr", "f4_b_no_ldstr",
        "f4_b_direct_to_vgpr", "f4_b_half_plr"])
def test_each_mn_contiguous_f4_rule_rejects_on_its_own(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem,
                                                       overrides, reason):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem, mi=_MNF4_MI, **{**_MNF4, **overrides})
    assert sol.get("Valid") is False
    assert reason in out


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
    moves = [str(i).split("//")[0].split() for i in packedAccSingles(sol, mulAlpha=False).items()]
    alphas = [str(i).split("//")[0] for i in packedAccSingles(sol, mulAlpha=True).items()]
    assert len(moves) == len(alphas) == len(arch2acc)
    for d, (move, alpha) in enumerate(zip(moves, alphas)):
        s = arch2acc[d]
        src = "v[vgprValuC+%u]" % (s // 2)
        # Element s is in half s%2 of its MI output VGPR; the temp gets it in its low half.
        if s % 2:
            assert move[0] == "v_lshrrev_b32" and move[-2].rstrip(",") in ("16", "0x10") and move[-1] == src, move
        else:
            assert move[0] == "v_mov_b32" and move[-1] == src, move
        assert src in alpha and "s[sgprAlpha]" in alpha and "op_sel_hi:[0,1]" in alpha
        assert ("op_sel:[0,1]" in alpha) == bool(s % 2)
    assert sorted(arch2acc.values()) == list(range(len(arch2acc)))


@pytest.mark.parametrize("sourceSwap", [False, True])
def test_packed_templates_move_pairs_and_scale_them_by_alpha(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                            _rocisa_gfx1250, sourceSwap):
    from tensilelite.KernelWriterModules import accToArchMapper, mapAcctoArchRegs, moveMIoutToArch, mulMIoutAlphaToArch

    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=(16, 16, 64, 1, 1, 2, 2, 1, 1), SourceSwap=sourceSwap,
                       VectorWidthA=1 if sourceSwap else 2)
    assert sol.get("Valid") is True, out
    _, arch2acc = accToArchMapper(sol)
    pairs = [(arch2acc[d], arch2acc[d + 1]) for d in range(0, len(arch2acc), 2)]
    reads = [str(i).split("//")[0] for i in mapAcctoArchRegs(sol).items()]
    assert reads == [str(i).split("//")[0] for i in moveMIoutToArch(sol, 0).items()]
    alphas = [str(i) for i in mulMIoutAlphaToArch(sol, 0).items()]
    assert len(reads) == len(alphas) == len(pairs)
    contiguous = [s1 == s0 + 1 and s0 % 2 == 0 for s0, s1 in pairs]
    for (s0, s1), whole, read, alpha in zip(pairs, contiguous, reads, alphas):
        if whole:
            assert read.split()[0] == "v_mov_b32" and "v[vgprValuC+%u]" % (s0 // 2) in read
            assert "v_pk_mul_f16" in alpha and "v_pack_b32_f16" not in alpha
        else:
            assert read.split()[0] == "v_pack_b32_f16"
            assert "v[vgprValuC+%u]" % (s0 // 2) in read and "v[vgprValuC+%u]" % (s1 // 2) in read
            assert alpha.index("v_pack_b32_f16") < alpha.index("v_pk_mul_f16")


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
                                       (packedAccSingles(sol, mulAlpha=False), packedArchSinglesToAcc(sol), 1, 80)):
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


@pytest.mark.parametrize("gwvw", [1, 2, 3, 4])
def test_f16_to_fp8_pack_widens_clamps_and_converts_each_element(_rocisa_gfx1250, gwvw):
    from tensilelite.Components.PackData import PackData_F16toFLOAT8
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    vgprs = KernelWriterAssembly.FP8CVTVgprStruct(vgprFp8Temp=40, vgprFp8NanInf=41, vgprFp8Min=42, vgprFp8Max=43,
                                                  vgprFp8F32Pair=44)
    insts = _instructions(PackData_F16toFLOAT8()(gwvw, 0, 0, vgprs, 10, 1, inputPrefix="ValuC+"))
    for mnemonic in ("v_cvt_f32_f16", "v_cmp_class_f32", "v_med3_f32", "v_cndmask_b32"):
        assert sum(i.startswith(mnemonic) for i in insts) == gwvw, mnemonic
    converts = [i for i in insts if i.startswith("v_cvt_pk_fp8_f32")]
    assert len(converts) == (gwvw + 1) // 2
    # Element pairs 2,3 land in the high word of the destination VGPR.
    assert [("op_sel:[0,0,1]" in i) for i in converts] == [p % 2 == 1 for p in range(len(converts))]


@pytest.mark.parametrize("gwvw", [4, 8, 16])
def test_f16_to_fp8_pack_clamps_pairs_and_converts_eight_at_once(_rocisa_gfx1250, gwvw):
    from tensilelite.Components.PackData import PackData_F16toFLOAT8
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    vgprs = KernelWriterAssembly.FP8CVTVgprStruct(vgprFp8Temp=40, vgprFp8NanInf=41, vgprFp8Min=42, vgprFp8Max=43,
                                                  vgprFp8F32Pair=44, vgprFp8Pk8Src=48, vgprFp8ClampPk=52)
    insts = _instructions(PackData_F16toFLOAT8()(gwvw, 0, 0, vgprs, 10, 1, inputPrefix="ValuC+"))
    maxima = [i for i in insts if i.startswith("v_pk_maximum3_f16")]
    minima = [i for i in insts if i.startswith("v_pk_minimum3_f16")]
    converts = [i for i in insts if i.startswith("v_cvt_scalef32_pk8_fp8_f16")]
    assert len(maxima) == len(minima) == gwvw // 2
    assert len(converts) == (gwvw + 7) // 8
    assert all(re.search(r"\bv52, v52$", i) for i in maxima) and all(re.search(r"\bv53, v53$", i) for i in minima)
    assert all("v[48:51]" in i and i.endswith("1.0") for i in converts)
    assert len(insts) == len(maxima) + len(minima) + len(converts)


def _f16_value(h):
    """(sign bit, magnitude) of an f16 pattern; the magnitude is a Fraction, inf or NaN, so -0 and NaNs keep a sign."""
    from fractions import Fraction
    e, m = (h >> 10) & 31, h & 1023
    if e == 31:
        return h >> 15, float("nan") if m else float("inf")
    return h >> 15, Fraction(m, 1 << 24) if e == 0 else Fraction(1024 + m, 1 << 10) * Fraction(2) ** (e - 15)


def _isnan(x):
    return isinstance(x[1], float) and x[1] != x[1]


def _order(x):
    # Order of non-NaN signed values, -0 below +0.
    return (-x[1] if x[0] else x[1], 1 - x[0])


def _med3(x, lo, hi):
    return sorted((x, lo, hi), key=_order)[1]


def _cvt_pk_fp8_f32(x):
    """v_cvt_pk_fp8_f32 on a signed value with magnitude <= 448: OCP e4m3, nearest even; every NaN gives 0xff."""
    from fractions import Fraction
    import math
    if _isnan(x):
        return 0xff
    sign = 0x80 if x[0] else 0
    a = Fraction(x[1])
    if a == 0:
        return sign
    e = a.numerator.bit_length() - a.denominator.bit_length()
    e = max(e - 1 if Fraction(2) ** e > a else e, -6)
    q = a / Fraction(2) ** (e - 3)
    r = math.floor(q)
    r += 1 if q - r > Fraction(1, 2) or (q - r == Fraction(1, 2) and r % 2) else 0
    if r == 16:
        e, r = e + 1, 8
    return sign | (r if r < 8 else ((e + 7) << 3) | (r - 8))


def _pk_clamp_f16(h):
    """v_pk_maximum3_f16 with -448 twice, then v_pk_minimum3_f16 with 448 twice, on one f16 bit pattern: IEEE maximum
    and minimum return a NaN operand quieted, and -0 stays -0."""
    if h & 0x7c00 == 0x7c00 and h & 0x3ff:
        return h | 0x200
    if h & 0x7fff > 0x5f00:
        return 0xdf00 if h & 0x8000 else 0x5f00
    return h


def _cvt_scalef32_pk8_fp8_f16(h):
    """v_cvt_scalef32_pk8_fp8_f16 at scale 1.0 on one f16 bit pattern of magnitude <= 448, on the bits: the significand
    is rounded to e4m3's quantum (3 fraction bits, 2^-9 below 2^-6), nearest even; every NaN gives 0xff."""
    sign = (h >> 8) & 0x80
    e, m = (h >> 10) & 31, h & 0x3ff
    if e == 31:
        assert m, "an infinity reaches the converter only through the clamp, as 448"
        return 0xff
    sig, exp = (m, -24) if e == 0 else (0x400 | m, e - 25)
    if sig == 0:
        return sign
    q = max(sig.bit_length() - 1 + exp - 3, -9)
    shift = q - exp
    if shift <= 0:
        r = sig << -shift
    else:
        r, rem, halfway = sig >> shift, sig & ((1 << shift) - 1), 1 << (shift - 1)
        r += rem > halfway or (rem == halfway and r & 1)
    code = 8 * q + 72 + r
    assert code <= 0x7e, hex(h)
    return sign | code


def _dest_f8_bytes(h):
    lo, hi = _f16_value(0xdf00), _f16_value(0x5f00)
    x = _f16_value(h)
    old = _cvt_pk_fp8_f32(x if _isnan(x) else _med3(x, lo, hi))
    new = _cvt_scalef32_pk8_fp8_f16(_pk_clamp_f16(h))
    return old, new


def test_dest_f8_clamp_sequences_agree_on_every_f16():
    """The two Dest-f8 sequences modeled apart, on every f16 bit pattern: (1) on exact values, widen to f32,
    v_cmp_class_f32 keeps a NaN, otherwise v_med3_f32 with -448 and 448, then v_cvt_pk_fp8_f32; (2) on the f16 bits,
    v_pk_maximum3_f16 / v_pk_minimum3_f16, then v_cvt_scalef32_pk8_fp8_f16 at scale 1.0. The boundary bytes,
    NaNs included, are what both instructions gave on gfx1250 silicon."""
    for h in range(1 << 16):
        old, new = _dest_f8_bytes(h)
        assert old == new, hex(h)
    for h, byte in ((0x7c00, 0x7e), (0xfc00, 0xfe), (0x5f04, 0x7e), (0xdf04, 0xfe), (0x8000, 0x80), (0x1800, 0x01),
                    (0x7e00, 0xff), (0xfe00, 0xff), (0x7c01, 0xff), (0xfc01, 0xff)):
        assert _dest_f8_bytes(h) == (byte, byte), hex(h)


@pytest.mark.parametrize("gwvw", [1, 2, 4])
@pytest.mark.parametrize("withC", [False, True])
@pytest.mark.parametrize("applyAlpha", [False, True])
def test_f16_to_bf16_pack_scales_in_f32(_rocisa_gfx1250, gwvw, withC, applyAlpha):
    from tensilelite.Components.PackData import PackData_F16toBF16
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    vgprs = KernelWriterAssembly.BF16CVTVgprStruct(vgprBf16Temp=40, vgprBf16Mask=41, vgprFp32Nan=42, vgprBf16Inc=43,
                                                   vgprF32Pair=44, vgprAlphaBeta=46)
    insts = _instructions(PackData_F16toBF16()(gwvw, 0, 0, vgprs, 10, 1, dataC=60 if withC else None,
                                               applyAlpha=applyAlpha, inputPrefix="ValuC+"))
    assert sum(i.startswith("v_cvt_f32_f16") for i in insts) == gwvw
    assert sum(i.startswith("v_mul_f32") and re.search(r"\bv46\b", i) is not None for i in insts) \
        == (gwvw if applyAlpha else 0)
    assert sum(i.startswith(("v_fmac_f32", "v_mac_f32")) and re.search(r"\bv47\b", i) is not None for i in insts) \
        == (gwvw if withC else 0)


@pytest.mark.parametrize("dest, epilogue", [
    ("h", ["__builtin_fmaf16(", "res[e] = __builtin_bit_cast(uint16_t, r);"]),
    ("F8", ["__builtin_fmaf16(", "__builtin_amdgcn_fmed3f(f, -448.0f, 448.0f)", "__builtin_amdgcn_cvt_pk_fp8_f32"]),
    ("B", ["float r = (float)alpha * (float)acc;", "__builtin_fmaf((float)beta", "0x7fff"]),
])
def test_split_k_reduction_adds_f16_partials_in_f32(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest, epilogue):
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
        assert "sum[e] = (float)__builtin_bit_cast(tensile_half, (uint16_t)((e & 1) ? (w >> 16) : w));" in source
        assert "v_fma_mix_f32 %0, %1, 1.0, %2 op_sel_hi:[1,1,0]" in source
        assert "(arg.W + idW + (g0 + u) * strideW);" in source
        assert "tensile_half acc = (tensile_half)sum[e];" in source
        # alpha == 0 never multiplies a slice that overflowed to inf.
        assert "if(alpha != (tensile_half)0)" in source
        for text in epilogue:
            assert text in source, text


def test_fp8_output_is_f16_accumulation():
    problem = ProblemType(_problem_config(DestDataType="F8"), False)
    assert isF16Accumulation(problem)
    assert "_F8F8H_" in str(problem)


def test_bf16_output_is_f16_accumulation():
    problem = ProblemType(_problem_config(DestDataType="B"), False)
    assert isF16Accumulation(problem)
    assert "_F8BH_" in str(problem)


@pytest.mark.parametrize("dest", ["F8", "B"])
@pytest.mark.parametrize("problem, overrides", [
    ({}, {}),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True}),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True}),
    ({"DataTypeA": "F4", "DataTypeB": "F4"}, {"ConvertAfterDS": True}),
], ids=["f8f8", "f8f4", "f4f8", "f4f4"])
@pytest.mark.parametrize("strict", [False, True], ids=["gfx1250", "strict"])
def test_fp8_and_bf16_outputs_are_valid(_gp_gfx1250, gfx1250_iim, gfx1250_strict_iim, assembler, capsys, problem,
                                        overrides, strict, dest):
    sol, out = _derive(gfx1250_strict_iim if strict else gfx1250_iim, assembler, capsys,
                       problem={**problem, "DestDataType": dest}, **overrides)
    assert sol.get("Valid") is True, out
    assert sol["MIRegPerOut"] == 0.5


@pytest.mark.parametrize("problem, overrides", [
    ({}, {}),
    ({"DestDataType": "F8"}, {"GlobalSplitU": 2, "GlobalSplitUAlgorithm": "MultipleBuffer"}),
    ({"DestDataType": "B"}, {"StaggerU": 0}),
    ({"DataTypeA": "F4", "DataTypeB": "F4"}, {"ConvertAfterDS": True, "TransposeLDS": 1, "LocalReadVectorWidth": 16}),
], ids=["h", "f8_gsu2", "bf16_no_stagger", "f4f4"])
@pytest.mark.parametrize("strict", [False, True], ids=["gfx1250", "strict"])
def test_lds_segment_b_rotate_is_valid(_gp_gfx1250, gfx1250_iim, gfx1250_strict_iim, assembler, capsys, problem,
                                       overrides, strict):
    params = {"DepthU": 256, "HalfPLR": 0, "TDMSplit": False, "LDSSegmentInterleave": 1, "LDSSegmentBRotate": 1,
              "TransposeLDS": -1, "LocalReadVectorWidth": -1, "VectorWidthA": -1, "SourceSwap": True,
              "ExpandPointerSwap": False, "AssertSummationElementMultiple": 128, **overrides}
    sol, out = _derive(gfx1250_strict_iim if strict else gfx1250_iim, assembler, capsys, problem=problem,
                       mi=(16, 16, 128, 1, 1, 4, 4, 2, 2), **params)
    assert sol.get("Valid") is True, out
    assert sol["LDSSegmentBRotate"] == 1 and sol["MIRegPerOut"] == 0.5


@pytest.mark.parametrize("problem", [{"DataTypeB": "F4", "TransposeB": True},
                                     {"DataTypeA": "F4", "DataTypeB": "F4", "TransposeB": True}], ids=["f8f4", "f4f4"])
@pytest.mark.parametrize("brotate", [0, 1])
@pytest.mark.parametrize("strict", [False, True], ids=["gfx1250", "strict"])
def test_lds_segment_interleave_with_mn_major_fp4_b_is_valid(_gp_gfx1250, gfx1250_iim, gfx1250_strict_iim, assembler,
                                                             capsys, problem, brotate, strict):
    params = {"DepthU": 256, "HalfPLR": 0, "TDMSplit": False, "LDSSegmentInterleave": 1, "LDSSegmentBRotate": brotate,
              "TransposeLDS": -1, "LocalReadVectorWidth": -1, "VectorWidthA": -1, "VectorWidthB": 1, "SourceSwap": True,
              "ExpandPointerSwap": False, "AssertSummationElementMultiple": 128, "AssertFree1ElementMultiple": 2,
              "ConvertAfterDS": True, "LDSTrInst": True}
    sol, out = _derive(gfx1250_strict_iim if strict else gfx1250_iim, assembler, capsys, problem=problem,
                       mi=(16, 16, 128, 1, 1, 4, 4, 2, 2), **params)
    assert sol.get("Valid") is True, out
    assert sol["LDSSegmentBRotate"] == brotate and sol["LDSSegInterleaveOffsets"]["bMnMajor"]


def _mn_major_b_state(**overrides):
    state = {"LDSSegmentInterleave": 1, "UnrollMajorLDSB": False, "enableLDSTrB": True, "VectorWidthB": 1,
             "ProblemType": ProblemType(_problem_config(DataTypeB="F4", TransposeB=True), False)}
    state.update(overrides)
    return state


@pytest.mark.parametrize("overrides", [
    {"LDSSegmentInterleave": -1}, {"UnrollMajorLDSB": True}, {"enableLDSTrB": False}, {"VectorWidthB": 2},
    {"ProblemType": ProblemType(_problem_config(TransposeB=True), False)},
    {"ProblemType": ProblemType(_problem_config(DataTypeB="F4", TransposeB=True, HighPrecisionAccumulate=True), False)},
], ids=["auto", "unroll_major", "no_transpose_read", "vector_width_2", "fp8_b", "f32_accumulation"])
def test_mn_major_b_interleave_needs_a_request_and_an_f16_accumulated_fp4_b(overrides):
    assert segMnMajorBRequested(_mn_major_b_state())
    assert not segMnMajorBRequested(_mn_major_b_state(**overrides))


def test_fp8_output_stochastic_rounding_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": "F8", "StochasticRounding": True})
    assert sol.get("Valid") is False
    assert "supports only alpha/beta in the epilogue" in out


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


# =========================================================================== #
# rocisa spelling of the f16-accumulate WMMA.
# =========================================================================== #
@pytest.fixture
def _rocisa_gfx1250(gfx1250_iim):
    from rocisa import rocIsa

    ri = rocIsa.getInstance()
    ri.init(ISA_GFX1250, shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
    ri.setKernel(ISA_GFX1250, 32)


@pytest.mark.parametrize("k, accType, mnemonic", [
    (64, "INST_F16", "v_wmma_f16_16x16x64_fp8_fp8"),
    (128, "INST_F16", "v_wmma_f16_16x16x128_fp8_fp8"),
    (128, "INST_F32", "v_wmma_f32_16x16x128_fp8_fp8"),
])
def test_fp8_wmma_spelling(_rocisa_gfx1250, k, accType, mnemonic):
    from rocisa.container import vgpr
    from rocisa.enum import InstType
    from rocisa.instruction import MFMAInstruction

    accRegs = 4 if accType == "INST_F16" else 8
    numRegs = k // 8
    inst = MFMAInstruction(instType=InstType.INST_F8, accType=getattr(InstType, accType), variant=[16, 16, k, 1],
                           mfma1k=False, acc=vgpr(0, accRegs), a=vgpr(16, numRegs), b=vgpr(32, numRegs),
                           acc2=vgpr(0, accRegs))
    assert str(inst).split()[0] == mnemonic


# =========================================================================== #
# Emitted assembly (--build-only; no GPU needed).
# =========================================================================== #
def _assembly_files(config, tensile_args, tmp_path, target="gfx1250"):
    output_dir = tmp_path / "output"
    Tensile.Tensile([str(_GFX12_CONFIGS / config), str(output_dir), "--build-only",
                     "--global-parameters", "KeepBuildTmp=True", *tensile_args, "--gpu-targets", target])
    files = {p.name: p.read_text() for p in output_dir.rglob("*.s")}
    assert files, "no assembly generated"
    return files


def _build_assembly(config, tensile_args, tmp_path, target="gfx1250"):
    return "\n".join(_assembly_files(config, tensile_args, tmp_path, target).values())


_REG = r"(v\[vgprValuC(?:\+\d+)?\]|v\d+)"


def _width_one_trace(text, store):
    # Follows every width-1 D store (`store` is its mnemonic) back to the instruction that filled its
    # register from an MI output VGPR, and every single C load to the beta FMA that consumes it. A store
    # from a register nothing filled reads an even element in place (its temp is its MI output VGPR).
    base = re.search(r"^\.set vgprValuC, (\d+)", text, re.M)
    base = int(base.group(1)) if base else 0

    def reg(token):
        m = re.fullmatch(r"v\[vgprValuC(?:\+(\d+))?\]", token)
        return base + int(m.group(1) or 0) if m else int(token[1:])

    source, stores, pendingC = {}, collections.Counter(), set()
    count = collections.Counter()
    for line in text.splitlines():
        if re.match(r"label_\w+:", line):
            # Each store variant starts from the MI outputs again.
            source.clear()
            continue
        fill = re.match(r"(v_mov_b32|v_lshrrev_b32|v_pk_mul_f16) %s, (?:16, |0x10, |s\[sgprAlpha\], )?"
                        r"v\[vgprValuC\+(\d+)\](.*)" % _REG, line)
        if fill and not fill.group(4).lstrip().startswith(","):
            acc, alpha = base + int(fill.group(3)), fill.group(1) == "v_pk_mul_f16"
            count["doubleAlpha"] += alpha and source.get(acc, (0, 0, False))[2]
            half = int(fill.group(1) == "v_lshrrev_b32" or "op_sel:[0,1]" in fill.group(4))
            source[reg(fill.group(2))] = (acc, half, alpha)
            count["fills"] += 1
            continue
        m = re.match(r"buffer_load_d16_b16 %s, .*// load C" % _REG, line)
        if m:
            assert reg(m.group(1)) not in pendingC, "two C values share a VGPR: " + line
            pendingC.add(reg(m.group(1)))
        m = re.match(r"v_pk_fma_f16 %s, %s, s\[sgprBeta\]" % (_REG, _REG), line) \
            or re.match(r"v_lshlrev_b32 %s, (?:16|0x10), %s" % (_REG, _REG), line) \
            or re.match(r"v_cvt_f32_bf16 %s, %s" % (_REG, _REG), line)
        if m:
            pendingC.discard(reg(m.groups()[-1]))
        m = re.match(r"(buffer_store_\w+) %s, .*// store D" % _REG, line)
        if m and m.group(1) == store:
            r = reg(m.group(2))
            count["direct"] += r not in source
            acc, half, _ = source.pop(r, (r, 0, False))
            stores[(acc, half)] += 1
        elif m:
            source.clear()
    assert not pendingC
    return stores, count


@pytest.mark.parametrize("config, store", [
    ("f8f8hh_gfx1250.yaml", "buffer_store_b16"),
    ("f8f4hh_gfx1250.yaml", "buffer_store_b16"),
    ("f8f8f8h_gfx1250.yaml", "buffer_store_b8"),
    ("f8f8bh_gfx1250.yaml", "buffer_store_b16"),
])
def test_width_one_stores_each_element_from_a_low_half(gfx1250_iim, tensile_args, tmp_path, config, store):
    files = _assembly_files(config, tensile_args, tmp_path)
    for name, text in files.items():
        # No D store or C load ever addresses a high half.
        assert not re.search(r"^(buffer_store_d16_hi\S* .*// store D|buffer_load_d16_hi\S* .*// load C)", text, re.M), name
    widthOne = {name: text for name, text in files.items() if re.search(r"^%s .*// store D" % store, text, re.M)}
    assert widthOne, "no kernel stores a single element"
    for name, text in widthOne.items():
        stores, count = _width_one_trace(text, store)
        vgprs = {v for v, _ in stores}
        # Each MI output VGPR is stored from its low half and from its high half equally often (an even
        # element may be stored straight from its MI output VGPR), and alpha is applied to each element once.
        assert vgprs and all(stores[(v, 0)] == stores[(v, 1)] > 0 for v in vgprs), (name, stores)
        assert count["doubleAlpha"] == 0, name


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
        store = re.match(r"buffer_store_b(\d+) .*// store T?D\b", line)
        if store:
            elements = int(store.group(1)) // 8 // bytesPerElement
            sections[key]["elements"] += elements
            sections[key]["singles"] += elements == 1
        sections[key]["alpha"] += bool(re.match(r"v_pk_mul_f16 \S+, s\[sgprAlpha\]", line))
        sections[key]["arithmetic"] += bool(re.match(r"v_(?:pk_)?(?:mul|fma)_f(?:16|32) ", line))
    return sections


@pytest.mark.parametrize("config", ["f16acc_gsu_edge_beta0_gfx1250.yaml", "f16acc_overflow_gfx1250.yaml"])
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
            elif dest != "B":
                # One packed multiply per pair of elements, or per element when each is stored alone.
                assert c["alpha"] * (1 if c["singles"] == c["elements"] else 2) == c["elements"], (name, label, c)
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


def test_streamk_partials_are_added_in_f32_and_rounded_once(gfx1250_iim, tensile_args, tmp_path):
    # Linear slots hold raw f16 partials, added with v_fma_mix_f32; tree slots hold f32 sums, added with v_add_f32.
    # Either way the sums are rounded to f16 once and the final store scales them like a GSU1 store. The
    # space-filling walk's s_cselect_b64 and s_flbit_i32_b32 go through StinkyTofu on gfx1250 as well; only the
    # walk emits s_flbit_i32_b32.
    files = _assembly_files("f16acc_streamk_gfx1250.yaml", tensile_args, tmp_path)
    seen = collections.Counter()
    walk = set()
    for name, text in files.items():
        if re.search(r"^\s*s_flbit_i32_b32 ", text, re.M):
            seen["space filling"] += 1
            walk.update(re.findall(r"^\s*(s_cselect_b64|s_flbit_i32_b32) ", text, re.M))
        loops = _streamk_f16_partial_loops(text)
        if not loops:
            assert "store f16 partial" not in text and "store f32 sums" not in text, name
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
        assert "round the sum to f16" in text, name
        assert ("store f32 sums" in text) == ("tree" in kinds) and ("store f16 partial" in text) == ("linear" in kinds), name
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
            if dest == "B":
                assert c["alpha"] == 0, (name, label, c)
            else:
                assert c["alpha"] * (1 if c["singles"] == c["elements"] else 2) == c["elements"], (name, label, c)
                seen["MBSK singles"] += c["singles"] > 0
    assert seen["MBSK"] and seen["MBSK singles"], seen


def test_fp8_output_saturates_and_converts_packed_f16(gfx1250_iim, tensile_args, tmp_path):
    files = _assembly_files("f8f8f8h_gfx1250.yaml", tensile_args, tmp_path)
    text = "\n".join(files.values())
    assert "v_wmma_f16_16x16x64_fp8_fp8" in text
    assert "v_wmma_f16_16x16x128_fp8_fp8" in text
    assert "v_wmma_f32_" not in text
    # alpha in f16, beta in f16 on C widened from fp8 (exact), then f16 -> f32 -> saturated fp8.
    assert "v_pk_mul_f16" in text
    assert re.search(r"^v_cvt_pk_f32_fp8 ", text, re.M) and re.search(r"^v_cvt_pk_f16_f32 ", text, re.M)
    assert re.search(r"^v_pk_fma_f16 .*s\[sgprBeta\].*op_sel_hi:\[1,0,1\]", text, re.M)
    assert re.search(r"^v_cvt_f32_f16 \S+, \S+\.h", text, re.M)
    assert re.search(r"^v_cvt_pk_fp8_f32 ", text, re.M)
    # The class mask is NaN only (0x3): inf from an f16 overflow is clamped to +/-448 like a finite overflow.
    assert re.search(r"^v_mov_b32 \S+, 0x3 +// Nan$", text, re.M) and "0x207" not in text
    assert re.search(r"0x43[eE]00000", text) and re.search(r"0xc3[eE]00000", text)
    # Store widths of 4 and more clamp f16 pairs (NaN kept) and convert eight values at once.
    assert re.search(r"^v_cvt_scalef32_pk8_fp8_f16 \S+, \S+, 1\.0", text, re.M)
    assert re.search(r"0xdf00df00", text) and re.search(r"0x5f005f00", text)
    assert re.search(r"^buffer_store_b8 .*// store D", text, re.M)
    assert re.search(r"^buffer_store_b(32|64) .*// store D", text, re.M)
    for name, body in files.items():
        widened = len(re.findall(r"^v_cvt_f32_f16 ", body, re.M))
        # Every widened value is clamped to [-448, 448] once, unless it is NaN, before v_cvt_pk_fp8_f32.
        assert widened == len(re.findall(r"^v_med3_\w*f32 ", body, re.M)) \
            == len(re.findall(r"^v_cmp_class_f32\w* ", body, re.M)), name
        assert len(re.findall(r"^v_pk_maximum3_f16 ", body, re.M)) == len(re.findall(r"^v_pk_minimum3_f16 ", body, re.M)), name
        assert widened or re.search(r"^v_cvt_scalef32_pk8_fp8_f16 ", body, re.M), name


def test_fp8_kernels_accumulate_and_store_packed_f16(gfx1250_iim, tensile_args, tmp_path):
    text = _build_assembly("f8f8hh_gfx1250.yaml", tensile_args, tmp_path)
    assert "v_wmma_f16_16x16x64_fp8_fp8" in text
    assert "v_wmma_f16_16x16x128_fp8_fp8" in text
    assert "v_wmma_f32_" not in text
    # alpha and beta*C on packed f16 pairs, and the SourceSwap pair gather.
    assert "v_pk_mul_f16" in text
    assert "v_pk_fma_f16" in text
    assert "v_pack_b32_f16" in text
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
    assert re.search(r"^v_cvt_f32_f16 \S+, \S+\.h +// f16 to f32", text, re.M)
    assert re.search(r"^v_cvt_f32_f16 .*// alpha to f32", text, re.M) and re.search(r"^v_cvt_f32_f16 .*// beta to f32", text, re.M)
    assert re.search(r"^buffer_store_b16 .*// store D", text, re.M)
    assert re.search(r"^buffer_store_b(32|64|128) .*// store D", text, re.M)
    for name, body in files.items():
        widened = len(re.findall(r"^v_cvt_f32_f16 .*// f16 to f32", body, re.M))
        alphas = len(re.findall(r"^v_mul_f32 .*// \*= alpha", body, re.M))
        betas = len(re.findall(r"^v_fmac_f32 .*// \+= C\*beta", body, re.M))
        packed = len(re.findall(r"^v_cvt_pk_bf16_f32 ", body, re.M))
        rounded = len(re.findall(r"^v_add3_u32 .*increment for rounding", body, re.M))
        assert widened and alphas <= widened and betas <= widened, name
        # One conversion per pair (or per single element), or the integer rounding per value.
        assert widened / 2 <= packed <= widened or rounded == widened, (name, widened, packed, rounded)


def test_fp8_output_clamp_keeps_nan_and_saturates_inf(_rocisa_gfx1250):
    import math
    import struct

    from tensilelite.Components.PackData import PackData_F16toFLOAT8

    Cvt = collections.namedtuple("Cvt", "vgprFp8NanInf vgprFp8Temp vgprFp8Min vgprFp8Max vgprFp8F32Pair")
    pairs = [(math.inf, -math.inf), (math.nan, 1.5), (65504.0, -65504.0), (464.0, 0.5)]
    # The kernel loads 0x3 (NaN classes) and +/-448 into these registers; v10..v13 hold the packed f16 pairs.
    regs = {"v20": 0x3, "v22": -448.0, "v23": 448.0}
    for k, pair in enumerate(pairs):
        regs["v%u" % (10 + k)] = struct.unpack("<I", struct.pack("<ee", *pair))[0]

    def classBits(x):
        if math.isnan(x):
            return 0x2
        if math.isinf(x):
            return 0x200 if x > 0 else 0x4
        return 0x100 if x > 0 else 0x40

    converted = []
    module = PackData_F16toFLOAT8()(8, 20, 20, Cvt(20, 21, 22, 23, 24), 4, 1)
    for inst in module.flatitems():
        op, args = str(inst).split("//")[0].split(None, 1)
        dst, *src = [t.strip().split()[0] for t in args.split(",")]
        if op == "v_cvt_f32_f16":
            word = regs[src[0].split(".")[0]] >> (16 if src[0].endswith(".h") else 0)
            regs[dst] = struct.unpack("<e", struct.pack("<H", word & 0xffff))[0]
        elif op == "v_cmp_class_f32":
            regs[dst] = bool(classBits(regs[src[0]]) & regs[src[1]])
        elif op == "v_med3_f32":
            regs[dst] = max(min(regs[src[0]], regs[src[2]]), regs[src[1]])
        elif op == "v_cndmask_b32":
            regs[dst] = regs[src[1]] if regs[src[2]] else regs[src[0]]
        else:
            assert op == "v_cvt_pk_fp8_f32", op
            converted.append((regs[src[0]], regs[src[1]]))
    expected = [(448.0, -448.0), (math.nan, 1.5), (448.0, -448.0), (448.0, 0.5)]
    assert len(converted) == len(expected)
    for got, want in zip(converted, expected):
        for g, w in zip(got, want):
            assert (math.isnan(g) and math.isnan(w)) or g == w, (converted, expected)


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


@pytest.fixture(scope="module")
def _strict_assembles_f16_wmma(gfx1250_cxx, tmp_path_factory):
    probe = tmp_path_factory.mktemp("strict_probe") / "wmma.s"
    probe.write_text("v_wmma_f16_16x16x64_fp8_fp8 v[0:3], v[8:15], v[16:23], v[0:3]\n")
    result = subprocess.run([gfx1250_cxx, "-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1250-strict",
                             "-c", "-o", os.devnull, str(probe)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("this assembler rejects the f16-accumulate fp8 WMMA for gfx1250-strict (AMD LLVM 24 accepts it)")


@pytest.mark.parametrize("config", ["f8f8hh_gfx1250-strict.yaml", "f8f4hh_gfx1250-strict.yaml",
                                    "f4f8hh_gfx1250-strict.yaml", "f4f4hh_gfx1250-strict.yaml",
                                    "f8f8f8h_gfx1250-strict.yaml", "f8f4f8h_gfx1250-strict.yaml",
                                    "f8f8bh_gfx1250-strict.yaml", "f8f4bh_gfx1250-strict.yaml",
                                    "f4f8f8h_gfx1250-strict.yaml", "f4f4f8h_gfx1250-strict.yaml",
                                    "f4f8bh_gfx1250-strict.yaml", "f4f4bh_gfx1250-strict.yaml",
                                    "f16acc_overflow_gfx1250-strict.yaml", "f16acc_largek_gfx1250-strict.yaml"])
def test_gfx1250_strict_kernels_decode_fp4_without_converts(_strict_assembles_f16_wmma, tensile_args, tmp_path, config):
    text = _build_assembly(config, tensile_args, tmp_path, "gfx1250-strict")
    assert re.search(r"v_wmma_f16_16x16x(64|128)_fp8_fp8", text)
    assert "v_cvt_scale_pk8" not in text
    assert "v_cvt_scalef32_pk8_fp8_f16" not in text and "v_pk_maximum3_f16" not in text
    assert ("0x4c484440" in text) == config.startswith(("f8f4", "f4"))
    assert _converted_operands(text, "v_perm_b32") == _fp4_operands(config)


# =========================================================================== #
# The VALU fp4 -> fp8 decode, run on a small model of the instructions it uses.
# =========================================================================== #
def _e2m1ToE4m3(nibble):
    return (0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c)[nibble & 7] | (0x80 if nibble & 8 else 0)


def _runDecode(insts, regs):
    def value(token):
        return regs[token] if token.startswith("v[") else int(token, 0)

    def perm(hi, lo, sel):
        data = lo.to_bytes(4, "little") + hi.to_bytes(4, "little")
        out = 0
        for k in range(4):
            s = (sel >> 8 * k) & 0xff
            assert s < 8, "the decode only selects bytes"
            out |= data[s] << 8 * k
        return out

    for inst in insts:
        op, args = str(inst).split("//")[0].split(None, 1)
        dst, *src = [t.strip() for t in args.split(",")]
        s = [value(t) for t in src]
        regs[dst] = {
            "v_mov_b32": lambda: s[0],
            "v_and_b32": lambda: s[0] & s[1],
            "v_lshlrev_b32": lambda: (s[1] << s[0]) & 0xffffffff,
            "v_lshrrev_b32": lambda: s[1] >> s[0],
            "v_and_or_b32": lambda: (s[0] & s[1]) | s[2],
            "v_or_b32": lambda: s[0] | s[1],
            "v_perm_b32": lambda: perm(*s),
        }[op]()


@pytest.mark.parametrize("inPlace", [False, True], ids=["separate", "in_place"])
def test_fp4_to_fp8_decode_matches_e2m1(_rocisa_gfx1250, inPlace):
    import random

    from rocisa.container import vgpr

    from tensilelite.Components.LocalRead import fp4ToFp8Decode

    rng = random.Random(1250)
    words = [b * 0x01010101 for b in range(256)] + [rng.getrandbits(32) for _ in range(512)]
    src, dst0, dst1 = vgpr("F4Src"), vgpr("F4Src" if inPlace else "F8Lo"), vgpr("F8Hi")
    insts = fp4ToFp8Decode(src, dst0, dst1, vgpr("T0"), vgpr("T1"), vgpr("Sel"), vgpr("Table"))
    for word in words:
        regs = {str(src): word}
        _runDecode(insts, regs)
        expected = [_e2m1ToE4m3(word >> 4 * e) for e in range(8)]
        got = list(regs[str(dst0)].to_bytes(4, "little") + regs[str(dst1)].to_bytes(4, "little"))
        assert got == expected, hex(word)
