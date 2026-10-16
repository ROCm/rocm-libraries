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

from Tensile import Tensile
from Tensile.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.Types import IsaVersion
from Tensile.KernelHelperNaming import conversionKernelNames
from Tensile.SolutionStructs.Problem import ProblemType, isF16Accumulation
from Tensile.SolutionStructs.Solution import Solution

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


# =========================================================================== #
# Solution rules on real gfx1250 capabilities.
# =========================================================================== #
@pytest.fixture(scope="module")
def gfx1250_cxx():
    from Tensile.Toolchain.Validators import validateToolchain

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
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(gfx1250_cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
    from Tensile.Common.ValidParameters import validParameters

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
    from Tensile.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

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
    assert conversionKernelNames(sol) == []


def test_stagger_u_is_off(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, out
    # One f16 rounding per 64-deep step: with 2048 accumulated, a block sum of 1 rounds back to 2048, so a workgroup
    # whose K loop starts later in K gives a different D.
    assert sol["StaggerU"] == 0
    assert sol["InternalSupportParams"]["SupportCustomStaggerU"] is False


@pytest.mark.parametrize("problem, overrides", [
    ({}, {}),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True}),
], ids=["f8f8", "f8f4"])
def test_valid_on_gfx1250_strict(_gp_gfx1250, gfx1250_strict_iim, assembler, capsys, problem, overrides):
    sol, out = _derive(gfx1250_strict_iim, assembler, capsys, problem=problem, **overrides)
    assert sol.get("Valid") is True, out
    assert sol["MIRegPerOut"] == 0.5


@pytest.mark.parametrize("overrides", [
    {"GlobalSplitU": 2},
    {"AdaptiveGemmGSUA": 1},
    {"GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"},
], ids=["gsu2", "adaptive_gsua", "mbsk"])
def test_gsu_merge_paths_are_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys, overrides):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is False
    assert "f16 accumulation does not support GlobalSplitU" in out


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


_SPLIT = "does not support GlobalSplitU, AdaptiveGemmGSUA, MultipleBufferSingleKernel, LocalSplitU or StreamK"
_STORE = "does not support StoreRemapVectorWidth, CompactLoopStore or UseSubtileImpl"
_EPILOGUE = "supports only alpha/beta in the epilogue"
_F4A = "reads F4 A K-contiguous from unroll-major LDS only"
_F4B = "reads F4 B K-contiguous from unroll-major LDS only"
_CAD = "converts F4 to F8 after the LDS read and needs ConvertAfterDS"


# Each case changes one condition of a valid configuration.
@pytest.mark.parametrize("problem, overrides, reason", [
    ({}, {"mi": (16, 16, 32, 1, 1, 1, 1, 1, 1)}, "supports only the 16x16x64 and 16x16x128 WMMA"),
    ({}, {"WorkGroup": [16, 16, 2]}, _SPLIT),
    ({}, {"StreamK": 3}, _SPLIT),
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
    ({"DataTypeA": "F4", "TransposeA": False}, {"ConvertAfterDS": True}, _F4A),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True, "TransposeLDS": 0}, _F4A),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True, "DirectToVgprA": True}, _F4A),
    ({"DataTypeA": "F4"}, {"ConvertAfterDS": True, "DirectToLds": 1}, _F4A),
    ({"DataTypeB": "F4", "TransposeB": True}, {"ConvertAfterDS": True}, _F4B),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True, "TransposeLDS": 0}, _F4B),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True, "DirectToVgprB": True}, _F4B),
    ({"DataTypeB": "F4"}, {"ConvertAfterDS": True, "DirectToLds": 1}, _F4B),
    ({}, {"mi": (16, 16, 64, 1, 1, 3, 1, 1, 1), "SourceSwap": True, "VectorWidthA": 3},
     "need a power-of-two StoreVectorWidth"),
], ids=["mi_16x16x32", "lsu2", "streamk", "store_remap", "compact_loop_store", "subtile", "bias", "activation",
        "gradient", "scale_ab", "scale_cd", "scale_alpha_vec", "amax_d", "gate_residual", "sparse",
        "mx_block_a", "mx_block_b", "convert_after_ds_without_f4", "f4_a_without_convert_after_ds",
        "f4_b_without_convert_after_ds", "f4_a_tlu", "f4_a_not_unroll_major", "f4_a_direct_to_vgpr",
        "f4_a_direct_to_lds", "f4_b_tlu", "f4_b_not_unroll_major", "f4_b_direct_to_vgpr", "f4_b_direct_to_lds",
        "svw3"])
def test_each_rule_rejects_on_its_own(_gp_gfx1250, gfx1250_iim, assembler, capsys, problem, overrides, reason):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem=problem, **overrides)
    assert sol.get("Valid") is False
    assert reason in out


@pytest.mark.parametrize("dest", ["h", "F8", "B"])
def test_store_width_must_be_a_power_of_two(_gp_gfx1250, gfx1250_iim, assembler, capsys, dest):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problem={"DestDataType": dest}, mi=(16, 16, 64, 1, 1, 6, 1, 1, 1),
                       SourceSwap=True, VectorWidthA=6)
    assert sol.get("Valid") is False
    assert "need a power-of-two StoreVectorWidth" in out


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
    from Tensile.Components.NotLocalFullTileElements import NotLocalFullTileElementsMFMA

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
    from Tensile.KernelWriterModules import accToArchMapper, packedAccSingles

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
    (128, "INST_F32", "v_wmma_f32_16x16x128_f8f6f4"),
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
    assert re.search(r"^v_cvt_pk_fp8_f32 .*op_sel:\[0,0,1\]", text, re.M)
    # The class mask is NaN only (0x3): inf from an f16 overflow is clamped to +/-448 like a finite overflow.
    assert re.search(r"^v_mov_b32 \S+, 0x3 +// Nan$", text, re.M) and "0x207" not in text
    assert re.search(r"0x43[eE]00000", text) and re.search(r"0xc3[eE]00000", text)
    assert "v_cvt_scalef32_pk8_fp8_f16" not in text
    assert re.search(r"^buffer_store_b8 .*// store D", text, re.M)
    assert re.search(r"^buffer_store_b(32|64) .*// store D", text, re.M)
    for name, body in files.items():
        widened = len(re.findall(r"^v_cvt_f32_f16 ", body, re.M))
        # Every widened value is clamped to [-448, 448] once, unless it is NaN, before v_cvt_pk_fp8_f32.
        assert widened and widened == len(re.findall(r"^v_med3_\w*f32 ", body, re.M)) \
            == len(re.findall(r"^v_cmp_class_f32\w* ", body, re.M)), name


def test_fp8_kernels_accumulate_and_store_packed_f16(gfx1250_iim, tensile_args, tmp_path):
    text = _build_assembly("f8f8hh_gfx1250.yaml", tensile_args, tmp_path)
    assert "v_wmma_f16_16x16x64_fp8_fp8" in text
    assert "v_wmma_f16_16x16x128_fp8_fp8" in text
    assert "v_wmma_f32_" not in text
    # alpha and beta*C on packed f16 pairs, and the SourceSwap pair gather.
    assert "v_pk_mul_f16" in text
    assert "v_pk_fma_f16" in text
    assert "v_pack_b32_f16" in text


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

    from Tensile.Components.PackData import PackData_F16toFLOAT8

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

    from Tensile.Components.LocalRead import fp4ToFp8Decode

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
