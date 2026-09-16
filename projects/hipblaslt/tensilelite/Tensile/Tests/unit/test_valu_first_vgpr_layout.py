# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import rocisa

from Tensile.BenchmarkProblems import _generate_single_solution
from Tensile.BenchmarkStructs import (
    BenchmarkProcess,
    constructForkPermutations,
)
from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.DataType import DataType
from Tensile.Common.GlobalParameters import (
    assignGlobalParameters,
    globalParameters,
)
from Tensile.Common.Types import DebugConfig, makeDebugConfig
from Tensile.Common.ValidParameters import validParameters
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.SolutionStructs.Naming import getKernelFileBase
from Tensile.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

pytestmark = pytest.mark.unit


def _writer(*, num_vgpr_buffer=1, inc_lds_buf_switch=False):
    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        IncLdsBufSwitch=inc_lds_buf_switch,
        archCaps={"VgprBank": True},
        asmCaps={"HasWMMA_V1": True, "HasWMMA_V3": False},
        c=SimpleNamespace(startVgprValu=0),
        numItersPLR=0,
        numVgprBuffer=num_vgpr_buffer,
    )
    return writer


def _kernel(mac_data_type="H", **overrides):
    kernel = {
        "BufferLoad": True,
        "ConvertAfterDS": False,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "EnableMatrixInstruction": True,
        "ISA": (11, 5, 1),
        "InnerUnroll": 1,
        "LoopIters": 1,
        "PrefetchGL2": 0,
        "PrefetchGlobalRead": 1,
        "ProblemType": {
            "MacDataTypeA": DataType(mac_data_type),
            "MacDataTypeB": DataType(mac_data_type),
            "MXBlockA": False,
            "MXBlockB": False,
            "Sparse": 0,
        },
        "StoreSwapAddr": False,
        "UseDotInstruction": False,
        "enableTDMA": False,
        "enableTDMB": False,
    }
    kernel.update(overrides)
    return kernel


def _kernel_with_problem_override(key, value):
    kernel = _kernel()
    kernel["ProblemType"] = deepcopy(kernel["ProblemType"])
    kernel["ProblemType"][key] = value
    return kernel


def _representative_config(data_type):
    bias_type = "h" if data_type == "H" else "b"
    problem_type = {
        "OperationType": "GEMM",
        "DataType": data_type,
        "DataTypeA": data_type,
        "DataTypeB": data_type,
        "DestDataType": data_type,
        "ComputeDataType": "S",
        "HighPrecisionAccumulate": True,
        "TransposeA": False,
        "TransposeB": True,
        "UseBeta": True,
        "Batched": True,
        "BiasSrc": "D",
        "UseBias": 1,
        "BiasDataTypeList": [bias_type],
        "UseScaleAlphaVec": 1,
        "Activation": True,
        "ActivationType": "hipblaslt_all",
        "SupportUserArgs": True,
    }
    solution_parameters = {
        "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
        "WavefrontSize": 32,
        "AssertFree0ElementMultiple": 8,
        "AssertFree1ElementMultiple": 8,
        "AssertSummationElementMultiple": 16,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ClusterLocalRead": 0,
        "DepthU": 16,
        "VectorWidthA": 1,
        "VectorWidthB": 2,
        "MIArchVgpr": True,
        "StaggerU": 32,
        "StaggerUStride": 256,
        "WorkGroupMapping": 8,
        "LocalReadVectorWidth": 16,
        "ScheduleGlobalRead": 1,
        "ScheduleLocalWrite": 1,
        "ScheduleIterAlg": 3,
        "ExpandPointerSwap": False,
        "TransposeLDS": 0,
        "LdsBlockSizePerPadA": 0,
        "LdsBlockSizePerPadB": 0,
        "LdsPadA": 0,
        "LdsPadB": 0,
        "1LDSBuffer": 1,
        "SourceSwap": True,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "StoreVectorWidth": -1,
        "StoreRemapVectorWidth": 0,
        "StorePriorityOpt": False,
        "GroupLoadStore": False,
        "NumElementsPerBatchStore": 10,
        "StoreSyncOpt": 0,
    }
    size_group = {
        "InitialSolutionParameters": None,
        "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
        "ForkParameters": [
            {key: [value]} for key, value in solution_parameters.items()
        ],
        "BenchmarkJoinParameters": None,
        "BenchmarkFinalParameters": [
            {"ProblemSizes": [{"Exact": [8192, 8192, 1, 8192]}]},
            {"BiasTypeArgs": [bias_type]},
            {"FactorDimArgs": [0]},
            {"ActivationArgs": [[{"Enum": "none"}]]},
        ],
    }
    return problem_type, size_group


@contextmanager
def _isolated_globals(isa_info_map):
    saved_global = deepcopy(dict(globalParameters))
    saved_valid = deepcopy(dict(validParameters))
    try:
        assignGlobalParameters({}, isa_info_map)
        yield
    finally:
        globalParameters.clear()
        globalParameters.update(saved_global)
        validParameters.clear()
        validParameters.update(saved_valid)


def _representative_writer(data_type):
    compiler = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    isa = gfxToIsa("gfx1151")
    isa_info_map = makeIsaInfoMap([isa], compiler)
    assembler = makeAssemblyToolchain(compiler, bundler, "default").assembler

    with _isolated_globals(isa_info_map):
        problem_type, size_group = _representative_config(data_type)
        benchmark_process = BenchmarkProcess(problem_type, size_group, False)
        benchmark_step = benchmark_process[0]
        permutations = list(
            constructForkPermutations(
                benchmark_step.forkParams, benchmark_step.paramGroups
            )
        )
        solution = _generate_single_solution(
            permutations[0],
            benchmark_process.problemType,
            benchmark_step.constantParams,
            assembler,
            makeDebugConfig({}),
            isa_info_map,
        )
        assert solution is not None
        kernel = generateKernelObjectsFromSolutions([solution])[0]
        kernel.duplicate = False
        kernel["BaseName"] = getKernelFileBase(False, kernel)

        rocisa_instance = rocisa.rocIsa.getInstance()
        rocisa_instance.init(tuple(kernel["ISA"]), str(compiler))
        rocisa_instance.setKernel(
            tuple(kernel["ISA"]), kernel["WavefrontSize"]
        )
        writer = KernelWriterAssembly(assembler, DebugConfig())
        writer.setRocIsa(
            rocisa_instance.getData(), rocisa_instance.getOutputOptions()
        )
        error, _ = writer.getSourceFileString(kernel)

    assert error == 0
    return writer, kernel


@pytest.mark.parametrize("mac_data_type", ["H", "B"])
def test_simple_16bit_wmma_uses_valu_first_layout(mac_data_type):
    assert _writer().canUseValuFirstVgprLayout(_kernel(mac_data_type))


@pytest.mark.parametrize(
    "kernel_overrides, writer_overrides",
    [
        ({"BufferLoad": False}, {}),
        ({"ConvertAfterDS": True}, {}),
        ({"DirectToLdsA": True}, {}),
        ({"DirectToLdsB": True}, {}),
        ({"DirectToVgprA": True}, {}),
        ({"DirectToVgprB": True}, {}),
        ({"EnableMatrixInstruction": False}, {}),
        ({"LoopIters": 2}, {}),
        ({"PrefetchGL2": 1}, {}),
        ({"PrefetchGlobalRead": 0}, {}),
        ({"StoreSwapAddr": True}, {}),
        ({"enableTDMA": True}, {}),
        ({"enableTDMB": True}, {}),
        ({}, {"inc_lds_buf_switch": True}),
        ({}, {"num_vgpr_buffer": 2}),
    ],
)
def test_unsupported_lifetimes_keep_generic_layout(
    kernel_overrides, writer_overrides
):
    assert not _writer(**writer_overrides).canUseValuFirstVgprLayout(
        _kernel(**kernel_overrides)
    )


@pytest.mark.parametrize("key", ["Sparse", "MXBlockA", "MXBlockB"])
def test_extended_operands_keep_generic_layout(key):
    assert not _writer().canUseValuFirstVgprLayout(
        _kernel_with_problem_override(key, 1)
    )


def test_non_16bit_operands_keep_generic_layout():
    assert not _writer().canUseValuFirstVgprLayout(_kernel("S"))


def test_valu_alignment_separates_a_from_c_and_selects_b_bank():
    writer = _writer()
    kernel = _kernel()

    assert writer.alignVgprForValuA(kernel, 128, True) == 130
    assert writer.alignVgprForValuB(kernel, 162) == 163
    assert writer.alignVgprForValuB(kernel, 164) == 165


@pytest.mark.parametrize("data_type", ["H", "B"])
def test_representative_gfx1151_layout_has_disjoint_ordered_regions(
    data_type,
):
    writer, kernel = _representative_writer(data_type)
    state = writer.states

    end_c = state.c.startVgprValu + state.c.numVgprValu
    end_a = state.a.startVgprValu + state.a.numVgprValu
    end_b = state.b.startVgprValu + state.b.numVgprValu
    end_local_read_b = (
        state.b.startVgprLocalReadAddr + state.b.numVgprLocalReadAddr
    )

    assert state.numVgprBuffer == 1
    assert writer.canUseValuFirstVgprLayout(kernel)
    assert end_c == 128
    assert (
        state.startVgpr
        == state.startVgprMisc
        == state.a.startVgprValu
        == 130
    )
    assert end_a <= state.b.startVgprValu
    assert (
        end_b
        == state.firstVgprForReads
        == writer.startVgprGlobalReadOffsetA
    )
    assert state.firstVgprForReads <= state.a.startVgprLocalWriteAddr
    assert state.a.startVgprLocalWriteAddr < state.a.startVgprG2L
    assert state.a.startVgprG2L < state.b.startVgprG2L
    assert state.b.startVgprG2L < state.a.startVgprLocalReadAddr
    assert state.a.startVgprLocalReadAddr < state.b.startVgprLocalReadAddr
    assert end_local_read_b == state.lastVgprForReads
    assert state.lastVgprForReads + 1 == state.totalVgprs == 219
    assert writer.vgprPool.size() == 219
