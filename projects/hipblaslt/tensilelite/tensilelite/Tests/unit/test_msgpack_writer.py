# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ solution-state module, when it has been built."""

import pytest


def _writer():
    return pytest.importorskip("tensilelite_write")


def _small_solution_state():
    """Derived solution state the SolutionState gtests build."""
    return {
        "SolutionNameMin": "",
        "KernelNameMin": "small_kernel",
        "SolutionIndex": 7,
        "DebugKernel": False,
        "ProblemType": {
            "TotalIndices": 4,
            "IndicesSummation": [3],
            "NumIndicesC": 3,
            "IndicesBatch": [2],
            "IndicesFree": [0, 1],
            "IndexAssignmentsA": [0, 3, 2],
            "IndexAssignmentsB": [3, 1, 2],
            "DataType": "Half",
            "DestDataType": "Half",
            "ComputeDataType": "Float",
            "Batched": True,
            "ActivationComputeDataType": "Float",
            "TransposeA": False,
            "TransposeB": True,
            "ComplexConjugateA": False,
            "ComplexConjugateB": False,
            "HighPrecisionAccumulate": True,
            "TLUA": False,
            "TLUB": False,
            "SwizzleTensorA": False,
            "SwizzleTensorB": False,
            "FusedGemmA2A": False,
            "UseBeta": True,
            "StridedBatched": True,
        },
        "ISA": [9, 4, 2],
        "_GlobalAccumulation": "null",
        "PrefetchGlobalRead": 2,
        "EnableMatrixInstruction": False,
        "ThreadTile0": 4,
        "ThreadTile1": 2,
        "NumElementsPerBatchStore": 0,
        "NumElementsPerThread": 4,
        "NumThreads": 64,
        "WavefrontSize": 64,
        "SpaceFillingAlgo": [],
        "WorkGroupMapping": 5,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        "WorkGroup": [64, 1, 1],
        "MacroTile0": 32,
        "MacroTile1": 8,
        "ThreadTile": [4, 2, 1],
        "MatrixInstruction": [16, 16, 16, 1],
        "GlobalReadVectorWidthA": 4,
        "GlobalReadVectorWidthB": 2,
        "StoreVectorWidth": 4,
        "DepthU": 16,
        "GlobalSplitU": 1,
        "KernelLanguage": "Assembly",
        "_WorkspaceSizePerElemC": 4,
        "_WorkspaceSizePerElemBias": 0,
        "ActivationFused": True,
        "WorkGroupMappingXCC": 4,
        "WorkGroupMappingXCCGroup": 8,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "CUOccupancy": 3,
        "MathClocksUnrolledLoop": 11,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "UseCustomMainLoopSchedule": 0,
        "NonTemporalD": 0,
        "WaveSeparateGlobalReadA": 0,
        "WaveSeparateGlobalReadB": 0,
        "UnrollLoopSwapGlobalReadOrder": 0,
        "NumLoadsCoalescedA": 1,
        "NumLoadsCoalescedB": 1,
        "MIWaveGroup": [1, 1],
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "LocalSplitU": 1,
        "ClusterDim": [1, 1, 1],
        "PackedC0IndicesX": [],
        "InternalSupportParams": {
            "KernArgsVersion": 2,
            "SupportUserGSU": True,
            "SupportCustomWGM": False,
            "SupportCustomStaggerU": True,
            "UseUniversalArgs": True,
            "UseSFC": False,
        },
    }


def test_encode_round_trips_a_nested_library_map():
    writer = _writer()
    value = {"gfx950": {"softmax": {"H": [{"co_path": "extop.co", "io_type": "H", "m": 8}]}}}
    assert writer.decode(writer.encode(value)) == value


def test_solution_from_state_kernel_name():
    writer = _writer()
    record = writer.solution_from_state(_small_solution_state())
    assert record["kernelName"] == "small_kernel"


def test_problem_type_index_layout_names_a_mirrored_summation():
    """The operation identifier comes from problemIndexLayout, including a batch and a mirror."""
    _writer()
    from Tensile.Contractions import ProblemType

    problem = _small_solution_state()["ProblemType"]
    problem["MirrorDimsA"] = [3]
    problem["Batched"] = True
    problem["ActivationComputeDataType"] = "Float"
    built = ProblemType.FromOriginalState(problem)
    assert built.operationIdentifier == "Contraction_l_AiLk_Bljk_Cijk_Dijk"
    assert built.indexNames == ("iLk", "ljk", "ijk", "ijk", "l")
    assert built.boundIndices[0].aMirror is True
    assert built.batchIndices[0].d == 2


def test_problem_type_index_layout_keeps_summation_order():
    """Summation order, batch, and both mirrors come from problemIndexLayout.

    IndicesSummation lists 4 before 3. Walking problem indices would put 3
    first and would name the contraction differently.
    """
    _writer()
    from Tensile.Contractions import ProblemType

    problem = _small_solution_state()["ProblemType"]
    problem.update({
        "TotalIndices": 5,
        "IndicesSummation": [4, 3],
        "IndexAssignmentsA": [0, 4, 3, 2],
        "IndexAssignmentsB": [4, 1, 3, 2],
        "MirrorDimsA": [4],
        "MirrorDimsB": [3],
    })
    built = ProblemType.FromOriginalState(problem)
    assert built.operationIdentifier == "Contraction_lm_AiLmk_BljMk_Cijk_Dijk"
    assert built.indexNames == ("iLmk", "ljMk", "ijk", "ijk", "lm")
    assert built.boundIndices[0] is built.indices[4]
    assert built.boundIndices[1] is built.indices[3]
    assert built.boundIndices[0].aMirror is True
    assert built.boundIndices[1].bMirror is True
    assert built.batchIndices[0] is built.indices[2]
    assert built.batchIndices[0].d == 2
