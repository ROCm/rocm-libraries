# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LibraryIO MessagePack writes go through the C++ writer when it was built."""

import gzip
import importlib.util
import zlib
from pathlib import Path

import msgpack
import pytest

from Tensile.LibraryIO import writeMsgPack, writeMsgPackIndexed


def _writer():
    if importlib.util.find_spec("tensilelite.tensilelite_write") is None:
        pytest.skip("tensilelite_write extension was not built")
    from tensilelite import tensilelite_write

    return tensilelite_write


def _kernels_lite_state():
    fixture = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "configs"
        / "SolutionLibraries"
        / "KernelsLite.dat.gz"
    )
    raw = gzip.decompress(fixture.read_bytes())
    state = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    assert isinstance(state, dict)
    assert "solutions" in state and "library" in state
    return state


def _unpack_zlib(path):
    return zlib.decompress(path.read_bytes())


def test_library_io_msgpack_writer_documents_load(tmp_path):
    writer = _writer()
    state = _kernels_lite_state()

    eager_base = tmp_path / "eager.dat"
    writeMsgPack(str(eager_base), state)
    eager_raw = _unpack_zlib(tmp_path / "eager.dat.zlib")
    eager = msgpack.unpackb(eager_raw, raw=False, strict_map_key=False)
    assert "solutions" in eager
    assert "library" in eager
    assert "format_version" not in eager
    assert "solutions_index" not in eager
    # The module loads the document it just wrote.
    assert writer.write_library(eager_raw)

    indexed_base = tmp_path / "indexed.dat"
    writeMsgPackIndexed(str(indexed_base), state)
    indexed_raw = _unpack_zlib(tmp_path / "indexed.dat.zlib")
    indexed = msgpack.unpackb(indexed_raw, raw=False, strict_map_key=False)
    assert indexed["format_version"] == 2
    assert isinstance(indexed["solutions_blob"], (bytes, bytearray))
    assert len(indexed["solutions_index"]) % 3 == 0
    assert len(indexed["solutions_index"]) > 0
    assert "solutions" not in indexed
    keys = list(indexed)
    if "version" in indexed:
        assert keys == [
            "format_version",
            "version",
            "solutions_index",
            "solutions_blob",
            "library",
        ]
    else:
        assert keys == [
            "format_version",
            "solutions_index",
            "solutions_blob",
            "library",
        ]
    # Loading the indexed document is the C++ reader's job. write_library does
    # that load, then emits the eager form.
    rewritten = writer.write_library(indexed_raw)
    rewritten_doc = msgpack.unpackb(rewritten, raw=False, strict_map_key=False)
    assert "solutions" in rewritten_doc
    assert "format_version" not in rewritten_doc
    assert "solutions_blob" not in rewritten_doc


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
