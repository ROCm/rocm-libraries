# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only unit coverage of the output-peerD option that ClientWriter passes to the client.

The option is only written for a problem type with OutputPeerD, so the client config text of every
other problem stays exactly as it was.
"""

import copy

import pytest

from tensilelite import ClientWriter
from tensilelite.Common.GlobalParameters import globalParameters, restoreDefaultGlobalParameters
from tensilelite.Contractions import ProblemType
from tensilelite.SolutionStructs.Problem import ProblemSizesMockDummy

pytestmark = pytest.mark.unit


@pytest.fixture
def _restore_gp():
    saved = copy.deepcopy(dict(globalParameters))
    try:
        yield
    finally:
        globalParameters.clear()
        globalParameters.update(saved)


class _FactorDimArgs:
    factorDims = [0]


def _ini(tmp_path, output_peer_d):
    restoreDefaultGlobalParameters()
    globalParameters["CpuOnly"] = False
    problem_type = ProblemType.FromOriginalState(
        {
            "OperationType": "GEMM",
            "DataType": "S",
            "DestDataType": "S",
            "ComputeDataType": "S",
            "TransposeA": 0,
            "TransposeB": 0,
            "Batched": True,
            "UseBeta": True,
            "TotalIndices": 4,
            "NumIndicesC": 3,
            "IndicesFree": [0, 1],
            "IndicesBatch": [2],
            "IndicesSummation": [3],
            "IndexAssignmentsA": [0, 3, 2],
            "IndexAssignmentsB": [3, 1, 2],
            "ComplexConjugateA": False,
            "ComplexConjugateB": False,
            "ActivationComputeDataType": "S",
            "UseBias": 1,
            "BiasSrc": "D",
            "HighPrecisionAccumulate": False,
            "OutputPeerD": output_peer_d,
        }
    )
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    path = source / "ClientParameters.ini"
    ClientWriter.writeClientConfigIni(
        forBenchmark=True,
        problemSizes=ProblemSizesMockDummy(),
        biasTypeArgs="",
        factorDimArgs=_FactorDimArgs(),
        activationArgs="",
        icacheFlushArgs="",
        problemType=problem_type,
        sourceDir=str(source),
        codeObjectFiles=["TensileLibrary_gfx1250.co"],
        resultsFileName="/DATA/benchmark.csv",
        parametersFilePath=str(path),
        deviceId=0,
        gfxName="gfx1250",
        libraryFile="/LIB/gfx1250/TensileLibrary.dat",
    )
    return path.read_text().splitlines()


def test_output_peer_d_is_written_only_when_the_problem_has_it(tmp_path, _restore_gp):
    assert "output-peerD=True" in _ini(tmp_path, True)
    assert not any(line.startswith("output-peerD") for line in _ini(tmp_path, False))
