# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only unit coverage of the GateMulOnly problem-type flag.

With UseGateResidual a kernel stores D = gate * act(...) + gate. GateMulOnly drops the "+ gate"
term (D = gate * act(...)) and reuses everything else of the gate. These tests cover the Python
side: the validation, the naming, the parsing, the selection predicate and the client option.
"""

import copy

import pytest

from tensilelite import ClientWriter
from tensilelite.Common.GlobalParameters import globalParameters, restoreDefaultGlobalParameters
from tensilelite.Contractions import ProblemType as ContractionsProblemType
from tensilelite.SolutionStructs.Problem import ProblemSizesMockDummy, ProblemType

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Problem type: default, validation and name
# ---------------------------------------------------------------------------
def _problem_type(**overrides):
    cfg = {"DataType": 0}
    cfg.update(overrides)
    return ProblemType(cfg, False)


def test_gate_mul_only_is_off_by_default():
    assert _problem_type()["GateMulOnly"] is False
    assert _problem_type(UseGateResidual=True)["GateMulOnly"] is False


def test_gate_mul_only_needs_the_gate(capsys):
    with pytest.raises(SystemExit):
        _problem_type(GateMulOnly=True)
    assert "GateMulOnly needs UseGateResidual" in capsys.readouterr().out


def test_the_problem_type_name_tells_the_two_gate_forms_apart():
    plain = str(_problem_type())
    residual = str(_problem_type(UseGateResidual=True))
    mul = str(_problem_type(UseGateResidual=True, GateMulOnly=True))
    assert "Gate" not in plain
    assert "GateRes" in residual and "GateMul" not in residual
    assert "GateMul" in mul and "GateRes" not in mul
    assert mul == residual.replace("GateRes", "GateMul")


# ---------------------------------------------------------------------------
# Contractions problem type: parsing and the selection predicate
# ---------------------------------------------------------------------------
def _contractions_problem_type(**overrides):
    config = {
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
        "HighPrecisionAccumulate": False,
    }
    config.update(overrides)
    return ContractionsProblemType.FromOriginalState(config)


@pytest.mark.parametrize("given, expected", [(None, False), (False, False), (True, True)])
def test_contractions_problem_type_parses_the_flag(given, expected):
    overrides = {"UseGateResidual": True, "GateResidualDataTypeList": []}
    if given is not None:
        overrides["GateMulOnly"] = given
    assert _contractions_problem_type(**overrides).gateMulOnly is expected


def _gate_mul_predicates(problemType):
    predicates = problemType.predicates(includeBatch=True, includeOperation=True, includeType=True)
    return [p for p in predicates if p.tag == "GateMulOnly"]


@pytest.mark.parametrize("flag", [False, True])
def test_a_gate_kernel_carries_the_flag_in_its_selection_predicate(flag):
    found = _gate_mul_predicates(
        _contractions_problem_type(
            UseGateResidual=True, GateResidualDataTypeList=[], GateMulOnly=flag
        )
    )
    assert len(found) == 1
    assert found[0].value is flag


def test_a_kernel_without_the_gate_has_no_gate_mul_predicate():
    # UseGateResidual false already rejects every gate problem.
    assert _gate_mul_predicates(_contractions_problem_type()) == []


# ---------------------------------------------------------------------------
# Client option
# ---------------------------------------------------------------------------
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


def _ini(tmp_path, gate_mul_only):
    restoreDefaultGlobalParameters()
    globalParameters["CpuOnly"] = False
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
        problemType=_contractions_problem_type(
            UseBias=1,
            BiasSrc="D",
            UseGateResidual=True,
            GateResidualDataTypeList=[],
            GateMulOnly=gate_mul_only,
        ),
        sourceDir=str(source),
        codeObjectFiles=["TensileLibrary_gfx1250.co"],
        resultsFileName="/DATA/benchmark.csv",
        parametersFilePath=str(path),
        deviceId=0,
        gfxName="gfx1250",
        libraryFile="/LIB/gfx1250/TensileLibrary.dat",
    )
    return path.read_text().splitlines()


def test_gate_mul_only_is_written_only_when_the_problem_has_it(tmp_path, _restore_gp):
    assert "gate-mul-only=True" in _ini(tmp_path, True)
    residual = _ini(tmp_path, False)
    assert "use-gate-residual=True" in residual
    assert not any(line.startswith("gate-mul-only") for line in residual)

