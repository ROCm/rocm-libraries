# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only unit coverage of the --split-devices options that ClientWriter passes to the client.

SplitDevices / SplitDim / SplitPlacementA..D are global parameters; writeClientConfigIni turns them
into the client's ``split-devices``, ``split-dim`` and ``split-placement`` options (only when the
split is on, so the config text of every other run is unchanged).
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


def _problem_type():
    return ProblemType.FromOriginalState(
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
        }
    )


class _FactorDimArgs:
    factorDims = [0]


def _write_ini(tmp_path, **gp):
    """The client config text for the default parameters with `gp` applied."""
    restoreDefaultGlobalParameters()
    globalParameters["CpuOnly"] = False
    globalParameters.update(gp)
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
        problemType=_problem_type(),
        sourceDir=str(source),
        codeObjectFiles=["TensileLibrary_gfx1250.co"],
        resultsFileName="/DATA/benchmark.csv",
        parametersFilePath=str(path),
        deviceId=0,
        gfxName="gfx1250",
        libraryFile="/LIB/gfx1250/TensileLibrary.dat",
    )
    return path.read_text()


def _options(ini):
    return dict(line.split("=", 1) for line in ini.splitlines() if "=" in line)


def test_no_split_options_are_written_when_the_split_is_off(tmp_path, _restore_gp):
    options = _options(_write_ini(tmp_path))
    assert not any(name.startswith("split-") for name in options)


def test_a_single_device_is_not_a_split(tmp_path, _restore_gp):
    options = _options(_write_ini(tmp_path, SplitDevices=1, SplitDim="N"))
    assert "split-devices" not in options


def test_the_split_options_are_written(tmp_path, _restore_gp):
    options = _options(_write_ini(tmp_path, SplitDevices=2, SplitDim="N"))
    assert options["split-devices"] == "2"
    assert options["split-dim"] == "N"
    # Replicated unless a tensor is asked to be shared.
    assert options["split-placement"] == "RRRR"


def test_a_split_joins_every_launch_unless_free_run_is_set(tmp_path, _restore_gp):
    # The client joins by default, so the option is only written to turn the join off.
    assert "split-free-run" not in _options(_write_ini(tmp_path, SplitDevices=2))
    options = _options(_write_ini(tmp_path, SplitDevices=2, SplitFreeRun=True))
    assert options["split-free-run"] == "True"
    # Without a split there is nothing to join.
    assert "split-free-run" not in _options(_write_ini(tmp_path, SplitFreeRun=True))


def test_each_tensor_has_its_own_placement(tmp_path, _restore_gp):
    options = _options(
        _write_ini(
            tmp_path,
            SplitDevices=2,
            SplitDim="N",
            SplitPlacementA="Shared",
            SplitPlacementD="Shared",
        )
    )
    assert options["split-placement"] == "SRRS"
    options = _options(_write_ini(tmp_path, SplitDevices=2, SplitDim="M", SplitPlacementB="Shared"))
    assert options["split-placement"] == "RSRR"


@pytest.mark.parametrize("tensor", ["A", "B", "C", "D"])
def test_an_unknown_placement_is_refused(tmp_path, _restore_gp, tensor):
    with pytest.raises(SystemExit):
        _write_ini(tmp_path, SplitDevices=2, **{"SplitPlacement" + tensor: "Everywhere"})


@pytest.mark.parametrize(
    "splitDim, tensor",
    [("M", "A"), ("M", "C"), ("N", "B"), ("N", "C")],
)
def test_only_the_unsplit_operand_and_d_can_be_shared(tmp_path, _restore_gp, splitDim, tensor):
    # The client refuses the same placements (client/include/SplitPlacement.hpp).
    with pytest.raises(SystemExit):
        _write_ini(tmp_path, SplitDevices=2, SplitDim=splitDim, **{"SplitPlacement" + tensor: "Shared"})


def test_an_unknown_split_dim_is_refused(tmp_path, _restore_gp):
    with pytest.raises(SystemExit):
        _write_ini(tmp_path, SplitDevices=2, SplitDim="K")
