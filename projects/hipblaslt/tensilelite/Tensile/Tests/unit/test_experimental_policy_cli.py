# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI policy values and augmented benchmark sweeps must reach generation."""

from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
import yaml

from config_harness import _isolated_globals_with_isa, _toolchain_for, solutions_from_config
from Tensile.BenchmarkStructs import _expandGroupedParameters, constructLazyForkPermutations
from Tensile.Common.GlobalParameters import globalParameters
from Tensile.ExperimentalLibrary import (
    ExperimentalLibraryError, augment_config, main, parse_set_arg,
)

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parents[1] / "common/streamk/data_parallel_static_sgemm.yaml"


@pytest.fixture(autouse=True)
def serial_generation(monkeypatch):
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)


def _config(legacy=False):
    config = yaml.safe_load(_FIXTURE.read_text())
    fork = config["BenchmarkProblems"][0][1]["ForkParameters"]
    selectors = {"TileProcessingStrategy", "WorkAssignment", "StreamK", "StreamKForceDPOnly"}
    fork[:] = [entry for entry in fork if not selectors.intersection(entry)]
    if legacy:
        fork.extend([{"StreamK": [3]}, {"StreamKForceDPOnly": [0]}])
    else:
        fork.extend([{"TileProcessingStrategy": ["DataParallel"]}, {"WorkAssignment": ["StaticGrid"]}])
    return config


def _cli_augment(tmp_path, config, sets):
    source, output = tmp_path / "input.yaml", tmp_path / "augmented.yaml"
    source.write_text(yaml.safe_dump(config))
    argv = ["augment", "--config", str(source), "--out", str(output),
            "--feature-name", "policy-regression", "--arch", "gfx942"]
    for setting in sets:
        argv.extend(["--set", setting])
    assert main(argv) == 0
    return output


def _candidates(size_group):
    common = {key: values for entry in size_group.get("BenchmarkCommonParameters", [])
              for key, values in entry.items()}
    fork = {key: values for entry in size_group["ForkParameters"] for key, values in entry.items()}
    groups = _expandGroupedParameters(fork.pop("Groups", []))
    return list(constructLazyForkPermutations(dict(common, **fork), groups))


def test_none_string_is_scoped_to_the_policy_enum():
    assert parse_set_arg("TileProcessingStrategy=None,StreamK") == (
        "TileProcessingStrategy", ["None", "StreamK"]
    )
    assert parse_set_arg("UnrelatedParameter=None") == ("UnrelatedParameter", [None])
    assert parse_set_arg("UnrelatedParameter=[None,1]") == ("UnrelatedParameter", [[None, 1]])


@pytest.mark.parametrize("value,expected", [("None", "0,1"), ("None,StreamK", "0,1,2")])
def test_cli_where_matches_disabled_policy(tmp_path, capsys, value, expected):
    source = tmp_path / "logic.yaml"
    source.write_text(yaml.safe_dump([[
        {"SolutionIndex": 0, "StreamK": 0},
        {"SolutionIndex": 1, "TileProcessingStrategy": "None"},
        {"SolutionIndex": 2, "TileProcessingStrategy": "StreamK"},
        {"SolutionIndex": 3, "TileProcessingStrategy": "DataParallel"},
    ]]))
    assert main(["list-solutions", "--logic-src", str(source), "--indices-only",
                 "--where", "TileProcessingStrategy=" + value]) == 0
    assert capsys.readouterr().out.strip() == expected


@pytest.mark.parametrize("legacy,settings,expected", [
    (False, ["StreamKForceDPOnly=0"], {("StreamK", "StaticGrid")}),
    (True, ["TileProcessingStrategy=DataParallel"], {("DataParallel", "StaticGrid")}),
    (False, ["StreamKForceDPOnly=0,1"], {("StreamK", "StaticGrid"), ("DataParallel", "StaticGrid")}),
    (True, ["TileProcessingStrategy=None", "GlobalSplitU=1"], {("None", "StaticGrid")}),
    (True, ["TileProcessingStrategy=None,StreamK", "GlobalSplitU=1"], {("None", "StaticGrid"), ("StreamK", "StaticGrid")}),
])
def test_cli_augmentation_generates_requested_policies(tmp_path, legacy, settings, expected):
    output = _cli_augment(tmp_path, _config(legacy), settings)
    solutions = solutions_from_config(output, arch="gfx942")
    assert len(solutions) == len(expected)
    assert {(s["TileProcessingStrategy"], s["WorkAssignment"]) for s in solutions} == expected
    assert all(s["Valid"] for s in solutions)


def test_augmentation_preserves_group_correlations_and_later_overrides(tmp_path):
    config = _config(legacy=True)
    group = config["BenchmarkProblems"][0][1]
    group["BenchmarkCommonParameters"].extend([
        {"StreamK": [3]}, {"StreamKForceDPOnly": [0]}, {"PersistentXCCMapping": [0, 8]},
    ])
    group["ForkParameters"] = [entry for entry in group["ForkParameters"]
                               if not {"StreamK", "StreamKForceDPOnly"}.intersection(entry)]
    group["ForkParameters"].append({"Groups": [
        [{"NonTemporalA": [0, 1]}],
        [{"StreamK": 3, "DepthU": [16, 32]}, {"StreamK": 5, "DepthU": 16}],
        [{"DepthU": 16}],
        [{"StreamKForceDPOnly": 0, "WorkGroupMapping": [1, 2]}],
        [{"VectorWidthA": 1}],
    ]})
    before = deepcopy(config)
    output = _cli_augment(tmp_path, config, ["TileProcessingStrategy=StreamK"])
    augmented = yaml.safe_load(output.read_text())
    result = augmented["BenchmarkProblems"][0][1]
    assert result["BenchmarkFinalParameters"] == before["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"]
    assert augmented["GlobalParameters"] == before["GlobalParameters"]
    candidates = _candidates(result)
    actual = {(s["TileProcessingStrategy"], s["WorkAssignment"], s["DepthU"],
               s["PersistentXCCMapping"], s["NonTemporalA"], s["WorkGroupMapping"])
              for s in candidates}
    expected = {("StreamK", assignment, 16, xcc, nontemporal, wgm)
                for assignment, xcc, nontemporal, wgm
                in product(("StaticGrid", "Hybrid"), (0, 8), (0, 1), (1, 2))}
    assert actual == expected
    assert len(candidates) == len(expected)
    assert all(s["MatrixInstruction"] == [16, 16, 4, 1, 1, 1, 1, 2, 2] for s in candidates)
    solutions = solutions_from_config(output, arch="gfx942")
    assert len(solutions) == len(expected)


def test_every_problem_size_group_receives_policy_override():
    config = _config()
    problem = config["BenchmarkProblems"][0]
    problem.append(deepcopy(problem[1]))
    config["BenchmarkProblems"].append(deepcopy(problem))
    augment_config(config, [parse_set_arg("StreamKForceDPOnly=0,1")])
    for problem in config["BenchmarkProblems"]:
        for group in problem[1:]:
            assert {s["TileProcessingStrategy"] for s in _candidates(group)} == {"DataParallel", "StreamK"}


@pytest.mark.parametrize("legacy", [False, True])
def test_shared_alias_override_takes_precedence_across_common_fork_and_groups(legacy):
    source, override = (("StreamKWorkStealing", "WorkQueueStealing") if legacy else
                        ("WorkQueueStealing", "StreamKWorkStealing"))
    config = {"BenchmarkProblems": [[{}, {
        "BenchmarkCommonParameters": [{"TileProcessingStrategy": ["StreamK"]}, {source: [1]}],
        "ForkParameters": [{"WorkAssignment": ["Hybrid"]}, {"Groups": [[{source: 1, "DepthU": [16, 32]}]]}],
    }]]}
    augment_config(config, [(override, [0, 1])])
    assert {(s["WorkQueueStealing"], s["DepthU"]) for s in _candidates(config["BenchmarkProblems"][0][1])} == set(product((0, 1), (16, 32)))


def test_unsupported_candidates_are_filtered_without_losing_valid_sweep_choices():
    config = {"BenchmarkProblems": [[{}, {"ForkParameters": [
        {"TileProcessingStrategy": ["DataParallel", "StreamK"]},
        {"WorkAssignment": ["StaticGrid", "Hybrid"]},
    ]}]]}
    augment_config(config, [("PersistentXCCMapping", [0, 8])])
    candidates = _candidates(config["BenchmarkProblems"][0][1])
    assert {(s["TileProcessingStrategy"], s["WorkAssignment"], s["PersistentXCCMapping"])
            for s in candidates} == {
        (strategy, assignment, xcc)
        for (strategy, assignment), xcc in product(
            (("DataParallel", "StaticGrid"), ("StreamK", "StaticGrid"), ("StreamK", "Hybrid")), (0, 8)
        )
    }


@pytest.mark.parametrize("settings,match", [
    (["StreamK=4", "WorkAssignment=Hybrid"], "Conflicting"),
    (["StreamKXCCMapping=4", "PersistentXCCMapping=8"], "Conflicting"),
    (["TileProcessingStrategy=NotAStrategy"], "TileProcessingStrategy must"),
    (["WorkAssignment=NotAnAssignment"], "WorkAssignment must"),
    (["StreamK=True,3"], "Legacy StreamK must"),
    (["TileProcessingStrategy=DataParallel", "WorkAssignment=Hybrid"], "No supported execution-policy combinations"),
])
def test_augmentation_rejects_explicit_conflicts_and_malformed_selectors(settings, match):
    with pytest.raises(ExperimentalLibraryError, match=match):
        augment_config(_config(legacy=True), [parse_set_arg(setting) for setting in settings])


def test_nonpolicy_augmentation_preserves_groups():
    config = _config()
    group = config["BenchmarkProblems"][0][1]
    groups = [[{"TileProcessingStrategy": "DataParallel", "DepthU": [16, 32]}], [{"NonTemporalA": [0, 1]}]]
    group["ForkParameters"].append({"Groups": deepcopy(groups)})
    augment_config(config, [("PrefetchGlobalRead", [1, 2])])
    assert next(entry["Groups"] for entry in group["ForkParameters"] if "Groups" in entry) == groups


def _main_cli_policy_lines(tmp_path, yaml_globals, overrides):
    from Tensile.ClientWriter import writeClientConfigIni
    from Tensile.Contractions import ProblemType as ContractionProblemType
    from Tensile.SolutionStructs import FactorDimArgs
    from Tensile.SolutionStructs.Problem import ProblemSizesMockDummy, ProblemType
    from Tensile.Tensile import Tensile

    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"GlobalParameters": dict(PrintLevel=0, **yaml_globals)}))
    argv = [str(config), str(tmp_path / "output"), "--gpu-targets", "gfx942", "--cpu-only"]
    if overrides:
        argv.extend(["--global-parameters", *overrides])
    with _isolated_globals_with_isa(_toolchain_for("gfx942")[1]):
        # No benchmark steps are needed to exercise the real entry point's
        # YAML/CLI precedence and the resulting client configuration.
        Tensile(argv)
        problem = ProblemType({"OperationType": "GEMM", "DataType": "s", "Batched": True}, False)
        ini = tmp_path / "ClientParameters.ini"
        writeClientConfigIni(
            forBenchmark=True, problemSizes=ProblemSizesMockDummy(),
            biasTypeArgs="", factorDimArgs=FactorDimArgs(problem, []),
            activationArgs="", icacheFlushArgs="",
            problemType=ContractionProblemType.FromOriginalState(problem.state),
            sourceDir=str(tmp_path), codeObjectFiles=[], resultsFileName=str(tmp_path / "results.csv"),
            parametersFilePath=str(ini), deviceId=0, gfxName="gfx942",
            libraryFile=str(tmp_path / "TensileLibrary.dat"),
        )
    return [line for line in ini.read_text().splitlines()
            if line.startswith(("streamk-hybrid-mode=", "hybrid-assignment-policy="))]


@pytest.mark.parametrize("yaml_globals,overrides,expected", [
    ({}, ["StreamKHybridMode=[1]"], ["DynamicWorkQueue"]),
    ({}, ["HybridAssignmentPolicy=['DynamicWorkQueue']"], ["DynamicWorkQueue"]),
    ({"HybridAssignmentPolicy": ["Auto"]}, ["StreamKHybridMode=1"], ["DynamicWorkQueue"]),
    ({"StreamKHybridMode": [1]}, ["HybridAssignmentPolicy='Auto'"], ["Auto"]),
    ({"HybridAssignmentPolicy": ["DynamicWorkQueue"]}, ["StreamKHybridMode=[0]"], []),
    ({"StreamKHybridMode": [1]}, [], ["DynamicWorkQueue"]),
    ({}, ["StreamKHybridMode=[0,1,2]",
          "HybridAssignmentPolicy=['Default','DynamicWorkQueue','Auto']"],
     ["Default", "DynamicWorkQueue", "Auto"]),
    pytest.param({}, ["StreamKHybridMode=(1,)"], ["DynamicWorkQueue"], id="legacy-tuple"),
    pytest.param({"StreamKHybridMode": [1]}, ["HybridAssignmentPolicy=('Default','Auto')"],
                 ["Default", "Auto"], id="canonical-tuple-overrides-yaml"),
])
def test_main_cli_hybrid_policy_reaches_client_config(tmp_path, yaml_globals, overrides, expected):
    assert _main_cli_policy_lines(tmp_path, yaml_globals, overrides) == [
        "hybrid-assignment-policy=" + name for name in expected
    ]


def test_main_cli_rejects_conflicting_hybrid_aliases(tmp_path):
    with pytest.raises(ValueError, match="Conflicting StreamKHybridMode and HybridAssignmentPolicy"):
        _main_cli_policy_lines(tmp_path, {}, [
            "StreamKHybridMode=[1]", "HybridAssignmentPolicy=['Auto']",
        ])
