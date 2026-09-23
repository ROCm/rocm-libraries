# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise canonical policy through benchmark permutations and solution identity."""

from pathlib import Path

import pytest
import yaml

from config_harness import solutions_from_config
from Tensile.SolutionStructs.Naming import getKeyNoInternalArgs, getSolutionNameFull

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parent / "characterization/_codegen/data/test_data/_designed/gfx942/streamk.yaml"


def _config(selectors):
    config = yaml.safe_load(_FIXTURE.read_text())
    group = config["BenchmarkProblems"][0][1]
    policy_fields = {
        "TileProcessingStrategy", "WorkAssignment", "StreamK", "StreamKForceDPOnly",
        "StreamKXCCMapping", "PersistentXCCMapping", "StreamKWorkStealing", "WorkQueueStealing",
        "StreamKAtomic", "GlobalSplitU",
    }
    group["ForkParameters"] = [
        item for item in group["ForkParameters"] if not set(item) & policy_fields
    ] + [{"StreamKAtomic": [0]}, {"GlobalSplitU": [1]}] + [{key: value} for key, value in selectors.items()]
    return config


def _derive(tmp_path, name, config):
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return solutions_from_config(path, arch="gfx942")


@pytest.mark.parametrize("strategy,assignment,mode,force,fragment", (
    ("None", "StaticGrid", 0, 0, "TPSN"),
    ("DataParallel", "StaticGrid", 3, 1, "TPSDP"),
    ("StreamK", "StaticGrid", 3, 0, "TPSSK"),
    ("StreamK", "DynamicWorkQueue", 4, 0, "TPSSK"),
    ("StreamK", "Hybrid", 5, 0, "TPSSK"),
))
def test_legacy_and_canonical_benchmark_input_have_one_identity(tmp_path, strategy, assignment, mode, force, fragment):
    canonical = _derive(tmp_path, "canonical", _config({"TileProcessingStrategy": [strategy], "WorkAssignment": [assignment]}))
    legacy = _derive(tmp_path, "legacy", _config({"StreamK": [mode], "StreamKForceDPOnly": [force]}))
    assert len(canonical) == len(legacy) == 1
    state = canonical[0]
    assert state["TileProcessingStrategy"] == strategy
    assert state["WorkAssignment"] == assignment
    assert state["_PersistentLoop"] is (strategy != "None")
    assert "StreamK" not in state and "StreamKForceDPOnly" not in state
    assert getKeyNoInternalArgs(state, False) == getKeyNoInternalArgs(legacy[0], False)
    name = getSolutionNameFull(state, False)
    assert name == getSolutionNameFull(legacy[0], False)
    assert fragment in name
    if strategy == "None":
        assert "_WA" not in name
    else:
        assert {"StaticGrid": "WASG", "DynamicWorkQueue": "WADWQ", "Hybrid": "WAH"}[assignment] in name
    if strategy != "StreamK":
        assert "SKFTR" not in name


def test_baseline_and_streamk_sweep_preserves_inactive_default(tmp_path):
    states = _derive(tmp_path, "baseline_sweep", _config({"TileProcessingStrategy": ["None", "StreamK"]}))
    assert {(s["TileProcessingStrategy"], s["WorkAssignment"]) for s in states} == {
        ("None", "StaticGrid"), ("StreamK", "StaticGrid"),
    }


@pytest.mark.parametrize("assignment", ("DynamicWorkQueue", "Hybrid"))
def test_none_ignores_known_assignment_in_benchmark_input(tmp_path, assignment):
    inactive = _derive(tmp_path, "inactive", _config({
        "TileProcessingStrategy": ["None"], "WorkAssignment": [assignment],
    }))
    omitted = _derive(tmp_path, "omitted", _config({"TileProcessingStrategy": ["None"]}))
    assert len(inactive) == len(omitted) == 1
    assert inactive[0]["WorkAssignment"] == "StaticGrid"
    assert inactive[0]["_PersistentLoop"] is False
    assert getKeyNoInternalArgs(inactive[0], False) == getKeyNoInternalArgs(omitted[0], False)
    assert getSolutionNameFull(inactive[0], False) == getSolutionNameFull(omitted[0], False)


def test_parameter_groups_keep_explicit_selector_provenance_separate(tmp_path):
    config = _config({})
    config["BenchmarkProblems"][0][1]["ForkParameters"].append({"Groups": [[
        {"TileProcessingStrategy": "DataParallel"},
        {"StreamK": 5},
    ]]})
    states = _derive(tmp_path, "mixed_groups", config)
    assert {(s["TileProcessingStrategy"], s["WorkAssignment"]) for s in states} == {
        ("DataParallel", "StaticGrid"), ("StreamK", "Hybrid"),
    }


@pytest.mark.parametrize("strategy,assignment,mode,force", (
    ("None", "StaticGrid", 0, 0),
    ("DataParallel", "StaticGrid", 3, 1),
    ("StreamK", "StaticGrid", 3, 0),
    ("StreamK", "DynamicWorkQueue", 4, 0),
    ("StreamK", "Hybrid", 5, 0),
))
def test_logic_extraction_regenerates_legacy_and_canonical_identity(tmp_path, strategy, assignment, mode, force):
    from Tensile.TensileLibLogicToYaml import formForkParams

    canonical = _derive(tmp_path, "source", _config({
        "TileProcessingStrategy": [strategy], "WorkAssignment": [assignment],
        "PersistentXCCMapping": [8 if mode else 0],
        "WorkQueueStealing": [int(mode in (4, 5))],
    }))[0]
    legacy = dict(canonical)
    legacy.pop("TileProcessingStrategy")
    legacy.pop("WorkAssignment")
    legacy["StreamK"] = mode
    legacy["StreamKForceDPOnly"] = force
    legacy["StreamKXCCMapping"] = legacy.pop("PersistentXCCMapping")
    legacy["StreamKWorkStealing"] = legacy.pop("WorkQueueStealing")
    regenerated = []
    for name, source in (("canonical", canonical), ("legacy", legacy)):
        config = _config({})
        group = config["BenchmarkProblems"][0][1]
        group.update(yaml.safe_load(yaml.dump(formForkParams(dict(source), False), sort_keys=False)))
        states = _derive(tmp_path, name, config)
        assert len(states) == 1
        regenerated.append(states[0])
    assert regenerated[0]["TileProcessingStrategy"] == strategy
    assert regenerated[0]["WorkAssignment"] == assignment
    assert getKeyNoInternalArgs(regenerated[0], False) == getKeyNoInternalArgs(regenerated[1], False)
    assert getSolutionNameFull(regenerated[0], False) == getSolutionNameFull(regenerated[1], False)
