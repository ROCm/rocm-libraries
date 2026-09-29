# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate adaptive GSU epilogues through the same configs used by common CI."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
@pytest.mark.parametrize("opt_nll", [0, 1])
@pytest.mark.parametrize("strategy", ["None", "DataParallel", "StreamK"])
def test_adaptive_gsu_store_modes(tmp_path, arch, opt_nll, strategy):
    # The final gsuasb group enables adaptive GSU with bias and activation.
    # Its optimized no-load loop and ordinary epilogue both hit store selection.
    fixture = Path(__file__).parents[1] / "common/gemm/gsuasb.yaml"
    config = yaml.safe_load(fixture.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][-1:]
    # ScaleAB disables OptNoLoadLoop during derivation. Keep both store paths
    # reachable so the test covers the optimized-loop failure reported in CI.
    config["BenchmarkProblems"][0][0]["UseScaleAB"] = ""
    config["GlobalParameters"]["CpuThreads"] = 1
    group = config["BenchmarkProblems"][0][1]
    overrides = {
        "TileProcessingStrategy": strategy,
        "WorkAssignment": "StaticGrid",
        "GlobalSplitU": 2,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "AdaptiveGemmGSUA": 1,
        "OptNoLoadLoop": opt_nll,
    }
    group["ForkParameters"] = [
        p for p in group["ForkParameters"] if not overrides.keys() & p.keys()
    ] + [{key: [value]} for key, value in overrides.items()]
    path = tmp_path / "adaptive_gsu.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    kernels = emit_kernels_from_config(path, arch=arch, limit=1)
    assert len(kernels) == 1
    name, source, error = kernels[0]
    (tmp_path / "adaptive_gsu.s").write_text(source)
    assert error == 0

    # Ordinary kernels choose MB versus MBSK from the runtime synchronizer.
    # Persistent kernels disable adaptive GSU during solution derivation.
    adaptive = strategy == "None"
    assert bool(re.search(r"s_cmp_eq_u64 s\[sgprSynchronizer[^\n]*Check for synchronizer", source)) == adaptive
    assert bool(re.search(r"^label_GW_B\w+_MBSK\w*:", source, re.MULTILINE)) == adaptive
    assert ("long branch if Synchronizer is null" in source) == adaptive
    if adaptive:
        assert re.search(r"^label_GW_B\w+_MB\w*:", source, re.MULTILINE)
        assert ("OptNLL_MB" in source) == bool(opt_nll)
    assert_assembles(source, name)


@pytest.mark.parametrize("algorithm", ["MultipleBuffer", "MultipleBufferSingleKernel"])
def test_gfx1250_adaptive_gsu_store_modes(tmp_path, algorithm):
    # The external gfx1250 job also exercises adaptive GSU through this fixture.
    # Keep one TDM tile shape and test both declared accumulation algorithms.
    fixture = Path(__file__).parents[1] / "common/gemm/gfx12/gsu_gfx1250.yaml"
    config = yaml.safe_load(fixture.read_text())
    config["GlobalParameters"]["CpuThreads"] = 1
    for parameter in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        for key, values in parameter.items():
            parameter[key] = [values[0]]
            if key == "AdaptiveGemmGSUA":
                parameter[key] = [1]
            elif key == "GlobalSplitUAlgorithm":
                parameter[key] = [algorithm]
    path = tmp_path / "adaptive_gsu_gfx1250.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    kernels = emit_kernels_from_config(path, arch="gfx1250", limit=1)
    assert len(kernels) == 1
    name, source, error = kernels[0]
    (tmp_path / "adaptive_gsu_gfx1250.s").write_text(source)
    assert error == 0
    assert re.search(r"^label_GW_B\w+_MBSK\w*:", source, re.MULTILINE)
    assert re.search(r"^label_GW_B\w+_MB\w*:", source, re.MULTILINE)
    assert_assembles(source, name)


@pytest.mark.parametrize("strategy,gsu", [
    ("None", -1), ("None", 1), ("None", 2),
    ("DataParallel", 1), ("StreamK", 1),
])
def test_atomic_dest_preserves_policy_and_workspace_contract(tmp_path, strategy, gsu):
    # AtomicDest accumulates into D without a workspace and rejects persistent
    # execution. Exercise that boundary after legacy selectors are normalized.
    fixture = Path(__file__).parents[1] / "common/gemm/gsuasb.yaml"
    config = yaml.safe_load(fixture.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    config["BenchmarkProblems"][0][0].update(DataType="B", DestDataType="B")
    config["GlobalParameters"]["CpuThreads"] = 1
    group = config["BenchmarkProblems"][0][1]
    overrides = {
        "TileProcessingStrategy": strategy, "GlobalSplitU": gsu,
        "GlobalSplitUAlgorithm": "AtomicDest", "AssertFree0ElementMultiple": 2,
    }
    group["ForkParameters"] = [
        {key: [values[0]] for key, values in parameter.items()}
        for parameter in group["ForkParameters"] if not overrides.keys() & parameter.keys()
    ] + [{key: [value]} for key, value in overrides.items()]
    path = tmp_path / "atomic_dest.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    solutions = solutions_from_config(path, arch="gfx950")
    if strategy != "None":
        assert not solutions
        # Prove that the same kernel is otherwise a valid persistent solution.
        for parameter in group["ForkParameters"]:
            if "GlobalSplitUAlgorithm" in parameter:
                parameter["GlobalSplitUAlgorithm"] = ["MultipleBuffer"]
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        accepted = solutions_from_config(path, arch="gfx950")
        assert len(accepted) == 1
        assert accepted[0]["TileProcessingStrategy"] == strategy
        return

    assert len(solutions) == 1
    assert solutions[0]["_WorkspaceSizePerElemC"] == 0
    assert solutions[0]["GlobalSplitUAlgorithm"] == "AtomicDest"
    kernels = emit_kernels_from_config(path, arch="gfx950", limit=1)
    assert len(kernels) == 1
    name, source, error = kernels[0]
    assert error == 0
    assert "buffer_atomic_pk_add_bf16" in source
    assert_assembles(source, name)
