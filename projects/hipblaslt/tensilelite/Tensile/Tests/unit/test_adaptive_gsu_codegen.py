# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate adaptive GSU epilogues through the same configs used by common CI."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config

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
