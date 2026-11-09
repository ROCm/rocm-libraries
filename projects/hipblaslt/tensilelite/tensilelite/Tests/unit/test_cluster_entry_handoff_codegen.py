# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise cluster entry ordering through Python reconstruction and the backend."""
from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).resolve().parents[1] / "common/gemm/gfx1250/cluster_entry_handoff.yaml"
_LAYOUTS = {(1, 4), (2, 2), (4, 1)}


@pytest.fixture(scope="module")
def kernels():
    results = emit_kernels_from_config(
        _CONFIG, limit=12, arch="gfx1250", canonical=False, expected_fork_count=12,
    )
    assert len(results) == 12
    indexed = {}
    for name, source, error in results:
        assert error == 0, (name, error)
        # The full kernel symbol encodes the public parameters, unlike the
        # shortened filename returned by the harness.
        sia = int(re.search(r"_SIA([04])_", source).group(1))
        clone = int(re.search(r"_ICIW([01])_", source).group(1))
        layout = tuple(map(int, re.search(r"_MIWT(\d+)_(\d+)_", source).groups()))
        key = (sia, clone, layout)
        assert key not in indexed
        indexed[key] = (name, source)
    assert set(indexed) == {(sia, clone, layout)
                            for sia in (0, 4) for clone in (0, 1) for layout in _LAYOUTS}
    return indexed


@pytest.mark.parametrize("sia", [0, 4])
@pytest.mark.parametrize("clone", [0, 1])
def test_cluster_entry_handoff_codegen(kernels, sia, clone):
    for layout in sorted(_LAYOUTS):
        name, source = kernels[sia, clone, layout]
        assert_assembles(source, name)
        instructions = [line.split("//", 1)[0].strip() for line in source.splitlines()]
        instructions = [line for line in instructions
                        if line.startswith(("s_", "v_", "tensor_", "ds_", "global_", "buffer_"))]
        first_load = next(i for i, line in enumerate(instructions)
                          if line.startswith("tensor_load_to_lds"))
        if sia == 0:
            # A second immediate pair would be redundant: Python's surviving
            # prefetch join already precedes the first loop cluster signal.
            assert instructions[first_load - 1] == "s_barrier_wait -3", name
            assert "For stream-k / persistent loop" in source
        else:
            # The LDS-token rebuild removes the prefetch join, so the entry
            # handoff must precede the early signal in the cloned or main loop.
            assert instructions[first_load - 3:first_load] == [
                "s_barrier_wait -3", "s_barrier_signal -1", "s_barrier_wait -1",
            ], name
            assert "For stream-k / persistent loop" not in source
        entry_wait = max(i for i in range(first_load)
                         if instructions[i] == "s_barrier_wait -3")
        next_signal = next(i for i in range(first_load, len(instructions))
                           if instructions[i] == "s_barrier_signal -3")
        assert any(instructions[i:i + 2] == ["s_barrier_signal -1", "s_barrier_wait -1"]
                   for i in range(entry_wait + 1, next_signal - 1)), name
        assert ("label_InitCIterWmma_" in source) == bool(clone), name


@pytest.mark.parametrize("sia", [0, 4])
@pytest.mark.parametrize("depth_u", [128, 256])
def test_nonpersistent_cluster_entry_reuses_prefetch_join(tmp_path, sia, depth_u):
    # DU128 has effective PLR0 and needs an entry join. DU256 has effective
    # PLR1: its existing prefetch join is sufficient under the nonzero-K guard.
    config = yaml.safe_load(_CONFIG.read_text())
    forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
    overrides = {
        "TileProcessingStrategy": ["None"], "GlobalSplitU": [1],
        "ScheduleIterAlg": [sia], "InitCIterWmma": [1], "DepthU": [depth_u],
        "Groups": [[{"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]}]],
    }
    for fork in forks:
        for key in fork.keys() & overrides.keys():
            fork[key] = overrides[key]
    path = tmp_path / "nonpersistent.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    results = emit_kernels_from_config(
        path, limit=1, arch="gfx1250", canonical=False, expected_fork_count=1,
    )
    assert len(results) == 1
    name, source, error = results[0]
    assert error == 0, (name, error)
    assert_assembles(source, name)
    instructions = [line.split("//", 1)[0].strip() for line in source.splitlines()]
    instructions = [line for line in instructions
                    if line.startswith(("s_", "v_", "tensor_", "ds_", "global_", "buffer_"))]
    first_load = next(i for i, line in enumerate(instructions)
                      if line.startswith("tensor_load_to_lds"))
    if depth_u == 128:
        assert instructions[first_load - 3:first_load] == [
            "s_barrier_wait -3", "s_barrier_signal -1", "s_barrier_wait -1",
        ], name
    else:
        assert instructions[first_load - 1] == "s_barrier_wait -3", name
    entry_wait = max(i for i in range(first_load)
                     if instructions[i] == "s_barrier_wait -3")
    next_signal = next(i for i in range(first_load, len(instructions))
                       if instructions[i] == "s_barrier_signal -3")
    assert any(instructions[i:i + 2] == ["s_barrier_signal -1", "s_barrier_wait -1"]
               for i in range(entry_wait + 1, next_signal - 1)), name


@pytest.mark.parametrize("arch, skipped", [
    ("gfx942", True), ("gfx1250", False), ("gfx1250-strict", False),
])
def test_cluster_regression_arch_selection(monkeypatch, arch, skipped):
    common = _CONFIG.parents[2]
    monkeypatch.syspath_prepend(str(common))
    from config_helpers import configMarks
    marks = configMarks(str(_CONFIG),
                        str(common.parent), [arch])
    assert any(mark.name == "skip" for mark in marks) == skipped
    assert any(mark.name == "common" for mark in marks)
    assert any(mark.name == "gfx1250-strict" for mark in marks)
