# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import pytest
import yaml
from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).parent / "data/test_data/_designed/gfx942/complex_b_read_wait.yaml"


@pytest.mark.parametrize(
    "conjugate_a,conjugate_b", [(False, False), (False, True), (True, False), (True, True)]
)
def test_complex_b_reads_complete_before_first_iteration(conjugate_a, conjugate_b, tmp_path):
    config = yaml.safe_load(_CONFIG.read_text())
    problem = config["BenchmarkProblems"][0][0]
    problem["ComplexConjugateA"] = conjugate_a
    problem["ComplexConjugateB"] = conjugate_b
    path = tmp_path / "complex.yaml"
    path.write_text(yaml.safe_dump(config))

    states = derive_states(path, arch="gfx942", limit_solutions=1)
    assert len(states) == 1
    state = states[0]
    assert state["UnrollMajorLDSB"]
    assert state["MIWaveTile"] == [7, 4]
    assert state["PrefetchLocalRead"] == 1
    assert state["ClusterLocalRead"] == 1

    results = emit_kernels_from_config(path, arch="gfx942", limit=1)
    assert len(results) == 1
    _name, source, error = results[0]
    assert error == 0

    # All loop variants must wait for the prefetched B tile before arithmetic
    # can consume its later operands, including the no-load and PGR1 paths.
    first_iteration_waits = [
        line
        for line in source.splitlines()
        if line.startswith("s_waitcnt") and "for iteration == 0" in line
    ]
    assert len(first_iteration_waits) == 4
    assert all(re.match(r"s_waitcnt lgkmcnt\(0\)", line) for line in first_iteration_waits)

    open_loop = source.index("label_openLoopL:")
    prefetch = source[source.index("/* local read prefetch b */") : open_loop]
    assert prefetch.count("ds_read_b64") == 4
    for label in ("label_LoopBeginL:", "label_LoopEndL:"):
        start = source.index(label, open_loop)
        first_arithmetic = re.search(r"^v_(?:add|mfma)_", source[start:], re.M)
        assert first_arithmetic is not None
        before_use = source[start : start + first_arithmetic.start()]
        assert "s_waitcnt lgkmcnt(0)" in before_use
