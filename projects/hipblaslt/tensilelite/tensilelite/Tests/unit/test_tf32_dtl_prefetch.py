# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check the CMS entry path as well as the repeated loop's partial VMEM waits."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit


def _assert_prefetch_ready(assembly, loads_a):
    # Keep the last prefetched set pending, just as the nonzero prologue wait
    # permits. The first loop iteration must retire it before next-bank reads.
    prefetch = assembly.split("// PGR=2 but only 1 loop", 2)[2]
    prefetch = prefetch.split("label_skipPGR2_1:", 1)[0]
    load_pattern = r"^buffer_load_.*vgprGlobalReadOffset([AB]).*\blds\b"
    initial = re.findall(load_pattern, prefetch, re.MULTILINE)
    assert len(initial) == loads_a + 4
    assert f"s_waitcnt vmcnt({loads_a + 4})" in assembly

    # Test both SIMD schedules. Unlike the steady-state CMS validator, seed
    # the queue from the actual prologue, not a previous main-loop iteration.
    paths = re.findall(
        r"^label_LoopBeginL_\d+:\n(.*?)(?=^s_cbranch_scc0 label_LoopBeginL_\d+)",
        assembly, re.MULTILINE | re.DOTALL,
    )
    assert len(paths) == 2
    for body in paths:
        pending = [(tensor, True) for tensor in initial]
        swapped = set()
        checked = set()
        for line in body.splitlines():
            load = re.match(load_pattern, line)
            if load:
                pending.append((load[1], False))
            wait = re.match(r"s_waitcnt .*vmcnt\((\d+)\)", line)
            if wait:
                count = int(wait[1])
                pending = pending[-count:] if count else []
            swap = re.match(r"v_xor_b32 v\[vgprLocalReadAddr([AB])", line)
            if swap:
                swapped.add(swap[1])
            read = re.match(r"ds_read_.*v\[vgprLocalReadAddr([AB])", line)
            if read and read[1] in swapped:
                tensor = read[1]
                assert (tensor, True) not in pending, (
                    f"prefetched {tensor} is still pending at {line}: {pending}"
                )
                checked.add(tensor)
        assert checked == {"A", "B"}


@pytest.mark.parametrize("tile_m", [192, 256])
@pytest.mark.parametrize("stream_k", [0, 3])
def test_tf32_dtl_prefetch_matches_loop_waits(tile_m, stream_k, tmp_path):
    config = yaml.safe_load(
        (Path(__file__).parent / "test_data" / "tf32_dtl_prefetch.yaml").read_text()
    )
    params = config["BenchmarkProblems"][0][1]
    params["ForkParameters"].extend([
        {"MatrixInstruction": [[16, 16, 32, 1, 1, tile_m // 32, 4, 2, 2]]},
        {"StreamK": [stream_k]},
    ])
    path = tmp_path / "tf32.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx950")
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    _assert_prefetch_ready(assembly, tile_m // 32)
