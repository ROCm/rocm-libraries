# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check that a 16-bit DTL custom-schedule loop's partial VMEM waits hold on entry from the prologue."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_LOAD = re.compile(r"^buffer_load_.*vgprGlobalReadOffset([AB]).*\blds\b")
_WAIT = re.compile(r"^s_waitcnt .*vmcnt\((\d+)\)")
_SWAP = re.compile(r"^v_xor_b32 v\[vgprLocalReadAddr([AB])")
_READ = re.compile(r"^ds_read_.*v\[vgprLocalReadAddr([AB])")


def _prefetched_set(assembly):
    # The second prefetched set is the one still pending when the loop starts.
    prefetch = assembly.split("// PGR=2 but only 1 loop", 2)[2]
    prefetch = prefetch.split("label_skipPGR2_1:", 1)[0]
    return [m[1] for m in (_LOAD.match(line) for line in prefetch.splitlines()) if m]


def _assert_prefetch_ready(assembly):
    initial = _prefetched_set(assembly)
    assert set(initial) == {"A", "B"}
    assert f"s_waitcnt vmcnt({len(initial)})" in assembly

    paths = re.findall(
        r"^label_LoopBeginL_\d+:\n(.*?)(?=^s_cbranch_scc0 label_LoopBeginL_\d+)",
        assembly, re.MULTILINE | re.DOTALL,
    )
    assert len(paths) == 2
    for body in paths:
        # Seed the in-order VMEM queue from the actual prologue rather than a
        # previous iteration, then require every read of a swapped buffer to
        # find its prefetched loads retired.
        pending = [(tensor, True) for tensor in initial]
        swapped = set()
        checked = set()
        for line in body.splitlines():
            if load := _LOAD.match(line):
                pending.append((load[1], False))
            if wait := _WAIT.match(line):
                count = int(wait[1])
                pending = pending[-count:] if count else []
            if swap := _SWAP.match(line):
                swapped.add(swap[1])
            read = _READ.match(line)
            if read and read[1] in swapped:
                tensor = read[1]
                assert (tensor, True) not in pending, (
                    f"prefetched {tensor} is still pending at {line}: {pending}"
                )
                checked.add(tensor)
        assert checked == {"A", "B"}


_TN_256x160 = dict(transpose=(True, False), tlds=1, ldstr=False,
                   mi=[16, 16, 32, 1, 1, 8, 5, 2, 2], size=[512, 320, 1, 512])
# Mirrors the 224x128 NT schedule with A and B switched.
_NT_128x224 = dict(transpose=(False, True), tlds=0, ldstr=True,
                   mi=[16, 16, 32, 1, 1, 4, 7, 2, 2], size=[256, 448, 1, 512])


@pytest.mark.parametrize("case, stream_k", [
    (_TN_256x160, 0),
    (_TN_256x160, 3),
    (_NT_128x224, 0),
], ids=["256x160_tn", "256x160_tn_sk3", "128x224_nt"])
def test_dtl_custom_schedule_prefetch_matches_loop_waits(case, stream_k, tmp_path):
    config = yaml.safe_load(
        (Path(__file__).parent / "test_data" / "dtl_custom_schedule_prefetch_order.yaml").read_text()
    )
    problem, params = config["BenchmarkProblems"][0]
    problem["TransposeA"], problem["TransposeB"] = case["transpose"]
    forks = params["ForkParameters"]
    for fork in forks:
        if "TransposeLDS" in fork:
            fork["TransposeLDS"] = [case["tlds"]]
        if "LDSTrInst" in fork:
            fork["LDSTrInst"] = [case["ldstr"]]
    forks.extend([
        {"MatrixInstruction": [case["mi"]]},
        {"StreamK": [stream_k]},
    ])
    params["BenchmarkFinalParameters"] = [{"ProblemSizes": [{"Exact": case["size"]}]}]
    path = tmp_path / "dtl_custom_schedule.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx950")
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    assert "_CMS_" in assembly
    _assert_prefetch_ready(assembly)
