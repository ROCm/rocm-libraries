# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 A2A+GEMM (FusedA2A: A2AGemm) codegen characterization.

Checks that a FusedA2A: A2AGemm solution emits the shard loop, the enqueuer
election, the SDMA ring instructions and the A2ASubRounds-dependent round
count, flag-slot and sub-round instructions, and that the result assembles for
gfx950.
"""

import math
import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_COMM = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # characterization/
        os.pardir,  # unit/
        os.pardir,  # tensilelite/Tests/
        "common", "comm", "gfx950",
    )
)
_CONFIGS = [os.path.join(_COMM, c) for c in ("a2a_gemm_loop.yaml", "a2a_gemm_loop_s2.yaml")]
_SUB_ROUNDS = {"a2a_gemm_loop.yaml": 1, "a2a_gemm_loop_s2.yaml": 2}

_A2A_GEMM_MARKERS = (
    "label_A2AShardLoopBegin:",
    "label_A2AWaitFlag:",
    "label_A2ATransitionEnd:",
    "label_A2ASkipEnqueue:",
    "s_atomic_inc",
    "s_atomic_umax_x2",
)


def _sub_round_lines(subRounds):
    """Full-line patterns of the A2ASubRounds-dependent instructions -> expected count."""
    split = subRounds > 1
    lines = {
        r"s_mov_b32 s\[sgprA2AShardCounter\], s\[sgprA2AW\] +// rounds = W": int(not split),
        rf"s_lshl_b32 s\[sgprA2AShardCounter\], s\[sgprA2AShardCounter\], {int(math.log2(subRounds))} +// \(W-1\) \* S": int(split),
        r"s_add_u32 s\[sgprA2AShardCounter\], s\[sgprA2AShardCounter\], 1 +// rounds = 1 \+ \(W-1\) \* S": int(split),
        r"s_lshl_b32 (s\d+), \1, 2 +// flag slot byte offset": subRounds + 1,
        rf"s_lshl_b32 (s\d+), \1, {int(math.log2(subRounds))} +// \(bLast\*\(W-1\) \+ i\) \* S": subRounds if split else 0,
        rf"s_lshl_b32 (s\d+), \1, {int(math.log2(subRounds))} +// slot \* S": int(split),
        r"s_add_u32 (s\d+), \1, s\d+ +// \+ sub-round": int(split),
        rf"s_and_b32 (s\d+), \1, {subRounds - 1} +// sub-round = \(1 - rounds left\) mod S": 4 if split else 0,
    }
    for h in range(1, subRounds):
        lines[rf"s_add_u32 (s\d+), \1, {h} +// \+ sub-round"] = 1
    return lines


@pytest.mark.parametrize("config", _CONFIGS, ids=os.path.basename)
def test_a2a_gemm_gfx950_emits_assembly(config):
    results = emit_kernels_from_config(config, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), [
        (b, e) for (b, _s, e) in results if e != 0
    ]
    expected = _sub_round_lines(_SUB_ROUNDS[os.path.basename(config)])
    for base, src, _err in results:
        assert_assembles(src, base)
        missing = [m for m in _A2A_GEMM_MARKERS if m not in src]
        assert not missing, f"Kernel {base!r}: missing A2AGemm markers {missing}"
        lines = src.splitlines()
        counts = {p: sum(1 for line in lines if re.fullmatch(p, line)) for p in expected}
        assert counts == expected, f"Kernel {base!r}: A2ASubRounds lines {counts} != {expected}"


def test_a2a_gemm_gfx950_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for every emitted A2AGemm kernel."""
    digest = sorted(
        (
            {"basename": b, "err": e}
            for config in _CONFIGS
            for (b, _s, e) in emit_kernels_from_config(config, limit=8, arch=_ARCH)
        ),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
