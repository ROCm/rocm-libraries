# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 A2A+GEMM (FusedA2A: A2AGemm) codegen characterization.

Checks that a FusedA2A: A2AGemm solution emits the shard loop, the enqueuer
election, the SDMA ring instructions and the A2ASubRounds-dependent round
count, flag-slot and sub-round instructions, no tail loop, an LDS write-pointer
reset in the round transition and, unless both A and B are DirectToLds, one
barrier there, and that the result assembles for gfx950.
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
_CONFIG = os.path.join(_COMM, "a2a_gemm_loop.yaml")
_SUB_ROUNDS = re.compile(r"\.amdhsa_kernel \S*_AASR(\d+)_")

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


def _transition(src):
    start = src.index("/* A2A_TRANSITION begin */")
    return src[start:src.index("label_A2ATransitionEnd:", start)]


@pytest.mark.parametrize("group", [0, 1])
def test_a2a_gemm_gfx950_emits_assembly(group):
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH, size_group_index=group)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), [
        (b, e) for (b, _s, e) in results if e != 0
    ]
    for base, src, _err in results:
        assert_assembles(src, base)
        missing = [m for m in _A2A_GEMM_MARKERS if m not in src]
        assert not missing, f"Kernel {base!r}: missing A2AGemm markers {missing}"
        subRounds = _SUB_ROUNDS.search(src)
        assert subRounds, f"Kernel {base!r}: no AASR<n> in the kernel name"
        expected = _sub_round_lines(int(subRounds.group(1)))
        lines = src.splitlines()
        counts = {p: sum(1 for line in lines if re.fullmatch(p, line)) for p in expected}
        assert counts == expected, f"Kernel {base!r}: A2ASubRounds lines {counts} != {expected}"
        assert "label_SkipTailLoopL:" not in src, f"Kernel {base!r}: tail loop emitted"
        transition = _transition(src)
        for tc in "AB":
            reset = rf"^\w+ [sv]\[[sv]gprLocalWriteAddr{tc}\], .*$"
            assert re.search(reset, transition, re.M), \
                f"Kernel {base!r}: no LocalWriteAddr{tc} reset in the transition"
        dtl = "/* DirectToLdsA=True */" in src and "/* DirectToLdsB=True */" in src
        barriers = len(re.findall(r"^s_barrier\b", transition, re.M))
        assert barriers == (0 if dtl else 1), \
            f"Kernel {base!r}: {barriers} s_barrier in the transition, DirectToLds={dtl}"


@pytest.mark.parametrize("group", [0, 1])
def test_a2a_gemm_gfx950_golden(snapshot, group):
    """Order-invariant golden: pin {basename, err} for every emitted A2AGemm kernel."""
    digest = sorted(
        (
            {"basename": b, "err": e}
            for (b, _s, e) in emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH,
                                                       size_group_index=group)
        ),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
