# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math

import pytest

from lib import evaluate as ev
from lib import features as fs
from lib import hipblaslt_launch as hl
from lib import identity_ceiling as ic

BASE = dict(
    mt_m=128, mt_n=128, mt_k=64, mi_m=16, mi_n=16, mi_k=32, occupancy=1,
    cache_hints_a=0, cache_hints_b=0, grvw_a=8, grvw_b=8, gwvw_d=4,
)  # fmt: skip
PLAIN = {"tile_processing_strategy": 0, "work_assignment": 0}
HYBRID = {"tile_processing_strategy": 2, "work_assignment": 2, "work_group_x": 256}
# Solution 0 and 1 share signature A in two launch classes; 3 is a copy of 0
# in another vector width; 2 has signature B.
KERNELS = {
    0: (dict(BASE), PLAIN),
    1: (dict(BASE), HYBRID),
    2: (dict(BASE, mt_n=64), PLAIN),
    3: (dict(BASE, grvw_a=4), PLAIN),
}


def gemm(m, latencies, screened=(), warmup=None, **extra):
    """`latencies` timed (warm-up only for `screened`), plus warm-up rows
    `warmup`."""
    rows = [(sid, us, sid in screened) for sid, us in latencies.items()]
    rows += [(sid, us, True) for sid, us in (warmup or {}).items()]
    cands = []
    for sid, us, skip in rows:
        cfg, attrs = KERNELS[sid]
        cands.append(dict(cfg, sol_idx_global=sid, us=us, is_skip=skip,
                          attributes=dict(attrs)))  # fmt: skip
    g = dict(m=m, n=4096, k=4096, batch_count=1, transA="N", transB="T",
             a_type="bf16_r", b_type="bf16_r", c_type="bf16_r", d_type="bf16_r",
             compute_type="f32_r", candidates=cands)  # fmt: skip
    g.update(extra)
    return g


GEMMS = [
    gemm(1, {0: 10.0, 1: 5.0, 2: 8.0, 3: 10.0}),
    # Solution 0 not admitted: signature A runs solution 1.
    gemm(2, {1: 6.0, 2: 7.0, 3: 9.0}),
    # Warm-up rows only.
    gemm(3, {0: 4.0, 1: 2.0}, screened=(0, 1)),
    # A timed row wins over a warm-up row of the same solution.
    gemm(4, {0: 12.0, 2: 13.0}, warmup={0: 3.0}),
    gemm(5, {}),
]


def test_signature_ceiling():
    c = ic.ceiling(GEMMS, ic.signature_grouping, "signature")
    assert (c.n_kernels, c.n_groups, c.n_shared, c.max_group) == (4, 2, 1, 3)
    eff = {k[0]: v for k, v in c.per_gemm.items()}
    assert eff == {1: 5.0 / 8.0, 2: 1.0, 3: 2.0 / 4.0, 4: 1.0}
    assert c.geomean == pytest.approx(math.prod(eff.values()) ** 0.25)
    assert c.n_losing == 2
    assert c.to_dict()["n_gemms"] == 4


def test_engine_ceiling_splits_launch_classes():
    launch = hl.HipblasltLaunch("gfx950")
    both = ic.engine_and_signature(GEMMS, launch=launch)
    engine = both["engine"]
    assert (engine.n_groups, engine.n_shared, engine.max_group) == (3, 1, 2)
    eff = {k[0]: v for k, v in engine.per_gemm.items()}
    assert eff == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    assert engine.geomean > both["signature"].geomean
    # Without a launch model every kernel of a signature launches alike.
    assert ic.ceiling(GEMMS, ic.engine_grouping()).per_gemm == (
        both["signature"].per_gemm
    )
    # A declared attribute splits groups like the engine's view key.
    spec = (fs.AttributeFeature("work_assignment"),)
    split = ic.ceiling(GEMMS, ic.engine_grouping(spec))
    assert split.n_groups == 3


def test_pool_order_decides_the_member_that_runs():
    pools = ev.build_pools(GEMMS)
    reversed_pools = {k: list(reversed(v)) for k, v in pools.items()}
    c = ic.ceiling(GEMMS[:1], ic.signature_grouping, pools=reversed_pools)
    # Solution 3 runs first for signature A: 10 us against 8 us of B.
    assert list(c.per_gemm.values()) == [5.0 / 8.0]
    one = ic.ceiling(GEMMS[:2], lambda k: 0, pools=reversed_pools)
    # One group runs its first measured kernel in pool order.
    assert [v for v in one.per_gemm.values()] == [5.0 / 10.0, 6.0 / 9.0]
    every = ic.ceiling(GEMMS, lambda k: k["sol_idx_global"])
    assert every.geomean == 1.0 and every.n_losing == 0


def test_contexts_and_pools_are_separate():
    other = gemm(1, {0: 10.0, 2: 5.0}, cu_budget=64, schedule="dynamic")
    fp16 = gemm(1, {1: 3.0}, a_type="f16_r", b_type="f16_r")
    c = ic.ceiling(GEMMS[:1] + [other, fp16], ic.signature_grouping)
    assert len(c.per_gemm) == 3
    assert c.n_kernels == 4 + 1
    assert sorted(c.per_gemm.values()) == [5.0 / 8.0, 1.0, 1.0]


def test_shipped_kernels_keep_the_full_best():
    # Without solution 2, GEMM 1 runs signature A's first kernel against the
    # best of every kernel measured.
    c = ic.ceiling(GEMMS[:2], ic.signature_grouping, kernels={0, 1, 3})
    assert list(c.per_gemm.values()) == [5.0 / 10.0, 6.0 / 6.0]
    assert (c.n_kernels, c.n_groups) == (3, 1)
    # Launch classes of a signature stay apart under the engine's grouping...
    engine = ic.engine_grouping(launch=hl.HipblasltLaunch("gfx950"))
    alone = ic.ceiling(GEMMS[:1], engine, kernels={1, 2})
    joined = ic.ceiling(GEMMS[:1], engine, kernels={0, 1, 2, 3})
    assert alone.per_gemm == joined.per_gemm
    # ...while a kernel joining a signature group ahead of its members can
    # lower the ceiling.
    assert ic.ceiling(GEMMS[:1], ic.signature_grouping, kernels={1, 2}).geomean == 1.0
    assert (
        ic.ceiling(GEMMS[:1], ic.signature_grouping, kernels={0, 1, 2}).geomean
        == 5.0 / 8.0
    )
    assert ic.ceiling(GEMMS, engine, kernels=()).geomean == 0.0
    both = ic.engine_and_signature(GEMMS[:1], launch=hl.HipblasltLaunch("gfx950"),
                                   kernels={0, 1, 2})  # fmt: skip
    assert both["engine"].geomean == 1.0 and both["signature"].geomean == 5.0 / 8.0
