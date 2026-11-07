# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""HIP-graph capture/replay of a hipDNN plan.

torch.cuda.graph() captures on its own side stream, so the first captured call
re-binds the hipDNN handle from the eager stream to the capture stream inside
the capture. Replays then run with new buffer contents, and eager calls after
the capture re-bind back. Every result must match the reference.
"""

import pytest
import torch

from hipdnn_vllm import stats

from .test_rms_norm import TOL, ref_rms_norm

EPS = 1e-6


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=str)
def test_capture_replay_with_changed_inputs(C, dtype):
    torch.manual_seed(0)
    rows, hidden = 64, 4096
    x = torch.randn(rows, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")

    # Eager warm-up on the default stream: builds the plan, binds that stream.
    torch.testing.assert_close(torch.ops.hipdnn_vllm.rms_norm(x, w, EPS), ref_rms_norm(x, w, EPS), **TOL[dtype])
    torch.cuda.synchronize()

    stats.reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = torch.ops.hipdnn_vllm.rms_norm(x, w, EPS)
    snap = C.stats_snapshot()
    captured = [r for r in snap["counters"] if r["op"] == "rms_norm" and r["phase"] == "capture"]
    assert captured and captured[0]["route"] == "hipdnn", snap["counters"]
    assert snap["runtime"]["plan_builds"] == 0  # capture reused the eager plan

    for step in range(5):
        x.copy_(torch.randn_like(x) * (step + 1))
        w.copy_(torch.randn_like(w))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, ref_rms_norm(x, w, EPS), **TOL[dtype])

    # Replays run no host code, so they add no counts.
    after = [r for r in C.stats_snapshot()["counters"] if r["op"] == "rms_norm"]
    assert sum(r["count"] for r in after) == sum(r["count"] for r in captured)

    # Eager again on the default stream (re-bind back) still correct.
    x2 = torch.randn_like(x)
    torch.testing.assert_close(torch.ops.hipdnn_vllm.rms_norm(x2, w, EPS), ref_rms_norm(x2, w, EPS), **TOL[dtype])


def test_two_graphs_share_one_plan_with_different_pointers(C):
    """One cached plan, captured twice against different buffers; replaying
    the graphs alternately must keep each graph's own pointers."""
    dtype = torch.bfloat16
    xs = [torch.randn(16, 2048, dtype=dtype, device="cuda") for _ in range(2)]
    w = torch.randn(2048, dtype=dtype, device="cuda")
    torch.ops.hipdnn_vllm.rms_norm(xs[0], w, EPS)
    torch.cuda.synchronize()
    builds = C.stats_snapshot()["runtime"]["plan_builds"]

    graphs, outs = [], []
    pool = torch.cuda.graph_pool_handle()
    for x in xs:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            outs.append(torch.ops.hipdnn_vllm.rms_norm(x, w, EPS))
        graphs.append(g)
    assert C.stats_snapshot()["runtime"]["plan_builds"] == builds

    for _ in range(3):
        for i in (1, 0):
            xs[i].copy_(torch.randn_like(xs[i]))
            graphs[i].replay()
        torch.cuda.synchronize()
        for i in (0, 1):
            torch.testing.assert_close(outs[i], ref_rms_norm(xs[i], w, EPS), **TOL[dtype])


def test_capture_on_user_stream(C):
    """Capture on an explicit non-default stream, eager on another: each call
    binds its own stream, and nothing syncs or allocates illegally in capture."""
    dtype = torch.float16
    x = torch.randn(8, 1024, dtype=dtype, device="cuda")
    w = torch.randn(1024, dtype=dtype, device="cuda")
    s_eager = torch.cuda.Stream()
    s_cap = torch.cuda.Stream()
    with torch.cuda.stream(s_eager):
        torch.ops.hipdnn_vllm.rms_norm(x, w, EPS)
    s_eager.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s_cap):
        y = torch.ops.hipdnn_vllm.rms_norm(x, w, EPS)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref_rms_norm(x, w, EPS), **TOL[dtype])


def _rms_counts(C):
    out = {}
    for r in C.stats_snapshot()["counters"]:
        if r["op"] == "rms_norm":
            key = (r["route"], r["reason"], r["phase"])
            out[key] = out.get(key, 0) + r["count"]
    return out


def test_captured_plan_survives_cache_eviction(C):
    """Replays never touch the plan cache, so a plan fetched under capture is
    pinned: clearing or shrinking the LRU must not release it while the graph
    can still replay its launches."""
    dtype = torch.float16
    hidden = 2048
    x = torch.randn(48, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    torch.ops.hipdnn_vllm.rms_norm(x, w, EPS)  # eager build
    torch.cuda.synchronize()
    pinned0 = C.stats_snapshot()["runtime"]["plan_cache_pinned"]

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = torch.ops.hipdnn_vllm.rms_norm(x, w, EPS)
    assert C.stats_snapshot()["runtime"]["plan_cache_pinned"] == pinned0 + 1

    try:
        C.plan_cache_clear()
        C.set_plan_cache_size(1)
        for rows in (3, 5, 7):  # churn the one LRU slot
            torch.ops.hipdnn_vllm.rms_norm(torch.randn(rows, hidden, dtype=dtype, device="cuda"), w, EPS)
        rt = C.stats_snapshot()["runtime"]
        assert rt["plan_cache_size"] == 1, rt
        assert rt["plan_cache_pinned"] == pinned0 + 1, rt
        for step in range(3):
            x.copy_(torch.randn_like(x) * (step + 1))
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(y, ref_rms_norm(x, w, EPS), **TOL[dtype])
    finally:
        C.set_plan_cache_size(4096)


def test_capture_miss_falls_back_without_building_and_is_not_cached(C):
    """A signature first met under capture builds nothing: the call falls back
    with CAPTURE_INVARIANT. That decline is not cached, so the next eager call
    builds the plans and a later capture runs on hipDNN."""
    dtype = torch.bfloat16
    hidden = 1792
    x = torch.randn(5, hidden, dtype=dtype, device="cuda")
    w = torch.randn(hidden, dtype=dtype, device="cuda")
    C.norm_rows_reset()
    C.norm_set_exact_rows([1])

    def builds():  # plan builds since the last stats.reset()
        return C.stats_snapshot()["runtime"]["plan_builds"]

    try:
        stats.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, False)
        assert builds() == 0
        assert _rms_counts(C) == {("fallback", "CAPTURE_INVARIANT", "capture"): 1}, _rms_counts(C)
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, ref_rms_norm(x, w, EPS), **TOL[dtype])

        # Eager: builds (the capture-time miss was not cached as a decline).
        stats.reset()
        torch.testing.assert_close(
            torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, False), ref_rms_norm(x, w, EPS), **TOL[dtype]
        )
        assert builds() > 0
        assert _rms_counts(C) == {("hipdnn", "", "eager"): 2}, _rms_counts(C)  # 5 rows = 4 + 1

        stats.reset()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            y2 = torch.ops.hipdnn_vllm.rms_norm_rows(x, w, EPS, False)
        assert builds() == 0
        assert _rms_counts(C) == {("hipdnn", "", "capture"): 2}, _rms_counts(C)
        x.copy_(torch.randn_like(x))
        g2.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y2, ref_rms_norm(x, w, EPS), **TOL[dtype])
    finally:
        C.norm_rows_reset()
