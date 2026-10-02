# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural tests for SoftwarePipeline.run_ping_pong_dynamic.

The dynamic ping-pong walks a *runtime* reduction extent, so the loop it emits
cannot be checked by counting unrolled iterations the way the compile-time
variant can. These tests build tiny kernels with recording stub callbacks and
assert on the emitted op structure instead:

  - the scf.for bounds: lower = k_lo (or a const 0), upper = k_extent, step =
    the const 2*block_k
  - one prologue issue at k_first, then exactly two issues and two computes
    inside the loop body, bound to the right offsets and buffers
  - the k_zero_fill select exists iff k_zero_fill is given, with operands
    (cmp_lt(k1, k_extent), k1, k_zero_fill), and it feeds the phase-A issue
  - the drain (s_waitcnt vmcnt=0 + barrier) directly after the loop
  - the ValueError guards

CPU-only; no GPU, torch or lowering involved. The C++ twin is byte-compared by
the dynamic_helpers parity family (tests/instances/parity/).
"""

from __future__ import annotations

import pytest

from rocke.core.ir import F16, F32, I32, IRBuilder
from rocke.helpers.pipeline import SoftwarePipeline
from rocke.helpers.schedule import SchedulePolicy


BLOCK_K = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(region, out):
    for op in region.ops:
        out.append(op)
        for r in op.regions:
            _flatten(r, out)
    return out


def _const_value(v):
    """The int an arith.constant Value holds, or None if v is not a constant."""
    op = v.op
    if op is None or op.name != "arith.constant":
        return None
    return op.attrs["value"]


class _Recorder:
    """Stub issue/compute callbacks that remember what they were called with.

    issue emits one ``arith.xor`` (k ^ marker) and compute one ``arith.smax``
    per state value, so each call leaves exactly one identifiable op behind
    and the tests can find it by identity rather than by position.
    """

    def __init__(self, b, marker):
        self.b = b
        self.marker = marker
        self.issues = []  # (k, buf, op)
        self.computes = []  # (k, buf, op)

    def issue(self, k, buf):
        v = self.b.xor(k, self.marker)
        self.issues.append((k, buf, v.op))

    def compute(self, k, buf, state):
        out = [self.b.smax(s, k) for s in state]
        self.computes.append((k, buf, out[0].op))
        return out


def _build(
    *,
    with_lo=False,
    with_zero_fill=False,
    schedule=None,
    pipe=None,
    n_state=1,
    mask=False,
):
    b = IRBuilder("pp_dyn_test")
    b.kernel.attrs["max_workgroup_size"] = 256
    k_extent = b.param("K", I32)
    k_lo = b.param("k_lo", I32) if with_lo else None
    k_zero = b.param("k_zero", I32) if with_zero_fill else None
    marker = b.param("marker", I32)
    bufs = [
        (
            b.smem_alloc(F16, [64], name_hint=f"a{i}"),
            b.smem_alloc(F16, [64], name_hint=f"b{i}"),
        )
        for i in range(2)
    ]
    inits = [b.const_i32(0) for _ in range(n_state)]
    rec = _Recorder(b, marker)
    pipe = pipe or SoftwarePipeline(num_iters=0)
    n_top_before = len(b.kernel.body.ops)
    results = pipe.run_ping_pong_dynamic(
        b,
        k_extent=k_extent,
        block_k=BLOCK_K,
        k_lo=k_lo,
        k_zero_fill=k_zero,
        mask_tail_state=mask,
        buffers=bufs,
        iter_args=[(f"s{i}", v) for i, v in enumerate(inits)],
        issue_load_fn=rec.issue,
        compute_fn=rec.compute,
        schedule=schedule,
    )
    new_top = b.kernel.body.ops[n_top_before:]
    for_ops = [op for op in new_top if op.name == "scf.for"]
    assert len(for_ops) == 1, [op.name for op in new_top]
    return {
        "b": b,
        "k_extent": k_extent,
        "k_lo": k_lo,
        "k_zero": k_zero,
        "bufs": bufs,
        "rec": rec,
        "results": results,
        "new_top": new_top,
        "for_op": for_ops[0],
    }


def _body(ctx):
    return ctx["for_op"].regions[0].ops


def _iv_adds(ctx):
    """The k1 = k + bk and k2 = k + 2bk adds, in emission order."""
    for_op = ctx["for_op"]
    iv_name = for_op.attrs["iv"]
    return [
        op
        for op in _body(ctx)
        if op.name == "arith.add" and op.operands[0].name == iv_name
    ]


# ---------------------------------------------------------------------------
# Loop bounds / step
# ---------------------------------------------------------------------------


class TestLoopShape:
    def test_step_is_two_block_k_const(self):
        ctx = _build()
        step = ctx["for_op"].operands[2]
        assert _const_value(step) == 2 * BLOCK_K

    def test_upper_bound_is_k_extent(self):
        ctx = _build()
        assert ctx["for_op"].operands[1] is ctx["k_extent"]

    def test_lower_bound_defaults_to_const_zero(self):
        ctx = _build(with_lo=False)
        assert _const_value(ctx["for_op"].operands[0]) == 0

    def test_lower_bound_is_k_lo(self):
        ctx = _build(with_lo=True)
        assert ctx["for_op"].operands[0] is ctx["k_lo"]

    def test_iv_is_named_k_pipe(self):
        ctx = _build()
        assert ctx["for_op"].attrs["iv"] == "%k_pipe"

    def test_iter_args_are_threaded_through(self):
        ctx = _build(n_state=2)
        for_op = ctx["for_op"]
        assert for_op.attrs["num_iter_args"] == 2
        assert len(ctx["results"]) == 2
        assert ctx["results"] == list(for_op.results)
        yields = [op for op in _body(ctx) if op.name == "scf.yield"]
        assert len(yields) == 1 and _body(ctx)[-1] is yields[0]
        assert len(yields[0].operands) == 2

    def test_k1_k2_are_iv_plus_bk_and_2bk(self):
        ctx = _build()
        adds = _iv_adds(ctx)
        assert len(adds) == 2
        assert _const_value(adds[0].operands[1]) == BLOCK_K
        # k2 reuses the step constant rather than materializing a new one.
        assert adds[1].operands[1] is ctx["for_op"].operands[2]


# ---------------------------------------------------------------------------
# Issue / compute placement
# ---------------------------------------------------------------------------


class TestIssueCompute:
    @pytest.mark.parametrize("with_lo", [False, True])
    def test_prologue_issue_uses_k_first_into_buf0(self, with_lo):
        ctx = _build(with_lo=with_lo)
        k, buf, op = ctx["rec"].issues[0]
        assert k is ctx["for_op"].operands[0]
        if with_lo:
            assert k is ctx["k_lo"]
        else:
            assert _const_value(k) == 0
        assert buf is ctx["bufs"][0]
        # Emitted before the loop, at the top level.
        assert op in ctx["new_top"]
        assert ctx["new_top"].index(op) < ctx["new_top"].index(ctx["for_op"])

    def test_body_has_two_issues_and_two_computes(self):
        ctx = _build()
        rec = ctx["rec"]
        assert len(rec.issues) == 3  # prologue + 2 phases
        assert len(rec.computes) == 2
        body_ids = {id(op) for op in _body(ctx)}
        in_body_issues = [i for i in rec.issues if id(i[2]) in body_ids]
        in_body_computes = [c for c in rec.computes if id(c[2]) in body_ids]
        assert len(in_body_issues) == 2
        assert len(in_body_computes) == 2
        # Nothing else in the loop issues or computes behind the helper's back.
        flat = _flatten(ctx["for_op"].regions[0], [])
        assert sum(op.name == "arith.xor" for op in flat) == 2

    def test_phase_offsets_and_buffers(self):
        ctx = _build()
        rec = ctx["rec"]
        adds = _iv_adds(ctx)
        k1, k2 = adds[0].results[0], adds[1].results[0]
        buf0, buf1 = ctx["bufs"]
        (_, _, _), (ia_k, ia_buf, _), (ib_k, ib_buf, _) = rec.issues
        (ca_k, ca_buf, _), (cb_k, cb_buf, _) = rec.computes
        # Phase A: prefetch k1 into buf1, compute k out of buf0.
        assert ia_k is k1 and ia_buf is buf1
        assert ca_k.name == ctx["for_op"].attrs["iv"] and ca_buf is buf0
        # Phase B: prefetch k2 into buf0, compute k1 out of buf1.
        assert ib_k is k2 and ib_buf is buf0
        assert cb_k is k1 and cb_buf is buf1

    def test_each_phase_issues_before_it_computes(self):
        ctx = _build()
        body = _body(ctx)
        rec = ctx["rec"]
        ia, ib = rec.issues[1][2], rec.issues[2][2]
        ca, cb = rec.computes[0][2], rec.computes[1][2]
        pos = {id(op): i for i, op in enumerate(body)}
        assert pos[id(ia)] < pos[id(ca)] < pos[id(ib)] < pos[id(cb)]

    def test_compute_state_chains_between_phases(self):
        ctx = _build()
        rec = ctx["rec"]
        a_out = rec.computes[0][2].results[0]
        b_op = rec.computes[1][2]
        # Phase B consumes phase A's state, and the yield returns phase B's.
        assert b_op.operands[0] is a_out
        yield_op = _body(ctx)[-1]
        assert yield_op.operands[0] is b_op.results[0]


# ---------------------------------------------------------------------------
# k_zero_fill select
# ---------------------------------------------------------------------------


class TestZeroFill:
    def test_no_select_without_zero_fill(self):
        ctx = _build(with_zero_fill=False)
        flat = _flatten(ctx["for_op"].regions[0], [])
        assert not any(op.name == "arith.select" for op in flat)
        assert not any(op.name == "arith.cmp" for op in flat)
        # Phase A prefetches k1 directly.
        k1 = _iv_adds(ctx)[0].results[0]
        assert ctx["rec"].issues[1][0] is k1

    @pytest.mark.parametrize("with_lo", [False, True])
    def test_select_feeds_phase_a_issue(self, with_lo):
        ctx = _build(with_zero_fill=True, with_lo=with_lo)
        body = _body(ctx)
        selects = [op for op in body if op.name == "arith.select"]
        assert len(selects) == 1
        sel = selects[0]
        k1 = _iv_adds(ctx)[0].results[0]
        cond, lhs, rhs = sel.operands
        assert lhs is k1
        assert rhs is ctx["k_zero"]
        cmp_op = cond.op
        assert cmp_op.name == "arith.cmp" and cmp_op.attrs["pred"] == "lt"
        assert cmp_op.operands[0] is k1
        assert cmp_op.operands[1] is ctx["k_extent"]
        # The select is what phase A prefetches; phase B still uses plain k2.
        assert ctx["rec"].issues[1][0] is sel.results[0]
        k2 = _iv_adds(ctx)[1].results[0]
        assert ctx["rec"].issues[2][0] is k2
        # Compute B keeps the unredirected k1 offset.
        assert ctx["rec"].computes[1][0] is k1

    @pytest.mark.parametrize("with_zero_fill", [False, True])
    def test_mask_tail_state_gates_phase_b(self, with_zero_fill):
        ctx = _build(with_zero_fill=with_zero_fill, mask=True, n_state=2)
        body = _body(ctx)
        k1 = _iv_adds(ctx)[0].results[0]
        cmps = [op for op in body if op.name == "arith.cmp"]
        # One compare, shared by the zero-fill select when there is one.
        assert len(cmps) == 1
        assert cmps[0].operands[0] is k1
        assert cmps[0].operands[1] is ctx["k_extent"]
        cond = cmps[0].results[0]
        # Phase B's new state is committed only when tile k+1 exists; the
        # yield carries select(k1 < K, phase-B state, phase-A state).
        comp_a, comp_b = ctx["rec"].computes
        yield_op = body[-1]
        assert yield_op.name == "scf.yield"
        for i, v in enumerate(yield_op.operands):
            sel = v.op
            assert sel.name == "arith.select"
            c, new, old = sel.operands
            assert c is cond
            assert new.op.name == "arith.smax" and new.op.operands[1] is k1
            assert old.op.name == "arith.smax"
            assert old.op.operands[1] is not k1
        assert comp_b[2] is yield_op.operands[0].op.operands[1].op

    def test_no_mask_yields_phase_b_state_directly(self):
        ctx = _build(mask=False)
        yield_op = _body(ctx)[-1]
        assert yield_op.operands[0].op is ctx["rec"].computes[1][2]

    def test_select_emitted_before_first_barrier(self):
        ctx = _build(with_zero_fill=True)
        names = [op.name for op in _body(ctx)]
        assert names.index("arith.select") < names.index("tile.sync")


# ---------------------------------------------------------------------------
# Barriers / waitcnt
# ---------------------------------------------------------------------------


class TestBarriers:
    def test_drain_follows_loop(self):
        ctx = _build()
        top = ctx["new_top"]
        i = top.index(ctx["for_op"])
        tail = top[i + 1 :]
        assert [op.name for op in tail] == ["tile.s_waitcnt", "tile.sync"]
        assert tail[0].attrs["vmcnt"] == 0

    def test_default_phase_barriers(self):
        # Defaults: sync_before_issue and sync_after_wait, no waitcnt.
        ctx = _build()
        names = [op.name for op in _body(ctx)]
        assert names.count("tile.sync") == 4
        assert "tile.s_waitcnt" not in names
        assert "tile.sync_lds_only" not in names

    def test_overlap_vmcnt_uses_partial_wait_and_lds_barriers(self):
        pipe = SoftwarePipeline(num_iters=0, wait_vmcnt=True, overlap_vmcnt=True)
        ctx = _build(pipe=pipe)
        body = _body(ctx)
        names = [op.name for op in body]
        assert names.count("tile.sync_lds_only") == 4
        assert "tile.sync" not in names
        waits = [op for op in body if op.name == "tile.s_waitcnt"]
        assert [w.attrs["vmcnt"] for w in waits] == [1, 1]
        # The post-loop drain is still a full vmcnt(0) + full barrier.
        top = ctx["new_top"]
        tail = top[top.index(ctx["for_op"]) + 1 :]
        assert [op.name for op in tail] == ["tile.s_waitcnt", "tile.sync"]
        assert tail[0].attrs["vmcnt"] == 0

    def test_wait_without_overlap_drains_fully(self):
        pipe = SoftwarePipeline(num_iters=0, wait_vmcnt=True)
        ctx = _build(pipe=pipe)
        waits = [op for op in _body(ctx) if op.name == "tile.s_waitcnt"]
        assert [w.attrs["vmcnt"] for w in waits] == [0, 0]

    def test_no_barriers_when_disabled(self):
        pipe = SoftwarePipeline(
            num_iters=0, sync_before_issue=False, sync_after_wait=False
        )
        ctx = _build(pipe=pipe)
        names = [op.name for op in _body(ctx)]
        assert "tile.sync" not in names and "tile.sync_lds_only" not in names

    def test_interwave_schedule_brackets_each_compute(self):
        sched = SchedulePolicy.for_pipeline("interwave")
        ctx = _build(schedule=sched)
        body = _body(ctx)
        prios = [
            (i, op.attrs["level"])
            for i, op in enumerate(body)
            if op.name == "tile.s_setprio"
        ]
        assert [lvl for _, lvl in prios] == [
            sched.compute_high_prio,
            sched.compute_low_prio,
        ] * 2
        pos = {id(op): i for i, op in enumerate(body)}
        for n, (_, _, cop) in enumerate(ctx["rec"].computes):
            hi_at, lo_at = prios[2 * n][0], prios[2 * n + 1][0]
            assert hi_at < pos[id(cop)] < lo_at

    def test_non_interwave_schedule_emits_nothing(self):
        ctx = _build(schedule=SchedulePolicy.for_pipeline("mem"))
        assert not any(op.name == "tile.s_setprio" for op in _body(ctx))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def _call(self, *, buffers, block_k):
        b = IRBuilder("pp_dyn_err")
        k = b.param("K", I32)
        init = b.const_f32(0.0)
        n_ops = len(b.kernel.body.ops)
        with pytest.raises(ValueError) as ei:
            SoftwarePipeline(num_iters=0).run_ping_pong_dynamic(
                b,
                k_extent=k,
                block_k=block_k,
                buffers=buffers,
                iter_args=[("acc", init)],
                issue_load_fn=lambda k, buf: None,
                compute_fn=lambda k, buf, s: s,
            )
        # Validation happens before anything is emitted.
        assert len(b.kernel.body.ops) == n_ops
        assert init.type is F32
        return str(ei.value)

    def test_one_buffer_pair_rejected(self):
        msg = self._call(buffers=[(None, None)], block_k=BLOCK_K)
        assert "2 buffer pairs" in msg

    def test_three_buffer_pairs_rejected(self):
        # Only the 2-buffer rotation exists; a third pair must not be ignored.
        msg = self._call(buffers=[(None, None)] * 3, block_k=BLOCK_K)
        assert "exactly 2 buffer pairs" in msg

    def test_no_buffers_rejected(self):
        msg = self._call(buffers=[], block_k=BLOCK_K)
        assert "2 buffer pairs" in msg

    @pytest.mark.parametrize("block_k", [0, -32])
    def test_non_positive_block_k_rejected(self, block_k):
        msg = self._call(buffers=[(None, None), (None, None)], block_k=block_k)
        assert "block_k must be positive" in msg


def test_run_ping_pong_dynamic_accepts_no_loop_carried_state():
    """With no iter_args the loop still builds (the C++ twin accepts 0 too)."""
    from rocke.core.ir import I32, IRBuilder
    from rocke.helpers.pipeline import SoftwarePipeline

    b = IRBuilder("pp_no_state")
    k_extent = b.param("K", I32)
    issued = []
    results = SoftwarePipeline(num_iters=1, double_buffer=True).run_ping_pong_dynamic(
        b,
        k_extent=k_extent,
        block_k=16,
        buffers=[(None, None), (None, None)],
        iter_args=[],
        issue_load_fn=lambda k, buf: issued.append(k),
        compute_fn=lambda k, buf, state: state,
    )
    assert results == []
    assert len(issued) == 3  # prologue + one per phase
