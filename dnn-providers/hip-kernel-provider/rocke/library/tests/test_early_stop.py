# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""EarlyStop: a kernel whose warmup is FACTOR x off the best is not timed."""

import rocke.runtime as rt_mod

from benchmarks.common.early_stop import EarlyStop


def _fake_timer(monkeypatch, per_kernel_ms):
    calls = []

    def fake(fn, *, warmup, iters, stream=0):
        calls.append((fn, warmup, iters))
        return per_kernel_ms[fn]

    monkeypatch.setattr(rt_mod, "time_launches", fake)
    return calls


def test_skips_kernels_far_off_the_best(monkeypatch):
    fast, ok, slow = object(), object(), object()
    calls = _fake_timer(monkeypatch, {fast: 1.0, ok: 4.0, slow: 6.0})
    stop = EarlyStop(5.0, after=1)
    # The first kernel is always measured: warmup and timed loop in one call.
    assert stop.measure(fast, warmup=3, iters=10) == 1.0
    assert calls[-1] == (fast, 3, 10)
    # Within the factor: the warmup is timed separately, then the loop.
    assert stop.measure(ok, warmup=3, iters=10) == 4.0
    assert calls[-2:] == [(ok, 0, 3), (ok, 0, 10)]
    # Beyond it: only the warmup runs, and the kernel is reported.
    n = len(calls)
    assert stop.measure(slow, warmup=3, iters=10) is None
    assert calls[n:] == [(slow, 0, 3)]
    assert stop.n_stopped == 1 and stop.best_ms == 1.0


def test_factor_zero_disables(monkeypatch):
    fast, slow = object(), object()
    _fake_timer(monkeypatch, {fast: 1.0, slow: 100.0})
    stop = EarlyStop(0, after=1)
    stop.measure(fast, warmup=3, iters=10)
    assert stop.measure(slow, warmup=3, iters=10) == 100.0
    assert stop.n_stopped == 0


def test_waits_for_the_first_kernels(monkeypatch):
    fast, slow = object(), object()
    _fake_timer(monkeypatch, {fast: 1.0, slow: 100.0})
    stop = EarlyStop(5.0, after=3)
    stop.measure(fast, warmup=3, iters=10)
    # Kernels 2 and 3 are timed in full however slow they are...
    assert stop.measure(slow, warmup=3, iters=10) == 100.0
    assert stop.measure(slow, warmup=3, iters=10) == 100.0
    # ...from the fourth on early stopping applies.
    assert stop.measure(slow, warmup=3, iters=10) is None


def _reset_handoff(monkeypatch):
    import benchmarks.common.early_stop as es

    monkeypatch.setattr(es, "_seeds", None)
    monkeypatch.setattr(es, "_records", {})
    monkeypatch.setattr(es, "_record_path", None)
    return es


def test_direct_and_implicit_problems_share_a_case_key():
    """The two scripts' problem types meet on one key, so a seed carries over."""
    from kernels.common._conv_implicit_gemm_common import ConvProblem
    from kernels.common.conv_direct_grouped import DirectConvProblem

    from benchmarks.common.early_stop import case_key

    direct = DirectConvProblem(N=2, H=16, W=16, groups=4, cpg=8, kpg=8)
    igemm = ConvProblem(N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1, groups=4)
    assert case_key(direct, "fp16", "fwd") == case_key(igemm, "fp16", "fwd")
    assert case_key(direct, "fp16", "fwd") != case_key(igemm, "bf16", "fwd")
    assert case_key(direct, "fp16", "fwd") != case_key(igemm, "fp16", "dgrad")


def test_record_then_seed_hands_the_best_over(monkeypatch, tmp_path):
    """One run records its best; the next starts from it, from the first kernel."""
    from argparse import Namespace

    from kernels.common._conv_implicit_gemm_common import ConvProblem

    fast, slow, ok = object(), object(), object()
    _fake_timer(monkeypatch, {fast: 1.0, slow: 9.0, ok: 3.0})
    problem = ConvProblem(N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1)
    record = tmp_path / "best.json"

    es = _reset_handoff(monkeypatch)
    args = Namespace(
        early_stop=5.0, early_stop_after=100, early_stop_record=str(record)
    )
    first = es.EarlyStop.for_case(args, problem, "fp16", "fwd")
    first.measure(slow, warmup=3, iters=10)
    first.measure(fast, warmup=3, iters=10)
    assert record.is_file()

    es = _reset_handoff(monkeypatch)
    args = Namespace(early_stop=5.0, early_stop_after=100, early_stop_seed=str(record))
    second = es.EarlyStop.for_case(args, problem, "fp16", "fwd")
    assert second.best_ms == 1.0
    # Seeded: `after` does not hold the first kernels back.
    assert second.measure(slow, warmup=3, iters=10) is None
    assert second.measure(ok, warmup=3, iters=10) == 3.0
    # A case the seed does not know starts from nothing.
    other = es.EarlyStop.for_case(args, problem, "fp16", "dgrad")
    assert other.best_ms is None


def test_all_stopped_against_a_seed_says_so(monkeypatch):
    slow = object()
    _fake_timer(monkeypatch, {slow: 9.0})
    stop = EarlyStop(5.0, after=100, seed_ms=1.0)
    assert stop.measure(slow, warmup=3, iters=10) is None
    assert "all 1 early-stopped" in stop.summary()
    assert "1.000 ms" in stop.summary()


def test_failed_kernels_neither_prune_nor_record(monkeypatch, tmp_path):
    """A kernel that failed --verify is timed but cannot become the best: a
    wrong-but-fast one would otherwise prune every correct kernel, here and in
    the sweep seeded from the record."""
    import json
    from argparse import Namespace

    from kernels.common._conv_implicit_gemm_common import ConvProblem

    wrong, ok, slow = object(), object(), object()
    _fake_timer(monkeypatch, {wrong: 0.1, ok: 2.0, slow: 9.0})
    problem = ConvProblem(N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1)
    record = tmp_path / "best.json"

    es = _reset_handoff(monkeypatch)
    args = Namespace(early_stop=5.0, early_stop_after=1, early_stop_record=str(record))
    stop = es.EarlyStop.for_case(args, problem, "fp16", "fwd")
    assert stop.measure(wrong, warmup=3, iters=10, passed=False) == 0.1
    assert stop.best_ms is None and stop.n_measured == 0
    assert not record.exists()
    # The failed kernel did not count towards `after` either: this one is the
    # first eligible kernel and is measured in full.
    assert stop.measure(slow, warmup=3, iters=10, passed=True) == 9.0
    assert stop.measure(ok, warmup=3, iters=10) == 2.0
    assert stop.best_ms == 2.0
    assert list(json.loads(record.read_text()).values()) == [2.0]
