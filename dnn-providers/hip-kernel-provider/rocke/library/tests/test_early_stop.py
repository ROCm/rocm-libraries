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
