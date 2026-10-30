# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Selection logic of benchmarks.common.batched_timing.rank, with the device
# stubbed out. What needs a GPU is the timing itself; what needs testing is
# what the ranking does with the times -- which candidates reach the
# full-fidelity round, which are dropped, and what happens to one that cannot
# be loaded or whose launch fails. Those are host decisions, so they are
# tested on the host: time_batch and prime_device are replaced by fakes that
# report a fixed time per candidate.

from __future__ import annotations

import pytest
from benchmarks.common import batched_timing


class _FakeDevice:
    """Stands in for the device: candidate ``i`` runs for ``times[i]`` ms.

    ``None`` in ``times`` is a candidate whose launch raises -- ``time_batch``
    reports that as a ``None`` time, so the fake does the same.
    """

    def __init__(self, times, *, probe_error: float = 1.0):
        self.times = list(times)
        self.probe_error = probe_error
        self.probed = 0
        self.measured = 0
        self.primed = 0
        self.batches = 0

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(batched_timing, "time_batch", self._time_batch)
        monkeypatch.setattr(batched_timing, "time_batch_staged", self._time_staged)
        monkeypatch.setattr(batched_timing, "prime_device", self._prime)

    def _time_staged(self, runs, *, warmup, iters, stream=0):
        # The finalists' round, in the default (staged) protocol.
        self.batches += 1
        out = []
        for run in runs:
            ms = self.times[run()]
            if ms is not None:
                self.measured += 1
            out.append(ms)
        return out

    def _prime(self, run, **_kwargs) -> None:
        self.primed += 1

    def _time_batch(self, runs, *, warmup, iters, stream=0, chunk=0):
        self.batches += 1
        probing = iters == batched_timing.DEFAULT_PROBE_ITERS
        out = []
        for run in runs:
            i = run()
            ms = self.times[i]
            if ms is None:
                out.append(None)
                continue
            if probing:
                self.probed += 1
                # A probe is a one- or two-launch sample: close, not exact.
                out.append(ms * self.probe_error)
            else:
                self.measured += 1
                out.append(ms)
        return out

    def make_run(self, i):
        return lambda i=i: i


def _rank(device, n, **kwargs):
    kwargs.setdefault("warmup", 3)
    kwargs.setdefault("iters", 10)
    return batched_timing.rank(n, device.make_run, **kwargs)


def test_only_the_fastest_reach_the_full_round(monkeypatch):
    times = [10.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0, 6.0, 5.0]
    device = _FakeDevice(times)
    device.install(monkeypatch)

    ranked = _rank(device, len(times), finalists=3, chunk=4)

    # Every candidate is probed once; only the three fastest are measured.
    assert device.probed == len(times)
    assert device.measured == 3
    assert sorted(r.index for r in ranked if r.is_finalist) == [1, 3, 5]
    # A finalist reports the full-fidelity time, never the probe's.
    assert [r.ms for r in ranked if r.index == 3] == [1.0]


def test_a_candidate_dropped_by_the_probe_keeps_its_probe_time(monkeypatch):
    times = [1.0, 5.0, 6.0]
    device = _FakeDevice(times, probe_error=1.1)
    device.install(monkeypatch)

    ranked = _rank(device, len(times), finalists=1, chunk=8)

    dropped = [r for r in ranked if not r.is_finalist]
    assert [r.index for r in dropped] == [1, 2]
    # It is reportable -- that is what the sweep prints for it -- but it is
    # the probe's number, so it is kept apart from ``ms``.
    assert all(r.probe_ms is not None and r.ms is None for r in dropped)
    assert dropped[0].probe_ms == pytest.approx(5.5)


def test_the_finalist_cut_spans_chunks(monkeypatch):
    # The fastest candidate sits in the last chunk, so a cut applied per chunk
    # rather than across the case would already have dropped it.
    times = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    device = _FakeDevice(times)
    device.install(monkeypatch)

    ranked = _rank(device, len(times), finalists=2, chunk=3)

    assert sorted(r.index for r in ranked if r.is_finalist) == [7, 8]


def test_a_candidate_that_cannot_be_loaded_is_marked_failed(monkeypatch):
    times = [4.0, 1.0, 2.0]
    device = _FakeDevice(times)
    device.install(monkeypatch)

    def make_run(i):
        return None if i == 1 else device.make_run(i)

    ranked = batched_timing.rank(
        len(times), make_run, warmup=3, iters=10, finalists=8, chunk=8
    )

    assert [r.index for r in ranked if r.failed] == [1]
    assert ranked[1].probe_ms is None and ranked[1].ms is None
    # The rest are unaffected: the load failure costs no launch at all.
    assert device.probed == 2


def test_a_launch_that_fails_never_ranks(monkeypatch):
    # ``None`` from time_batch is a candidate whose launch raised.
    times = [4.0, None, 2.0]
    device = _FakeDevice(times)
    device.install(monkeypatch)

    ranked = _rank(device, len(times), finalists=8, chunk=8)

    assert [r.index for r in ranked if r.failed] == [1]
    assert sorted(r.index for r in ranked if r.is_finalist) == [0, 2]


def test_the_device_is_primed_once_per_case(monkeypatch):
    device = _FakeDevice([3.0] * 12)
    device.install(monkeypatch)

    _rank(device, 12, finalists=2, chunk=3)

    # Four probe chunks, one priming: the clocks are brought up before the
    # first measured chunk and the chunk drains are too short to lose them.
    assert device.primed == 1


def test_the_finalist_hook_decides_what_may_rank(monkeypatch):
    times = [3.0, 1.0, 2.0]
    device = _FakeDevice(times)
    device.install(monkeypatch)
    checked = []

    def on_finalist(i, run):
        checked.append(i)
        return i != 1  # the fastest one is wrong

    ranked = _rank(device, len(times), finalists=3, chunk=8, on_finalist=on_finalist)

    # Checked fastest-first, and every finalist is checked before it is timed.
    assert checked == [1, 2, 0]
    assert ranked[1].passed is False
    # It is still timed, so its row carries a number for the caller to report.
    assert ranked[1].ms == 1.0
    assert ranked[0].passed is True


def test_scratch_is_released_for_everything_that_cannot_rank(monkeypatch):
    times = [9.0, 1.0, 8.0, None, 2.0, 7.0]
    device = _FakeDevice(times)
    device.install(monkeypatch)
    dropped = []

    ranked = _rank(device, len(times), finalists=2, chunk=3, on_drop=dropped.append)

    # Everything but the two finalists: the slow ones and the one that failed.
    assert sorted(dropped) == [0, 2, 3, 5]
    assert sorted(r.index for r in ranked if r.is_finalist) == [1, 4]
    # A finalist's scratch is still held -- it was timed with it.
    assert not set(dropped) & {1, 4}


def test_a_full_scratch_budget_closes_the_chunk_early(monkeypatch):
    device = _FakeDevice([1.0] * 8)
    device.install(monkeypatch)
    held = {"bytes": 0}
    sizes = []

    def make_run(i):
        held["bytes"] += 100
        sizes.append(len(sizes))
        return device.make_run(i)

    batches = []
    real_time_batch = device._time_batch

    def counting(runs, **kwargs):
        if kwargs["iters"] == batched_timing.DEFAULT_PROBE_ITERS:
            batches.append(len(runs))
        return real_time_batch(runs, **kwargs)

    monkeypatch.setattr(batched_timing, "time_batch", counting)

    batched_timing.rank(
        8,
        make_run,
        warmup=3,
        iters=10,
        finalists=8,
        chunk=100,
        held_bytes=lambda: held["bytes"],
        chunk_bytes=250,
        # Nothing is ever released here, so the budget is passed for good
        # after the third bind; every later chunk holds one candidate.
        on_drop=lambda i: None,
    )

    # The chunk that crosses the budget keeps the candidate that crossed it.
    assert batches == [3, 1, 1, 1, 1, 1]


def test_without_a_budget_the_count_alone_chunks(monkeypatch):
    device = _FakeDevice([1.0] * 5)
    device.install(monkeypatch)
    batches = []
    real_time_batch = device._time_batch

    def counting(runs, **kwargs):
        if kwargs["iters"] == batched_timing.DEFAULT_PROBE_ITERS:
            batches.append(len(runs))
        return real_time_batch(runs, **kwargs)

    monkeypatch.setattr(batched_timing, "time_batch", counting)

    batched_timing.rank(5, device.make_run, warmup=3, iters=10, finalists=5, chunk=2)

    assert batches == [2, 2, 1]


def test_probe_results_are_reported_as_they_land(monkeypatch):
    device = _FakeDevice([5.0, 4.0, 3.0, 2.0, 1.0])
    device.install(monkeypatch)
    # Read inside the callback: the Ranked handed to it is the live one, and a
    # finalist's gets its ``ms`` filled in later.
    seen = []

    def on_probe(r):
        seen.append((r.index, r.probe_ms, r.ms))

    _rank(device, 5, finalists=1, chunk=2, on_probe=on_probe)

    # One callback per candidate, in candidate order, before any full-fidelity
    # timing -- this is what a long case prints its progress from.
    assert [i for i, _, _ in seen] == [0, 1, 2, 3, 4]
    assert all(probe is not None for _, probe, _ in seen)
    assert all(ms is None for _, _, ms in seen)


def test_staged_timing_changes_only_how_the_finalists_are_timed(monkeypatch):
    """--timing staged (the default): the probe is the same batched block round,
    and only the finalists go through the stalled single-call protocol."""
    times = [4.0, 1.0, 3.0, 2.0]
    device = _FakeDevice(times)
    device.install(monkeypatch)
    staged = []
    assert batched_timing.DEFAULT_TIMING == "staged"

    def _staged(runs, *, warmup, iters, stream=0):
        staged.append((len(runs), warmup, iters))
        # A stalled sample pays the event pair the block loop amortises away.
        return [times[run()] + 0.5 for run in runs]

    monkeypatch.setattr(batched_timing, "time_batch_staged", _staged)

    ranked = _rank(device, len(times), finalists=2)

    assert device.probed == len(times)
    assert device.measured == 0
    assert staged == [(2, 3, 10)]
    assert {r.index: r.ms for r in ranked if r.is_finalist} == {1: 1.5, 3: 2.5}


def test_block_timing_never_touches_the_stall_gate(monkeypatch):
    device = _FakeDevice([1.0, 2.0])
    device.install(monkeypatch)

    def _staged(*_args, **_kwargs):
        raise AssertionError("block timing went through the staged path")

    monkeypatch.setattr(batched_timing, "time_batch_staged", _staged)
    ranked = _rank(device, 2, finalists=2, timing="block")
    assert device.measured == 2
    assert all(r.is_finalist for r in ranked)


def test_an_unknown_timing_is_refused(monkeypatch):
    device = _FakeDevice([1.0])
    device.install(monkeypatch)
    with pytest.raises(ValueError, match="timing"):
        _rank(device, 1, timing="events")
