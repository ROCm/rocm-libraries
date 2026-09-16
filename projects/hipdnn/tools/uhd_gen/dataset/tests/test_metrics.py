# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The arithmetic that turns a measured time into a rate.

Worth its own suite because nothing downstream can catch an error here. A wrong count does not
raise -- it produces a plausible throughput, and every model trained on it ranks kernels by a
number that is wrong by a constant factor nobody sees. The counts themselves come from the
engine that solved the problem, so what remains to test is how a row is read: which values mean
"reported", which mean "absent", and that absent never becomes a winner.
"""

from __future__ import annotations

import pytest

from uhd_gen.dataset.metrics import derive_metrics, reported


def test_metrics_are_derived_from_the_engines_counts_and_the_time():
    got = derive_metrics(dict(flops=2.0e12, bytes=12.0e9), time_ms=1.0)

    assert got["tflops"] == pytest.approx(2.0e12 / 1e-3 / 1e12)
    assert got["gbs"] == pytest.approx(12.0e9 / 1e-3 / 1e9)


def test_the_counts_are_the_engines_own_rather_than_a_declarations():
    """What the removal of publish.py's `--opmeta` settled.

    A corpus_gen declaration names its parameters in corpus_gen's vocabulary and an engine binds
    its own; nothing checks that the two agree (RFC 0020 §6), and a resolvable-but-wrong name
    yields a plausible throughput rather than an error. It was wrong on measured data too: the
    sdpa declaration counts every tensor `heads` wide, so a grouped-query corpus had its `bytes`
    overstated by up to 7.8x. The engine's count is right because it knows what it read -- so a
    row carrying the problem parameters but no cost has nothing to derive from, by design.
    """
    query = dict(batch=2, heads=32, seqlen_q=1024, seqlen_k=1024, head_dim=128, dtype="fp16")
    assert derive_metrics(query, time_ms=1.0) == {"tflops": None, "gbs": None}


def test_a_memory_bound_operation_reports_bandwidth_and_no_throughput():
    """An engine may publish bytes and no flops. A FLOP count for layernorm is a convention
    rather than a fact, and it is memory-bound anyway, so `gbs` is what ranks it."""
    got = derive_metrics(dict(bytes=4.0e9), time_ms=1.0)

    assert got["tflops"] is None
    assert got["gbs"] == pytest.approx(4.0e9 / 1e-3 / 1e9)


def test_each_metric_stands_on_its_own_count():
    """Neither metric may borrow the other's presence: an engine reporting one cost and not the
    other gets exactly the metric it can support."""
    assert derive_metrics(dict(flops=1.0e12), 1.0)["gbs"] is None
    assert derive_metrics(dict(flops=1.0e12), 1.0)["tflops"] is not None
    assert derive_metrics(dict(bytes=1.0e9), 1.0)["tflops"] is None
    assert derive_metrics(dict(bytes=1.0e9), 1.0)["gbs"] is not None


def test_the_rate_scales_with_the_time():
    fast = derive_metrics(dict(flops=1.0e12, bytes=1.0e9), time_ms=1.0)
    slow = derive_metrics(dict(flops=1.0e12, bytes=1.0e9), time_ms=2.0)

    assert slow["tflops"] == pytest.approx(fast["tflops"] / 2)
    assert slow["gbs"] == pytest.approx(fast["gbs"] / 2)


@pytest.mark.parametrize("time_ms", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_no_measurement_yields_null_never_a_winner(time_ms):
    """The direction of failure that decided null over zero.

    A zero time is impossible rather than fast, and dividing by it gives an infinity that
    outranks every real measurement. Nothing here may manufacture a value from an absent one.
    """
    got = derive_metrics(dict(flops=1.0e12, bytes=1.0e9), time_ms)
    assert got == {"tflops": None, "gbs": None}


@pytest.mark.parametrize("absent", ["", None, float("nan"), float("inf"), 0.0, -1.0, "n/a"])
def test_an_unfilled_cost_column_reads_as_absent_rather_than_as_zero(absent):
    """A corpus is written with an empty restval, so a column the engine did not fill arrives as
    `''` or NaN. Either turning into a 0.0 would give a real measurement a rate of exactly zero,
    which sorts last and looks like a slow kernel rather than like a missing count. A negative
    or non-finite count is a producer bug, and reads the same way for the same reason.
    """
    assert reported(dict(flops=absent), "flops") is None
    assert derive_metrics(dict(flops=absent, bytes=absent), time_ms=1.0) == {
        "tflops": None, "gbs": None
    }


def test_a_cost_arrives_readable_however_the_reader_typed_it():
    """A CSV read without a dtype hint gives strings; pandas gives numpy floats. Both are the
    engine's count, and the metric may not depend on which reader was upstream."""
    assert reported({"flops": "2e12"}, "flops") == pytest.approx(2.0e12)
    assert reported({"flops": 2}, "flops") == pytest.approx(2.0)
    assert reported({}, "flops") is None
