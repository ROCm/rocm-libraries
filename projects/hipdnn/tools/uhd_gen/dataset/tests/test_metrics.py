# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for turning a measured time and the engine's cost counts into rates."""

from __future__ import annotations

import pytest

from uhd_gen.dataset.metrics import derive_metrics, reported


def test_metrics_are_derived_from_the_engines_counts_and_the_time():
    got = derive_metrics(dict(flops=2.0e12, bytes=12.0e9), time_ms=1.0)

    assert got["tflops"] == pytest.approx(2.0e12 / 1e-3 / 1e12)
    assert got["gbs"] == pytest.approx(12.0e9 / 1e-3 / 1e9)


def test_the_counts_are_the_engines_own_rather_than_a_declarations():
    """Problem parameters without an engine-reported cost give no metric."""
    query = dict(
        batch=2, heads=32, seqlen_q=1024, seqlen_k=1024, head_dim=128, dtype="fp16"
    )
    assert derive_metrics(query, time_ms=1.0) == {"tflops": None, "gbs": None}


def test_a_memory_bound_operation_reports_bandwidth_and_no_throughput():
    """Layernorm reports bytes only; `gbs` is what ranks it."""
    got = derive_metrics(dict(bytes=4.0e9), time_ms=1.0)

    assert got["tflops"] is None
    assert got["gbs"] == pytest.approx(4.0e9 / 1e-3 / 1e9)


def test_each_metric_stands_on_its_own_count():
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
    """A zero time would give an infinite rate that outranks every real measurement."""
    got = derive_metrics(dict(flops=1.0e12, bytes=1.0e9), time_ms)
    assert got == {"tflops": None, "gbs": None}


@pytest.mark.parametrize(
    "absent", ["", None, float("nan"), float("inf"), 0.0, -1.0, "n/a"]
)
def test_an_unfilled_cost_column_reads_as_absent_rather_than_as_zero(absent):
    """An empty or invalid count must not become 0.0, which reads as a slow kernel."""
    assert reported(dict(flops=absent), "flops") is None
    assert derive_metrics(dict(flops=absent, bytes=absent), time_ms=1.0) == {
        "tflops": None,
        "gbs": None,
    }


def test_a_cost_arrives_readable_however_the_reader_typed_it():
    """CSV strings and numeric values parse the same."""
    assert reported({"flops": "2e12"}, "flops") == pytest.approx(2.0e12)
    assert reported({"flops": 2}, "flops") == pytest.approx(2.0)
    assert reported({}, "flops") is None
