# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks for the ordinary and fixed-binding decode timing contracts."""

from contextlib import contextmanager
import importlib
import sys
from enum import IntEnum
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(params=["gfx942", "gfx950"])
def benchmark(request):
    return importlib.import_module(
        f"benchmarks.{request.param}.attention.decode.benchmark_decode_live"
    )


def fake_runtime(monkeypatch, benchmark, path, *, scoped=False, fail_timer=False):
    events = []

    class AttentionMaskType(IntEnum):
        BOTTOM_RIGHT_CAUSAL = 2

    def dispatch(request):
        if benchmark.__name__.split(".")[1] == "gfx950":
            assert request.mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL
        return SimpleNamespace(
            spec=SimpleNamespace(path=path, kernel_name=lambda: "selected_route")
        )

    def module(name, **attributes):
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
        return stub

    def ordinary(**arguments):
        assert "execution" not in arguments
        assert "tuning_spec" not in arguments
        assert arguments["backend"] == ("tiled" if path == "2d" else "3d")
        events.append("ordinary")

    @contextmanager
    def replay_scope():
        events.append("enter")
        try:
            yield lambda: events.append("replay")
        finally:
            events.append("scope_drain")

    def prepare(**arguments):
        assert arguments["backend"] == "3d"
        events.append("prepare")
        return SimpleNamespace(
            replay_scope=replay_scope, close=lambda: events.append("close")
        )

    def timer(call, *, warmup, iters, stream):
        assert (warmup, iters, stream) == (1, 2, 7)
        events.append("timer")
        for _ in range(warmup + iters):
            call()
        if fail_timer:
            raise RuntimeError("timing failed")
        return 0.125

    module("torch", empty_like=lambda q: object())
    kernels = module(
        "kernels",
        UnifiedAttentionProblem=lambda **kw: SimpleNamespace(**kw),
        run_unified_attention_torch=ordinary,
    )
    # Ordinary mode must work without importing the optional preparation API.
    if scoped:
        kernels.prepare_unified_attention_torch = prepare
    module("kernels.common.attention_unified", _resolve_attention_arch=lambda: "gfx950")
    module(
        "dispatch.attention",
        AttentionMaskType=AttentionMaskType,
        AttentionRequest=lambda **kw: SimpleNamespace(**kw),
        dispatch_attention=dispatch,
    )
    module(
        "rocke.runtime",
        time_launches=timer,
        synchronize_and_release=lambda stream: events.append("drain"),
    )
    monkeypatch.setattr(benchmark, "_bench_stream_handle", lambda: 7)
    shape = benchmark.DecodeShape(
        batch=1,
        seqlen_q=1,
        seqlen_k=2048,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        dtype="bf16",
        label="decode",
    )
    data = dict(
        q=object(),
        kc=object(),
        vc=object(),
        cu_q=object(),
        kv_lens=object(),
        scale=0.125,
        block_table=object(),
        softcap=0.0,
        sinks=None,
        alibi_slopes=None,
        qq_bias=None,
        k_scale=1.0,
        v_scale=1.0,
    )
    return shape, data, events


@pytest.mark.parametrize("path", ["2d", "3d"])
def test_default_times_ordinary_calls_without_preparation(monkeypatch, benchmark, path):
    shape, data, events = fake_runtime(monkeypatch, benchmark, path)

    assert benchmark._run_dsl(shape, data, 120, warmup=1, iters=2) == (
        0.125,
        path,
        "selected_route",
    )
    assert events == ["timer", "ordinary", "ordinary", "ordinary", "drain"]


def test_scoped_3d_prepares_outside_timing_and_drains_before_close(
    monkeypatch, benchmark
):
    shape, data, events = fake_runtime(monkeypatch, benchmark, "3d", scoped=True)

    assert benchmark._run_dsl(shape, data, 120, warmup=1, iters=2, timing="scoped") == (
        0.125,
        "3d",
        "selected_route",
    )
    assert events == [
        "prepare",
        "enter",
        "timer",
        "replay",
        "replay",
        "replay",
        "scope_drain",
        "close",
    ]


def test_scoped_2d_keeps_ordinary_calls(monkeypatch, benchmark):
    shape, data, events = fake_runtime(monkeypatch, benchmark, "2d")

    assert benchmark._run_dsl(shape, data, 120, warmup=1, iters=2, timing="scoped") == (
        0.125,
        "2d",
        "selected_route",
    )
    assert events == ["timer", "ordinary", "ordinary", "ordinary", "drain"]


def test_scoped_timing_failure_drains_and_closes_execution(monkeypatch, benchmark):
    shape, data, events = fake_runtime(
        monkeypatch, benchmark, "3d", scoped=True, fail_timer=True
    )

    assert benchmark._run_dsl(shape, data, 120, warmup=1, iters=2, timing="scoped") == (
        None,
        None,
        None,
    )
    assert events[-2:] == ["scope_drain", "close"]
