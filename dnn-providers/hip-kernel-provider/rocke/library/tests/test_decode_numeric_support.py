# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host checks for decode prerequisites, launch ordering and scoped cleanup."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests._decode_numeric_support import decode_launch_session, gpu


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
def test_decode_gpu_uses_detected_supported_target(arch, monkeypatch):
    from rocke.runtime import hip_module

    torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.0"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hip_module, "get_device_arch", lambda: arch)
    assert gpu.__wrapped__() == (torch, arch)


@pytest.mark.parametrize(
    "hip,available,arch,reason",
    [
        (None, True, "gfx950", "ROCm torch"),
        ("7.0", False, None, "ROCm GPU"),
        ("7.0", True, "gfx1151", "decode validation targets"),
    ],
)
def test_decode_gpu_skips_only_missing_prerequisites(
    hip, available, arch, reason, monkeypatch
):
    from rocke.runtime import hip_module

    torch = SimpleNamespace(
        version=SimpleNamespace(hip=hip),
        cuda=SimpleNamespace(is_available=lambda: available),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    def detect_arch():
        assert hip and available
        return arch

    monkeypatch.setattr(hip_module, "get_device_arch", detect_arch)
    with pytest.raises(pytest.skip.Exception, match=reason):
        gpu.__wrapped__()


def test_decode_gpu_skips_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(pytest.skip.Exception, match="ROCm torch required"):
        gpu.__wrapped__()


def test_decode_gpu_does_not_hide_device_probe_errors(monkeypatch):
    from rocke.runtime import hip_module

    torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.0"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    def detect_arch():
        raise RuntimeError("device probe failed")

    monkeypatch.setattr(hip_module, "get_device_arch", detect_arch)
    with pytest.raises(RuntimeError, match="device probe failed"):
        gpu.__wrapped__()


def _runtime(events, args, fault=None):
    def record(event):
        events.append(event)
        if fault == event:
            raise RuntimeError(event)

    def check_arguments(kwargs):
        assert kwargs.keys() == args.keys()
        assert all(kwargs[key] is value for key, value in args.items())

    owner = SimpleNamespace(
        launch=lambda: record("owner.launch"), close=lambda: record("owner.close")
    )
    graph = SimpleNamespace(replay=lambda: record("graph.replay"))

    def prepare(**kwargs):
        check_arguments(kwargs)
        record("prepare")
        return owner

    def run(**kwargs):
        check_arguments(kwargs)
        record("run")

    @contextmanager
    def capture(captured):
        assert captured is graph
        record("capture.enter")
        try:
            yield
        finally:
            record("capture.exit")

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            CUDAGraph=lambda: graph,
            graph=capture,
            synchronize=lambda: record("synchronize"),
        )
    )
    au = SimpleNamespace(
        prepare_unified_attention_torch=prepare,
        run_unified_attention_torch=run,
        clear_attention_3d_graph_cache=lambda: record("clear"),
    )
    return torch, au


@pytest.mark.parametrize("mode", ["eager", "prepared", "graph", "external"])
def test_decode_launch_order_and_original_buffers(mode):
    events = []
    args = dict(
        out=object(), block_table=object(), lse=object(), use_graph=mode == "graph"
    )
    torch, au = _runtime(events, args)
    with decode_launch_session(torch, au, args, mode) as launch:
        launch()
        launch()
    setup = []
    if mode in ("prepared", "external"):
        setup.append("prepare")
    if mode == "external":
        setup += ["capture.enter", "owner.launch", "capture.exit"]
    call = {
        "eager": "run",
        "graph": "run",
        "prepared": "owner.launch",
        "external": "graph.replay",
    }[mode]
    teardown = (
        ["owner.close", "clear"] if mode in ("prepared", "external") else ["clear"]
    )
    assert events == setup + [call, "synchronize"] * 2 + teardown


@pytest.mark.parametrize(
    "fault", ["prepare", "capture.enter", "owner.launch", "assertion", "owner.close"]
)
def test_decode_scope_failure_closes_owner_and_clears_cache(fault):
    events = []
    args = dict(out=object(), block_table=object(), use_graph=False)
    torch, au = _runtime(events, args, fault=fault)
    mode = "external" if fault == "capture.enter" else "prepared"
    error = AssertionError if fault == "assertion" else RuntimeError
    with pytest.raises(error, match=fault):
        with decode_launch_session(torch, au, args, mode) as launch:
            if fault == "assertion":
                raise AssertionError("assertion")
            launch()
    assert events[-1] == "clear"
    if fault == "prepare":
        assert events == ["prepare", "clear"]
    else:
        assert events[-2:] == ["owner.close", "clear"]
