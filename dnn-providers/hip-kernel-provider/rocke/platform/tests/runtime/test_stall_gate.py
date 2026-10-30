# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Staged timing: one stalled event pair per call, as hipDNN's benchmark times.

The fake-gate tests need no GPU: they pin the order of one sample (arm, start
event, the call, stop event, release, wait) and what happens when the call
raises or the watchdog had to release the gate. The last tests run on a GPU
when there is one: a real wait on a real stream, released by the host and, for
a call that blocks the host, by the watchdog.
"""

from __future__ import annotations

import os

import pytest
from rocke.runtime.stall_gate import StallTimeout, time_launches_staged

_STREAM = 5
_GPU = pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="needs an AMD GPU")


class _FakeEvent:
    def __init__(self, log, name):
        self._log = log
        self._name = name

    def record(self, stream):
        self._log.append(f"record:{self._name}")

    def synchronize(self):
        self._log.append(f"sync:{self._name}")

    def elapsed_to(self, other):
        return 2.0

    def destroy(self):
        self._log.append(f"destroy:{self._name}")


class _FakeRuntime:
    def __init__(self):
        self.log: list[str] = []
        self._events = 0

    def sync(self):
        self.log.append("device_sync")

    def wait_stream(self, stream):
        self.log.append("wait_stream")

    def event(self):
        self._events += 1
        return _FakeEvent(self.log, "start" if self._events == 1 else "stop")


class _FakeGate:
    def __init__(self, log, timeout_on=None):
        self._log = log
        self._arms = 0
        self._timeout_on = timeout_on

    def arm(self, stream):
        self._arms += 1
        self._log.append("arm")

    def release(self):
        self._log.append("release")

    def timed_out(self):
        return self._arms == self._timeout_on

    def close(self):
        self._log.append("close")


def _fn(log, raise_on=None):
    calls = [0]

    def fn():
        calls[0] += 1
        log.append("fn")
        if calls[0] == raise_on:
            raise RuntimeError("launch failed")

    return fn


def test_every_call_is_its_own_stalled_sample():
    rt = _FakeRuntime()
    gate = _FakeGate(rt.log)
    out = time_launches_staged(
        _fn(rt.log), warmup=1, iters=2, stream=_STREAM, gate=gate, runtime=rt
    )
    one = ["arm", "record:start", "fn", "record:stop", "release", "sync:stop"]
    assert rt.log == ["device_sync"] + one * 3 + [
        "destroy:start",
        "destroy:stop",
        "wait_stream",
    ]
    # The warmup sample is discarded; a caller's gate is not closed.
    assert out == [2.0, 2.0]


def test_a_raise_releases_and_drains_before_it_propagates():
    rt = _FakeRuntime()
    gate = _FakeGate(rt.log)
    with pytest.raises(RuntimeError, match="launch failed"):
        time_launches_staged(
            _fn(rt.log, raise_on=2), warmup=0, iters=3, gate=gate, runtime=rt
        )
    tail = rt.log[rt.log.index("fn", rt.log.index("fn") + 1) :]
    assert tail[:3] == ["fn", "release", "device_sync"]
    assert "close" not in rt.log


def test_a_watchdog_release_invalidates_the_measurement():
    rt = _FakeRuntime()
    gate = _FakeGate(rt.log, timeout_on=2)
    with pytest.raises(StallTimeout):
        time_launches_staged(_fn(rt.log), warmup=0, iters=3, gate=gate, runtime=rt)
    assert rt.log.count("fn") == 2
    assert rt.log[-3:] == ["device_sync", "destroy:start", "destroy:stop"]


def _busy_launch():
    """A launch of a one-instruction kernel, or skip when none can be built."""
    import struct

    from rocke.runtime.hip_module import Runtime, get_device_arch

    try:
        from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    except Exception as e:  # noqa: BLE001  # pragma: no cover - environment dependent
        pytest.skip(f"comgr unavailable: {e}")
    arch = get_device_arch()
    if not arch:
        pytest.skip("no HIP device")
    ir = (
        'target triple = "amdgcn-amd-amdhsa"\n'
        "define amdgpu_kernel void @nop(ptr addrspace(1) %p) {\n"
        "  ret void\n"
        "}\n"
    )
    try:
        hsaco, _timings = build_hsaco_from_llvm_ir(ir, isa=f"amdgcn-amd-amdhsa--{arch}")
    except Exception as e:  # noqa: BLE001  # pragma: no cover - environment dependent
        pytest.skip(f"cannot build a kernel here: {e}")
    rt = Runtime()
    fn = rt.load_module(hsaco).get_function("nop")
    return rt, rt.prepare_launch(fn, (1, 1, 1), (64, 1, 1), struct.pack("<Q", 0))


@_GPU
def test_real_gate_times_every_call():
    _rt, run = _busy_launch()
    out = time_launches_staged(run, warmup=2, iters=5)
    assert len(out) == 5
    assert all(ms > 0 for ms in out)


@_GPU
def test_real_watchdog_unblocks_a_call_that_waits_on_the_stalled_stream():
    from rocke.runtime.stall_gate import StallGate

    rt, run = _busy_launch()

    def blocking():
        run()
        rt.stream_sync(0)

    with StallGate(timeout_s=0.2) as gate:
        with pytest.raises(StallTimeout):
            time_launches_staged(blocking, warmup=0, iters=1, gate=gate)
        # The gate stays usable after a timeout.
        assert len(time_launches_staged(run, warmup=0, iters=2, gate=gate)) == 2
