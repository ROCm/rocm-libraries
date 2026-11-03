# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The timing loops drain the device on every exit, including a raise.

:func:`time_launches` and the timed path of :func:`launch_kernel` run their
launches fire-and-forget, so a callable that raises mid-loop leaves earlier
launches in flight. Callers free the buffers those launches use as soon as the
exception reaches them, so the loops drain before letting it propagate. The
fake-runtime tests need no GPU; the last test runs on one when present.
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest
import rocke.runtime.launcher as L
from rocke.runtime.hip_module import HipError

# The fake runtime's stream: a nonzero handle is used as-is (stream 0 would
# resolve through torch). It is not a real HIP stream.
_STREAM = 5


class _FakeEvent:
    def __init__(self, log):
        self._log = log

    def record(self, stream):
        self._log.append("record")

    def synchronize(self):
        self._log.append("event_sync")

    def elapsed_to(self, other):
        return 10.0

    def destroy(self):
        self._log.append("destroy")


class _FakeRuntime:
    def __init__(self, sync_error: Exception | None = None):
        self.log: list[str] = []
        self._sync_error = sync_error
        self._warmup_synced = False

    def sync(self):
        self.log.append("sync")
        if self._sync_error is not None and self._warmup_synced:
            raise self._sync_error
        self._warmup_synced = True

    def wait_stream(self, stream):
        self.log.append(f"wait_stream({stream})")

    def event(self):
        return _FakeEvent(self.log)


def _uses(rt):
    return mock.patch.object(L, "_runtime", return_value=rt)


def _raise_on_call(n, log, exc):
    calls = [0]

    def fn():
        calls[0] += 1
        log.append("fn")
        if calls[0] == n:
            raise exc

    return fn


def test_success_drains_the_stream_once():
    rt = _FakeRuntime()
    fn = _raise_on_call(0, rt.log, None)
    with _uses(rt):
        ms = L.time_launches(fn, warmup=1, iters=2, stream=_STREAM)
    assert ms == 5.0
    assert rt.log[-1] == f"wait_stream({_STREAM})"
    # The only device-wide sync is the one ending the warmup.
    assert rt.log.count("sync") == 1


@pytest.mark.parametrize(
    ("fail_on", "where"),
    [(1, "warmup"), (3, "timed loop")],
)
def test_a_raise_drains_before_it_propagates(fail_on, where):
    rt = _FakeRuntime()
    err = HipError(f"launch failed in the {where}")
    fn = _raise_on_call(fail_on, rt.log, err)
    with _uses(rt), pytest.raises(HipError) as info:
        L.time_launches(fn, warmup=2, iters=4, stream=_STREAM)
    assert info.value is err
    # The drain is the last thing that happens, after the failing call.
    assert rt.log[-1] == "sync"
    assert rt.log.index("fn") < len(rt.log) - 1
    if where == "timed loop":
        # Both events are still destroyed.
        assert rt.log.count("destroy") == 2


def test_a_failed_drain_does_not_mask_the_launch_error():
    drain_err = HipError("hipDeviceSynchronize: device error")
    rt = _FakeRuntime(sync_error=drain_err)
    err = HipError("launch failed")
    fn = _raise_on_call(3, rt.log, err)
    with _uses(rt), pytest.raises(HipError) as info:
        L.time_launches(fn, warmup=2, iters=4, stream=_STREAM)
    assert info.value is err
    if sys.version_info >= (3, 11):
        assert any("device error" in n for n in info.value.__notes__)


def test_keyboard_interrupt_is_not_drained():
    # A drain would block for good on the hung kernel an interrupt signals.
    rt = _FakeRuntime()
    fn = _raise_on_call(3, rt.log, KeyboardInterrupt())
    with _uses(rt), pytest.raises(KeyboardInterrupt):
        L.time_launches(fn, warmup=2, iters=4, stream=_STREAM)
    assert rt.log.count("sync") == 1  # the warmup's only


@pytest.mark.parametrize("gpu_timer", [True, False])
def test_launch_kernel_timing_drains_on_a_raise(gpu_timer):
    rt = _FakeRuntime()
    err = RuntimeError("callable failed")
    fn = _raise_on_call(3, rt.log, err)
    s = L.StreamConfig(
        stream_id=_STREAM,
        time_kernel=True,
        cold_niters=1,
        nrepeat=4,
        is_gpu_timer=gpu_timer,
    )
    with _uses(rt), pytest.raises(RuntimeError) as info:
        L.launch_kernel(s, lambda _s: fn())
    assert info.value is err
    assert rt.log[-1] == "sync"
    if gpu_timer:
        assert rt.log.count("destroy") == 2


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="needs an AMD GPU")
def test_real_runtime_drains_queued_work_on_a_raise():
    """On HIP: the drain runs against the real runtime and succeeds."""
    rt = L._runtime()
    try:
        nbytes = 64 << 20
        buf = rt.alloc(nbytes)
    except HipError as e:
        pytest.skip(f"no usable HIP device: {e}")
    calls = [0]

    def fn():
        calls[0] += 1
        rt.memset(buf, calls[0] & 0xFF, nbytes)
        if calls[0] == 4:
            raise HipError("injected launch failure")

    try:
        spy = mock.patch.object(rt, "sync", wraps=rt.sync)
        with spy as sync, pytest.raises(HipError, match="inj") as info:
            # A real stream this time: the fake handle is not one.
            L.time_launches(fn, warmup=2, iters=8, stream=0)
        # The warmup's sync, then the drain -- which did not fail.
        assert sync.call_count == 2
        assert not getattr(info.value, "__notes__", None)
        assert not rt._pending_args
    finally:
        rt.free(buf)
