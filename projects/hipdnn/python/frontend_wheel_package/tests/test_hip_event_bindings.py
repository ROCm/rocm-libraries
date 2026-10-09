#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Smoke tests for direct HIP runtime bindings."""

import subprocess
import sys
import textwrap

import pytest

import hipdnn_frontend as fe


@pytest.mark.gpu
def test_hip_event_timing_smoke() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")

    start = fe.HipEvent()
    assert str(start.ptr()) in repr(start)
    stop = fe.HipEvent()

    start.record(0)
    stop.record(0)
    stop.synchronize()

    assert start.elapsed_time(stop) >= 0.0


@pytest.mark.gpu
def test_elapsed_time_rejects_reversed_events() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")

    start = fe.HipEvent()
    stop = fe.HipEvent()

    # Record in reverse order: stop completes before start, so the true elapsed
    # time from start to stop is negative. HIP itself reports success with that
    # negative duration; the binding must reject it rather than hand a caller a
    # timing that silently violates the ">= 0" invariant every other test relies on.
    stop.record(0)
    fe.hip_device_synchronize()
    start.record(0)
    fe.hip_device_synchronize()

    with pytest.raises(RuntimeError):
        start.elapsed_time(stop)


@pytest.mark.gpu
def test_stall_gate_orders_events() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")
    if not fe.hip_can_use_stream_wait_value():
        pytest.skip("Device does not support hipStreamWaitValue32")

    start = fe.HipEvent()
    stop = fe.HipEvent()

    fe.hip_device_synchronize()
    with fe.HipStallGate() as gate:
        gate.arm(0)
        start.record(0)
        stop.record(0)
        gate.release()
        stop.synchronize()

    assert start.elapsed_time(stop) >= 0.0


@pytest.mark.gpu
def test_stall_gate_close_is_idempotent_and_blocks_reuse() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")
    if not fe.hip_can_use_stream_wait_value():
        pytest.skip("Device does not support hipStreamWaitValue32")

    gate = fe.HipStallGate()
    gate.close()
    gate.close()  # idempotent: a second close() is a no-op, not an error

    with pytest.raises(RuntimeError):
        gate.arm(0)
    with pytest.raises(RuntimeError):
        gate.release()
    with pytest.raises(RuntimeError):
        gate.timed_out()


@pytest.mark.gpu
def test_stall_gate_context_manager_closes_on_exit() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")
    if not fe.hip_can_use_stream_wait_value():
        pytest.skip("Device does not support hipStreamWaitValue32")

    with fe.HipStallGate() as gate:
        assert not gate.timed_out()

    with pytest.raises(RuntimeError):
        gate.timed_out()


# A child-process timeout bounds the test if the watchdog fails to release the stream.
_TIMEOUT_SCRIPT = textwrap.dedent(
    """
    import hipdnn_frontend as fe

    gate = fe.HipStallGate()
    gate.arm(0)
    # Blocks until the watchdog writes the signal: only the host can release, and the
    # host is right here, waiting on the stream it stalled.
    fe.hip_stream_synchronize(0)
    assert gate.timed_out(), "watchdog released the stall but did not report it"

    # A timeout belongs to one attempt, not the gate or the Python module.
    gate.arm(0)
    gate.release()
    fe.hip_stream_synchronize(0)
    assert not gate.timed_out()
    gate.close()

    with fe.HipStallGate() as other:
        other.arm(0)
        other.release()
        fe.hip_stream_synchronize(0)
        assert not other.timed_out()
    print("OK")
    """
)


@pytest.mark.gpu
def test_stall_gate_timeout_does_not_disable_reuse_or_other_gates() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")
    if not fe.hip_can_use_stream_wait_value():
        pytest.skip("Device does not support hipStreamWaitValue32")

    # Generous relative to the gate's 2 s budget, so a slow runner cannot fail this.
    completed = subprocess.run(
        [sys.executable, "-c", _TIMEOUT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout
