#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Smoke tests for direct HIP runtime bindings."""

import subprocess
import sys
import textwrap

import pytest

import hipdnn_frontend as fe


_REQUIRED_API = (
    "HipEvent",
    "HipStallGate",
    "hip_stream_synchronize",
    "hip_get_device_count",
    "hip_device_synchronize",
    "hip_can_use_stream_wait_value",
)


def test_hip_event_symbols_are_exported() -> None:
    missing = [name for name in _REQUIRED_API if not hasattr(fe, name)]
    assert missing == []


@pytest.mark.gpu
def test_hip_event_timing_smoke() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")

    start = fe.HipEvent()
    stop = fe.HipEvent()

    start.record(0)
    stop.record(0)
    stop.synchronize()

    assert start.elapsed_time(stop) >= 0.0


@pytest.mark.gpu
def test_stall_gate_orders_events() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")
    if not fe.hip_can_use_stream_wait_value():
        pytest.skip("Device does not support hipStreamWaitValue32")

    gate = fe.HipStallGate()
    start = fe.HipEvent()
    stop = fe.HipEvent()

    fe.hip_device_synchronize()
    gate.arm(0)
    start.record(0)
    stop.record(0)
    gate.release()
    stop.synchronize()

    assert start.elapsed_time(stop) >= 0.0


@pytest.mark.gpu
def test_stall_gate_raises_after_destroy() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")
    if not fe.hip_can_use_stream_wait_value():
        pytest.skip("Device does not support hipStreamWaitValue32")

    gate = fe.HipStallGate()
    gate.destroy()

    # A destroyed gate must reject arm/release with a clear error instead of
    # issuing stream ops on freed signal memory.
    with pytest.raises(RuntimeError, match="destroyed"):
        gate.arm(0)
    with pytest.raises(RuntimeError, match="destroyed"):
        gate.release()


# Runs in a child process: the watchdog latch that this test trips is process-wide and
# has no Python reset, so tripping it in-process would silently unstall every later test.
_TIMEOUT_SCRIPT = textwrap.dedent(
    """
    import hipdnn_frontend as fe

    gate = fe.HipStallGate()
    gate.arm(0)
    # Blocks until the watchdog writes the signal: only the host can release, and the
    # host is right here, waiting on the stream it stalled.
    fe.hip_stream_synchronize(0)
    assert gate.timed_out(), "watchdog released the stall but did not report it"

    # Reuse after a timeout must raise: stalling is off from here on, so a caller that
    # got no exception would keep reading host-inflated times as device time.
    try:
        gate.arm(0)
    except RuntimeError as err:
        assert "watchdog" in str(err), err
    else:
        raise AssertionError("arm() silently declined after the watchdog timeout")

    assert not gate.timed_out(), "a declined arm still reports the earlier timeout"
    print("OK")
    """
)


@pytest.mark.gpu
def test_stall_gate_reports_timeout_and_refuses_reuse() -> None:
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
