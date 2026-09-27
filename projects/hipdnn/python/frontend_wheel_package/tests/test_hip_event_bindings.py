#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Smoke tests for direct HIP runtime bindings."""

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
