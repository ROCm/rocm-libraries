#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Smoke tests for direct HIP runtime bindings."""

import pytest

import hipdnn_frontend as fe


_REQUIRED_API = (
    "HipEvent",
    "hip_event_create",
    "hip_event_record",
    "hip_event_synchronize",
    "hip_event_elapsed_time",
    "hip_stream_synchronize",
    "hip_device_synchronize",
    "hip_get_device_count",
    "hipEventCreate",
    "hipEventRecord",
    "hipEventSynchronize",
    "hipEventElapsedTime",
)


def test_hip_event_symbols_are_exported() -> None:
    missing = [name for name in _REQUIRED_API if not hasattr(fe, name)]
    assert missing == []


def test_hip_event_timing_smoke() -> None:
    if fe.hip_get_device_count() <= 0:
        pytest.skip("No HIP GPU available")

    start = fe.hip_event_create()
    stop = fe.hip_event_create()

    fe.hip_event_record(start, 0)
    fe.hip_event_record(stop, 0)
    fe.hip_event_synchronize(stop)

    assert fe.hip_event_elapsed_time(start, stop) >= 0.0


def test_hip_runtime_api_name_aliases_smoke() -> None:
    if fe.hipGetDeviceCount() <= 0:
        pytest.skip("No HIP GPU available")

    start = fe.hipEventCreate()
    stop = fe.hipEventCreate()

    fe.hipEventRecord(start, 0)
    fe.hipEventRecord(stop, 0)
    fe.hipEventSynchronize(stop)

    assert fe.hipEventElapsedTime(start, stop) >= 0.0
