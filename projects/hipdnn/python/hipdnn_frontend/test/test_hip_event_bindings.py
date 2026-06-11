#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Smoke tests for direct HIP runtime bindings."""

import pytest

import hipdnn_frontend as fe


_REQUIRED_API = (
    "HipEvent",
    "hip_stream_synchronize",
    "hip_get_device_count",
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
