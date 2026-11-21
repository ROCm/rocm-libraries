# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for manually computed timing records."""

import importlib

import pytest

timing = importlib.import_module("tensilelite.Common.TimingInstrumentation")
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("duration_ms, expected_ms", [(7.5, 7.5), (0.0, 0.0), (-3.0, 0.0)])
def test_record_timing_gates_buffer_writes_and_saturates_negative_durations(
    monkeypatch, enabled, duration_ms, expected_ms
):
    buffer = [("existing", 1.0)]
    monkeypatch.setattr(timing, "_timing_buffer", buffer)
    monkeypatch.setitem(timing.globalParameters, "TimingInstrumentation", enabled)

    timing.record_timing("verification", duration_ms)

    expected = [("existing", 1.0)]
    if enabled:
        expected.append(("verification", expected_ms))
    assert buffer == expected
