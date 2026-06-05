# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.host (CPU time probe + host memory snapshot)."""

import resource
import sys
import types
from unittest.mock import patch

import pytest

from dnn_benchmarking.metrics import host
from dnn_benchmarking.metrics._diagnostic import reset as _reset_warns


@pytest.fixture(autouse=True)
def _reset_warn_state():
    _reset_warns()
    yield
    _reset_warns()


def _fake_rusage(utime: float, stime: float):
    """Build a struct_rusage-shaped object matching what getrusage returns."""
    obj = types.SimpleNamespace()
    obj.ru_utime = utime
    obj.ru_stime = stime
    return obj


class TestCpuTimeProbe:
    def test_delta_in_milliseconds(self):
        samples = [_fake_rusage(0.0, 0.0), _fake_rusage(0.1, 0.05)]
        with patch.object(resource, "getrusage", side_effect=samples):
            with host.CpuTimeProbe() as probe:
                pass
        assert probe.delta is not None
        assert probe.delta.user_time_ms == pytest.approx(100.0)
        assert probe.delta.kernel_time_ms == pytest.approx(50.0)

    def test_failure_in_start_yields_none_delta(self):
        with patch.object(resource, "getrusage", side_effect=OSError("denied")):
            with host.CpuTimeProbe() as probe:
                pass
        assert probe.delta is None

    def test_failure_in_end_does_not_raise_and_leaves_delta_none(self):
        # First call (enter) succeeds; second call (exit) raises.
        sequence = [_fake_rusage(0.0, 0.0), OSError("end failed")]

        def _side(*_args, **_kwargs):
            value = sequence.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        with patch.object(resource, "getrusage", side_effect=_side):
            with host.CpuTimeProbe() as probe:
                pass
        assert probe.delta is None


class TestHostMemorySnapshot:
    def test_returns_stable_keys_when_psutil_missing(self):
        # Simulate ImportError by removing psutil from sys.modules and
        # blocking re-import via meta_path.
        with patch.dict(sys.modules, {"psutil": None}):
            snap = host.host_memory_snapshot()
        assert set(snap.keys()) == {"host_rss_mb", "host_ram_available_mb"}
        assert snap["host_rss_mb"] is None
        assert snap["host_ram_available_mb"] is None

    def test_psutil_path_returns_floats(self):
        fake_psutil = types.ModuleType("psutil")

        class _Mem:
            rss = 256 * 1024 * 1024  # 256 MiB

        class _Proc:
            def memory_info(self):
                return _Mem()

        class _VMem:
            available = 16 * 1024 * 1024 * 1024  # 16 GiB

        fake_psutil.Process = lambda: _Proc()
        fake_psutil.virtual_memory = lambda: _VMem()
        # psutil.Error is referenced inside the except handler — need it
        # to exist as a class so the import-level reference resolves.
        fake_psutil.Error = Exception

        with patch.dict(sys.modules, {"psutil": fake_psutil}):
            snap = host.host_memory_snapshot()
        assert snap["host_rss_mb"] == pytest.approx(256.0)
        assert snap["host_ram_available_mb"] == pytest.approx(16384.0)


class TestIsPsutilAvailable:
    def test_returns_false_without_module(self):
        with patch.dict(sys.modules, {"psutil": None}):
            assert host.is_psutil_available() is False
