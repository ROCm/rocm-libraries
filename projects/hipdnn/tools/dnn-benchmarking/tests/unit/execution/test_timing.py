# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Timer and GPU timing utilities."""

import sys
import time
import types

import pytest

import dnn_benchmarking.execution.timing as timing_module
from dnn_benchmarking.execution.timing import (
    GpuTimer,
    GpuTimerInterface,
    HipGpuTimer,
    Timer,
    create_gpu_timer,
    get_available_backends,
    is_gpu_timing_available,
)

# Import shared test fixture
from tests.conftest import DummyHipTimer


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_measures_elapsed_time(self) -> None:
        """Test that timer measures elapsed time correctly."""
        with Timer() as t:
            time.sleep(0.01)  # 10ms sleep

        # Should be at least 10ms, allow for some variance
        assert t.elapsed_ms >= 10.0
        assert t.elapsed_ms < 100.0  # Sanity check

    def test_timer_elapsed_seconds(self) -> None:
        """Test elapsed_s property."""
        with Timer() as t:
            time.sleep(0.01)  # 10ms sleep

        assert t.elapsed_s >= 0.01
        assert t.elapsed_s < 0.1

    def test_timer_zero_when_not_started(self) -> None:
        """Test timer returns zero before use."""
        t = Timer()
        assert t.elapsed_ms == 0.0
        assert t.elapsed_s == 0.0

    def test_timer_reusable(self) -> None:
        """Test that timer can be reused."""
        t = Timer()

        with t:
            time.sleep(0.005)  # 5ms
        first_elapsed = t.elapsed_ms

        with t:
            time.sleep(0.01)  # 10ms
        second_elapsed = t.elapsed_ms

        # Second measurement should be longer
        assert second_elapsed > first_elapsed

    def test_timer_with_exception(self) -> None:
        """Test that timer still records time when exception occurs."""
        t = Timer()

        with pytest.raises(ValueError):
            with t:
                time.sleep(0.005)
                raise ValueError("test error")

        # Time should still be recorded
        assert t.elapsed_ms >= 5.0


class TestGpuTimerInterface:
    """Tests for the unified GPU timer interface."""

    def test_interface_is_abstract(self) -> None:
        """Verify GpuTimerInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GpuTimerInterface()  # type: ignore

    def test_interface_defines_required_methods(self) -> None:
        """Verify interface defines required abstract methods."""
        assert hasattr(GpuTimerInterface, "start")
        assert hasattr(GpuTimerInterface, "stop")
        assert hasattr(GpuTimerInterface, "elapsed_ms")
        assert hasattr(GpuTimerInterface, "synchronize")
        assert hasattr(GpuTimerInterface, "backend_name")

    def test_interface_has_context_manager_protocol(self) -> None:
        """Verify interface supports context manager."""
        assert hasattr(GpuTimerInterface, "__enter__")
        assert hasattr(GpuTimerInterface, "__exit__")


class TestBackendDetection:
    """Tests for backend availability detection."""

    def test_get_available_backends_returns_list(self) -> None:
        """Test that get_available_backends returns a list."""
        backends = get_available_backends()
        assert isinstance(backends, list)

    def test_get_available_backends_only_valid_values(self, monkeypatch) -> None:
        """Test that only valid backend names are returned."""
        monkeypatch.setattr(timing_module, "_is_hip_available", lambda: True)
        backends = get_available_backends()
        assert backends == ["hip"]

    def test_is_gpu_timing_available_matches_backends(self, monkeypatch) -> None:
        """Test consistency between availability functions."""
        monkeypatch.setattr(timing_module, "_is_hip_available", lambda: True)
        backends = get_available_backends()
        assert is_gpu_timing_available() == (len(backends) > 0)

    def test_cpu_only_torch_does_not_enable_gpu_timing(self, monkeypatch) -> None:
        """CPU-only torch is importable but must not enable GPU timers."""
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(timing_module, "_is_hip_available", lambda: False)

        assert timing_module.torch_support.module_available() is True
        assert timing_module.torch_support.gpu_available() is False
        assert get_available_backends() == []
        assert is_gpu_timing_available() is False


class TestFactoryFunction:
    """Tests for create_gpu_timer factory."""

    def test_invalid_backend_raises_error(self) -> None:
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_gpu_timer("invalid")  # type: ignore

    def test_hip_backend_unavailable_raises_error(self, monkeypatch) -> None:
        """Test that requesting unavailable HIP backend raises RuntimeError."""

        def raise_unavailable():
            raise RuntimeError("HIP GPU timing not available")

        monkeypatch.setattr(timing_module, "_require_hip_runtime", raise_unavailable)
        with pytest.raises(RuntimeError, match="HIP GPU timing not available"):
            create_gpu_timer("hip")

    def test_auto_no_backend_returns_none(self, monkeypatch) -> None:
        """Test that auto with no backends returns None (graceful fallback)."""
        monkeypatch.setattr(timing_module, "_is_hip_available", lambda: False)
        assert create_gpu_timer("auto") is None

    def test_auto_creates_timer_when_available(self, monkeypatch) -> None:
        """Test auto-detection creates a timer when available."""
        monkeypatch.setattr(timing_module, "_is_hip_available", lambda: True)
        monkeypatch.setattr(timing_module, "HipGpuTimer", DummyHipTimer)
        timer = create_gpu_timer("auto")
        assert isinstance(timer, GpuTimerInterface)
        assert timer.backend_name == "hip"


class TestHipGpuTimerBackwardCompat:
    """Tests for backward compatibility aliases."""

    def test_gputimer_alias(self) -> None:
        """Test that GpuTimer is alias for HipGpuTimer."""
        assert GpuTimer is HipGpuTimer


class TestGpuTimerContextManager:
    """Tests for GPU timer context manager behavior."""

    def test_context_manager_calls_start_and_stop(self) -> None:
        """Test that context manager calls start on enter and stop on exit."""
        call_order = []

        class TrackingTimer(GpuTimerInterface):
            @property
            def backend_name(self) -> str:
                return "tracking"

            def start(self) -> None:
                call_order.append("start")

            def stop(self) -> None:
                call_order.append("stop")

            def synchronize(self) -> None:
                pass

            def elapsed_ms(self) -> float:
                return 1.0

        timer = TrackingTimer()
        with timer:
            call_order.append("inside")

        assert call_order == ["start", "inside", "stop"]

    def test_context_manager_stop_called_on_exception(self) -> None:
        """Test that stop is called even when exception occurs."""
        stop_called = False

        class ExceptionTimer(GpuTimerInterface):
            @property
            def backend_name(self) -> str:
                return "exception"

            def start(self) -> None:
                pass

            def stop(self) -> None:
                nonlocal stop_called
                stop_called = True

            def synchronize(self) -> None:
                pass

            def elapsed_ms(self) -> float:
                return 1.0

        timer = ExceptionTimer()
        with pytest.raises(RuntimeError):
            with timer:
                raise RuntimeError("test error")

        assert stop_called is True


class TestDirectHipTimers:
    """Tests for direct hipdnn_frontend HIP API usage."""

    def test_hip_gpu_timer_uses_bound_hip_events(self, monkeypatch) -> None:
        calls = []
        events = []

        class FakeEvent:
            def __init__(self) -> None:
                events.append(self)
                calls.append(("create", self))

            def record(self, stream: int) -> None:
                calls.append(("record", self, stream))

            def synchronize(self) -> None:
                calls.append(("synchronize", self))

            def elapsed_time(self, stop) -> float:
                calls.append(("elapsed", self, stop))
                return 1.25

        class FakeHipdnn:
            @staticmethod
            def hip_get_device_count() -> int:
                return 1

            HipEvent = FakeEvent

        monkeypatch.setattr(timing_module, "hipdnn", FakeHipdnn)

        timer = HipGpuTimer(stream=123)
        timer.start()
        timer.stop()
        timer.synchronize()

        assert timer.elapsed_ms() == 1.25
        assert len(events) == 2
        assert calls == [
            ("create", events[0]),
            ("create", events[1]),
            ("record", events[0], 123),
            ("record", events[1], 123),
            ("synchronize", events[1]),
            ("synchronize", events[1]),
            ("elapsed", events[0], events[1]),
        ]

    def test_hip_gpu_timer_synchronize_stream_records_event_on_stream(
        self, monkeypatch
    ) -> None:
        calls = []
        events = []

        class FakeEvent:
            def __init__(self) -> None:
                events.append(self)
                calls.append(("create", self))

            def record(self, stream: int) -> None:
                calls.append(("record", self, stream))

            def synchronize(self) -> None:
                calls.append(("synchronize", self))

            def elapsed_time(self, stop) -> float:
                return 0.0

        class FakeHipdnn:
            @staticmethod
            def hip_get_device_count() -> int:
                return 1

            HipEvent = FakeEvent

        monkeypatch.setattr(timing_module, "hipdnn", FakeHipdnn)

        timer = HipGpuTimer(stream=456)
        timer.synchronize_stream()

        assert len(events) == 2
        assert calls == [
            ("create", events[0]),
            ("create", events[1]),
            ("record", events[1], 456),
            ("synchronize", events[1]),
        ]


class TestTimingSanity:
    """Sanity tests for timing behavior."""

    def test_dummy_timer_implements_interface(self) -> None:
        """Test that DummyHipTimer properly implements the interface."""
        timer = DummyHipTimer()

        # Verify all interface methods work
        assert timer.backend_name == "hip"
        timer.start()
        timer.stop()
        timer.synchronize()
        assert timer.elapsed_ms() == 0.0

    def test_dummy_timer_context_manager(self) -> None:
        """Test that DummyHipTimer works as context manager."""
        timer = DummyHipTimer()
        with timer:
            pass
        assert timer.elapsed_ms() == 0.0
