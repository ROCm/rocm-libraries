# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Timer and GPU timing utilities."""

import time

import pytest

import dnn_benchmarking.execution.timing as timing_module
from dnn_benchmarking.execution.timing import (
    GpuTimer,
    GpuTimerInterface,
    Timer,
    TorchGpuTimer,
    create_gpu_timer,
    get_available_backends,
    is_gpu_timing_available,
)

# Import shared test fixture
from tests.conftest import DummyTorchTimer


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
        monkeypatch.setattr(timing_module, "_is_torch_available", lambda: True)
        backends = get_available_backends()
        assert backends == ["torch"]

    def test_is_gpu_timing_available_matches_backends(self, monkeypatch) -> None:
        """Test consistency between availability functions."""
        monkeypatch.setattr(timing_module, "_is_torch_available", lambda: True)
        backends = get_available_backends()
        assert is_gpu_timing_available() == (len(backends) > 0)


class TestFactoryFunction:
    """Tests for create_gpu_timer factory."""

    def test_invalid_backend_raises_error(self) -> None:
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_gpu_timer("invalid")  # type: ignore

    def test_torch_backend_unavailable_raises_error(self, monkeypatch) -> None:
        """Test that requesting unavailable torch backend raises RuntimeError."""
        monkeypatch.setattr(timing_module, "_is_torch_available", lambda: False)
        with pytest.raises(RuntimeError, match="PyTorch GPU not available"):
            create_gpu_timer("torch")

    def test_auto_no_backend_returns_none(self, monkeypatch) -> None:
        """Test that auto with no backends returns None (graceful fallback)."""
        monkeypatch.setattr(timing_module, "_is_torch_available", lambda: False)
        assert create_gpu_timer("auto") is None

    def test_auto_creates_timer_when_available(self, monkeypatch) -> None:
        """Test auto-detection creates a timer when available."""
        monkeypatch.setattr(timing_module, "_is_torch_available", lambda: True)
        monkeypatch.setattr(timing_module, "TorchGpuTimer", DummyTorchTimer)
        timer = create_gpu_timer("auto")
        assert isinstance(timer, GpuTimerInterface)
        assert timer.backend_name == "torch"


class TestTorchGpuTimerBackwardCompat:
    """Tests for backward compatibility aliases."""

    def test_gputimer_alias(self) -> None:
        """Test that GpuTimer is alias for TorchGpuTimer."""
        assert GpuTimer is TorchGpuTimer


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

            def elapsed_ms(self) -> float:
                return 1.0

        timer = ExceptionTimer()
        with pytest.raises(RuntimeError):
            with timer:
                raise RuntimeError("test error")

        assert stop_called is True


class TestTimingSanity:
    """Sanity tests for timing behavior."""

    def test_dummy_timer_implements_interface(self) -> None:
        """Test that DummyTorchTimer properly implements the interface."""
        timer = DummyTorchTimer()

        # Verify all interface methods work
        assert timer.backend_name == "torch"
        timer.start()
        timer.stop()
        assert timer.elapsed_ms() == 0.0

    def test_dummy_timer_context_manager(self) -> None:
        """Test that DummyTorchTimer works as context manager."""
        timer = DummyTorchTimer()
        with timer:
            pass
        assert timer.elapsed_ms() == 0.0
