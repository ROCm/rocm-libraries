# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for hipDNN executor E2E timing synchronization."""

import sys
import time
import types
from typing import Any, Dict

import pytest

import dnn_benchmarking.execution.executor as executor_module
from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.execution.timing import GpuTimerInterface


class DummyResult:
    """Minimal execution result stub."""

    def is_bad(self) -> bool:
        return False

    def get_message(self) -> str:
        return ""


class DummyGraph:
    """Minimal hipDNN graph stub."""

    def execute(
        self, handle: Any, variant_pack: Dict[int, int], workspace_ptr: int
    ) -> DummyResult:
        return DummyResult()


class DummyTorchTimer(GpuTimerInterface):
    """Minimal torch timer for executor tests."""

    @property
    def backend_name(self) -> str:
        return "torch"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def elapsed_ms(self) -> float:
        return 1.0


def _make_executor(gpu_backend: str) -> executor_module.Executor:
    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=1)
    executor = executor_module.Executor("{}", config, gpu_backend=gpu_backend)
    executor._graph = DummyGraph()
    executor._workspace_ptr = 0
    return executor


def test_warmup_synchronizes_once_after_untimed_iterations(monkeypatch) -> None:
    """hipDNN warmup work is drained before the measured loop starts."""
    calls: list[str] = []

    class TrackingGraph:
        def execute(
            self, handle: Any, variant_pack: Dict[int, int], workspace_ptr: int
        ) -> DummyResult:
            calls.append("execute")
            return DummyResult()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("sync"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=3, benchmark_iters=1)
    executor = executor_module.Executor("{}", config, gpu_backend="none")
    executor._graph = TrackingGraph()
    executor._workspace_ptr = 0

    executor.warmup(handle=None, variant_pack={})

    assert calls == ["execute", "execute", "execute", "sync"]


def test_warmup_zero_iterations_does_not_synchronize(monkeypatch) -> None:
    """No warmup iterations means no extra pre-benchmark synchronization."""
    calls: list[str] = []

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("sync"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    executor = _make_executor("none")
    executor.warmup(handle=None, variant_pack={})

    assert calls == []


def test_benchmark_synchronizes_each_measured_iteration(monkeypatch) -> None:
    """Each timed hipDNN iteration executes once and then synchronizes once."""
    calls: list[str] = []

    class TrackingGraph:
        def execute(
            self, handle: Any, variant_pack: Dict[int, int], workspace_ptr: int
        ) -> DummyResult:
            calls.append("execute")
            return DummyResult()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("sync"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=3)
    executor = executor_module.Executor("{}", config, gpu_backend="none")
    executor._graph = TrackingGraph()
    executor._workspace_ptr = 0

    result = executor.benchmark(handle=None, variant_pack={})

    assert calls == ["execute", "sync", "execute", "sync", "execute", "sync"]
    assert len(result.e2e_timings) == 3


def test_gpu_timer_start_stop_inside_timed_region(monkeypatch) -> None:
    """Ensure GPU timer start/stop are invoked within the Timer context.

    start() and stop() must be called inside the Timer so that E2E timing
    covers the full GPU work interval. elapsed_ms() is called outside since
    it only reads back the recorded duration.
    """
    in_timer = {"value": False}
    start_called_in_timer = {"value": False}
    stop_called_in_timer = {"value": False}
    elapsed_called_in_timer = {"value": False}

    class FakeTimer:
        def __enter__(self) -> "FakeTimer":
            in_timer["value"] = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            in_timer["value"] = False

        @property
        def elapsed_ms(self) -> float:
            return 1.0

    class TrackingGpuTimer(GpuTimerInterface):
        """GPU timer that tracks when start/stop/elapsed_ms are called."""

        @property
        def backend_name(self) -> str:
            return "torch"

        def start(self) -> None:
            start_called_in_timer["value"] = in_timer["value"]

        def stop(self) -> None:
            stop_called_in_timer["value"] = in_timer["value"]

        def elapsed_ms(self) -> float:
            elapsed_called_in_timer["value"] = in_timer["value"]
            return 1.0

    monkeypatch.setattr(executor_module, "Timer", FakeTimer)
    monkeypatch.setattr(
        executor_module, "create_gpu_timer", lambda backend: TrackingGpuTimer()
    )

    executor = _make_executor("torch")
    result = executor.benchmark(handle=None, variant_pack={})

    # start() and stop() should be called inside the timer context
    assert start_called_in_timer["value"] is True
    assert stop_called_in_timer["value"] is True
    # elapsed_ms() should be called outside the timer context
    assert elapsed_called_in_timer["value"] is False
    assert result.kernel_timings is not None
    assert result.metadata is not None
    assert result.metadata.gpu_backend == "torch"


def test_e2e_timing_at_least_as_long_as_kernel(monkeypatch) -> None:
    """Ensure E2E timing >= kernel timing for every iteration.

    With correct timer ordering, the wall-clock Timer wraps gpu_timer
    start/stop and synchronization, so E2E must always be >= kernel.
    We simulate sync overhead in stop() so that the real Timer measures
    more wall-clock time than the fixed kernel duration.
    """
    KERNEL_MS = 0.5
    SYNC_DELAY_S = 0.01  # 10ms simulated sync overhead

    class SlowSyncGpuTimer(GpuTimerInterface):
        """GPU timer where stop() has measurable latency (simulating sync)."""

        @property
        def backend_name(self) -> str:
            return "torch"

        def start(self) -> None:
            pass

        def stop(self) -> None:
            time.sleep(SYNC_DELAY_S)

        def elapsed_ms(self) -> float:
            return KERNEL_MS

    monkeypatch.setattr(
        executor_module, "create_gpu_timer", lambda backend: SlowSyncGpuTimer()
    )

    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=5)
    executor = executor_module.Executor("{}", config, gpu_backend="torch")
    executor._graph = DummyGraph()
    executor._workspace_ptr = 0

    result = executor.benchmark(handle=None, variant_pack={})

    assert result.kernel_timings is not None
    assert len(result.e2e_timings) == 5
    assert len(result.kernel_timings) == 5

    for i, (e2e, kernel) in enumerate(zip(result.e2e_timings, result.kernel_timings)):
        assert (
            e2e >= kernel
        ), f"Iteration {i}: E2E ({e2e:.3f}ms) must be >= kernel ({kernel:.3f}ms)"


def test_e2e_timings_recorded_without_gpu_timing(monkeypatch) -> None:
    """Ensure E2E timings are recorded even when GPU timing is disabled."""

    class FakeTimer:
        def __enter__(self) -> "FakeTimer":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            pass

        @property
        def elapsed_ms(self) -> float:
            return 2.0

    monkeypatch.setattr(executor_module, "Timer", FakeTimer)

    executor = _make_executor("none")
    result = executor.benchmark(handle=None, variant_pack={})

    assert result.e2e_timings == [2.0]
    assert result.kernel_timings is None
