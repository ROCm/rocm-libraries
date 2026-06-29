# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for hipDNN executor E2E timing synchronization."""

import time
import sys
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


class DummyHipTimer(GpuTimerInterface):
    """Minimal HIP timer for executor tests."""

    @property
    def backend_name(self) -> str:
        return "hip"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def elapsed_ms(self) -> float:
        return 1.0


class TrackingStreamTimer:
    """Stream-scoped HIP timer stub that records synchronization calls."""

    def __init__(self, calls: list[str], stream: int = 0) -> None:
        self.calls = calls
        self.stream = stream

    def synchronize_stream(self) -> None:
        self.calls.append("sync")


class FakeHandle:
    """Minimal handle stub with a configurable HIP stream."""

    def __init__(self, stream: int) -> None:
        self.stream = stream

    def get_stream(self) -> int:
        return self.stream


def _make_executor(collect_kernel_timing: bool = True) -> executor_module.Executor:
    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=1)
    executor = executor_module.Executor(
        "{}", config, collect_kernel_timing=collect_kernel_timing
    )
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

    monkeypatch.setattr(
        executor_module,
        "HipGpuTimer",
        lambda stream=0: TrackingStreamTimer(calls, stream),
    )

    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=3, benchmark_iters=1)
    executor = executor_module.Executor("{}", config, collect_kernel_timing=False)
    executor._graph = TrackingGraph()
    executor._workspace_ptr = 0

    executor.warmup(handle=None, variant_pack={})

    assert calls == ["execute", "execute", "execute", "sync"]


def test_warmup_zero_iterations_does_not_synchronize(monkeypatch) -> None:
    """No warmup iterations means no extra pre-benchmark synchronization."""
    calls: list[str] = []

    monkeypatch.setattr(
        executor_module,
        "HipGpuTimer",
        lambda stream=0: pytest.fail("zero warmup must not synchronize"),
    )

    executor = _make_executor(collect_kernel_timing=False)
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

    monkeypatch.setattr(
        executor_module,
        "HipGpuTimer",
        lambda stream=0: TrackingStreamTimer(calls, stream),
    )

    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=3)
    executor = executor_module.Executor("{}", config, collect_kernel_timing=False)
    executor._graph = TrackingGraph()
    executor._workspace_ptr = 0

    result = executor.benchmark(handle=None, variant_pack={})

    assert calls == ["execute", "sync", "execute", "sync", "execute", "sync"]
    assert len(result.e2e_timings) == 3


def test_benchmark_uses_handle_stream_for_timing_and_sync(monkeypatch) -> None:
    """hipDNN timing and fallback synchronization use the handle stream."""
    timer_streams: list[int] = []
    sync_streams: list[int] = []

    monkeypatch.setattr(
        executor_module,
        "create_gpu_timer",
        lambda stream=0: timer_streams.append(stream) or DummyHipTimer(),
    )
    monkeypatch.setattr(
        executor_module,
        "HipGpuTimer",
        lambda stream=0: sync_streams.append(stream) or TrackingStreamTimer([], stream),
    )

    timed_executor = _make_executor(collect_kernel_timing=True)
    timed_result = timed_executor.benchmark(handle=FakeHandle(123), variant_pack={})

    fallback_executor = _make_executor(collect_kernel_timing=False)
    fallback_executor.benchmark(handle=FakeHandle(456), variant_pack={})

    assert timer_streams == [123]
    assert sync_streams == [456]
    assert timed_result.kernel_timings == [1.0]


def test_handle_stream_change_after_prepare_raises() -> None:
    """Reject hipDNN handle stream drift so warmup and benchmark use one stream."""
    executor = _make_executor(collect_kernel_timing=False)
    executor._execution_stream = 123

    with pytest.raises(executor_module.ExecutionError, match="stream changed"):
        executor.benchmark(handle=FakeHandle(456), variant_pack={})


def test_gpu_timer_start_stop_elapsed_inside_timed_region(monkeypatch) -> None:
    """Ensure GPU timer start/stop/elapsed are invoked within Timer.

    elapsed_ms() synchronizes the stop event, so it must be called inside the
    Timer context for E2E timing to cover the full GPU work interval.
    """
    in_timer = {"value": False}
    start_called_in_timer = {"value": False}
    stop_called_in_timer = {"value": False}
    sync_called = {"value": False}
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
        """GPU timer that tracks when start/stop/synchronize/elapsed_ms are called."""

        @property
        def backend_name(self) -> str:
            return "hip"

        def start(self) -> None:
            start_called_in_timer["value"] = in_timer["value"]

        def stop(self) -> None:
            stop_called_in_timer["value"] = in_timer["value"]

        def synchronize(self) -> None:
            sync_called["value"] = True

        def elapsed_ms(self) -> float:
            elapsed_called_in_timer["value"] = in_timer["value"]
            return 1.0

    monkeypatch.setattr(executor_module, "Timer", FakeTimer)
    monkeypatch.setattr(
        executor_module,
        "create_gpu_timer",
        lambda stream=0: TrackingGpuTimer(),
    )

    executor = _make_executor(collect_kernel_timing=True)
    result = executor.benchmark(handle=None, variant_pack={})

    # start(), stop(), and elapsed_ms() should be called inside the timer context.
    assert start_called_in_timer["value"] is True
    assert stop_called_in_timer["value"] is True
    assert elapsed_called_in_timer["value"] is True
    assert sync_called["value"] is False
    assert result.kernel_timings is not None
    assert result.metadata is not None
    assert result.metadata.timing_backend == "hip"


def test_e2e_timing_at_least_as_long_as_kernel(monkeypatch) -> None:
    """Ensure E2E timing >= kernel timing for every iteration.

    With correct timer ordering, the wall-clock Timer wraps gpu_timer
    start/stop and elapsed_ms(). elapsed_ms() synchronizes the stop event, so
    E2E must always be >= kernel.
    We simulate sync overhead in elapsed_ms() so that the real Timer measures
    more wall-clock time than the fixed kernel duration.
    """
    KERNEL_MS = 0.5
    SYNC_DELAY_S = 0.01  # 10ms simulated sync overhead

    class SlowSyncGpuTimer(GpuTimerInterface):
        """GPU timer where elapsed_ms() has measurable synchronization latency."""

        @property
        def backend_name(self) -> str:
            return "hip"

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_ms(self) -> float:
            time.sleep(SYNC_DELAY_S)
            return KERNEL_MS

    monkeypatch.setattr(
        executor_module,
        "create_gpu_timer",
        lambda stream=0: SlowSyncGpuTimer(),
    )

    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=5)
    executor = executor_module.Executor("{}", config, collect_kernel_timing=True)
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
    monkeypatch.setattr(
        executor_module,
        "HipGpuTimer",
        lambda stream=0: TrackingStreamTimer([], stream),
    )

    executor = _make_executor(collect_kernel_timing=False)
    result = executor.benchmark(handle=None, variant_pack={})

    assert result.e2e_timings == [2.0]
    assert result.kernel_timings is None


def test_cpu_only_torch_does_not_affect_hip_synchronization(monkeypatch) -> None:
    """CPU-only torch must not be consulted for hipDNN timing synchronization."""
    sync_calls: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def synchronize() -> None:
            raise AssertionError("hipDNN timing must not call torch.cuda.synchronize")

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        executor_module,
        "HipGpuTimer",
        lambda stream=0: TrackingStreamTimer(sync_calls, stream),
    )

    executor = _make_executor(collect_kernel_timing=False)
    result = executor.benchmark(handle=None, variant_pack={})

    assert result.e2e_timings
    assert sync_calls == ["sync"]
