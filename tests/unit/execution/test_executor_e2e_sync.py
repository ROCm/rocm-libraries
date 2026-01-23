"""Tests for hipDNN executor E2E timing synchronization."""

import sys
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

    def execute(self, handle: Any, variant_pack: Dict[int, int], workspace_ptr: int) -> DummyResult:
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


def test_sync_called_inside_timed_region(monkeypatch) -> None:
    """Ensure GPU timer elapsed_ms (which syncs) is invoked within the Timer context.

    The elapsed_ms() method internally calls synchronize on the stop event,
    so calling it inside the Timer ensures E2E timing includes sync time.
    """
    in_timer = {"value": False}
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
        """GPU timer that tracks when elapsed_ms is called."""

        @property
        def backend_name(self) -> str:
            return "torch"

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def elapsed_ms(self) -> float:
            elapsed_called_in_timer["value"] = in_timer["value"]
            return 1.0

    monkeypatch.setattr(executor_module, "Timer", FakeTimer)
    monkeypatch.setattr(executor_module, "create_gpu_timer", lambda backend: TrackingGpuTimer())

    executor = _make_executor("torch")
    result = executor.benchmark(handle=None, variant_pack={})

    # elapsed_ms() should be called inside the timer context
    # (this ensures E2E timing includes the sync that elapsed_ms performs)
    assert elapsed_called_in_timer["value"] is True
    assert result.kernel_timings is not None
    assert result.metadata is not None
    assert result.metadata.gpu_backend == "torch"


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
