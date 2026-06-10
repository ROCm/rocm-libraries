# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Timing utilities for benchmark execution.

GPU Kernel Timing
-----------------
GPU timings are collected with HIP runtime events exposed by
``hipdnn_frontend``. This keeps benchmark timing on the same runtime used by
hipDNN itself and avoids depending on PyTorch's CUDA/ROCm wrappers for event
creation, event synchronization, or elapsed-time calculation.

Stream Context
--------------
hipDNN handles default to the native HIP default stream. ``HipGpuTimer`` records
events on the stream pointer provided by the executor, so external-library work
and timing markers are enqueued in the same HIP stream.
"""

import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, Literal, Optional, Type

import hipdnn_frontend as hipdnn


_HIP_EVENT_API = ("HipEvent", "hip_get_device_count")


def _validate_hip_event_api() -> None:
    """Verify hipdnn_frontend provides the HIP event bindings."""
    missing = [name for name in _HIP_EVENT_API if not hasattr(hipdnn, name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"hipdnn_frontend is missing HIP event bindings: {joined}")


def _require_hip_runtime() -> Any:
    """Return hipdnn_frontend when at least one HIP device is visible."""
    _validate_hip_event_api()
    if int(hipdnn.hip_get_device_count()) <= 0:
        raise RuntimeError("HIP GPU timing not available: no HIP devices are visible")
    return hipdnn


from ..common import torch_support


def _is_torch_available() -> bool:
    """Backward-compatible alias for PyTorch GPU timing availability."""
    return torch_support.gpu_available()


def _is_hip_available() -> bool:
    """Return True when direct HIP timing bindings and a HIP device exist."""
    try:
        _require_hip_runtime()
        return True
    except Exception:
        return False


def get_available_backends() -> List[str]:
    """Return list of available GPU timer backends."""
    return ["hip"] if _is_hip_available() else []


def is_gpu_timing_available() -> bool:
    """Check if direct HIP GPU timing is available."""
    return _is_hip_available()


class GpuTimerInterface(ABC):
    """Abstract interface for GPU kernel timing.

    Supports context manager protocol for convenient timing blocks,
    as well as explicit start/stop/synchronize control for fine-grained timing.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name (e.g., 'hip')."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Record the start timestamp on the GPU stream."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Record the stop timestamp on the GPU stream."""
        ...

    @abstractmethod
    def synchronize(self) -> None:
        """Block until the stop timestamp has completed."""
        ...

    @abstractmethod
    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed time in milliseconds.

        Must be called after stop(). Blocks until GPU operations complete.
        """
        ...

    def __enter__(self) -> "GpuTimerInterface":
        """Context manager entry - records start."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit - records stop."""
        self.stop()


class HipGpuTimer(GpuTimerInterface):
    """GPU kernel timing using direct HIP runtime events."""

    @property
    def backend_name(self) -> str:
        """Return 'hip' as the backend name."""
        return "hip"

    def __init__(self, stream: int = 0) -> None:
        """Initialize GPU timer with HIP events.

        Args:
            stream: HIP stream pointer encoded as an integer. ``0`` is the
                default stream.

        Raises:
            RuntimeError: If HIP event bindings or a HIP device are unavailable.
        """
        self._hipdnn = _require_hip_runtime()
        self._stream = int(stream)
        self._start_event = self._hipdnn.HipEvent()
        self._stop_event = self._hipdnn.HipEvent()
        self._recorded_stop_event: Optional[Any] = None

    def start(self) -> None:
        """Record the start event on the configured HIP stream."""
        self._start_event.record(self._stream)
        self._recorded_stop_event = None

    def _record_stop_event(self) -> None:
        """Record the stop event on the configured HIP stream."""
        self._stop_event.record(self._stream)
        self._recorded_stop_event = self._stop_event

    def stop(self) -> None:
        """Record the stop event on the configured HIP stream."""
        self._record_stop_event()

    def synchronize(self) -> None:
        """Block until the recorded stop event has completed."""
        if self._recorded_stop_event is None:
            raise RuntimeError("HIP timer stop event has not been recorded")
        self._recorded_stop_event.synchronize()

    def synchronize_stream(self) -> None:
        """Record and wait for an event on the configured HIP stream."""
        self._record_stop_event()
        self.synchronize()

    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed time in milliseconds."""
        self.synchronize()
        return float(self._start_event.elapsed_time(self._stop_event))


def create_gpu_timer(
    backend: Optional[Literal["hip", "auto"]] = "auto",
    stream: int = 0,
) -> Optional[GpuTimerInterface]:
    """Create a GPU timer for the specified or detected backend.

    Args:
        backend: Timer backend to use:
            - "hip": Force direct HIP event timing
            - "auto": Auto-detect direct HIP event timing
        stream: HIP stream pointer encoded as an integer.

    Returns:
        GpuTimerInterface implementation, or None when ``auto`` cannot find one.

    Raises:
        RuntimeError: If a requested backend is not available.
        ValueError: If invalid backend is specified.
    """
    if backend == "auto" or backend is None:
        if _is_hip_available():
            return HipGpuTimer(stream)
        return None

    if backend == "hip":
        return HipGpuTimer(stream)

    raise ValueError(f"Unknown backend: {backend}")


# Backward compatibility alias.
GpuTimer = HipGpuTimer


class Timer:
    """Context manager for measuring wall-clock execution time.

    Uses time.perf_counter() for high-resolution timing.

    Example:
        with Timer() as t:
            # code to time
            pass
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self) -> None:
        """Initialize timer with zero elapsed time."""
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop timing."""
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self._end - self._start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self._end - self._start
