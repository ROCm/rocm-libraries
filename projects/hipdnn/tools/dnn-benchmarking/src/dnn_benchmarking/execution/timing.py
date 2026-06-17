# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Timing utilities for benchmark execution.

GPU Kernel Timing
-----------------
Two GPU timer backends are supported:

* ``hip`` — HIP runtime events exposed by ``hipdnn_frontend``. This keeps
  benchmark timing on the same runtime used by hipDNN itself and avoids
  depending on PyTorch's CUDA/ROCm wrappers for event creation, event
  synchronization, or elapsed-time calculation. Preferred on ROCm hosts.
* ``torch`` — ``torch.cuda.Event`` timing through PyTorch's CUDA/ROCm
  facade. The only backend available on CUDA hosts, where
  ``hipdnn_frontend`` is not installed.

Stream Context
--------------
hipDNN handles default to the native HIP default stream. ``HipGpuTimer``
records events on the stream pointer provided by the executor, so
external-library work and timing markers are enqueued in the same HIP
stream. ``TorchGpuTimer`` records events on the provided torch stream
(PyTorch's default stream when omitted).
"""

import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, Optional, Type

from ..common import torch_support
from ..config.benchmark_config import TimingBackendName


_HIP_EVENT_API = ("HipEvent", "hip_get_device_count")

# Lazily imported hipdnn_frontend module. Resolved on first HIP-timer use so
# the tool stays importable on hosts without hipDNN (e.g. CUDA machines
# running the PyTorch backend). Tests may inject a fake module here.
hipdnn: Optional[Any] = None


def _load_hipdnn() -> Any:
    """Return hipdnn_frontend, importing it lazily on first use."""
    global hipdnn
    if hipdnn is not None:
        return hipdnn
    try:
        import hipdnn_frontend
    except ImportError as e:
        raise RuntimeError(
            f"HIP GPU timing not available: hipdnn_frontend is not importable: {e}"
        ) from e
    hipdnn = hipdnn_frontend
    return hipdnn


def _validate_hip_event_api(module: Any) -> None:
    """Verify hipdnn_frontend provides the HIP event bindings."""
    missing = [name for name in _HIP_EVENT_API if not hasattr(module, name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"hipdnn_frontend is missing HIP event bindings: {joined}")


def _require_hip_runtime() -> Any:
    """Return hipdnn_frontend when at least one HIP device is visible."""
    module = _load_hipdnn()
    _validate_hip_event_api(module)
    if int(module.hip_get_device_count()) <= 0:
        raise RuntimeError("HIP GPU timing not available: no HIP devices are visible")
    return module


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
    backends: List[str] = []
    if _is_hip_available():
        backends.append(TimingBackendName.HIP.value)
    if torch_support.gpu_available():
        backends.append(TimingBackendName.TORCH.value)
    return backends


def is_gpu_timing_available() -> bool:
    """Check if any GPU timing backend is available."""
    return len(get_available_backends()) > 0


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
        return TimingBackendName.HIP.value

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


class TorchGpuTimer(GpuTimerInterface):
    """GPU kernel timing using torch.cuda events.

    Used on CUDA hosts where hipdnn HIP event bindings are unavailable.
    Events are recorded on the provided torch stream so timing brackets
    the same stream the executor enqueues graph work on.
    """

    @property
    def backend_name(self) -> str:
        """Return 'torch' as the backend name."""
        return TimingBackendName.TORCH.value

    def __init__(self, stream: Optional[Any] = None) -> None:
        """Initialize GPU timer with torch CUDA events.

        Args:
            stream: torch.cuda.Stream to record events on. ``None`` uses
                PyTorch's default stream, which corresponds to the native
                CUDA/HIP default stream and so also captures work submitted
                by external libraries.

        Raises:
            RuntimeError: If PyTorch GPU support is unavailable.
        """
        if not torch_support.gpu_available():
            raise RuntimeError(
                "Torch GPU timing not available: torch reports no usable GPU"
            )
        import torch

        self._stream = stream if stream is not None else torch.cuda.default_stream()
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._stop_event = torch.cuda.Event(enable_timing=True)
        self._stop_recorded = False

    def start(self) -> None:
        """Record the start event on the configured torch stream."""
        self._start_event.record(self._stream)
        self._stop_recorded = False

    def stop(self) -> None:
        """Record the stop event on the configured torch stream."""
        self._stop_event.record(self._stream)
        self._stop_recorded = True

    def synchronize(self) -> None:
        """Block until the recorded stop event has completed."""
        if not self._stop_recorded:
            raise RuntimeError("Torch timer stop event has not been recorded")
        self._stop_event.synchronize()

    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed time in milliseconds."""
        self.synchronize()
        return float(self._start_event.elapsed_time(self._stop_event))


def create_gpu_timer(
    backend: TimingBackendName = TimingBackendName.AUTO,
    stream: int = 0,
    torch_stream: Optional[Any] = None,
) -> Optional[GpuTimerInterface]:
    """Create a GPU timer for the specified or detected backend.

        backend: Timer backend enum to use. ``AUTO`` prefers direct HIP
            event timing and falls back to torch event timing only when
            ``torch_stream`` is provided.
        stream: HIP stream pointer encoded as an integer (hip backend).
        torch_stream: torch.cuda.Stream for the torch backend. ``auto``
            only falls back to torch timing when this is set, because
            callers driving work through raw HIP streams have no torch
            stream for the events to bracket.

    Returns:
        GpuTimerInterface implementation, or None when ``auto`` cannot find one.

    Raises:
        RuntimeError: If a requested backend is not available.
        TypeError: If backend is not a TimingBackendName.
    """
    if not isinstance(backend, TimingBackendName):
        raise TypeError("backend must be a TimingBackendName")

    if backend is TimingBackendName.AUTO:
        if _is_hip_available():
            return HipGpuTimer(stream)
        if torch_stream is not None and torch_support.gpu_available():
            return TorchGpuTimer(torch_stream)
        return None

    if backend is TimingBackendName.HIP:
        return HipGpuTimer(stream)

    if backend is TimingBackendName.TORCH:
        return TorchGpuTimer(torch_stream)

    if backend is TimingBackendName.NONE:
        return None

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
