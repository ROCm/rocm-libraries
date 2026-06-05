# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Timing utilities for benchmark execution.

GPU Kernel Timing
-----------------
This module provides GPU kernel timing using PyTorch CUDA/ROCm events.

Stream Context Issue
--------------------
PyTorch maintains its own stream management separate from the underlying
CUDA/HIP runtime. By default, torch.cuda.Event.record() records on
torch.cuda.current_stream(), which is PyTorch's per-thread managed stream.

However, external libraries like hipDNN execute kernels via the native C API,
which uses the CUDA/HIP default stream (stream 0), not PyTorch's current stream.

Solution
--------
TorchGpuTimer explicitly records events on torch.cuda.default_stream(), which
corresponds to the native CUDA/HIP default stream (stream 0). This ensures we
correctly capture kernel execution from external libraries.

Verification
------------
Testing confirms that torch.cuda.default_stream() captures hipDNN kernel
execution with <5% difference compared to events recorded after full device
synchronization, providing accurate pure GPU kernel timing.
"""

import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import List, Literal, Optional, Type


def _is_torch_available() -> bool:
    """Check if PyTorch GPU support is available.

    Returns:
        True if torch.cuda.is_available() returns True, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """Return list of available GPU timer backends.

    Returns:
        List of backend names (e.g., ["torch"] or []).
    """
    backends = []
    if _is_torch_available():
        backends.append("torch")
    return backends


def is_gpu_timing_available() -> bool:
    """Check if any GPU timing backend is available.

    Returns:
        True if PyTorch GPU timing is available, False otherwise.
    """
    return _is_torch_available()


class GpuTimerInterface(ABC):
    """Abstract interface for GPU kernel timing.

    Supports context manager protocol for convenient timing blocks,
    as well as explicit start/stop control for fine-grained timing.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name (e.g., 'torch')."""
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


class TorchGpuTimer(GpuTimerInterface):
    """GPU kernel timing using PyTorch CUDA/ROCm events.

    Uses torch.cuda.Event for timing on supported GPUs (NVIDIA or AMD ROCm).

    Events are recorded on the default stream to capture kernels launched by
    external libraries (e.g., hipDNN) that execute on the native CUDA/HIP
    default stream rather than PyTorch's current stream.

    Example:
        timer = TorchGpuTimer()
        timer.start()
        # GPU kernel execution
        timer.stop()
        elapsed = timer.elapsed_ms()
    """

    @property
    def backend_name(self) -> str:
        """Return 'torch' as the backend name."""
        return "torch"

    def __init__(self) -> None:
        """Initialize GPU timer with PyTorch events.

        Raises:
            RuntimeError: If PyTorch GPU is not available.
        """
        if not _is_torch_available():
            raise RuntimeError("PyTorch GPU not available for GPU timing")

        import torch

        self._torch = torch
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._stop_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        """Record the start event on the default CUDA stream.

        Uses torch.cuda.default_stream() to ensure we capture kernels
        launched by external libraries (e.g., hipDNN) that use the native
        CUDA/HIP default stream.
        """
        stream = self._torch.cuda.default_stream()
        self._start_event.record(stream=stream)

    def stop(self) -> None:
        """Record the stop event on the default CUDA stream.

        Uses torch.cuda.default_stream() to ensure we capture kernels
        launched by external libraries (e.g., hipDNN) that use the native
        CUDA/HIP default stream.
        """
        stream = self._torch.cuda.default_stream()
        self._stop_event.record(stream=stream)

    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds between start and stop events.
        """
        self._stop_event.synchronize()
        return self._start_event.elapsed_time(self._stop_event)


def create_gpu_timer(
    backend: Optional[Literal["torch", "auto"]] = "auto",
) -> Optional[GpuTimerInterface]:
    """Create a GPU timer for the specified or detected backend.

    Args:
        backend: Timer backend to use:
            - "torch": Force PyTorch backend (CUDA or ROCm)
            - "auto": Auto-detect (uses PyTorch if available)

    Returns:
        GpuTimerInterface implementation.

    Raises:
        RuntimeError: If requested backend is not available.
        ValueError: If invalid backend is specified.
    """
    if backend == "auto" or backend is None:
        if _is_torch_available():
            return TorchGpuTimer()
        return None

    if backend == "torch":
        if not _is_torch_available():
            raise RuntimeError("PyTorch GPU not available for GPU timing")
        return TorchGpuTimer()

    raise ValueError(f"Unknown backend: {backend}")


# Backward compatibility alias
GpuTimer = TorchGpuTimer


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
