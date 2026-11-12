# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-released device stall, for timing one call the way hipDNN does.

:func:`rocke.runtime.time_launches` times N calls back to back between one
pair of events and divides. hipDNN's benchmark (``dnn-benchmark``, "staged"
timing) times every call on its own instead: it parks a wait packet on the
stream, records the start event, enqueues the work, records the stop event and
only then releases the wait from the host. The event span is gap-free device
time -- the host's submission cost is outside it -- but it does include the
fixed device cost of a start/stop event pair around a single call, which a
back-to-back loop amortises away. A sweep that is compared against hipDNN or
MIOpen numbers therefore has to measure the same way, or its small kernels look
faster than they will in that comparison.

:class:`StallGate` is the ctypes twin of ``hipdnn_data_sdk::utilities::StallGate``
(``hipStreamWaitValue32`` on HIP signal memory, released by a host store, with a
watchdog), and :func:`time_launches_staged` is the twin of dnn-benchmark's
``StalledRegionTimer.measure`` loop.

The watchdog
------------
The release is a host action that runs after the measured call returns. A call
that blocks the host on the stalled stream (a fenced launch, a synchronous
``hipMemset``) would wait forever for a stream only the host can release, so a
watchdog thread writes the release itself after ``timeout_s``. That turns the
hang into one slow, invalid sample: :meth:`StallGate.timed_out` reports it and
:func:`time_launches_staged` raises :class:`StallTimeout` rather than return a
number that includes host time.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable, List, Optional

from .hip_module import (
    HipError,
    Runtime,
    _check,
    _hipDeviceGetAttribute,
    _hipExtMallocWithFlags,
    _hipFree,
    _hipGetDevice,
    _hipStreamWaitValue32,
)

__all__ = ["StallGate", "StallTimeout", "default_gate", "time_launches_staged"]

# hipDeviceAttributeCanUseStreamWaitValue (AMD-specific block of the
# hipDeviceAttribute_t enum), hipMallocSignalMemory, hipStreamWaitValueGte.
_HIP_ATTR_CAN_USE_STREAM_WAIT_VALUE = 10013
_HIP_MALLOC_SIGNAL_MEMORY = 0x2
_HIP_STREAM_WAIT_VALUE_GTE = 0x0

#: Host time allowed between arm and release, as in hipDNN's StallGate.
DEFAULT_TIMEOUT_S = 2.0


class StallTimeout(RuntimeError):
    """The watchdog, not the host, released the gate: the sample holds host time."""


class StallGate:
    """Hold every later item on a stream until the host releases it.

    Usage per sample: :meth:`arm`, record the start event, enqueue the work,
    record the stop event, :meth:`release`, synchronize the stop event. The
    stream must have drained past the previous wait before the next
    :meth:`arm` -- synchronizing the stop event is what guarantees it.

    One gate arms one stream at a time. :meth:`close` (or leaving a ``with``
    block) releases any pending wait, stops the watchdog and frees the signal.
    """

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s = float(timeout_s) if timeout_s > 0 else DEFAULT_TIMEOUT_S
        self._signal = 0
        self._cv = threading.Condition()
        self._armed = False
        self._timed_out = False
        self._stop = False
        self._deadline = 0.0
        self._watchdog: Optional[threading.Thread] = None

        device = ctypes.c_int(0)
        _check(_hipGetDevice(ctypes.byref(device)), "hipGetDevice")
        can = ctypes.c_int(0)
        _check(
            _hipDeviceGetAttribute(
                ctypes.byref(can), _HIP_ATTR_CAN_USE_STREAM_WAIT_VALUE, device.value
            ),
            "hipDeviceGetAttribute(CanUseStreamWaitValue)",
        )
        if not can.value:
            raise HipError(
                f"device {device.value} does not support hipStreamWaitValue32; "
                f"staged timing needs it"
            )
        # Signal memory is an 8-byte HSA signal; the 32-bit wait reads its low word.
        ptr = ctypes.c_void_p(0)
        _check(
            _hipExtMallocWithFlags(ctypes.byref(ptr), 8, _HIP_MALLOC_SIGNAL_MEMORY),
            "hipExtMallocWithFlags(hipMallocSignalMemory)",
        )
        self._signal = int(ptr.value or 0)
        self._write(1)

    def _write(self, value: int) -> None:
        ctypes.c_uint32.from_address(self._signal).value = value

    def _watch(self) -> None:
        with self._cv:
            while not self._stop:
                if not self._armed:
                    self._cv.wait()
                    continue
                left = self._deadline - time.monotonic()
                if left > 0:
                    self._cv.wait(left)
                    continue
                # The blocked host is waiting on this stream; this store both
                # ends the stall and unblocks it.
                self._write(1)
                self._armed = False
                self._timed_out = True

    def arm(self, stream: int) -> None:
        """Enqueue a wait on ``stream`` that holds everything queued after it."""
        if not self._signal:
            raise HipError("StallGate is closed")
        with self._cv:
            if self._armed:
                raise HipError("StallGate is already armed")
            self._timed_out = False
        if self._watchdog is None:
            self._watchdog = threading.Thread(
                target=self._watch, name="rocke-stall-gate", daemon=True
            )
            self._watchdog.start()
        self._write(0)
        status = _hipStreamWaitValue32(
            ctypes.c_void_p(int(stream)),
            ctypes.c_void_p(self._signal),
            1,
            _HIP_STREAM_WAIT_VALUE_GTE,
            0xFFFFFFFF,
        )
        if status != 0:
            # No wait was enqueued to ever raise the signal; put it back.
            self._write(1)
            _check(status, "hipStreamWaitValue32")
        with self._cv:
            self._armed = True
            self._deadline = time.monotonic() + self._timeout_s
            self._cv.notify_all()

    def release(self) -> None:
        """Let the stalled work run. A no-op when not armed."""
        with self._cv:
            if not self._armed:
                return
            self._write(1)
            self._armed = False
            self._cv.notify_all()

    def timed_out(self) -> bool:
        """Whether the watchdog, not :meth:`release`, ended the last :meth:`arm`."""
        with self._cv:
            return self._timed_out

    def close(self) -> None:
        """Release, stop the watchdog and free the signal. Idempotent."""
        with self._cv:
            if self._signal:
                self._write(1)
            self._armed = False
            self._stop = True
            self._cv.notify_all()
        if self._watchdog is not None:
            self._watchdog.join()
            self._watchdog = None
        if self._signal:
            # hipFree synchronizes the device first, so the released wait
            # packet has retired before its signal goes away.
            _hipFree(ctypes.c_void_p(self._signal))
            self._signal = 0

    def __enter__(self) -> "StallGate":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - HIP may be gone at teardown
            pass


_default_gate: Optional[StallGate] = None


def default_gate() -> StallGate:
    """The process's shared gate, created on first use and closed at exit.

    A gate costs a signal allocation, a watchdog thread and, on close, a device
    drain; a sweep that times thousands of kernels one at a time shares one
    rather than paying that per kernel. A gate is bound to the device current
    when it is made, which is the one device a sweep process uses.
    """
    global _default_gate
    if _default_gate is None:
        import atexit

        _default_gate = StallGate()
        atexit.register(_default_gate.close)
    return _default_gate


def time_launches_staged(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    stream: int = 0,
    gate: Optional[StallGate] = None,
    runtime: Optional[Runtime] = None,
) -> List[float]:
    """Per-call device times of ``fn`` in milliseconds, one stalled sample each.

    Every warmup and timed call is its own sample: arm the gate, record the
    start event, call ``fn``, record the stop event, release, wait on the stop
    event. Warmup samples are discarded. The caller summarises the returned
    list (dnn-benchmark reports the median).

    ``fn`` runs under :func:`~rocke.runtime.launcher.no_fence`, so a launcher
    call inside it does not synchronize with the host. ``stream`` is resolved
    as :func:`~rocke.runtime.time_launches` resolves it, and ``gate`` defaults
    to :func:`default_gate`. Raises :class:`StallTimeout` when ``fn`` blocked
    the host on the stalled stream (the watchdog released it); the device is
    drained before any exception propagates.
    """
    from .launcher import no_fence
    from .torch_interop import resolve_stream

    rt = runtime if runtime is not None else Runtime()
    stream = resolve_stream(stream)
    g = default_gate() if gate is None else gate
    start = rt.event()
    stop = rt.event()
    out: List[float] = []
    try:
        rt.sync()
        with no_fence():
            for i in range(int(warmup) + int(iters)):
                g.arm(stream)
                try:
                    start.record(stream=stream)
                    fn()
                    stop.record(stream=stream)
                finally:
                    g.release()
                stop.synchronize()
                if g.timed_out():
                    raise StallTimeout(
                        "the measured call blocked the host on the stalled "
                        "stream; the stall watchdog released it"
                    )
                if i >= warmup:
                    out.append(start.elapsed_to(stop))
    except Exception:
        try:
            rt.sync()
        except Exception:  # noqa: BLE001, S110 - the call's error is the one to see
            pass
        raise
    finally:
        start.destroy()
        stop.destroy()
    rt.wait_stream(stream)
    return out
