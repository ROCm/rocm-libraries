# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Host-side probes: CPU time delta and host memory snapshot.

``CpuTimeProbe`` is a context manager that diffs the calling process's
user/kernel CPU time across a block, sampled via ``os.times`` so it behaves
the same on POSIX and Windows. ``host_memory_snapshot`` returns process RSS
and host RAM availability via ``psutil`` if installed. Both degrade
gracefully: on any failure they yield ``None`` values rather than raising.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ._diagnostic import warn_once


def _process_cpu_times() -> Optional[Tuple[float, float]]:
    """Return ``(user, kernel)`` CPU time for the calling process, in seconds.

    Uses ``os.times``, the cross-platform stdlib accessor (backed by
    ``GetProcessTimes`` on Windows). The probe wraps the whole benchmark
    loop and reports a per-iteration average, so ``os.times``'s clock-tick
    resolution is immaterial here. Returns ``None`` if it is unavailable,
    so callers can degrade gracefully.
    """
    try:
        times = os.times()
        return (times.user, times.system)
    except (AttributeError, OSError) as e:
        warn_once("cpu_time", f"os.times() failed: {e}")
        return None


@dataclass
class CpuTimeDelta:
    """CPU time consumed by the calling process between two probe samples.

    Attributes:
        user_time_ms: User-space CPU time in milliseconds.
        kernel_time_ms: Kernel-space CPU time in milliseconds.
    """

    user_time_ms: float
    kernel_time_ms: float


class CpuTimeProbe:
    """Context manager that measures CPU user/kernel time over a block.

    Usage::

        with CpuTimeProbe() as probe:
            do_work()
        delta = probe.delta  # CpuTimeDelta or None on failure
    """

    def __init__(self) -> None:
        self._start: Optional[Tuple[float, float]] = None
        self.delta: Optional[CpuTimeDelta] = None

    def __enter__(self) -> "CpuTimeProbe":
        self._start = _process_cpu_times()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._start is None:
            return
        end = _process_cpu_times()
        if end is None:
            return
        start_user, start_kernel = self._start
        end_user, end_kernel = end
        self.delta = CpuTimeDelta(
            user_time_ms=(end_user - start_user) * 1000.0,
            kernel_time_ms=(end_kernel - start_kernel) * 1000.0,
        )


def is_psutil_available() -> bool:
    """Return True when the psutil import succeeds."""
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


def host_memory_snapshot() -> Dict[str, Optional[float]]:
    """Return process RSS and host RAM availability in MB.

    Both values are ``None`` if ``psutil`` is unavailable or the read
    fails. The two-key shape is stable so callers can serialise without
    branching on availability.

    Returns:
        Dict with keys ``host_rss_mb`` and ``host_ram_available_mb``.
    """
    out: Dict[str, Optional[float]] = {
        "host_rss_mb": None,
        "host_ram_available_mb": None,
    }
    try:
        import psutil
    except ImportError:
        warn_once("psutil", "module not installed; host memory metrics disabled")
        return out

    try:
        proc = psutil.Process()
        out["host_rss_mb"] = proc.memory_info().rss / (1024.0 * 1024.0)
    except (psutil.Error, OSError) as e:
        warn_once("psutil", f"process RSS read failed: {e}")

    try:
        out["host_ram_available_mb"] = psutil.virtual_memory().available / (
            1024.0 * 1024.0
        )
    except (psutil.Error, OSError) as e:
        warn_once("psutil", f"virtual_memory read failed: {e}")

    return out
