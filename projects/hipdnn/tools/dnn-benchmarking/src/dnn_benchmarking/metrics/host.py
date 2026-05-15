# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Host-side probes: CPU time delta and host memory snapshot.

``CpuTimeProbe`` is a context manager that diffs ``resource.getrusage``
across a block to produce user/kernel CPU time consumed by the calling
process. ``host_memory_snapshot`` returns process RSS and host RAM
availability via ``psutil`` if installed. Both degrade gracefully:
on any failure they yield ``None`` values rather than raising.
"""

import resource
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ._diagnostic import warn_once


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
        self._start: Optional[Any] = None
        self.delta: Optional[CpuTimeDelta] = None

    def __enter__(self) -> "CpuTimeProbe":
        try:
            self._start = resource.getrusage(resource.RUSAGE_SELF)
        except (OSError, ValueError) as e:
            warn_once("cpu_time", f"start sample failed: {e}")
            self._start = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._start is None:
            return
        try:
            end = resource.getrusage(resource.RUSAGE_SELF)
        except (OSError, ValueError) as e:
            warn_once("cpu_time", f"end sample failed: {e}")
            return
        self.delta = CpuTimeDelta(
            user_time_ms=(end.ru_utime - self._start.ru_utime) * 1000.0,
            kernel_time_ms=(end.ru_stime - self._start.ru_stime) * 1000.0,
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
