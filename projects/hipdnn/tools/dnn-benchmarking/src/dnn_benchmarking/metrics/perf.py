# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Linux perf stat wrapper for CPU-side hardware counters.

Wraps the workload in ``perf stat -x, -e <events>``, parses the CSV
output, and folds CPU cycles/instructions/IPC into ``extra_metrics["perf"]``.

Two tiers of events:

* User-space (``cycles:u``, ``instructions:u``) — always available to
  the running user.
* Kernel-space (``cycles:k``, ``instructions:k``) — require
  ``/proc/sys/kernel/perf_event_paranoid <= 1`` (or ``CAP_PERFMON`` on
  the perf binary). Dropped silently when the kernel doesn't permit
  them; the recorded paranoid value tells the user why the kernel
  fields are None.

Missing perf binary is a single warn_once + skipped metrics dict;
nothing about ``--perf`` is fatal.
"""

import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._diagnostic import warn_once

PERF_EVENTS_USER = [
    "cycles:u",
    "instructions:u",
    "task-clock",
    "context-switches",
    "page-faults",
]

PERF_EVENTS_KERNEL = [
    "cycles:k",
    "instructions:k",
]


def _read_perf_paranoid() -> Optional[int]:
    """Return the kernel perf_event_paranoid setting, or None if unreadable."""
    try:
        with open("/proc/sys/kernel/perf_event_paranoid", "r") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _kernel_events_allowed(paranoid: Optional[int]) -> bool:
    # Documented kernel rule: cycles:k / instructions:k require
    # paranoid <= 1. paranoid 2 blocks kernel events; 3 blocks all
    # unprivileged tracing; 4 blocks even cycles:u on some kernels.
    return paranoid is not None and paranoid <= 1


def _build_argv(
    events: List[str],
    csv_path: Path,
    inner_argv: List[str],
) -> List[str]:
    return [
        "perf",
        "stat",
        "-x,",
        "-o",
        str(csv_path),
        "-e",
        ",".join(events),
        "--",
        *inner_argv,
    ]


def _parse_perf_csv(csv_path: Path) -> Dict[str, Any]:
    """Parse the seven-column ``perf stat -x,`` output.

    Format per row: ``<value>,<unit>,<event>,<run-time-ns>,<percent>,<metric>,<metric-unit>``
    Header lines (``# ...``) and blanks are skipped. Values that report
    ``<not counted>`` or ``<not supported>`` map to None for that event.
    """
    out: Dict[str, Any] = {}
    if not csv_path.exists():
        return out
    for raw in csv_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # perf inserts the message in the value column when it can't
        # measure (commas in messages would break naive split, so guard).
        cols = line.split(",")
        if len(cols) < 3:
            continue
        value_str, _unit, event = cols[0], cols[1], cols[2]
        if value_str.startswith("<"):
            out[event] = None
            continue
        try:
            value = float(value_str)
            out[event] = int(value) if value.is_integer() else value
        except ValueError:
            out[event] = None
    return out


def run(
    inner_argv: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run perf stat, parse CSV, return extra_metrics slice. Never raises."""
    binary = shutil.which("perf")
    if binary is None:
        warn_once("perf", "perf binary not found on PATH; skipping CPU counters")
        return {"perf": {"skipped": "perf binary not found on PATH"}}

    paranoid = _read_perf_paranoid()
    kernel_ok = _kernel_events_allowed(paranoid)
    events = list(PERF_EVENTS_USER)
    if kernel_ok:
        events.extend(PERF_EVENTS_KERNEL)

    host_dir = out_dir / socket.gethostname()
    host_dir.mkdir(parents=True, exist_ok=True)
    csv_path = host_dir / "perf.csv"
    argv = _build_argv(events, csv_path, inner_argv)

    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        warn_once("perf", f"perf invocation failed: {e}")
        return {"perf": {"skipped": f"perf invocation failed: {e}"}}

    parsed = _parse_perf_csv(csv_path)

    def _get(name: str) -> Optional[float]:
        v = parsed.get(name)
        return v if isinstance(v, (int, float)) else None

    cycles_user = _get("cycles:u")
    instr_user = _get("instructions:u")
    ipc_user: Optional[float] = None
    if cycles_user and instr_user is not None and cycles_user > 0:
        ipc_user = float(instr_user) / float(cycles_user)

    result: Dict[str, Any] = {
        "cycles_user": cycles_user,
        "instructions_user": instr_user,
        "ipc_user": ipc_user,
        "cycles_kernel": _get("cycles:k") if kernel_ok else None,
        "instructions_kernel": _get("instructions:k") if kernel_ok else None,
        "task_clock_ms": _get("task-clock"),
        "context_switches": _get("context-switches"),
        "page_faults": _get("page-faults"),
        "kernel_perf_paranoid": paranoid,
        "csv_path": str(csv_path),
    }
    if not kernel_ok:
        result["kernel_events_skipped_reason"] = (
            f"perf_event_paranoid={paranoid} (kernel events require <= 1)"
        )
    if proc.returncode != 0:
        result["returncode"] = proc.returncode
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        if tail:
            result["error_tail"] = tail
        warn_once(
            "perf",
            f"perf stat exited {proc.returncode}; partial counters may be present",
        )
    return {"perf": result}
