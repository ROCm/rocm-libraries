# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Coverage counters and the per-process report.

Two counter sources are merged into one report:

* ``_C`` counts its own op calls (atomic increments in C++, no Python on the hot
  path), plus plan builds, build time and plan-cache size.
* Python routing layers that decide before reaching ``_C`` (the attention
  backend's pre-filter and fallback) count here through :func:`record`.

Counters are keyed by ``(op, route, reason, phase)``. ``phase`` is ``capture``
when the increment happened while the current stream was capturing a HIP graph,
and ``eager`` otherwise. Replays of a captured graph run no host code, so they
are not counted.

The report also lists ``attention_plans``: the distinct paged-attention plan
sizes ``(path, B, max_seq_len_q)`` built in this process, each with the reason
it was built (``bound`` for the per-process upper bound, ``capture`` for a
CUDA-graph capture size, ``eager`` for an eager power-of-two bucket). Their
number is bounded; see :mod:`hipdnn_vllm.attention.paged`.

The report is ``report-<pid>.json`` under ``HIPDNN_VLLM_REPORT_DIR``,
written at exit. A one-line summary is logged every
``HIPDNN_VLLM_REPORT_INTERVAL`` seconds while counters change, and the report
is refreshed with it. The file name holds only the pid: the rank (in the
JSON) can change once ``torch.distributed`` is torn down, and a second file
for the same process would double its counts for readers that sum
``report-*.json``. Writes are serialized, because the exit hook may run while
the summary thread is writing.

vLLM's EngineCore and workers are ``multiprocessing`` children. Those end in
``os._exit`` and never run ``atexit``, so the exit write is also registered as a
``multiprocessing`` finalizer, which their bootstrap runs before exiting.
"""

from __future__ import annotations

import atexit
import json
import logging
import multiprocessing.util
import os
import sys
import threading
import time
from collections import Counter
from typing import Any

from .config import Config

__all__ = ["record", "note_attention_plan", "snapshot", "write_report", "start", "reset"]

logger = logging.getLogger("hipdnn_vllm")

ROUTE_HIPDNN = "hipdnn"
ROUTE_FALLBACK = "fallback"

_lock = threading.Lock()
_write_lock = threading.Lock()
_counts: Counter = Counter()
# (path, B, max_seq_len_q) -> why it was first built ("bound", "capture", "eager").
_attention_plans: dict[tuple[str, int, int], str] = {}
_config: Config | None = None
# pid of the process whose exit hooks and summary thread are running. A forked
# child inherits the parent's module state but neither its thread nor its
# multiprocessing finalizers, so it must start its own.
_started_pid: int | None = None
_thread: threading.Thread | None = None
_stop = threading.Event()


def _reinit_locks_in_child() -> None:
    # A fork while the summary thread held a lock would leave it held forever
    # in the child, which has no such thread.
    global _lock, _write_lock
    _lock = threading.Lock()
    _write_lock = threading.Lock()


os.register_at_fork(after_in_child=_reinit_locks_in_child)


def _capturing() -> bool:
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - no device in this process
        return False


def record(op: str, route: str, reason: str = "", n: int = 1) -> None:
    """Count ``n`` calls of ``op`` that took ``route`` for ``reason``."""
    phase = "capture" if _capturing() else "eager"
    with _lock:
        _counts[(op, route, reason, phase)] += n
    if _config is not None and _started_pid != os.getpid():
        start(_config)


def note_attention_plan(path: str, batch: int, max_seq_len_q: int, origin: str) -> None:
    """Record a paged-attention plan size built in this process (first origin wins)."""
    with _lock:
        _attention_plans.setdefault((path, int(batch), int(max_seq_len_q)), origin)


def _attention_plans_snapshot() -> dict[str, Any]:
    sizes = [
        {"path": path, "batch": b, "max_seq_len_q": sq, "origin": origin}
        for (path, b, sq), origin in sorted(_attention_plans.items())
    ]
    by_origin = Counter(s["origin"] for s in sizes)
    return {"count": len(sizes), "by_origin": dict(sorted(by_origin.items())), "sizes": sizes}


def _c_snapshot() -> dict[str, Any] | None:
    mod = sys.modules.get("hipdnn_vllm._C")
    if mod is None:
        return None
    return mod.stats_snapshot()


def snapshot() -> dict[str, Any]:
    """The full report as a JSON-serialisable dict."""
    with _lock:
        py_rows = [
            {"op": op, "route": route, "reason": reason, "phase": phase, "count": cnt}
            for (op, route, reason, phase), cnt in sorted(_counts.items())
        ]
        plans = _attention_plans_snapshot()
    c = _c_snapshot()
    rows = py_rows + (c["counters"] if c else [])
    rows.sort(key=lambda r: (r["op"], r["route"], r["reason"], r["phase"]))
    fallback_reasons = Counter()
    for r in rows:
        if r["route"] == ROUTE_FALLBACK:
            fallback_reasons[r["reason"]] += r["count"]
    return {
        "pid": os.getpid(),
        "rank": _rank(),
        "argv0": os.path.basename(sys.argv[0]) if sys.argv else "",
        "time": time.time(),
        "counters": rows,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "runtime": c["runtime"] if c else None,
        "attention_plans": plans,
    }


def _rank() -> str:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            dist = torch.distributed
            if dist.is_available() and dist.is_initialized():
                return str(dist.get_rank())
        except Exception:  # noqa: BLE001
            pass
    for name in ("RANK", "LOCAL_RANK"):
        if os.environ.get(name):
            return os.environ[name]
    return "0"


def _has_data(snap: dict[str, Any]) -> bool:
    return bool(snap["counters"]) or snap["runtime"] is not None


def write_report(directory: str | None = None) -> str | None:
    """Write the report JSON and return its path. Returns ``None`` when no
    directory is configured or nothing was recorded in this process."""
    directory = directory or (_config.report_dir if _config else None)
    if not directory:
        return None
    with _write_lock:
        # Snapshot under the lock too, so a later write never loses to an older one.
        snap = snapshot()
        if not _has_data(snap):
            return None
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"report-{snap['pid']}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    return path


def _summary_line(snap: dict[str, Any]) -> str:
    by_route = Counter()
    for r in snap["counters"]:
        by_route[(r["op"], r["route"])] += r["count"]
    parts = [f"{op}:{route}={cnt}" for (op, route), cnt in sorted(by_route.items())]
    return "hipdnn_vllm coverage: " + (" ".join(parts) if parts else "no calls")


def _periodic(interval: float) -> None:
    last = None
    while not _stop.wait(interval):
        snap = snapshot()
        line = _summary_line(snap)
        if line != last:
            logger.info(line)
            last = line
            # Also refresh the report file: a process killed without running
            # atexit (SIGKILL at engine shutdown) still leaves its counts.
            try:
                write_report()
            except OSError as ex:
                logger.warning("hipdnn_vllm: could not write report: %s", ex)


def _at_exit() -> None:
    # Registered both with atexit and as a multiprocessing finalizer; a main
    # process runs both, so only the first call writes.
    if _stop.is_set():
        return
    _stop.set()
    try:
        path = write_report()
        if path:
            logger.info("hipdnn_vllm report written to %s", path)
    except Exception as ex:  # noqa: BLE001 - never fail interpreter shutdown
        logger.warning("hipdnn_vllm: could not write report: %s", ex)


def configure(config: Config) -> None:
    """Remember the config. Threads and exit hooks start on first use, so
    processes that never route an op (the API server) do nothing."""
    global _config
    _config = config


def start(config: Config) -> None:
    global _started_pid, _thread, _config
    with _lock:
        if _started_pid == os.getpid():
            return
        _started_pid = os.getpid()
        _config = config
    atexit.register(_at_exit)
    multiprocessing.util.Finalize(None, _at_exit, exitpriority=0)
    if config.report_interval > 0:
        _thread = threading.Thread(
            target=_periodic, args=(config.report_interval,), name="hipdnn-vllm-stats", daemon=True
        )
        _thread.start()


def reset() -> None:
    """Clear Python and C++ counters (tests and benchmarks)."""
    with _lock:
        _counts.clear()
        _attention_plans.clear()
    mod = sys.modules.get("hipdnn_vllm._C")
    if mod is not None:
        mod.stats_reset()
