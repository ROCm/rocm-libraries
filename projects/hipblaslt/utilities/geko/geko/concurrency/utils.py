# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Concurrency utilities for the GEKO framework."""

from typing import List, Sequence, TypeVar, Callable
from threading import current_thread, main_thread

import joblib
import signal
import os
import time
import subprocess
import logging

logger = logging.getLogger("GEKO")

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["parallel_for", "wait_process_or_stop", "install_stop_handlers", "restore_stop_handlers"]


def parallel_for(fn: Callable[[T], R], seq: Sequence[T], n_jobs: int = 64) -> List[R]:
    """Execute a function in parallel over a sequence.

    Args:
        fn: Function to apply to each element.
        seq: Sequence of elements to process.
        n_jobs: Number of parallel jobs.

    Returns:
        List of results from applying fn to each element in seq
    """
    if not seq:
        return []

    # Avoid oversubscription and Windows spawn overhead for tiny batches.
    max_workers = max(1, min(len(seq), n_jobs, os.cpu_count() or 1))
    return joblib.Parallel(n_jobs=max_workers)(joblib.delayed(fn)(el) for el in seq)


def _terminate_process_tree_windows(proc: subprocess.Popen, terminate_timeout: float) -> None:
    # On Windows, taskkill /T reliably tears down the full child tree.
    subprocess.run(
        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        proc.wait(timeout=terminate_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _terminate_process_tree_posix(proc: subprocess.Popen, proc_name: str, terminate_timeout: float) -> None:
    # On POSIX, kill the process group if the child is its group leader.
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    if pgid == proc.pid:
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=terminate_timeout)
            return
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Config={proc_name} did not exit after SIGTERM; sending SIGKILL to process group"
            )
            os.killpg(pgid, signal.SIGKILL)
            proc.wait()
            return

    # Fallback when child was not started in a dedicated process group.
    proc.terminate()
    try:
        proc.wait(timeout=terminate_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def wait_process_or_stop(
    proc: subprocess.Popen,
    stop_event,
    proc_name: str,
    poll_interval: float = 1.0,
    terminate_timeout: float = 30.0,
    progress_path=None,
    stall_timeout: float = 0.0,
) -> bool:
    """Wait for process completion, a stop request, or a progress stall.

    Args:
        proc: Child process to monitor.
        stop_event: Event-like object with wait(timeout) and is_set() methods.
        proc_name: Process name to wait for or stop.
        poll_interval: Seconds between stop checks while process is running.
        terminate_timeout: Seconds to wait after terminate() before kill().
        progress_path: File whose mtime indicates forward progress (the worker's
            tensilelite log). Required for stall detection; ignored when None.
        stall_timeout: Seconds without progress before the worker is considered
            stalled and torn down. 0 disables the check.

    Returns:
        True if the worker was killed for stalling, False otherwise.

    Detection is mtime-based rather than a wall-clock budget per shape: shapes
    legitimately differ by an order of magnitude in runtime, but a healthy worker
    writes to its log once per solution, so "no write for N seconds" separates a
    dead worker from a slow one far better than any total-time cap.

    Caveat: with Tensile's "print only winners" flag, the log is only written on
    a new best, not per solution, so a long stretch without a better solution can
    look identical to a stall. If stalls get reported on otherwise-healthy runs,
    check whether that flag is enabled before assuming the detector is broken.
    """
    def _terminate_process_tree() -> None:
        if os.name == "nt":
            _terminate_process_tree_windows(proc, terminate_timeout)
        else:
            _terminate_process_tree_posix(proc, proc_name, terminate_timeout)

    last_progress = time.monotonic()
    last_mtime = None

    while proc.poll() is None:
        if stop_event.wait(timeout=poll_interval):
            logger.warning(
                f"Stop requested while running config={proc_name}; terminating subprocess"
            )
            _terminate_process_tree()
            return False

        if not stall_timeout or progress_path is None:
            continue

        try:
            mtime = os.path.getmtime(progress_path)
        except OSError:
            # Progress file missing or unreadable. Keep waiting: the stall
            # clock starts from when this call began, so a worker that never
            # produces the file is still caught.
            mtime = last_mtime

        if mtime != last_mtime:
            last_mtime, last_progress = mtime, time.monotonic()
            continue

        stalled_for = time.monotonic() - last_progress
        if stalled_for >= stall_timeout:
            logger.warning(
                f"Config={proc_name} made no progress for {stalled_for:.0f}s "
                f"(stall_timeout={stall_timeout:.0f}s); terminating to free the GPU slot"
            )
            _terminate_process_tree()
            return True

    return False


def install_stop_handlers(stop_event) -> tuple[object | None, object | None]:
    """Install SIGINT/SIGTERM handlers that set stop_event and log the stop request.

    Signal handlers can only be installed from the main thread. When a Runner
    is nested inside worker threads, this function becomes a no-op and returns
    ``(None, None)`` so callers can safely restore conditionally.

    Returns:
        Tuple containing previous SIGINT and SIGTERM handlers for restoration.
    """
    if current_thread() is not main_thread():
        logger.debug("Skipping stop handler installation outside the main thread")
        return None, None

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None

    def _request_stop(signum, _frame) -> None:
        stop_event.set()
        logger.warning(f"Received signal {signum}, stopping new work and finalizing active workers")

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    return prev_sigint, prev_sigterm


def restore_stop_handlers(prev_handlers: tuple[object | None, object | None]) -> None:
    """Restore SIGINT/SIGTERM handlers from install_stop_handlers return value."""
    prev_sigint, prev_sigterm = prev_handlers
    if prev_sigint is None and prev_sigterm is None:
        return

    signal.signal(signal.SIGINT, prev_sigint)
    if hasattr(signal, "SIGTERM") and prev_sigterm is not None:
        signal.signal(signal.SIGTERM, prev_sigterm)
