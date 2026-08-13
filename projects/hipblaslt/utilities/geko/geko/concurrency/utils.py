# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Concurrency utilities for the GEKO framework."""

from typing import List, Sequence, TypeVar, Callable
from threading import current_thread, main_thread

import joblib
import signal
import os
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
) -> None:
    """Wait for process completion or terminate it if stop_event is set.

    Args:
        proc: Child process to monitor.
        stop_event: Event-like object with wait(timeout) and is_set() methods.
        proc_name: Process name to wait for or stop.
        poll_interval: Seconds between stop checks while process is running.
        terminate_timeout: Seconds to wait after terminate() before kill().
    """
    while proc.poll() is None:
        if not stop_event.wait(timeout=poll_interval):
            continue

        logger.warning(
            f"Stop requested while running config={proc_name}; terminating subprocess"
        )
        if os.name == "nt":
            _terminate_process_tree_windows(proc, terminate_timeout)
        else:
            _terminate_process_tree_posix(proc, proc_name, terminate_timeout)
        break


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
