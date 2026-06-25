################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

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
    return joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(fn)(el) for el in seq)


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
    def _terminate_process_tree() -> None:
        # On Windows, taskkill /T reliably tears down the full child tree.
        if os.name == "nt":
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
            return

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

    while proc.poll() is None:
        if not stop_event.wait(timeout=poll_interval):
            continue

        logger.warning(
            f"Stop requested while running config={proc_name}; terminating subprocess"
        )
        _terminate_process_tree()
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
