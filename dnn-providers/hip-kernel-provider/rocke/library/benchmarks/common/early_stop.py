# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Early stopping for the conv benchmark sweeps.

A sweep times thousands of kernels for one problem, and most of the clock goes
to the timed loops of kernels that were never going to rank. The warmup
launches run anyway, so they are timed too: a kernel whose warmup is already
``factor`` times slower than the best kernel measured so far is reported and
skipped instead of being timed.
"""

from __future__ import annotations

from typing import Callable, Optional

DEFAULT_FACTOR = 5.0
# Kernels timed in full before early stopping engages: until then the best
# result is too young to judge the rest against.
DEFAULT_AFTER = 100


def add_early_stop_arg(parser) -> None:
    """The ``--early-stop FACTOR`` flag, shared by both conv benchmarks."""
    parser.add_argument(
        "--early-stop",
        type=float,
        default=DEFAULT_FACTOR,
        dest="early_stop",
        metavar="FACTOR",
        help="skip timing a kernel whose warmup is more than FACTOR times slower "
        f"than the best kernel measured so far (default: {DEFAULT_FACTOR:g}; "
        "0 disables)",
    )
    parser.add_argument(
        "--early-stop-after",
        type=int,
        default=DEFAULT_AFTER,
        dest="early_stop_after",
        metavar="N",
        help="time the first N kernels of a sweep in full before --early-stop "
        f"starts skipping (default: {DEFAULT_AFTER})",
    )


class EarlyStop:
    """Times kernels for one sweep, skipping the hopeless ones.

    :meth:`measure` replaces ``time_launches``: it times the ``warmup``
    launches first (they also serve as the warmup) and, when their average is
    more than ``factor`` times the best result so far, returns ``None`` without
    running the timed loop. The first ``after`` kernels of a sweep are always
    measured in full, so the best result is established before it is used.
    """

    def __init__(
        self, factor: float = DEFAULT_FACTOR, after: int = DEFAULT_AFTER
    ) -> None:
        self.factor = factor
        self.after = after
        self.n_measured = 0
        self.best_ms: Optional[float] = None
        self.n_stopped = 0
        self.last_warmup_ms: Optional[float] = None

    def measure(
        self,
        fn: Callable[[], None],
        *,
        warmup: int,
        iters: int,
        stream: int = 0,
    ) -> Optional[float]:
        from rocke.runtime import time_launches

        engaged = (
            self.factor > 0
            and self.best_ms is not None
            and self.n_measured >= self.after
            and warmup > 0
        )
        if engaged:
            warm = time_launches(fn, warmup=0, iters=warmup, stream=stream)
            self.last_warmup_ms = warm
            if warm > self.factor * self.best_ms:
                self.n_stopped += 1
                return None
            # The probe was the warmup; only the timed loop is left.
            ms = time_launches(fn, warmup=0, iters=iters, stream=stream)
        else:
            ms = time_launches(fn, warmup=warmup, iters=iters, stream=stream)
        self.n_measured += 1
        if self.best_ms is None or ms < self.best_ms:
            self.best_ms = ms
        return ms

    def report(self, label: str) -> None:
        """Print the line for a kernel :meth:`measure` just skipped."""
        print(
            f"  [early-stop] {label}: warmup {self.last_warmup_ms:.3f} ms > "
            f"{self.factor:g} x best {self.best_ms:.3f} ms",
            flush=True,
        )

    def summary(self) -> str:
        return f"{self.n_stopped} early-stopped" if self.n_stopped else ""
