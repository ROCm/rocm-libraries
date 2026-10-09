# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Early stopping for the conv benchmark sweeps.

A sweep times thousands of kernels for one problem, and most of the clock goes
to the timed loops of kernels that were never going to rank. The warmup
launches run anyway, so they are timed too: a kernel whose warmup is already
``factor`` times slower than the best kernel measured so far is reported and
skipped instead of being timed. A kernel that failed verification is still
timed but never becomes the best: a wrong-but-fast kernel would otherwise prune
every correct one, here and, through the record file, in a seeded sweep.

The best time can also come from another run: ``--early-stop-record FILE``
keeps each case's best time in FILE (rewritten whenever one improves, so a run
that is killed still leaves what it measured), and
``--early-stop-seed FILE`` starts each case's sweep from the times in such a
file. ``benchmark_conv_compare.py`` uses that to seed the implicit-GEMM sweep
with the direct-conv results. Cases are matched by their normalized geometry,
dtype and direction, so the two scripts' problem types meet on one key.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Optional

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
        f"starts skipping (default: {DEFAULT_AFTER}); a seeded case skips from "
        "the first kernel",
    )
    parser.add_argument(
        "--early-stop-seed",
        default=None,
        dest="early_stop_seed",
        metavar="FILE",
        help="JSON of per-case best times (as written by --early-stop-record) "
        "to start each case's early stopping from",
    )
    parser.add_argument(
        "--early-stop-record",
        default=None,
        dest="early_stop_record",
        metavar="FILE",
        help="keep each case's best measured time in FILE (JSON)",
    )


def case_key(problem: object, dtype: str, direction: str) -> str:
    """One key for a conv case, whichever script's problem type describes it."""
    from kernels.common.conv_args import ConvGeometry

    if hasattr(problem, "KH"):  # DirectConvProblem
        geom = ConvGeometry.from_direct_problem(problem)
    else:
        geom = ConvGeometry.from_problem(problem)
    fields = ",".join(f"{k}={v}" for k, v in sorted(asdict(geom).items()))
    return f"{direction}/{dtype}/{fields}"


# Per-process state behind --early-stop-seed / --early-stop-record, set up on
# the first EarlyStop.for_case call.
_seeds: Optional[Dict[str, float]] = None
_records: Dict[str, float] = {}
_record_path: Optional[Path] = None


def _load(args) -> Dict[str, float]:
    global _seeds, _record_path
    if _seeds is None:
        seed_path = getattr(args, "early_stop_seed", None)
        _seeds = {}
        if seed_path and Path(seed_path).is_file():
            text = Path(seed_path).read_text()
            _seeds = {k: float(v) for k, v in json.loads(text).items()}
        record_path = getattr(args, "early_stop_record", None)
        _record_path = Path(record_path) if record_path else None
    return _seeds


def _record(key: str, ms: float) -> None:
    prev = _records.get(key)
    if prev is not None and ms >= prev:
        return
    _records[key] = ms
    if _record_path is not None:
        tmp = _record_path.with_name(_record_path.name + ".tmp")
        tmp.write_text(json.dumps(_records, indent=1, sort_keys=True) + "\n")
        tmp.replace(_record_path)


class EarlyStop:
    """Times kernels for one sweep, skipping the hopeless ones.

    :meth:`measure` replaces ``time_launches``: it times the ``warmup``
    launches first (they also serve as the warmup) and, when their average is
    more than ``factor`` times the best result so far, returns ``None`` without
    running the timed loop. The first ``after`` kernels of a sweep are always
    measured in full, so the best result is established before it is used.
    A kernel measured with ``passed=False`` (it failed verification) neither
    counts towards ``after`` nor updates the best result or the record file.
    """

    def __init__(
        self,
        factor: float = DEFAULT_FACTOR,
        after: int = DEFAULT_AFTER,
        *,
        seed_ms: Optional[float] = None,
        record_key: Optional[str] = None,
    ) -> None:
        self.factor = factor
        self.after = after
        self.n_measured = 0
        # A seed is a time another run measured on this case: it is trusted
        # from the first kernel, so ``after`` does not apply to it.
        self.seed_ms = seed_ms
        self.best_ms: Optional[float] = seed_ms
        self.n_stopped = 0
        self.last_warmup_ms: Optional[float] = None
        self.record_key = record_key

    @classmethod
    def for_case(cls, args, problem: object, dtype: str, direction: str) -> "EarlyStop":
        """The early stopping for one case: seeded and recorded per the flags."""
        seeds = _load(args)
        key = case_key(problem, dtype, direction)
        return cls(
            args.early_stop,
            args.early_stop_after,
            seed_ms=seeds.get(key),
            record_key=key if getattr(args, "early_stop_record", None) else None,
        )

    def measure(
        self,
        fn: Callable[[], None],
        *,
        warmup: int,
        iters: int,
        stream: int = 0,
        passed: Optional[bool] = None,
    ) -> Optional[float]:
        from rocke.runtime import time_launches

        engaged = (
            self.factor > 0
            and self.best_ms is not None
            and (self.seed_ms is not None or self.n_measured >= self.after)
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
        if passed is False:
            return ms
        self.n_measured += 1
        if self.best_ms is None or ms < self.best_ms:
            self.best_ms = ms
        if self.record_key is not None:
            _record(self.record_key, ms)
        return ms

    def report(self, label: str) -> None:
        """Print the line for a kernel :meth:`measure` just skipped."""
        seeded = " (seed)" if self.best_ms == self.seed_ms else ""
        print(
            f"  [early-stop] {label}: warmup {self.last_warmup_ms:.3f} ms > "
            f"{self.factor:g} x best {self.best_ms:.3f} ms{seeded}",
            flush=True,
        )

    def summary(self) -> str:
        if not self.n_stopped:
            return ""
        if not self.n_measured and self.seed_ms is not None:
            return (
                f"all {self.n_stopped} early-stopped: none came within "
                f"{self.factor:g} x the seeded best {self.seed_ms:.3f} ms"
            )
        return f"{self.n_stopped} early-stopped"
