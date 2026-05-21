# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repeated-run benchmark summaries.

This is the Python equivalent of the runbook's benchmark hygiene section:
run enough attempts, optionally discard the first cold run, report median and
spread, and keep correctness separate from timing unless requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class BenchmarkSummary:
    """Summary statistics over repeated `run_manifest` measurements."""

    ms: Sequence[float]
    tflops: Sequence[float]
    gbps: Sequence[float]
    max_abs_diff: float = 0.0
    bad_count: int = 0
    total: int = 0
    discarded_first: bool = False

    @property
    def attempts(self) -> int:
        return len(self.tflops)

    @property
    def median_tflops(self) -> float:
        return float(statistics.median(self.tflops)) if self.tflops else 0.0

    @property
    def min_tflops(self) -> float:
        return float(min(self.tflops)) if self.tflops else 0.0

    @property
    def max_tflops(self) -> float:
        return float(max(self.tflops)) if self.tflops else 0.0

    @property
    def mean_tflops(self) -> float:
        return float(statistics.mean(self.tflops)) if self.tflops else 0.0

    @property
    def stdev_tflops(self) -> float:
        return float(statistics.stdev(self.tflops)) if len(self.tflops) > 1 else 0.0

    @property
    def spread_pct(self) -> float:
        median = self.median_tflops
        if median == 0.0:
            return 0.0
        return (self.max_tflops - self.min_tflops) / median * 100.0

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "ms": list(self.ms),
            "tflops": list(self.tflops),
            "gbps": list(self.gbps),
            "median_tflops": self.median_tflops,
            "min_tflops": self.min_tflops,
            "max_tflops": self.max_tflops,
            "mean_tflops": self.mean_tflops,
            "stdev_tflops": self.stdev_tflops,
            "spread_pct": self.spread_pct,
            "max_abs_diff": self.max_abs_diff,
            "bad_count": self.bad_count,
            "total": self.total,
            "discarded_first": self.discarded_first,
        }


def summarize_runs(
    *,
    ms: Iterable[float],
    tflops: Iterable[float],
    gbps: Iterable[float],
    max_abs_diff: float = 0.0,
    bad_count: int = 0,
    total: int = 0,
    discard_first: bool = False,
) -> BenchmarkSummary:
    """Build a `BenchmarkSummary` from already-collected runs."""

    ms_l = list(ms)
    tf_l = list(tflops)
    gb_l = list(gbps)
    if not (len(ms_l) == len(tf_l) == len(gb_l)):
        raise ValueError("ms, tflops, and gbps must have the same length")
    if discard_first and len(tf_l) > 1:
        ms_l = ms_l[1:]
        tf_l = tf_l[1:]
        gb_l = gb_l[1:]
    return BenchmarkSummary(
        ms=ms_l,
        tflops=tf_l,
        gbps=gb_l,
        max_abs_diff=float(max_abs_diff),
        bad_count=int(bad_count),
        total=int(total),
        discarded_first=bool(discard_first),
    )


def benchmark_manifest(
    manifest_path: Path,
    hsaco_path: Optional[Path] = None,
    *,
    attempts: int = 5,
    verify_first: bool = True,
    discard_first: bool = False,
) -> BenchmarkSummary:
    """Run one manifest repeatedly and summarize the measurements.

    `verify_first=True` verifies only the first run. The timing for that run
    is still valid because `run_manifest` times before copying D2H and running
    the CPU reference, but callers can set `discard_first=True` if they want to
    throw away the first run for cache/clock warm-up reasons.
    """

    if attempts <= 0:
        raise ValueError("attempts must be > 0")

    # Import lazily so `python -m ck_dsl.run_manifest` does not see
    # ck_dsl.run_manifest pre-imported by ck_dsl.__init__ via this module.
    from ..run_manifest import run_manifest

    ms: List[float] = []
    tflops: List[float] = []
    gbps: List[float] = []
    max_abs = 0.0
    bad = 0
    total = 0

    for i in range(attempts):
        summary = run_manifest(
            Path(manifest_path),
            Path(hsaco_path) if hsaco_path is not None else None,
            verify=(verify_first and i == 0),
        )
        ms.append(summary.ms)
        tflops.append(summary.tflops)
        gbps.append(summary.gbps)
        if i == 0:
            max_abs = summary.max_abs_diff
            bad = summary.bad_count
            total = summary.total

    return summarize_runs(
        ms=ms,
        tflops=tflops,
        gbps=gbps,
        max_abs_diff=max_abs,
        bad_count=bad,
        total=total,
        discard_first=discard_first,
    )
