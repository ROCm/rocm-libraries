# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sustained re-measurement of the finalists of power-capped GEMMs.

A long or compute-bound GEMM runs at the device's power limit in sustained
use, and the limit slows some kernels more than others. Adaptive timing's
windows of a few milliseconds per kernel end before the clock settles, so
they rank such a GEMM's kernels at a clock it never runs at for long.
stage03d times the finalists of these GEMMs again in long self-soaked
windows: each kernel warms up for `warmup_time` ms, long enough to reach
the power limit, and is then timed for at least `measure_time` ms (at most
`max_measure_time`), in `runs` runs, its rows a label source of their own
(stage04 `is_sustained`, preferred by lib/repeats.py).

A GEMM is power-capped (`power_capped`) when its winner (its fastest tested
row) takes at least `min_us`, or reaches `compute_bound_tflops` while taking
at least `compute_bound_min_us`. Its finalists (`finalists`) are its
`top_k` fastest candidates and every candidate within `tolerance` of the
winner (tested rows by their latency, skip rows by their warm-up), at most
`max_kernels`, the fastest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import AbstractSet, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import rebench as rb
from .bench_log import Problem
from .bench_yaml import set_line_knob

Shape = Tuple[int, int, int, int]

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "min_us": 100.0,
    "compute_bound_tflops": 700.0,
    "compute_bound_min_us": 100.0,
    "top_k": 4,
    "tolerance": 0.08,
    "max_kernels": 12,
    "runs": 1,
    "warmup_time": 250.0,
    "measure_time": 300.0,
    "max_measure_time": 400.0,
    "gemms_per_block": 24,
}
# The adaptive timing fields a sustained line sets; the line keeps the rest.
WINDOW_KEYS = ("warmup_time", "measure_time", "max_measure_time")


def settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """`bench.sustained` of a run config over DEFAULTS."""
    block = (cfg.get("bench") or {}).get("sustained") or {}
    return {k: block.get(k, d) for k, d in DEFAULTS.items()}


def tflops(shape: Shape, us: float) -> float:
    """The rate of a GEMM of `shape` that takes `us`."""
    m, n, k, b = shape
    return 2.0 * m * n * k * b / us / 1e6


def power_capped(shape: Shape, winner_us: float, st: Mapping[str, Any]) -> str:
    """Why a GEMM whose winner takes `winner_us` runs at the power limit:
    "compute_bound", "long", or "" when it does not."""
    if winner_us >= float(st["compute_bound_min_us"]) and tflops(
        shape, winner_us
    ) >= float(st["compute_bound_tflops"]):
        return "compute_bound"
    if winner_us >= float(st["min_us"]):
        return "long"
    return ""


@dataclass
class Finalists:
    """The finalists of one power-capped GEMM, fastest first."""

    shape: Shape
    winner_us: float
    reason: str
    candidates: List[rb.Candidate]

    @property
    def sol_indices(self) -> List[int]:
        return [c.sol_index for c in self.candidates]


def finalists(
    prob: Problem, st: Mapping[str, Any], blocked: AbstractSet[int] = frozenset()
) -> Optional[Finalists]:
    """The finalists of a complete problem whose GEMM is power-capped (see
    the module docstring), rows of the `blocked` solutions left out; None
    for any other problem or fewer than two candidates."""
    if not prob.complete:
        return None
    shape = prob.shape()
    tested = rb.tested_latencies(prob, blocked)
    if shape is None or not tested:
        return None
    winner_us = min(tested.values())
    reason = power_capped(shape, winner_us, st)
    if not reason:
        return None
    best: Dict[int, rb.Candidate] = {}
    for t in prob.tested:
        if t.sol_index is not None and t.sol_index in tested:
            us = tested[t.sol_index]
            if t.sol_index not in best:
                best[t.sol_index] = rb.Candidate(t.rank, t.sol_index, us, False)
    for s in prob.skipped:
        us = s.warm_up_us
        if s.sol_index is None or s.sol_index in best or us is None:
            continue
        if s.sol_index in blocked:
            continue
        if math.isfinite(us) and us > 0:
            best[s.sol_index] = rb.Candidate(s.rank, s.sol_index, us, True)
    order = sorted(best.values(), key=lambda c: (c.us, c.rank))
    limit = winner_us * (1.0 + float(st["tolerance"]))
    chosen = [c for i, c in enumerate(order) if i < int(st["top_k"]) or c.us <= limit]
    chosen = chosen[: int(st["max_kernels"])]
    if len(chosen) < 2:
        return None
    return Finalists(shape, winner_us, reason, chosen)


def sustained_line(
    line: str,
    kernels: Sequence[int],
    st: Mapping[str, Any],
    rotating_mib: Optional[int] = None,
) -> str:
    """`line` restamped to time exactly `kernels` (`rebench.rebench_line`: no
    skip screen) in the long windows of `st`."""
    out = rb.rebench_line(line, kernels, rotating_mib=rotating_mib)
    for key in WINDOW_KEYS:
        out = set_line_knob(out, key, float(st[key]))
    return out


def config_errors(cfg: Mapping[str, Any]) -> List[str]:
    """The rules of `bench.sustained` the schema cannot state."""
    st = settings(cfg)
    if float(st["measure_time"]) > float(st["max_measure_time"]):
        return ["bench.sustained.measure_time exceeds max_measure_time"]
    return []
