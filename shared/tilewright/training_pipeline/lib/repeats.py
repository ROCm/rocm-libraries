# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One label per (GEMM, kernel) from its repeated measurements.

A kernel can have several rows for one GEMM: its shape benched again in
another round or by another bench of the run, a retried problem, or the
runs of the sustained finalist re-measurement (stage03d, rows with
`sustained`). Its label comes from one class of its rows, the first it has
of: sustained rows, tested rows, skip rows (a single cold warm-up call, an
upper bound on its latency). Within that class the readings merge by the
aggregation (`AGGREGATIONS`, `train.repeat_aggregation`):

  median      the median reading; the mean of the two middle readings of an
              even count
  median_low  the lower of the two middle readings of an even count, so
              always one of the readings
  min         the fastest reading

Every reading carries noise, so the fastest of several is biased low, and
more so the more readings a kernel has: the minimum favours the kernels
measured most often (a winner's curse), the median does not. The candidate
of a kernel with several readings is a copy of its median reading's row (the
lower one of an even count, the fastest under `min`), so it keeps that
reading's rank and noise statistics, with the label as its latency and the
number of readings as `n_repeats`; a kernel with one row keeps that row.

Only a GEMM's finalists have sustained rows, and the long windows read them
slower than the short windows read the rest. So that its kernels compare
under one protocol, a GEMM with sustained rows labels every other kernel
with its short-window latency times the median ratio of sustained to
short-window latency over the finalists that have a timed short-window row
(`sustained_ratio`); the candidate keeps the short-window latency as
`us_short`, and a finalist its own as well.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

AGGREGATIONS = ("median", "median_low", "min")
DEFAULT_AGGREGATION = "median"
# The rule stage05 merged repeats by before it was an option: model options
# recorded without `repeat_aggregation` were trained with it.
PREDATING_AGGREGATION = "min"


def config_aggregation(cfg: Optional[Mapping[str, Any]]) -> str:
    """The `train.repeat_aggregation` of a run config, else the default."""
    train = (cfg or {}).get("train") or {}
    return check(str(train.get("repeat_aggregation") or DEFAULT_AGGREGATION))


def check(aggregation: str) -> str:
    """`aggregation`, or ValueError when it is not one of AGGREGATIONS."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"repeat aggregation {aggregation!r} is not one of {list(AGGREGATIONS)}"
        )
    return aggregation


def aggregate(values: Sequence[float], aggregation: str) -> float:
    """The label of one kernel's readings of one GEMM."""
    if not values:
        raise ValueError("no readings to aggregate")
    if check(aggregation) == "min":
        return min(values)
    if aggregation == "median_low":
        return statistics.median_low(values)
    return statistics.median(values)


def _merge_class(rows: Sequence[Mapping[str, Any]], aggregation: str) -> Dict[str, Any]:
    """A copy of the representative row of `rows` (one class of one
    kernel's rows) with their label as `us` and their count as
    `n_repeats`."""
    group = sorted(rows, key=lambda r: float(r["us"]))
    rep = group[0] if check(aggregation) == "min" else group[(len(group) - 1) // 2]
    out = dict(rep)
    out["us"] = aggregate([float(r["us"]) for r in group], aggregation)
    out["n_repeats"] = len(group)
    return out


def merge_rows(rows: Sequence[Mapping[str, Any]], aggregation: str) -> Dict[str, Any]:
    """The candidate of the rows (`us`, `is_skip`, `sustained`,
    `is_origami_pick`) of one kernel of one GEMM; see the module docstring.
    A finalist also gets the label of its other rows as `us_short`, with
    `short_timed` saying whether that comes from tested rows. A kernel any
    of whose rows was the Origami pick stays the Origami pick."""
    sustained = [r for r in rows if r.get("sustained")]
    short = [r for r in rows if not r.get("sustained")]
    tested = [r for r in short if not r.get("is_skip")]
    if sustained:
        out = _merge_class(sustained, aggregation)
        out["is_skip"] = False
        if short:
            out["us_short"] = _merge_class(tested or short, aggregation)["us"]
            out["short_timed"] = bool(tested)
    else:
        out = _merge_class(tested or short, aggregation)
    if any(r.get("is_origami_pick") for r in rows):
        out["is_origami_pick"] = True
    return out


def sustained_ratio(candidates: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """The median sustained over short-window latency of a GEMM's merged
    finalists that have a timed short-window row; None without one."""
    ratios = [
        float(c["us"]) / float(c["us_short"])
        for c in candidates
        if c.get("sustained") and c.get("short_timed") and float(c["us_short"]) > 0
    ]
    return statistics.median(ratios) if ratios else None


def _scaled(
    candidates: List[Mapping[str, Any]], ratio: float
) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for c in candidates:
        if c.get("sustained"):
            out.append(c)
            continue
        base = float(c.get("us_short", c["us"]))
        out.append(dict(c, us=base * ratio, us_short=base))
    return out


def merge_candidates(
    candidates: Iterable[Mapping[str, Any]], aggregation: str
) -> List[Mapping[str, Any]]:
    """The candidates of one GEMM, one per solution index in the order the
    indices first appear: a kernel's only row as it is, else its rows
    merged (`merge_rows`); with sustained rows, every other kernel scaled by
    `sustained_ratio`. Merging merged candidates again changes nothing."""
    check(aggregation)
    by_sid: Dict[int, List[Mapping[str, Any]]] = {}
    for c in candidates:
        by_sid.setdefault(int(c["sol_idx_global"]), []).append(c)
    out = [
        rows[0] if len(rows) == 1 else merge_rows(rows, aggregation)
        for rows in by_sid.values()
    ]
    ratio = sustained_ratio(out)
    return _scaled(out, ratio) if ratio is not None else out
