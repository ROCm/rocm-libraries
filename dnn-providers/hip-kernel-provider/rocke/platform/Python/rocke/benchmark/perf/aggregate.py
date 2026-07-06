# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sampling primitive - reduce K single-run records to one aggregate (pure).

A single profiled run is noisy even with the clock-invariant cycle metric (cache
state, scheduling, contention still vary - see the design doc's noise appendix). So
the honest unit of comparison is *several* runs of the same kernel reduced to a
median plus a spread, not one shot.

`aggregate(records)` takes K records that share an identity
(`schema.identity` = arch + kernel_name + shape) and returns ONE record where:
  - every counter is the **median** across the K runs;
  - `wall.ms_median` / `wall.ms_spread_pct` + `wall.samples` summarize wall time;
  - `spread` carries the relative spread of the primary metric (`busy_cycles`) and
    of wall ms, so a consumer can gate a regression on `max(threshold, k*spread)`
    instead of a bare delta;
  - `derived` is recomputed from the median counters;
  - `n_samples` records K.

Pure and stdlib-only: writes nothing, runs no profiler. Mixing identities is a
caller bug, so it raises rather than silently averaging unrelated kernels.
"""
from __future__ import annotations

import copy
import statistics
from typing import Any, Mapping, Optional, Sequence

from . import schema as _schema


def _median(vals: Sequence[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _as_int_if_whole(x: float) -> float:
    return int(x) if x == int(x) else x


def _spread_pct(vals: Sequence[float]) -> Optional[float]:
    """Relative spread (max-min)/|median| * 100, or None if undefined.

    Peak-to-peak (not stdev) because K is small and we want the worst observed
    wobble, which is what a noise-aware regression gate should tolerate.
    """
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    med = statistics.median(vals)
    if med == 0:
        return None
    return (max(vals) - min(vals)) / abs(med) * 100.0


def _median_counters(records: Sequence[Mapping[str, Any]]) -> dict:
    """Per-counter median across records (union of keys; missing values skipped)."""
    keys: set[str] = set()
    for r in records:
        keys.update((r.get("counters") or {}).keys())
    out: dict = {}
    for k in keys:
        vals = [(r.get("counters") or {}).get(k) for r in records]
        m = _median([v for v in vals if v is not None])
        if m is not None:
            out[k] = _as_int_if_whole(m)
    return out


def _median_wall(records: Sequence[Mapping[str, Any]]) -> dict:
    """Aggregate the wall section: median ms + spread + per-run samples."""
    ms_vals = [(r.get("wall") or {}).get("ms_median") for r in records]
    ms_vals = [v for v in ms_vals if v is not None]
    wall: dict = {}
    if ms_vals:
        wall["ms_median"] = statistics.median(ms_vals)
        wall["ms_spread_pct"] = _spread_pct(ms_vals)
        wall["samples"] = [{"ms": v} for v in ms_vals]
    # Throughput fields (medianed when present in the inputs).
    for k in ("tflops", "gbs", "pct_peak"):
        vals = [(r.get("wall") or {}).get(k) for r in records]
        m = _median([v for v in vals if v is not None])
        if m is not None:
            wall[k] = m
    return wall


def _derived(counters: Mapping[str, Any]) -> dict:
    """Recompute derived ratios from median counters (same defs as the harness)."""
    d: dict = {}
    if counters.get("total_clocks"):
        d["busy_fraction"] = counters.get("busy_cycles", 0) / counters["total_clocks"]
    hits = counters.get("l2_hit", 0)
    misses = counters.get("l2_miss", 0)
    if (hits + misses) > 0:
        d["l2_hit_rate"] = hits / (hits + misses)
    return d


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict:
    """Reduce K same-identity records to one median+spread record.

    Raises ValueError on an empty input or on mixed identities (aggregating across
    kernels/shapes/arches is always a caller bug). The run/kernel metadata is taken
    from the first record (identical by construction).
    """
    if not records:
        raise ValueError("aggregate() needs at least one record")
    ids = {_schema.identity(r) for r in records}
    if len(ids) != 1:
        raise ValueError(f"records span multiple identities: {sorted(ids)}")

    base = records[0]
    counters = _median_counters(records)
    wall = _median_wall(records)

    busy = [(r.get("counters") or {}).get(_schema.PRIMARY_METRIC) for r in records]
    spread = {"ms_pct": wall.get("ms_spread_pct"),
              "busy_cycles_pct": _spread_pct([v for v in busy if v is not None])}

    out = copy.deepcopy(dict(base))
    out["wall"] = wall
    out["counters"] = counters
    out["derived"] = _derived(counters)
    out["captured_counters"] = sorted(counters)
    out["spread"] = spread
    out["n_samples"] = len(records)
    _schema.validate(out)
    return out
