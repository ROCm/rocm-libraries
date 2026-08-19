# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sampling primitive - reduce K single-run records to one aggregate (pure).

A single profiled run is noisy even with the clock-invariant cycle metric (cache
state, scheduling, and contention still vary run to run). So the honest unit of
comparison is *several* runs of the same kernel reduced to a median plus a spread,
not one shot.

`aggregate(records)` takes K records that share an identity
(`schema.identity` = arch + kernel_name + shape) and returns ONE record where:
  - every counter is the **median** across the K runs;
  - `wall` and `profiled` timings are each medianed (ms + spread + per-run samples);
  - `spread` carries the relative spread of the primary metric (`busy_cycles`) and
    of wall ms, so a consumer can gate a regression on `max(threshold, k*spread)`
    instead of a bare delta;
  - `derived` is recomputed from the median counters (+ profiler_overhead_pct);
  - `n_samples` records K;
  - `counter_samples` (opt-in per-dispatch, if present) is collected from every
    input record and tagged with `sample_index`, preserving the raw evidence
    behind an aggregate without conflating dispatch ids across repeats.

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
    """Per-counter median across records, requiring every run to capture the key."""
    keys = set((records[0].get("counters") or {}).keys())
    for r in records[1:]:
        keys.intersection_update((r.get("counters") or {}).keys())
    out: dict = {}
    for k in keys:
        vals = [(r.get("counters") or {}).get(k) for r in records]
        m = _median(vals) if all(v is not None for v in vals) else None
        if m is not None:
            out[k] = _as_int_if_whole(m)
    return out


def _median_timing(records: Sequence[Mapping[str, Any]], section: str) -> dict:
    """Aggregate a timing section ('wall' or 'profiled'): median ms + spread +
    per-run samples + medianed throughput fields."""
    ms_vals = [(r.get(section) or {}).get("ms_median") for r in records]
    ms_vals = [v for v in ms_vals if v is not None]
    out: dict = {}
    if len(ms_vals) == len(records):
        out["ms_median"] = statistics.median(ms_vals)
        out["ms_spread_pct"] = _spread_pct(ms_vals)
        out["samples"] = [{"ms": v} for v in ms_vals]
    for k in ("tflops", "gbs", "pct_peak"):
        vals = [(r.get(section) or {}).get(k) for r in records]
        m = _median(vals) if all(v is not None for v in vals) else None
        if m is not None:
            out[k] = m
    return out


def _derived(counters: Mapping[str, Any]) -> dict:
    """Recompute derived ratios from median counters (same defs as the harness)."""
    d: dict = {}
    busy = counters.get("busy_cycles")
    total = counters.get("total_clocks")
    if busy is not None and total:
        d["busy_fraction"] = busy / total
    hits = counters.get("l2_hit")
    misses = counters.get("l2_miss")
    if hits is not None and misses is not None and (hits + misses) > 0:
        d["l2_hit_rate"] = hits / (hits + misses)
    return d


def _all_counter_samples(records: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Preserve raw dispatch samples from every run with their source run index."""
    out: list[dict] = []
    for sample_index, record in enumerate(records):
        for sample in record.get("counter_samples") or []:
            out.append({**sample, "sample_index": sample_index})
    return out


def _aggregate_verification(records: Sequence[Mapping[str, Any]]) -> dict:
    """Combine explicit verification results without inheriting only run zero."""
    results = [record.get("verify") or {} for record in records]
    statuses = [result.get("ok") for result in results]
    if any(status is False for status in statuses):
        ok = False
    elif statuses and all(status is True for status in statuses):
        ok = True
    else:
        return {}
    out = {"ok": ok}
    diffs = [result.get("max_abs_diff") for result in results]
    if all(value is not None for value in diffs):
        out["max_abs_diff"] = max(diffs)
    for key in ("bad_count", "total"):
        values = [result.get(key) for result in results]
        if all(value is not None for value in values):
            out[key] = sum(values)
    return out


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
    wall = _median_timing(records, "wall")
    profiled = _median_timing(records, "profiled")

    busy = [(r.get("counters") or {}).get(_schema.PRIMARY_METRIC) for r in records]
    spread = {
        "ms_pct": wall.get("ms_spread_pct"),
        "busy_cycles_pct": (
            _spread_pct(busy) if all(v is not None for v in busy) else None
        ),
    }

    derived = _derived(counters)
    if profiled.get("ms_median") and wall.get("ms_median"):
        derived["profiler_overhead_pct"] = (
            (profiled["ms_median"] - wall["ms_median"]) / wall["ms_median"] * 100.0
        )

    out = copy.deepcopy(dict(base))
    out["wall"] = wall
    out["profiled"] = profiled
    out["counters"] = counters
    out["derived"] = derived
    out["captured_counters"] = sorted(counters)
    out["verify"] = _aggregate_verification(records)
    out["spread"] = spread
    out["n_samples"] = len(records)
    if any("counter_samples" in r for r in records):
        out["counter_samples"] = _all_counter_samples(records)
    _schema.validate(out)
    return out
