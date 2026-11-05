# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Geometry of routing cells: the box a cell label covers along M, N and K,
how far (in octaves) a shape lies from the box's inner sides, and the
neighbouring-leaf GEMMs within a margin outside it.

A side is inner when the routing sends shapes beyond it to another cell: the
tier thresholds of lib/grid.py and the split thresholds of a sub-cell label.
The bottom of the Tiny/TinyK tiers and the top of the Large/LargeK tiers are
the grid's outer edges. Integer bounds are inclusive, so a side sits half a
unit outside the last value the cell covers; a shape on either side of a
threshold is the same distance from it.
"""
from __future__ import annotations

import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from . import grid as _grid
from . import subcells as _sc

BOX_AXES: Tuple[str, ...] = ("M", "N", "K")

# (lo side, hi side) of a cell along one axis; None is an outer edge.
Sides = Tuple[Optional[float], Optional[float]]
_UNBOUNDED = 1 << 62


def _tier_bounds(axis: str, tier: str) -> Tuple[int, Optional[int]]:
    if axis == "K":
        lo, hi = _grid.TIER_RANGES_K[tier]
        top = _grid.K_TIERS_ALL[-1]
    else:
        lo, hi = _grid.TIER_RANGES_MN[tier]
        top = _grid.M_TIERS[-1]
    return int(lo), (None if tier == top else int(hi))


def cell_box(label: str) -> Dict[str, Sides]:
    """{axis: (lo side, hi side)} of `label` (a base cell or a sub-cell);
    a side is a coordinate between two integers, None an outer edge."""
    tiers = dict(zip(BOX_AXES, _sc.base_cell(label).split("|")[:3]))
    box: Dict[str, Sides] = {}
    for axis in BOX_AXES:
        lo, hi = _tier_bounds(axis, tiers[axis])
        top = hi if hi is not None else _UNBOUNDED
        lo, top = _sc.restrict_axis_bounds(label, axis, lo, top)
        box[axis] = (
            None if lo <= 1 else lo - 0.5,
            None if top == _UNBOUNDED else top + 0.5,
        )
    return box


def _coords(m: int, n: int, k: int) -> Dict[str, float]:
    return {"M": float(m), "N": float(n), "K": float(k)}


def boundary_distance(box: Mapping[str, Sides], m: int, n: int, k: int) -> float:
    """Octaves from a shape inside `box` to its nearest inner side; inf when
    the box has none."""
    best = math.inf
    for axis, v in _coords(m, n, k).items():
        lo, hi = box[axis]
        if lo is not None:
            best = min(best, abs(math.log2(v / lo)))
        if hi is not None:
            best = min(best, abs(math.log2(hi / v)))
    return best


def octaves_outside(box: Mapping[str, Sides], m: int, n: int, k: int) -> float:
    """How far a shape lies outside `box`: the largest per-axis excess in
    octaves, 0 inside. Only inner sides can be crossed: the outer edges
    bound every shape."""
    worst = 0.0
    for axis, v in _coords(m, n, k).items():
        lo, hi = box[axis]
        if lo is not None and v < lo:
            worst = max(worst, math.log2(lo / v))
        elif hi is not None and v > hi:
            worst = max(worst, math.log2(v / hi))
    return worst


def same_batch_tier(label: str, batch: int) -> bool:
    return _sc.base_cell(label).split("|")[3] == _grid.b_tier(int(batch))


def _covers(label: str, leaf: str) -> bool:
    return leaf == label or leaf.startswith(label + "#")


def margin_members(
    label: str,
    pool: Mapping[str, Sequence[Dict[str, Any]]],
    margin: float,
) -> List[Dict[str, Any]]:
    """GEMMs of `pool` ({leaf: [gemm]}) outside `label`'s box by at most
    `margin` octaves, in the same batch tier, from leaves `label` does not
    cover. The pool decides isolation: pass only GEMMs that are training
    GEMMs of their own cell."""
    if margin <= 0:
        return []
    box = cell_box(label)
    out: List[Dict[str, Any]] = []
    for leaf in sorted(pool):
        if _covers(label, leaf):
            continue
        for g in pool[leaf]:
            if not same_batch_tier(label, g["batch_count"]):
                continue
            d = octaves_outside(box, g["m"], g["n"], g["k"])
            if 0 < d <= margin:
                out.append(g)
    return out


def margin_pools(
    cells: Mapping[str, Sequence[Dict[str, Any]]],
    labels: Iterable[str],
    margin: float,
    own_training: Callable[[Sequence[Dict[str, Any]]], Sequence[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """{label: margin GEMMs} for every label: `margin_members` over the
    pool of each leaf's own training GEMMs, `own_training(cells[leaf])`, so
    a GEMM another cell holds out for validation never enters a margin."""
    if margin <= 0:
        return {lbl: [] for lbl in labels}
    pool = {leaf: list(own_training(gs)) for leaf, gs in cells.items()}
    return {lbl: margin_members(lbl, pool, margin) for lbl in labels}


def distance_bucket(d: float, edges: Iterable[float] = (0.25, 0.5, 1.0)) -> str:
    """'<a', 'a-b', ..., '>=z' label of distance `d` over the bucket edges."""
    edges = list(edges)
    if d < edges[0]:
        return f"<{edges[0]:g}"
    for a, b in zip(edges, edges[1:]):
        if d < b:
            return f"{a:g}-{b:g}"
    return f">={edges[-1]:g}"
