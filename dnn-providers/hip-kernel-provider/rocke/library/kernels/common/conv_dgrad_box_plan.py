# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tap-box packing plans for the dgrad ``pp_boxes`` pixel-list kernels.

A ``pp_boxes`` 7 binary runs records of up to ``tile_m`` dX pixels; a record
runs the filter offsets of its *box* (one interval of filter offsets per
spatial axis) on every row, so a row whose own offsets are a strict subset of
the box wastes MFMAs on zero rows. ``pp_boxes`` 7 packs the exact pixel
classes into whole tiles and merges the class remainders greedily, which
grows the boxes of many tiles to (nearly) the full filter.

``pp_boxes`` 8 (this module) plans the records by the makespan of the
one-round split-K launch instead: for a target slice length ``T`` (K tiles)
every box ``B`` gets the split degree ``S_B = ceil(K tiles of B / T)``, and
the pixels are assigned to boxes that contain their offsets so that the
launch's workgroups, sum over the records of tiles * N tiles * S_B, fit the
CU budget. The smallest feasible ``T`` wins; among its plans the one with the
fewest MFMA K tiles. The assignment is solved by a small exact search over
the candidate boxes (see :func:`plan_boxes`), so the plan is a deterministic
function of the problem and the tile.

Pure host-side planning: the binary is the ``pp_boxes`` 7 pixel-list kernel.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

Box = Tuple[int, int, int, int, int, int]  # (zlo, zhi, ylo, yhi, xlo, xhi)


def _axis_ranges(n_in: int, n_out: int, pad: int, filt: int, dil: int = 1) -> List[Tuple[int, int]]:
    """Filter offsets [lo, hi] each input coordinate can use (stride 1)."""
    out = []
    for h in range(n_in):
        lo = max(0, h + pad - n_out + 1)
        hi = min(filt - 1, h + pad)
        out.append((lo, hi))
    return out


def pixel_classes(p) -> Dict[Box, List[int]]:
    """dX pixels grouped by their exact filter-offset box (stride 1, dilation 1)."""
    Di = p.Di if p.is_3d else 1
    dr = _axis_ranges(p.Di, p.Do, p.pD, p.Z) if p.is_3d else [(0, 0)]
    hr = _axis_ranges(p.Hi, p.Ho, p.pH, p.Y)
    wr = _axis_ranges(p.Wi, p.Wo, p.pW, p.X)
    hw = p.Hi * p.Wi
    cl: Dict[Box, List[int]] = {}
    for n in range(p.N):
        for d in range(Di):
            for h in range(p.Hi):
                for w in range(p.Wi):
                    key = dr[d] + hr[h] + wr[w]
                    cl.setdefault(key, []).append((n * Di + d) * hw + h * p.Wi + w)
    return cl


def box_taps(b: Box) -> int:
    return (b[1] - b[0] + 1) * (b[3] - b[2] + 1) * (b[5] - b[4] + 1)


def _contains(big: Box, c: Box) -> bool:
    return all(big[2 * i] <= c[2 * i] and c[2 * i + 1] <= big[2 * i + 1] for i in range(3))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _solve_milp(counts: Dict[Box, int], boxes: Sequence[Box], cost: Dict[Box, int], taps_k: Dict[Box, int],
                tile_m: int, n_tiles: int, budget: int):
    """Assign the class pixels to boxes containing them: minimise the MFMA K tiles
    (sum of tiles * K tiles of the box) with the launch's workgroups (sum of tiles *
    split * N tiles) within ``budget``. A small MILP (scipy's HiGHS); None if
    infeasible or scipy is missing."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:  # pragma: no cover - planner falls back to pp_boxes 7
        return None
    classes = sorted(counts)
    boxes = sorted(boxes)
    pairs = [(ci, bi) for ci, c in enumerate(classes) for bi, b in enumerate(boxes) if _contains(b, c)]
    nx, nb = len(pairs), len(boxes)
    obj = np.concatenate([np.zeros(nx), np.array([float(taps_k[b]) for b in boxes])])
    A = np.zeros((len(classes) + nb + 1, nx + nb))
    lo = np.zeros(len(classes) + nb + 1)
    hi = np.zeros(len(classes) + nb + 1)
    for k, (ci, bi) in enumerate(pairs):
        A[ci, k] = 1
        A[len(classes) + bi, k] = 1
    for ci, c in enumerate(classes):
        lo[ci] = hi[ci] = counts[c]
    for bi, b in enumerate(boxes):
        A[len(classes) + bi, nx + bi] = -tile_m
        lo[len(classes) + bi] = -np.inf
        A[-1, nx + bi] = cost[b] * n_tiles
    lo[-1], hi[-1] = 0, budget
    res = milp(obj, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nx + nb),
               bounds=Bounds(0, np.inf), options={"time_limit": 30.0})
    if res.x is None or res.status != 0:
        return None
    x = np.round(res.x).astype(int)
    plan = []
    for bi, b in enumerate(boxes):
        t = int(x[nx + bi])
        if not t:
            continue
        members = [(classes[ci], int(x[k])) for k, (ci, bb) in enumerate(pairs) if bb == bi and x[k]]
        plan.append((b, t, members))
    return plan


@lru_cache(maxsize=64)
def _plan_cached(key) -> Tuple:
    (counts_items, tile_m, n_tiles, kpt, max_split, budget) = key
    counts = dict(counts_items)
    classes = list(counts)
    # candidate boxes: per axis an interval [lo, hi] from the class bounds.
    axis_iv = []
    for a in range(3):
        lo_set = sorted({c[2 * a] for c in classes})
        hi_set = sorted({c[2 * a + 1] for c in classes})
        axis_iv.append([(lo, hi) for lo in lo_set for hi in hi_set if lo <= hi])
    boxes = [i + j + k for i in axis_iv[0] for j in axis_iv[1] for k in axis_iv[2]]
    boxes = [b for b in boxes if any(_contains(b, c) for c in classes)]
    taps_k = {b: box_taps(b) * kpt for b in boxes}
    need = max(taps_k[c] for c in classes)  # the longest class must fit max_split
    for T in range(_ceil_div(need, max_split), max(taps_k.values()) + 1):
        cost = {b: _ceil_div(taps_k[b], T) for b in boxes}
        cand = [b for b in boxes if cost[b] <= max_split]
        plan = _solve_milp(counts, cand, cost, taps_k, tile_m, n_tiles, budget)
        if plan is not None:
            return T, tuple((b, t, tuple(m)) for b, t, m in plan)
    return None


def plan_boxes(p, tile_m: int, tile_n: int, tile_k: int, max_split: int, cus: int):
    """``(T, [(box, tiles, split, [pixel rows])])`` of the ``pp_boxes`` 8 plan, or None."""
    cl = pixel_classes(p)
    counts = tuple(sorted((c, len(v)) for c, v in cl.items()))
    kpt = _ceil_div(p.kpg, tile_k)
    n_tiles = _ceil_div(p.cpg, tile_n)
    res = _plan_cached((counts, tile_m, n_tiles, kpt, max(1, max_split), cus))
    if res is None:
        return None
    T, plan = res
    pools = {c: list(v) for c, v in cl.items()}
    out = []
    for b, tiles, members in plan:
        rows: List[int] = []
        for c, n in members:
            rows += pools[c][:n]
            pools[c] = pools[c][n:]
        out.append((b, tiles, _ceil_div(box_taps(b) * kpt, T), sorted(rows)))
    return T, out
