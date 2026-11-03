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
split-K launch instead. For a candidate slice length ``T`` (K tiles) every
tap box ``B`` gets the split degree ``S_B = ceil(K tiles of B / T)``; a small
MILP (scipy's HiGHS, no time limit: at most 5 classes per axis, so <= 125
classes x 216 boxes) assigns the pixel classes to boxes that contain their
offsets, minimising the MFMA K tiles with the launch's workgroups (tiles x N
tiles x S_B) within the CU budget (``pp_box_cus``). Each candidate is scored
by the simulated greedy dispatch of its blocks on the CUs; the lowest
makespan wins, then the fewest K tiles. The plan is a deterministic function
of the problem, the tile and the budget (cached per process). Without scipy
(or a feasible plan) the launch falls back to the ``pp_boxes`` 7 packing.

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
               bounds=Bounds(0, np.inf))  # no time limit: a deterministic optimum (<= 125 classes x 216 boxes)
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


def _makespan(plan, cost, taps_k, n_tiles: int, cus: int) -> int:
    """K tiles per CU of the launch: its blocks (records longest slice first, a
    record's tiles x N tiles x slices) dispatched in order to the CU that frees
    first."""
    import heapq

    recs = sorted(plan, key=lambda e: (-_ceil_div(taps_k[e[0]], cost[e[0]]), e[0]))
    h = [(0, i) for i in range(cus)]
    mk = 0
    for b, t, _m in recs:
        S = cost[b]
        per = _ceil_div(taps_k[b], S)
        for _ in range(t * n_tiles):
            for s_ in range(S):
                ln = max(0, min(per, taps_k[b] - s_ * per))
                t0, c = heapq.heappop(h)
                heapq.heappush(h, (t0 + ln, c))
                mk = max(mk, t0 + ln)
    return mk


@lru_cache(maxsize=64)
def _plan_cached(key) -> Tuple:
    (counts_items, tile_m, n_tiles, kpt, max_split, budget, round_cus) = key
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
    full = [b for b in boxes if all(_contains(b, c) for c in classes)]
    # Candidate slice lengths: those at which some box's slices are exact.
    cands = sorted({_ceil_div(taps_k[b], k) for b in boxes for k in range(1, max_split + 1)})
    cands = [t for t in cands if t >= _ceil_div(need, max_split)]
    best = None
    for T in cands:
        if best is not None and T >= best[0][0]:
            break  # the makespan is at least T
        cost = {b: _ceil_div(taps_k[b], T) for b in boxes}
        cand = [b for b in boxes if cost[b] <= max_split]
        if not any(b in cand for b in full):
            continue  # every class needs a home: the full box must be allowed
        plan = _solve_milp(counts, cand, cost, taps_k, tile_m, n_tiles, budget)
        if plan is None:
            continue
        mk = _makespan(plan, cost, taps_k, n_tiles, round_cus)
        steps = sum(t * taps_k[b] for b, t, _ in plan)
        if best is None or (mk, steps) < best[0]:
            best = ((mk, steps), T, tuple((b, t, tuple(m)) for b, t, m in plan))
    if best is not None:
        return best[1], best[2]
    return None


def plan_boxes(p, tile_m: int, tile_n: int, tile_k: int, max_split: int, cus: int, round_cus: int = 228):
    """``(T, [(box, tiles, split, [pixel rows])])`` of the ``pp_boxes`` 8 plan, or None."""
    cl = pixel_classes(p)
    counts = tuple(sorted((c, len(v)) for c, v in cl.items()))
    kpt = _ceil_div(p.kpg, tile_k)
    n_tiles = _ceil_div(p.cpg, tile_n)
    res = _plan_cached((counts, tile_m, n_tiles, kpt, max(1, max_split), cus, round_cus))
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
