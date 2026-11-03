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
MILP (scipy's HiGHS, deterministic node / gap limits, no time limit: <= 125
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

import os as _os
from functools import lru_cache

try:  # the exact assignment (HiGHS MILP); without scipy the greedy twin plans
    import scipy.optimize as _sco  # noqa: F401

    _SOLVER = _os.environ.get("ROCKE_PBX8_SOLVER", "milp")
except ImportError:  # pragma: no cover
    _SOLVER = "greedy"
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


def pixel_classes(p, interior: int = 0) -> Dict[Box, List[int]]:
    """dX pixels grouped by their exact filter-offset box (stride 1, dilation 1).

    ``interior`` (KernelIdentity.dx_interior_axes bit mask, 1 W / 2 H / 4 D;
    0 = none): the (F - 1) / 2 halo shell of those axes gets no pixel."""
    Di = p.Di if p.is_3d else 1
    sd = (p.Z - 1) // 2 if (p.is_3d and interior & 4) else 0
    sh = (p.Y - 1) // 2 if interior & 2 else 0
    sw = (p.X - 1) // 2 if interior & 1 else 0
    dr = _axis_ranges(p.Di, p.Do, p.pD, p.Z) if p.is_3d else [(0, 0)]
    hr = _axis_ranges(p.Hi, p.Ho, p.pH, p.Y)
    wr = _axis_ranges(p.Wi, p.Wo, p.pW, p.X)
    hw = p.Hi * p.Wi
    cl: Dict[Box, List[int]] = {}
    for n in range(p.N):
        for d in range(sd, Di - sd):
            for h in range(sh, p.Hi - sh):
                for w in range(sw, p.Wi - sw):
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
               bounds=Bounds(0, np.inf),
               # Deterministic work limits only (a time limit would make the plan
               # depend on the machine's load): a node budget and a relative MIP gap.
               options={"node_limit": 4000, "mip_rel_gap": 2e-3})
    if res.x is None:
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


def _hull(a: Box, c: Box) -> Box:
    return (min(a[0], c[0]), max(a[1], c[1]), min(a[2], c[2]), max(a[3], c[3]),
            min(a[4], c[4]), max(a[5], c[5]))


def _solve_greedy(counts: Dict[Box, int], boxes: Sequence[Box], cost: Dict[Box, int], taps_k: Dict[Box, int],
                  tile_m: int, n_tiles: int, budget: int):
    """Pure-Python twin of :func:`_solve_milp` (same result format): every class
    starts in its own exact box; then (A) the spare rows of a box's last tile take
    the partial-tile rows of boxes it contains (a whole tile less, nothing more),
    and (B) while over ``budget``, the two partial tiles whose merge into their hull
    box saves workgroups at the least K-tile increase are merged. None if the
    budget cannot be met. Deterministic (fixed orders, exact ties broken by box)."""
    allowed = set(boxes)
    groups: Dict[Box, Dict[Box, int]] = {}
    for c in sorted(counts):
        if c not in allowed:
            return None
        groups.setdefault(c, {})[c] = counts[c]

    def rows(b):
        return sum(groups.get(b, {}).values())

    def tiles(b):
        return _ceil_div(rows(b), tile_m)

    def wgs():
        return sum(tiles(b) * cost[b] for b in groups) * n_tiles

    def move(src, dst, n):
        """Move n rows of src's group (classes in sorted order, last first) to dst."""
        g = groups[src]
        d = groups.setdefault(dst, {})
        for c in sorted(g, reverse=True):
            if not n:
                break
            k = min(n, g[c])
            g[c] -= k
            d[c] = d.get(c, 0) + k
            n -= k
            if not g[c]:
                del g[c]
        if not g:
            del groups[src]

    def fill():
        changed = True
        while changed:
            changed = False
            for B in sorted(groups, key=lambda b: (-cost[b], -taps_k[b], b)):
                if B not in groups:
                    continue
                spare = tiles(B) * tile_m - rows(B)
                best = None
                for g in sorted(groups):
                    if g == B or not _contains(B, g):
                        continue
                    r = rows(g) % tile_m
                    if r and r <= spare:
                        key = (cost[g], taps_k[g], g)
                        if best is None or key > best[0]:
                            best = (key, g, r)
                if best is not None:
                    move(best[1], B, best[2])
                    changed = True

    fill()
    while wgs() > budget:
        best = None
        cur_w = wgs()
        part = sorted(g for g in groups if rows(g) % tile_m)
        for i, g1 in enumerate(part):
            r1 = rows(g1) % tile_m
            for g2 in part[i + 1:]:
                r2 = rows(g2) % tile_m
                U = _hull(g1, g2)
                if U not in allowed:
                    continue
                # tiles after moving both partial tiles into U
                old = {g1: tiles(g1), g2: tiles(g2)}
                old_u = tiles(U) if U not in (g1, g2) else None
                ru = (rows(U) if U not in (g1, g2) else 0) + r1 + r2
                if U == g1:
                    ru = rows(g1) + r2
                elif U == g2:
                    ru = rows(g2) + r1
                t1 = _ceil_div(rows(g1) - r1, tile_m) if U != g1 else 0
                t2 = _ceil_div(rows(g2) - r2, tile_m) if U != g2 else 0
                tu = _ceil_div(ru, tile_m)
                dw = (t1 - old[g1]) * cost[g1] + (t2 - old[g2]) * cost[g2] + tu * cost[U]
                ds = (t1 - old[g1]) * taps_k[g1] + (t2 - old[g2]) * taps_k[g2] + tu * taps_k[U]
                if old_u is not None:
                    dw -= old_u * cost[U]
                    ds -= old_u * taps_k[U]
                if dw >= 0:
                    continue
                key = (ds / -dw, ds, U, g1, g2)
                if best is None or key < best[0]:
                    best = (key, g1, g2, r1, r2, U)
        if best is None:
            return None
        _k, g1, g2, r1, r2, U = best
        if U != g1:
            move(g1, U, r1)
        if U != g2:
            move(g2, U, r2)
        fill()
        if wgs() >= cur_w:  # pragma: no cover - a merge always saves workgroups
            return None
    plan = []
    for b in sorted(groups):
        t = tiles(b)
        if t:
            plan.append((b, t, sorted(groups[b].items())))
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
    # A problem whose unsplit tiles alone exceed the budget (big grids) runs
    # several rounds anyway: the budget grows by whole rounds until a plan fits.
    min_wgs = _ceil_div(sum(counts.values()), tile_m) * n_tiles
    budget = max(budget, _ceil_div(min_wgs, round_cus) * round_cus)
    best = None
    for T in cands:
        if best is not None and T >= best[0][0]:
            break  # the makespan is at least T
        cost = {b: _ceil_div(taps_k[b], T) for b in boxes}
        cand = [b for b in boxes if cost[b] <= max_split]
        if not any(b in cand for b in full):
            continue  # every class needs a home: the full box must be allowed
        solver = _solve_greedy if _SOLVER == "greedy" else _solve_milp
        plan = solver(counts, cand, cost, taps_k, tile_m, n_tiles, budget)
        if plan is None:
            continue
        mk = _makespan(plan, cost, taps_k, n_tiles, round_cus)
        steps = sum(t * taps_k[b] for b, t, _ in plan)
        if best is None or (mk, steps) < best[0]:
            best = ((mk, steps), T, tuple((b, t, tuple(m)) for b, t, m in plan))
    if best is not None:
        return best[1], best[2]
    return None


def records_makespan(sub_gemms, kpg: int, tile_k: int, cus: int) -> Tuple[int, int]:
    """``(makespan, K tiles)`` of a pp_boxes record list: the simulated makespan
    (blocks in order to the CU that frees first, as :func:`_makespan`) and the
    launch's total MFMA K tiles."""
    import heapq

    kt = _ceil_div(kpg, tile_k)
    h = [(0, i) for i in range(cus)]
    mk = tot = 0
    for sg in sub_gemms:
        span = sg.y_dot_slice * sg.x_dot_slice * sg.gemm_k_padded * kt
        S = max(1, sg.b_x_stride)
        per = _ceil_div(span, S)
        for i in range(sg.block_end - sg.block_start):
            ln = max(0, min(per, span - (i % S) * per))
            tot += ln
            t0, c = heapq.heappop(h)
            heapq.heappush(h, (t0 + ln, c))
            mk = max(mk, t0 + ln)
    return mk, tot


def plan_boxes(p, tile_m: int, tile_n: int, tile_k: int, max_split: int, cus: int, round_cus: int = 228,
               interior: int = 0):
    """``(T, [(box, tiles, split, [pixel rows])])`` of the ``pp_boxes`` 8 plan, or None.
    ``interior``: as :func:`pixel_classes` (dx_interior: the shell gets no row)."""
    cl = pixel_classes(p, interior)
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
