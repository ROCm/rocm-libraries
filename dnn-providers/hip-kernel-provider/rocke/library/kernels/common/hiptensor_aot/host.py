# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side argument builders of the generic kernels (reference for hipTensor's C++ planner).

hipTensor's rocKE backend computes these arguments in C++ (``rocke_plan.cpp``); this module is
the Python statement of the same rules, used by the numeric tests to launch the kernels:

* :func:`ew_args` -- canonical modes of an elementwise problem (extent-1 modes dropped, ordered
  by the output strides, contiguous modes merged), the kernel slots, magic numbers and grid;
* :func:`reduce_args` -- kept / reduced modes, row or col kernel, split and grid.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .common import MAX_RANK, as_i32_bits, magic_u31

I32_MAX = (1 << 31) - 1
GRID_YZ_MAX = 65535


def canonicalize(
    dims: List[Tuple[int, List[int]]], key: int
) -> List[Tuple[int, List[int]]]:
    """``dims``: ``(extent, [stride per tensor])``. Drops extent-1 modes, orders by the strides of
    tensor ``key`` and merges modes contiguous in every tensor."""
    dims = [d for d in dims if d[0] != 1]
    dims.sort(key=lambda d: abs(d[1][key]))
    out: List[Tuple[int, List[int]]] = []
    for e, st in dims:
        if out:
            le, ls = out[-1]
            if all(st[t] == ls[t] * le for t in range(len(st))) and le * e <= I32_MAX:
                out[-1] = (le * e, ls)
                continue
        out.append((e, list(st)))
    return out


def _magic(vals: Dict[str, int], m: str, s: str, d: int) -> None:
    mm, ss = magic_u31(d)
    vals[m] = as_i32_bits(mm)
    vals[s] = ss


def ew_args(dims, names: Sequence[str], block: int = 256):
    """Kernel argument values (sizes, strides, magic numbers) and grid of a generic elementwise
    launch over canonical ``dims`` whose stride lists follow ``names`` (e.g. ``["A", "D"]``).
    """
    rank = len(dims)
    inner = min(3, rank)
    prod = lambda ds: __import__("math").prod(d[0] for d in ds)  # noqa: E731
    while inner > 1 and prod(dims[:inner]) > I32_MAX:
        inner -= 1
    if rank - inner > MAX_RANK - 3:
        raise ValueError("too many modes")
    slots = [(1, [0] * len(names)) for _ in range(MAX_RANK)]
    for i in range(inner):
        slots[i] = dims[i]
    for i in range(inner, rank):
        slots[3 + i - inner] = dims[i]
    n_in = prod(slots[:3])
    n_out = prod(slots[3:])
    vals: Dict[str, int] = dict(n_in=n_in, n_out=n_out, ystep=min(n_out, GRID_YZ_MAX))
    for i, (e, st) in enumerate(slots):
        vals[f"e{i}"] = e
        vals[f"lim{i}"] = e
        for t, n in enumerate(names):
            vals[f"s{n}{i}"] = st[t] if e != 1 else 0
    for i in (0, 1, 3, 4, 5, 6):
        _magic(vals, f"m{i}", f"s{i}", slots[i][0])
    idx64 = any(
        sum((e - 1) * abs(st[t]) for e, st in dims) > I32_MAX for t in range(len(names))
    )
    grid = (-(-n_in // block), vals["ystep"], 1)
    return vals, grid, idx64


def reduce_args(kept, red, n_cus: int = 256, block: int = 256, split=None):
    """``kept``: ``(extent, [A, C, D strides])``, ``red``: ``(extent, [A stride])`` canonical
    (ordered by A's strides). Returns ``(mode, values, grid, finalize_grid, split)``."""
    import math

    kept = list(kept) + [(1, [0, 0, 0])] * (MAX_RANK - len(kept))
    red = list(red) + [(1, [0])] * (MAX_RANK - len(red))
    n_out = math.prod(k[0] for k in kept)
    n_red = math.prod(r[0] for r in red)
    row = n_red > 1 and (n_out == 1 or abs(red[0][1][0]) < abs(kept[0][1][0]))
    vals: Dict[str, int] = dict(n_out=n_out)
    for i in range(MAX_RANK):
        e, (sa, sc, sd) = kept[i]
        vals[f"ke{i}"] = e
        vals[f"kA{i}"] = sa if e != 1 else 0
        vals[f"kC{i}"] = sc if e != 1 else 0
        vals[f"kD{i}"] = sd if e != 1 else 0
        er, (sr,) = red[i]
        vals[f"re{i}"] = er
        vals[f"rA{i}"] = sr if er != 1 else 0
        if i < MAX_RANK - 1:
            _magic(vals, f"km{i}", f"ks{i}", e)
            _magic(vals, f"rm{i}", f"rs{i}", er)
    if row:
        nchunk = -(-red[0][0] // 64)
        trips = nchunk * (n_red // red[0][0])
        if split is None:
            split = max(1, min(trips, -(-(8 * n_cus * 4) // n_out)))
        chunk = -(-trips // split)
        split = -(-trips // chunk)
        vals.update(nchunk=nchunk, jchunk=chunk, split=split)
        _magic(vals, "jm", "js", nchunk)
        grid = (-(-n_out // (block // 64)), split, 1)
    else:
        outer = n_out // kept[0][0]
        gx = -(-kept[0][0] // block)
        gy = min(outer, GRID_YZ_MAX)
        if split is None:
            split = max(1, min(n_red, -(-(4 * n_cus) // (gx * gy))))
        chunk = -(-n_red // split)
        split = -(-n_red // chunk)
        vals.update(n_red=n_red, rchunk=chunk, ystep=gy, n_outer=outer, split=split)
        grid = (gx, gy, split)
    return ("row" if row else "col"), vals, grid, (-(-n_out // block), 1, 1), split
