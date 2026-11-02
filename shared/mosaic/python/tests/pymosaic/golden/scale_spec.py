# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared structure registry + seeded random-layout generator for the
*scale* golden tests.

The curated ``golden_vectors.json`` stores full per-index vectors for a
small, human-readable set. That doesn't scale to tens of thousands of
layouts. Instead, the scale path:

  1. Generates N random layouts over a FIXED set of structures (this
     module). The structures are the compile-time nesting profiles; the
     leaf values are random. This mirrors the JIT exactly: one compiled
     module per structure, many value sets through it.
  2. Has ``bench_golden.cpp`` compute, from the mosaic C++ headers, a
     64-bit FNV-1a hash of the full offset enumeration ``[L(0)..L(n-1)]``
     for each layout (plus size / cosize / repr). That hash is the
     ground truth.
  3. Has the Python ``Layout`` reproduce the enumeration via ``enumerate()``
     and checks the same hash.

The ``STRUCTURES`` list here MUST stay in lockstep with the ``sid`` switch
in ``bench_golden.cpp`` (same index, same nesting profile, same DFS leaf
order). Leaf order is left-to-right depth-first, matching both
``mosaic::flatten`` (C++) and ``_render.flatten`` (Python).
"""

from __future__ import annotations

import random
from typing import Any

from pymosaic import _render

# Each structure is a nesting profile: nested tuples with ``None`` at the
# leaves. Index == the ``sid`` understood by bench_golden.cpp's switch — the
# two MUST stay in lockstep (same index, same profile, same DFS leaf order).
#
# This set deliberately spans a wide range of compositions: flat ranks 1-5,
# 2-level layouts with assorted top/sub ranks, scalar+tuple mixes at every
# position, 1-tuple modes, and 3- and 4-level deep nestings.
_ = None
STRUCTURES: list[Any] = [
    (_,),  # 0  flat rank-1
    (_, _),  # 1  flat rank-2
    (_, _, _),  # 2  flat rank-3
    (_, _, _, _),  # 3  flat rank-4
    (_, _, _, _, _),  # 4  flat rank-5
    ((_, _), (_, _)),  # 5  2-level 2x2
    ((_, _, _), (_, _)),  # 6  2-level 3+2
    ((_, _), (_, _), (_, _)),  # 7  rank-3 of pairs
    ((_, _), _),  # 8  nested + scalar
    (_, (_, _)),  # 9  scalar + nested
    ((_, _), _, _),  # 10 nested + 2 scalars
    (_, (_, _), _),  # 11 scalar + nested + scalar
    ((_, _), (_, _), _),  # 12 2 pairs + scalar
    ((_,), (_,)),  # 13 two 1-tuples
    ((_,),),  # 14 rank-1 of a 1-tuple
    (((_, _), _), (_,)),  # 15 3-level
    ((_, (_, _)), _),  # 16 3-level mixed + scalar
    (((_, _), (_, _)), (_, _)),  # 17 pair-of-pairs + pair
    ((((_, _), _), _), _),  # 18 4-level deep
    (((_, _), _), ((_, _), _)),  # 19 blocked-of-blocked (symmetric)
    (_, ((_, _), _)),  # 20 scalar + 3-level mode
    ((_, _, _), _),  # 21 triple + scalar
    ((_, _), (_, _, _), _),  # 22 pair + triple + scalar
    (((_,),),),  # 23 deep single (rank-1 chain)
]

_MASK64 = 0xFFFFFFFFFFFFFFFF
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def fnv1a_offsets(offsets: list[int]) -> int:
    """64-bit FNV-1a over the little-endian 8-byte encoding of each offset.

    Byte-for-byte identical to ``fnv1a_offsets`` in bench_golden.cpp so the
    Python enumeration can be checked against the C++ ground-truth hash.
    """
    h = _FNV_OFFSET
    for o in offsets:
        v = o & _MASK64
        for b in range(8):
            h ^= (v >> (8 * b)) & 0xFF
            h = (h * _FNV_PRIME) & _MASK64
    return h


def _renest(profile: Any, it) -> Any:
    """Re-nest a flat iterator of leaf values into ``profile``'s shape."""
    if isinstance(profile, tuple):
        return tuple(_renest(p, it) for p in profile)
    return next(it)


def _leaf_count(profile: Any) -> int:
    if isinstance(profile, tuple):
        return sum(_leaf_count(p) for p in profile)
    return 1


# Per-leaf-count value range for shape leaves — keeps the total domain size
# modest (enumerate() is O(size)) while still covering a wide value space.
def _shape_leaf_range(n_leaves: int) -> int:
    if n_leaves <= 1:
        return 48
    if n_leaves == 2:
        return 10
    if n_leaves == 3:
        return 6
    if n_leaves == 4:
        return 4
    if n_leaves == 5:
        return 3
    return 2


def generate_specs(n: int, seed: int = 0xC0FFEE) -> list[dict]:
    """Generate ``n`` random layout specs, deterministically from ``seed``.

    Each spec dict carries:
      - ``sid``           : structure index into STRUCTURES
      - ``shape``/``stride``: nested int-tuples (for Python Layout construction)
      - ``shape_leaves`` / ``stride_leaves``: flat DFS leaf lists (for the
        whitespace-delimited file bench_golden.cpp reads)

    Strides are a mix of compact row-major, compact col-major, and small
    random positive values — all congruent to the shape by construction.
    """
    rng = random.Random(seed)
    specs: list[dict] = []
    for _ in range(n):
        sid = rng.randrange(len(STRUCTURES))
        profile = STRUCTURES[sid]
        k = _leaf_count(profile)
        hi = _shape_leaf_range(k)

        shape_leaves = [rng.randint(1, hi) for _ in range(k)]
        shape = _renest(profile, iter(shape_leaves))

        mode = rng.randrange(3)
        if mode == 0:
            stride = _render.compact_row_major(shape)
        elif mode == 1:
            stride = _render.compact_col_major(shape)
        else:
            stride = _renest(profile, iter(rng.randint(1, 8) for _ in range(k)))

        specs.append(
            {
                "sid": sid,
                "shape": shape,
                "stride": stride,
                "shape_leaves": list(_render.flatten(shape)),
                "stride_leaves": list(_render.flatten(stride)),
            }
        )
    return specs


def write_cpp_input(specs: list[dict], path) -> None:
    """Write the whitespace-delimited spec file bench_golden.cpp consumes.

    One line per layout::  ``sid k  s0 .. s{k-1}  d0 .. d{k-1}``
    """
    lines = []
    for s in specs:
        leaves = s["shape_leaves"] + s["stride_leaves"]
        k = len(s["shape_leaves"])
        lines.append(f"{s['sid']} {k} " + " ".join(str(x) for x in leaves))
    # Leading count lets the C++ side reserve / sanity-check.
    with open(path, "w") as f:
        f.write(f"{len(specs)}\n")
        f.write("\n".join(lines))
        f.write("\n")
