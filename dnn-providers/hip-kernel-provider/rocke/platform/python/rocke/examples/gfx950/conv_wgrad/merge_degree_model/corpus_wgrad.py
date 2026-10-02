#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Load the measured depthwise-wgrad ``(tile_n, group_merge)`` surface.

``measure_degrees.py`` writes one CSV per ``tile_n`` rung, each row a ranked
kernel config for one shape. This module collapses those rows into the only
thing the policy question needs: for every shape, the best observed time at
every ``(tile_n, Gm)`` pair, minimised over everything else the sweep varied
(split-K, pipeline, epilogue, warp shape).

Two parsing quirks are worth knowing about:

* **``Gm`` is recovered from the kernel name, not from a column.** The ranked-row
  CSV writer emits no ``group_merge`` field, and the merged and unmerged kernels
  for one shape are otherwise indistinguishable in the row. ``gm<N>`` is in the
  name because ``ConvGroupedSpec.kernel_name`` puts it there -- merging changes
  the GEMM, so it must not share a compile-cache key.
* **The ``shape`` column is the driver's own descriptor string**, so the filter
  extent and group count are re-parsed out of it rather than trusted from the
  separate columns, which carry the *post-normalisation* values.

Nothing here is specific to the shipped policy; ``score_policy.py`` is what
applies a policy to this surface.
"""

import csv
import re
import sys
from collections import defaultdict

_GM = re.compile(r"_gm(\d+)_")
_SHAPE = re.compile(r"N(\d+)H(\d+)W(\d+)C(\d+)_K(\d+)Y(\d+)X(\d+)(?:G(\d+))?")

#: The degrees the kernel gate offers, in ascending order.
LADDER = (2, 4, 8, 16, 32, 64)


def gm_of(kernel_name):
    """Merge degree encoded in a kernel name; 1 when the kernel is unmerged."""
    m = _GM.search(kernel_name)
    return int(m.group(1)) if m else 1


def parse_shape(s):
    """Filter/tensor extents out of the driver's shape descriptor."""
    m = _SHAPE.match(s)
    if not m:
        return None
    n, h, w, c, k, y, x, g = m.groups()
    return dict(
        N=int(n),
        H=int(h),
        W=int(w),
        C=int(c),
        K=int(k),
        Y=int(y),
        X=int(x),
        G=int(g) if g else 1,
    )


def load(paths):
    """Read sweep CSVs -> ``(best, meta)``.

    ``best[shape][(tile_n, gm)]`` is the fastest observed time for that pair;
    ``meta[shape]`` carries the geometry the policy is a function of.
    """
    best = defaultdict(dict)
    meta = {}
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                ms = r.get("rocke_ms")
                if not ms:
                    continue
                try:
                    ms = float(ms)
                except ValueError:
                    continue
                if ms <= 0:
                    continue
                key = (int(r["tile_n"]), gm_of(r["kernel_name"]))
                s = r["shape"]
                d = best[s]
                if key not in d or ms < d[key]:
                    d[key] = ms
                if s not in meta:
                    m = parse_shape(s) or {}
                    m["groups"] = int(r["groups"])
                    m["dtype"] = r["dtype"]
                    m["sH"] = int(r["sH"])
                    meta[s] = m
    return best, meta


def max_admissible(spatial, groups, tile_n):
    """Largest gate-admissible degree: ladder, divisibility, and tile bound.

    The same three request-dependent clauses as
    ``kernels.common.conv_implicit_gemm_wgrad.wgrad_group_merge_available``,
    and -- with ``tile_n`` pinned to the shipped tile -- the whole of the policy
    ``dispatch.grouped_convolution._wgrad_merge_degree`` ships.
    """
    best = 1
    for gm in LADDER:
        if gm <= groups and groups % gm == 0 and spatial * gm <= tile_n:
            best = gm
    return best


def main():
    paths = sys.argv[1:] or ["deg_tn64.csv"]
    best, meta = load(paths)
    tiles = sorted({tn for d in best.values() for tn, _ in d})
    print(f"{len(best)} shapes, tile_n values {tiles}, from {paths}\n")
    print(f"{'shape':<40} {'sp':>3} {'G':>5} {'oracle(tn,gm)':>14} {'x base':>8}")
    rows = []
    for s, d in best.items():
        base = d.get((64, 1))
        if base is None:
            continue
        otn, ogm = min(d, key=lambda k: d[k])
        rows.append((s, meta[s], (otn, ogm), base / d[(otn, ogm)]))
    rows.sort(key=lambda r: -r[3])
    for s, m, (otn, ogm), sp in rows:
        spatial = m.get("Y", 0) * m.get("X", 0)
        print(
            f"{s:<40} {spatial:>3} {m['groups']:>5} "
            f"{f'({otn},{ogm})':>14} {sp:>8.2f}"
        )


if __name__ == "__main__":
    main()
