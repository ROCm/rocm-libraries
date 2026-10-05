#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Score candidate depthwise-wgrad merge policies against the measured surface.

This is the file that decided what ships. It evaluates the policy dispatch now
uses -- *keep the shipped tile, merge as hard as the kernel gate allows* -- side
by side with the three richer policies that were fitted and rejected, so the
rejection can be re-derived rather than taken on trust.

The policy space has three knobs, all of which collapsed to their neutral value:

``tiles``
    Which ``tile_n`` rungs the policy may spend. Widening the tile is the only
    way a large filter merges at all (at the shipped width a 7x7 stage cannot
    merge, since ``49 * 2`` already exceeds it), so a ladder reaches shapes the
    single-tile rule cannot. It also carries the whole regression tail.
``target``
    Stop climbing rungs once the degree reaches this. Motivated by vector width
    -- the merged run is ``Gm`` elements long and buffer loads saturate -- but
    at a fixed tile the ``spatial * Gm <= tile_n`` bound binds first on nearly
    every shape, so it only ever interacts with ``tiles``.
``floor``
    Refuse to merge past the point where fewer than ``groups // floor`` merged
    groups remain, on the theory that merging divides the grid and starves the
    machine. Split-K is resolved *after* merging and refills the grid, so this
    mostly declines free speedup.

**Read the geometric mean, not the aggregate.** An aggregate ``sum(oracle) /
sum(pick)`` is time-weighted: one expensive shape can hide a hundred cheap
regressions, and during this fit it did -- it ranked the grid floor as worth
having by a margin smaller than run-to-run noise, while the per-shape geometric
mean showed it losing on more shapes than it won. Both are reported; only the
geometric mean, measured against *the point that ships today* rather than
against the unreachable oracle, decided anything.

Usage::

    python3 score_policy.py deg_tn64.csv deg_tn128.csv deg_tn256.csv
    python3 score_policy.py --shipped deg_tn64.csv      # one rung is enough
"""

import argparse
import math
import sys

from corpus_wgrad import LADDER, load, max_admissible

#: The shipped wgrad tile. Pinned, not swept: see ``_GFX950_TILE_N``.
SHIPPED_TILE_N = 64


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _admissible(spatial, groups, tile_n, floor=1):
    """``max_admissible`` with an optional surviving-group floor applied."""
    cap = groups // max(floor, 1)
    best = 1
    for gm in LADDER:
        if gm <= cap and gm <= groups and groups % gm == 0 and spatial * gm <= tile_n:
            best = gm
    return best


def shipped(spatial, groups):
    """The policy dispatch ships: one tile, maximum admissible degree."""
    if groups <= 1:
        return SHIPPED_TILE_N, 1
    return SHIPPED_TILE_N, max_admissible(spatial, groups, SHIPPED_TILE_N)


def ladder(spatial, groups, tiles, target=8, floor=1):
    """A rejected policy: climb ``tile_n`` until the degree reaches ``target``.

    Falls back to the narrowest rung offering the best degree any rung offers,
    so a filter too large to merge anywhere keeps the shipped tile rather than
    paying for width it gets nothing from. (Letting it widen regardless was the
    worst variant measured, by a wide margin.)
    """
    if groups <= 1:
        return SHIPPED_TILE_N, 1
    widest = (SHIPPED_TILE_N, 1)
    for tn in tiles:
        gm = _admissible(spatial, groups, tn, floor)
        if gm >= target:
            return tn, gm
        if gm > widest[1]:
            widest = (tn, gm)
    return widest


def score(best, meta, pick_fn):
    """Apply a policy and summarise it against both reference points."""
    vs_base, vs_oracle, by_gm, tiles_used = [], [], {}, {}
    exact = 0
    regressions = []
    for s, d in best.items():
        m = meta[s]
        base = d.get((SHIPPED_TILE_N, 1))
        key = pick_fn(m["Y"] * m["X"], m["groups"])
        if base is None or key not in d:
            continue  # that rung was not measured for this shape
        tiles_used[key[0]] = tiles_used.get(key[0], 0) + 1
        pick = d[key]
        ratio = base / pick
        vs_base.append(ratio)
        vs_oracle.append(min(d.values()) / pick)
        by_gm.setdefault(key[1], []).append(ratio)
        if ratio < 0.98:
            regressions.append((ratio, s, key))
        at_tile = {g: v for (t, g), v in d.items() if t == key[0]}
        if at_tile and min(at_tile, key=lambda g: at_tile[g]) == key[1]:
            exact += 1
    regressions.sort()
    return dict(
        n=len(vs_base),
        vs_base=sorted(vs_base),
        geo_base=_geomean(vs_base),
        geo_oracle=_geomean(vs_oracle),
        exact=exact,
        regressions=regressions,
        by_gm=by_gm,
        tiles=tiles_used,
    )


def _report(label, r, verbose=False):
    v = r["vs_base"]
    print(f"{label:<34} n={r['n']}")
    print(
        f"  vs shipped (tile {SHIPPED_TILE_N}, Gm=1): "
        f"geomean {r['geo_base']:.3f}x  median {v[len(v) // 2]:.3f}x  "
        f"p90 {v[9 * len(v) // 10]:.2f}x  max {v[-1]:.1f}x"
    )
    worst = r["regressions"][0] if r["regressions"] else None
    print(
        f"  regressions (<0.98x)   : {len(r['regressions'])}/{r['n']}"
        + (f"   worst {worst[0]:.3f}x at {worst[2]}" if worst else "")
    )
    print(f"  vs full (tile, Gm) oracle : geomean {r['geo_oracle']:.3f}")
    print(f"  best degree at its own tile on {r['exact']}/{r['n']} shapes")
    print(f"  tile mix: {dict(sorted(r['tiles'].items()))}")
    if verbose:
        print(f"  {'Gm':>4} {'n':>4} {'geomean vs shipped':>20}")
        for k in sorted(r["by_gm"]):
            print(f"  {k:>4} {len(r['by_gm'][k]):>4} {_geomean(r['by_gm'][k]):>19.3f}x")
        for ratio, s, key in r["regressions"][:10]:
            print(f"    {ratio:.3f}x  {key}  {s}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+", help="deg_tn*.csv from measure_degrees.py")
    ap.add_argument(
        "--shipped",
        action="store_true",
        help="score only the shipped policy (needs just the tile-64 sweep)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    best, meta = load(args.csv)
    if not best:
        print("no measured corpus present -- run measure_degrees.py first")
        return 0
    tiles = tuple(sorted({tn for d in best.values() for tn, _ in d}))
    print(f"{len(best)} shapes, measured tiles {tiles}\n")

    _report("SHIPPED: one tile, max degree", score(best, meta, shipped), True)
    if args.shipped:
        return 0
    if len(tiles) < 2:
        print("(only one rung measured -- the rejected ladders need the others)")
        return 0
    for floor in (1, 4):
        _report(
            f"rejected: ladder, floor={floor}",
            score(best, meta, lambda sp, g, f=floor: ladder(sp, g, tiles, floor=f)),
            args.verbose,
        )
    _report(
        "rejected: ladder, no target cap",
        score(best, meta, lambda sp, g: ladder(sp, g, tiles, target=10**9)),
        args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
