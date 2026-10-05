#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Score what dispatch actually ships against the measured corpus.

model.py is the readable reference; what runs in production is
``dispatch.grouped_convolution.fwd_group_merge_for_geometry()``, which inlines
the same arithmetic so that ``dispatch`` keeps importing nothing. Two copies of
one formula is a drift hazard, so this script closes the loop a reviewer would
otherwise have to close by eye:

  1. **Do the two copies agree?** Every measured shape is put through both; any
     disagreement is a defect, not a tuning difference, and is reported per
     shape.

  2. **Does the shipped policy reproduce the fitted model's quality?** Scored as
     the geometric mean over shapes of the realised fraction -- what share of
     each shape's own measured best the single modelled pick keeps. A ratio
     within a shape, so no absolute figure is involved.

The second question is also the comparison a reviewer wants against the rule
this replaced, so the previous three-cap heuristic (tile fit, a flat ceiling, an
occupancy floor) is reimplemented here and scored side by side on the same rows.
It is kept *here* rather than in dispatch precisely because it is dead code
whose only remaining job is to be the baseline in this report.

Needs a locally generated degrees.csv -- see README.md.

Usage:
    python3 score_shipped.py [--verbose]
"""

import argparse
import math
import sys
from pathlib import Path

try:
    from rocke.assets import library_root
except ImportError:  # loose script, rocke not installed in this interpreter
    # examples/gfx950/conv_fwd/merge_degree_model/ -> platform/python
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from rocke.assets import library_root

import corpus_fwd
import model

# ``library`` is build-time-only and never installed.
sys.path.insert(0, str(library_root()))

from dispatch.grouped_convolution import (  # noqa: E402
    _GFX950_TILE_K,
    _GFX950_TILE_M,
    _GFX950_TILE_N,
    fwd_group_merge_for_geometry,
)

# The rule this replaced, preserved only as the baseline for the report below:
# fit the N tile, never exceed a flat ceiling, and refuse to merge away so many
# CTAs that the machine runs dry. Hand-fitted to a 35-shape corpus.
_LEGACY_CEILING = 32
_LEGACY_MIN_CTAS = 512


def legacy_pick(groups, m, y, x, tile_m, tile_n):
    """Pre-model three-cap rule, descending so the largest legal degree wins."""
    if y * x <= 1:
        return 1
    for gm in (64, 32, 16, 8, 4, 2):
        if gm > min(tile_n, _LEGACY_CEILING) or groups % gm:
            continue
        if m > 0 and -(-m // tile_m) * (groups // gm) < _LEGACY_MIN_CTAS:
            continue
        return gm
    return 1


def geomean(xs):
    return (
        math.exp(sum(math.log(max(v, 1e-9)) for v in xs) / len(xs))
        if xs
        else float("nan")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="list every shape the shipped policy leaves >5% short",
    )
    args = ap.parse_args()

    cfgs = corpus_fwd.measured()
    if not cfgs:
        raise SystemExit("no measured corpus present — see README.md")

    tm, tn, tk = _GFX950_TILE_M, _GFX950_TILE_N, _GFX950_TILE_K
    print(f"{len(cfgs)} measured shapes at the shipped tile {tm}x{tn}x{tk}\n")

    drift = []
    rows = []
    for c in cfgs:
        m = c.m()
        shipped = fwd_group_merge_for_geometry(
            c.groups,
            m,
            tm,
            tn,
            y=c.y,
            x=c.x,
            stride=c.stride,
            esize=c.esize,
            tile_k=tk,
        )
        local = model.pick(c.groups, m, c.y, c.x, c.stride, tm, tn, tk, c.esize)
        if shipped != local:
            drift.append((c.name, shipped, local))
        rows.append((c, shipped, legacy_pick(c.groups, m, c.y, c.x, tm, tn)))

    # --- 1. the two copies must be the same function ------------------------
    if drift:
        print(f"DRIFT: {len(drift)} shapes disagree between dispatch and model.py")
        for name, s, l in drift[:20]:
            print(f"  {name:<36} dispatch={s:<3} model.py={l}")
        print()
    else:
        print("dispatch and model.py agree on every shape\n")

    # --- 2. quality of the pick, shipped vs the rule it replaced ------------
    print("=" * 78)
    print("realised fraction of each shape's own measured best")
    print("=" * 78)
    print(f"{'policy':<12} {'geomean':>9} {'exact':>10} {'>5% short':>11} {'worst':>8}")
    print("-" * 78)
    for label, idx in (("shipped", 1), ("legacy", 2)):
        fr = [r[0].realised(r[idx]) for r in rows]
        exact = sum(r[idx] == r[0].oracle for r in rows)
        short = sum(f < 0.95 for f in fr)
        print(
            f"{label:<12} {geomean(fr):>9.4f} {exact:>6}/{len(rows):<3} "
            f"{short:>11} {min(fr):>8.3f}"
        )

    if args.verbose:
        print()
        print("shapes the shipped policy leaves more than 5% short:")
        bad = sorted(
            ((r[0].realised(r[1]), r[0], r[1]) for r in rows),
            key=lambda t: t[0],
        )
        for frac, c, gm in bad:
            if frac >= 0.95:
                break
            print(
                f"  {c.name:<38} picked {gm:<3} oracle {c.oracle:<3} "
                f"realised {frac:.3f}"
            )

    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
