#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Score the shipped degree model at every tile in tiles.py.

Two questions, deliberately kept apart:

  1. *Does the degree model transfer?* Per tile, compare the degree it picks
     against the degree that actually won at that same tile. The model was
     fitted at 64x64x64 only, and tile_m enters its cost in three places, so
     this is the test for whether those three uses were separately identified
     or merely co-fitted. Scored as geomean of (chosen / best-at-that-tile),
     which is a ratio of ratios and carries no absolute figure.

  2. *Is 64x64x64 the right tile?* Per shape, compare the best achievable at
     each tile against the best achievable at the shipped tile. This is a
     separate axis from the degree and is reported as a speedup ratio.

Usage:
    python3 score_tiles.py [--dir tiles_out]
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import model
from tiles import TILES

HERE = Path(__file__).resolve().parent
_SHAPE_RE = re.compile(r"N(\d+)H(\d+)W(\d+)C(\d+)_K(\d+)Y(\d+)X(\d+)G(\d+)")


def slug(spec):
    return spec.replace("/", "_")


def load(path):
    """Raw per-(shape, degree) best TFLOP/s from one single-tile CSV.

    Returns ``{request_key: {degree: tflops}}`` plus the geometry needed to ask
    the model for a pick. Only verified rows count -- a degree that is fast
    because the diagonal mask is wrong must never set an oracle.
    """
    groups, geom = defaultdict(dict), {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("passed", "").strip() != "True":
                continue
            if row.get("direction", "fwd") != "fwd":
                continue
            try:
                tf = float(row["rocke_tflops"])
            except (TypeError, ValueError):
                continue
            if tf <= 0.0:
                continue
            mm = _SHAPE_RE.match(row["shape"])
            if not mm:
                continue
            key = (
                row["shape"], row["dtype"], row["sH"], row["sW"],
                row["pH"], row["pW"], row["dH"], row["dW"],
            )
            if key not in geom:
                n, hi, wi, _c, _k, y, x, g = (int(v) for v in mm.groups())
                sh, ph, dh = int(row["sH"]), int(row["pH"]), int(row["dH"])
                sw, pw, dw = int(row["sW"]), int(row["pW"]), int(row["dW"])
                ho = (hi + 2 * ph - dh * (y - 1) - 1) // sh + 1
                wo = (wi + 2 * pw - dw * (x - 1) - 1) // sw + 1
                nm = f"N{n}_H{hi}W{wi}_G{g}_Y{y}X{x}"
                if sh != 1:
                    nm += f"_s{sh}"
                if dh != 1:
                    nm += f"_d{dh}"
                geom[key] = dict(
                    name=nm, groups=g, m=n * ho * wo, y=y, x=x, stride=sw,
                    esize=2 if row["dtype"] != "fp32" else 4,
                    n=n, hi=hi, wi=wi, wo=wo, dilation=dh, dtype=row["dtype"],
                )
            gm = int(row["group_merge"] or 1)
            groups[key][gm] = max(groups[key].get(gm, 0.0), tf)
    return groups, geom


def geomean(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tiles_out")
    args = ap.parse_args()
    d = HERE / args.dir

    per_tile, best_at = {}, defaultdict(dict)

    print("=" * 92)
    print("1. does the degree model transfer?  (fitted at 64x64x64 only)")
    print("=" * 92)
    print(
        f"{'tile':<18} {'n':>5} {'geomean':>9} {'exact':>7} "
        f"{'<0.95':>6} {'worst':>7}  source"
    )
    print("-" * 92)

    for spec, src in TILES:
        path = d / f"{slug(spec)}.csv"
        if not path.exists():
            print(f"{spec:<18} {'--':>5}  (not measured yet)")
            continue
        tm, tn, tk = (int(v) for v in spec.split("/")[0].split("x"))
        curves, geom = load(path)

        fracs, exact, bad, worst, worst_name = [], 0, 0, 1.0, ""
        for key, best in curves.items():
            if not best:
                continue
            g = geom[key]
            for gm, tf in best.items():
                best_at[g["name"]][(spec, gm)] = tf
            pick = model.pick(
                groups=g["groups"], m=g["m"], y=g["y"], x=g["x"],
                stride=g["stride"], tile_m=tm, tile_n=tn, tile_k=tk,
                esize=g["esize"],
            )
            if pick not in best:
                continue  # model chose a degree the sweep never built
            top = max(best.values())
            frac = best[pick] / top
            fracs.append(frac)
            oracle = max(best, key=lambda k: best[k])
            exact += pick == oracle
            bad += frac < 0.95
            if frac < worst:
                worst, worst_name = frac, g["name"]
        per_tile[spec] = (fracs, exact, bad, worst, worst_name)
        print(
            f"{spec:<18} {len(fracs):>5} {geomean(fracs):>9.4f} {exact:>7} "
            f"{bad:>6} {worst:>7.3f}  {src}"
        )

    for spec, (_f, _e, _b, w, nm) in per_tile.items():
        if w < 0.80:
            print(f"    worst at {spec}: {nm} -> {w:.3f}")

    # --- 2. which tile is actually best ------------------------------------
    ctrl = TILES[0][0]
    print()
    print("=" * 92)
    print(f"2. is {ctrl} the right tile?  best-at-tile / best-at-control, per shape")
    print("=" * 92)
    print(f"{'tile':<18} {'n':>5} {'geomean':>9} {'wins':>6} {'best ratio':>11}")
    print("-" * 92)

    for spec, _src in TILES:
        if spec == ctrl:
            continue
        ratios, wins, top_r, top_n = [], 0, 0.0, ""
        for name, obs in best_at.items():
            here = [v for (s, _g), v in obs.items() if s == spec]
            base = [v for (s, _g), v in obs.items() if s == ctrl]
            if not here or not base:
                continue
            r = max(here) / max(base)
            ratios.append(r)
            wins += r > 1.0
            if r > top_r:
                top_r, top_n = r, name
        if not ratios:
            continue
        print(
            f"{spec:<18} {len(ratios):>5} {geomean(ratios):>9.4f} "
            f"{wins:>6} {top_r:>10.2f}x  ({top_n})"
        )

    # per-shape best tile, as a histogram
    champion = defaultdict(int)
    for name, obs in best_at.items():
        by_tile = defaultdict(float)
        for (s, _g), v in obs.items():
            by_tile[s] = max(by_tile[s], v)
        if by_tile:
            champion[max(by_tile, key=lambda k: by_tile[k])] += 1
    if champion:
        print()
        print("best tile per shape:")
        for spec, n in sorted(champion.items(), key=lambda kv: -kv[1]):
            print(f"  {spec:<18} {n:>4}")


if __name__ == "__main__":
    main()
