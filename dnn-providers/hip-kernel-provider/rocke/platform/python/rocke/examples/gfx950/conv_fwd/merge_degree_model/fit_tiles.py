#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Refit the degree model on the pooled 8-tile corpus.

The shipped constants were fitted at 64x64x64 and nothing else. tile_m enters
the cost in three separate places -- the CTA count, the memory bucket's bytes
per tile, and the brake's working-set footprint -- so with tile_m constant only
the products ``64*W`` and the ratio ``64/C`` were ever identified. Any one of
the three could be carrying the other two's weight and the fit could not tell.

Pooling eight tiles makes tile_m vary over {32, 64, 128, 256} and tile_k over
{16, 32, 64}, which is what separates them. The split holds out whole *shapes*,
not shape-tile pairs: a shape present in TRAIN at one tile and in TEST at
another would leak its degree curve across the split, and the degree curve is
precisely what is under test.

Usage:
    python3 fit_tiles.py [--draws 300000] [--seeds 3]
"""

import argparse
import random
from pathlib import Path

import numpy as np

import fitfast
import model
from corpus_fwd import Config
from score_tiles import load
from tiles import TILES

HERE = Path(__file__).resolve().parent


def configs_for(spec, d):
    """Config list for one tile, normalised against that tile's own Gm=1."""
    path = d / f"{spec.replace('/', '_')}.csv"
    if not path.exists():
        return []
    curves, geom = load(path)
    out = []
    for key, best in curves.items():
        if 1 not in best or best[1] <= 0.0:
            continue  # no unmerged baseline -> no normalisable curve
        g = geom[key]
        base = best[1]
        out.append(
            Config(
                g["name"], g["n"], g["hi"], g["wi"], g["groups"], g["y"],
                g["x"], g["stride"], g["dtype"],
                {k: v / base for k, v in best.items()}, "measured",
                mo=g["m"], dilation=g["dilation"], wo_=g["wo"],
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dir", default="tiles_out")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--brake", default="all", choices=("mem", "all"))
    ap.add_argument("--combine", default="sum", choices=("max", "sum"))
    args = ap.parse_args()
    d = HERE / args.dir

    # --- load every tile, remembering which tile each config came from -----
    by_tile = {}
    for spec, _src in TILES:
        cfgs = configs_for(spec, d)
        if cfgs:
            by_tile[spec] = cfgs
    names = sorted({c.name for cfgs in by_tile.values() for c in cfgs})
    print(f"{len(by_tile)} tiles, {len(names)} distinct shapes")

    # --- split on the SHAPE, so no curve straddles train/test --------------
    rng = random.Random(1234)
    shuffled = names[:]
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * args.test_frac))
    test_names = set(shuffled[:cut])
    print(f"train {len(names) - len(test_names)} shapes, test {len(test_names)}\n")

    def tab(which):
        """Concatenated tables over all tiles, each with its own tile triple."""
        parts = []
        for spec, cfgs in by_tile.items():
            tm, tn, tk = (int(v) for v in spec.split("/")[0].split("x"))
            sel = [c for c in cfgs if (c.name in test_names) == (which == "test")]
            if not sel:
                continue
            parts.append(fitfast.tabulate(sel, tile_m=tm, tile_n=tn, tile_k=tk))
        keys = parts[0].keys()
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}

    def tab_one(spec, which="test"):
        tm, tn, tk = (int(v) for v in spec.split("/")[0].split("x"))
        sel = [c for c in by_tile[spec]
               if (c.name in test_names) == (which == "test")]
        return fitfast.tabulate(sel, tile_m=tm, tile_n=tn, tile_k=tk), sel

    Ttr = tab("train")
    seeds = list(range(args.seeds))

    shipped = np.array([getattr(model.DEFAULT, f) for f in fitfast.FIELDS])

    print("refitting on pooled TRAIN ...", flush=True)
    p, key = fitfast.restarts(Ttr, args.draws, seeds, args.combine,
                              brake=args.brake)
    print(f"  pooled train geomean {key[0]:.4f}")
    print("  shipped: " + "  ".join(f"{f}={v:.5g}"
                                    for f, v in zip(fitfast.FIELDS, shipped)))
    print("  refit  : " + "  ".join(f"{f}={v:.5g}"
                                    for f, v in zip(fitfast.FIELDS, p)))

    print()
    print("=" * 92)
    print("held-out TEST geomean, per tile")
    print("=" * 92)
    print(
        f"{'tile':<18} {'n':>5} {'shipped':>9} {'pooled':>9} {'per-tile':>9} "
        f"{'d(pool)':>8} {'headroom':>9}"
    )
    print("-" * 92)
    tot_s, tot_r, tot_o = [], [], []
    for spec, _src in TILES:
        if spec not in by_tile:
            continue
        Tv, sel = tab_one(spec)
        if not sel:
            continue
        ev = lambda q: fitfast.summarise(
            *fitfast.evaluate(Tv, q[None], args.combine, args.brake), Tv
        )[0][0]
        gs, gr = ev(shipped), ev(p)
        # Upper bound for this functional form at this tile: fit the tile's own
        # TRAIN rows and score its own TEST rows. If this is barely above the
        # pooled number, the form transfers and only the constants were off; if
        # it is far above, the form itself is missing a tile_m mechanism.
        Tt, _ = tab_one(spec, "train")
        po, _ = fitfast.restarts(Tt, args.draws, seeds, args.combine,
                                 brake=args.brake)
        go = ev(po)
        tot_s.append(gs)
        tot_r.append(gr)
        tot_o.append(go)
        print(
            f"{spec:<18} {len(sel):>5} {gs:>9.4f} {gr:>9.4f} {go:>9.4f} "
            f"{gr - gs:>+8.4f} {go - gr:>+9.4f}"
        )
    print("-" * 92)
    print(
        f"{'mean over tiles':<18} {'':>5} {np.mean(tot_s):>9.4f} "
        f"{np.mean(tot_r):>9.4f} {np.mean(tot_o):>9.4f} "
        f"{np.mean(tot_r) - np.mean(tot_s):>+8.4f} "
        f"{np.mean(tot_o) - np.mean(tot_r):>+9.4f}"
    )


if __name__ == "__main__":
    main()
