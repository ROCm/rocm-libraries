#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure the (tile_n, group-merge) surface for a corpus of depthwise wgrad shapes.

The sweep driver has no tile-pinning switch, and a full tile x warp x pipeline
x epilogue x split-K sweep per shape is untenable for a 200-shape corpus. But
the swept axes are module globals read at call time, so pinning them collapses
the per-shape work to the axes actually under test, without touching the
shipped benchmark.

Why this sweeps a *surface* and not a curve
-------------------------------------------
The forward harness pins one tile and sweeps the degree, because forward's tile
bound (``Gm <= tile_n``) is slack for every degree the gate offers -- the tile
is not part of the decision. Wgrad's bound is ``spatial * Gm <= tile_n``, which
binds hard: at the shipped ``tile_n = 64`` a 3x3 stage caps at Gm=4, a 5x5 at
Gm=2, and a 7x7 cannot merge at all. The degree you are allowed to pick is a
function of the tile you picked, so the two have to be measured jointly and the
policy has to answer with both.

Hence ``--tile``: one run pins one tile, the tile lands in the CSV, and runs at
different tiles concatenate into one corpus that the fit reads as a surface.

What is pinned (gfx950 wgrad, from dispatch.grouped_convolution):
    tile 64x64x64, warp 2x2, warp_tile_mn 32, pipeline "mem", epilogue
    "default"

What is swept: group_merge over the full admissible set, and split_k over the
auto ladder. split_k has to stay swept rather than pinned -- merging divides the
grid by Gm and split-K multiplies it back, so the best degree at a fixed split_k
is not the best degree, and a curve measured that way scores the two knobs
against each other instead of against the baseline.

Usage:
    python3 measure_degrees.py shapes_wgrad.txt -o deg_tn64.csv
    python3 measure_degrees.py shapes_wgrad.txt -o deg_tn128.csv \
        --tile 64x128x64/2x2/32
    python3 measure_degrees.py shapes_wgrad.txt -o deg_tn256.csv \
        --tile 64x256x64/2x2/32
"""

import argparse
import sys
from pathlib import Path

try:
    from rocke.assets import library_root, platform_root
except ImportError:  # loose script, rocke not installed in this interpreter
    # examples/gfx950/conv_wgrad/merge_degree_model/ -> platform/python
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from rocke.assets import library_root, platform_root

# gfx950 wgrad depthwise candidate, as dispatch builds it today (Gm=1).
PIN = {
    "_TILE_MN": (64,),
    "_TILE_MN_GFX1250": (64,),
    "_TILE_K": (64,),
    "_WARP_MN": (2,),
    "_WARP_MN_GFX1250": (2,),
    "_WARP_TILE_MN": (32,),
    "_PIPELINES": ("mem",),
    "_EPILOGUES": ("default",),
}

# The full degree ladder the kernel gate offers. The shipped benchmark default
# stops at 16, which is enough for forward but truncates wgrad: a 1x1 or 3x3
# depthwise stage at tile_n=256 admits 32 and 64, and those are exactly the
# rungs that tell us whether the curve has turned over yet.
GROUP_MERGE_SWEEP = (2, 4, 8, 16, 32, 64)


def parse_tile(spec):
    """``MxNxK/wmXwn/atom`` -> ``(tile_m, tile_n, tile_k, warp_m, warp_n, atom)``."""
    mnk, warps, atom = spec.split("/")
    tm, tn, tk = (int(v) for v in mnk.split("x"))
    wm, wn = (int(v) for v in warps.split("x"))
    return tm, tn, tk, wm, wn, int(atom)


def pin_for_tile(tile):
    """PIN dict holding the sweep to the M/N values of one tile."""
    tm, tn, tk, wm, wn, atom = tile
    mn = (tm,) if tm == tn else (tm, tn)
    wmn = (wm,) if wm == wn else (wm, wn)
    return dict(
        PIN,
        _TILE_MN=mn,
        _TILE_MN_GFX1250=mn,
        _TILE_K=(tk,),
        _WARP_MN=wmn,
        _WARP_MN_GFX1250=wmn,
        _WARP_TILE_MN=(atom,),
    )


class _ExactTile:
    """``itertools`` shim that collapses the sweep's square tile cross product.

    ``_TILE_MN`` and ``_WARP_MN`` each feed *both* the M and N axis of
    ``itertools.product``, so pinning an asymmetric tile by narrowing the
    globals cannot work: ``(64, 256)`` x ``(2,)`` also builds 64x64, 256x256
    and 256x64 -- four tiles where one was asked for, four times the kernels to
    compile and measure. Every wgrad tile this harness cares about is
    asymmetric (``tile_m`` stays 64 because depthwise ``wg_M`` is 1), so this
    is the common path here, not a corner.

    The benchmark reads ``itertools`` out of its own module namespace, so
    rebinding that name lets the exact (tile_m, tile_n, warp_m, warp_n) survive
    and drops the rest, without editing the shipped sweep. Only the 6-iterable
    combo product is touched, and only when its first two axes are the same
    object -- every other ``product`` call in the module passes straight
    through.

    Note the arity: wgrad's combo product has 6 iterables where forward's has
    8, because wgrad carries pipeline/epilogue/split_k/Gm outside the product
    rather than inside it. A shim written against the forward arity silently
    does nothing here.
    """

    def __init__(self, inner, tile):
        self._inner = inner
        self._want = (tile[0], tile[1], tile[3], tile[4])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def product(self, *iters, **kw):
        out = self._inner.product(*iters, **kw)
        if len(iters) != 6 or tuple(iters[0]) != tuple(iters[1]):
            return out
        want = self._want
        return [c for c in out if (c[0], c[1], c[3], c[4]) == want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes", help="MIOpenDriver command file from gen_shapes.py")
    ap.add_argument("-o", "--out", default="degrees.csv")
    ap.add_argument("--jobs", type=int, default=8, help="0 = all cores")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument(
        "--tile",
        default=None,
        help="tile to pin, as MxNxK/warpMxwarpN/warpTileMN (default: the "
        "shipped 64x64x64/2x2/32). The tile lands in the CSV, so runs at "
        "different tiles concatenate into one corpus.",
    )
    ap.add_argument(
        "--group-merge",
        default=",".join(str(g) for g in GROUP_MERGE_SWEEP),
        help="comma-separated degree ladder to sweep",
    )
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    tile = parse_tile(args.tile) if args.tile else None
    pin = pin_for_tile(tile) if tile else PIN

    # ``library`` is build-time-only and never installed, so it has to go on the
    # path by hand; ``benchmarks/common`` because the sweep driver's own
    # siblings import each other by bare module name.
    lib = library_root()
    sys.path.insert(0, str(lib))
    sys.path.insert(0, str(platform_root() / "python"))
    sys.path.insert(0, str(lib / "benchmarks" / "common"))

    import itertools

    import benchmark_implicit_gemm_conv as B

    missing = [k for k in pin if not hasattr(B, k)]
    if not hasattr(B, "_GROUP_MERGE_SWEEP"):
        missing.append("_GROUP_MERGE_SWEEP")
    if missing:
        sys.exit(f"benchmark module has no {missing} — sweep axes moved, update PIN")
    for k, v in pin.items():
        setattr(B, k, v)
    B._GROUP_MERGE_SWEEP = tuple(int(v) for v in args.group_merge.split(",") if v)
    if tile and (tile[0] != tile[1] or tile[3] != tile[4]):
        B.itertools = _ExactTile(itertools, tile)
    print(f"pinned sweep axes to tile {args.tile or '64x64x64/2x2/32 (shipped)'}:")
    for k, v in pin.items():
        print(f"  {k:<22}= {v}")
    print(f"  {'_GROUP_MERGE_SWEEP':<22}= {B._GROUP_MERGE_SWEEP}")

    argv = [
        "benchmark_implicit_gemm_conv",
        "--miopen-file",
        args.shapes,
        "--direction",
        "wgrad",
        "--arch",
        args.arch,
        # 0 = sweep the auto split-K ladder. Pinning it would score merging
        # against a split_k chosen for the unmerged grid, which is the one
        # comparison guaranteed to flatter merging.
        "--split-k",
        "0",
        # Merged tiles cannot use the packed-atomic epilogue -- it has no way to
        # drop an off-diagonal group pair -- so the merged family is two-stage
        # only. Forcing two-stage for the unmerged arm too keeps the baseline
        # on the same pipeline as the thing it is a baseline for.
        "--two-stage",
        "always",
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--jobs",
        str(args.jobs),
        "--csv",
        args.out,
        "--top",
        "8",
        # The CSV cap defaults to 5 ranked rows per case, which silently drops
        # the *slowest* configs -- and on depthwise the slowest is Gm=1, the
        # baseline every curve is normalised against. Keep the whole sweep.
        "--csv-top",
        "999",
        # Verifies every kernel, not just the first (the flag's help text is
        # stale). A merged degree that is fast because the diagonal mask is
        # wrong must not be allowed to set an oracle.
        "--verify",
        *args.extra,
    ]
    sys.argv = argv
    print("argv:", " ".join(argv[1:]), flush=True)
    raise SystemExit(B.main())


if __name__ == "__main__":
    main()
