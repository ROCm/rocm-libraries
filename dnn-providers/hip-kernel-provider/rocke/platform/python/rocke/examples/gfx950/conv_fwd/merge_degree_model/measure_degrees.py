#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure the group-merge degree curve for a corpus of depthwise fwd shapes.

The sweep driver has no tile-pinning switch, and a full tile x warp x pipeline
x epilogue sweep per shape is untenable for 400 shapes. But the swept axes are
module globals read at call time, so pinning them to the single configuration
dispatch actually ships collapses the per-shape work to exactly the degree axis
-- the one thing under test -- without touching the shipped benchmark.

What is pinned (gfx950 fwd, from dispatch.grouped_convolution):
    tile 64x64x64, warp 2x2, warp_tile_mn 32, pipeline "mem", epilogue
    "cshuffle"

What is swept: group_merge over the full admissible set. ``--group-merge-window
-1 --unmerged-frac 1.0`` are forced, because anything narrower would score the
degree policy against itself.

Usage:
    python3 measure_degrees.py shapes_fwd.txt -o degrees.csv [--jobs 32]
"""

import argparse
import sys
from pathlib import Path

try:
    from rocke.assets import library_root, platform_root
except ImportError:  # loose script, rocke not installed in this interpreter
    # examples/gfx950/conv_fwd/merge_degree_model/ -> platform/python
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from rocke.assets import library_root, platform_root

# gfx950 forward depthwise merged candidate, as dispatch builds it.
PIN = {
    "_TILE_MN": (64,),
    "_TILE_MN_GFX1250": (64,),
    "_TILE_K": (64,),
    "_WARP_MN": (2,),
    "_WARP_MN_GFX1250": (2,),
    "_WARP_TILE_MN": (32,),
    "_PIPELINES": ("mem",),
    "_EPILOGUES": ("cshuffle",),
}


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
    globals cannot work: ``(128, 32)`` x ``(4, 1)`` also builds 128x128, 32x128
    and 32x32 at every legal warp split -- nine tiles where one was asked for,
    nine times the kernels to compile and measure.

    The benchmark reads ``itertools`` out of its own module namespace, so
    rebinding that name lets the exact (tile_m, tile_n, warp_m, warp_n) survive
    and drops the rest, without editing the shipped sweep. Only the 8-iterable
    combo product is touched, and only when its first two axes are the same
    object -- every other ``product`` call in the module passes straight
    through.
    """

    def __init__(self, inner, tile):
        self._inner = inner
        self._want = (tile[0], tile[1], tile[3], tile[4])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def product(self, *iters, **kw):
        out = self._inner.product(*iters, **kw)
        if len(iters) != 8 or tuple(iters[0]) != tuple(iters[1]):
            return out
        want = self._want
        return [c for c in out if (c[0], c[1], c[3], c[4]) == want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes", help="MIOpenDriver command file from gen_shapes.py")
    ap.add_argument("-o", "--out", default="degrees.csv")
    ap.add_argument("--jobs", type=int, default=0, help="0 = all cores")
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
    if missing:
        sys.exit(f"benchmark module has no {missing} — sweep axes moved, update PIN")
    for k, v in pin.items():
        setattr(B, k, v)
    if tile and (tile[0] != tile[1] or tile[3] != tile[4]):
        B.itertools = _ExactTile(itertools, tile)
    print(f"pinned sweep axes to tile {args.tile or '64x64x64/2x2/32 (shipped)'}:")
    for k, v in pin.items():
        print(f"  {k:<22}= {v}")

    argv = [
        "benchmark_implicit_gemm_conv",
        "--miopen-file", args.shapes,
        "--direction", "fwd",
        "--arch", args.arch,
        "--group-merge-window", "-1",   # full degree axis: no policy feedback
        "--unmerged-frac", "1.0",
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--jobs", str(args.jobs),
        "--csv", args.out,
        "--top", "8",
        # The CSV cap defaults to 5 ranked rows per case, which silently drops
        # the *slowest* degrees -- and on depthwise the slowest is Gm=1, the
        # baseline every curve is normalised against. Keep the whole sweep.
        "--csv-top", "999",
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
