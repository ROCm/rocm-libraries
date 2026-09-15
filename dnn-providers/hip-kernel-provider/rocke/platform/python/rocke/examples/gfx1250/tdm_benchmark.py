# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A/B the gfx1250 global->LDS load paths at a fixed tile.

The staged sweep searches the whole space and ``universal_gemm_verify`` picks
its own tile, so neither answers "is TDM faster than direct-to-LDS for *this*
tile". This runs one shape across the load paths and prints TFLOP/s side by
side. Every candidate is verified against the host reference before it is
timed, so a wrong kernel reports WRONG rather than a fast number.

    python3 -m rocke.examples.gfx1250.tdm_benchmark \\
        --m 4096 --n 4096 --k 4096 --tile 256 256 64 --warps 4 4 --lds-k-pad 8
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from rocke.sweep import build_all_instances

from .bf16_gemm_sweep import (
    PreparedProblem,
    _make_spec,
    benchmark_record,
    verify_record,
)

# Trait overrides per load path. "vgpr" is the baseline that stages A/B through
# registers; the rest are the two direct-to-LDS forms and the two TDM depths.
LOAD_PATHS: Dict[str, Dict[str, object]] = {
    "vgpr": {},
    "dtl": {"direct_to_lds": True},
    "dtl_pf": {"direct_to_lds": True, "dtl_prefetch": True},
    "tdm1": {"tdm": True, "tdm_depth": 1},
    "tdm2": {"tdm": True, "tdm_depth": 2},
}
DEFAULT_PATHS = ("dtl_pf", "tdm1", "tdm2")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--tile", type=int, nargs=3, default=(256, 256, 64),
                   metavar=("M", "N", "K"))
    p.add_argument("--warps", type=int, nargs=2, default=(4, 4), metavar=("M", "N"))
    p.add_argument("--lds-k-pad", type=int, default=8)
    p.add_argument("--pipeline", default="mem")
    p.add_argument("--waves-per-eu", type=int, default=None)
    p.add_argument("--paths", default=",".join(DEFAULT_PATHS),
                   help=f"comma-separated subset of {sorted(LOAD_PATHS)}, or 'all'")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--cache-dir", type=Path,
                   default=Path.home() / ".cache" / "rocke_tdm_benchmark")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    names = sorted(LOAD_PATHS) if args.paths == "all" else args.paths.split(",")
    unknown = [n for n in names if n not in LOAD_PATHS]
    if unknown:
        raise SystemExit(f"unknown load path(s) {unknown}; pick from {sorted(LOAD_PATHS)}")

    config = {
        "target": {
            "arch": "gfx1250",
            "isa": "amdgcn-amd-amdhsa--gfx1250",
            "dtype": args.dtype,
            "layout": "RCR",
            "wave_size": 32,
            "warp_tile": [16, 16, 32],
        }
    }
    tile_m, tile_n, tile_k = args.tile
    warp_m, warp_n = args.warps
    base = _make_spec(
        config, tile_m, tile_n, tile_k, warp_m, warp_n, args.pipeline,
        scheduler="intrawave", epilogue="default",
        waves_per_eu=args.waves_per_eu, lds_swizzle=False, lds_k_pad=args.lds_k_pad,
    )

    shape = (args.m, args.n, args.k)
    print(f"{args.dtype} {args.m}x{args.n}x{args.k}  tile={tile_m}x{tile_n}x{tile_k} "
          f"warps={warp_m}x{warp_n} lds_k_pad={args.lds_k_pad}")
    problem = PreparedProblem.create(shape, args.n, with_reference=True, dtype=args.dtype)
    results: List[Tuple[str, float]] = []
    try:
        for name in names:
            spec = replace(base, trait=replace(base.trait, **LOAD_PATHS[name]))
            record = build_all_instances(
                [spec], cache_dir=args.cache_dir, arch="gfx1250",
                isa=config["target"]["isa"], parallel=1,
            )[0]
            if not record.ok:
                print(f"  {name:8} BUILD FAILED  {record.error.splitlines()[0][:90]}")
                continue
            check: Dict[str, object] = {}
            if not verify_record(problem, record, check, tolerance=args.tolerance):
                print(f"  {name:8} WRONG  {check.get('incorrect')}/{check.get('elements')} "
                      f"elements, max|diff|={check.get('max_abs_diff')}")
                continue
            timing = benchmark_record(
                problem, record, arch="gfx1250",
                warmup=args.warmup, iters=args.iters, attempts=args.attempts,
            )
            results.append((name, float(timing["tflops"])))
            print(f"  {name:8} {timing['tflops']:8.1f} TFLOP/s  "
                  f"({timing['ms']:.3f} ms)  verified")
    finally:
        problem.close()

    if len(results) > 1:
        best_name, best = max(results, key=lambda r: r[1])
        print(f"\nfastest: {best_name} at {best:.1f} TFLOP/s")
        for name, tflops in results:
            if name != best_name:
                print(f"  {name:8} {tflops / best:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
