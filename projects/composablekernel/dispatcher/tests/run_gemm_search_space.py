#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Dispatcher universal GEMM search space runner.

Enumerates the full GemmKernelConfig search space by calling expand_sweep
with the tile_engine default_ci_config.json across all (dtype, layout) strata,
samples a budget-limited subset with a daily rotating seed, compiles each
config via the bridge, runs on GPU, and reports correctness + TFLOPS.

This is the dispatcher equivalent of tile_engine's gemm_universal_benchmark.py
driven with TILE_ENGINE_SAMPLING_TIER=daily.

Usage:
    python3 run_gemm_search_space.py --arch gfx942 --budget 500
    python3 run_gemm_search_space.py --dtypes fp16,bf16 --layouts rcr,rrr
    python3 run_gemm_search_space.py --budget 0   # full space, no cap
    python3 run_gemm_search_space.py --budget 500 --seed 42 --json results.json
"""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

_DISPATCHER = Path(__file__).parent.parent
sys.path.insert(0, str(_DISPATCHER / "python"))

import numpy as np

from ctypes_utils import detect_gpu_arch
from gemm_utils import GemmProblem, GpuGemmRunner, expand_sweep, setup_multiple_gemm_dispatchers

_CI_CONFIG = (
    _DISPATCHER.parent / "tile_engine/ops/gemm/configs/default_ci_config.json"
)
_DTYPES = ["fp16", "bf16", "fp8", "bf8"]
_LAYOUTS = ["rcr", "rrr", "crr", "ccr"]
_RTOL = 2e-2


def _daily_seed() -> int:
    return int(hashlib.md5(date.today().isoformat().encode()).hexdigest(), 16) % (2**31)


def _sample(strata: dict, budget: int, seed: int) -> list:
    """Stratified budget-limited sample: equal share per non-empty stratum."""
    nonempty = {k: v for k, v in strata.items() if v}
    if not nonempty:
        return []
    rng = random.Random(seed)
    n = len(nonempty)
    per = max(1, budget // n)
    remainder = budget - per * n
    selected = []
    for i, (_, configs) in enumerate(nonempty.items()):
        alloc = min(per + (remainder if i == n - 1 else 0), len(configs))
        selected.extend(rng.sample(configs, alloc))
    rng.shuffle(selected)
    return selected[:budget]


def _emulate(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        rounded = (u32 + ((u32 >> 16) & 1) + np.uint32(0x7FFF)) >> 16
        return (rounded.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float16).astype(np.float32)


def _max_rel(out: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(out - ref))) / (float(np.max(np.abs(ref))) + 1e-12)


def run(args) -> int:
    arch = args.arch or detect_gpu_arch()
    dtypes = [d.strip() for d in args.dtypes.split(",")]
    layouts = [l.strip() for l in args.layouts.split(",")]
    budget = args.budget
    seed = args.seed if args.seed is not None else _daily_seed()
    size = args.size

    print(f"arch={arch}  dtypes={dtypes}  layouts={layouts}")
    print(f"budget={budget if budget > 0 else 'unlimited'}  seed={seed}  size={size}")

    # --- Enumerate: same JSON config as tile_engine, across all strata ---
    print("\nEnumerating search space...")
    t0 = time.time()
    strata = {}
    for dtype in dtypes:
        for layout in layouts:
            configs = expand_sweep(str(_CI_CONFIG), arch=arch,
                                   dtype=dtype, layout=layout)
            strata[f"{dtype}/{layout}"] = configs
    total = sum(len(v) for v in strata.values())
    print(f"  {total} valid configs across {len(strata)} strata ({time.time()-t0:.1f}s)")
    for key, cfgs in sorted(strata.items()):
        print(f"    {key}: {len(cfgs)}")

    # --- Sample ---
    if budget > 0 and budget < total:
        configs = _sample(strata, budget=budget, seed=seed)
        print(f"\nSampled {len(configs)}/{total} configs (budget={budget})")
    else:
        configs = [c for v in strata.values() for c in v]
        print(f"\nRunning all {len(configs)} configs (no cap)")

    if not configs:
        print("ERROR: no configs to run")
        return 1

    # --- Build ---
    print(f"\nBuilding {len(configs)} .so files (parallel JIT)...")
    t0 = time.time()
    so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)
    build_ok = sum(1 for p in so_paths if p is not None)
    print(f"  Built {build_ok}/{len(configs)} in {time.time()-t0:.1f}s "
          f"({len(configs)-build_ok} failed)")

    # --- Run ---
    print(f"\nRunning {build_ok} kernels on GPU (M=N=K={size})...")
    problem = GemmProblem(M=size, N=size, K=size)
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)

    results = []
    n_pass = n_fail = n_build_fail = 0

    for cfg, so in zip(configs, so_paths):
        if so is None:
            n_build_fail += 1
            results.append({"name": cfg.name, "status": "build_fail"})
            continue
        try:
            result = GpuGemmRunner(lib_path=so).run(A, B, problem)
        except Exception as exc:
            n_fail += 1
            results.append({"name": cfg.name, "status": "run_error", "error": str(exc)})
            continue
        if not result.success:
            n_fail += 1
            results.append({"name": cfg.name, "status": "run_fail",
                            "error": f"status={result.status}"})
            continue
        ref = _emulate(_emulate(A, cfg.dtype_a) @ _emulate(B, cfg.dtype_a), cfg.dtype_a)
        mr = _max_rel(result.output, ref)
        ok = mr <= _RTOL
        n_pass += ok
        n_fail += not ok
        results.append({"name": cfg.name, "status": "pass" if ok else "fail",
                        "tflops": round(result.tflops, 3), "max_rel": round(mr, 6)})

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} pass / {n_fail} fail / {n_build_fail} build-fail "
          f"/ {len(configs)} total")

    if args.json:
        out = {"arch": arch, "size": size, "seed": seed, "budget": budget,
               "total_configs": total, "n_pass": n_pass, "n_fail": n_fail,
               "n_build_fail": n_build_fail, "kernels": results}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Results written to {args.json}")

    if n_fail > 0:
        print("\nFailed kernels:")
        for r in results:
            if r["status"] not in ("pass", "build_fail"):
                print(f"  {r['name']}: {r['status']} "
                      f"{r.get('error', r.get('max_rel', ''))}")

    return 0 if n_fail == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dispatcher universal GEMM search space runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arch", default=None,
                   help="GPU arch (default: auto-detect via rocminfo)")
    p.add_argument("--dtypes", default=",".join(_DTYPES),
                   help=f"Comma-separated dtypes (default: {','.join(_DTYPES)})")
    p.add_argument("--layouts", default=",".join(_LAYOUTS),
                   help=f"Comma-separated layouts (default: {','.join(_LAYOUTS)})")
    p.add_argument("--budget", type=int, default=500,
                   help="Max configs to run; 0 = no cap (default: 500)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; default = daily rotating seed")
    p.add_argument("--size", type=int, default=1024,
                   help="M=N=K problem size (default: 1024)")
    p.add_argument("--json", default=None,
                   help="Write results JSON to this path")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
