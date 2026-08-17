#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Dispatcher universal GEMM search space runner.

Enumerates the full GemmKernelConfig search space (mirroring tile engine's
daily-tier coverage), samples a budget-limited subset, compiles each config
via the bridge, runs on GPU, and reports correctness + TFLOPS.

Usage:
    python3 run_gemm_search_space.py --arch gfx942 --budget 500
    python3 run_gemm_search_space.py --dtypes fp16,bf16 --layouts rcr,rrr
    python3 run_gemm_search_space.py --budget 500 --seed 42 --json results.json
    python3 run_gemm_search_space.py --budget 0   # run full search space (no cap)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure dispatcher python and codegen are on path.
_DISPATCHER = Path(__file__).parent.parent
sys.path.insert(0, str(_DISPATCHER / "python"))
sys.path.insert(0, str(_DISPATCHER / "codegen"))

import numpy as np

from ctypes_utils import detect_gpu_arch
from gemm_utils import GemmProblem, GpuGemmRunner, setup_multiple_gemm_dispatchers
from gemm_search_space import (
    DTYPES,
    LAYOUTS,
    daily_seed,
    enumerate_configs,
    sample_configs,
)

_RTOL = 2e-2


def _emulate(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        rounded = (u32 + ((u32 >> 16) & 1) + np.uint32(0x7FFF)) >> 16
        return (rounded.astype(np.uint32) << 16).view(np.float32)
    if dtype in ("fp8", "bf8"):
        # Approximate: cast to fp16 precision for reference.
        return x.astype(np.float16).astype(np.float32)
    return x.astype(np.float16).astype(np.float32)


def _reference(A, B, dtype):
    return _emulate(_emulate(A, dtype) @ _emulate(B, dtype), dtype)


def _max_rel(out: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref))) + 1e-12
    return float(np.max(np.abs(out - ref))) / denom


def _np_dtype(dtype: str) -> np.dtype:
    if dtype in ("fp16", "bf16"):
        return np.float16
    if dtype in ("fp8", "bf8"):
        return np.uint8
    return np.float32


def run(args) -> int:
    arch = args.arch or detect_gpu_arch()
    dtypes = [d.strip() for d in args.dtypes.split(",")]
    layouts = [l.strip() for l in args.layouts.split(",")]
    budget = args.budget
    seed = args.seed if args.seed is not None else daily_seed()
    size = args.size

    print(f"arch={arch}  dtypes={dtypes}  layouts={layouts}")
    print(f"budget={budget if budget > 0 else 'unlimited'}  seed={seed}  size={size}")

    # --- Enumerate ---
    print("\nEnumerating search space...")
    t0 = time.time()
    strata = enumerate_configs(arch=arch, dtypes=dtypes, layouts=layouts)
    total_configs = sum(len(v) for v in strata.values())
    print(f"  {total_configs} valid configs across {len(strata)} strata "
          f"({time.time()-t0:.1f}s)")
    for key, cfgs in sorted(strata.items()):
        print(f"    {key}: {len(cfgs)}")

    # --- Sample ---
    if budget > 0 and budget < total_configs:
        configs = sample_configs(strata, budget=budget, seed=seed)
        print(f"\nSampled {len(configs)}/{total_configs} configs (budget={budget})")
    else:
        configs = [c for v in strata.values() for c in v]
        print(f"\nRunning all {len(configs)} configs (no budget cap)")

    if not configs:
        print("ERROR: no configs to run")
        return 1

    # --- Build ---
    print(f"\nBuilding {len(configs)} .so files (parallel JIT)...")
    t0 = time.time()
    so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)
    build_ok = sum(1 for p in so_paths if p is not None)
    build_fail = len(so_paths) - build_ok
    print(f"  Built {build_ok}/{len(configs)} in {time.time()-t0:.1f}s "
          f"({build_fail} failed)")

    # --- Run ---
    print(f"\nRunning {build_ok} kernels on GPU...")
    problem = GemmProblem(M=size, N=size, K=size)
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)

    results = []
    n_pass = n_fail = n_build_fail = 0

    for cfg, so in zip(configs, so_paths):
        dtype = cfg.dtype_a
        label = cfg.name

        if so is None:
            n_build_fail += 1
            results.append({"name": label, "status": "build_fail"})
            continue

        try:
            runner = GpuGemmRunner(lib_path=so)
            result = runner.run(A, B, problem)
        except Exception as exc:
            n_fail += 1
            results.append({"name": label, "status": "run_error", "error": str(exc)})
            continue

        if not result.success:
            n_fail += 1
            results.append({"name": label, "status": "run_fail",
                            "error": f"status={result.status}"})
            continue

        ref = _reference(A, B, dtype)
        mr = _max_rel(result.output, ref)
        ok = mr <= _RTOL
        n_pass += ok
        n_fail += not ok
        results.append({
            "name": label,
            "status": "pass" if ok else "fail",
            "tflops": round(result.tflops, 3),
            "max_rel": round(mr, 6),
        })

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} pass / {n_fail} fail / {n_build_fail} build-fail "
          f"/ {len(configs)} total")

    if args.json:
        out = {
            "arch": arch, "size": size, "seed": seed,
            "budget": budget, "total_configs": total_configs,
            "n_pass": n_pass, "n_fail": n_fail, "n_build_fail": n_build_fail,
            "kernels": results,
        }
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
    parser = argparse.ArgumentParser(
        description="Dispatcher universal GEMM search space runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default=None,
                        help="GPU architecture (default: auto-detect via rocminfo)")
    parser.add_argument("--dtypes", default=",".join(DTYPES),
                        help=f"Comma-separated dtypes (default: {','.join(DTYPES)})")
    parser.add_argument("--layouts", default=",".join(LAYOUTS),
                        help=f"Comma-separated layouts (default: {','.join(LAYOUTS)})")
    parser.add_argument("--budget", type=int, default=500,
                        help="Max configs to run; 0 = no cap (default: 500)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed; default = daily rotating seed")
    parser.add_argument("--size", type=int, default=1024,
                        help="M=N=K problem size (default: 1024)")
    parser.add_argument("--json", default=None,
                        help="Write results JSON to this path")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
