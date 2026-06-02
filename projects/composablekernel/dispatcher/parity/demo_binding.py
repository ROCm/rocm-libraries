#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
T2.2 demo — shows the full DispatcherLib workflow end-to-end.

Requires ``libdispatcher_gemm.so`` built and accessible.  Build it with::

    hipcc -fPIC -shared -o libdispatcher_gemm.so \\
          dispatcher_capi.cpp \\
          -I<ck_include_root> \\
          -include <output_dir>/dispatcher_wrappers/register_all_kernels.hpp

Usage::

    # Default: looks for ./libdispatcher_gemm.so; runs first kernel on 512×512×512.
    python demo_binding.py

    # Custom library path and problem size:
    python demo_binding.py --so /path/to/libdispatcher_gemm.so --mnk 1024 1024 1024

    # List kernels only, do not run:
    python demo_binding.py --list-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _tflops(M: int, N: int, K: int, elapsed_ms: float) -> float:
    return 2.0 * M * N * K / (elapsed_ms * 1e-3) / 1e12


def _cpu_gemm_fp16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CPU fp32 reference for verification: C = A @ B (fp32 accumulation)."""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)


def run_demo(so_path: str, M: int, N: int, K: int, list_only: bool = False) -> int:
    """
    Full T2.2 demo:
      1. Import DispatcherLib.
      2. List all registered kernels.
      3. Pick one that supports (M, N, K).
      4. Allocate numpy A (M×K), B (K×N), run GEMM → C.
      5. Verify C against CPU fp32 reference.
      6. Report TFLOP/s.

    Returns 0 on success, 1 on error.
    """
    # 1. Import and load
    try:
        from dispatcher_binding import DispatcherLib, DispatcherError
    except ImportError as exc:
        print(f"[DEMO] ERROR: cannot import dispatcher_binding: {exc}", file=sys.stderr)
        return 1

    try:
        lib = DispatcherLib(so_path)
    except FileNotFoundError as exc:
        print(f"[DEMO] ERROR: {exc}", file=sys.stderr)
        print("[DEMO] Build the shared library first (see module docstring).", file=sys.stderr)
        return 1

    print(f"[DEMO] Loaded {so_path}  version={lib.version()}")

    # 2. List kernels
    names = lib.kernel_names()
    count = lib.kernel_count()
    print(f"[DEMO] {count} kernel(s) registered:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")

    if list_only or count == 0:
        return 0

    # 3. Pick first kernel that supports (M, N, K)
    handle = None
    chosen_name = None
    for name in names:
        h = lib.find_kernel(name)
        if lib.supports(h, M, N, K):
            handle = h
            chosen_name = name
            break

    if handle is None:
        print(f"[DEMO] No kernel supports {M}×{N}×{K} — try a tile-aligned size.", file=sys.stderr)
        return 1

    print(f"[DEMO] Selected kernel: {chosen_name}")

    # 4. Allocate numpy arrays (host; GPU pointer forwarding requires hipMalloc).
    #    The binding accepts host arrays for structure demonstration.  On a real
    #    GPU node, allocate via hip.malloc / torch.cuda and pass device pointers.
    rng = np.random.default_rng(42)
    # Bounded init: A ∈ [−0.75, 0.75], B ∈ [−0.5, 0.5]
    a = ((rng.integers(0, 7, (M, K)) - 3) * 0.25).astype(np.float16)
    b = ((rng.integers(0, 5, (K, N)) - 2) * 0.25).astype(np.float16)

    print(f"[DEMO] A shape={a.shape} dtype={a.dtype}, B shape={b.shape} dtype={b.dtype}")

    # 5. Run GEMM
    try:
        c, elapsed_ms = lib.run_gemm(handle, a, b)
    except Exception as exc:
        print(f"[DEMO] run_gemm raised {type(exc).__name__}: {exc}", file=sys.stderr)
        print("[DEMO] (Expected on CPU-only hosts — GPU pointer access faults at kernel launch.)",
              file=sys.stderr)
        return 1

    tf = _tflops(M, N, K, elapsed_ms)
    print(f"[DEMO] GEMM {M}×{N}×{K}  elapsed={elapsed_ms:.3f} ms  TFLOP/s={tf:.2f}")

    # 6. Verify against CPU reference
    c_ref = _cpu_gemm_fp16(a, b)
    abs_err = np.abs(c.astype(np.float32) - c_ref.astype(np.float32))
    rel_err = abs_err / (np.abs(c_ref.astype(np.float32)) + 1e-6)
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    pct_pass = 100.0 * (abs_err < 1e-2).mean()

    print(f"[DEMO] max_abs_err={max_abs:.4f}  max_rel_err={max_rel:.4f}  "
          f"elements_within_1e-2={pct_pass:.1f}%")

    atol = 1e-3 * (K ** 0.5)
    rtol = 1e-2
    passed = bool((abs_err < atol + rtol * np.abs(c_ref.astype(np.float32))).all())
    if passed:
        print("[DEMO] PASSED — dispatcher output matches CPU fp32 reference.")
    else:
        print("[DEMO] FAILED — numerical mismatch exceeds tolerance.", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so", default="./libdispatcher_gemm.so",
                    help="Path to libdispatcher_gemm.so (default: ./libdispatcher_gemm.so)")
    ap.add_argument("--mnk", nargs=3, type=int, metavar=("M", "N", "K"),
                    default=[512, 512, 512],
                    help="Problem size M N K (default: 512 512 512)")
    ap.add_argument("--list-only", action="store_true",
                    help="List registered kernels without running GEMM")
    args = ap.parse_args()

    M, N, K = args.mnk
    return run_demo(args.so, M, N, K, list_only=args.list_only)


if __name__ == "__main__":
    raise SystemExit(main())
