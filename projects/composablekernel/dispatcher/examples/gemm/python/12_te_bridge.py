#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 12: Tile Engine -> Dispatcher bridge

Unlike examples 01-11 (which drive the Dispatcher's native ctypes Registry),
this example exercises the *Tile Engine -> Dispatcher bridge* in
``dispatcher/python/gemm_utils.py``. The bridge is the path the Tile Engine
itself uses: one common ``GemmKernelConfig`` feeds codegen, force-include
compile, and a flat extern "C" ABI, and ``GpuGemmRunner`` runs the resulting
.so against a NumPy reference.

It demonstrates the generality the bridge gained over the original fp16/rcr-only
slice by running multiple (dtype, layout) pairs: fp16 across the four A/B
row/col layout combinations supported by universal GEMM (row-major C), plus a
couple of bf16 cases.

Usage:
    python3 12_te_bridge.py
    python3 12_te_bridge.py --size 1024
    python3 12_te_bridge.py --rtol 2e-2
    python3 12_te_bridge.py --arch gfx950
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np  # noqa: E402

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    GemmProblem,
    GpuGemmRunner,
    setup_multiple_gemm_dispatchers,
)
from ctypes_utils import detect_gpu_arch  # noqa: E402

# A single algorithm known to compile and run on gfx942. Only the Signature
# (dtype + layout) varies per case; the Algorithm is held fixed so the example
# isolates the bridge's dtype/layout generality.
_ALGO = dict(
    tile_m=64,
    tile_n=64,
    tile_k=64,
    wave_m=4,
    wave_n=1,
    wave_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
    pipeline="compv3",
    scheduler="intrawave",
    epilogue="cshuffle",
    pad_m=False,
    pad_n=False,
    pad_k=False,
)

# (dtype, layout) pairs. Column-major C (e.g. rcc) is rejected at build by the
# universal GEMM, so every case keeps row-major C -- which leaves exactly four
# A/B combinations (rcr/rrr/ccr/crr). Both dtypes cover all four so the matrix
# matches the docstring's claim.
_CASES = [
    ("fp16", "rcr"),
    ("fp16", "rrr"),
    ("fp16", "ccr"),
    ("fp16", "crr"),
    ("bf16", "rcr"),
    ("bf16", "rrr"),
    ("bf16", "ccr"),
    ("bf16", "crr"),
]

_LAYOUT_WORD = {"r": "row", "c": "col"}


def _emulate(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round fp32 inputs to the kernel's storage dtype so the CPU reference
    matches what the GPU actually multiplies."""
    if dtype == "bf16":
        u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        rounded = (u32 + ((u32 >> 16) & 1) + np.uint32(0x7FFF)) >> 16
        return (rounded.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float16).astype(np.float32)


def _config(dtype: str, layout: str, arch: str) -> GemmKernelConfig:
    la, lb, lc = layout
    return GemmKernelConfig(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=dtype,
        layout_a=_LAYOUT_WORD[la],
        layout_b=_LAYOUT_WORD[lb],
        layout_c=_LAYOUT_WORD[lc],
        gfx_arch=arch,
        **_ALGO,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tile Engine -> Dispatcher bridge example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=512, help="M=N=K (default 512)")
    parser.add_argument(
        "--rtol", type=float, default=2e-2, help="relative tolerance (default 2e-2)"
    )
    parser.add_argument(
        "--arch",
        default=detect_gpu_arch(),
        help="GPU target arch (default: auto-detected via rocminfo)",
    )
    args = parser.parse_args()

    problem = GemmProblem(M=args.size, N=args.size, K=args.size)
    configs = [_config(dt, lay, args.arch) for dt, lay in _CASES]

    print(f"Building {len(configs)} bridge kernels (codegen + hipcc)...")
    so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)

    np.random.seed(42)
    A = (np.random.randn(problem.M, problem.K) * 0.1).astype(np.float32)
    B = (np.random.randn(problem.K, problem.N) * 0.1).astype(np.float32)

    n_pass = 0
    for (dtype, layout), so in zip(_CASES, so_paths):
        tag = f"{dtype}/{layout}"
        if so is None:
            print(f"  {tag:10s} BUILD FAILED")
            continue
        runner = GpuGemmRunner(lib_path=so)
        result = runner.run(A, B, problem)
        if not result.success:
            print(f"  {tag:10s} RUN FAILED (status {result.status})")
            continue
        # Emulate both the input quantization (A,B stored as dtype) and the
        # output store: the GPU writes C back as dtype_c, so round the fp32
        # accumulator to dtype too before comparing.
        ref = _emulate(_emulate(A, dtype) @ _emulate(B, dtype), dtype)
        # Global relative error (normalize by the largest reference magnitude):
        # per-element ratios explode on the near-zero entries that K-length
        # accumulation of zero-mean data produces, so they are not meaningful.
        denom = float(np.max(np.abs(ref))) + 1e-12
        max_rel = float(np.max(np.abs(result.output - ref))) / denom
        ok = max_rel <= args.rtol
        n_pass += ok
        print(
            f"  {tag:10s} tflops={result.tflops:7.1f}  "
            f"max_rel={max_rel:.2e}  {'PASS' if ok else 'FAIL'}"
        )

    total = len(configs)
    print(f"\n{n_pass}/{total} passed")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
