# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import ctypes
import struct

import numpy as np

from ck_dsl.helpers.compile import compile_kernel
from ck_dsl.instances.common.vector_scale import (
    VectorScaleSpec, build_vectorscale, vectorscale_grid,
)
from ck_dsl.runtime.hip_module import Runtime

def _as_u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx942")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    spec = VectorScaleSpec(
        block_size=args.block_size,
    )
    art = compile_kernel(build_vectorscale(spec), arch=args.arch)
  

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    A = np.random.default_rng(0).standard_normal(args.n).astype(np.float32)
    A_dev = rt.alloc(A.nbytes)
    C_dev = rt.alloc(args.n * 4)
    rt.memcpy_h2d(A_dev, _as_u8(A), A.nbytes)
    rt.memset(C_dev, 0, args.n * 4)

    grid = vectorscale_grid(args.n, spec)
    block = (spec.block_size, 1, 1)
    pack = struct.pack("<QQfi", A_dev, C_dev, args.alpha, args.n)
    rt.launch(fn, grid, block, pack, stream=0)
    rt.stream_sync(0)

    out_buf = (ctypes.c_uint8 * (args.n * 4))()
    rt.memcpy_d2h(out_buf, C_dev, args.n * 4)
    C_out = (np.frombuffer(bytes(out_buf), dtype=np.float32))

    rt.free(A_dev)
    rt.free(C_dev)

    ref = args.alpha * A
    diff = np.abs(C_out - ref)
    max_abs = float(diff.max())
    bad = int((diff > args.tol).sum())
    ok = max_abs <= args.tol
    print(
        f"[{args.arch}] HIP-path elementwise-add N={args.n}: "
        f"max_abs_diff={max_abs:.3e} bad={bad}/{args.n} tol={args.tol:.0e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
