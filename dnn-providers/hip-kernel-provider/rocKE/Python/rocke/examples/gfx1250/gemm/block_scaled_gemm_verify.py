# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1250 K=64 FP8/BF8 block-scaled WMMA GEMM.

Computes ``C[m, n] = sum_kg a_scale[m, kg] * b_scale[kg, n]
                       * sum_{k in group} A[m, k] * B[n, k]`` (RCR, ``C = A @ B^T``)
on a gfx1250 device and compares against a numpy fp32 reference. A/B carry
fp8e4m3 / bf8e5m2 mantissas (encoded via ml_dtypes); per-block f32 scales use
``group_size_k = block_k``.

A wrong K=64 fragment lane map produces a transposed / permuted result and
fails verify, so a PASS at multiple shapes confirms the mapping.

Must run on a gfx1250 device:

  PYTHONPATH=Python python3 -m rocke.examples.gfx1250.gemm.block_scaled_gemm_verify \
      --m 16 --n 16 --k 128 --dtype fp8e4m3
"""

from __future__ import annotations

import argparse
import ctypes
import struct

from rocke.helpers import compile_kernel
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    build_block_scaled_gemm,
)
from rocke.runtime.hip_module import Runtime


def _ml_fp8_dtype(np_mod, dtype: str):
    import ml_dtypes

    if dtype == "fp8e4m3":
        return ml_dtypes.float8_e4m3fn
    if dtype == "bf8e5m2":
        return ml_dtypes.float8_e5m2
    raise SystemExit(f"unsupported low-bit dtype {dtype!r}")


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1250")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--block-k", type=int, default=128)
    p.add_argument("--dtype", default="fp8e4m3", choices=("fp8e4m3", "bf8e5m2"))
    p.add_argument("--tol", type=float, default=2e-2)
    args = p.parse_args()

    import numpy as np

    for d, name, mult in (
        (args.m, "M", 16),
        (args.n, "N", 16),
        (args.k, "K", args.block_k),
    ):
        if d % mult:
            raise SystemExit(f"{name}={d} must be a multiple of {mult}")
    if args.block_k % 64:
        raise SystemExit("block_k must be a multiple of 64")

    M, N, K, BK = args.m, args.n, args.k, args.block_k
    groups = K // BK
    fp8 = _ml_fp8_dtype(np, args.dtype)
    rng = np.random.default_rng(0xB10C)

    # Small, mostly fp8-representable magnitudes so quantisation noise is tiny.
    A_f = (rng.integers(-4, 5, size=(M, K)).astype(np.float32)) * 0.5
    B_f = (rng.integers(-4, 5, size=(N, K)).astype(np.float32)) * 0.5
    A_q = A_f.astype(fp8)
    B_q = B_f.astype(fp8)
    A_dq = A_q.astype(np.float32)
    B_dq = B_q.astype(np.float32)

    A_scale = (rng.uniform(0.5, 1.5, size=(M, groups))).astype(np.float32)
    B_scale = (rng.uniform(0.5, 1.5, size=(groups, N))).astype(np.float32)

    # Reference: per-group scaled accumulation.
    ref = np.zeros((M, N), dtype=np.float32)
    for kg in range(groups):
        sl = slice(kg * BK, (kg + 1) * BK)
        partial = A_dq[:, sl] @ B_dq[:, sl].T  # (M, N)
        ref += partial * np.outer(A_scale[:, kg], B_scale[kg, :])

    spec = BlockScaledGemmSpec(
        name=f"verify_{args.arch}",
        M=M,
        N=N,
        K=K,
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c="bf16",
        scale_dtype="fp32",
        block_k=BK,
    )
    art = compile_kernel(build_block_scaled_gemm(spec, arch=args.arch), arch=args.arch)
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa})")

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    import ml_dtypes

    A_bytes = A_q.view(np.uint8)
    B_bytes = B_q.view(np.uint8)
    out = np.zeros((M, N), dtype=ml_dtypes.bfloat16)

    ad = rt.alloc(A_bytes.nbytes)
    bd = rt.alloc(B_bytes.nbytes)
    asd = rt.alloc(A_scale.nbytes)
    bsd = rt.alloc(B_scale.nbytes)
    cd = rt.alloc(out.nbytes)
    rt.memcpy_h2d(ad, u8(A_bytes), A_bytes.nbytes)
    rt.memcpy_h2d(bd, u8(B_bytes), B_bytes.nbytes)
    rt.memcpy_h2d(asd, u8(A_scale), A_scale.nbytes)
    rt.memcpy_h2d(bsd, u8(B_scale), B_scale.nbytes)
    rt.memset(cd, 0, out.nbytes)

    grid = block_scaled_gemm_grid(spec)
    block = (spec.block_size, 1, 1)
    packed = struct.pack("<QQQQQiii", ad, bd, asd, bsd, cd, M, N, K)
    rt.launch(fn, grid, block, packed)
    rt.sync()
    rt.memcpy_d2h(u8_out := (ctypes.c_uint8 * int(out.nbytes))(), cd, out.nbytes)
    out = np.frombuffer(bytes(u8_out), dtype=ml_dtypes.bfloat16).reshape(M, N)

    got = out.astype(np.float32)
    diff = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1.0)
    max_rel = float((diff / denom).max())
    max_abs = float(diff.max())
    for ptr in (ad, bd, asd, bsd, cd):
        rt.free(ptr)
    module.unload()

    ok = max_rel <= args.tol
    tag = "PASS (lane map confirmed)" if ok else "FAIL (lane map wrong)"
    print(
        f"[{args.arch}] block-scaled {args.dtype} {M}x{N}x{K} bk{BK} groups={groups}: "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} tol={args.tol:.0e} -> {tag}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
