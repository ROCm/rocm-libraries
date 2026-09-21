# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest-runner problem builder for MatMulNBits kernels."""

from __future__ import annotations

import os
import struct
from typing import Optional, Tuple

from ....runtime.hip_module import Runtime
from .._matmul_nbits_common import (
    MatMulNBitsSpec,
    matmul_nbits_reference,
    pack_i4_weights_for_matmul_nbits,
)
from ..gemm_universal import TileSpec
from .utils import as_u8_buffer, nbytes, require_numpy


def _spec_from_manifest(manifest: dict) -> MatMulNBitsSpec:
    return MatMulNBitsSpec(
        name=str(manifest.get("kernel_name", "manifest_matmul_nbits")),
        N=int(manifest["N"]),
        K=int(manifest["K"]),
        tile=TileSpec(
            tile_m=int(manifest["block_m"]),
            tile_n=int(manifest["block_n"]),
            tile_k=int(manifest["block_k"]),
            warp_m=int(manifest.get("warp_m", 2)),
            warp_n=int(manifest.get("warp_n", 2)),
            warp_k=int(manifest.get("warp_k", 1)),
            warp_tile_m=int(manifest.get("warp_tile_m", 16)),
            warp_tile_n=int(manifest.get("warp_tile_n", 16)),
            warp_tile_k=int(manifest.get("warp_tile_k", 16)),
        ),
        group_size=int(manifest.get("group_size", 32)),
        scale_dtype=str(manifest.get("scale_dtype", "fp16")),
        family=str(manifest.get("family", "large_n")),
    )


def run_matmul_nbits_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """fp16-activation / packed-int4-weight matmul (gfx1151 ``matmul_nbits``)."""
    np = require_numpy()
    spec = _spec_from_manifest(manifest)
    N = spec.N
    K = spec.K
    group = spec.group_size
    scale_dtype = spec.scale_dtype
    if shape is None:
        ds = manifest.get("default_shape", [128, N, K])
        M = int(ds[0])
    else:
        sm, sn, sk = shape
        if sn != N or sk != K:
            raise ValueError(
                f"matmul_nbits shape N/K ({sn},{sk}) != manifest ({N},{K})"
            )
        M = int(sm)
    if K % 2:
        raise ValueError(f"K ({K}) must be even to pack two int4 per byte")
    if K % group:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group})")

    np_scale = np.float16 if scale_dtype in ("f16", "fp16") else np.float32
    rng = np.random.default_rng(0x4B17)
    A = rng.integers(-4, 5, size=(M, K), dtype=np.int16).astype(np.float16)
    W = rng.integers(-8, 8, size=(N, K), dtype=np.int16)
    scales = (
        rng.integers(1, 5, size=(N, K // group)).astype(np.float32) * 0.03125
    ).astype(np_scale)
    packed = pack_i4_weights_for_matmul_nbits(W, spec)
    C = np.empty((M, N), dtype=np.float16)

    block_m = int(manifest["block_m"])
    if M % block_m:
        raise ValueError(
            f"M ({M}) must be divisible by block_m ({block_m}); partial-M tiles "
            "are not supported by the matmul_nbits kernels"
        )

    grid = (
        (N + int(manifest["block_n"]) - 1) // int(manifest["block_n"]),
        (M + block_m - 1) // block_m,
        1,
    )
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * M * N * K
    scale_bytes = np.dtype(np_scale).itemsize
    bytes_xfer = (
        2.0 * (M * K + M * N)
        + float(N * (K // 2))
        + scale_bytes * float(N * (K // group))
    )

    def make_args(rt: Runtime):
        A_dev = rt.alloc(nbytes(A))
        B_dev = rt.alloc(nbytes(packed))
        S_dev = rt.alloc(nbytes(scales))
        C_dev = rt.alloc(nbytes(C))
        rt.memcpy_h2d(A_dev, as_u8_buffer(A), nbytes(A))
        rt.memcpy_h2d(B_dev, as_u8_buffer(packed), nbytes(packed))
        rt.memcpy_h2d(S_dev, as_u8_buffer(scales), nbytes(scales))
        rt.memset(C_dev, 0, nbytes(C))
        return struct.pack("<QQQQi", A_dev, B_dev, S_dev, C_dev, M), (
            A_dev,
            B_dev,
            S_dev,
            C_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(as_u8_buffer(C), ptrs[3], nbytes(C))
        ref = matmul_nbits_reference(A, packed, scales, spec).astype(np.float16)
        Cf = C.astype(np.float32)
        reff = ref.astype(np.float32)
        tol = 1e-2
        err = np.abs(Cf - reff)
        bad = err > tol + tol * np.abs(reff)
        if os.environ.get("ROCKE_NBITS_DEBUG"):
            _nbits_debug_dump(np, Cf, reff, err, M, N, K, group)
        return float(err.max()), int(np.count_nonzero(bad)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def _nbits_debug_dump(np, Cf, reff, diff, M, N, K, group):
    """Locate matmul_nbits mismatches and inspect C/reference relationships."""
    tol = 1e-2
    bad = diff > tol
    nbad = int(bad.sum())
    print(
        f"[nbits-debug] shape M={M} N={N} K={K} group={group} "
        f"bad={nbad}/{Cf.size} max={float(diff.max()):.4g}"
    )
    if nbad == 0:
        return
    rows = np.where(bad.any(axis=1))[0]
    cols = np.where(bad.any(axis=0))[0]
    print(
        f"[nbits-debug] bad rows (M): {rows[:32].tolist()}"
        f"{' ...' if rows.size > 32 else ''} (count={rows.size})"
    )
    print(
        f"[nbits-debug] bad cols (N): {cols[:32].tolist()}"
        f"{' ...' if cols.size > 32 else ''} (count={cols.size})"
    )
    col_counts = bad.sum(axis=0)
    nz_cols = np.where(col_counts > 0)[0]
    print(
        "[nbits-debug] bad-count by N col (nonzero): "
        + ", ".join(f"n{c}:{int(col_counts[c])}" for c in nz_cols[:24])
    )
    bi, bj = np.where(bad)
    print("[nbits-debug] sample bad coords (m,n): C, ref, ratio")
    for k in range(min(12, bi.size)):
        m, n = int(bi[k]), int(bj[k])
        c, r = float(Cf[m, n]), float(reff[m, n])
        ratio = (c / r) if r != 0 else float("inf")
        print(f"  ({m:>4},{n:>4})  C={c:>10.4f}  ref={r:>10.4f}  C/ref={ratio:>8.4f}")
