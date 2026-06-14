# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest-runner problem builder for implicit-GEMM convolution."""

from __future__ import annotations

import struct
from typing import Optional, Tuple

from ....runtime.hip_module import Runtime
from .utils import as_u8_buffer, nbytes, require_numpy


def run_conv_manifest_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    np = require_numpy()
    cv = [int(x) for x in manifest["conv"]]
    if len(cv) < 13:
        raise ValueError("conv manifest needs [N,H,W,C,K,R,S,sH,sW,pH,pW,dH,dW]")
    N, H, W, C, K, R, S, sH, sW, pH, pW, dH, dW = cv[:13]
    groups = int(manifest.get("groups", 1))
    cpg = int(manifest.get("cpg", C // groups))
    kpg = int(manifest.get("kpg", K // groups))
    if groups * cpg != C or groups * kpg != K:
        raise ValueError(
            f"invalid grouping groups={groups} cpg={cpg} kpg={kpg} C={C} K={K}"
        )

    dtype = str(manifest.get("dtype", "fp16"))
    if dtype == "bf16":
        # numpy has no bfloat16; allocate as uint16 (same byte width) and let
        # the kernel interpret the bits as bf16. Reference check is skipped for bf16
        # because numpy can't compute a bf16 reference.
        np_dtype = np.uint16
    elif dtype == "fp32":
        np_dtype = np.float32
    else:
        np_dtype = np.float16

    rng = np.random.default_rng(1234)
    if dtype == "bf16":
        # Generate as float32, then take the upper 2 bytes of each f32 word as
        # bf16 bits (little-endian: bytes [2:4]). This produces valid bf16-encoded
        # values rather than integer bit patterns (.astype(uint16) would truncate
        # the float to an integer first, giving garbage — often zeros or NaNs).
        def _to_bf16(x):
            f32 = np.ascontiguousarray(x, dtype=np.float32)
            return f32.view(np.uint8).reshape(-1, 4)[:, 2:].copy().view(np.uint16).reshape(x.shape)
        A = _to_bf16(rng.random((N, H, W, C), dtype=np.float32) * 0.04 - 0.02)
        B = _to_bf16(rng.random((K, R, S, cpg), dtype=np.float32) * 0.04 - 0.02)
    else:
        A = (rng.random((N, H, W, C), dtype=np.float32) * 0.04 - 0.02).astype(np_dtype)
        B = (rng.random((K, R, S, cpg), dtype=np.float32) * 0.04 - 0.02).astype(np_dtype)
    Ho = (H + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (W + 2 * pW - dW * (S - 1) - 1) // sW + 1
    D = np.empty((N, Ho, Wo, K), dtype=np_dtype)

    if "grid_explicit" in manifest:
        gx, gy, gz = [int(x) for x in manifest["grid_explicit"]]
    else:
        bm = int(manifest["block_m"])
        bn = int(manifest["block_n"])
        M = N * H * W
        gx, gy, gz = (
            (K + bn - 1) // bn,
            (M + bm - 1) // bm,
            int(manifest.get("grid_z", 1)),
        )
        if manifest.get("grid_order") == "MN":
            gx, gy = gy, gx
    grid = (gx, gy, gz)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * N * H * W * K * R * S * cpg
    bytes_xfer = float(A.itemsize) * (A.size + B.size + D.size)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(nbytes(A))
        B_dev = rt.alloc(nbytes(B))
        D_dev = rt.alloc(nbytes(D))
        rt.memcpy_h2d(A_dev, as_u8_buffer(A), nbytes(A))
        rt.memcpy_h2d(B_dev, as_u8_buffer(B), nbytes(B))
        rt.memset(D_dev, 0, nbytes(D))
        if int(manifest.get("sig_has_bytes", 1)):
            args = struct.pack(
                "<QQQiii", A_dev, B_dev, D_dev, nbytes(A), nbytes(B), nbytes(D)
            )
        else:
            args = struct.pack("<QQQ", A_dev, B_dev, D_dev)
        return args, (A_dev, B_dev, D_dev)

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, D.size
        if dtype == "bf16":
            # numpy can't compute a bf16 reference; skip numerical check.
            return 0.0, 0, D.size
        rt.memcpy_d2h(as_u8_buffer(D), ptrs[2], nbytes(D))
        Ap = np.pad(A, ((0, 0), (pH, pH), (pW, pW), (0, 0)), mode="constant")
        ref = np.zeros_like(D, dtype=np.float32)
        for r in range(R):
            for s in range(S):
                row_start = r * dH
                col_start = s * dW
                x = Ap[
                    :,
                    row_start : row_start + Ho * sH : sH,
                    col_start : col_start + Wo * sW : sW,
                    :,
                ]
                for g in range(groups):
                    xs = x[..., g * cpg : (g + 1) * cpg].astype(np.float32)
                    ws = B[g * kpg : (g + 1) * kpg, r, s, :].astype(np.float32)
                    ref[..., g * kpg : (g + 1) * kpg] += np.einsum(
                        "nhwc,kc->nhwk", xs, ws, optimize=True
                    )
        # Cast reference back to the kernel's output dtype for a fair comparison.
        ref_out = ref.astype(np_dtype)
        tol = 1e-4 if dtype == "fp32" else 1e-2
        diff = np.abs(D.astype(np.float32) - ref_out.astype(np.float32))
        return float(diff.max()), int(np.count_nonzero(diff > tol)), D.size

    return make_args, grid, block, flop, bytes_xfer, check
