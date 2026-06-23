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

    is_3d = manifest.get("conv_layout") == "implicit_gemm_3d"

    cv = [int(x) for x in manifest["conv"]]
    if is_3d:
        if len(cv) < 18:
            raise ValueError(
                "3-D conv manifest needs [N,Di,Hi,Wi,C,K,Z,Y,X,sD,sH,sW,pD,pH,pW,dD,dH,dW]"
            )
        N, Di, Hi, Wi, C, K, Z, Y, X, sD, sH, sW, pD, pH, pW, dD, dH, dW = cv[:18]
    else:
        if len(cv) < 13:
            raise ValueError("conv manifest needs [N,Hi,Wi,C,K,Y,X,sH,sW,pH,pW,dH,dW]")
        N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, dH, dW = cv[:13]
        Di, Z, sD, pD, dD = 1, 1, 1, 0, 1

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

    rng = np.random.default_rng(1234)  # noqa: F841
    if is_3d:
        A = np.random.uniform(-5.0, 5.0, size=(N, Di, Hi, Wi, C)).astype(np.float32)
        B = np.random.uniform(-5.0, 5.0, size=(K, Z, Y, X, cpg)).astype(np.float32)
        if dtype == "bf16":
            A = (A.view(np.uint32) >> 16).view(np.float32)
            B = (B.view(np.uint32) >> 16).view(np.float32)
        else:
            A = A.astype(np_dtype)
            B = B.astype(np_dtype)

        Do = (Di + 2 * pD - dD * (Z - 1) - 1) // sD + 1
        Ho = (Hi + 2 * pH - dH * (Y - 1) - 1) // sH + 1
        Wo = (Wi + 2 * pW - dW * (X - 1) - 1) // sW + 1
        D = np.empty((N, Do, Ho, Wo, K), dtype=np_dtype)
    else:
        A = np.random.uniform(-5.0, 5.0, size=(N, Hi, Wi, C)).astype(np.float32)
        B = np.random.uniform(-5.0, 5.0, size=(K, Y, X, cpg)).astype(np.float32)
        if dtype == "bf16":
            A = (A.view(np.uint32) >> 16).view(np.float32)
            B = (B.view(np.uint32) >> 16).view(np.float32)
        else:
            A = A.astype(np_dtype)
            B = B.astype(np_dtype)

        Ho = (Hi + 2 * pH - dH * (Y - 1) - 1) // sH + 1
        Wo = (Wi + 2 * pW - dW * (X - 1) - 1) // sW + 1
        D = np.empty((N, Ho, Wo, K), dtype=np_dtype)

    if "grid_explicit" in manifest:
        gx, gy, gz = [int(x) for x in manifest["grid_explicit"]]
    else:
        bm = int(manifest["block_m"])
        bn = int(manifest["block_n"])
        M = N * Do * Ho * Wo if is_3d else N * Ho * Wo
        gx, gy, gz = (
            (K + bn - 1) // bn,
            (M + bm - 1) // bm,
            int(manifest.get("grid_z", 1)),
        )
        if manifest.get("grid_order") == "MN":
            gx, gy = gy, gx
    grid = (gx, gy, gz)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = (
        2.0 * N * Do * Ho * Wo * K * Z * Y * X * cpg
        if is_3d
        else 2.0 * N * Ho * Wo * K * Y * X * cpg
    )
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
        rt.memcpy_d2h(as_u8_buffer(D), ptrs[2], nbytes(D))
        ref = np.zeros_like(D, dtype=np.float32)
        if is_3d:
            Ap = np.pad(
                A, ((0, 0), (pD, pD), (pH, pH), (pW, pW), (0, 0)), mode="constant"
            )
            for z in range(Z):
                for y in range(Y):
                    for x in range(X):
                        d_start = z * dD
                        r_start = y * dH
                        c_start = x * dW
                        inp = Ap[
                            :,
                            d_start : d_start + Do * sD : sD,
                            r_start : r_start + Ho * sH : sH,
                            c_start : c_start + Wo * sW : sW,
                            :,
                        ]
                        for g in range(groups):
                            xs = inp[..., g * cpg : (g + 1) * cpg].astype(np.float32)
                            ws = B[g * kpg : (g + 1) * kpg, z, y, x, :].astype(
                                np.float32
                            )
                            ref[..., g * kpg : (g + 1) * kpg] += np.einsum(
                                "ndhwc,kc->ndhwk", xs, ws, optimize=True
                            )
        else:
            Ap = np.pad(A, ((0, 0), (pH, pH), (pW, pW), (0, 0)), mode="constant")
            for y in range(Y):
                for x in range(X):
                    row_start = y * dH
                    col_start = x * dW
                    inp = Ap[
                        :,
                        row_start : row_start + Ho * sH : sH,
                        col_start : col_start + Wo * sW : sW,
                        :,
                    ]
                    for g in range(groups):
                        xs = inp[..., g * cpg : (g + 1) * cpg].astype(np.float32)
                        ws = B[g * kpg : (g + 1) * kpg, y, x, :].astype(np.float32)
                        ref[..., g * kpg : (g + 1) * kpg] += np.einsum(
                            "nhwc,kc->nhwk", xs, ws, optimize=True
                        )
        # Cast reference back to the kernel's output dtype for a fair comparison.
        ref_out = ref.astype(np_dtype)
        tol = 1e-4 if dtype == "fp32" else 1e-2
        if dtype == "bf16":
            # D is uint16 (bf16 bits); zero-extend each element to uint32, then
            # shift into the float32 exponent/mantissa position.
            D_casted = D.astype(np.uint32) << 16
            D_casted = D_casted.view(np.float32)
        else:
            D_casted = D.astype(np.float32)
        err = np.abs(D_casted - ref_out)
        threshold = tol + tol * np.abs(ref_out)
        bad = err > threshold
        return float(err.max()), int(np.count_nonzero(bad)), D.size

    return make_args, grid, block, flop, bytes_xfer, check
