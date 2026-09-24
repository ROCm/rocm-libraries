# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest-runner problem builders for GEMM-family kernels."""

from __future__ import annotations

import struct
from typing import Optional, Tuple

from ....runtime.hip_module import Runtime
from .utils import as_u8_buffer, nbytes, require_numpy


def _gemm_dtype_from_manifest(manifest: dict):
    """Return (np_storage_dtype, is_bf16) from args_signature ptr type."""
    import numpy as np

    sig = manifest.get("args_signature", [])
    ptr_type = next((a.get("type", "") for a in sig if a.get("name") == "A"), "")
    is_bf16 = "bf16" in ptr_type
    return np.float16, is_bf16


def run_gemm_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    np = require_numpy()
    if shape is None:
        ds = manifest.get("default_shape", [3328, 4096, 4096])
        M, N, K = int(ds[0]), int(ds[1]), int(ds[2])
    else:
        M, N, K = shape
    _, is_bf16 = _gemm_dtype_from_manifest(manifest)
    rng = np.random.default_rng(0xC0FFEE)
    A_f32 = None  # float32 inputs for bf16 reference, set when is_bf16
    B_f32 = None
    if is_bf16:
        # bf16 = upper 16 bits of fp32. For small integers (-5..5) the lower
        # 16 bits of fp32 are zero, so truncation is exact (no rounding).
        A_f32 = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
        B_f32 = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)
        # Encode as bf16 raw bytes and store behind a float16 view for transfer
        A = (A_f32.view(np.uint32) >> 16).astype(np.uint16).view(np.float16)
        B = (B_f32.view(np.uint32) >> 16).astype(np.uint16).view(np.float16)
    else:
        A = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float16)
        B = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float16)
    C = np.empty((M, N), dtype=np.float16)
    gx = (N + int(manifest["block_n"]) - 1) // int(manifest["block_n"])
    gy = (M + int(manifest["block_m"]) - 1) // int(manifest["block_m"])
    if manifest.get("grid_order") == "MN":
        gx, gy = gy, gx
    grid = (gx, gy, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * M * N * K
    bytes_xfer = 2.0 * (M * K + N * K + M * N)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(nbytes(A))
        B_dev = rt.alloc(nbytes(B))
        C_dev = rt.alloc(nbytes(C))
        rt.memcpy_h2d(A_dev, as_u8_buffer(A), nbytes(A))
        rt.memcpy_h2d(B_dev, as_u8_buffer(B), nbytes(B))
        rt.memset(C_dev, 0, nbytes(C))
        return struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K), (
            A_dev,
            B_dev,
            C_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(as_u8_buffer(C), ptrs[2], nbytes(C))
        if is_bf16:
            # fp32 accumulation is exact for small-integer inputs; cast result
            # to bf16 by taking upper 16 bits (same truncation the kernel does).
            ref_raw = (A_f32 @ B_f32.T).view(np.uint32) >> 16
            ref_f32 = ref_raw.astype(np.uint16).view(np.float16).astype(np.float32)
            out_f32 = C.view(np.uint16).view(np.float16).astype(np.float32)
        else:
            ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)
            ref_f32 = ref.astype(np.float32)
            out_f32 = C.astype(np.float32)
        tol = 1e-2
        err = np.abs(out_f32 - ref_f32)
        bad = err > tol + tol * np.abs(ref_f32)
        return float(err.max()), int(np.count_nonzero(bad)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def run_gemm_iu8_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Native integer WMMA GEMM (int8 in / i32 out): ``C = A @ B.T``, exact."""
    np = require_numpy()
    if shape is None:
        ds = manifest.get("default_shape", [256, 256, 256])
        M, N, K = int(ds[0]), int(ds[1]), int(ds[2])
    else:
        M, N, K = shape
    if K % 4:
        raise ValueError(f"iu8 GEMM needs K multiple of 4 (i32 packing), got K={K}")
    rng = np.random.default_rng(0xC0FFEE)
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(N, K), dtype=np.int8)
    A_p = np.ascontiguousarray(A).view(np.int32)
    B_p = np.ascontiguousarray(B).view(np.int32)
    C = np.empty((M, N), dtype=np.int32)
    gx = (N + int(manifest["block_n"]) - 1) // int(manifest["block_n"])
    gy = (M + int(manifest["block_m"]) - 1) // int(manifest["block_m"])
    if manifest.get("grid_order") == "MN":
        gx, gy = gy, gx
    grid = (gx, gy, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * M * N * K
    bytes_xfer = 1.0 * (M * K + N * K) + 4.0 * (M * N)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(nbytes(A_p))
        B_dev = rt.alloc(nbytes(B_p))
        C_dev = rt.alloc(nbytes(C))
        rt.memcpy_h2d(A_dev, as_u8_buffer(A_p), nbytes(A_p))
        rt.memcpy_h2d(B_dev, as_u8_buffer(B_p), nbytes(B_p))
        rt.memset(C_dev, 0, nbytes(C))
        return struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K), (
            A_dev,
            B_dev,
            C_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(as_u8_buffer(C), ptrs[2], nbytes(C))
        ref = A.astype(np.int32) @ B.astype(np.int32).T
        err = np.abs(C.astype(np.int64) - ref.astype(np.int64)).astype(np.float64)
        bad = err > 0.0
        return float(err.max()), int(np.count_nonzero(bad)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def run_batched_gemm_manifest_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Batched RCR GEMM: A[B,M,K] x Bmat[B,N,K] -> C[B,M,N]."""
    np = require_numpy()
    ds = manifest.get("default_shape", [8, 1024, 1024, 1024])
    if len(ds) != 4:
        raise ValueError("batched_gemm_fp16 default_shape must be [B, M, N, K]")
    BATCH, M, N, K = [int(x) for x in ds]
    rng = np.random.default_rng(0xBADC0DE)
    A = rng.integers(-5, 6, size=(BATCH, M, K), dtype=np.int16).astype(np.float16)
    Bm = rng.integers(-5, 6, size=(BATCH, N, K), dtype=np.int16).astype(np.float16)
    C = np.empty((BATCH, M, N), dtype=np.float16)
    grid = (
        (N + int(manifest["block_n"]) - 1) // int(manifest["block_n"]),
        (M + int(manifest["block_m"]) - 1) // int(manifest["block_m"]),
        BATCH,
    )
    block = (int(manifest["threads_per_block"]), 1, 1)
    stride_a = M * K
    stride_b = N * K
    stride_c = M * N
    flop = 2.0 * BATCH * M * N * K
    bytes_xfer = 2.0 * BATCH * (M * K + N * K + M * N)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(nbytes(A))
        B_dev = rt.alloc(nbytes(Bm))
        C_dev = rt.alloc(nbytes(C))
        rt.memcpy_h2d(A_dev, as_u8_buffer(A), nbytes(A))
        rt.memcpy_h2d(B_dev, as_u8_buffer(Bm), nbytes(Bm))
        rt.memset(C_dev, 0, nbytes(C))
        return struct.pack(
            "<QQQiiiiii",
            A_dev,
            B_dev,
            C_dev,
            M,
            N,
            K,
            stride_a,
            stride_b,
            stride_c,
        ), (A_dev, B_dev, C_dev)

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(as_u8_buffer(C), ptrs[2], nbytes(C))
        ref = np.empty_like(C)
        for bi in range(BATCH):
            ref[bi] = (A[bi].astype(np.float32) @ Bm[bi].astype(np.float32).T).astype(
                np.float16
            )
        ref_f32 = ref.astype(np.float32)
        tol = 1e-2
        err = np.abs(C.astype(np.float32) - ref_f32)
        bad = err > tol + tol * np.abs(ref_f32)
        return float(err.max()), int(np.count_nonzero(bad)), C.size

    return make_args, grid, block, flop, bytes_xfer, check
