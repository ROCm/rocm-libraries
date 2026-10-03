# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest-runner problem builders for GEMM-family kernels."""

from __future__ import annotations

import struct
from typing import Optional, Tuple

from ....runtime.hip_module import Runtime
from .utils import as_u8_buffer, nbytes, require_numpy


def _gemm_is_bf16(manifest: dict) -> bool:
    """Whether the GEMM operands are bf16, per the ``A`` ptr type.

    The manifest ``kind`` is ``gemm_fp16`` for every GEMM, so the element
    type is carried by ``args_signature`` (``ptr<bf16, global>`` vs
    ``ptr<f16, global>``, emitted by ``helpers.manifest.gemm_args_signature``).
    Both are 2 bytes wide, so only the interpretation differs.
    """
    sig = manifest.get("args_signature", [])
    ptr_type = next((a.get("type", "") for a in sig if a.get("name") == "A"), "")
    return "bf16" in ptr_type


def run_gemm_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    np = require_numpy()
    if shape is None:
        ds = manifest.get("default_shape", [3328, 4096, 4096])
        M, N, K = int(ds[0]), int(ds[1]), int(ds[2])
    else:
        M, N, K = shape
    is_bf16 = _gemm_is_bf16(manifest)
    rng = np.random.default_rng(0xC0FFEE)
    A_f32 = None  # float32 inputs for bf16 reference, set when is_bf16
    B_f32 = None
    if is_bf16:
        # Small integers (-5..5) are exactly representable in bf16; their fp32
        # lower 16 bits are zero so truncation == RNE for the inputs.
        A_f32 = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
        B_f32 = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)
        # Encode as bf16 raw bytes stored behind a float16 view for device transfer.
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
            from ....dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16

            # Reference: fp32 accumulation (exact for small-integer inputs),
            # then round-to-nearest-even to bf16 matching the kernel's fptrunc.
            ref_u16 = _bf16_from_f32(np, A_f32 @ B_f32.T)
            ref_f32 = _f32_from_bf16(np, ref_u16)
            # Decode raw output bytes as bf16, not fp16.
            out_f32 = _f32_from_bf16(np, C.view(np.uint16))
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


def _fp8e4m3_decode_table(np):
    """All 256 OCP e4m3 (``fn``) codes decoded to f32, built from the format.

    ``s.eeee.mmm``, bias 7. ``e == 0`` is subnormal — ``(-1)^s * 2^-6 * m/8``;
    otherwise ``(-1)^s * 2^(e-7) * (1 + m/8)``. The format carries no
    infinities, and ``e == 0xF, m == 0x7`` (``0x7f`` / ``0xff``) is NaN, so the
    largest finite magnitude is 448.

    Derived rather than tabulated on purpose: a hand-typed table is a second
    definition of the wire format that can drift from the hardware's.
    """
    code = np.arange(256, dtype=np.uint32)
    sign = np.where((code >> 7) & 1, np.float32(-1.0), np.float32(1.0))
    exp = ((code >> 3) & 0xF).astype(np.int32)
    man = (code & 0x7).astype(np.float32)
    sub = np.float32(2.0**-6) * (man / np.float32(8.0))
    nor = np.ldexp(np.float32(1.0) + man / np.float32(8.0), exp - 7)
    val = (sign * np.where(exp == 0, sub, nor)).astype(np.float32)
    val[(exp == 0xF) & ((code & 0x7) == 0x7)] = np.float32("nan")
    return val


def _fp8e4m3_encode(np, x):
    """Encode f32 values to OCP e4m3 bytes, rejecting anything inexact.

    This runner's verification argument rests on the device seeing exactly the
    numbers the reference multiplies, so a silent round here would quietly
    demote the gate to a tolerance check. Vectorised via the inverted decode
    table — a Python loop over a 4096x8192 operand is not viable.
    """
    table = _fp8e4m3_decode_table(np)
    keep = np.isfinite(table)
    vals = table[keep]
    codes = np.arange(256, dtype=np.uint8)[keep]
    order = np.argsort(vals, kind="stable")
    vals, codes = vals[order], codes[order]
    # -0.0 and +0.0 compare equal; np.unique folds them and return_index keeps
    # the first occurrence, which the stable sort left as +0.0 (code 0x00).
    uniq, first = np.unique(vals, return_index=True)
    uniq_codes = codes[first]
    xf = np.ascontiguousarray(x, dtype=np.float32).ravel()
    idx = np.minimum(np.searchsorted(uniq, xf), uniq.size - 1)
    if not np.array_equal(uniq[idx], xf):
        raise ValueError("fp8e4m3 encode: value not exactly representable in e4m3")
    return uniq_codes[idx].reshape(np.shape(x))


def run_gemm_fp8_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """fp8 WMMA GEMM (e4m3 A/B in, bf16 C out): ``C = A @ B.T``, exact.

    Inputs are integers in -5..5, every one of which is exact in e4m3 (four
    significand bits). With ``|sum| <= 25 * K`` the fp32 accumulator is exact
    for any K below ~670k, so summation order cannot change the result and the
    only rounding left is the kernel's final deterministic RNE to bf16 — which
    the reference reproduces. Hence an exact compare rather than a tolerance.
    """
    np = require_numpy()
    if shape is None:
        ds = manifest.get("default_shape", [4096, 4096, 8192])
        M, N, K = int(ds[0]), int(ds[1]), int(ds[2])
    else:
        M, N, K = shape
    if 25 * K >= (1 << 24):
        raise ValueError(
            f"gemm_fp8 verify: K={K} overflows the exact-fp32-accumulate "
            f"argument (25*K must stay below 2^24); lower K or widen the check"
        )
    rng = np.random.default_rng(0xC0FFEE)
    A_f32 = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
    B_f32 = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)
    A = _fp8e4m3_encode(np, A_f32)
    B = _fp8e4m3_encode(np, B_f32)
    # C is bf16; carry the raw halves as uint16 so no float16 view lies about
    # the encoding on the way to and from the device.
    C = np.empty((M, N), dtype=np.uint16)
    gx = (N + int(manifest["block_n"]) - 1) // int(manifest["block_n"])
    gy = (M + int(manifest["block_m"]) - 1) // int(manifest["block_m"])
    if manifest.get("grid_order") == "MN":
        gx, gy = gy, gx
    grid = (gx, gy, 1)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * M * N * K
    bytes_xfer = 1.0 * (M * K + N * K) + 2.0 * (M * N)

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
        from ....dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16

        ref_f32 = _f32_from_bf16(np, _bf16_from_f32(np, A_f32 @ B_f32.T))
        out_f32 = _f32_from_bf16(np, C)
        err = np.abs(out_f32 - ref_f32)
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
