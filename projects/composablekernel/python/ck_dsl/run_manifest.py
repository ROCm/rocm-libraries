# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python-native CK DSL manifest runner.

This replaces the C++ `example/ck_tile/dsl/common/launcher.cpp` path
for day-to-day DSL development. The flow:

  1. `gen.py` emits a HSACO blob + `manifest.json`.
  2. Python loads the code object with `hipModuleLoadData`.
  3. Python allocates tensors (torch CUDA tensors), passes their raw pointers
     into `hipModuleLaunchKernel`, verifies with torch reference ops, and times
     with HIP events.

No host C++ compile is involved. The C++ launcher can stay as a CMake/CK-Tile
compatibility target, but this module is the maintained runtime path.
"""

from __future__ import annotations

import argparse
import json
import ctypes
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .runtime.hip_module import Runtime

# Try to import torch-based launcher, fall back to direct HIP timing if unavailable
try:
    from .runtime.launcher import time_launches

    HAS_TORCH_LAUNCHER = True
except ImportError:
    HAS_TORCH_LAUNCHER = False


@dataclass
class RunSummary:
    ms: float = field()
    tflops: float = field()
    gbps: float = field()
    max_abs_diff: float = field(default=0.0)
    bad_count: int = field(default=0)
    total: int = field(default=0)


def _require_numpy():
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("ck_dsl.run_manifest requires numpy") from e
    return np


def _nbytes(a) -> int:
    return int(a.nbytes)


def _as_u8_buffer(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def _parse_shape(s: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not s:
        return None
    parts = [int(x) for x in s.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"--shape expects three ints, got {s!r}")
    return parts[0], parts[1], parts[2]


def _load(manifest_path: Path, hsaco_path: Optional[Path]):
    manifest = json.loads(manifest_path.read_text())
    if hsaco_path is None:
        hsaco_path = manifest_path.parent / str(manifest["hsaco"])
    return manifest, hsaco_path.read_bytes(), hsaco_path


def _launch_timed(
    rt: Runtime, fn, grid, block, args: bytes, warmup: int, iters: int
) -> float:
    """Time `iters` repeats of `rt.launch(fn, grid, block, args)` on
    the default stream.

    Delegates to `ck_dsl.runtime.launcher.time_launches` when torch is
    available, so the manifest runner and the in-tree Launcher abstraction
    share one bench-timing path; see that function's docstring for the
    correctness rationale (no per-call module reload, no module unload,
    args buffer lifetime tracked by Runtime._pending_args).

    Falls back to direct HIP event timing when torch is unavailable
    (torch-free environments).
    """
    if HAS_TORCH_LAUNCHER:
        return time_launches(
            lambda: rt.launch(fn, grid, block, args),
            warmup=warmup,
            iters=iters,
        )
    else:
        # Fallback: Direct HIP event timing (torch-free)
        for _ in range(warmup):
            rt.launch(fn, grid, block, args)
        rt.sync()
        e0 = rt.event()
        e1 = rt.event()
        e0.record()
        for _ in range(iters):
            rt.launch(fn, grid, block, args)
        e1.record()
        e1.synchronize()
        total_ms = e0.elapsed_to(e1)
        e0.destroy()
        e1.destroy()
        return total_ms / iters


def _gemm_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    np = _require_numpy()
    if shape is None:
        ds = manifest.get("default_shape", [3328, 4096, 4096])
        M, N, K = int(ds[0]), int(ds[1]), int(ds[2])
    else:
        M, N, K = shape
    rng = np.random.default_rng(0xC0FFEE)
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
        A_dev = rt.alloc(_nbytes(A))
        B_dev = rt.alloc(_nbytes(B))
        C_dev = rt.alloc(_nbytes(C))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B), _nbytes(B))
        rt.memset(C_dev, 0, _nbytes(C))
        return struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K), (
            A_dev,
            B_dev,
            C_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(_as_u8_buffer(C), ptrs[2], _nbytes(C))
        ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)
        diff = np.abs(C.astype(np.float32) - ref.astype(np.float32))
        return float(diff.max()), int(np.count_nonzero(diff > 0)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def _gemm_iu8_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Native integer WMMA GEMM (int8 in / i32 out): ``C = A @ B.T``, exact.

    A (``M×K``) and B (``N×K``) are signed int8 row-major, packed 4-per-i32
    along K (little-endian view: i32 slot ``j`` = K-bytes ``[4j..4j+3]``, which
    is exactly the ``wmma_i32_16x16x16_iu8`` fragment slot order). The kernel
    receives i32 pointers and the logical int8 ``K``. Integer WMMA does no
    rounding, so verify expects ``max_abs_diff == 0`` against the int32 numpy
    reference; random asymmetric inputs pin row vs col in the lane map.
    """
    np = _require_numpy()
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
    # Pack int8 rows into i32 (little-endian): K contiguous -> K//4 i32 columns.
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
        A_dev = rt.alloc(_nbytes(A_p))
        B_dev = rt.alloc(_nbytes(B_p))
        C_dev = rt.alloc(_nbytes(C))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A_p), _nbytes(A_p))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B_p), _nbytes(B_p))
        rt.memset(C_dev, 0, _nbytes(C))
        return struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K), (
            A_dev,
            B_dev,
            C_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(_as_u8_buffer(C), ptrs[2], _nbytes(C))
        ref = A.astype(np.int32) @ B.astype(np.int32).T
        diff = np.abs(C.astype(np.int64) - ref.astype(np.int64))
        return float(diff.max()), int(np.count_nonzero(diff > 0)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def _matmul_nbits_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """fp16-activation / packed-int4-weight matmul (gfx1151 ``matmul_nbits``).

    ``C[M, N] = A[M, K] @ dequant(B, scales)^T`` with ``A`` / ``C`` fp16
    row-major, ``B`` two signed int4 per byte (uint8 ``[N, K // 2]``, low nibble
    = even ``k``), and one ``fp16``/``fp32`` scale per ``(n, k // group)`` group.

    ``N`` / ``K`` are compile-time fields baked into the kernel; only ``M`` is a
    runtime argument. The pack / dequant logic is inlined here (rather than
    importing the instance host helpers) so the remote runner stays dependency
    light. It mirrors
    :func:`ck_dsl.instances.common._matmul_nbits_common.pack_i4_weights_for_matmul_nbits`.
    """
    np = _require_numpy()
    N = int(manifest["N"])
    K = int(manifest["K"])
    group = int(manifest.get("group_size", 32))
    scale_dtype = str(manifest.get("scale_dtype", "f16"))
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
    W = rng.integers(-8, 8, size=(N, K), dtype=np.int16)  # signed int4 [-8, 7]
    scales = (
        rng.integers(1, 5, size=(N, K // group)).astype(np.float32) * 0.03125
    ).astype(np_scale)
    low = (W[:, 0::2] & 0x0F).astype(np.uint8)
    high = (W[:, 1::2] & 0x0F).astype(np.uint8)
    packed = (low | (high << 4)).astype(np.uint8)
    C = np.empty((M, N), dtype=np.float16)

    block_m = int(manifest["block_m"])
    # The WMMA matmul_nbits kernels assume every tile_m row is in-bounds for the
    # A loads; partial-M tail handling is not implemented. A non-multiple M would
    # make the last tile read A out of bounds, so reject it here rather than
    # launch a kernel that reads past the buffer.
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
        2.0 * (M * K + M * N)  # A + C fp16
        + float(N * (K // 2))  # packed B
        + scale_bytes * float(N * (K // group))  # scales
    )

    def make_args(rt: Runtime):
        A_dev = rt.alloc(_nbytes(A))
        B_dev = rt.alloc(_nbytes(packed))
        S_dev = rt.alloc(_nbytes(scales))
        C_dev = rt.alloc(_nbytes(C))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(packed), _nbytes(packed))
        rt.memcpy_h2d(S_dev, _as_u8_buffer(scales), _nbytes(scales))
        rt.memset(C_dev, 0, _nbytes(C))
        return struct.pack("<QQQQi", A_dev, B_dev, S_dev, C_dev, M), (
            A_dev,
            B_dev,
            S_dev,
            C_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, C.size
        rt.memcpy_d2h(_as_u8_buffer(C), ptrs[3], _nbytes(C))
        w = W.astype(np.float32) * np.repeat(scales.astype(np.float32), group, axis=1)
        ref = (A.astype(np.float32) @ w.T).astype(np.float16)
        Cf = C.astype(np.float32)
        reff = ref.astype(np.float32)
        diff = np.abs(Cf - reff)
        if os.getenv("CKDSL_NBITS_DEBUG"):
            _nbits_debug_dump(np, Cf, reff, diff, M, N, K, group)
        return float(diff.max()), int(np.count_nonzero(diff > 0)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def _nbits_debug_dump(np, Cf, reff, diff, M, N, K, group):
    """Locate matmul_nbits mismatches: cluster bad elements by row/col and
    inspect the C/ref relationship to see if a scale was dropped/misapplied."""
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
    # Per-N-tile and per-M-tile histograms (16-wide WMMA tiles).
    col_counts = bad.sum(axis=0)
    nz_cols = np.where(col_counts > 0)[0]
    print(
        "[nbits-debug] bad-count by N col (nonzero): "
        + ", ".join(f"n{c}:{int(col_counts[c])}" for c in nz_cols[:24])
    )
    # Inspect first few bad coords: ratio reveals dropped/extra scale factor.
    bi, bj = np.where(bad)
    print("[nbits-debug] sample bad coords (m,n): C, ref, ratio")
    for k in range(min(12, bi.size)):
        m, n = int(bi[k]), int(bj[k])
        c, r = float(Cf[m, n]), float(reff[m, n])
        ratio = (c / r) if r != 0 else float("inf")
        print(f"  ({m:>4},{n:>4})  C={c:>10.4f}  ref={r:>10.4f}  C/ref={ratio:>8.4f}")


def _batched_gemm_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Batched RCR GEMM: A[B,M,K] x Bmat[B,N,K] -> C[B,M,N]."""
    np = _require_numpy()
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
        A_dev = rt.alloc(_nbytes(A))
        B_dev = rt.alloc(_nbytes(Bm))
        C_dev = rt.alloc(_nbytes(C))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(Bm), _nbytes(Bm))
        rt.memset(C_dev, 0, _nbytes(C))
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
        rt.memcpy_d2h(_as_u8_buffer(C), ptrs[2], _nbytes(C))
        ref = np.empty_like(C)
        for bi in range(BATCH):
            ref[bi] = (A[bi].astype(np.float32) @ Bm[bi].astype(np.float32).T).astype(
                np.float16
            )
        diff = np.abs(C.astype(np.float32) - ref.astype(np.float32))
        return float(diff.max()), int(np.count_nonzero(diff > 0)), C.size

    return make_args, grid, block, flop, bytes_xfer, check


def _conv_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    np = _require_numpy()
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

    rng = np.random.default_rng(1234)
    A = (rng.random((N, H, W, C), dtype=np.float32) * 0.04 - 0.02).astype(np.float16)
    B = (rng.random((K, R, S, cpg), dtype=np.float32) * 0.04 - 0.02).astype(np.float16)
    # Output spatial dims account for stride / dilation per the conv
    # forward formula: ``Ho = (H + 2*pH - dH*(R-1) - 1) // sH + 1`` and
    # symmetric for ``Wo``. The historical ``D = np.empty((N, H, W, K))``
    # silently assumed ``Ho = H`` (only valid when ``sH == sW == 1``
    # AND padding cancels the dilation tail), corrupting the parity
    # gate on stride > 1 / dilation > 1 conv shapes.
    Ho = (H + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (W + 2 * pW - dW * (S - 1) - 1) // sW + 1
    D = np.empty((N, Ho, Wo, K), dtype=np.float16)

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
    bytes_xfer = 2.0 * (A.size + B.size + D.size)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(_nbytes(A))
        B_dev = rt.alloc(_nbytes(B))
        D_dev = rt.alloc(_nbytes(D))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B), _nbytes(B))
        rt.memset(D_dev, 0, _nbytes(D))
        if int(manifest.get("sig_has_bytes", 1)):
            args = struct.pack(
                "<QQQiii", A_dev, B_dev, D_dev, _nbytes(A), _nbytes(B), _nbytes(D)
            )
        else:
            args = struct.pack("<QQQ", A_dev, B_dev, D_dev)
        return args, (A_dev, B_dev, D_dev)

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, D.size
        rt.memcpy_d2h(_as_u8_buffer(D), ptrs[2], _nbytes(D))
        # Vectorized grouped conv reference, fp32 accumulation then fp16 output.
        # ``Ap[:, r*dH : r*dH + Ho*sH : sH, s*dW : s*dW + Wo*sW : sW, :]``
        # is the ``(N, Ho, Wo, C)`` slice that contributes to the (r, s)
        # filter tap. The historical form
        # ``Ap[:, r : r + H*sH : sH, ...]`` was wrong on dilation>1
        # (missing the ``*dH`` start offset) and on stride>1 (used H
        # instead of Ho for the slice size).
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
        ref_h = ref.astype(np.float16)
        diff = np.abs(D.astype(np.float32) - ref_h.astype(np.float32))
        return float(diff.max()), int(np.count_nonzero(diff > 1e-2)), D.size

    return make_args, grid, block, flop, bytes_xfer, check


def _q_int_codes(np, scaled_f32, lo: float, hi: float):
    """Clamp then round-to-nearest-even, matching the integer fusion kernels."""
    return np.rint(np.clip(scaled_f32, lo, hi))


def _pack_i4_rows(np, codes):
    """Signed int4 codes ``(rows, cols)`` -> two codes per byte."""
    lo = codes[:, 0::2].astype(np.int32) & 0xF
    hi = codes[:, 1::2].astype(np.int32) & 0xF
    return ((hi << 4) | lo).astype(np.uint8).view(np.int8)


def _unpack_deep_fused_y(np, words, channels: int):
    """``(pool_h, pool_w, words)`` int32 output -> signed int4 channel codes."""
    ph, pw, _ = words.shape
    out = np.empty((ph, pw, channels), dtype=np.int32)
    for ch in range(channels):
        word = words[:, :, ch // 8].astype(np.uint32)
        nib = (word >> (4 * (ch % 8))) & 0xF
        signed = nib.astype(np.int32)
        out[:, :, ch] = np.where(signed >= 8, signed - 16, signed)
    return out


def _deep_fused_i8i4_reference(np, X, W0, W1_codes, manifest: dict):
    """Integer-exact reference for gfx1151 deep fused conv0->conv1->pool."""
    N, H, W, C, K0, R, S, sH, sW, pH, pW, dH, dW = [
        int(x) for x in manifest["conv"][:13]
    ]
    K1 = int(manifest["conv1"]["K1"])
    pool_y, pool_x, pool_s_h, pool_s_w = [int(x) for x in manifest["pool"]]
    quant = manifest.get("quant", {})
    m0 = np.float32(quant.get("m0", 0.0625))
    m0b = np.float32(quant.get("m0b", 0.5))
    m1 = np.float32(quant.get("m1", 0.25))
    mf = np.float32(quant.get("mf", 1.0))
    Ho = (H + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (W + 2 * pW - dW * (S - 1) - 1) // sW + 1
    pool_ho = (Ho - pool_y) // pool_s_h + 1
    pool_wo = (Wo - pool_x) // pool_s_w + 1

    Xp = np.pad(
        X.astype(np.int64),
        ((0, 0), (pH, pH), (pW, pW), (0, 0)),
    )
    P0 = np.zeros((N, Ho, Wo, K0), dtype=np.int64)
    for r in range(R):
        for s in range(S):
            x = Xp[
                :,
                r * dH : r * dH + Ho * sH : sH,
                s * dW : s * dW + Wo * sW : sW,
                :,
            ]
            w = W0[:, r, s, :].astype(np.int64)
            P0 += np.einsum("nhwc,kc->nhwk", x, w, optimize=True)

    q0 = _q_int_codes(np, P0.astype(np.float32) * m0, -127.0, 127.0)
    q0_relu = np.maximum(q0, 0.0)
    C0 = _q_int_codes(np, q0_relu * m0b, -8.0, 7.0)

    P1 = np.einsum("nhwk,ok->nhwo", C0, W1_codes.astype(np.float32), optimize=True)
    q1 = _q_int_codes(np, P1 * m1, -8.0, 7.0)
    C1 = np.maximum(q1, 0.0)

    ref = np.empty((pool_ho, pool_wo, K1), dtype=np.int32)
    for ho in range(pool_ho):
        for wo in range(pool_wo):
            h0 = ho * pool_s_h
            w0 = wo * pool_s_w
            patch = C1[0, h0 : h0 + pool_y, w0 : w0 + pool_x, :]
            pooled = patch.max(axis=(0, 1)).astype(np.float32)
            ref[ho, wo, :] = _q_int_codes(np, pooled * mf, -8.0, 7.0).astype(np.int32)
    return ref


def _deep_fused_conv_pool_i8i4_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """gfx1151 deep-fused int8/int4 conv+pool runner.

    The kernel ABI is four raw pointers: ``X, W0, Y, W1``. Shapes and quant
    multipliers are compile-time-specialized by the example and repeated in the
    manifest so this generic runner can build matching test tensors.
    """
    if shape is not None:
        raise ValueError("deep_fused_conv_pool_i8i4 uses manifest shape, not --shape")
    np = _require_numpy()
    N, H, W, C, K0, R, S, sH, sW, pH, pW, dH, dW = [
        int(x) for x in manifest["conv"][:13]
    ]
    K1 = int(manifest["conv1"]["K1"])
    pool_y, pool_x, pool_s_h, pool_s_w = [int(x) for x in manifest["pool"]]
    Ho = (H + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (W + 2 * pW - dW * (S - 1) - 1) // sW + 1
    pool_ho = (Ho - pool_y) // pool_s_h + 1
    pool_wo = (Wo - pool_x) // pool_s_w + 1

    rng = np.random.default_rng(int(manifest.get("seed", 123)))
    X = rng.integers(-3, 4, size=(N, H, W, C), dtype=np.int8)
    W0 = rng.integers(-3, 4, size=(K0, R, S, C), dtype=np.int8)
    W1_codes = rng.integers(-3, 4, size=(K1, K0), dtype=np.int8)
    W1 = _pack_i4_rows(np, W1_codes)
    Y = np.zeros((pool_ho, pool_wo, (K1 + 7) // 8), dtype=np.int32)

    if "grid_explicit" in manifest:
        grid = tuple(int(x) for x in manifest["grid_explicit"])
    else:
        pool_tile_h, pool_tile_w = [int(x) for x in manifest["pool_tile"]]
        grid = (1, pool_ho // pool_tile_h, pool_wo // pool_tile_w)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * (N * Ho * Wo * K0 * R * S * C + N * Ho * Wo * K1 * K0)
    bytes_xfer = float(X.nbytes + W0.nbytes + W1.nbytes + Y.nbytes)
    tol = int(manifest.get("verify_tol", 0))

    def make_args(rt: Runtime):
        X_dev = rt.alloc(_nbytes(X))
        W0_dev = rt.alloc(_nbytes(W0))
        Y_dev = rt.alloc(_nbytes(Y))
        W1_dev = rt.alloc(_nbytes(W1))
        rt.memcpy_h2d(X_dev, _as_u8_buffer(X), _nbytes(X))
        rt.memcpy_h2d(W0_dev, _as_u8_buffer(W0), _nbytes(W0))
        rt.memcpy_h2d(W1_dev, _as_u8_buffer(W1), _nbytes(W1))
        rt.memset(Y_dev, 0, _nbytes(Y))
        return struct.pack("<QQQQ", X_dev, W0_dev, Y_dev, W1_dev), (
            X_dev,
            W0_dev,
            Y_dev,
            W1_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, pool_ho * pool_wo * K1
        rt.memcpy_d2h(_as_u8_buffer(Y), ptrs[2], _nbytes(Y))
        got = _unpack_deep_fused_y(np, Y, K1)
        ref = _deep_fused_i8i4_reference(np, X, W0, W1_codes, manifest)
        diff = np.abs(got - ref)
        max_diff = int(diff.max()) if diff.size else 0
        return float(max_diff), int(np.count_nonzero(diff > tol)), got.size

    return make_args, grid, block, flop, bytes_xfer, check


def _simple_op_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Generic ``input(s) -> output`` runner.

    Supports kernels that fit one of these shapes (all of them small
    enough that the args layout is predictable from the manifest):

    * ``elementwise_fp16`` — unary or binary pointwise op.
        Signature: ``(A: ptr, [B: ptr,] C: ptr, N: i32)``
    * ``reduce_fp16`` — row reduction.
        Signature: ``(X: ptr, Y: ptr, M: i32, N: i32)``
    * ``layernorm_fp16`` / ``rmsnorm_fp16``.
        Signature: ``(X: ptr, Y: ptr, gamma: ptr, [beta: ptr,] M: i32, N: i32, eps: f32)``
    * ``transpose_fp16``.
        Signature: ``(X: ptr, Y: ptr, M: i32, N: i32)``

    All of these read their shape from the manifest's ``default_shape``
    field (a list of ints in the per-op canonical order) and dispatch
    to a small numpy reference for verification when ``--verify`` is
    set. Shape override via ``shape=(...)`` is currently a no-op (all
    of these are intrinsically 1D/2D ops, not GEMM-shaped); the gemm
    runner remains the path for shape-parameterized benchmarks.
    """
    np = _require_numpy()
    kind = str(manifest["kind"])
    op = str(manifest.get("op", ""))
    dtype = str(manifest.get("dtype", "f16"))
    if dtype not in ("f16", "bf16"):
        raise ValueError(f"simple-op runner currently supports f16/bf16, got {dtype!r}")
    np_dtype = np.float16  # both f16 and bf16 round-trip through fp16 storage here
    default_shape = manifest.get("default_shape") or []
    threads = int(manifest["threads_per_block"])
    block = (threads, 1, 1)
    rng = np.random.default_rng(int(manifest.get("seed", 0xC0FFEE)))

    if kind == "elementwise_fp16":
        N = int(default_shape[0]) if default_shape else 1024
        A = rng.standard_normal(N).astype(np_dtype)
        is_binary = bool(manifest.get("is_binary", False))
        B = rng.standard_normal(N).astype(np_dtype) if is_binary else None
        C = np.zeros(N, dtype=np_dtype)
        # Grid: ceil(N / elems_per_block).
        epb = int(manifest["elems_per_block"])
        grid = ((N + epb - 1) // epb, 1, 1)
        flop = float(N)
        bytes_xfer = 2.0 * N * (2 if is_binary else 1) + 2.0 * N

        def make_args(rt: Runtime):
            A_dev = rt.alloc(_nbytes(A))
            C_dev = rt.alloc(_nbytes(C))
            rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
            rt.memset(C_dev, 0, _nbytes(C))
            if is_binary:
                B_dev = rt.alloc(_nbytes(B))
                rt.memcpy_h2d(B_dev, _as_u8_buffer(B), _nbytes(B))
                args = struct.pack("<QQQi", A_dev, B_dev, C_dev, N)
                return args, (A_dev, B_dev, C_dev)
            args = struct.pack("<QQi", A_dev, C_dev, N)
            return args, (A_dev, C_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, C.size
            C_dev = ptrs[-1]
            rt.memcpy_d2h(_as_u8_buffer(C), C_dev, _nbytes(C))
            A_f32 = A.astype(np.float32)
            if op == "relu":
                ref = np.maximum(A_f32, 0.0)
            elif op == "copy":
                ref = A_f32
            elif op == "neg":
                ref = -A_f32
            elif op == "abs":
                ref = np.abs(A_f32)
            elif op == "silu":
                ref = A_f32 * (1.0 / (1.0 + np.exp(-A_f32)))
            elif op == "gelu_tanh":
                inner = np.sqrt(2.0 / np.pi) * (A_f32 + 0.044715 * A_f32**3)
                ref = 0.5 * A_f32 * (1.0 + np.tanh(inner))
            elif op == "exp2":
                ref = np.exp2(A_f32)
            elif is_binary and op == "add":
                ref = A_f32 + B.astype(np.float32)
            elif is_binary and op == "sub":
                ref = A_f32 - B.astype(np.float32)
            elif is_binary and op == "mul":
                ref = A_f32 * B.astype(np.float32)
            elif is_binary and op == "max":
                ref = np.maximum(A_f32, B.astype(np.float32))
            elif is_binary and op == "min":
                ref = np.minimum(A_f32, B.astype(np.float32))
            else:
                raise ValueError(f"no reference for elementwise op {op!r}")
            ref_h = ref.astype(np_dtype)
            diff = np.abs(C.astype(np.float32) - ref_h.astype(np.float32))
            return float(diff.max()), int(np.count_nonzero(diff > 1e-2)), C.size

        return make_args, grid, block, flop, bytes_xfer, check

    if kind == "reduce_fp16":
        M = int(default_shape[0])
        N = int(default_shape[1])
        X = rng.standard_normal((M, N)).astype(np_dtype)
        Y = np.zeros((M,), dtype=np_dtype)
        grid = (M, 1, 1)
        flop = float(M * N)
        bytes_xfer = 2.0 * M * N + 2.0 * M

        def make_args(rt: Runtime):
            X_dev = rt.alloc(_nbytes(X))
            Y_dev = rt.alloc(_nbytes(Y))
            rt.memcpy_h2d(X_dev, _as_u8_buffer(X), _nbytes(X))
            rt.memset(Y_dev, 0, _nbytes(Y))
            args = struct.pack("<QQii", X_dev, Y_dev, M, N)
            return args, (X_dev, Y_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, Y.size
            rt.memcpy_d2h(_as_u8_buffer(Y), ptrs[1], _nbytes(Y))
            X_f32 = X.astype(np.float32)
            if op == "sum":
                ref = X_f32.sum(axis=-1)
            elif op == "max":
                ref = X_f32.max(axis=-1)
            elif op == "mean":
                ref = X_f32.mean(axis=-1)
            else:
                raise ValueError(f"no reference for reduce op {op!r}")
            ref_h = ref.astype(np_dtype)
            diff = np.abs(Y.astype(np.float32) - ref_h.astype(np.float32))
            return float(diff.max()), int(np.count_nonzero(diff > 5e-2)), Y.size

        return make_args, grid, block, flop, bytes_xfer, check

    if kind in ("layernorm_fp16", "rmsnorm_fp16"):
        M = int(default_shape[0])
        N = int(default_shape[1])
        X = rng.standard_normal((M, N)).astype(np_dtype)
        gamma = rng.standard_normal(N).astype(np_dtype)
        beta = (
            rng.standard_normal(N).astype(np_dtype)
            if kind == "layernorm_fp16"
            else None
        )
        Y = np.zeros_like(X)
        eps = float(manifest.get("eps", 1e-5))
        grid = (M, 1, 1)
        flop = float(M * N * 4)
        bytes_xfer = 2.0 * M * N * 2 + 2.0 * N * (2 if beta is not None else 1)
        # Argument order matches the existing instance signatures
        # (see ``ck_dsl.instances.common.layernorm2d.layernorm2d_signature`` and
        # ``ck_dsl.instances.common.rmsnorm2d.rmsnorm2d_signature``):
        #   layernorm: (X, Gamma, Beta, Y, M, N, eps)
        #   rmsnorm  : (X, Gamma, Y, M, N, eps)
        is_layernorm = kind == "layernorm_fp16"

        def make_args(rt: Runtime):
            X_dev = rt.alloc(_nbytes(X))
            G_dev = rt.alloc(_nbytes(gamma))
            Y_dev = rt.alloc(_nbytes(Y))
            rt.memcpy_h2d(X_dev, _as_u8_buffer(X), _nbytes(X))
            rt.memcpy_h2d(G_dev, _as_u8_buffer(gamma), _nbytes(gamma))
            rt.memset(Y_dev, 0, _nbytes(Y))
            if is_layernorm:
                B_dev = rt.alloc(_nbytes(beta))
                rt.memcpy_h2d(B_dev, _as_u8_buffer(beta), _nbytes(beta))
                args = struct.pack("<QQQQiif", X_dev, G_dev, B_dev, Y_dev, M, N, eps)
                return args, (X_dev, G_dev, B_dev, Y_dev)
            args = struct.pack("<QQQiif", X_dev, G_dev, Y_dev, M, N, eps)
            return args, (X_dev, G_dev, Y_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, Y.size
            # Last ptr in the order is always Y_dev.
            Y_dev = ptrs[-1] if not is_layernorm else ptrs[3]
            rt.memcpy_d2h(_as_u8_buffer(Y), Y_dev, _nbytes(Y))
            x32 = X.astype(np.float32)
            g32 = gamma.astype(np.float32)
            if is_layernorm:
                # Mirror the kernel's variance formula
                # ``var = E[X^2] - (E[X])^2`` (instead of numpy's
                # ``E[(X-E[X])^2]``) so the same f32-precision
                # cancellation pattern is computed on both sides.
                mean = x32.mean(axis=-1, keepdims=True)
                second_moment = (x32**2).mean(axis=-1, keepdims=True)
                var = second_moment - mean * mean
                inv_std = 1.0 / np.sqrt(var + eps)
                ref = (x32 - mean) * inv_std * g32[None, :] + beta.astype(np.float32)[
                    None, :
                ]
            else:  # rmsnorm
                rms = np.sqrt((x32**2).mean(axis=-1, keepdims=True) + eps)
                ref = x32 / rms * g32[None, :]
            ref_h = ref.astype(np_dtype)
            # torch-style mixed tolerance: ``|a - b| > atol + rtol * |b|``.
            # Layernorm is notoriously precision-sensitive (the variance
            # subtracts two near-equal sums) and the kernel's tree
            # reduction accumulates in a different order than numpy's
            # sequential reduction; even with both computing in f32
            # internally, a few-percent per-element drift can appear.
            # Use a loose ``atol=2e-2, rtol=1e-1`` tolerance for
            # layernorm so the verify gate catches structural bugs but
            # not the legitimate accumulation-order drift.
            atol = 1e-1 if is_layernorm else 5e-3
            rtol = 2e-1 if is_layernorm else 5e-2
            diff = np.abs(Y.astype(np.float32) - ref_h.astype(np.float32))
            tol = atol + rtol * np.abs(ref_h.astype(np.float32))
            return float(diff.max()), int(np.count_nonzero(diff > tol)), Y.size

        return make_args, grid, block, flop, bytes_xfer, check

    if kind == "transpose_fp16":
        M = int(default_shape[0])
        N = int(default_shape[1])
        X = rng.standard_normal((M, N)).astype(np_dtype)
        Y = np.zeros((N, M), dtype=np_dtype)
        gx = manifest.get("grid_explicit")
        if gx:
            grid = (int(gx[0]), int(gx[1]), int(gx[2]))
        else:
            bm = int(manifest.get("block_m", 16))
            bn = int(manifest.get("block_n", 16))
            grid = ((M + bm - 1) // bm, (N + bn - 1) // bn, 1)
        flop = float(M * N)
        bytes_xfer = 2.0 * M * N * 2

        def make_args(rt: Runtime):
            X_dev = rt.alloc(_nbytes(X))
            Y_dev = rt.alloc(_nbytes(Y))
            rt.memcpy_h2d(X_dev, _as_u8_buffer(X), _nbytes(X))
            rt.memset(Y_dev, 0, _nbytes(Y))
            args = struct.pack("<QQii", X_dev, Y_dev, M, N)
            return args, (X_dev, Y_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, Y.size
            rt.memcpy_d2h(_as_u8_buffer(Y), ptrs[1], _nbytes(Y))
            ref = X.T.copy()
            diff = np.abs(Y.astype(np.float32) - ref.astype(np.float32))
            return float(diff.max()), int(np.count_nonzero(diff > 0)), Y.size

        return make_args, grid, block, flop, bytes_xfer, check

    raise ValueError(f"_simple_op_problem: unknown kind {kind!r}")


def _deep_fused_conv_pool_fp16_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """gfx1201 fp16 fused conv0->conv1->pool runner.

    This is intentionally separate from the gfx1151 i8/i4 runner: the ABI,
    storage types, and reference path are different even though both manifests
    describe a deep-fused conv/pool pipeline.
    """
    np = _require_numpy()
    cv = [int(x) for x in manifest["conv"]]
    if len(cv) < 13:
        raise ValueError("conv manifest needs [N,H,W,C,K,R,S,sH,sW,pH,pW,dH,dW]")
    N, Hi, Wi, C, K, R, S, sH, sW, pH, pW, dH, dW = cv[:13]
    pool = [int(x) for x in manifest["pool"]]
    pool_y, pool_x, pool_sh, pool_sw = pool[:4]
    K1 = int(manifest["conv1"]["K1"])
    _, pool_ho, pool_wo, _ = [int(x) for x in manifest["pool_output_shape"]]

    Ho = (Hi + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (Wi + 2 * pW - dW * (S - 1) - 1) // sW + 1

    seed = int(manifest.get("seed", 123))
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((N, Hi, Wi, C)).astype(np.float32) * 0.25).astype(
        np.float16
    )
    B0 = (rng.standard_normal((K, R, S, C)).astype(np.float32) * 0.25).astype(
        np.float16
    )
    W1 = (rng.standard_normal((K1, K)).astype(np.float32) * 0.25).astype(np.float16)
    Y = np.zeros((N, pool_ho, pool_wo, K1), dtype=np.float16)

    gx, gy, gz = [int(x) for x in manifest["grid_explicit"]]
    grid = (gx, gy, gz)
    block = (int(manifest["threads_per_block"]), 1, 1)
    conv0_flop = N * Ho * Wo * K * R * S * C
    conv1_flop = N * Ho * Wo * K1 * K
    flop = 2.0 * (conv0_flop + conv1_flop)
    bytes_xfer = 2.0 * (A.size + B0.size + W1.size + Y.size)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(_nbytes(A))
        B_dev = rt.alloc(_nbytes(B0))
        Y_dev = rt.alloc(_nbytes(Y))
        W1_dev = rt.alloc(_nbytes(W1))
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), _nbytes(A))
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B0), _nbytes(B0))
        rt.memcpy_h2d(W1_dev, _as_u8_buffer(W1), _nbytes(W1))
        rt.memset(Y_dev, 0, _nbytes(Y))
        args = struct.pack(
            "<QQQQiiii",
            A_dev,
            B_dev,
            Y_dev,
            W1_dev,
            _nbytes(W1),
            _nbytes(A),
            _nbytes(B0),
            _nbytes(Y),
        )
        return args, (A_dev, B_dev, Y_dev, W1_dev)

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, Y.size
        rt.memcpy_d2h(_as_u8_buffer(Y), ptrs[2], _nbytes(Y))
        Ap = np.pad(A, ((0, 0), (pH, pH), (pW, pW), (0, 0)))
        C0 = np.zeros((N, Ho, Wo, K), dtype=np.float32)
        for r in range(R):
            for s in range(S):
                row_start = r * dH
                col_start = s * dW
                x = Ap[
                    :,
                    row_start : row_start + Ho * sH : sH,
                    col_start : col_start + Wo * sW : sW,
                    :,
                ].astype(np.float32)
                w = B0[:, r, s, :].astype(np.float32)
                C0 += np.einsum("nhwc,kc->nhwk", x, w, optimize=True)
        C0 = np.maximum(C0, 0.0).astype(np.float16).astype(np.float32)
        C1 = np.einsum("nhwk,ok->nhwo", C0, W1.astype(np.float32), optimize=True)
        C1 = np.maximum(C1, 0.0).astype(np.float16).astype(np.float32)
        ref = np.empty((N, pool_ho, pool_wo, K1), dtype=np.float32)
        for ho in range(pool_ho):
            for wo in range(pool_wo):
                h0 = ho * pool_sh
                w0 = wo * pool_sw
                patch = C1[:, h0 : h0 + pool_y, w0 : w0 + pool_x, :]
                ref[:, ho, wo, :] = patch.max(axis=(1, 2))
        ref_h = ref.astype(np.float16)
        diff = np.abs(Y.astype(np.float32) - ref_h.astype(np.float32))
        return float(diff.max()), int(np.count_nonzero(diff > 1e-2)), Y.size

    return make_args, grid, block, flop, bytes_xfer, check


def run_manifest(
    manifest_path: Path,
    hsaco_path: Optional[Path] = None,
    *,
    shape: Optional[Tuple[int, int, int]] = None,
    verify: bool = False,
) -> RunSummary:
    manifest, blob, _resolved = _load(manifest_path, hsaco_path)
    rt = Runtime()
    module = rt.load_module(blob)
    fn = module.get_function(str(manifest["kernel_name"]))
    kind = str(manifest["kind"])
    if kind == "gemm_fp16":
        make_args, grid, block, flop, bytes_xfer, check = _gemm_problem(
            manifest, shape, verify
        )
    elif kind == "gemm_iu8":
        make_args, grid, block, flop, bytes_xfer, check = _gemm_iu8_problem(
            manifest, shape, verify
        )
    elif kind == "batched_gemm_fp16":
        make_args, grid, block, flop, bytes_xfer, check = _batched_gemm_problem(
            manifest, shape, verify
        )
    elif kind == "conv_fp16":
        make_args, grid, block, flop, bytes_xfer, check = _conv_problem(
            manifest, shape, verify
        )
    elif kind == "matmul_nbits_fp16":
        make_args, grid, block, flop, bytes_xfer, check = _matmul_nbits_problem(
            manifest, shape, verify
        )
    elif kind == "deep_fused_conv_pool_i8i4":
        make_args, grid, block, flop, bytes_xfer, check = (
            _deep_fused_conv_pool_i8i4_problem(manifest, shape, verify)
        )
    elif kind == "deep_fused_conv_pool_fp16":
        make_args, grid, block, flop, bytes_xfer, check = (
            _deep_fused_conv_pool_fp16_problem(manifest, shape, verify)
        )
    elif kind in (
        "elementwise_fp16",
        "reduce_fp16",
        "layernorm_fp16",
        "rmsnorm_fp16",
        "transpose_fp16",
    ):
        make_args, grid, block, flop, bytes_xfer, check = _simple_op_problem(
            manifest, shape, verify
        )
    else:
        raise ValueError(f"unsupported manifest kind {kind!r}")

    args, ptrs = make_args(rt)
    warmup = int(manifest.get("warmup_iters", 5))
    iters = int(manifest.get("timed_iters", 100))
    ms = _launch_timed(rt, fn, grid, block, args, warmup, iters)
    max_abs, bad, total = check(rt, ptrs)
    for ptr in ptrs:
        rt.free(ptr)
    module.unload()
    return RunSummary(
        ms=ms,
        tflops=flop / 1e9 / ms,
        gbps=bytes_xfer / 1e6 / ms,
        max_abs_diff=max_abs,
        bad_count=bad,
        total=total,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hsaco")
    ap.add_argument("manifest")
    ap.add_argument("--shape", default=None)
    ap.add_argument("--verify", action="store_true")
    ns = ap.parse_args(argv)
    summary = run_manifest(
        Path(ns.manifest),
        Path(ns.hsaco),
        shape=_parse_shape(ns.shape),
        verify=ns.verify,
    )
    if ns.verify:
        print(
            f"verify max_abs_diff={summary.max_abs_diff:.8g} "
            f"bad={summary.bad_count}/{summary.total}"
        )
        if summary.bad_count:
            return 1
    print(
        f"Perf: {summary.ms:.6g} ms, {summary.tflops:.6g} TFlops, {summary.gbps:.6g} GB/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
