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
import struct
from dataclasses import dataclass
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
    ms: float
    tflops: float
    gbps: float
    max_abs_diff: float = 0.0
    bad_count: int = 0
    total: int = 0


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
    grid = (
        (N + int(manifest["block_n"]) - 1) // int(manifest["block_n"]),
        (M + int(manifest["block_m"]) - 1) // int(manifest["block_m"]),
        1,
    )
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
    D = np.empty((N, H, W, K), dtype=np.float16)

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
        Ap = np.pad(A, ((0, 0), (pH, pH), (pW, pW), (0, 0)), mode="constant")
        ref = np.zeros_like(D, dtype=np.float32)
        for r in range(R):
            for s in range(S):
                x = Ap[:, r : r + H * sH : sH, s : s + W * sW : sW, :]
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
        # (see ``ck_dsl.instances.layernorm2d.layernorm2d_signature`` and
        # ``ck_dsl.instances.rmsnorm2d.rmsnorm2d_signature``):
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
    elif kind == "batched_gemm_fp16":
        make_args, grid, block, flop, bytes_xfer, check = _batched_gemm_problem(
            manifest, shape, verify
        )
    elif kind == "conv_fp16":
        make_args, grid, block, flop, bytes_xfer, check = _conv_problem(
            manifest, shape, verify
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
