# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest-runner problem builder for simple elementwise/reduce/norm ops."""

from __future__ import annotations

import struct
from typing import Optional, Tuple

from ....runtime.hip_module import Runtime
from .utils import as_u8_buffer, nbytes, require_numpy


def run_simple_op_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Generic ``input(s) -> output`` runner for small/simple op manifests."""
    np = require_numpy()
    kind = str(manifest["kind"])
    op = str(manifest.get("op", ""))
    dtype = str(manifest.get("dtype", "f16"))
    if dtype not in ("f16", "bf16"):
        raise ValueError(f"simple-op runner currently supports f16/bf16, got {dtype!r}")
    np_dtype = np.float16
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
        epb = int(manifest["elems_per_block"])
        grid = ((N + epb - 1) // epb, 1, 1)
        flop = float(N)
        bytes_xfer = 2.0 * N * (2 if is_binary else 1) + 2.0 * N

        def make_args(rt: Runtime):
            A_dev = rt.alloc(nbytes(A))
            C_dev = rt.alloc(nbytes(C))
            rt.memcpy_h2d(A_dev, as_u8_buffer(A), nbytes(A))
            rt.memset(C_dev, 0, nbytes(C))
            if is_binary:
                B_dev = rt.alloc(nbytes(B))
                rt.memcpy_h2d(B_dev, as_u8_buffer(B), nbytes(B))
                args = struct.pack("<QQQi", A_dev, B_dev, C_dev, N)
                return args, (A_dev, B_dev, C_dev)
            args = struct.pack("<QQi", A_dev, C_dev, N)
            return args, (A_dev, C_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, C.size
            C_dev = ptrs[-1]
            rt.memcpy_d2h(as_u8_buffer(C), C_dev, nbytes(C))
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
            ref_f32 = ref_h.astype(np.float32)
            tol = 1e-2
            err = np.abs(C.astype(np.float32) - ref_f32)
            bad = err > tol + tol * np.abs(ref_f32)
            return float(err.max()), int(np.count_nonzero(bad)), C.size

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
            X_dev = rt.alloc(nbytes(X))
            Y_dev = rt.alloc(nbytes(Y))
            rt.memcpy_h2d(X_dev, as_u8_buffer(X), nbytes(X))
            rt.memset(Y_dev, 0, nbytes(Y))
            args = struct.pack("<QQii", X_dev, Y_dev, M, N)
            return args, (X_dev, Y_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, Y.size
            rt.memcpy_d2h(as_u8_buffer(Y), ptrs[1], nbytes(Y))
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
            ref_f32 = ref_h.astype(np.float32)
            tol = 5e-2
            err = np.abs(Y.astype(np.float32) - ref_f32)
            bad = err > tol + tol * np.abs(ref_f32)
            return float(err.max()), int(np.count_nonzero(bad)), Y.size

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
        is_layernorm = kind == "layernorm_fp16"

        def make_args(rt: Runtime):
            X_dev = rt.alloc(nbytes(X))
            G_dev = rt.alloc(nbytes(gamma))
            Y_dev = rt.alloc(nbytes(Y))
            rt.memcpy_h2d(X_dev, as_u8_buffer(X), nbytes(X))
            rt.memcpy_h2d(G_dev, as_u8_buffer(gamma), nbytes(gamma))
            rt.memset(Y_dev, 0, nbytes(Y))
            if is_layernorm:
                B_dev = rt.alloc(nbytes(beta))
                rt.memcpy_h2d(B_dev, as_u8_buffer(beta), nbytes(beta))
                args = struct.pack("<QQQQiif", X_dev, G_dev, B_dev, Y_dev, M, N, eps)
                return args, (X_dev, G_dev, B_dev, Y_dev)
            args = struct.pack("<QQQiif", X_dev, G_dev, Y_dev, M, N, eps)
            return args, (X_dev, G_dev, Y_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, Y.size
            Y_dev = ptrs[-1] if not is_layernorm else ptrs[3]
            rt.memcpy_d2h(as_u8_buffer(Y), Y_dev, nbytes(Y))
            x32 = X.astype(np.float32)
            g32 = gamma.astype(np.float32)
            if is_layernorm:
                mean = x32.mean(axis=-1, keepdims=True)
                second_moment = (x32**2).mean(axis=-1, keepdims=True)
                var = second_moment - mean * mean
                inv_std = 1.0 / np.sqrt(var + eps)
                ref = (x32 - mean) * inv_std * g32[None, :] + beta.astype(np.float32)[
                    None, :
                ]
            else:
                rms = np.sqrt((x32**2).mean(axis=-1, keepdims=True) + eps)
                ref = x32 / rms * g32[None, :]
            ref_h = ref.astype(np_dtype)
            atol = 1e-1 if is_layernorm else 5e-3
            rtol = 2e-1 if is_layernorm else 5e-2
            ref_f32 = ref_h.astype(np.float32)
            err = np.abs(Y.astype(np.float32) - ref_f32)
            bad = err > atol + rtol * np.abs(ref_f32)
            return float(err.max()), int(np.count_nonzero(bad)), Y.size

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
            X_dev = rt.alloc(nbytes(X))
            Y_dev = rt.alloc(nbytes(Y))
            rt.memcpy_h2d(X_dev, as_u8_buffer(X), nbytes(X))
            rt.memset(Y_dev, 0, nbytes(Y))
            args = struct.pack("<QQii", X_dev, Y_dev, M, N)
            return args, (X_dev, Y_dev)

        def check(rt: Runtime, ptrs):
            if not verify:
                return 0.0, 0, Y.size
            rt.memcpy_d2h(as_u8_buffer(Y), ptrs[1], nbytes(Y))
            ref = X.T.copy()
            ref_f32 = ref.astype(np.float32)
            err = np.abs(Y.astype(np.float32) - ref_f32)
            bad = err > 0.0
            return float(err.max()), int(np.count_nonzero(bad)), Y.size

        return make_args, grid, block, flop, bytes_xfer, check

    raise ValueError(f"run_simple_op_manifest_problem: unknown kind {kind!r}")
