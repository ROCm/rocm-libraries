# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# New kernel, written for this provider against flydsl 0.3.4 (pip wheel).
#   toolchain: ROCm 7.1.0
#
# Provenance: the rotate-half (NeoX) formulation follows upstream FlyDSL's
#   https://github.com/ROCm/FlyDSL
#   commit:   226cde8d2a84bc2687c8d69587607fddf0c16f2f  (v0.5.0)
#   path:     kernels/attention/fused_rope_cache_kernel.py
# but no code is taken from it. That kernel fuses RoPE with a KV-cache write and
# bakes everything this op leaves to the graph: head_dim, the head counts, a
# packed [T, H, D] layout, cos/sin caches indexed by a position tensor, and
# full rotation only (it raises on rotary_dim != head_dim). Porting it would
# have kept none of its structure, so this is a fresh kernel instead:
#   - raw FP32 angles in, sin/cos computed in-kernel (cuDNN RoPE semantics);
#   - partial rotation of any even-width segment [o, o + r) of the last dim;
#   - head_dim, rope_dim, rope_offset and output_scale are runtime arguments,
#     so one object per dtype serves every shape;
#   - every operand is addressed through runtime element strides, so BSHD,
#     BHSD and broadcast (stride-0) freqs need no copies.
#
# Upstream-first: a runtime-shape RoPE builder belongs in FlyDSL; this copy
# exists so the provider is not blocked on that.

"""Rotary position embedding (rotate-half) forward, cuDNN RoPE semantics.

For x of shape [d0, d1, d2, D] (lower ranks are padded with leading unit
axes), r = rope_dim, h = r / 2, o = rope_offset, s = output_scale and f the
raw angles broadcast to x (freqs' last dim is r), for 0 <= i < h:

    y[o + i]     = s * (x[o + i] cos f[i]         - x[o + i + h] sin f[i])
    y[o + i + h] = s * (x[o + i + h] cos f[i + h] + x[o + i] sin f[i + h])

and y[j] = s * x[j] outside [o, o + r). Arithmetic is f32.

Launch: one workgroup per row, grid (d2, d1, d0), ``BLOCK_THREADS`` threads.
A row's D - h work items are: the passthrough elements before the segment,
the h rotation pairs, then the passthrough elements after it. Each work item
reads and writes only its own elements, so y may alias x.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec

from kernels.common.kernels_common import dtype_to_elem_type

KERNEL_NAME = "rope_fwd"

# One wave64 workgroup per row. A Qwen row (D = 256, r = 64) is 224 work items.
BLOCK_THREADS = 64

SUPPORTED_DTYPES = ("bf16", "f16", "f32")


def build_rope_module(dtype_str: str, BLOCK_THREADS: int = BLOCK_THREADS):
    """Build the RoPE forward module for one element type of x and y.

    Returns the ``@flyc.jit`` launcher. Its arguments are, in order: ``X``,
    ``Freqs``, ``Y`` (pointers; ``Freqs`` is f32); the four element strides of
    x, of freqs and of y, outermost first (Int64; a broadcast freqs axis has
    stride 0); ``head_dim``, ``rope_dim``, ``rope_offset`` (Int32);
    ``output_scale`` (Float32); then the grid extents ``d2, d1, d0`` (Int32)
    and the stream. The kernel takes the same list without the grid extents.
    """
    if dtype_str not in SUPPORTED_DTYPES:
        raise ValueError(
            f"dtype_str must be one of {SUPPORTED_DTYPES}, got {dtype_str!r}"
        )
    IS_F32 = dtype_str == "f32"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def rope_fwd_kernel(
        X: fx.Pointer,
        Freqs: fx.Pointer,
        Y: fx.Pointer,
        x_s0: fx.Int64,
        x_s1: fx.Int64,
        x_s2: fx.Int64,
        x_s3: fx.Int64,
        f_s0: fx.Int64,
        f_s1: fx.Int64,
        f_s2: fx.Int64,
        f_s3: fx.Int64,
        y_s0: fx.Int64,
        y_s1: fx.Int64,
        y_s2: fx.Int64,
        y_s3: fx.Int64,
        head_dim: fx.Int32,
        rope_dim: fx.Int32,
        rope_offset: fx.Int32,
        output_scale: fx.Float32,
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)

        def _as_ptr(ptr, dtype):
            return fx.recast_iter(
                fx.PointerType.get(dtype.ir_type, ptr.address_space), ptr
            )

        x_ptr = _as_ptr(X, elem_dtype)
        f_ptr = _as_ptr(Freqs, fx.Float32)
        y_ptr = _as_ptr(Y, elem_dtype)

        def _load(ptr, idx):
            view = fx.make_view(fx.add_offset(ptr, idx), fx.make_layout(1, 1))
            return Vec(view.load())[0]

        def _load_x(idx):
            value = _load(x_ptr, idx)
            if const_expr(IS_F32):
                return value
            return value.to(fx.Float32)

        def _store_y(idx, value):
            if const_expr(not IS_F32):
                value = value.to(elem_dtype)
            view = fx.make_view(fx.add_offset(y_ptr, idx), fx.make_layout(1, 1))
            view.store(Vec.filled(1, value, elem_dtype))

        i2 = fx.Int64(fx.block_idx.x)
        i1 = fx.Int64(fx.block_idx.y)
        i0 = fx.Int64(fx.block_idx.z)
        x_row = i0 * x_s0 + i1 * x_s1 + i2 * x_s2
        f_row = i0 * f_s0 + i1 * f_s1 + i2 * f_s2
        y_row = i0 * y_s0 + i1 * y_s1 + i2 * y_s2

        tid = fx.Int32(fx.thread_idx.x)
        half = rope_dim // fx.Int32(2)
        seg_end = rope_offset + half
        items = head_dim - half
        scale = output_scale

        for base, _state in range(0, items, BLOCK_THREADS, init=[]):
            k = tid + fx.Int32(base)
            if k < items:
                in_pair = (k >= rope_offset) & (k < seg_end)
                if in_pair:
                    i = k - rope_offset
                    lo = fx.Int64(k)
                    hi = fx.Int64(k + half)
                    x_lo = _load_x(x_row + lo * x_s3)
                    x_hi = _load_x(x_row + hi * x_s3)
                    f_lo = _load(f_ptr, f_row + fx.Int64(i) * f_s3)
                    f_hi = _load(f_ptr, f_row + fx.Int64(i + half) * f_s3)
                    y_lo = scale * (x_lo * fmath.cos(f_lo) - x_hi * fmath.sin(f_lo))
                    y_hi = scale * (x_hi * fmath.cos(f_hi) + x_lo * fmath.sin(f_hi))
                    _store_y(y_row + lo * y_s3, y_lo)
                    _store_y(y_row + hi * y_s3, y_hi)
                else:
                    # Before the segment k is the element; past its first half,
                    # k + h is (the second half is served by the pairs).
                    j = fx.Int64((k < rope_offset).select(k, k + half))
                    _store_y(y_row + j * y_s3, scale * _load_x(x_row + j * x_s3))

    @flyc.jit
    def launch_rope_fwd(
        X: fx.Pointer,
        Freqs: fx.Pointer,
        Y: fx.Pointer,
        x_s0: fx.Int64,
        x_s1: fx.Int64,
        x_s2: fx.Int64,
        x_s3: fx.Int64,
        f_s0: fx.Int64,
        f_s1: fx.Int64,
        f_s2: fx.Int64,
        f_s3: fx.Int64,
        y_s0: fx.Int64,
        y_s1: fx.Int64,
        y_s2: fx.Int64,
        y_s3: fx.Int64,
        head_dim: fx.Int32,
        rope_dim: fx.Int32,
        rope_offset: fx.Int32,
        output_scale: fx.Float32,
        d2: fx.Int32,
        d1: fx.Int32,
        d0: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rope_fwd_kernel(
            X,
            Freqs,
            Y,
            x_s0,
            x_s1,
            x_s2,
            x_s3,
            f_s0,
            f_s1,
            f_s2,
            f_s3,
            y_s0,
            y_s1,
            y_s2,
            y_s3,
            head_dim,
            rope_dim,
            rope_offset,
            output_scale,
        )
        launcher.launch(
            grid=(d2, d1, d0),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_rope_fwd.block_threads = BLOCK_THREADS
    return launch_rope_fwd
