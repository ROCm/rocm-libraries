# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# New kernel, written for this provider against flydsl 0.3.4 (pip wheel).
#   toolchain: ROCm 7.1.0
#
# Provenance: upstream FlyDSL has no standalone pointwise kernel. Its device-side
# activation helpers (https://github.com/ROCm/FlyDSL, commit 226cde8d (v0.5.0),
# kernels/common/act.py) are written against the v0.5 API (rocdl.exp2 / rocdl.rcp
# on raw MLIR values) and fused into MoE epilogues; no code is taken from them.
# The activations here use the 0.3.4 `math` dialect wrappers instead.
#
# Upstream-first: a runtime-shape elementwise builder belongs in FlyDSL; this copy
# exists so the provider is not blocked on that.

"""Elementwise unary and binary ops with broadcasting, cuDNN pointwise semantics.

One object per (mode, in0 dtype, in1 dtype, out dtype). Every shape, stride and
broadcast is a runtime argument: the host collapses the output to at most four
axes ``[d0, d1, d2, d3]`` and gives each operand one element stride per axis, 0
on an axis it broadcasts along, so

    out[c] = f(a[sum c_i * sa_i], b[sum c_i * sb_i])

for every coordinate ``c`` of the output. Arithmetic is f32 whatever the storage
types; a 16-bit result is rounded to nearest even.

Modes (``param`` is the swish or softplus beta; the other modes ignore it):

    sigmoid   1 / (1 + exp(-a))
    swish     a / (1 + exp(-param * a))
    exp       exp(a)
    softplus  (max(z, 0) + log1p(exp(-|z|))) / param,  z = param * a
    neg       -a
    identity  a                      (a dtype cast when in0 and out differ)
    add, sub, mul, div               a op b

Launch: ``BLOCK_THREADS`` threads per workgroup, one output element per thread,
or with ``vec`` set one ``vector_width`` run of elements per thread. ``vec`` is for
the host to set only when the output and every input are one dense run of ``n``
elements (all strides 1 after collapsing); the last partial run is finished
element by element. Each output element is read from and written by one thread
only, so the output may alias an input of the same layout.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec

from kernels.common.kernels_common import dtype_to_elem_type

KERNEL_NAME = "pointwise"

BLOCK_THREADS = 256

SUPPORTED_DTYPES = ("bf16", "f16", "f32")
UNARY_MODES = ("sigmoid", "swish", "exp", "softplus", "neg", "identity")
BINARY_MODES = ("add", "sub", "mul", "div")

_ELEM_BYTES = {"bf16": 2, "f16": 2, "f32": 4}


def vector_width(*dtype_strs: str) -> int:
    """Elements per thread on the dense path: one 16-byte access of the widest operand."""
    return 16 // max(_ELEM_BYTES[d] for d in dtype_strs)


def _compute(mode: str, a, b, param):
    """``mode`` applied to f32 scalars."""
    one = fx.Float32(1.0)
    zero = fx.Float32(0.0)
    if const_expr(mode == "add"):
        return a + b
    if const_expr(mode == "sub"):
        return a - b
    if const_expr(mode == "mul"):
        return a * b
    if const_expr(mode == "div"):
        return a / b
    if const_expr(mode == "sigmoid"):
        return one / (one + fmath.exp(zero - a))
    if const_expr(mode == "swish"):
        return a / (one + fmath.exp(zero - param * a))
    if const_expr(mode == "exp"):
        return fmath.exp(a)
    if const_expr(mode == "softplus"):
        z = param * a
        positive = (z > zero).select(z, zero)
        return (positive + fmath.log1p(fmath.exp(zero - fmath.absf(z)))) / param
    if const_expr(mode == "neg"):
        return zero - a
    if const_expr(mode == "identity"):
        return a
    raise ValueError(f"unknown pointwise mode {mode!r}")


def build_pointwise_module(
    mode: str,
    in0_dtype: str,
    in1_dtype: str | None,
    out_dtype: str,
    BLOCK_THREADS: int = BLOCK_THREADS,
):
    """Build the pointwise module for one mode and one set of storage types.

    ``in1_dtype`` is None for a unary mode. Returns the ``@flyc.jit`` launcher. Its
    arguments are, in order: ``Out``, ``A``, ``B`` (pointers; a unary object never
    reads ``B``); ``n`` (output elements) and ``vec`` (Int32); the output extents
    ``d1, d2, d3`` of the collapsed axes 1..3 (Int32; axis 0 is implied by ``n``);
    the four element strides of the output, of ``A`` and of ``B``, outermost first
    (Int64); ``param`` (Float32); then the grid size (Int32) and the stream. The
    kernel takes the same list without the grid size.
    """
    binary = mode in BINARY_MODES
    if mode not in UNARY_MODES + BINARY_MODES:
        raise ValueError(f"unknown pointwise mode {mode!r}")
    if binary != (in1_dtype is not None):
        raise ValueError(
            f"mode {mode!r} is {'binary' if binary else 'unary'}; in1_dtype "
            f"{in1_dtype!r} disagrees"
        )
    for dtype_str in (in0_dtype, in1_dtype or in0_dtype, out_dtype):
        if dtype_str not in SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype_str!r}"
            )
    b_dtype_str = in1_dtype if binary else in0_dtype
    VEC = (
        vector_width(in0_dtype, b_dtype_str, out_dtype)
        if binary
        else vector_width(in0_dtype, out_dtype)
    )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def pointwise_kernel(
        Out: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        n: fx.Int32,
        vec: fx.Int32,
        d1: fx.Int32,
        d2: fx.Int32,
        d3: fx.Int32,
        so0: fx.Int64,
        so1: fx.Int64,
        so2: fx.Int64,
        so3: fx.Int64,
        sa0: fx.Int64,
        sa1: fx.Int64,
        sa2: fx.Int64,
        sa3: fx.Int64,
        sb0: fx.Int64,
        sb1: fx.Int64,
        sb2: fx.Int64,
        sb3: fx.Int64,
        param: fx.Float32,
    ):
        a_elem = dtype_to_elem_type(in0_dtype)
        b_elem = dtype_to_elem_type(b_dtype_str)
        out_elem = dtype_to_elem_type(out_dtype)

        def _as_ptr(ptr, dtype):
            return fx.recast_iter(
                fx.PointerType.get(dtype.ir_type, ptr.address_space), ptr
            )

        a_ptr = _as_ptr(A, a_elem)
        b_ptr = _as_ptr(B, b_elem)
        out_ptr = _as_ptr(Out, out_elem)

        def _load(ptr, idx, width):
            view = fx.make_view(fx.add_offset(ptr, idx), fx.make_layout(width, 1))
            return Vec(view.load())

        def _to_f32(value, dtype_str):
            if const_expr(dtype_str == "f32"):
                return value
            return value.to(fx.Float32)

        def _store(idx, values):
            out = Vec.from_elements(values, fx.Float32)
            if const_expr(out_dtype != "f32"):
                out = out.to(out_elem)
            view = fx.make_view(
                fx.add_offset(out_ptr, idx), fx.make_layout(len(values), 1)
            )
            view.store(out)

        def _element(a_idx, b_idx):
            a = _to_f32(_load(a_ptr, a_idx, 1)[0], in0_dtype)
            b = a
            if const_expr(binary):
                b = _to_f32(_load(b_ptr, b_idx, 1)[0], b_dtype_str)
            return _compute(mode, a, b, param)

        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        thread = bid * BLOCK_THREADS + tid

        if vec != 0:
            # Dense: output and inputs are one run of n elements, stride 1.
            first = thread * VEC
            if first + VEC <= n:
                at = fx.Int64(first)
                a_run = _load(a_ptr, at, VEC)
                b_run = a_run
                if const_expr(binary):
                    b_run = _load(b_ptr, at, VEC)
                values = []
                for j in range_constexpr(VEC):
                    a = _to_f32(a_run[j], in0_dtype)
                    b = a
                    if const_expr(binary):
                        b = _to_f32(b_run[j], b_dtype_str)
                    values.append(_compute(mode, a, b, param))
                _store(at, values)
            else:
                for j in range_constexpr(VEC):
                    if first + j < n:
                        at = fx.Int64(first + j)
                        _store(at, [_element(at, at)])
        else:
            if thread < n:
                c3 = thread % d3
                rest = thread // d3
                c2 = rest % d2
                rest = rest // d2
                c1 = rest % d1
                c0 = rest // d1
                i0 = fx.Int64(c0)
                i1 = fx.Int64(c1)
                i2 = fx.Int64(c2)
                i3 = fx.Int64(c3)
                a_idx = i0 * sa0 + i1 * sa1 + i2 * sa2 + i3 * sa3
                b_idx = i0 * sb0 + i1 * sb1 + i2 * sb2 + i3 * sb3
                o_idx = i0 * so0 + i1 * so1 + i2 * so2 + i3 * so3
                _store(o_idx, [_element(a_idx, b_idx)])

    @flyc.jit
    def launch_pointwise(
        Out: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        n: fx.Int32,
        vec: fx.Int32,
        d1: fx.Int32,
        d2: fx.Int32,
        d3: fx.Int32,
        so0: fx.Int64,
        so1: fx.Int64,
        so2: fx.Int64,
        so3: fx.Int64,
        sa0: fx.Int64,
        sa1: fx.Int64,
        sa2: fx.Int64,
        sa3: fx.Int64,
        sb0: fx.Int64,
        sb1: fx.Int64,
        sb2: fx.Int64,
        sb3: fx.Int64,
        param: fx.Float32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = pointwise_kernel(
            Out,
            A,
            B,
            n,
            vec,
            d1,
            d2,
            d3,
            so0,
            so1,
            so2,
            so3,
            sa0,
            sa1,
            sa2,
            sa3,
            sb0,
            sb1,
            sb2,
            sb3,
            param,
        )
        launcher.launch(
            grid=(grid, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_pointwise.block_threads = BLOCK_THREADS
    launch_pointwise.vector_width = VEC
    return launch_pointwise
