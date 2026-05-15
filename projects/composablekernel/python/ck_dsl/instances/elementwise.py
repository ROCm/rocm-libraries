# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Elementwise kernel instance builder (CK Tile ``21_elementwise`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/21_elementwise``. Emits a
single AMDGPU kernel that walks one contiguous N-element tensor with
vectorised global loads/stores and applies a fused unary or binary
operation per element.

What we cover today:

* Unary ops: ``copy``, ``neg``, ``abs``, ``relu``, ``gelu_tanh``, ``silu``, ``exp2``
* Binary ops: ``add``, ``sub``, ``mul``, ``max``, ``min``
* Dtypes: ``f16`` and ``bf16`` for I/O (compute is f32 internally)

The kernel processes the buffer as a single contiguous run of ``numel``
elements; multi-dimensional torch tensors must be ``contiguous()``. This
mirrors CK Tile's ``elementwise_example`` strategy.

The kernel uses :class:`ck_dsl.helpers.TensorView` for global I/O and
:func:`ck_dsl.helpers.io.io_ir_type` for the dtype dispatch; the per-
element math is the canonical exp2-based pattern (no ``llvm.tanh``,
which the AMDGPU backend can't lower).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import I32, IRBuilder, KernelDef, PtrType, Value
from ..helpers.io import io_ir_type
from ..helpers.spec import SignatureBuilder, kernel_name_join
from ..helpers.tensor_view import make_global_view


UnaryOp = Literal["copy", "neg", "abs", "relu", "gelu_tanh", "silu", "exp2"]
BinaryOp = Literal["add", "sub", "mul", "max", "min"]
DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class ElementwiseSpec:
    """One elementwise kernel instance."""

    op: str
    dtype: DType = "f16"
    block_size: int = 256
    vec: int = 8
    name: str = "ck_dsl_elementwise"

    def is_unary(self) -> bool:
        return self.op in ("copy", "neg", "abs", "relu", "gelu_tanh", "silu", "exp2")

    def is_binary(self) -> bool:
        return self.op in ("add", "sub", "mul", "max", "min")

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.op,
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
        )

    def elems_per_block(self) -> int:
        return self.block_size * self.vec


def is_valid_spec(spec: ElementwiseSpec) -> Tuple[bool, str]:
    if not (spec.is_unary() or spec.is_binary()):
        return False, f"unknown op {spec.op!r}"
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"block_size {spec.block_size} not in {{64, 128, 256, 512, 1024}}"
    if spec.vec not in (2, 4, 8):
        return False, f"vec {spec.vec} not in {{2, 4, 8}}"
    return True, "ok"


# ---------------------------------------------------------------------
# Op kernels (f32 scalar arithmetic).
# ---------------------------------------------------------------------


def _tanh_via_exp2(b: IRBuilder, x: Value) -> Value:
    """tanh(x) = (e^(2x) - 1) / (e^(2x) + 1), via exp2.

    Same primitive set as :func:`apply_softcap_runtime` in
    ``attention_unified``; we avoid ``llvm.tanh.f32`` because the
    AMDGPU backend doesn't lower it (it produces a
    ``CODEGEN_BC_TO_RELOCATABLE`` failure in ``amd_comgr``).
    """
    c_2log2e = b.const_f32(2.0 * 1.4426950408889634)
    c_one = b.const_f32(1.0)
    e2x = b.exp2(b.fmul(c_2log2e, x))
    return b.fmul(b.fsub(e2x, c_one), b.rcp(b.fadd(e2x, c_one)))


def _sigmoid_via_exp2(b: IRBuilder, x: Value) -> Value:
    """sigmoid(x) = 1 / (1 + e^-x) via exp2."""
    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one = b.const_f32(1.0)
    return b.rcp(b.fadd(one, b.exp2(b.fmul(c_neg_log2e, x))))


def _apply_unary(b: IRBuilder, x: Value, op: str) -> Value:
    if op == "copy":
        return x
    if op == "neg":
        return b.fsub(b.const_f32(0.0), x)
    if op == "abs":
        return b.fmax(x, b.fsub(b.const_f32(0.0), x))
    if op == "relu":
        return b.fmax(x, b.const_f32(0.0))
    if op == "exp2":
        return b.exp2(x)
    if op == "silu":
        return b.fmul(x, _sigmoid_via_exp2(b, x))
    if op == "gelu_tanh":
        # GELU (tanh approx): 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c_half = b.const_f32(0.5)
        c_one = b.const_f32(1.0)
        c_sq2_over_pi = b.const_f32(0.7978845608028654)
        c_a = b.const_f32(0.044715)
        x3 = b.fmul(b.fmul(x, x), x)
        inner = b.fmul(c_sq2_over_pi, b.fadd(x, b.fmul(c_a, x3)))
        return b.fmul(b.fmul(c_half, x), b.fadd(c_one, _tanh_via_exp2(b, inner)))
    raise ValueError(f"unsupported unary op {op!r}")


def _apply_binary(b: IRBuilder, a: Value, c: Value, op: str) -> Value:
    if op == "add":
        return b.fadd(a, c)
    if op == "sub":
        return b.fsub(a, c)
    if op == "mul":
        return b.fmul(a, c)
    if op == "max":
        return b.fmax(a, c)
    if op == "min":
        return b.fmin(a, c)
    raise ValueError(f"unsupported binary op {op!r}")


# ---------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------


def build_elementwise(spec: ElementwiseSpec) -> KernelDef:
    """Build the IR for one elementwise instance.

    Kernel signature (depends on op):
      * unary  : ``(A: ptr, C: ptr, N: i32)``
      * binary : ``(A: ptr, B: ptr, C: ptr, N: i32)``
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid elementwise spec: {why}")

    io_ty = io_ir_type(spec.dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    A = b.param("A", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    if spec.is_binary():
        Bp = b.param(
            "B", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
        )
    C = b.param("C", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    N = b.param("N", I32)

    # 1D views over the contiguous buffer. The shape entry is just
    # informational (no rank-1 bounds check); offset computation only
    # uses the stride which defaults to 1.
    a_view = make_global_view(A, shape=(1,), dtype=io_ty)
    c_view = make_global_view(C, shape=(1,), dtype=io_ty)
    b_view = make_global_view(Bp, shape=(1,), dtype=io_ty) if spec.is_binary() else None

    tid = b.thread_id_x()
    bid = b.block_id_x()
    c_vec = b.const_i32(spec.vec)
    c_chunk = b.const_i32(spec.block_size * spec.vec)

    block_base = b.mul(bid, c_chunk)
    thread_base = b.add(block_base, b.mul(tid, c_vec))

    fast_lim = b.add(thread_base, c_vec)
    in_fast = b.cmp_le(fast_lim, N)

    def emit_vec_path() -> None:
        a_scalars = a_view.load_vec_as_f32(b, [thread_base], n=spec.vec)
        if spec.is_binary():
            b_scalars = b_view.load_vec_as_f32(b, [thread_base], n=spec.vec)
            results = [
                _apply_binary(b, a_scalars[i], b_scalars[i], spec.op)
                for i in range(spec.vec)
            ]
        else:
            results = [_apply_unary(b, a_scalars[i], spec.op) for i in range(spec.vec)]
        c_view.store_vec_from_f32(b, [thread_base], values=results)

    def emit_scalar_path() -> None:
        for i in range(spec.vec):
            idx = b.add(thread_base, b.const_i32(i))
            in_bounds = b.cmp_lt(idx, N)
            with b.scf_if(in_bounds):
                a = b.cast_to_f32(a_view.load_scalar(b, [idx]))
                if spec.is_binary():
                    bv = b.cast_to_f32(b_view.load_scalar(b, [idx]))
                    r = _apply_binary(b, a, bv, spec.op)
                else:
                    r = _apply_unary(b, a, spec.op)
                c_view.store_scalar(b, [idx], b.cast_f32_to(r, io_ty))

    with b.scf_if(in_fast):
        emit_vec_path()
    with b.scf_if(b.lnot(in_fast)):
        emit_scalar_path()

    return b.kernel


def elementwise_grid(numel: int, spec: ElementwiseSpec) -> Tuple[int, int, int]:
    chunk = spec.elems_per_block()
    grid_x = (numel + chunk - 1) // chunk
    return (grid_x, 1, 1)


def elementwise_signature(spec: ElementwiseSpec):
    sb = SignatureBuilder().ptr("A", spec.dtype)
    if spec.is_binary():
        sb.ptr("B", spec.dtype)
    return sb.ptr("C", spec.dtype).scalar("N", "i32").build()
