# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""2D pooling kernel instance (CK Tile ``36_pooling`` 2D counterpart).

DSL counterpart of CK Tile's ``example/ck_tile/36_pooling`` (which ships
a 3D NDHWC ``MaxPool`` example). For v1 we implement the 2D NHWC case
covering ``max`` / ``avg`` / ``sum`` reductions; the 3D extension is a
v2 follow-up (mostly an extra spatial dim in the descriptor + the inner
loop).

Layout::

    Input  : NHWC, shape (N, H, W, C)
    Output : NHWC, shape (N, Ho, Wo, C)

    Ho = (H + 2*pH - ((Y - 1) * dH + 1)) / sH + 1
    Wo = (W + 2*pW - ((X - 1) * dW + 1)) / sW + 1

For each output ``(n, ho, wo, c)`` the kernel walks the ``(y, x)``
window and reduces over the in-bounds input cells. Pad cells contribute
``-inf`` (max), ``0`` (sum), or are skipped from the count (avg).

Implementation notes:

* One thread per output element. The block covers a flat slab of
  ``block_size`` consecutive output positions; ``thread_id_x`` indexes
  into the slab.
* Input offsets go through a coordinate-transform descriptor that
  encodes the convolution-style affine map ``hi = ho*sH - pH + y*dH``
  (``embed`` transform with a ``lo`` / ``hi`` bound). Out-of-bounds
  cells flip the descriptor's ``valid`` predicate; the kernel masks
  them via ``select(valid, loaded, neutral)`` where the neutral value
  depends on the reduction op.
* Reductions accumulate in f32 to keep parity with CK Tile's
  ``ComputeDataType = float`` convention; the result is cast back to
  the I/O dtype at the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F16, I32, IRBuilder, KernelDef, PtrType, Value
from ..helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ..transforms import TensorDescriptor, embed


DType = Literal["f16", "bf16"]
PoolOp = Literal["max", "avg", "sum"]


@dataclass(frozen=True)
class PoolingProblem:
    """2D pooling shape parameters (NHWC input / output)."""

    N: int
    H: int
    W: int
    C: int

    Y: int  # window height
    X: int  # window width

    sH: int = 1  # stride
    sW: int = 1
    pH: int = 0  # left pad (also used as right pad for now)
    pW: int = 0
    dH: int = 1  # dilation
    dW: int = 1

    @property
    def Ho(self) -> int:
        return (self.H + 2 * self.pH - ((self.Y - 1) * self.dH + 1)) // self.sH + 1

    @property
    def Wo(self) -> int:
        return (self.W + 2 * self.pW - ((self.X - 1) * self.dW + 1)) // self.sW + 1

    @property
    def total_out(self) -> int:
        return self.N * self.Ho * self.Wo * self.C

    def short(self) -> str:
        return (
            f"N{self.N}H{self.H}W{self.W}C{self.C}"
            f"_Y{self.Y}X{self.X}_s{self.sH}x{self.sW}_p{self.pH}x{self.pW}"
        )


@dataclass(frozen=True)
class Pooling2DSpec:
    """One concrete pooling kernel configuration."""

    problem: PoolingProblem
    dtype: DType = "f16"
    op: PoolOp = "max"
    block_size: int = 256
    name: str = "ck_dsl_pooling2d"

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.problem.short(),
            self.dtype,
            self.op,
            f"b{self.block_size}",
        )


def is_valid_spec(spec: Pooling2DSpec) -> Tuple[bool, str]:
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=2,  # vec is unused for one-thread-per-output layout; pass a benign value
        )
    )
    if not ok:
        return ok, why
    if spec.op not in ("max", "avg", "sum"):
        return False, f"unsupported pool op {spec.op!r}"
    p = spec.problem
    if p.Y <= 0 or p.X <= 0:
        return False, f"window dims must be positive (Y={p.Y}, X={p.X})"
    if p.Ho <= 0 or p.Wo <= 0:
        return False, (
            f"output spatial dims must be positive "
            f"(Ho={p.Ho}, Wo={p.Wo}); check pad/stride/window vs H/W"
        )
    return True, "ok"


def _make_input_descriptor(p: PoolingProblem) -> TensorDescriptor:
    """Input descriptor: ``(n, ho, y, wo, x, c) -> NHWC linear offset``.

    Two ``embed`` transforms encode the conv-style affine map for the
    spatial dims; each carries a ``lo`` / ``hi`` bound so out-of-bounds
    cells (the padded zone) flip the validity predicate.
    """
    return TensorDescriptor.naive(
        "X_nhwc",
        lengths=[p.N, p.H, p.W, p.C],
        dtype=F16,
        coord_names=["n", "hi", "wi", "c"],
    ).transform(
        embed(
            upper=["ho", "y"],
            into="hi",
            strides=[p.sH, p.dH],
            offset=-p.pH,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=["wo", "x"],
            into="wi",
            strides=[p.sW, p.dW],
            offset=-p.pW,
            lo=0,
            hi=p.W,
        ),
    )


def _neutral_value(b: IRBuilder, op: PoolOp) -> Value:
    """f32 neutral element for the reduction."""
    if op == "max":
        return b.const_f32(float("-inf"))
    if op in ("sum", "avg"):
        return b.const_f32(0.0)
    raise ValueError(f"no neutral value for op {op!r}")


def _combine(b: IRBuilder, op: PoolOp, acc: Value, x: Value) -> Value:
    """Reduction step in f32."""
    if op == "max":
        return b.fmax(acc, x)
    if op in ("sum", "avg"):
        return b.fadd(acc, x)
    raise ValueError(f"no combiner for op {op!r}")


def build_pooling2d(spec: Pooling2DSpec) -> KernelDef:
    """Build the IR for one pooling instance.

    Kernel signature::

        (X: ptr<dtype, global>,   # NHWC input  [N, H,  W,  C]
         Y: ptr<dtype, global>,   # NHWC output [N, Ho, Wo, C]
         X_bytes: i32,            # buffer-rsrc byte length for X
         Y_bytes: i32)            # buffer-rsrc byte length for Y

    Grid: ``(ceil(N*Ho*Wo*C / block_size), 1, 1)``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid pooling2d spec: {why}")

    p = spec.problem
    from ..helpers.io import io_ir_type

    io_ty = io_ir_type(spec.dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    X_bytes = b.param("X_bytes", I32)
    Y_bytes = b.param("Y_bytes", I32)

    in_desc = _make_input_descriptor(p)

    c0 = b.const_i32(0)
    c_elem_bytes = b.const_i32(2)  # f16 / bf16 both 2 bytes
    c_total = b.const_i32(p.total_out)
    c_HoWoC = b.const_i32(p.Ho * p.Wo * p.C)
    c_WoC = b.const_i32(p.Wo * p.C)
    c_C = b.const_i32(p.C)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    out_idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)

    # Decompose the flat output index into (n, ho, wo, c).
    n_val = b.div(out_idx, c_HoWoC)
    rem_nhowoC = b.mod(out_idx, c_HoWoC)
    ho_val = b.div(rem_nhowoC, c_WoC)
    rem_howoC = b.mod(rem_nhowoC, c_WoC)
    wo_val = b.div(rem_howoC, c_C)
    c_val = b.mod(rem_howoC, c_C)

    in_bounds = b.cmp_lt(out_idx, c_total)

    x_rsrc = b.buffer_rsrc(X, X_bytes)
    y_rsrc = b.buffer_rsrc(Y, Y_bytes)

    # Reduction over the (Y, X) window. Both dims are compile-time so
    # the Python-level loop unrolls cleanly; for typical pooling
    # configurations (2x2, 3x3, 7x7) this is fine. Larger windows
    # would benefit from a runtime scf.for, which we can add later.
    neutral = _neutral_value(b, spec.op)
    acc = neutral
    valid_count = b.const_f32(0.0) if spec.op == "avg" else None

    for y_i in range(p.Y):
        c_y = b.const_i32(y_i)
        for x_i in range(p.X):
            c_x = b.const_i32(x_i)
            off, valid = in_desc.offset(
                b, n=n_val, ho=ho_val, y=c_y, wo=wo_val, x=c_x, c=c_val
            )
            off_bytes = b.mul(off, c_elem_bytes)
            safe_in_off = (
                b.select(valid, off_bytes, oob_sentinel)
                if valid is not None
                else off_bytes
            )
            # f16 / bf16 share the byte-wise buffer load surface
            # (``buffer_load_f16`` returns a 2-byte value either way;
            # the IR-level type tag drives the f32 cast).
            loaded_raw = b.buffer_load_f16(x_rsrc, safe_in_off, c0)
            loaded_f32 = b.cast_to_f32(loaded_raw)
            # Mask invalid loads with the neutral element so they don't
            # contribute to the reduction (especially relevant for max
            # pool: a buffer-OOB load returns 0, which would be the
            # wrong winner on negative-only inputs).
            if valid is not None:
                masked = b.select(valid, loaded_f32, neutral)
            else:
                masked = loaded_f32
            acc = _combine(b, spec.op, acc, masked)
            if spec.op == "avg":
                # Count only the valid cells. ``valid`` is i1; convert
                # to f32 via select(0.0, 1.0).
                contrib = (
                    b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
                    if valid is not None
                    else b.const_f32(1.0)
                )
                valid_count = b.fadd(valid_count, contrib)

    # Finalise: avg = sum / count (count >= 1 in practice because every
    # output position has at least one valid window cell; we guard
    # against div-by-zero with a 1.0 fallback for the all-pad case).
    if spec.op == "avg":
        safe_count = b.fmax(valid_count, b.const_f32(1.0))
        result_f32 = b.fmul(acc, b.rcp(safe_count))
    else:
        result_f32 = acc

    result = b.cast_f32_to(result_f32, io_ty)

    # Store: tail-of-grid threads (out_idx >= total) get their store
    # masked off via the buffer-rsrc OOB sentinel.
    safe_out_off = b.select(in_bounds, b.mul(out_idx, c_elem_bytes), oob_sentinel)
    b.buffer_store_f16(y_rsrc, safe_out_off, c0, result)

    return b.kernel


def pooling2d_grid(spec: Pooling2DSpec) -> Tuple[int, int, int]:
    """Return the launch grid: one thread per output element."""
    return ceil_div_grid((spec.problem.total_out, spec.block_size))


def pooling2d_signature(spec: Pooling2DSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("X_bytes", "i32")
        .scalar("Y_bytes", "i32")
        .build()
    )
