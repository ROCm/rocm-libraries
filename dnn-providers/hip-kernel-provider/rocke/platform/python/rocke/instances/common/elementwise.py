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

CK Tile parity shape:

* Each CTA owns one ``block_size * vec``-element slab of the contiguous
  run (``S::Block_M`` in CK Tile terms; here a 1D row instead of a 2D
  M tile because there is no reduction).
* Each thread reads ``vec`` consecutive elements via
  :meth:`TileWindow.load_vec_as_f32` -- the per-lane CK Tile
  ``cast_tile<ComputeDataType>(load_tile(...))`` analogue -- promotes to
  ``f32``, applies the per-element op (``ElementWiseOperation{}`` in
  CK Tile), and writes back via :meth:`TileWindow.store_vec_from_f32`.
* The ``ElementWiseOperation`` math is f32 because that is the CK Tile
  ``ComputeDataType`` convention; the per-lane math IS per-element
  scalar f32 because each lane already owns one f32 register per slot
  (there is no SIMD-over-N-lane f32 instruction on AMDGPU; the packed
  ``v_pk_*`` family is f16 / bf16 only, and going through those would
  require us to drop the f32 compute precision).
* When the trailing chunk is partial (``thread_base + vec > N``), the
  kernel falls through to a per-element scalar loop guarded by
  ``cmp_lt(idx, N)``.

The per-element math uses the canonical ``exp2``-based sigmoid /
``tanh`` pattern (no ``llvm.tanh``, which the AMDGPU backend can't
lower); negation goes through :meth:`IRBuilder.fneg` rather than
``fsub(0.0, x)`` so the LLVM lowering emits a single ``v_neg``-form
sign-flip instead of a one-op ``v_sub``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.activations import _sigmoid_via_exp2, _tanh_via_exp2
from ...helpers.distribution import (
    TileDistributionEncoding,
    make_static_distributed_tensor,
    make_static_tile_distribution,
)
from ...helpers.io import io_ir_type
from ...helpers.spec import SignatureBuilder, kernel_name_join
from ...helpers.tensor_view import make_global_view, make_tile_window


UnaryOp = Literal[
    "copy",
    "neg",
    "abs",
    "relu",
    "gelu_tanh",
    "quick_gelu",
    "silu",
    "swish",
    "tanh",
    "sigmoid",
    "exp2",
]
BinaryOp = Literal["add", "sub", "mul", "max", "min", "swiglu", "geglu"]
DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class ElementwiseSpec:
    """One elementwise kernel instance."""

    op: str
    dtype: DType = "f16"
    block_size: int = 256
    vec: int = 8
    name: str = "rocke_elementwise"

    def is_unary(self) -> bool:
        return self.op in (
            "copy",
            "neg",
            "abs",
            "relu",
            "gelu_tanh",
            "quick_gelu",
            "silu",
            "swish",
            "tanh",
            "sigmoid",
            "exp2",
        )

    def is_binary(self) -> bool:
        return self.op in ("add", "sub", "mul", "max", "min", "swiglu", "geglu")

    def is_bias(self) -> bool:
        """P80: ``op`` names with the ``bias_`` prefix are 3-operand
        ``(x, bias, ...)`` ops where ``bias`` is broadcast along the
        batch axis (stride 0). Currently shipped: ``"bias_add"``,
        ``"bias_add_relu"``, ``"bias_add_silu"``, ``"bias_add_gelu_tanh"``.
        Mirrors CK Tile ``21_elementwise``'s multi-D operand convention.
        """
        return self.op.startswith("bias_")

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


def _gelu_tanh(b: IRBuilder, x: Value) -> Value:
    """Tanh-approximation GELU in f32.

    ``gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``

    Factored out so SwiGLU's GELU sibling (``geglu``) shares the exact
    same constant pool and op chain rather than duplicating the
    arithmetic at the binary call site.
    """
    c_half = b.const_f32(0.5)
    c_one = b.const_f32(1.0)
    c_sq2_over_pi = b.const_f32(0.7978845608028654)
    c_a = b.const_f32(0.044715)
    # Cube x via (x*x)*x: depth-2 dependency chain, same op count as
    # the natural ``x*x*x`` triple-mul; we write it this way so the
    # IR builder folds the constant-1 stride on the third multiply
    # rather than two consecutive ``v_mul`` issues.
    x2 = b.fmul(x, x)
    x3 = b.fmul(x2, x)
    inner = b.fmul(c_sq2_over_pi, b.fadd(x, b.fmul(c_a, x3)))
    return b.fmul(b.fmul(c_half, x), b.fadd(c_one, _tanh_via_exp2(b, inner)))


def _apply_unary(b: IRBuilder, x: Value, op: str) -> Value:
    if op == "copy":
        return x
    if op == "neg":
        # ``arith.fneg`` lowers to a single sign-bit flip; the
        # ``fsub(0.0, x)`` form used to emit a constant-load +
        # ``v_sub`` pair pre-canonicalisation.
        return b.fneg(x)
    if op == "abs":
        # ``|x| = max(x, -x)`` is the AMDGPU-friendly form; the
        # ``v_max3_f32`` family selects the max directly without the
        # ``fsub(0.0, x)`` constant load.
        return b.fmax(x, b.fneg(x))
    if op == "relu":
        return b.fmax(x, b.const_f32(0.0))
    if op == "exp2":
        return b.exp2(x)
    if op == "tanh":
        return _tanh_via_exp2(b, x)
    if op == "sigmoid":
        return _sigmoid_via_exp2(b, x)
    if op in ("silu", "swish"):
        # SiLU / Swish: x * sigmoid(x). The two names are interchangeable
        # in the literature; ``swish`` is the original Google paper name
        # and ``silu`` is the variant adopted by PyTorch.
        return b.fmul(x, _sigmoid_via_exp2(b, x))
    if op == "quick_gelu":
        # Quick GELU: x * sigmoid(1.702 * x). The activation used by
        # OpenAI's GPT-2 / CLIP era models -- a numerically cheap
        # approximation of the exact GELU that's a single sigmoid plus
        # one multiply.
        c_1702 = b.const_f32(1.702)
        return b.fmul(x, _sigmoid_via_exp2(b, b.fmul(c_1702, x)))
    if op == "gelu_tanh":
        return _gelu_tanh(b, x)
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
    if op == "swiglu":
        # SwiGLU: silu(a) * c. The "Swish-Gated Linear Unit" used in
        # LLaMA-style FFNs; we keep ``a`` on the activation side and
        # ``c`` on the gate side to match PyTorch's
        # ``F.silu(gate) * value`` convention when ``a == gate`` and
        # ``c == value``.
        return b.fmul(b.fmul(a, _sigmoid_via_exp2(b, a)), c)
    if op == "geglu":
        # GeGLU: gelu(a) * c -- same gating shape as SwiGLU but with
        # tanh-approx GELU on the activation side. Used in PaLM and
        # GLU-variant T5 papers. The activation reuses
        # :func:`_gelu_tanh` so the constant pool stays unique.
        return b.fmul(_gelu_tanh(b, a), c)
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

    # CK Tile distribution of the per-block fast tile.
    #
    # The fast path owns a contiguous ``block_size * vec`` slab; we model
    # it as a single 1D X dim hierarchically split as
    # ``Hs = ((block_size, vec),)``:
    #
    #   * level 0 (``block_size``) is the lane/thread axis -> P0
    #   * level 1 (``vec``)        is the per-thread vector  -> Y0
    #
    # ``calculate_x`` then reconstructs the in-tile element index as
    # ``x = tid * vec + y`` -- exactly the ``thread_base + i`` decode the
    # hand-rolled path used, but now expressed through the idiomatic
    # ``TileDistribution`` + ``StaticDistributedTensor`` surface (the same
    # machinery ``examples/common/distribution_2d_add_demo.py`` drives).
    encoding = TileDistributionEncoding(
        Hs=((spec.block_size, spec.vec),),
        Ps2RHs_major=((1,),),
        Ps2RHs_minor=((0,),),
        Ys2RHs_major=(1,),
        Ys2RHs_minor=(1,),
    )
    distribution = make_static_tile_distribution(encoding)
    tile_elems = spec.block_size * spec.vec

    # 1D views over the contiguous buffer. The shape entry is just
    # informational (no rank-1 bounds check); offset computation only
    # uses the stride which defaults to 1.
    a_view = make_global_view(A, shape=(tile_elems,), dtype=io_ty)
    c_view = make_global_view(C, shape=(tile_elems,), dtype=io_ty)
    b_view = (
        make_global_view(Bp, shape=(tile_elems,), dtype=io_ty)
        if spec.is_binary()
        else None
    )

    tid = b.thread_id_x()
    bid = b.block_id_x()
    c_vec = b.const_i32(spec.vec)
    c_chunk = b.const_i32(tile_elems)

    block_base = b.mul(bid, c_chunk)
    thread_base = b.add(block_base, b.mul(tid, c_vec))

    fast_lim = b.add(thread_base, c_vec)
    in_fast = b.cmp_le(fast_lim, N)

    # Per-block tile windows anchored at this CTA's slab origin. The
    # distribution P-coordinate is just the lane id; it feeds level 0 of
    # the H decomposition.
    a_tile = make_tile_window(a_view, lengths=(tile_elems,), origin=(block_base,))
    c_tile = make_tile_window(c_view, lengths=(tile_elems,), origin=(block_base,))
    b_tile = (
        make_tile_window(b_view, lengths=(tile_elems,), origin=(block_base,))
        if spec.is_binary()
        else None
    )
    ps = [[tid]]

    def emit_vec_path() -> None:
        # Tile-shaped fast path, CK Tile idiom: ``load_tile`` ingests the
        # per-thread Y span (f32-promoted) into a StaticDistributedTensor,
        # we sweep Y applying the per-element f32 op, and ``store_tile``
        # packs + writes the result. Mirrors CK Tile's
        # ``store_tile(y_window, cast_tile<YDataType>(y_tile))``.
        a_dt = a_tile.load(b, distribution=distribution, ps=ps)
        out_dt = make_static_distributed_tensor(distribution, dtype=io_ty)
        if spec.is_binary():
            b_dt = b_tile.load(b, distribution=distribution, ps=ps)
            for y in distribution.iterate_ys():
                out_dt.set(y, _apply_binary(b, a_dt.get(y), b_dt.get(y), spec.op))
        else:
            for y in distribution.iterate_ys():
                out_dt.set(y, _apply_unary(b, a_dt.get(y), spec.op))
        c_tile.store(b, out_dt, ps=ps)

    def emit_scalar_path() -> None:
        # Trailing-tail scalar fallback. The fast path's
        # ``thread_base + vec <= N`` predicate is false here, so each
        # of the up-to-``vec`` lanes individually probes ``idx < N``
        # before issuing its scalar load / op / store. The list
        # comprehension over ``range(spec.vec)`` is unrolled at IR
        # construction time so the inner branches all fold to direct
        # predicated ops.
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
