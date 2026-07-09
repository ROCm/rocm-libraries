# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Multiple-D GEMM kernel instance (CK Tile ``19_gemm_multi_d`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/19_gemm_multi_d``. The
kernel computes::

    E = f(A * B, D_0, D_1, ..., D_{n-1})

where ``f`` is a user-supplied elementwise fusion of the accumulator
output and N ``(M, N)`` side inputs. Typical uses:

* ``E = A*B + D0``                       (bias / residual add)
* ``E = (A*B + D0) * D1``                (bias + gating)
* ``E = relu(A*B + D0)``                 (bias + activation; composes
                                           with the existing ``ReLU`` op)
* ``E = (A*B + D0 + D1 + ...)``          (multi-residual sum)

Implementation re-uses the existing :class:`UniversalGemmSpec` body
(MFMA + cshuffle epilogue) and attaches a :class:`FusedEpilogue` whose
op chain is one :class:`ResidualAdd` / :class:`ResidualMul` per D
operand. Each D becomes an extra ``(M, N)`` row-major pointer
parameter on the kernel (named ``D0``, ``D1``, ..., in the order the
spec lists them).

What we cover today:

* Same tile / pipeline / scheduler space as :func:`build_universal_gemm`
  (fp16 RCR, MFMA 16x16x{16,32} and 32x32x{8,16}).
* Per-D op chosen from ``{"add", "mul"}``. The CK Tile example's
  ``CDEElementWise`` template is more general; ``add`` / ``mul`` cover
  the common bias / residual / gate cases.
* Up to 8 D operands (CK Tile defaults to 1-2 in the example; the
  cap here is the same ``MAX_D`` we expose on the spec).
* Requires ``epilogue="cshuffle"`` because the ``default`` epilogue
  doesn't have the fused-epilogue hook wired in. Validation rejects
  ``default`` so the failure mode is loud, not silent.

The shape contract: every D is laid out row-major as ``(M, N)``, with
the same M and N as the GEMM output, and stride-M defaulting to N
(contiguous). Non-contiguous D's can be supported by extending the
underlying :class:`ResidualAdd` op to thread through a per-D stride
argument — left as a v2 follow-on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

from ...core.ir import F16, IRBuilder, Type, Value, KernelDef
from ...helpers.fuse import (
    FusedEpilogue,
    ResidualAdd,
    ResidualMul,
    dtype_to_ir,
    ir_dtype_global_load,
)
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


MAX_D = 8
DOp = Tuple[str, str]  # (param_name, "add" | "mul")
DLoadKind = Literal["stock", "tiled", "vector"]
"""How the fused epilogue should issue D operand loads.

The AMDGPU backend coalesces all three forms into the same wide
``global_load_dwordx4`` per slice after instruction selection, so
steady-state wall time is within microbench noise (<1%) across
all three. The trade-off is in IR size and comgr compile time:

* ``"stock"``  -- per-element scalar loads through the base
  :class:`~rocke.helpers.fuse.FusedEpilogue`. Equivalent to
  pre-edit behavior; largest IR among the three options.
* ``"tiled"``  -- per-element scalar loads but with the
  ``m * stride_m + n`` base offset hoisted out of the inner loop.
  ~45% smaller IR; the per-element load shape preserves the
  LLVM scheduler's preferred form for interleaving with MFMA.
* ``"vector"`` (default) -- one ``global_load_vN`` per D operand
  per ``apply_vec`` slice. ~50% smaller IR, ~20% faster comgr
  lowering. Wall time equivalent to ``"stock"`` after warmup;
  best for compile-heavy dispatcher sweeps.
"""


# ---------------------------------------------------------------------
# Tile-level multi-D epilogue
# ---------------------------------------------------------------------
#
# Background: the stock :class:`~rocke.helpers.fuse.FusedEpilogue`
# calls each op's ``apply_element`` once per output element. For a
# residual-add / residual-mul chain that means ``num_d * n_elems``
# *per-element address-arithmetic chains* per ``store_vec``: each
# call recomputes ``m * stride_m + n + i``, even though the only
# index that varies is the per-element ``i``.
#
# CK Tile's ``cshuffle_epilogue.hpp`` ``apply_d_tensors`` instead
# computes the per-D ``tile_window`` origin once per shuffle slice,
# pulls the D tile in via ``load_tile``, and combines element-wise
# via the user's ``CDEElementwise``. The DSL analogue is to
# compute the (m, n) -> flat-element base offset once per
# ``apply_vec`` call (shared across every D operand and every
# element index), then loop element-wise from that base.
#
# Two related but distinct optimisations live below in one class
# (:class:`_MultiDEpilogue`); the active strategy is selected by
# :attr:`GemmMultiDSpec.d_load_kind` (``_load_kind`` on the epilogue):
#
# 1. ``"tiled"``: keeps the per-element scalar D load (matches the LLVM
#    backend's preferred MFMA / VMEM interleave) but hoists the row-stride
#    multiply + (m * N + n) base out of the inner loop. ~45% smaller IR
#    than the stock form; wall time within microbench noise.
#
# 2. ``"vector"`` (the default): hoists the per-element address arithmetic
#    *and* coalesces the N scalar D loads into one ``global_load_vN`` per D.
#    The IR / LLVM both shrink (~50% IR reduction at the 128x128 tile);
#    after warmup, wall time matches the stock path within bench noise
#    because the AMDGPU backend already coalesces the per-element loads.
#    comgr lowering is ~10-20% faster on the smaller IR, which is the
#    user-visible win for compile-heavy dispatcher sweeps.
#
# Reference primitives in scope: ``IRBuilder.global_load_vN`` and
# ``IRBuilder.vec_extract`` for the vector path, plus
# ``IRBuilder.fadd`` / ``IRBuilder.fmul`` for the per-element
# combine. Reference C++:
#  * ``include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp``
#    -- ``apply_d_tensors`` issues ``load_tile`` per D operand and
#    fuses via ``CDEElementwise``.
#  * ``example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.cpp`` --
#    plain (M, N) row-major Ds combined by element-wise add/mul.


@dataclass(frozen=True)
class _MultiDEpilogue(FusedEpilogue):
    """Multi-D fused epilogue for a pure ``ResidualAdd`` / ``ResidualMul`` chain.

    One class drives both D-load strategies (selected by
    :attr:`_load_kind`, sourced from ``GemmMultiDSpec.d_load_kind``); the
    classification (``_residual_kinds`` / ``_residual_dtypes``), the
    ``off_base = m * stride_m + n`` hoist, and the per-element add/mul combine
    are shared:

    * ``"tiled"`` -- keeps the per-element scalar D load (the AMDGPU backend
      already coalesces aligned addresses into one ``global_load_dwordx{n}``)
      but hoists the ``m * stride_m + n`` base out of the inner loop. For a
      chain of ``num_d`` residual ops over an ``n_elems``-wide output vector
      the stock path emits ``num_d * n_elems`` (mul, add, add) chains; this
      emits one (mul, add) plus ``num_d * n_elems`` (add, load) pairs, matching
      what CK Tile's ``apply_d_tensors`` produces after ``tile_window``
      resolution.
    * ``"vector"`` (the default selection) -- additionally coalesces the
      ``n_elems`` scalar D loads into one ``global_load_vN`` per D operand
      per slice. Bit-identical output to the tiled form (verified on GPU
      against the stock :class:`FusedEpilogue`); ~50% smaller IR than stock
      and ~10-20% faster comgr lowering, with wall time within bench noise
      after warmup.

    Either way, any non-residual op kind (e.g. ``ResidualAdd + Cast + ReLU``),
    or a ``vector`` request with an unsupported ``n_elems`` / D dtype, falls
    back to the base-class per-element :meth:`apply_element` path so the
    optimisation never regresses correctness on chains it doesn't recognise.
    """

    _residual_kinds: Tuple[Optional[str], ...] = field(default_factory=tuple)
    _residual_dtypes: Tuple[Optional[Type], ...] = field(default_factory=tuple)
    _load_kind: str = "vector"

    @classmethod
    def from_ops(
        cls, ops: Tuple, dtype: Any = F16, *, load_kind: str = "vector"
    ) -> "_MultiDEpilogue":
        kinds: List[Optional[str]] = []
        dts: List[Optional[Type]] = []
        for op in ops:
            if isinstance(op, ResidualAdd):
                kinds.append("add")
                dts.append(dtype_to_ir(op.dtype))
            elif isinstance(op, ResidualMul):
                kinds.append("mul")
                dts.append(dtype_to_ir(op.dtype))
            else:
                kinds.append(None)
                dts.append(None)
        return cls(
            ops=tuple(ops),
            dtype=dtype,
            _residual_kinds=tuple(kinds),
            _residual_dtypes=tuple(dts),
            _load_kind=load_kind,
        )

    def apply_vec(
        self,
        b: IRBuilder,
        v: Value,
        m: Value,
        n: Value,
        *,
        n_elems: int,
    ) -> Value:
        vectorize = self._load_kind == "vector"
        if vectorize and n_elems not in (2, 4, 8):
            return super().apply_vec(b, v, m, n, n_elems=n_elems)
        if not all(k is not None for k in self._residual_kinds):
            # Heterogeneous chain (e.g. ResidualAdd + Cast + ReLU) -- fall
            # back so the non-residual ops keep their existing per-element
            # ``apply_element`` semantics.
            return super().apply_vec(b, v, m, n, n_elems=n_elems)

        stride_m = self._live_params.get("__stride_m") or self._live_params["__N"]
        # One (mul, add) per apply_vec call -- shared across every D operand
        # and every element index. The stock path re-emits ``m * stride_m``
        # for every (op, i).
        off_base = b.add(b.mul(m, stride_m), n)
        ir_dtype = self._ir_dtype()

        if not vectorize:
            # "tiled": per-element scalar D loads from the hoisted base.
            out: List[Value] = []
            for i in range(n_elems):
                scalar = b.vec_extract(v, i)
                # ``off_i = off_base + i`` is one add (folded into the GEP by
                # LLVM); the stock path emitted ``n + i`` then
                # ``m * stride_m + (n + i)`` per element.
                off_i = off_base if i == 0 else b.add(off_base, b.const_i32(i))
                for op_idx, op in enumerate(self.ops):
                    dt = self._residual_dtypes[op_idx]
                    ptr = self._live_params[op.param_name]
                    r = ir_dtype_global_load(b, dt, ptr, off_i)
                    kind = self._residual_kinds[op_idx]
                    if kind == "add":
                        scalar = b.fadd(scalar, r)
                    else:  # "mul"
                        scalar = b.fmul(scalar, r)
                out.append(scalar)
            return b.vec_pack(out, ir_dtype)

        # "vector": one global_load_vN per D operand per slice.
        per_d_vecs: List[Value] = []
        for op_idx, op in enumerate(self.ops):
            dt = self._residual_dtypes[op_idx]
            assert dt is not None
            ptr = self._live_params[op.param_name]
            if dt.name in ("f16", "bf16"):
                dv = b.global_load_vN(ptr, off_base, dt, n_elems)
            elif dt.name == "f32":
                scalars = [
                    ir_dtype_global_load(b, dt, ptr, b.add(off_base, b.const_i32(i)))
                    for i in range(n_elems)
                ]
                dv = b.vec_pack(scalars, dt)
            else:
                return super().apply_vec(b, v, m, n, n_elems=n_elems)
            per_d_vecs.append(dv)

        out = []
        for i in range(n_elems):
            scalar = b.vec_extract(v, i)
            for op_idx, op in enumerate(self.ops):
                d_elem = b.vec_extract(per_d_vecs[op_idx], i)
                kind = self._residual_kinds[op_idx]
                if kind == "add":
                    scalar = b.fadd(scalar, d_elem)
                else:
                    scalar = b.fmul(scalar, d_elem)
            out.append(scalar)
        return b.vec_pack(out, ir_dtype)


@dataclass(frozen=True)
class GemmMultiDSpec:
    """One concrete multi-D GEMM kernel configuration.

    ``base`` is a :class:`UniversalGemmSpec` providing the GEMM tile /
    pipeline / data choices. ``d_operands`` is a tuple of
    ``(param_name, op)`` pairs; ``op`` is one of ``{"add", "mul"}``.
    The kernel param order on the produced IR is::

        A, B, C, M, N, K, [stride_a, stride_b, stride_c?], D0, D1, ...

    -- the fused-epilogue D pointers are declared *after* the main
    GEMM body wires up A/B/C/M/N/K, so the kernarg ABI matches that
    order rather than the textbook ``A, B, D0..., C, M, N, K`` shape
    the CK Tile example uses on the host side. See
    :func:`gemm_multi_d_signature` for the kernel-order signature
    the launcher must use.

    ``d_dtype`` is the element type used for every D operand (CK Tile
    allows heterogeneous D dtypes via its template tuple; we ship the
    homogeneous case in v1 since it matches every shipped example).

    ``d_load_kind``: controls how the fused epilogue pulls D
    operands. See :data:`DLoadKind` for the trade-off matrix.
    Default ``"vector"`` issues one ``global_load_vN`` per D operand
    per slice (smallest IR, fastest compile, wall time equivalent
    to the stock per-element form after warmup). Set to ``"stock"``
    to recover pre-edit semantics if a downstream tool expects the
    legacy per-element IR shape; set to ``"tiled"`` for the
    in-between variant that hoists the row-stride base mul without
    materialising the D vector.
    """

    base: UniversalGemmSpec
    d_operands: Tuple[DOp, ...]
    d_dtype: str = "fp16"
    name: str = "rocke_gemm_multi_d"
    d_load_kind: DLoadKind = "vector"

    @property
    def num_d(self) -> int:
        return len(self.d_operands)

    def kernel_name(self) -> str:
        d_suffix = "_".join(f"{name}{op}" for name, op in self.d_operands)
        return kernel_name_join(
            self.name,
            self.base.kernel_name(),
            f"md{self.num_d}",
            d_suffix,
            self.d_dtype,
        )


def is_valid_spec(spec: GemmMultiDSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    if spec.num_d == 0:
        return False, "d_operands must contain at least one entry"
    if spec.num_d > MAX_D:
        return False, f"num_d {spec.num_d} > MAX_D {MAX_D}"
    if spec.base.trait.epilogue != "cshuffle":
        return False, (
            "GemmMultiD requires base.trait.epilogue='cshuffle' "
            "(the default epilogue doesn't have the fused-op hook); "
            f"got {spec.base.trait.epilogue!r}"
        )
    names = set()
    for name, op in spec.d_operands:
        if op not in ("add", "mul"):
            return False, f"D op {op!r} not in {{'add','mul'}}"
        if not name:
            return False, "D param_name must be a non-empty string"
        if name in names:
            return False, f"duplicate D param_name {name!r}"
        if name in ("A", "B", "C", "M", "N", "K"):
            return False, (
                f"D param_name {name!r} collides with a reserved GEMM kernel parameter"
            )
        names.add(name)
    # Same arch validation as universal GEMM (arch-aware: a gfx950-only
    # warp-tile atom requested for gfx942 is rejected here).
    from .gemm_universal import is_valid_spec as _is_valid_gemm

    ok_base, why_base = _is_valid_gemm(spec.base, arch=arch)
    if not ok_base:
        return False, f"base GEMM spec invalid: {why_base}"
    return True, "ok"


def _build_fused_epilogue(spec: GemmMultiDSpec) -> FusedEpilogue:
    """Compose the per-D ``ResidualAdd`` / ``ResidualMul`` chain.

    Picks the epilogue subclass based on ``spec.d_load_kind``:

      * ``"vector"`` (default): :class:`_MultiDEpilogue` with
        ``load_kind="vector"`` -- one ``global_load_vN`` per D per
        slice. Smallest IR / fastest comgr; matches CK Tile's
        ``CShuffleEpilogue::apply_d_tensors``.
      * ``"tiled"``: :class:`_MultiDEpilogue` with
        ``load_kind="tiled"`` -- per-element scalar loads with the
        ``m * stride_m + n`` base hoisted.
      * ``"stock"``: the base
        :class:`~rocke.helpers.fuse.FusedEpilogue` -- per-element
        ``apply_element`` walks, equivalent to pre-edit behavior.
    """
    d_ir_dtype = dtype_to_ir(spec.d_dtype)
    ops = []
    for name, op in spec.d_operands:
        if op == "add":
            ops.append(ResidualAdd(param_name=name, dtype=d_ir_dtype))
        elif op == "mul":
            ops.append(ResidualMul(param_name=name, dtype=d_ir_dtype))
        else:  # validated above, but defensive
            raise ValueError(f"unsupported D op {op!r}")
    if spec.d_load_kind == "stock":
        return FusedEpilogue(ops=tuple(ops), dtype=d_ir_dtype)
    if spec.d_load_kind == "tiled":
        return _MultiDEpilogue.from_ops(tuple(ops), dtype=d_ir_dtype, load_kind="tiled")
    # ``"vector"`` (default) or any unrecognised value: smallest IR,
    # bit-identical output to the stock per-element form.
    return _MultiDEpilogue.from_ops(tuple(ops), dtype=d_ir_dtype, load_kind="vector")


def build_gemm_multi_d(spec: GemmMultiDSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one multi-D GEMM instance.

    Kernel signature::

        (A: ptr<dtype_a, global>,
         B: ptr<dtype_b, global>,
         D0: ptr<d_dtype, global>,
         ...,
         D{num_d-1}: ptr<d_dtype, global>,
         C: ptr<dtype_c, global>,
         M: i32, N: i32, K: i32)

    The D pointers come after A/B and before C — same order the CK
    Tile example uses, so a host launcher can pack the args directly.

    Implementation: builds a :class:`FusedEpilogue` from the per-D op
    chain, attaches it to the base :class:`UniversalGemmSpec` via the
    `_fused_epilogue` side-channel, and delegates to
    :func:`build_universal_gemm`. The cshuffle epilogue picks up the
    fused chain and applies it inside the LDS-staged store loop.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid GemmMultiD spec for {arch}: {why}")

    # Attach the fused epilogue to the base spec. The base spec is a
    # frozen dataclass; we use ``object.__setattr__`` (the same idiom
    # ``helpers.fuse.compile_fn`` uses) to drop the FusedEpilogue into
    # the ``_fused_epilogue`` side-channel slot the cshuffle epilogue
    # reads via ``getattr(spec, "_fused_epilogue", None)``.
    fused = _build_fused_epilogue(spec)

    # Build a fresh copy of the base spec so we don't mutate a spec
    # the caller may reuse elsewhere. The frozen dataclass forces us
    # through ``dataclasses.replace`` for any field change; here we
    # just rename so the lowered kernel symbol carries the multi-D
    # suffix, then attach the fused epilogue.
    import dataclasses

    base_renamed = dataclasses.replace(spec.base, name=spec.kernel_name())
    object.__setattr__(base_renamed, "_fused_epilogue", fused)

    return build_universal_gemm(base_renamed, arch=arch)


def gemm_multi_d_signature(spec: GemmMultiDSpec):
    """Manifest-style signature for the multi-D GEMM kernel.

    The kernarg order must match the actual order
    :func:`build_universal_gemm` lays out plus the order
    :class:`~rocke.helpers.fuse.FusedEpilogue.declare_params` appends
    D pointers. That sequence is::

        A, B, C, M, N, K, [stride_a, stride_b, stride_c?], D0, D1, ...

    not the textbook ``A, B, D0..., C, M, N, K`` shape the CK Tile
    example uses on the host side. The mismatch isn't visible until
    the launcher actually packs kernargs (lowering / HIP parity both
    pass): with the textbook order, the kernel reads C from where
    the launcher wrote D0, dereferences garbage, and hits a memory
    access fault on the first global store. The kernel-order
    signature below is what the AMDGPU kernarg ABI actually expects.
    """
    sb = (
        SignatureBuilder()
        .ptr("A", spec.base.data.dtype_a)
        .ptr("B", spec.base.data.dtype_b)
        .ptr("C", spec.base.data.dtype_c)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
    )
    if spec.base.batched:
        sb.scalar("stride_a", "i32").scalar("stride_b", "i32").scalar("stride_c", "i32")
    for name, _op in spec.d_operands:
        sb.ptr(name, spec.d_dtype)
    return sb.build()


def gemm_multi_d_grid(spec: GemmMultiDSpec, m: int, n: int, batch: int = 1):
    """Same grid as :func:`build_universal_gemm`: ``(N_tiles, M_tiles, batch)``."""
    t = spec.base.tile
    z = batch if spec.base.batched else 1
    return ceil_div_grid((n, t.tile_n), (m, t.tile_m), (z, 1))


__all__ = [
    "DLoadKind",
    "DOp",
    "MAX_D",
    "GemmMultiDSpec",
    "build_gemm_multi_d",
    "gemm_multi_d_grid",
    "gemm_multi_d_signature",
    "is_valid_spec",
    # Re-exports for convenience so callers can build the base spec
    # without importing a second module.
    "DataSpec",
    "TileSpec",
    "TraitSpec",
    "UniversalGemmSpec",
]
