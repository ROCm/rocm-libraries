# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Graph-level fusion driver.

A minimal but complete pipeline that takes a Python callable, identifies
a known fusion pattern (currently: ``matmul -> bias-add -> activation``),
and lowers it to a single CK DSL kernel via the existing
:class:`UniversalGemmSpec` plus a small "fused epilogue" hook.

Architecture (this is the framework you'd extend to match Triton-via-Inductor):

    Python fn
        │
        ▼
    [Graph capture]  ──── torch.fx.symbolic_trace
        │
        ▼
    [Pattern match]  ──── ``_PATTERN_TABLE`` (extensible)
        │
        ▼
    [Lower to CK DSL spec]  ──── ``FusionLowering.lower_<pattern>``
        │
        ▼
    [Autotuner.select() + launch]  ──── from :mod:`helpers.autotune`

Each fusion pattern produces:

  * a base :class:`UniversalGemmSpec` (the matmul shell)
  * a :class:`FusedEpilogue` describing the post-cshuffle op chain
  * an ``args_builder`` that knows how to marshal the user's tensors
    into the kernel's ``{A, B, bias, C, M, N, K}`` arg dict.

Adding a new pattern means:

  1. Extending :class:`FusedEpilogue` with new :class:`EpilogueOp`
     subclasses (e.g. ``Scale``, ``GELU``, ``Dropout``).
  2. Adding an entry to ``_PATTERN_TABLE`` that walks the FX graph
     and returns the spec + epilogue + args_builder.

This proves the architecture works end-to-end. Scaling it up to full
Inductor parity requires:

  * Many more patterns (attention QKV, layernorm, RMS norm, …).
  * A *scheduler* that picks fusion boundaries when many patterns
    could apply (the hardest part — Inductor inherits this from its
    cost model).
  * Buffer-reuse / memory-planning for intermediates that DO escape
    the fusion (Inductor's :class:`SchedulerNode` machinery).
  * Cross-kernel scheduling (we'd issue 2+ kernels with shared
    workspace, and our :class:`Autotuner` would tune each
    independently — already supported).

For now this module is the *proof* that none of these require new
infrastructure — only patterns and a scheduler on top of what's here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.ir import BF16, F16, F32, IRBuilder, PtrType, Type, Value
from .autotune import Autotuner, AutotuneConfig


__all__ = [
    "EpilogueOp",
    "BiasAdd",
    "Cast",
    "Clamp",
    "GELU",
    "ReLU",
    "ResidualAdd",
    "ResidualMul",
    "Scale",
    "SiLU",
    "FusedEpilogue",
    "FusionMatchError",
    "FusionPlan",
    "compile_fn",
    "explain_fn",
    "fuse_matmul_bias_relu",
    "dtype_to_ir",
    "ir_dtype_zero",
    "ir_dtype_const",
    "ir_dtype_global_load",
]


# ============================================================
# Dtype dispatch
# ============================================================
#
# Every helper that lives between the GEMM output and the global
# store has to know its element type. The CK DSL IR builder has
# type-specific entry points (``global_load_f16``,
# ``global_load_bf16``, …); the fusion layer abstracts them behind
# the four functions below so an :class:`EpilogueOp` author writes
# ``ir_dtype_global_load(b, dtype, ptr, idx)`` once and gets the
# right scalar load regardless of whether the surrounding kernel is
# fp16, bf16, or (eventually) fp8 / i8.

_DTYPE_STR_TO_IR = {
    "fp16": F16,
    "f16": F16,
    "half": F16,
    "float16": F16,  # torch.float16 -> "torch.float16"
    "bf16": BF16,
    "bfloat16": BF16,
    "fp32": F32,
    "f32": F32,
    "float": F32,
    "float32": F32,  # torch.float32
}


def dtype_to_ir(dtype) -> Type:
    """Resolve a Python-side dtype description to an IR ``Type``.

    Accepts the canonical CK Tile strings (``"fp16"``, ``"bf16"``,
    ``"fp32"``), their aliases, or an IR ``Type`` directly. Used by
    :class:`FusedEpilogue` and :class:`EpilogueOp` subclasses so the
    fusion layer doesn't grow N copies of every helper for every
    dtype.
    """
    if isinstance(dtype, Type):
        return dtype
    s = str(dtype).lower()
    if s.startswith("torch."):
        # Pure-python access path for `torch.float16` / `torch.bfloat16`
        # etc., without importing torch eagerly.
        s = s.split(".", 1)[-1]
    if s in _DTYPE_STR_TO_IR:
        return _DTYPE_STR_TO_IR[s]
    raise ValueError(f"unsupported epilogue dtype {dtype!r}")


def ir_dtype_zero(b: IRBuilder, dtype: Type) -> Value:
    """Element zero in the given type."""
    if dtype == F16:
        return b.trunc_f32_to_f16(b.const_f32(0.0))
    if dtype == BF16:
        # Reuse the f32 trunc helper if a bf16 path exists; otherwise
        # fall back to a bitcast through i16(0). Both lower to a
        # ``mov 0`` on AMDGPU.
        try:
            return b.trunc_f32_to_bf16(b.const_f32(0.0))
        except AttributeError:
            # Generic path: const_f32(0.0) -> cast_f32_to(bf16).
            return b.cast_f32_to(b.const_f32(0.0), BF16)
    if dtype == F32:
        return b.const_f32(0.0)
    raise NotImplementedError(f"ir_dtype_zero: unsupported {dtype}")


def ir_dtype_const(b: IRBuilder, dtype: Type, value: float) -> Value:
    """Per-element constant in the given type."""
    if dtype == F16:
        return b.trunc_f32_to_f16(b.const_f32(float(value)))
    if dtype == BF16:
        try:
            return b.trunc_f32_to_bf16(b.const_f32(float(value)))
        except AttributeError:
            return b.cast_f32_to(b.const_f32(float(value)), BF16)
    if dtype == F32:
        return b.const_f32(float(value))
    raise NotImplementedError(f"ir_dtype_const: unsupported {dtype}")


def ir_dtype_global_load(b: IRBuilder, dtype: Type, ptr: Value, idx: Value) -> Value:
    """Single-element global load dispatched on ``dtype``."""
    if dtype == F16:
        return b.global_load_f16(ptr, idx)
    if dtype == BF16:
        return b.global_load_bf16(ptr, idx)
    if dtype == F32:
        return b.global_load_f32(ptr, idx)
    raise NotImplementedError(f"ir_dtype_global_load: unsupported {dtype}")


# ============================================================
# Epilogue op IR (small, scalar-element-level for portability)
# ============================================================


class EpilogueOp:
    """One step in a fused post-cshuffle op chain.

    Each op operates *element-wise* on the loaded output vector that
    would otherwise be stored directly to global memory. Ops are
    composed left-to-right.

    Subclasses override:

      * :meth:`declare_params` — registers any extra kernel params
        (e.g. a ``bias`` pointer). Called once at kernel-build time.
      * :meth:`apply_element` — transforms one scalar element.
        Called per element, per lane, per output position.
      * :meth:`tag` — short label used in kernel-name suffix and the
        autotune cache key.

    Element-level (rather than vector-level) lets us reuse just the
    primitive scalar IR ops (``fadd``, ``fmax``, ``fmul``) without
    plumbing vector arithmetic into the IR builder.
    """

    def declare_params(self, b: IRBuilder) -> Dict[str, Value]:
        return {}

    def apply_element(
        self,
        b: IRBuilder,
        v: Value,
        *,
        m: Value,
        n: Value,
        elem_idx: int,
        params: Dict[str, Value],
    ) -> Value:
        raise NotImplementedError

    def tag(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BiasAdd(EpilogueOp):
    """Per-N bias: out[m, n] += bias[n].

    ``bias`` is a 1D tensor of length N in the same element type as
    the GEMM output (``dtype``). The pointer comes from a new kernel
    param; the same value is broadcast across every M-row.
    """

    param_name: str = "bias"
    dtype: Any = F16  # IR Type or any input accepted by ``dtype_to_ir``

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def declare_params(self, b: IRBuilder) -> Dict[str, Value]:
        p = b.param(
            self.param_name,
            PtrType(self._ir_dtype(), "global"),
            noalias=True,
            readonly=True,
            align=16,
        )
        return {self.param_name: p}

    def apply_element(
        self,
        b: IRBuilder,
        v: Value,
        *,
        m: Value,
        n: Value,
        elem_idx: int,
        params: Dict[str, Value],
    ) -> Value:
        bias_ptr = params[self.param_name]
        # Scalar global load of one element at offset ``n + elem_idx``.
        # The AMDGPU backend coalesces these into a single buffer
        # load if the vector ends up at a contiguous N stride.
        n_idx = b.add(n, b.const_i32(elem_idx))
        bv = ir_dtype_global_load(b, self._ir_dtype(), bias_ptr, n_idx)
        return b.fadd(v, bv)

    def tag(self) -> str:
        return f"bias_{self._ir_dtype().name}"


@dataclass(frozen=True)
class ReLU(EpilogueOp):
    """Element-wise ``max(v, 0)``. No extra params."""

    dtype: Any = F16

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def apply_element(self, b, v, *, m, n, elem_idx, params):
        return b.fmax(v, ir_dtype_zero(b, self._ir_dtype()))

    def tag(self) -> str:
        return f"relu_{self._ir_dtype().name}"


@dataclass(frozen=True)
class Scale(EpilogueOp):
    """Element-wise multiply by a scalar constant baked into the kernel."""

    scale: float
    dtype: Any = F16

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def apply_element(self, b, v, *, m, n, elem_idx, params):
        return b.fmul(v, ir_dtype_const(b, self._ir_dtype(), self.scale))

    def tag(self) -> str:
        return f"scale{self.scale:g}_{self._ir_dtype().name}"


def _ir_unary_via_f32(b: IRBuilder, dtype: Type, v: Value, fn) -> Value:
    """Promote ``v`` to f32, apply ``fn(b, v_f32)``, cast back to ``dtype``.

    Most transcendental epilogue activations (gelu, silu, tanh) need
    f32 internally for both accuracy and AMDGPU lowering. The
    f16/bf16 IR builders do not expose `exp2` / `tanh` on narrow
    types, so we round-trip via f32.
    """
    if dtype == F32:
        return fn(b, v)
    v32 = b.cast_to_f32(v)
    r32 = fn(b, v32)
    return b.cast_f32_to(r32, dtype)


def _sigmoid_via_exp2(b: IRBuilder, x: Value) -> Value:
    """1 / (1 + e^-x), implemented via exp2.

    ``exp(-x) = exp2(-x * log2(e))``. Matches the formula used in the
    ``elementwise`` instance and avoids ``math.exp`` (which the AMDGPU
    backend does not lower on its own).
    """
    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one = b.const_f32(1.0)
    return b.rcp(b.fadd(one, b.exp2(b.fmul(c_neg_log2e, x))))


def _tanh_via_exp2(b: IRBuilder, x: Value) -> Value:
    """tanh(x) = (e^{2x} - 1) / (e^{2x} + 1).

    Same primitive set the ``elementwise`` instance uses for
    ``gelu_tanh``; deliberately avoids ``math.tanh`` to keep the
    fused-epilogue body AMDGPU-lowerable without extra runtime
    libraries.
    """
    c_2log2e = b.const_f32(2.0 * 1.4426950408889634)
    one = b.const_f32(1.0)
    e2x = b.exp2(b.fmul(c_2log2e, x))
    return b.fmul(b.fsub(e2x, one), b.rcp(b.fadd(e2x, one)))


@dataclass(frozen=True)
class GELU(EpilogueOp):
    """GELU (tanh approximation): ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.

    The tanh-based approximation matches the formula PyTorch's
    ``gelu(approximate="tanh")`` uses, which is also the default for
    most transformer epilogues. Stays in f32 internally for
    accuracy, then casts back to the surrounding ``dtype``.
    """

    dtype: Any = F16

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def apply_element(self, b, v, *, m, n, elem_idx, params):
        def _gelu_f32(b: IRBuilder, x: Value) -> Value:
            c_half = b.const_f32(0.5)
            c_one = b.const_f32(1.0)
            c_sq2_over_pi = b.const_f32(0.7978845608028654)
            c_a = b.const_f32(0.044715)
            x3 = b.fmul(b.fmul(x, x), x)
            inner = b.fmul(c_sq2_over_pi, b.fadd(x, b.fmul(c_a, x3)))
            return b.fmul(b.fmul(c_half, x), b.fadd(c_one, _tanh_via_exp2(b, inner)))

        return _ir_unary_via_f32(b, self._ir_dtype(), v, _gelu_f32)

    def tag(self) -> str:
        return f"gelu_{self._ir_dtype().name}"


@dataclass(frozen=True)
class SiLU(EpilogueOp):
    """SiLU (aka swish): ``x * sigmoid(x)``.

    Implemented via the exp2-based sigmoid to keep the AMDGPU lower
    chain simple. Inputs/outputs share ``dtype``; the math is done
    in f32 for accuracy.
    """

    dtype: Any = F16

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def apply_element(self, b, v, *, m, n, elem_idx, params):
        def _silu_f32(b: IRBuilder, x: Value) -> Value:
            return b.fmul(x, _sigmoid_via_exp2(b, x))

        return _ir_unary_via_f32(b, self._ir_dtype(), v, _silu_f32)

    def tag(self) -> str:
        return f"silu_{self._ir_dtype().name}"


@dataclass(frozen=True)
class Clamp(EpilogueOp):
    """Element-wise ``min(max(v, lo), hi)``.

    ``lo`` and ``hi`` are scalar floats baked into the kernel. Useful
    for activations like ``relu6`` (``lo=0, hi=6``) or to clamp logits
    before a downstream cast.
    """

    lo: float
    hi: float
    dtype: Any = F16

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def apply_element(self, b, v, *, m, n, elem_idx, params):
        d = self._ir_dtype()
        lo = ir_dtype_const(b, d, self.lo)
        hi = ir_dtype_const(b, d, self.hi)
        return b.fmin(b.fmax(v, lo), hi)

    def tag(self) -> str:
        return f"clamp{self.lo:g}_{self.hi:g}_{self._ir_dtype().name}"


@dataclass(frozen=True)
class Cast(EpilogueOp):
    """Cast the streaming value to a new output ``dtype``.

    Only the ``apply_element`` return value is in the new dtype; the
    surrounding :class:`FusedEpilogue` is responsible for stitching
    the result back into a vector through :meth:`FusedEpilogue.apply_vec`.
    Place :class:`Cast` last in the chain so subsequent ops still see
    the original element type.
    """

    src_dtype: Any = F16
    dst_dtype: Any = F16

    def _src(self) -> Type:
        return dtype_to_ir(self.src_dtype)

    def _dst(self) -> Type:
        return dtype_to_ir(self.dst_dtype)

    def apply_element(self, b, v, *, m, n, elem_idx, params):
        src = self._src()
        dst = self._dst()
        if src == dst:
            return v
        if src == F32:
            return b.cast_f32_to(v, dst)
        if dst == F32:
            return b.cast_to_f32(v)
        # Narrow-to-narrow: go through f32 to keep the IR builder simple.
        return b.cast_f32_to(b.cast_to_f32(v), dst)

    def tag(self) -> str:
        return f"cast_{self._src().name}_to_{self._dst().name}"


@dataclass(frozen=True)
class ResidualAdd(EpilogueOp):
    """Element-wise ``out += residual[m, n]``.

    ``residual`` is a full ``(M, N)`` tensor passed as an extra kernel
    parameter; the per-element load uses the GEMM's running ``(m, n)``
    indices to compute a global offset.

    ``stride_m`` defaults to ``N`` (set at kernel build time via
    :meth:`FusedEpilogue.with_runtime`); pass an explicit value if the
    residual is non-contiguous. The default contiguous mode keeps the
    fast path tight: one scalar global load per element, which the
    AMDGPU backend coalesces back into a vector load when adjacent
    elements share a vector store.
    """

    param_name: str = "residual"
    dtype: Any = F16

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def declare_params(self, b: IRBuilder) -> Dict[str, Value]:
        p = b.param(
            self.param_name,
            PtrType(self._ir_dtype(), "global"),
            noalias=True,
            readonly=True,
            align=16,
        )
        # The residual stride along M is the kernel's existing ``N``
        # param. We re-lookup the param after the declare to thread
        # the SSA value through to ``apply_element``.
        return {self.param_name: p}

    def apply_element(
        self,
        b: IRBuilder,
        v: Value,
        *,
        m: Value,
        n: Value,
        elem_idx: int,
        params: Dict[str, Value],
    ) -> Value:
        ptr = params[self.param_name]
        n_idx = b.add(n, b.const_i32(elem_idx))
        # ``params['__stride_m']`` is injected by ``FusedEpilogue`` once
        # the kernel knows N at runtime. The fallback below uses
        # ``params['__N']`` when the runtime stride wasn't recorded
        # (smaller code, requires contiguous residual).
        stride_m = params.get("__stride_m")
        if stride_m is None:
            stride_m = params["__N"]
        off = b.add(b.mul(m, stride_m), n_idx)
        r = ir_dtype_global_load(b, self._ir_dtype(), ptr, off)
        return b.fadd(v, r)

    def tag(self) -> str:
        return f"resadd_{self._ir_dtype().name}"


@dataclass(frozen=True)
class ResidualMul(EpilogueOp):
    """Element-wise ``out *= residual[m, n]``.

    Same layout/contract as :class:`ResidualAdd` but with a multiply
    instead of an add. Useful for gated activations (``silu(x) * v``)
    where ``v`` is the second projection of an MLP.
    """

    param_name: str = "residual_mul"
    dtype: Any = F16

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def declare_params(self, b: IRBuilder) -> Dict[str, Value]:
        p = b.param(
            self.param_name,
            PtrType(self._ir_dtype(), "global"),
            noalias=True,
            readonly=True,
            align=16,
        )
        return {self.param_name: p}

    def apply_element(
        self,
        b: IRBuilder,
        v: Value,
        *,
        m: Value,
        n: Value,
        elem_idx: int,
        params: Dict[str, Value],
    ) -> Value:
        ptr = params[self.param_name]
        n_idx = b.add(n, b.const_i32(elem_idx))
        stride_m = params.get("__stride_m") or params["__N"]
        off = b.add(b.mul(m, stride_m), n_idx)
        r = ir_dtype_global_load(b, self._ir_dtype(), ptr, off)
        return b.fmul(v, r)

    def tag(self) -> str:
        return f"resmul_{self._ir_dtype().name}"


@dataclass(frozen=True)
class FusedEpilogue:
    """A chain of :class:`EpilogueOp`'s applied after the cshuffle LDS read.

    Hooked into :func:`_emit_epilogue_cshuffle` via the kernel spec's
    ``_fused_epilogue`` attribute. Provides both vector- and scalar-
    entry points; ``apply_vec`` is what the GEMM kernel calls.

    ``dtype`` is the element type of the values streaming through the
    epilogue (matches the GEMM ``C`` output). Each contained op
    inherits it via :meth:`with_dtype` when the user calls
    :meth:`compile_fn`; the default is fp16 for ergonomic Triton-style
    use.
    """

    ops: Tuple[EpilogueOp, ...]
    dtype: Any = F16
    # Populated lazily during kernel build: maps param_name -> SSA value.
    _live_params: Dict[str, Value] = field(default_factory=dict, repr=False)

    def _ir_dtype(self) -> Type:
        return dtype_to_ir(self.dtype)

    def with_dtype(self, dtype) -> "FusedEpilogue":
        """Return a copy of this epilogue with ``dtype`` propagated to
        every contained op (via ``dataclasses.replace`` where the op's
        dataclass exposes a ``dtype`` field; non-dtype ops pass
        through unchanged).
        """
        new_ops = []
        for op in self.ops:
            if dataclasses.is_dataclass(op) and "dtype" in {
                f.name for f in dataclasses.fields(op)
            }:
                new_ops.append(dataclasses.replace(op, dtype=dtype))
            else:
                new_ops.append(op)
        return FusedEpilogue(ops=tuple(new_ops), dtype=dtype)

    def declare_params(self, b: IRBuilder) -> Dict[str, Value]:
        """Walk every op's declare_params, accumulating the new params.

        Idempotent: subsequent calls return the same SSA values from
        the in-builder cache.
        """
        for op in self.ops:
            for name, val in op.declare_params(b).items():
                self._live_params[name] = val
        return dict(self._live_params)

    def record_runtime(
        self, b: IRBuilder, *, N: Value, stride_m: Optional[Value] = None
    ) -> None:
        """Record runtime values that residual-style ops need.

        ``N`` is the GEMM's runtime N dimension; residual ops use it
        as the M-stride for a contiguous ``(M, N)`` residual unless
        an explicit ``stride_m`` is provided. Call this once per
        kernel build, after :meth:`declare_params`, so the same
        :class:`FusedEpilogue` instance can be reused across many
        kernels at different shapes (each rebuild rebinds the SSA
        values to the new kernel's params).
        """
        self._live_params["__N"] = N
        if stride_m is not None:
            self._live_params["__stride_m"] = stride_m

    def apply_vec(
        self,
        b: IRBuilder,
        v: Value,
        m: Value,
        n: Value,
        *,
        n_elems: int,
    ) -> Value:
        """Apply the op chain element-wise to an N-vector.

        Re-packs the result so the GEMM kernel can store it via the
        same ``buffer_store_vN_<dtype>`` it would have used otherwise.
        """
        out = []
        for i in range(n_elems):
            scalar = b.vec_extract(v, i)
            for op in self.ops:
                scalar = op.apply_element(
                    b,
                    scalar,
                    m=m,
                    n=n,
                    elem_idx=i,
                    params=self._live_params,
                )
            out.append(scalar)
        return b.vec_pack(out, self._ir_dtype())

    def apply_scalar(self, b: IRBuilder, v: Value, m: Value, n: Value) -> Value:
        for op in self.ops:
            v = op.apply_element(
                b,
                v,
                m=m,
                n=n,
                elem_idx=0,
                params=self._live_params,
            )
        return v

    def kernel_name_suffix(self) -> str:
        return "_".join(op.tag() for op in self.ops) or "id"


# Convenience names that pattern matchers use.
def fuse_matmul_bias_relu(dtype=F16) -> FusedEpilogue:
    return FusedEpilogue(ops=(BiasAdd(dtype=dtype), ReLU(dtype=dtype)), dtype=dtype)


# ============================================================
# torch.fx graph-capture + pattern matcher (extensible)
# ============================================================


class FusionMatchError(RuntimeError):
    """Raised when no known pattern matches the traced graph."""


@dataclass(frozen=True)
class FusionPlan:
    """A normalized fusion decision.

    This is the small, explainable bridge between graph capture
    (FX nodes) and the typed epilogue / lowerer system. The plan is
    deliberately flat so it round-trips cleanly to JSON for the
    fusion-explainer (``explain_fn``).

    Attributes
    ----------
    pattern
        Short stable label identifying the pattern family (e.g.
        ``"matmul_bias_relu"`` or ``"pointwise_add"``). The lowering
        registry uses the prefix (``matmul`` / ``pointwise`` /
        ``reduce``) to pick a lowerer.
    a_arg_name, b_arg_name
        Names of the primary input placeholders. For pointwise / reduce
        plans ``b_arg_name`` may be an empty string.
    bias_arg_name
        Optional bias-style placeholder (1D, broadcast across M).
    residual_arg_names
        Names of any full (M, N) residual placeholders that participate
        in the epilogue (e.g. ``mm + bias + residual``).
    epilogue_template
        Ordered chain of :class:`EpilogueOp` instances, with each op's
        dtype updated lazily by :meth:`compile_fn`.
    explanation
        Free-form English explanation suitable for ``explain_fn``.
    extra_attrs
        Lowerer-specific attributes (e.g. ``unary_chain`` for
        pointwise plans, ``reduce_op`` for reductions).
    """

    pattern: str
    a_arg_name: str
    b_arg_name: str
    bias_arg_name: Optional[str]
    epilogue_template: FusedEpilogue
    explanation: str
    residual_arg_names: Tuple[str, ...] = ()
    extra_attrs: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "a_arg_name": self.a_arg_name,
            "b_arg_name": self.b_arg_name,
            "bias_arg_name": self.bias_arg_name,
            "residual_arg_names": list(self.residual_arg_names),
            "epilogue_ops": [op.tag() for op in self.epilogue_template.ops],
            "explanation": self.explanation,
            "extra_attrs": dict(self.extra_attrs),
        }


def _try_import_torch_fx():
    """Lazy import so non-torch users can still import this module."""
    try:
        import torch
        import torch.fx

        return torch, torch.fx
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(f"compile_fn() requires torch.fx; got: {e}") from e


def _target_name(target) -> str:
    return getattr(target, "__name__", str(target))


def _is_call(node, names: Tuple[str, ...]) -> bool:
    return (
        getattr(node, "op", None) == "call_function"
        and _target_name(node.target) in names
    )


def _placeholder_name(node) -> Optional[str]:
    if getattr(node, "op", None) != "placeholder":
        return None
    return node.target if hasattr(node, "target") else node.name


def _match_matmul_epilogue(graph_module) -> Optional[FusionPlan]:
    """Normalize a family of GEMM-rooted epilogues.

    Each step in the chain after ``matmul`` is one of:

    * ``relu(x)`` -> :class:`ReLU`
    * ``gelu(x)`` (including ``approximate='tanh'``) -> :class:`GELU`
    * ``silu(x)`` / ``F.silu(x)`` -> :class:`SiLU`
    * ``torch.clamp(x, lo, hi)`` -> :class:`Clamp`
    * ``x * scalar`` / ``scalar * x`` -> :class:`Scale`
    * ``x + placeholder`` (1D) -> :class:`BiasAdd`
    * ``x + placeholder`` (2D, same shape as ``x``) -> :class:`ResidualAdd`
    * ``x * placeholder`` (2D, same shape as ``x``) -> :class:`ResidualMul`
    * ``x.to(dtype)`` / ``.float()`` / ``.half()`` -> :class:`Cast`
      (must be the last op in the chain).

    The chain must be single-use and linear: every op consumes the
    previous op's output. This gives broad coverage of the
    ``mm + bias + activation + (optional residual)`` patterns Triton
    routinely fuses while staying predictable and easy to explain.
    """

    nodes = list(graph_module.graph.nodes)
    output_nodes = [n for n in nodes if n.op == "output"]
    if len(output_nodes) != 1:
        return None
    root = output_nodes[0].args[0]

    ops_rev: List[EpilogueOp] = []
    bias_name: Optional[str] = None
    residual_names: List[str] = []
    cur = root

    # Walk backward from output to matmul, peeling supported epilogue ops.
    while True:
        # ---- pure activations ----
        if _is_call(cur, ("relu", "F.relu")):
            ops_rev.append(ReLU())
            cur = cur.args[0]
            continue
        if _is_call(cur, ("gelu", "F.gelu")):
            ops_rev.append(GELU())
            cur = cur.args[0]
            continue
        if _is_call(cur, ("silu", "F.silu")):
            ops_rev.append(SiLU())
            cur = cur.args[0]
            continue

        # ---- clamp / clamp_min / clamp_max ----
        if _is_call(cur, ("clamp", "clamp_min", "clamp_max")):
            tgt = _target_name(cur.target)
            args = list(cur.args) + [None, None]
            kwargs = dict(cur.kwargs)
            x = args[0]
            if tgt == "clamp":
                lo = kwargs.get("min", args[1])
                hi = kwargs.get("max", args[2])
            elif tgt == "clamp_min":
                lo = kwargs.get("min", args[1])
                hi = None
            else:
                lo = None
                hi = kwargs.get("max", args[1])
            if not isinstance(lo, (int, float, type(None))) or not isinstance(
                hi, (int, float, type(None))
            ):
                return None
            ops_rev.append(
                Clamp(
                    lo=float(lo) if lo is not None else float("-inf"),
                    hi=float(hi) if hi is not None else float("inf"),
                )
            )
            cur = x
            continue

        # ---- elementwise mul (Scale or ResidualMul) ----
        if (
            _is_call(cur, ("mul", "__mul__"))
            or getattr(cur, "target", None) is operator.mul
        ):
            lhs, rhs = cur.args[0], cur.args[1]
            scalar = (
                rhs
                if isinstance(rhs, (int, float))
                else lhs
                if isinstance(lhs, (int, float))
                else None
            )
            if scalar is not None:
                next_node = lhs if scalar is rhs else rhs
                ops_rev.append(Scale(float(scalar)))
                cur = next_node
                continue
            # Tensor * placeholder -> ResidualMul.
            lhs_ph = _placeholder_name(lhs)
            rhs_ph = _placeholder_name(rhs)
            if lhs_ph and not rhs_ph:
                residual_names.append(lhs_ph)
                ops_rev.append(
                    ResidualMul(param_name=f"residual_mul_{len(residual_names) - 1}")
                )
                cur = rhs
                continue
            if rhs_ph and not lhs_ph:
                residual_names.append(rhs_ph)
                ops_rev.append(
                    ResidualMul(param_name=f"residual_mul_{len(residual_names) - 1}")
                )
                cur = lhs
                continue
            return None

        # ---- elementwise add (Bias or Residual) ----
        if (
            _is_call(cur, ("add", "__add__"))
            or getattr(cur, "target", None) is operator.add
        ):
            lhs, rhs = cur.args[0], cur.args[1]
            lhs_ph = _placeholder_name(lhs)
            rhs_ph = _placeholder_name(rhs)
            if lhs_ph and not rhs_ph:
                # ``bias`` is the placeholder; whether it's 1D vs 2D
                # is decided at runtime (we accept either).
                if bias_name is None:
                    bias_name = lhs_ph
                    ops_rev.append(BiasAdd())
                else:
                    residual_names.append(lhs_ph)
                    ops_rev.append(
                        ResidualAdd(param_name=f"residual_{len(residual_names) - 1}")
                    )
                cur = rhs
                continue
            if rhs_ph and not lhs_ph:
                if bias_name is None:
                    bias_name = rhs_ph
                    ops_rev.append(BiasAdd())
                else:
                    residual_names.append(rhs_ph)
                    ops_rev.append(
                        ResidualAdd(param_name=f"residual_{len(residual_names) - 1}")
                    )
                cur = lhs
                continue
            return None

        break

    if not _is_call(cur, ("matmul", "mm")):
        return None

    a_ph = _placeholder_name(cur.args[0])
    b_ph = _placeholder_name(cur.args[1])
    if not a_ph or not b_ph:
        return None

    # We walked backwards, so reverse to get execution order after GEMM.
    ops = tuple(reversed(ops_rev))
    pattern = "matmul" + (
        "_" + "_".join(op.tag().split("_", 1)[0] for op in ops) if ops else ""
    )
    explanation = (
        "single-output linear GEMM epilogue: "
        f"A={a_ph}, B={b_ph}, bias={bias_name}, residuals={residual_names}, "
        f"ops={[op.tag() for op in ops]}"
    )
    return FusionPlan(
        pattern=pattern,
        a_arg_name=a_ph,
        b_arg_name=b_ph,
        bias_arg_name=bias_name,
        epilogue_template=FusedEpilogue(ops=ops),
        explanation=explanation,
        residual_arg_names=tuple(residual_names),
    )


def _match_pointwise_chain(graph_module) -> Optional[FusionPlan]:
    """Match a chain of unary/binary pointwise ops with no GEMM at the root.

    Supported shapes:

    * Pure unary chain: ``f1(f2(...fn(A)...))`` with each ``fi`` in
      ``{relu, gelu, silu, neg, abs}``.
    * Pure binary chain over two placeholders: ``op(A, B)`` followed by
      optional unaries.

    The match emits a :class:`FusionPlan` with ``pattern="pointwise_*"``
    so the :class:`ElementwiseLowerer` can handle the plan instead of
    the GEMM lowerer. ``a_arg_name`` is the primary input placeholder;
    ``b_arg_name`` is the optional secondary (binary case) and
    ``bias_arg_name`` is unused.
    """

    nodes = list(graph_module.graph.nodes)
    output_nodes = [n for n in nodes if n.op == "output"]
    if len(output_nodes) != 1:
        return None
    root = output_nodes[0].args[0]

    unary_targets = {
        "relu": "relu",
        "F.relu": "relu",
        "gelu": "gelu_tanh",
        "F.gelu": "gelu_tanh",
        "silu": "silu",
        "F.silu": "silu",
        "neg": "neg",
        "__neg__": "neg",
        "abs": "abs",
    }
    binary_targets = {
        "add": "add",
        "__add__": "add",
        "sub": "sub",
        "__sub__": "sub",
        "mul": "mul",
        "__mul__": "mul",
        "max": "max",
        "min": "min",
    }

    chain_tags: List[str] = []
    cur = root
    while True:
        tgt = _target_name(getattr(cur, "target", None))
        if getattr(cur, "op", None) != "call_function":
            break
        if tgt in unary_targets:
            chain_tags.append(unary_targets[tgt])
            cur = cur.args[0]
            continue
        break

    a_ph = b_ph = None
    op_kind: Optional[str] = None

    if getattr(cur, "op", None) == "call_function":
        tgt = _target_name(cur.target)
        # operator.add etc resolve to short names too.
        if hasattr(operator, "add") and getattr(cur, "target", None) is operator.add:
            tgt = "add"
        elif getattr(cur, "target", None) is operator.mul:
            tgt = "mul"
        if tgt in binary_targets:
            lhs, rhs = cur.args[0], cur.args[1]
            l_ph = _placeholder_name(lhs)
            r_ph = _placeholder_name(rhs)
            if l_ph and r_ph:
                a_ph, b_ph = l_ph, r_ph
                op_kind = binary_targets[tgt]
    if a_ph is None:
        ph = _placeholder_name(cur)
        if not ph:
            return None
        a_ph = ph

    chain_tags = list(reversed(chain_tags))
    if op_kind is None and not chain_tags:
        return None
    pattern = "pointwise_" + (
        "_".join([op_kind] + chain_tags) if op_kind else "_".join(chain_tags)
    )
    explanation = (
        f"pointwise chain rooted at placeholder(s) {a_ph}"
        + (f", {b_ph}" if b_ph else "")
        + f": ops={[op_kind] if op_kind else []} -> {chain_tags}"
    )
    # Encode the chain into the FusionPlan as ``extra_attrs``; the
    # ElementwiseLowerer reads it back.
    return FusionPlan(
        pattern=pattern,
        a_arg_name=a_ph,
        b_arg_name=b_ph or "",
        bias_arg_name=None,
        epilogue_template=FusedEpilogue(ops=()),
        explanation=explanation,
        extra_attrs={"op_kind": op_kind, "unary_chain": tuple(chain_tags)},
    )


def _match_rowwise_reduction(graph_module) -> Optional[FusionPlan]:
    """Match ``torch.{sum,mean,max}(A, dim=-1)`` style row reductions.

    Returns a :class:`FusionPlan` whose ``pattern`` is
    ``"reduce_<op>"`` so the :class:`ReductionLowerer` can pick it
    up. The plan carries the reduction op kind in ``extra_attrs``;
    the lowerer turns it into a :class:`Reduce2DSpec`.
    """

    nodes = list(graph_module.graph.nodes)
    output_nodes = [n for n in nodes if n.op == "output"]
    if len(output_nodes) != 1:
        return None
    root = output_nodes[0].args[0]

    reduce_kinds = {
        "sum": "sum",
        "mean": "mean",
        "max": "max",
        "amax": "max",
    }
    tgt = _target_name(getattr(root, "target", None))
    if getattr(root, "op", None) != "call_function" or tgt not in reduce_kinds:
        return None
    if not root.args:
        return None
    x = root.args[0]
    dim = root.kwargs.get("dim", root.args[1] if len(root.args) > 1 else None)
    if dim is not None and dim not in (-1, 1):
        # Only last-dim row reductions are supported by the lowerer.
        return None
    ph = _placeholder_name(x)
    if not ph:
        return None
    op_kind = reduce_kinds[tgt]
    return FusionPlan(
        pattern=f"reduce_{op_kind}",
        a_arg_name=ph,
        b_arg_name="",
        bias_arg_name=None,
        epilogue_template=FusedEpilogue(ops=()),
        explanation=f"rowwise {op_kind} of placeholder {ph}",
        extra_attrs={"reduce_op": op_kind},
    )


_PATTERN_TABLE: List[Callable[[Any], Optional[FusionPlan]]] = [
    _match_matmul_epilogue,
    _match_rowwise_reduction,
    _match_pointwise_chain,
]


# ============================================================
# Compile driver
# ============================================================


def _make_gemm_configs(
    *,
    epilogue: FusedEpilogue,
    chiplet_wgms: Sequence[int] = (4, 8),
    name_prefix: str = "fused",
) -> List[AutotuneConfig]:
    """Build a small but representative tile/chiplet sweep, threading
    the fused epilogue through each spec.
    """
    from ..instances.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    ir_dtype = dtype_to_ir(epilogue.dtype)
    if ir_dtype == BF16:
        data = DataSpec(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32"
        )
        # Universal bf16 GEMM currently supports 16x16x16/32 atoms.
        tile_choices = [
            ("64x64x32", 64, 64, 32, 2, 2, 16, 16, 32),
            ("128x64x32", 128, 64, 32, 4, 2, 16, 16, 32),
            ("64x128x32", 64, 128, 32, 2, 4, 16, 16, 32),
        ]
    else:
        data = DataSpec(
            dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
        )
        tile_choices = [
            ("128x128x32", 128, 128, 32, 2, 2, 32, 32, 16),
            ("128x128x64", 128, 128, 64, 2, 2, 32, 32, 16),
            ("256x128x32", 256, 128, 32, 4, 2, 32, 32, 16),
        ]

    configs: List[AutotuneConfig] = []
    for tile_name, tm, tn, tk, wm, wn, wtm, wtn, wtk in tile_choices:
        for wgm in chiplet_wgms:
            chiplet = wgm > 0
            spec = UniversalGemmSpec(
                name=f"{name_prefix}_{epilogue.kernel_name_suffix()}_t{tm}x{tn}x{tk}_wgm{wgm}",
                tile=TileSpec(
                    tile_m=tm,
                    tile_n=tn,
                    tile_k=tk,
                    warp_m=wm,
                    warp_n=wn,
                    warp_k=1,
                    warp_tile_m=wtm,
                    warp_tile_n=wtn,
                    warp_tile_k=wtk,
                ),
                trait=TraitSpec(
                    pipeline="compv4",
                    epilogue="cshuffle",
                    chiplet_swizzle=chiplet,
                    chiplet_wgm=wgm,
                ),
                data=data,
            )
            # Attach the fused epilogue via a private side-channel attr.
            # Frozen dataclass: use ``object.__setattr__`` to bypass.
            object.__setattr__(spec, "_fused_epilogue", epilogue)
            configs.append(
                AutotuneConfig(
                    spec=spec,
                    name=f"{tile_name}_wgm{wgm}" + ("_chiplet" if chiplet else ""),
                )
            )
    return configs


def compile_fn(
    fn,
    *,
    cache_path: Optional[str] = None,
    verbose: bool = False,
):
    """Trace ``fn``, match a fusion pattern, return a launchable callable.

    Currently supports GEMM plus a linear epilogue chain drawn from
    ``BiasAdd``, ``Scale`` and ``ReLU``. Falls back to
    :class:`FusionMatchError` for any other graph; the user can extend
    :data:`_PATTERN_TABLE` to add more.

    Example::

        @ck_dsl.compile_fn
        def fused(A, B, bias):
            return torch.relu(torch.matmul(A, B) + bias)

        out = fused(A, B, bias)  # one CK DSL kernel, autotuned per shape.
    """
    torch, fx = _try_import_torch_fx()
    traced = fx.symbolic_trace(fn)

    match = None
    for pattern_fn in _PATTERN_TABLE:
        match = pattern_fn(traced)
        if match is not None:
            break
    if match is None:
        raise FusionMatchError(
            f"compile_fn: no known pattern matches the graph of {fn!r}. "
            f"Known patterns: {[p.__name__ for p in _PATTERN_TABLE]}"
        )

    # The dtype is resolved at first-call time (from the input tensors'
    # dtype). The epilogue chain + autotune config set is built lazily
    # per dtype, then cached: a single ``compile_fn`` decoration
    # handles fp16 + bf16 calls interchangeably, with the autotuner
    # storing separate winners per dtype.
    from ..core.lower_llvm import lower_kernel_to_llvm
    from ..runtime.comgr import build_hsaco_from_llvm_ir
    from ..runtime.launcher import (
        KernelLauncher,
        LaunchConfig,
        time_launches,
    )
    from ..instances.gemm_universal import build_universal_gemm
    import functools

    _per_dtype: Dict[Any, Tuple[Autotuner, List[AutotuneConfig]]] = {}

    @functools.lru_cache(maxsize=None)
    def _build_launcher(spec_id: int, dtype_name: str):
        # The dtype is part of the cache key because the same Spec
        # object id can correspond to multiple dtypes across the per-
        # dtype config lists (we rebuild specs per dtype below).
        all_cfgs = []
        for cfgs in (c for _atn, c in _per_dtype.values()):
            all_cfgs.extend(cfgs)
        cfg = next(c for c in all_cfgs if id(c.spec) == spec_id)
        spec = cfg.spec
        ir_dtype = dtype_to_ir(dtype_name)
        kernel = build_universal_gemm(spec)
        ir = lower_kernel_to_llvm(kernel)
        hsaco, _ = build_hsaco_from_llvm_ir(ir)
        # Kernel param order is A, B, C, M, N, K, bias.
        # All four ptrs share the GEMM element dtype.
        elem_ptr_ty = f"ptr<{ir_dtype.name}, global>"
        # The base signature is hardcoded fp16 in the manifest helper,
        # so build it manually to honor the actual dtype.
        full_sig = [
            {"name": "A", "type": elem_ptr_ty, "size_bytes": 8},
            {"name": "B", "type": elem_ptr_ty, "size_bytes": 8},
            {"name": "C", "type": elem_ptr_ty, "size_bytes": 8},
            {"name": "M", "type": "i32", "size_bytes": 4},
            {"name": "N", "type": "i32", "size_bytes": 4},
            {"name": "K", "type": "i32", "size_bytes": 4},
            {"name": "bias", "type": elem_ptr_ty, "size_bytes": 8},
        ]
        launcher = KernelLauncher(
            hsaco=hsaco,
            kernel_name=kernel.name,
            signature=full_sig,
        )
        block_size = spec.tile.warp_m * spec.tile.warp_n * spec.tile.warp_k * 64
        return launcher, spec, block_size

    def _launch_args(spec, M, N, K, A, B_rcr, bias, C):
        return (
            {"A": A, "B": B_rcr, "C": C, "bias": bias, "M": M, "N": N, "K": K},
            LaunchConfig(
                grid=(
                    (N + spec.tile.tile_n - 1) // spec.tile.tile_n,
                    (M + spec.tile.tile_m - 1) // spec.tile.tile_m,
                    1,
                ),
                block=(
                    spec.tile.warp_m * spec.tile.warp_n * spec.tile.warp_k * 64,
                    1,
                    1,
                ),
            ),
        )

    def _current_torch_stream() -> int:
        import torch as _t

        return int(_t.cuda.current_stream().cuda_stream)

    def _get_tuner(dtype_name: str) -> Autotuner:
        """Lazy per-dtype Autotuner. The first call for a given dtype
        builds the dtype-flavoured config list and registers an
        Autotuner for it; subsequent calls hit the cache.
        """
        if dtype_name in _per_dtype:
            return _per_dtype[dtype_name][0]
        ir_dtype = dtype_to_ir(dtype_name)
        epi = match.epilogue_template.with_dtype(ir_dtype)
        cfgs = _make_gemm_configs(
            epilogue=epi,
            name_prefix=f"fused_{ir_dtype.name}",
        )

        def bench(cfg, **kw):
            launcher, spec, _bs = _build_launcher(id(cfg.spec), dtype_name)
            M, N, K = kw["M"], kw["N"], kw["K"]
            A, B, bias = kw["A"], kw["B"], kw["bias"]
            import torch as _t

            # UniversalGemmSpec currently implements RCR: A[M,K] and
            # B[N,K]. PyTorch matmul receives B[K,N], so adapt the
            # runtime tensor here. A future RRR lowerer can remove this
            # copy by teaching the kernel a row-major B descriptor.
            B_rcr = B.t().contiguous()
            C = _t.empty((M, N), dtype=A.dtype, device=A.device)
            args, lc = _launch_args(spec, M, N, K, A, B_rcr, bias, C)
            stream = _current_torch_stream()
            lc_streamed = type(lc)(
                grid=lc.grid,
                block=lc.block,
                shared_bytes=lc.shared_bytes,
                stream=stream,
            )
            return time_launches(
                lambda: launcher(args, config=lc_streamed),
                warmup=kw.get("_warmup_iters", 10),
                iters=kw.get("_bench_iters", 50),
                stream=stream,
            )

        def launch(cfg, *, A, B, bias, C, M, N, K, **_):
            launcher, spec, _bs = _build_launcher(id(cfg.spec), dtype_name)
            B_rcr = B.t().contiguous()
            args, lc = _launch_args(spec, M, N, K, A, B_rcr, bias, C)
            stream = _current_torch_stream()
            lc_streamed = type(lc)(
                grid=lc.grid,
                block=lc.block,
                shared_bytes=lc.shared_bytes,
                stream=stream,
            )
            launcher(args, config=lc_streamed)

        tuner = Autotuner(
            configs=cfgs,
            key_fn=lambda *, M, N, K, dtype, **_: (M, N, K, dtype),
            bench_fn=bench,
            launch_fn=launch,
            cache_path=cache_path,
            verbose=verbose,
        )
        _per_dtype[dtype_name] = (tuner, cfgs)
        return tuner

    def wrapped(*args, **kwargs):
        import torch as _t

        bound_args = {}
        ordered_names = [match.a_arg_name, match.b_arg_name]
        if match.bias_arg_name is not None:
            ordered_names.append(match.bias_arg_name)
        for name, value in zip(ordered_names, args):
            bound_args[name] = value
        bound_args.update(kwargs)
        missing = [n for n in ordered_names if n not in bound_args]
        if missing:
            raise TypeError(f"missing fused kernel arg(s): {missing}")
        A = bound_args[match.a_arg_name]
        B = bound_args[match.b_arg_name]
        bias = bound_args.get(match.bias_arg_name) if match.bias_arg_name else None
        if (
            A.dim() != 2
            or B.dim() != 2
            or (match.bias_arg_name is not None and (bias is None or bias.dim() != 1))
        ):
            raise ValueError(
                f"compile_fn({match.pattern}): expected 2D A, 2D B"
                + (", 1D bias" if match.bias_arg_name else "")
                + f"; got {tuple(A.shape)}, {tuple(B.shape)}"
                + (f", {tuple(bias.shape)}" if bias is not None else "")
            )
        if A.dtype != B.dtype or (bias is not None and A.dtype != bias.dtype):
            raise ValueError(
                f"compile_fn({match.pattern}): inputs must share dtype; "
                f"got A={A.dtype}, B={B.dtype}"
                + (f", bias={bias.dtype}" if bias is not None else "")
            )
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"matmul dim mismatch: K={K} vs K2={K2}")
        if bias is not None and bias.shape[0] != N:
            raise ValueError(f"bias[{bias.shape[0]}] vs N={N}")
        dtype_map = {_t.float16: "fp16", _t.bfloat16: "bf16"}
        if A.dtype not in dtype_map:
            raise NotImplementedError(
                f"compile_fn: supported GEMM fusion dtypes are fp16/bf16; got {A.dtype}"
            )
        dtype = dtype_map[A.dtype]
        C = _t.empty((M, N), dtype=A.dtype, device=A.device)
        # If no bias exists in the plan, pass a harmless alias for the
        # runtime plumbing. It won't be used because no BiasAdd op
        # declared a bias parameter in the kernel.
        bias_for_runtime = bias if bias is not None else C
        _get_tuner(dtype)(
            A=A, B=B, bias=bias_for_runtime, C=C, M=M, N=N, K=K, dtype=dtype
        )
        return C

    wrapped.autotuners = _per_dtype
    wrapped.match = match.as_dict()
    return wrapped


def explain_fn(fn) -> Dict[str, Any]:
    """Return a structured explanation of whether ``fn`` is fusible.

    This is intentionally cheap: it only FX-traces and pattern-matches;
    it does not compile, autotune, or touch the GPU. Use it in tests or
    notebooks to answer "why didn't this graph fuse?"
    """
    _torch, fx = _try_import_torch_fx()
    traced = fx.symbolic_trace(fn)
    for pattern_fn in _PATTERN_TABLE:
        plan = pattern_fn(traced)
        if plan is not None:
            d = plan.as_dict()
            d["matched"] = True
            return d
    return {
        "matched": False,
        "reason": "no registered CK DSL fusion pattern matched",
        "known_patterns": [p.__name__ for p in _PATTERN_TABLE],
    }
