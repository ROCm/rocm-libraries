# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A small Python IR for the CK DSL.

Kernels and their bodies are first-class Python data structures (SSA
values, ops, regions), not C++ text in an f-string. Each operation
produces an SSA `Value` with a `Type`; control flow ops (`scf.for`)
carry nested `Region`s of ops.

The IR keeps its high-level CK Tile vocabulary (`tile.smem_alloc`,
`tile.smem_load_v4`, `tile.mfma_f32_16x16x16_f16`, `tile.sync`)
distinct from low-level loads/stores. A printer renders the IR in an
MLIR-like text form for inspection; a separate lowering pass walks the
IR to emit HIP.

Design constraints:
- No external dependencies; standard library only.
- Each Op records its source span (file/line) so the lowering can emit
 comments tying generated lines back to the Python authoring site.
- SSA values are uniquely named per kernel; the builder hands out
 `%vN` style identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ----------------------------- Types --------------------------------------


@dataclass(frozen=True)
class Type:
    name: str

    def __repr__(self) -> str:
        return self.name


I1 = Type("i1")
I8 = Type("i8")
I16 = Type("i16")
I32 = Type("i32")
I64 = Type("i64")
BF16 = Type("bf16")
F16 = Type("f16")
F32 = Type("f32")
FP8E4M3 = Type("fp8e4m3")
BF8E5M2 = Type("bf8e5m2")


# AMDGPU buffer-load AUX-byte cache-coherency hints. The AUX field of
# the ``raw_ptr_buffer_load[_lds]`` intrinsics encodes the GLC and SLC
# bits that bias the load's L1/L2 caching policy. Pass one of these
# constants as the ``coherency`` argument to
# ``async_buffer_load_lds_addr`` / ``buffer_load_vN_*``.
CACHE_ALL = 0  # Cache at all levels (default).
CACHE_GLOBAL = 1  # GLC set — skip L2; useful for one-shot loads.
CACHE_STREAM = 2  # SLC set — streaming hint (don't evict useful lines).
NON_TEMPORAL = 3  # GLC + SLC — bypass cache hierarchy entirely.


# ----- target-neutral MMA metadata ---------------------------------------
#
# ``IRBuilder.mma`` emits a single ``tile.mma`` op keyed by ``op_id``; the ISA
# backend lowers that op_id to the matching MFMA/WMMA call. To stay BYTE-
# IDENTICAL with the historical ISA-named emission, ``mma`` must size the result
# vector and pick the result-name hint exactly as the legacy method did. These
# tables encode both, keyed by op_id.
#
# ``_MMA_C_FRAG_LEN`` is duplicated here (rather than imported from
# ``core/arch``) so ``ir.py`` stays free of an ``arch`` import when a caller
# passes a bare op_id string; an ``MmaOp`` object always supplies its own
# ``c_frag_len`` and bypasses this table.
_MMA_C_FRAG_LEN: Dict[str, int] = {
    "mfma_f32_16x16x4_f32": 4,
    "mfma_f32_32x32x2_f32": 16,
    "mfma_f32_16x16x16_f16": 4,
    "mfma_f32_16x16x32_f16": 4,
    "mfma_f32_16x16x16_bf16": 4,
    "mfma_f32_16x16x32_bf16": 4,
    "mfma_f32_16x16x32_fp8": 4,
    "mfma_f32_16x16x32_bf8": 4,
    "mfma_f32_32x32x8_f16": 16,
    "mfma_f32_32x32x16_f16": 16,
    "mfma_f32_32x32x8_bf16": 16,
    "mfma_f32_32x32x16_bf16": 16,
    "mfma_f32_32x32x16_fp8": 16,
    "mfma_f32_32x32x16_bf8": 16,
    "mfma_f32_4x4x4_f16": 4,
    "mfma_f32_16x16x128_fp4": 4,
    "mfma_f32_16x16x96_fp6": 4,
    "mfma_f32_16x16x128_fp8": 4,
    "mfma_scale_f32_16x16x128_f8f6f4": 4,
    "wmma_f32_16x16x16_f16": 8,
    "wmma_f32_16x16x16_bf16": 8,
    "wmma_i32_16x16x16_iu8": 8,
    "wmma_i32_16x16x16_iu4": 8,
    "wmma_gfx12_f32_16x16x16_f16": 8,
    "wmma_gfx12_f32_16x16x16_bf16": 8,
    "wmma_gfx1250_f32_16x16x32_f16": 8,
    "wmma_gfx1250_f32_16x16x32_bf16": 8,
    "wmma_gfx1250_f32_16x16x64_fp8_fp8": 8,
    "wmma_gfx1250_f32_16x16x64_fp8_bf8": 8,
    "wmma_gfx1250_f32_16x16x64_bf8_fp8": 8,
    "wmma_gfx1250_f32_16x16x64_bf8_bf8": 8,
}

# op_id -> accumulator/result *element* type. Float atoms accumulate in f32;
# integer WMMA atoms (iu8/iu4) accumulate in i32. Used by ``IRBuilder.mma`` to
# size the result vector element type when ``op`` is a bare op_id string; an
# ``MmaOp`` object supplies its ``c_dtype`` directly and bypasses this table.
_MMA_C_INT_OP_IDS = frozenset(
    {
        "wmma_i32_16x16x16_iu8",
        "wmma_i32_16x16x16_iu4",
    }
)

# op_id -> the ``result_name_hint`` the legacy ISA-named method used. Most atoms
# used "acc"; a handful used distinct hints that must be preserved verbatim so
# the SSA value numbering (and thus the emitted text) is unchanged.
_MMA_RESULT_HINT: Dict[str, str] = {
    "mfma_f32_32x32x16_bf16": "acc32",
    "mfma_f32_16x16x128_fp4": "acc4",
    "mfma_f32_16x16x96_fp6": "acc6",
    "mfma_f32_16x16x128_fp8": "acc128",
    "mfma_scale_f32_16x16x128_f8f6f4": "mxacc",
}


def _mma_c_frag_len(op_id: str) -> int:
    try:
        return _MMA_C_FRAG_LEN[op_id]
    except KeyError:
        raise ValueError(
            f"unknown MMA op_id {op_id!r}; pass an MmaOp or one of "
            f"{sorted(_MMA_C_FRAG_LEN)}"
        )


@dataclass(frozen=True)
class VectorType(Type):
    elem: Type
    count: int

    def __init__(self, elem: Type, count: int) -> None:
        object.__setattr__(self, "name", f"vec<{elem.name}x{count}>")
        object.__setattr__(self, "elem", elem)
        object.__setattr__(self, "count", count)


@dataclass(frozen=True)
class PtrType(Type):
    pointee: Type
    space: str

    def __init__(self, pointee: Type, space: str) -> None:
        object.__setattr__(self, "name", f"ptr<{pointee.name},{space}>")
        object.__setattr__(self, "pointee", pointee)
        object.__setattr__(self, "space", space)


@dataclass(frozen=True)
class SmemType(Type):
    elem: Type
    shape: Tuple[int, ...]

    def __init__(self, elem: Type, shape: Sequence[int]) -> None:
        shape = tuple(int(x) for x in shape)
        s = "x".join(str(x) for x in shape)
        object.__setattr__(self, "name", f"smem<{elem.name}, [{s}]>")
        object.__setattr__(self, "elem", elem)
        object.__setattr__(self, "shape", shape)


# ----------------------------- Values / Ops ------------------------------


@dataclass
class Value:
    name: str
    type: Type
    op: Optional["Op"] = None

    def __repr__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        raise TypeError(
            "rocke SSA Value cannot be used as a Python bool. "
            "Use IRBuilder.static_if(...) for Python-time branches or "
            "IRBuilder.scf_if(...) for runtime branches."
        )


@dataclass
class Op:
    name: str
    operands: List[Value] = field(default_factory=list)
    results: List[Value] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    regions: List["Region"] = field(default_factory=list)
    loc: Optional[str] = None

    @property
    def result(self) -> Value:
        if len(self.results) != 1:
            raise ValueError(f"op {self.name!r} has {len(self.results)} results, not 1")
        return self.results[0]

    @property
    def is_pure(self) -> bool:
        if "pure" in self.attrs:
            return bool(self.attrs["pure"])
        return is_pure_op_name(self.name)


@dataclass
class Region:
    label: str
    ops: List[Op] = field(default_factory=list)


@dataclass
class Param:
    name: str
    type: Type
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelDef:
    name: str
    params: List[Param]
    body: Region
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_workgroup_size(self) -> int:
        """The upper bound on threads-per-block at launch time. This
        gets baked into the AMDGPU kernel attributes
        (`amdgpu-flat-work-group-size="64,N"`); launching with more
        than N threads/block triggers a HIP `unspecified launch
        failure`. Defaults to 256 for legacy kernels."""
        return int(self.attrs.get("max_workgroup_size", 256))


# ----------------------------- Builder -----------------------------------


class IRBuilder:
    def __init__(self, kernel_name: str) -> None:
        self._counter = 0
        self._region_stack: List[Region] = []
        self._params: List[Param] = []
        self._param_values: Dict[str, Value] = {}
        self.kernel = KernelDef(
            name=kernel_name,
            params=self._params,
            body=Region("entry"),
        )
        self._region_stack.append(self.kernel.body)

    # ----- naming -----

    def _fresh(self, prefix: str = "v") -> str:
        self._counter += 1
        return f"%{prefix}{self._counter}"

    # ----- region management -----

    def _emit(self, op: Op) -> None:
        self._region_stack[-1].ops.append(op)

    def push_region(self, region: Region) -> None:
        self._region_stack.append(region)

    def pop_region(self) -> None:
        self._region_stack.pop()

    # ----- params -----

    def param(self, name: str, t: Type, **attrs: Any) -> Value:
        """Declare a kernel parameter.

        Standard ``attrs`` recognised by the lowering:

        * ``noalias`` (bool) — emit ``noalias`` on the LLVM kernel arg.
        * ``readonly`` (bool) — emit ``readonly``.
        * ``writeonly`` (bool) — emit ``writeonly``.
        * ``align`` (int) — emit ``align N``.
        * ``addr_space`` (str) — for ``PtrType`` params, override the
          pointee address space:

          * ``"global"`` (default) → ``ptr addrspace(1)``
          * ``"constant"`` → ``ptr addrspace(4)`` (kernel-arg
            constant memory; AMDGPU backend uses scalar
            ``s_load_dwordxN`` instead of per-lane VMEM loads).
            Used by descriptor tables (P17): grouped-GEMM
            ``gemm_descs_const``, per-expert pointer arrays, etc.
        """
        if name in self._param_values:
            raise ValueError(f"duplicate kernel parameter {name!r}")
        v = Value(name=f"%{name}", type=t)
        self._param_values[name] = v
        self._params.append(Param(name=name, type=t, attrs=dict(attrs)))
        return v

    def get_param(self, name: str) -> Value:
        return self._param_values[name]

    # ----- compile-time loops -----

    def static_for(
        self,
        start: int,
        stop: int,
        step: int = 1,
        body: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Emit a compile-time unrolled loop.

        This intentionally emits no `scf.for`; it is a marker for kernels
        whose trip count is a Python-time constant and should become straight
        line IR.
        """
        if step == 0:
            raise ValueError("static_for step must not be 0")
        if body is None:
            return
        for i in range(start, stop, step):
            body(i)

    def unroll(self, start: int, stop: Optional[int] = None, step: int = 1) -> range:
        """Return a Python `range` and document intentional static unrolling."""
        if stop is None:
            start, stop = 0, start
        if step == 0:
            raise ValueError("unroll step must not be 0")
        return range(start, stop, step)

    # ----- generic op builder -----

    def _op(
        self,
        name: str,
        operands: Sequence[Value] = (),
        result_types: Sequence[Type] = (),
        attrs: Optional[Dict[str, Any]] = None,
        regions: Optional[Sequence[Region]] = None,
        result_name_hint: str = "v",
        loc: Optional[str] = None,
    ) -> Op:
        results = [Value(self._fresh(result_name_hint), t) for t in result_types]
        op = Op(
            name=name,
            operands=list(operands),
            results=results,
            attrs=dict(attrs or {}),
            regions=list(regions or []),
            loc=loc,
        )
        for r in results:
            r.op = op
        self._emit(op)
        return op

    # ----- arith -----

    def const_i32(self, value: int) -> Value:
        op = self._op(
            "arith.constant",
            result_types=[I32],
            attrs={"value": int(value), "ity": "i32"},
            result_name_hint="c",
        )
        return op.result

    def const_i64(self, value: int) -> Value:
        op = self._op(
            "arith.constant",
            result_types=[I64],
            attrs={"value": int(value), "ity": "i64"},
            result_name_hint="c",
        )
        return op.result

    def const_f32(self, value: float) -> Value:
        op = self._op(
            "arith.constant",
            result_types=[F32],
            attrs={"value": float(value), "ity": "f32"},
            result_name_hint="c",
        )
        return op.result

    def add(self, a: Value, b: Value) -> Value:
        return self._op("arith.add", [a, b], [a.type], result_name_hint="add").result

    def sub(self, a: Value, b: Value) -> Value:
        return self._op("arith.sub", [a, b], [a.type], result_name_hint="sub").result

    def mul(self, a: Value, b: Value) -> Value:
        return self._op("arith.mul", [a, b], [a.type], result_name_hint="mul").result

    def div(self, a: Value, b: Value) -> Value:
        return self._op("arith.div", [a, b], [a.type], result_name_hint="div").result

    def mod(self, a: Value, b: Value) -> Value:
        return self._op("arith.mod", [a, b], [a.type], result_name_hint="mod").result

    def fadd(self, a: Value, b: Value) -> Value:
        return self._op("arith.fadd", [a, b], [a.type], result_name_hint="fadd").result

    def fsub(self, a: Value, b: Value) -> Value:
        return self._op("arith.fsub", [a, b], [a.type], result_name_hint="fsub").result

    def fmul(self, a: Value, b: Value) -> Value:
        return self._op("arith.fmul", [a, b], [a.type], result_name_hint="fmul").result

    def fdiv(self, a: Value, b: Value) -> Value:
        return self._op("arith.fdiv", [a, b], [a.type], result_name_hint="fdiv").result

    def fneg(self, a: Value) -> Value:
        return self._op("arith.fneg", [a], [a.type], result_name_hint="fneg").result

    def fabs(self, a: Value) -> Value:
        """Floating-point absolute value (single-instruction on AMDGPU).

        Lowers to ``llvm.fabs.f32`` (or the matching f16 / bf16
        variant). On the AMDGPU backend this becomes a free input
        modifier on the consumer or a single-cycle ``v_max_f32 a, -a``
        — strictly cheaper than the historical ``fmax(a, fneg(a))``
        idiom which materialises ``-a`` as an SSA value and prevents
        the ``abs`` modifier from firing under multi-use of ``a``.

        Used by smoothquant / moe_smoothquant / add_rmsnorm /
        layernorm / rmsnorm amax sites (~30 sites today).
        """
        return self._op("arith.fabs", [a], [a.type], result_name_hint="fabs").result

    def fma(self, a: Value, b: Value, c: Value) -> Value:
        """Floating-point fused multiply-add: ``a * b + c``.

        Lowers to ``llvm.fmuladd.f32`` so the AMDGPU MachineCombiner
        always picks ``v_fma_f32`` (vs. the bare ``fmul`` + ``fadd``
        pair, which only fuses when the scheduler can prove
        ``contract`` semantics for both ops).
        """
        if not (a.type == b.type == c.type):
            raise ValueError(
                f"fma expects matching types; got {a.type.name}, "
                f"{b.type.name}, {c.type.name}"
            )
        return self._op("arith.fma", [a, b, c], [a.type], result_name_hint="fma").result

    def fmax3(self, a: Value, b: Value, c: Value) -> Value:
        """Three-way floating-point max — ``max(a, max(b, c))``.

        Lowers to ``llvm.amdgcn.fmed3.f32`` rearranged form when the
        AMDGPU backend can prove ordered semantics, otherwise to a
        single ``v_max3_f32`` op (gfx9+). One cycle vs the two-cycle
        ``fmax(fmax(a, b), c)`` chain. Used by smoothquant /
        add_rmsnorm amax + softmax pre-reduce.
        """
        if not (a.type == b.type == c.type):
            raise ValueError(
                f"fmax3 expects matching types; got {a.type.name}, "
                f"{b.type.name}, {c.type.name}"
            )
        return self._op(
            "arith.fmax3", [a, b, c], [a.type], result_name_hint="fmax3"
        ).result

    def fmin3(self, a: Value, b: Value, c: Value) -> Value:
        """Three-way floating-point min — ``min(a, min(b, c))``.

        Sibling of :meth:`fmax3`. Lowers to ``v_min3_f32`` (single
        cycle) on AMDGPU. Used by topk / smin epilogue chains.
        """
        if not (a.type == b.type == c.type):
            raise ValueError(
                f"fmin3 expects matching types; got {a.type.name}, "
                f"{b.type.name}, {c.type.name}"
            )
        return self._op(
            "arith.fmin3", [a, b, c], [a.type], result_name_hint="fmin3"
        ).result

    def cmp_lt(self, a: Value, b: Value) -> Value:
        return self._op(
            "arith.cmp", [a, b], [I1], attrs={"pred": "lt"}, result_name_hint="lt"
        ).result

    def cmp_le(self, a: Value, b: Value) -> Value:
        return self._op(
            "arith.cmp", [a, b], [I1], attrs={"pred": "le"}, result_name_hint="le"
        ).result

    def cmp_gt(self, a: Value, b: Value) -> Value:
        return self._op(
            "arith.cmp", [a, b], [I1], attrs={"pred": "gt"}, result_name_hint="gt"
        ).result

    def cmp_ge(self, a: Value, b: Value) -> Value:
        return self._op(
            "arith.cmp", [a, b], [I1], attrs={"pred": "ge"}, result_name_hint="ge"
        ).result

    def cmp_eq(self, a: Value, b: Value) -> Value:
        return self._op(
            "arith.cmp", [a, b], [I1], attrs={"pred": "eq"}, result_name_hint="eq"
        ).result

    def cmp_ne(self, a: Value, b: Value) -> Value:
        return self._op(
            "arith.cmp", [a, b], [I1], attrs={"pred": "ne"}, result_name_hint="ne"
        ).result

    def fcmp(self, pred: str, a: Value, b: Value) -> Value:
        if pred not in ("olt", "ole", "ogt", "oge", "oeq", "one", "ord", "uno"):
            raise ValueError(f"unsupported fcmp predicate {pred!r}")
        return self._op(
            "arith.fcmp", [a, b], [I1], attrs={"pred": pred}, result_name_hint="fcmp"
        ).result

    def fmax(self, a: Value, b: Value) -> Value:
        return self._op("arith.fmax", [a, b], [a.type], result_name_hint="fmax").result

    def fmin(self, a: Value, b: Value) -> Value:
        return self._op("arith.fmin", [a, b], [a.type], result_name_hint="fmin").result

    def exp2(self, a: Value) -> Value:
        return self._op("math.exp2", [a], [a.type], result_name_hint="exp2").result

    def log2(self, a: Value) -> Value:
        return self._op("math.log2", [a], [a.type], result_name_hint="log2").result

    def rcp(self, a: Value) -> Value:
        return self._op("math.rcp", [a], [a.type], result_name_hint="rcp").result

    def rcp_fast(self, a: Value) -> Value:
        """Fast (~1 ulp) hardware reciprocal: ``v_rcp_f32`` directly.

        Unlike :meth:`rcp` (which lowers to an IEEE-correct ``fdiv 1.0, x`` and
        gets expanded to the full ``v_div_scale``/``v_div_fmas``/``v_div_fixup``
        sequence in the LLVM backend), this maps straight to
        ``llvm.amdgcn.rcp.f32`` -- the same single-instruction reciprocal aiter
        uses in its SiLU sigmoid. f32 only.
        """
        return self._op("math.rcp_fast", [a], [a.type], result_name_hint="rcpf").result

    def sqrt(self, a: Value) -> Value:
        return self._op("math.sqrt", [a], [a.type], result_name_hint="sqrt").result

    def rsqrt(self, a: Value) -> Value:
        """1.0 / sqrt(a). Lowered to a single hardware reciprocal-sqrt on AMDGPU."""
        return self._op("math.rsqrt", [a], [a.type], result_name_hint="rsq").result

    def tanh(self, a: Value) -> Value:
        return self._op("math.tanh", [a], [a.type], result_name_hint="tanh").result

    def land(self, a: Value, b: Value) -> Value:
        """Bit-and (used for i1 conjunction and i32 masks)."""
        return self._op("arith.and", [a, b], [a.type], result_name_hint="and").result

    def lor(self, a: Value, b: Value) -> Value:
        return self._op("arith.or", [a, b], [a.type], result_name_hint="or").result

    def lnot(self, a: Value) -> Value:
        """Bitwise-NOT. For an i1 input this is the logical negation."""
        return self._op("arith.not", [a], [a.type], result_name_hint="not").result

    def smax(self, a: Value, b: Value) -> Value:
        return self._op("arith.smax", [a, b], [a.type], result_name_hint="smax").result

    def smin(self, a: Value, b: Value) -> Value:
        return self._op("arith.smin", [a, b], [a.type], result_name_hint="smin").result

    def zext(self, v: Value, target: Type) -> Value:
        return self._op("arith.zext", [v], [target], result_name_hint="zx").result

    def sext(self, v: Value, target: Type) -> Value:
        return self._op("arith.sext", [v], [target], result_name_hint="sx").result

    def trunc(self, v: Value, target: Type) -> Value:
        return self._op("arith.trunc", [v], [target], result_name_hint="tr").result

    def select(self, cond: Value, lhs: Value, rhs: Value) -> Value:
        return self._op(
            "arith.select", [cond, lhs, rhs], [lhs.type], result_name_hint="sel"
        ).result

    def masked_select(self, cond: Value, lhs: Value, rhs: Value) -> Value:
        return self.select(cond, lhs, rhs)

    def trunc_f32_to_f16(self, v: Value) -> Value:
        return self._op(
            "arith.trunc_f32_to_f16", [v], [F16], result_name_hint="t"
        ).result

    def rint_f32(self, v: Value) -> Value:
        """Round an f32 to the nearest integer (still f32), round-to-nearest-even.

        Lowers to ``llvm.rint.f32`` (``rintf`` in HIP), which honours the
        current rounding mode (RNE). Unlike :meth:`cvt_f32_to_i8_sat` this does
        not narrow to i8, so it works for the full f32 integer range -- the
        primitive an integer GEMM emulated via f16 WMMA needs to snap its
        sub-ULP-noisy accumulator back to the exact integer before requant.
        """
        if v.type.name != "f32":
            raise ValueError(f"rint_f32 expects f32 input, got {v.type.name}")
        return self._op("arith.rint_f32", [v], [F32], result_name_hint="rint").result

    def cast_to_f32(self, v: Value) -> Value:
        if v.type.name == "f32":
            return v
        if v.type.name not in ("f16", "bf16"):
            raise ValueError(f"cast_to_f32 unsupported from {v.type.name}")
        return self._op("arith.cast_to_f32", [v], [F32], result_name_hint="f32").result

    def cast_f32_to(self, v: Value, target: Type) -> Value:
        if v.type.name != "f32":
            raise ValueError("cast_f32_to expects f32 input")
        if target.name == "f32":
            return v
        if target.name not in ("f16", "bf16"):
            raise ValueError(f"cast_f32_to unsupported to {target.name}")
        return self._op(
            "arith.cast_f32_to",
            [v],
            [target],
            attrs={"target": target.name},
            result_name_hint="cast",
        ).result

    def sitofp_f32(self, v: Value) -> Value:
        """Signed-integer to f32 conversion (LLVM sitofp). Inputs must be `i32`.

        This is used by ALiBi-style biases that need `pos_in_f32 = (i32) pos`.
        """
        if v.type.name != "i32":
            raise ValueError(f"sitofp_f32 expects i32 input, got {v.type.name}")
        return self._op("arith.sitofp_f32", [v], [F32], result_name_hint="sitof").result

    def cvt_fp8_to_f32(self, v: Value) -> Value:
        """Convert one fp8e4m3 element to f32.

        Lowers to AMDGPU's `llvm.amdgcn.cvt.f32.fp8`. This is the foundation
        primitive for the FP8 K/V dequant path; FP8 in unified attention
        loads bytes from the cache, applies this conversion, multiplies by
        the per-tensor scale, and casts back to the Q dtype before the
        MFMA.
        """
        if v.type.name != "fp8e4m3":
            raise ValueError(f"cvt_fp8_to_f32 expects fp8e4m3 input, got {v.type.name}")
        return self._op(
            "arith.cvt_fp8_to_f32", [v], [F32], result_name_hint="dq8"
        ).result

    def cvt_bf8_to_f32(self, v: Value) -> Value:
        """Convert one bf8e5m2 element to f32. Lowers to
        ``llvm.amdgcn.cvt.f32.bf8`` (the e5m2 sibling of cvt_fp8_to_f32).
        """
        if v.type.name != "bf8e5m2":
            raise ValueError(f"cvt_bf8_to_f32 expects bf8e5m2 input, got {v.type.name}")
        return self._op(
            "arith.cvt_bf8_to_f32", [v], [F32], result_name_hint="dqb8"
        ).result

    def cvt_pk_f32_fp8x4(self, v: Value) -> Value:
        """Convert <4 x fp8e4m3> to <4 x f32> using AMDGPU's packed
        ``v_cvt_pk_f32_fp8`` (2 lowering intrinsic calls; 2x speedup vs
        4 scalar ``cvt_fp8_to_f32``).

        The hardware ``v_cvt_pk_f32_fp8`` reads an i32 containing 4 fp8
        bytes plus an i1 word-select (0 → bytes 0,1 → <2 x f32>;
        1 → bytes 2,3 → <2 x f32>). We bitcast the input <4 x fp8>
        to i32 and call the intrinsic twice to cover all 4 bytes;
        the result is a single <4 x f32>.

        Use this for FP8 K/V dequant loops where the loader gathers
        groups of 4-or-more contiguous fp8 elements per lane and then
        dequants them to f32 for the post-load scale multiply.
        Reference: AITER's ``to_float_fp8x4`` in
        ``csrc/include/attention_common.cuh`` uses the same primitive
        for its paged-attention FP8 K/V dequant chain.
        """
        if (
            not isinstance(v.type, VectorType)
            or v.type.elem.name != "fp8e4m3"
            or v.type.count != 4
        ):
            raise ValueError(
                f"cvt_pk_f32_fp8x4 expects vec<fp8e4m3x4> input, got {v.type.name}"
            )
        return self._op(
            "arith.cvt_pk_f32_fp8x4",
            [v],
            [VectorType(F32, 4)],
            result_name_hint="dq8x4",
        ).result

    def cvt_pk_f32_bf8x4(self, v: Value) -> Value:
        """e5m2 sibling of :meth:`cvt_pk_f32_fp8x4`. Same packing
        scheme but using ``llvm.amdgcn.cvt.pk.f32.bf8``.
        """
        if (
            not isinstance(v.type, VectorType)
            or v.type.elem.name != "bf8e5m2"
            or v.type.count != 4
        ):
            raise ValueError(
                f"cvt_pk_f32_bf8x4 expects vec<bf8e5m2x4> input, got {v.type.name}"
            )
        return self._op(
            "arith.cvt_pk_f32_bf8x4",
            [v],
            [VectorType(F32, 4)],
            result_name_hint="dqb8x4",
        ).result

    def cvt_scalef32_pk_f32_fp8x4(self, v: Value, scale_f32: Value) -> Value:
        """Convert ``<4 x fp8e4m3> -> <4 x f32>`` with **E8M0-scaled** fused
        cvt via AMDGPU's gfx950 ``v_cvt_scalef32_pk_f32_fp8``.

        IMPORTANT — E8M0 semantics, NOT plain f32 multiply:

            The hardware uses the f32 scale operand's *exponent bits
            only* (an E8M0 microscaling factor). The sign and mantissa
            of the f32 are SILENTLY DISCARDED. Effective scale is
            ``2^(unbiased_exp)`` rounded toward the f32's exponent
            bits. A scale of ``0.011`` (exp bits 120 ⇒ 2^-7 = 0.0078)
            produces outputs ~0.71x of expected; only power-of-two
            scales work bit-correctly.

            This is the MX (microscaling) instruction family: it's
            designed for tensors with block-scale metadata, not for
            arbitrary per-tensor f32 scales. The naming is misleading
            ("scalef32" suggests f32 mul) but matches the LLVM
            intrinsic naming.

            For arbitrary f32 scales (paged-attention
            ``k_scale``/``v_scale``), use :meth:`cvt_pk_f32_fp8x4`
            followed by an explicit ``fmul`` against the scale. The
            cost is ~4 extra packed muls per quad; correctness is
            non-negotiable.

        The intrinsic remains useful for MX-style workloads where the
        scale is naturally power-of-two. The signature accepts an f32
        scale parameter because that's what the LLVM intrinsic
        signature is; the *value* must encode an E8M0 exponent.

        ``scale_f32`` is a single ``f32`` value broadcast to all packs.
        Per AMDGPU spec, the hardware treats this as an "implicit scalar"
        per-call so the scale must be lane-uniform; pass a value loaded
        from a scalar tensor (k_scale / v_scale) which is naturally
        SGPR-resident.
        """
        if (
            not isinstance(v.type, VectorType)
            or v.type.elem.name != "fp8e4m3"
            or v.type.count != 4
        ):
            raise ValueError(
                f"cvt_scalef32_pk_f32_fp8x4 expects vec<fp8e4m3x4>, got {v.type.name}"
            )
        if scale_f32.type.name != "f32":
            raise ValueError(
                f"cvt_scalef32_pk_f32_fp8x4 scale must be f32, got {scale_f32.type.name}"
            )
        return self._op(
            "arith.cvt_scalef32_pk_f32_fp8",
            [v, scale_f32],
            [VectorType(F32, 4)],
            result_name_hint="sdq8x4",
        ).result

    def cvt_scalef32_pk_f32_bf8x4(self, v: Value, scale_f32: Value) -> Value:
        """e5m2 sibling of :meth:`cvt_scalef32_pk_f32_fp8x4`."""
        if (
            not isinstance(v.type, VectorType)
            or v.type.elem.name != "bf8e5m2"
            or v.type.count != 4
        ):
            raise ValueError(
                f"cvt_scalef32_pk_f32_bf8x4 expects vec<bf8e5m2x4>, got {v.type.name}"
            )
        if scale_f32.type.name != "f32":
            raise ValueError("cvt_scalef32_pk_f32_bf8x4 scale must be f32")
        return self._op(
            "arith.cvt_scalef32_pk_f32_bf8",
            [v, scale_f32],
            [VectorType(F32, 4)],
            result_name_hint="sdqb8x4",
        ).result

    def cvt_f32_to_fp8(self, v: Value) -> Value:
        """Round + saturate one f32 element to fp8e4m3 (round-to-nearest-even).

        Lowers to AMDGPU's ``llvm.amdgcn.cvt.pk.fp8.f32`` packing
        intrinsic (two f32 per call), with the second slot held at
        +0.0f and the low byte extracted as the single-element
        result. Out-of-range inputs are clamped to the fp8e4m3 range.
        """
        if v.type.name != "f32":
            raise ValueError(f"cvt_f32_to_fp8 expects f32 input, got {v.type.name}")
        return self._op(
            "arith.cvt_f32_to_fp8", [v], [FP8E4M3], result_name_hint="q8"
        ).result

    def cvt_pk_fp8_f32x4(self, v: Value) -> Value:
        """Round + saturate <4 x f32> to <4 x fp8e4m3> with AMDGPU's
        packed ``v_cvt_pk_fp8_f32`` instruction.

        The scalar :meth:`cvt_f32_to_fp8` wrapper intentionally exposes a
        simple one-element API but wastes the second conversion lane of
        the packed hardware instruction. Attention's native-fp8 path needs
        to quantise Q/P register vectors in groups of four; using this
        primitive turns four scalar cvts into two packed cvts, matching the
        FlyDSL ``pa_decode_fp8`` pattern for P->fp8 quantisation.
        """
        if (
            not isinstance(v.type, VectorType)
            or v.type.elem.name != "f32"
            or v.type.count != 4
        ):
            raise ValueError(
                f"cvt_pk_fp8_f32x4 expects vec<f32x4> input, got {v.type.name}"
            )
        return self._op(
            "arith.cvt_pk_fp8_f32x4",
            [v],
            [VectorType(FP8E4M3, 4)],
            result_name_hint="q8x4",
        ).result

    def cvt_f32_to_bf8(self, v: Value) -> Value:
        """Round + saturate one f32 element to bf8e5m2 (round-to-nearest-even).

        Lowers to ``llvm.amdgcn.cvt.pk.bf8.f32``; the e5m2 sibling of
        cvt_f32_to_fp8. ``bf8e5m2`` carries 5 exponent bits so its
        representable range reaches ~57344 at the cost of 1 mantissa bit.
        """
        if v.type.name != "f32":
            raise ValueError(f"cvt_f32_to_bf8 expects f32 input, got {v.type.name}")
        return self._op(
            "arith.cvt_f32_to_bf8", [v], [BF8E5M2], result_name_hint="qb8"
        ).result

    def cvt_pk_bf8_f32x4(self, v: Value) -> Value:
        """e5m2 sibling of :meth:`cvt_pk_fp8_f32x4`."""
        if (
            not isinstance(v.type, VectorType)
            or v.type.elem.name != "f32"
            or v.type.count != 4
        ):
            raise ValueError(
                f"cvt_pk_bf8_f32x4 expects vec<f32x4> input, got {v.type.name}"
            )
        return self._op(
            "arith.cvt_pk_bf8_f32x4",
            [v],
            [VectorType(BF8E5M2, 4)],
            result_name_hint="qb8x4",
        ).result

    def cvt_pk_i8_f32x4(self, v: Value) -> Value:
        """Round + saturate <4 x f32> to <4 x i8> as a single packed
        primitive (vs four scalar :meth:`cvt_f32_to_i8_sat`).

        The lowering uses ``llvm.rint.f32`` + ``smin`` / ``smax`` on
        the i32 result and then a ``llvm.amdgcn.perm`` byte-select to
        pack four i8 lanes into one i32, which is bitcast back to
        <4 x i8>. AMDGPU's pattern matcher folds the rint+min/max
        quad into 2-3 ``v_med3_i32`` plus a ``v_perm_b32`` -- ~6-8
        instructions vs ~20 for the scalar chain.
        """
        if (
            not isinstance(v.type, VectorType)
            or v.type.elem.name != "f32"
            or v.type.count != 4
        ):
            raise ValueError(
                f"cvt_pk_i8_f32x4 expects vec<f32x4> input, got {v.type.name}"
            )
        return self._op(
            "arith.cvt_pk_i8_f32x4",
            [v],
            [VectorType(I8, 4)],
            result_name_hint="qi8x4",
        ).result

    def cvt_f32_to_i8_sat(self, v: Value) -> Value:
        """Round + saturate one f32 element to i8 (round-to-nearest-even).

        ``(int8_t) min(127, max(-128, lrintf(v)))`` -- the canonical
        int8 dynamic-range quantisation primitive used by SmoothQuant
        and every per-row int8-quant epilogue.
        """
        if v.type.name != "f32":
            raise ValueError(f"cvt_f32_to_i8_sat expects f32 input, got {v.type.name}")
        return self._op(
            "arith.cvt_f32_to_i8_sat", [v], [I8], result_name_hint="qi8"
        ).result

    def clamp_f32(self, v: Value, lo: Value, hi: Value) -> Value:
        """``min(hi, max(lo, v))`` for f32 -- saturating clamp every
        quantisation path needs before rounding. The two ``fmin`` /
        ``fmax`` emissions fold into a single ``v_med3_f32`` on AMDGPU.
        """
        if v.type.name != "f32":
            raise ValueError(f"clamp_f32 expects f32 input, got {v.type.name}")
        return self.fmin(hi, self.fmax(lo, v))

    def xor(self, a: Value, b_v: Value) -> Value:
        """Bitwise XOR. Same type for both operands."""
        return self._op("arith.xor", [a, b_v], [a.type], result_name_hint="xor").result

    def shl(self, a: Value, b_v: Value) -> Value:
        """Logical left shift ``a << b_v``. Lowers to LLVM ``shl``.
        AMDGPU folds constant shifts into ``v_lshlrev_b32`` with the
        immediate baked in.
        """
        return self._op("arith.shl", [a, b_v], [a.type], result_name_hint="shl").result

    def lshr(self, a: Value, b_v: Value) -> Value:
        """Logical right shift ``a >> b_v`` (zero-fill). Lowers to
        LLVM ``lshr``; AMDGPU folds to ``v_lshrrev_b{32,64}``.
        """
        return self._op(
            "arith.lshr", [a, b_v], [a.type], result_name_hint="lshr"
        ).result

    def umul_hi_i32(self, a: Value, b_v: Value) -> Value:
        """Unsigned high 32 bits of an i32 * i32 product.

        Returns ``((u64)a * (u64)b) >> 32`` as an i32. Used by
        Philox4x32 RNG; the AMDGPU backend lowers it to ``v_mul_hi_u32``
        (a single instruction).
        """
        if a.type.name != "i32" or b_v.type.name != "i32":
            raise ValueError(
                f"umul_hi_i32 expects i32 operands, got {a.type.name} / {b_v.type.name}"
            )
        return self._op(
            "arith.umul_hi_i32", [a, b_v], [I32], result_name_hint="umh"
        ).result

    def global_atomic_add(
        self,
        ptr: Value,
        idx: Value,
        value: Value,
        *,
        ordering: str = "monotonic",
    ) -> Value:
        """Atomic add into a global tensor; returns the value at the slot
        *before* the add (LLVM ``atomicrmw add`` semantics).

        Supports ``i32`` and ``f32`` operands. Used by the MoE sort
        kernel and FMHA backward (fp32 dQ accumulate across the K loop).
        """
        if value.type.name not in ("i32", "f32"):
            raise ValueError(
                f"global_atomic_add supports i32 / f32, got {value.type.name}"
            )
        if ordering not in ("monotonic", "acquire", "release", "acq_rel", "seq_cst"):
            raise ValueError(f"unknown ordering {ordering!r}")
        return self._op(
            "memref.global_atomic_add",
            [ptr, idx, value],
            [value.type],
            attrs={"elem_type": value.type.name, "ordering": ordering},
            result_name_hint="atom_add",
        ).result

    def lds_atomic_add(
        self,
        smem: Value,
        indices: Sequence[Value],
        value: Value,
        *,
        ordering: str = "monotonic",
    ) -> Value:
        """Atomic add into an LDS slot; returns the pre-add value.

        Used by the MoE sort histogram pass (per-block per-expert
        local counters). Lowers to ``ds_add_u32`` / ``ds_pk_add_f32``.
        """
        if value.type.name not in ("i32", "f32"):
            raise ValueError(
                f"lds_atomic_add supports i32 / f32, got {value.type.name}"
            )
        if ordering not in ("monotonic", "acquire", "release", "acq_rel", "seq_cst"):
            raise ValueError(f"unknown ordering {ordering!r}")
        return self._op(
            "tile.lds_atomic_add",
            [smem, *indices, value],
            [value.type],
            attrs={
                "elem_type": value.type.name,
                "rank": len(indices),
                "ordering": ordering,
            },
            result_name_hint="lds_atom",
        ).result

    def global_atomic_add_pk_bf16(
        self,
        ptr: Value,
        idx: Value,
        value: Value,
        *,
        ordering: str = "monotonic",
    ) -> Value:
        """Packed-bf16 atomic add: two bf16 lanes per transaction.

        Lowers to AMDGPU's ``llvm.amdgcn.global.atomic.fadd.v2bf16``
        intrinsic (gfx940+); returns the pre-add value. ``value``
        must be a ``<2 x bf16>`` vector and the pointer must reach
        into a bf16 buffer with an even element index. Used by the
        FMHA-backward kernel's dQ accumulate path for direct-to-bf16
        landing.
        """
        if ordering not in ("monotonic", "acquire", "release", "acq_rel", "seq_cst"):
            raise ValueError(f"unknown ordering {ordering!r}")
        if not isinstance(value.type, VectorType):
            raise ValueError(
                f"global_atomic_add_pk_bf16 expects <2 x bf16> input, "
                f"got {value.type.name}"
            )
        if value.type.elem != BF16 or value.type.count != 2:
            raise ValueError(
                f"global_atomic_add_pk_bf16 expects <2 x bf16> input, "
                f"got {value.type.name}"
            )
        return self._op(
            "memref.global_atomic_add_pk_bf16",
            [ptr, idx, value],
            [value.type],
            attrs={"elem_type": "bf16", "vec": 2, "ordering": ordering},
            result_name_hint="atom_bf16",
        ).result

    def fp16_zero(self) -> Value:
        return self._op(
            "arith.constant",
            result_types=[F16],
            attrs={"value": 0.0, "ity": "f16"},
            result_name_hint="c",
        ).result

    def zero_vec_f32_4(self) -> Value:
        return self.zero_vec_f32(4)

    def zero_vec_f32(self, n: int) -> Value:
        """A `<n x float>` zero accumulator.

        16x16 MFMA atoms use n=4 (4 floats per lane); the 32x32 atoms
        used by every production CK dispatcher tile use n=16. Larger
        per-warp tiles (e.g. 4x4 MFMA per warp with the 32x32 atom)
        get one of these per atom, threaded through the K loop as
        loop-carried `iter_args`.
        """
        if n <= 0:
            raise ValueError(f"zero_vec_f32 needs positive n, got {n}")
        return self._op(
            "arith.constant_vec",
            result_types=[VectorType(F32, n)],
            attrs={"fill": 0.0, "elem": "f32", "vec": n},
            result_name_hint=f"cz{n}",
        ).result

    def zero_vec(self, elem: Type, n: int) -> Value:
        if elem == F32:
            return self.zero_vec_f32(n)
        if elem in (F16, BF16, FP8E4M3, BF8E5M2, I8, I32):
            return self._op(
                "arith.constant_vec",
                result_types=[VectorType(elem, n)],
                attrs={"fill": 0.0, "elem": elem.name, "vec": n},
                result_name_hint=f"cz{n}",
            ).result
        raise ValueError(f"zero_vec unsupported elem {elem.name}")

    # ----- gpu / runtime -----

    def thread_id_x(self) -> Value:
        return self._op(
            "gpu.thread_id",
            attrs={"axis": "x"},
            result_types=[I32],
            result_name_hint="tid",
        ).result

    def block_id_x(self) -> Value:
        return self._op(
            "gpu.block_id",
            attrs={"axis": "x"},
            result_types=[I32],
            result_name_hint="bid",
        ).result

    def block_id_y(self) -> Value:
        return self._op(
            "gpu.block_id",
            attrs={"axis": "y"},
            result_types=[I32],
            result_name_hint="bid",
        ).result

    def block_id_z(self) -> Value:
        return self._op(
            "gpu.block_id",
            attrs={"axis": "z"},
            result_types=[I32],
            result_name_hint="bid",
        ).result

    # ----- memory -----

    def smem_alloc(
        self, elem: Type, shape: Sequence[int], name_hint: str = "smem"
    ) -> Value:
        t = SmemType(elem, shape)
        return self._op(
            "tile.smem_alloc", result_types=[t], result_name_hint=name_hint
        ).result

    def global_load(
        self, ptr: Value, idx: Value, dtype: Type, *, align: int = 1
    ) -> Value:
        return self._op(
            "memref.global_load_typed",
            [ptr, idx],
            [dtype],
            attrs={"elem_type": dtype.name, "align": int(align)},
            result_name_hint="gl",
        ).result

    def global_load_f16(self, ptr: Value, idx: Value, *, align: int = 2) -> Value:
        return self._op(
            "memref.global_load",
            [ptr, idx],
            [F16],
            attrs={"align": int(align)},
            result_name_hint="gl",
        ).result

    def global_load_f32(self, ptr: Value, idx: Value, *, align: int = 4) -> Value:
        return self.global_load(ptr, idx, F32, align=align)

    def global_load_i32(self, ptr: Value, idx: Value, *, align: int = 4) -> Value:
        return self.global_load(ptr, idx, I32, align=align)

    def global_load_i64(self, ptr: Value, idx: Value, *, align: int = 8) -> Value:
        return self.global_load(ptr, idx, I64, align=align)

    def global_load_bf16(self, ptr: Value, idx: Value, *, align: int = 2) -> Value:
        return self.global_load(ptr, idx, BF16, align=align)

    def global_load_fp8e4m3(self, ptr: Value, idx: Value, *, align: int = 1) -> Value:
        return self.global_load(ptr, idx, FP8E4M3, align=align)

    def masked_global_load(
        self,
        ptr: Value,
        idx: Value,
        mask: Value,
        other: Value,
        dtype: Type,
        *,
        align: int = 1,
    ) -> Value:
        """OOB-safe masked global load.

        Unlike a raw global load + select, this clamps the index to a safe
        value when the mask is false. Some callers (e.g. ALiBi and QQ-bias
        in the unified attention kernels) supply indices that are negative
        or out-of-range when the mask is false; on AMDGPU a global_load with
        such an index would issue a real memory access and can fault when
        the resulting virtual address lands in unmapped memory. Clamping
        keeps the access in-bounds while still returning ``other`` for the
        masked positions.
        """
        if idx.type.name != "i32":
            raise ValueError("masked_global_load expects i32 index for clamp-safe load")
        safe_idx = self.select(mask, idx, self.const_i32(0))
        loaded = self.global_load(ptr, safe_idx, dtype, align=align)
        return self.select(mask, loaded, other)

    def global_store(
        self, ptr: Value, idx: Value, value: Value, *, align: int = 1
    ) -> None:
        self._op(
            "memref.global_store_typed",
            [ptr, idx, value],
            attrs={"elem_type": value.type.name, "align": int(align)},
        )

    def global_load_vN_f16(
        self, ptr: Value, idx: Value, n: int, *, align: Optional[int] = None
    ) -> Value:
        """Vectorised global load of N consecutive halves (N in {2,4,8})."""
        return self.global_load_vN(ptr, idx, F16, n, align=align)

    def global_load_vN(
        self,
        ptr: Value,
        idx: Value,
        dtype: Type,
        n: int,
        *,
        align: Optional[int] = None,
    ) -> Value:
        """Vectorised global load of N consecutive values.

        Supports the full element-type catalog the LLVM lowering already
        accepts: ``f16`` / ``bf16`` (N in {2, 4, 8}), ``f32`` / ``i32``
        (N in {2, 4, 8}), ``i16`` (N in {2, 4, 8}), ``fp8e4m3`` /
        ``bf8e5m2`` / ``i8`` (N in {2, 4, 8, 16}).

        Lowers to a single ``load <N x elem>`` from ``addrspace(1)``;
        AMDGPU's backend coalesces these into a single VMEM transaction
        (``global_load_dwordxN``) when the address is naturally aligned.

        The per-element size is folded into the default alignment so the
        common case (8 fp8 → 8-byte load, 4 f32 → 16-byte load) does
        not need an explicit ``align=`` kwarg.
        """
        if dtype.name in ("f16", "bf16", "i16"):
            elem_bytes = 2
            # n=16 (32-byte `global_load_dwordx8`) is needed for the RDNA WMMA
            # <16 x half> operand fragment; AMDGPU coalesces it when aligned.
            if n not in (2, 4, 8, 16):
                raise ValueError(f"unsupported vector width for global_load_vN: {n}")
        elif dtype.name in ("f32", "i32"):
            elem_bytes = 4
            if n not in (2, 4, 8):
                raise ValueError(
                    f"unsupported vector width for {dtype.name} global_load_vN: {n}"
                )
        elif dtype.name in ("fp8e4m3", "bf8e5m2", "i8"):
            elem_bytes = 1
            if n not in (2, 4, 8, 16):
                raise ValueError(
                    f"unsupported vector width for {dtype.name} global_load_vN: {n}"
                )
        else:
            raise ValueError(
                "global_load_vN supports f16/bf16/i16/f32/i32/fp8e4m3/bf8e5m2/i8, "
                f"got {dtype.name}"
            )
        return self._op(
            "memref.global_load_vN",
            [ptr, idx],
            [VectorType(dtype, n)],
            attrs={
                "elem_type": dtype.name,
                "vec": n,
                "align": int(align or (n * elem_bytes)),
            },
            result_name_hint=f"gv{n}",
        ).result

    def vector_binary(self, op_name: str, a: Value, b: Value) -> Value:
        if not isinstance(a.type, VectorType) or a.type != b.type:
            raise ValueError("vector_binary expects matching vector operands")
        return self._op(
            f"vector.{op_name}", [a, b], [a.type], result_name_hint=f"v{op_name}"
        ).result

    def vector_add(self, a: Value, b: Value) -> Value:
        return self.vector_binary("add", a, b)

    def vector_mul(self, a: Value, b: Value) -> Value:
        return self.vector_binary("mul", a, b)

    def vector_and(self, a: Value, b: Value) -> Value:
        return self.vector_binary("and", a, b)

    def vector_or(self, a: Value, b: Value) -> Value:
        return self.vector_binary("or", a, b)

    def vector_shl(self, a: Value, b: Value) -> Value:
        return self.vector_binary("shl", a, b)

    def vector_lshr(self, a: Value, b: Value) -> Value:
        return self.vector_binary("lshr", a, b)

    def vector_smax(self, a: Value, b: Value) -> Value:
        return self.vector_binary("smax", a, b)

    def vector_smin(self, a: Value, b: Value) -> Value:
        return self.vector_binary("smin", a, b)

    def vector_max(self, a: Value, b: Value) -> Value:
        return self.vector_binary("max", a, b)

    def vector_sub(self, a: Value, b: Value) -> Value:
        return self.vector_binary("sub", a, b)

    def vector_fma(self, a: Value, b: Value, c: Value) -> Value:
        """Packed FMA — ``a * b + c`` over matching vector operands.

        Lowers to ``llvm.fmuladd.v<N>x<elem>`` so the AMDGPU
        MachineCombiner picks ``v_pk_fma_f32`` / ``v_fma_f32`` chains
        without relying on the scheduler proving ``contract``
        semantics on a separate ``vector.mul`` + ``vector.add`` pair.
        """
        if not (isinstance(a.type, VectorType) and a.type == b.type == c.type):
            raise ValueError("vector_fma expects three matching vector operands")
        return self._op(
            "vector.fma", [a, b, c], [a.type], result_name_hint="vfma"
        ).result

    def vector_sum(self, v: Value) -> Value:
        if not isinstance(v.type, VectorType):
            raise ValueError("vector_sum expects vector")
        return self._op(
            "vector.sum", [v], [v.type.elem], result_name_hint="vsum"
        ).result

    def vector_reduce_max(self, v: Value) -> Value:
        if not isinstance(v.type, VectorType):
            raise ValueError("vector_reduce_max expects vector")
        return self._op(
            "vector.reduce_max", [v], [v.type.elem], result_name_hint="vmax"
        ).result

    def vector_splat(self, scalar: Value, n: int) -> Value:
        return self._op(
            "vector.splat",
            [scalar],
            [VectorType(scalar.type, n)],
            attrs={"vec": int(n)},
            result_name_hint="splat",
        ).result

    def vector_select(self, mask: Value, lhs: Value, rhs: Value) -> Value:
        if lhs.type != rhs.type:
            raise ValueError("vector_select lhs/rhs type mismatch")
        return self._op(
            "vector.select", [mask, lhs, rhs], [lhs.type], result_name_hint="vsel"
        ).result

    def vector_cmp(self, pred: str, a: Value, b: Value) -> Value:
        if not isinstance(a.type, VectorType) or a.type != b.type:
            raise ValueError("vector_cmp expects matching vector operands")
        return self._op(
            "vector.cmp",
            [a, b],
            [VectorType(I1, a.type.count)],
            attrs={"pred": pred},
            result_name_hint=f"vcmp_{pred}",
        ).result

    def vector_trunc(self, v: Value, target: Type) -> Value:
        if not isinstance(v.type, VectorType):
            raise ValueError("vector_trunc expects vector input")
        return self._op(
            "vector.trunc",
            [v],
            [VectorType(target, v.type.count)],
            attrs={"target": target.name},
            result_name_hint=f"vtr{v.type.count}",
        ).result

    def vector_sext(self, v: Value, target: Type) -> Value:
        """Sign-extend each lane of an integer vector to a wider element type
        (e.g. ``<N x i8>`` -> ``<N x i16>``). Lowers to an LLVM vector ``sext``;
        the AMDGPU backend keeps the packed layout so a following packed op
        (``v_pk_max_i16`` etc.) can consume it."""
        if not isinstance(v.type, VectorType):
            raise ValueError("vector_sext expects vector input")
        return self._op(
            "vector.sext",
            [v],
            [VectorType(target, v.type.count)],
            attrs={"target": target.name},
            result_name_hint=f"vsx{v.type.count}",
        ).result

    def smem_store_f16(
        self, smem: Value, indices: Sequence[Value], value: Value
    ) -> None:
        self._op(
            "tile.smem_store",
            [smem, *indices, value],
            attrs={"rank": len(indices), "elem_type": "f16"},
        )

    def smem_store_vN_f16(
        self, smem: Value, indices: Sequence[Value], value: Value, n: int
    ) -> None:
        """Vectorised LDS store of <n x f16> (N in {1,2,4,8}).

        The address space and alignment let the backend issue
        ds_write_b{16,32,64,128} instead of element-wise stores.
        """
        self.smem_store_vN(smem, indices, value, n)

    def smem_store_vN(
        self, smem: Value, indices: Sequence[Value], value: Value, n: int
    ) -> None:
        """Vectorised LDS store of N consecutive values.

        Supports scalar stores (`n=1`) plus vector stores for the element types
        accepted by :meth:`smem_load_vN`. For 8-bit element types, ``n=16`` maps
        to a 16-byte LDS transaction, which is the native-int WMMA staging shape.
        """
        if n == 1:
            # Single-element store; route through the scalar `tile.smem_store`.
            self._op(
                "tile.smem_store",
                [smem, *indices, value],
                attrs={"rank": len(indices), "elem_type": value.type.name},
            )
            return
        if not isinstance(value.type, VectorType):
            raise ValueError("smem_store_vN expects vector value for n > 1")
        elem_name = value.type.elem.name
        allowed_n = (
            (2, 4, 8, 16) if elem_name in ("i8", "fp8e4m3", "bf8e5m2") else (2, 4, 8)
        )
        if n not in allowed_n:
            raise ValueError(
                f"unsupported vector width for smem_store_vN of {elem_name}: {n} "
                f"(allowed: {allowed_n})"
            )
        elem_bytes = (
            1
            if elem_name in ("i8", "fp8e4m3", "bf8e5m2")
            else 4 if elem_name in ("f32", "i32") else 2
        )
        self._op(
            "tile.smem_store_vN",
            [smem, *indices, value],
            attrs={
                "rank": len(indices),
                "elem_type": elem_name,
                "vec": n,
                "align": n * elem_bytes,
            },
        )

    def smem_load_v4_f16(self, smem: Value, row: Value, col: Value) -> Value:
        return self._op(
            "tile.smem_load_v4",
            [smem, row, col],
            [VectorType(F16, 4)],
            attrs={"elem_type": "f16"},
            result_name_hint="a",
        ).result

    def smem_load_vN_f16(self, smem: Value, *indices, n: int = 0) -> Value:
        """LDS load of <N x f16> (N in {1,2,4,8}). N=8 emits `ds_read_b128`
        on AMDGPU when the address is 16-byte aligned.

        Pass one Value per array dim (1D, 2D, 3D, ...).
        N=1 returns `<1 x half>`; use `vec_extract(., 0)` to get a scalar half.
        """
        return self.smem_load_vN(smem, *indices, dtype=F16, n=n)

    def smem_load_vN(self, smem: Value, *indices, dtype: Type, n: int = 0) -> Value:
        """LDS load of ``<N x dtype>``. Supports 8-bit (fp8e4m3 / bf8e5m2 /
        i8), 16-bit (f16 / bf16) and 32-bit (f32 / i32) element types;
        AMDGPU lowers vector LDS loads to ``ds_read_b{8, 16, 32, 64, 128}``
        based on total payload size. The 8-bit variants must use ``n in {1,
        2, 4, 8, 16}`` so the resulting payload still maps to a single
        ``ds_read_b*`` instruction (n=16 → ds_read_b128).
        """
        if dtype.name not in ("f16", "bf16", "f32", "i32", "fp8e4m3", "bf8e5m2", "i8"):
            raise ValueError(
                "smem_load_vN supports f16 / bf16 / f32 / i32 / fp8e4m3 / "
                f"bf8e5m2 / i8, got {dtype.name}"
            )
        allowed_n = (
            (1, 2, 4, 8, 16)
            if dtype.name in ("fp8e4m3", "bf8e5m2", "i8")
            else (1, 2, 4, 8)
        )
        if n not in allowed_n:
            raise ValueError(
                f"unsupported vector width {n} for smem_load_vN of {dtype.name} "
                f"(allowed: {allowed_n})"
            )
        if not indices:
            raise ValueError("smem_load_vN needs at least one index")
        return self._op(
            "tile.smem_load_vN",
            [smem, *indices],
            [VectorType(dtype, n)],
            attrs={"elem_type": dtype.name, "vec": n, "rank": len(indices)},
            result_name_hint=f"av{n}",
        ).result

    # ----- target-neutral MMA -----

    def mma(
        self,
        op: Any,
        a: Value,
        b: Value,
        c: Value,
        *extra: Value,
    ) -> Value:
        """Target-neutral matrix-multiply-accumulate: ``D = A * B + C``.

        ``op`` is either an :class:`~rocke.core.arch.MmaOp` (preferred — its
        ``op_id`` and ``c_frag_len`` drive the lowering) or a raw ``op_id``
        string. This emits a single ``tile.mma`` op carrying the ``op_id`` as an
        attribute; the LLVM lowering dispatches that ``op_id`` through the ISA
        backend (:meth:`rocke.core.isa.ISABackend.emit_mma`), which emits the
        matching MFMA call on CDNA or the WMMA call on RDNA. **One kernel body,
        two ISAs.**

        The result vector type is ``<c_frag_len x float>`` (the per-lane
        accumulator length the atom produces). When ``op`` is an ``op_id``
        string the frag length is resolved from the static MMA fragment table.

        The ISA-named helpers (:meth:`mfma_f32_16x16x16_f16`,
        :meth:`wmma_f32_16x16x16_f16`, …) are thin wrappers over this method,
        kept for backward compatibility; they pass the ``op_id`` that reproduces
        their historical emission byte-for-byte.

        ``*extra`` carries the additional operands the scaled MX atoms need
        (``a_scale``, ``b_scale``); ordinary atoms take exactly ``a, b, c``.
        """
        op_id = op.op_id if hasattr(op, "op_id") else str(op)
        c_frag_len = (
            op.c_frag_len
            if hasattr(op, "c_frag_len") and op.c_frag_len
            else _mma_c_frag_len(op_id)
        )
        # Accumulator element type: integer WMMA atoms (iu8/iu4) accumulate in
        # i32; everything else in f32. Prefer the atom's own c_dtype when ``op``
        # is an MmaOp, else fall back to the op_id table.
        c_dtype = getattr(op, "c_dtype", None)
        is_int_acc = (
            c_dtype == "i32" if c_dtype is not None else op_id in _MMA_C_INT_OP_IDS
        )
        c_elem = I32 if is_int_acc else F32
        hint = _MMA_RESULT_HINT.get(op_id, "acc")
        return self._op(
            "tile.mma",
            [a, b, c, *extra],
            [VectorType(c_elem, c_frag_len)],
            attrs={"op_id": op_id},
            result_name_hint=hint,
        ).result

    def wmma_f32_16x16x16_f16(self, a: Value, b: Value, c: Value) -> Value:
        """RDNA3/3.5 (gfx11) WMMA: D[16x16] += A[16x16] * B[16x16], f16 in / f32 acc.

        Wave32 ABI (hardware-verified on gfx1151): per lane ``a`` and ``b`` are
        ``<16 x half>`` and the accumulator ``c`` / result are ``<8 x float>``.
        Fragment layout (lane ``l``): ``a`` = row ``l%16`` (K=0..15), ``b`` =
        col ``l%16`` (K=0..15); result slot ``i`` = ``(row 2i+l/16, col l%16)``.
        Lowered via :meth:`rocke.core.isa.Gfx11RdnaBackend.emit_wmma`; CDNA
        targets reject this op. Thin wrapper over :meth:`mma`.
        """
        return self.mma("wmma_f32_16x16x16_f16", a, b, c)

    def wmma_f32_16x16x16_bf16(self, a: Value, b: Value, c: Value) -> Value:
        """RDNA3/3.5 (gfx11) WMMA: D[16x16] += A[16x16] * B[16x16], bf16 in / f32 acc.

        Wave32 ABI (hardware-verified on gfx1151, ROCm 7.0.2 clang 20): per lane
        ``a`` and ``b`` are ``<16 x bfloat>`` and the accumulator ``c`` / result
        are ``<8 x float>``. The fragment layout matches the f16 WMMA (lane
        ``l``: ``a`` = row ``l%16`` K=0..15, ``b`` = col ``l%16`` K=0..15;
        result slot ``i`` = ``(row 2i+l/16, col l%16)``). The backend bitcasts
        the ``<16 x bfloat>`` operands to ``<16 x i16>`` for the intrinsic
        ``llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16``. Lowered via
        :meth:`rocke.core.isa.Gfx11RdnaBackend.emit_wmma`; CDNA targets reject
        this op. Thin wrapper over :meth:`mma`.
        """
        return self.mma("wmma_f32_16x16x16_bf16", a, b, c)

    def wmma_gfx12_f32_16x16x16_f16(self, a: Value, b: Value, c: Value) -> Value:
        """RDNA4 (gfx12) WMMA: D[16x16] += A[16x16] * B[16x16], f16 in / f32 acc.

        Unlike RDNA3/3.5, gfx12 drops the cross-half operand duplication: per
        lane ``a`` and ``b`` are ``<8 x half>`` and the accumulator ``c`` /
        result are ``<8 x float>``. Fragment layout (lane ``l``): ``a`` = row
        ``l%16`` K=``(l//16)*8 + i``, ``b`` = col ``l%16`` K=``(l//16)*8 + i``;
        result is column-distributed, slot ``i`` = ``(row (l//16)*8 + i,
        col l%16)``. Lowered via :meth:`rocke.core.isa.Gfx12RdnaBackend.emit_wmma`
        (intrinsic ``...v8f32.v8f16``). Thin wrapper over :meth:`mma`.
        """
        return self.mma("wmma_gfx12_f32_16x16x16_f16", a, b, c)

    def wmma_gfx12_f32_16x16x16_bf16(self, a: Value, b: Value, c: Value) -> Value:
        """RDNA4 (gfx12) WMMA bf16 variant. Same ``<8 x half>``-style fragment
        layout as :meth:`wmma_gfx12_f32_16x16x16_f16` but with ``<8 x bfloat>``
        operands bitcast to ``<8 x i16>`` for the intrinsic
        ``llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16``. Thin wrapper over
        :meth:`mma`."""
        return self.mma("wmma_gfx12_f32_16x16x16_bf16", a, b, c)

    def wmma_gfx1250_f32_16x16x32_f16(self, a: Value, b: Value, c: Value) -> Value:
        """gfx1250 (gfx1250) WMMA: D[16x16] += A[16x32] * B[32x16], f16 in / f32 acc.

        The gfx1250 (CDNA, GFX12 programming model) fp16 WMMA atom is K=32. Per
        lane ``a`` and ``b`` are ``<16 x half>`` (16 K-elements, the K dimension
        split across the two lane-halves) and the accumulator ``c`` / result are
        ``<8 x float>`` (same 16x16 column-distributed layout as gfx12). Lowered
        via :meth:`rocke.core.isa.Gfx1250Backend.emit_wmma` to the 8-operand
        intrinsic ``llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16``. Thin
        wrapper over :meth:`mma`."""
        return self.mma("wmma_gfx1250_f32_16x16x32_f16", a, b, c)

    def wmma_gfx1250_f32_16x16x32_bf16(self, a: Value, b: Value, c: Value) -> Value:
        """gfx1250 (gfx1250) WMMA bf16 variant. Same ``<16 x half>``-style K=32
        fragment layout as :meth:`wmma_gfx1250_f32_16x16x32_f16` but with
        ``<16 x bfloat>`` operands passed directly to the intrinsic
        ``llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16`` (no i16 bitcast,
        unlike gfx11/gfx12). Thin wrapper over :meth:`mma`."""
        return self.mma("wmma_gfx1250_f32_16x16x32_bf16", a, b, c)

    def wmma_gfx1250_f32_16x16x64_fp8_fp8(self, a: Value, b: Value, c: Value) -> Value:
        """gfx1250 (gfx1250) FP8 WMMA: D[16x16] += A[16x64] * B[64x16], f32 acc.

        The gfx1250 low-bit WMMA atom is **K=64** (distinct from the gfx12
        16x16x16 FP8 form, which is not selectable on gfx1250). Per lane ``a``
        and ``b`` are 32 fp8 bytes carried as ``<8 x i32>`` and the accumulator
        ``c`` / result are ``<8 x float>``. Lowered through
        :meth:`rocke.core.isa.Gfx1250Backend.emit_wmma` to
        ``llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32``. Thin wrapper over
        :meth:`mma`."""
        return self.mma("wmma_gfx1250_f32_16x16x64_fp8_fp8", a, b, c)

    def wmma_gfx1250_f32_16x16x64_fp8_bf8(self, a: Value, b: Value, c: Value) -> Value:
        """gfx1250 (gfx1250) mixed FP8(A)/BF8(B) K=64 WMMA. Same ``<8 x i32>``
        fragment ABI as :meth:`wmma_gfx1250_f32_16x16x64_fp8_fp8`. Thin wrapper
        over :meth:`mma`."""
        return self.mma("wmma_gfx1250_f32_16x16x64_fp8_bf8", a, b, c)

    def wmma_gfx1250_f32_16x16x64_bf8_fp8(self, a: Value, b: Value, c: Value) -> Value:
        """gfx1250 (gfx1250) mixed BF8(A)/FP8(B) K=64 WMMA. Thin wrapper over
        :meth:`mma`."""
        return self.mma("wmma_gfx1250_f32_16x16x64_bf8_fp8", a, b, c)

    def wmma_gfx1250_f32_16x16x64_bf8_bf8(self, a: Value, b: Value, c: Value) -> Value:
        """gfx1250 (gfx1250) BF8 K=64 WMMA. Thin wrapper over :meth:`mma`."""
        return self.mma("wmma_gfx1250_f32_16x16x64_bf8_bf8", a, b, c)

    def mfma_f32_16x16x16_f16(self, a: Value, b: Value, c: Value) -> Value:
        return self.mma("mfma_f32_16x16x16_f16", a, b, c)

    def mfma_f32_16x16x32_f16(self, a: Value, b: Value, c: Value) -> Value:
        """MFMA with K=32 per atom (gfx950 only).

        A and B per-lane operands are `<8 x half>` (vs `<4 x half>` for
        16x16x16); the accumulator stays `<4 x float>`. This halves the
        K-loop trip count for the same per-warp tile, and the wider
        operand load amortises the address arithmetic per MFMA.
        """
        return self.mma("mfma_f32_16x16x32_f16", a, b, c)

    def mfma_f32_16x16x16_bf16(self, a: Value, b: Value, c: Value) -> Value:
        return self.mma("mfma_f32_16x16x16_bf16", a, b, c)

    def mfma_f32_16x16x32_bf16(self, a: Value, b: Value, c: Value) -> Value:
        return self.mma("mfma_f32_16x16x32_bf16", a, b, c)

    def mfma_f32_16x16x32_fp8(self, a: Value, b: Value, c: Value) -> Value:
        return self.mma("mfma_f32_16x16x32_fp8", a, b, c)

    def mfma_f32_16x16x32_bf8(self, a: Value, b: Value, c: Value) -> Value:
        return self.mma("mfma_f32_16x16x32_bf8", a, b, c)

    def mfma_f32_32x32x16_bf16(self, a: Value, b: Value, c: Value) -> Value:
        """gfx950 wider MFMA shape — 32x32 output × 16 K-step per atom.

        Per-lane operand layout:
          - A: ``<8 x bfloat>`` (32 rows × 16 K / 64 lanes = 8 cells)
          - B: ``<8 x bfloat>``
          - C/D: ``<16 x float>`` (32×32 / 64 lanes = 16 outputs per lane)

        The 16-output-per-lane accumulator is what makes Triton's
        long-prefill softmax pattern efficient: per-row max is reduced
        across 4 lanes' worth of in-lane values using chained ``v_max3``
        (15 of them fold 16 values down to 1 per row in 15 instructions),
        and then ONE ``permlane32_swap`` + ``v_max3`` combines the two
        32-lane halves. Total cross-lane traffic for a 64-lane row-reduce
        with this layout: **2 VALU ops + 1 permlane swap**, vs 64
        ``ds_swizzle``/``ds_bpermute`` ops with the 16x16x32 layout.

        Same FLOPs/cycle (1024 cycles per call either way) but half the
        instruction count vs 16x16x32, which (a) halves SIMD issue
        pressure, (b) keeps each lane's accumulator dense for in-lane
        reduce, (c) reduces total ds_read_b128 K-load count.
        """
        return self.mma("mfma_f32_32x32x16_bf16", a, b, c)

    def mfma_f32_32x32x8_f16(self, a: Value, b: Value, c: Value) -> Value:
        """The 32x32x8 f16 MFMA atom — the default warp-tile every CK
        Tile dispatcher config from `default_config.json` uses on wave64.

        Wave64 layout: A is `<4 x half>`, B is `<4 x half>` per lane,
        accumulator is `<16 x float>` per lane (32*32 = 1024 outputs /
        64 lanes). One MFMA per K-step produces 16 floats per lane that
        we then truncate to f16 for the GEMM epilogue.
        """
        return self.mma("mfma_f32_32x32x8_f16", a, b, c)

    def mfma_f32_32x32x8_bf16(self, a: Value, b: Value, c: Value) -> Value:
        """The 32x32x8 bf16 MFMA atom (the ``.1k`` intrinsic) — CDNA3-legal.

        Unlike ``mfma_f32_32x32x16_bf16`` (CDNA4 / gfx950-only; the gfx942
        backend ``Cannot select`` it), the K=8 bf16 32x32 atom IS present on
        gfx942 (it lowers to ``llvm.amdgcn.mfma.f32.32x32x8bf16.1k`` ->
        ``v_mfma_f32_32x32x8_bf16``). It is the wide-fragment double-rate bf16
        path for gfx942 flash attention.

        Wave64 per-lane layout (identical C distribution to the f16 32x32x8 /
        32x32x16 atoms; only K-per-atom differs):
          - A: ``<4 x bfloat>`` (32 rows × 8 K / 64 lanes = 4 cells)
          - B: ``<4 x bfloat>``
          - C/D: ``<16 x float>`` (32×32 / 64 lanes = 16 outputs per lane)
        """
        return self.mma("mfma_f32_32x32x8_bf16", a, b, c)

    def mfma_f32_32x32x16_f16(self, a: Value, b: Value, c: Value) -> Value:
        """The 32x32x16 f16 MFMA atom (gfx950 only).

        Per lane: A `<8 x half>`, B `<8 x half>`, acc `<16 x float>`.
        Doubles K per atom over 32x32x8, halving the K-loop trip count
        for the same per-warp tile and the same accumulator footprint.
        """
        return self.mma("mfma_f32_32x32x16_f16", a, b, c)

    def mfma_f32_32x32x16_fp8(self, a: Value, b: Value, c: Value) -> Value:
        return self.mma("mfma_f32_32x32x16_fp8", a, b, c)

    def mfma_f32_32x32x16_bf8(self, a: Value, b: Value, c: Value) -> Value:
        return self.mma("mfma_f32_32x32x16_bf8", a, b, c)

    def inline_asm(
        self,
        template: str,
        constraints: str,
        operands: "Sequence[Value]" = (),
        result_type: Optional[Type] = None,
        *,
        sideeffect: bool = True,
        convergent: bool = False,
        result_name_hint: str = "asm",
    ) -> Optional[Value]:
        """General AMDGPU inline-asm IR op (ADDITIVE, golden-safe).

        Emits an LLVM ``call <ty> asm sideeffect[ convergent] "<template>",
        "<constraints>"(<typed operands>)`` at lowering time. This is the
        deterministic way to pin a machine instruction's operand register
        classes (e.g. force AGPR srcA/srcB on an MFMA via the ``a``
        constraint) which the typed intrinsics do not let us control.

        Args:
          template: the asm text with ``$0`` (output, if any) then ``$1..$N``
            inputs in the order they appear in ``operands`` (after the output).
          constraints: LLVM constraint string, e.g. ``"=v,0,a,a"``. AMDGPU
            letters: ``v``=VGPR, ``a``=AGPR, ``s``=SGPR in; ``=v``/``=a``/``=s``
            outputs; a digit (``0``) ties an input to that output.
          operands: the input Values, in constraint order (after the output).
          result_type: result Type, or ``None`` for a void asm.
          sideeffect: emit ``sideeffect`` (default True; prevents DCE/dup).
          convergent: also emit ``convergent`` (cannot be moved across waves).

        Returns the result Value, or ``None`` for a void asm.
        """
        rt = [result_type] if result_type is not None else []
        op = self._op(
            "tile.inline_asm",
            list(operands),
            rt,
            attrs={
                "template": str(template),
                "constraints": str(constraints),
                "sideeffect": bool(sideeffect),
                "convergent": bool(convergent),
            },
            result_name_hint=result_name_hint,
        )
        return op.result if result_type is not None else None

    def inline_asm_multi(
        self,
        template: str,
        constraints: str,
        operands: "Sequence[Value]" = (),
        *,
        result_types: "Sequence[Type]" = (),
        sideeffect: bool = True,
        convergent: bool = False,
        result_name_hint: str = "asm",
    ):
        """Multi-output AMDGPU inline-asm op (ADDITIVE, golden-safe).

        Same as :meth:`inline_asm` but for an asm with N (> 1) outputs, which
        LLVM models as a *literal struct* return (``{ <ty0>, <ty1>, ... }``)
        that the lowering unpacks with ``extractvalue`` (the precedent is
        :meth:`permlane32_swap`'s ``{ i32, i32 }`` asm). Returns a list of N
        result Values in declaration order. ``$0..$(N-1)`` are the outputs;
        inputs follow in ``operands`` order starting at ``$N``.

        Used by the nuclear clustered MFMA helper
        (:func:`helpers.asm.mfma_f8f6f4_agpr_cluster`) so a whole gate/up
        MFMA burst is one asm node (one schedule fence) rather than N.
        """
        rts = list(result_types)
        if len(rts) <= 1:
            r = self.inline_asm(
                template,
                constraints,
                operands,
                result_type=(rts[0] if rts else None),
                sideeffect=sideeffect,
                convergent=convergent,
                result_name_hint=result_name_hint,
            )
            return [r] if rts else []
        op = self._op(
            "tile.inline_asm",
            list(operands),
            rts,
            attrs={
                "template": str(template),
                "constraints": str(constraints),
                "sideeffect": bool(sideeffect),
                "convergent": bool(convergent),
            },
            result_name_hint=result_name_hint,
        )
        return list(op.results)

    def mfma_scale_f32_16x16x128_f8f6f4(
        self,
        a: Value,
        b: Value,
        c: Value,
        a_scale: Value,
        b_scale: Value,
    ) -> Value:
        """MX-MFMA with in-instruction scales (P15).

        ``mfma_scale_f32_16x16x128_f8f6f4`` is the gfx950 MX MFMA
        instruction family. Operands are 16-byte packed mantissa
        vectors (fp8/fp6/fp4) plus per-warp E8M0 scales applied
        in-instruction. The scale broadcast inside the instruction is
        per-output-row (vs per-input-row), which is what fixes the
        B_MX1 row-aware correctness gap that the post-hoc
        scale-apply chain has.

        Lowers to ``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4``.
        """
        return self.mma("mfma_scale_f32_16x16x128_f8f6f4", a, b, c, a_scale, b_scale)

    def mfma_f32_16x16x128_fp4(self, a: Value, b: Value, c: Value) -> Value:
        """fp4 MX MFMA (gfx950+, P52).

        Operands are 64-bit packed `<16 x fp4>` per lane;
        accumulator is `<4 x float>`. Scales are applied separately
        via :meth:`mfma_scale_f32_16x16x128_f8f6f4` for the
        production MX path.
        """
        return self.mma("mfma_f32_16x16x128_fp4", a, b, c)

    def mfma_f32_16x16x96_fp6(self, a: Value, b: Value, c: Value) -> Value:
        """fp6 MX MFMA (gfx950+, P52)."""
        return self.mma("mfma_f32_16x16x96_fp6", a, b, c)

    def register_p_from_qk_c(self, qk_c: Value, target_dtype: Type) -> Value:
        """Lane-XOR + bit-transpose to convert a `<16 x f32>` QK-MFMA
        accumulator into a `<8 x dtype>` PV-MFMA A-vector (P13).

        The historical implementation lives inline in
        ``attention_tiled_2d.py:_permute_p_c_to_a16 / _pack_p_a16 /
        _pack_p_a32`` (L1020-1078 of phase-0). This IR-level primitive
        wraps the lane-XOR + bit-transpose sequence so the
        ``use_register_pv`` path can be enabled by default and the
        per-element P_LDS publish is dropped from the kernel body.

        Reference: CK Tile ``block_fmha_fwd_v3_pipeline.hpp:471-479,
        700-734`` and ``MakePRegTileDistribution`` at
        ``block_fmha_fwd_v3_pipeline_default_policy.hpp:178-186``.
        """
        if target_dtype.name not in ("f16", "bf16"):
            raise ValueError(
                f"register_p_from_qk_c target must be f16/bf16, got {target_dtype.name}"
            )
        return self._op(
            "tile.register_p_from_qk_c",
            [qk_c],
            [VectorType(target_dtype, 8)],
            attrs={"target_dtype": target_dtype.name},
            result_name_hint="pa",
        ).result

    def smem_store_distributed(
        self,
        smem: Value,
        layout_attrs: Dict[str, Any],
        values: Value,
    ) -> None:
        """CShuffle-style distributed LDS store (P42).

        Takes a per-lane ``<c_per_lane x dtype>`` accumulator and an
        ``LdsLayout.cshuffle`` descriptor (passed via ``layout_attrs``)
        and writes the lane's slice of the warp tile to LDS in one
        ``ds_write_b<N>``-shaped op. Replaces the per-element
        ``smem_store(..., n=1)`` chain (~64 per-element ops per warp
        tile) in the cshuffle epilogue.

        Reference: CK Tile ``cshuffle_epilogue.hpp:316-384, 661-759``.
        """
        return self._op(
            "tile.smem_store_distributed",
            [smem, values],
            attrs=dict(layout_attrs),
        )

    def cooperative_global_store(
        self,
        ptr: Value,
        addrs: Value,
        values: Value,
    ) -> None:
        """Cooperative global store with per-lane address (P14).

        Takes a per-lane vector of f32 values + a matching per-lane
        vector of i32 addresses. Issues the per-lane stores while
        letting the AMDGPU backend coalesce adjacent lanes into one
        global_store_dwordxN transaction. Replaces the per-lane
        ``global_store(workspace_acc, …)`` chain in
        ``attention_tiled_3d.py:854``.
        """
        return self._op(
            "memref.cooperative_global_store",
            [ptr, addrs, values],
            attrs={
                "vec": (
                    int(values.type.count) if isinstance(values.type, VectorType) else 1
                )
            },
        )

    def mfma_f32_4x4x4_f16(self, a: Value, b: Value, c: Value) -> Value:
        """The 4x4x4 f16 MFMA atom — 16 independent 4x4 matmuls per wave.

        Per lane on wave64: A `<4 x half>`, B `<4 x half>`, acc
        `<4 x float>`. The single MFMA emits 16 independent 4x4
        matrix products in one go, indexed by `batch = lane / 4`.
        This is what our small-channel direct-conv kernel uses: treat
        `batch` as a group-in-workgroup index and run 16 independent
        4-channel convolution groups inside one wave.

        Lowers to `@llvm.amdgcn.mfma.f32.4x4x4f16` (3 immarg).
        """
        return self.mma("mfma_f32_4x4x4_f16", a, b, c)

    # ----- vector type casts (for packed buffer-load + LDS reads) -----

    def vec_bitcast(self, v: Value, target: Type) -> Value:
        """Bitcast a vector value to a target vector type of equal size.

        Used to flip between `<N x i32>` (the result of
        `buffer_load_dwordxN`) and `<2N x f16>` (the form the MFMA wants),
        or `<N x i32>` (raw 32-bit dwords) and `<N x float>`. The LLVM
        lowering emits an `addrspacecast`-less `bitcast`; the gfx
        backend folds it away.
        """
        return self._op(
            "vector.bitcast",
            [v],
            [target],
            attrs={"target": target.name},
            result_name_hint="bc",
        ).result

    # ----- uniform / wave-scalar helpers -----

    def readfirstlane(self, v: Value) -> Value:
        """`@llvm.amdgcn.readfirstlane(v)` — make `v` scalar (SGPR) by
        broadcasting lane 0's value across the whole wave.

        Use when computing an address that *is* the same across the
        wave (e.g. the LDS base byte offset for the current wave when
        issuing `raw_ptr_buffer_load_lds`). Without this the compiler
        may keep the address in VGPRs and refuse to emit the scalar
        form of the LDS instruction.
        """
        return self._op(
            "tile.readfirstlane",
            [v],
            [v.type],
            result_name_hint="ufm",
        ).result

    def pin_sgpr(self, v: Value) -> Value:
        """No-op asm constraint that forces ``v`` to stay in an SGPR
        across uses.

        Emits the AMDGPU idiom ``asm volatile("" : "+s"(x))`` —
        without it the register allocator may copy a value produced
        by :meth:`readfirstlane` back into a VGPR before re-using it
        across many uses (e.g. an SGPR LDS-base cursor that we bump
        with ``s_add_u32`` across an unrolled K-loop). Pinning saves
        both the round-trip ``v_readfirstlane_b32`` and the VGPR
        pressure on the LDS-base.

        Typical usage::

        lds_base = b.pin_sgpr(b.readfirstlane(b.cast_i32(addr)))
        # ... subsequent uses of lds_base land in SGPR-only paths ...
        """
        return self._op(
            "tile.pin_sgpr",
            [v],
            [v.type],
            result_name_hint="sgpr",
        ).result

    def to_sgpr_u32(self, v: Value) -> Value:
        """Convenience: ``pin_sgpr(readfirstlane(v))``.

        The canonical AMDGPU "lift this value into scalar registers"
        pattern. Use whenever you have a wave-uniform i32 (an LDS
        base, a global byte offset, a tile coord) that you want to
        keep in scalar registers across many uses.
        """
        return self.pin_sgpr(self.readfirstlane(v))

    def wave_all(self, predicate: Value) -> Value:
        """Wave-wide AND vote: ``__all_sync``-style ``i32`` result.

        Returns a wave-uniform i32 that is 1 iff *every* lane's
        ``predicate`` was non-zero, 0 otherwise. Lowered to
        ``__builtin_amdgcn_read_exec()`` AND/compare on AMDGPU
        (a single ``s_or_b64`` / ``s_cmp_eq`` pair) — no ds_bpermute,
        no LDS round-trip.

        Pairs naturally with adaptive online-softmax rescaling: when
        ``wave_all(max_diff < THRESHOLD)`` is 1, the workgroup can skip
        the rescale path entirely.
        """
        return self._op(
            "tile.wave_all",
            [predicate],
            [I32],
            result_name_hint="wave_all",
        ).result

    def wave_any(self, predicate: Value) -> Value:
        """Wave-wide OR vote: 1 iff *any* lane's predicate is non-zero.

        Lowered to a wave ballot + non-zero check. Useful for early
        bailout — e.g. "are any cells of this attention row finite?".
        """
        return self._op(
            "tile.wave_any",
            [predicate],
            [I32],
            result_name_hint="wave_any",
        ).result

    def wave_ballot(self, predicate: Value) -> Value:
        """Wave-wide ballot: returns a 64-bit mask of which lanes
        satisfied the predicate.

        Lowered to ``__builtin_amdgcn_ballot_w64`` (or the wave32
        equivalent on gfx12). For boolean reductions prefer the
        higher-level :meth:`wave_all` / :meth:`wave_any` which avoid
        forcing the mask materialisation on the consumer side.
        """
        return self._op(
            "tile.wave_ballot",
            [predicate],
            [I64],
            result_name_hint="ballot",
        ).result

    def ds_bpermute(self, addr: Value, data: Value) -> Value:
        """`__builtin_amdgcn_ds_bpermute(addr, data)` — wave64 cross-lane
        broadcast permute, using LDS as the shuffle vehicle.

        `addr` is a per-lane 32-bit value where bits [7:2] index the source
        lane (i.e. each lane reads from `lane = addr >> 2`); high bits are
        ignored. `data` is the per-lane 32-bit payload to broadcast.

        CK Tile's `warp_shuffle_*` helpers wrap this primitive
        (see `core/arch/utility.hpp::warp_shuffle_down`/`warp_shuffle`).
        We expose it directly because the kernel author already knows the
        target lane and the bit-cast of `data`.

        Both `addr` and `data` must be `i32`. For non-i32 payloads, callers
        should bitcast first.

        **Performance note (gfx950).** For wave64 XOR butterflies within
        each 16-lane row-group (the standard MFMA-16x16x32 softmax
        reduction pattern), prefer :meth:`ds_swizzle_xor` which uses the
        FFT-mode swizzle and is one LDS instruction vs ds_bpermute's
        2 instructions (vgpr address compute + ds_bpermute). Measured on
        the long-prefill ``bf16 n=16 q=1000 k=1050`` workload: ds_swizzle
        cut the inner-loop ds-unit ops from 64 → 16 and shaved 30+ µs
        per call. See the ISA op-by-op comparison in
        ``ck/dsl/unified_attention_results.md``.
        """
        if addr.type.name != "i32" or data.type.name != "i32":
            raise ValueError("ds_bpermute requires i32 addr + i32 data")
        return self._op(
            "tile.ds_bpermute",
            [addr, data],
            [I32],
            result_name_hint="bp",
        ).result

    def ds_swizzle_xor(self, data: Value, xor_mask: int) -> Value:
        """`__builtin_amdgcn_ds_swizzle(data, offset)` — wave-internal lane
        permutation in a single LDS instruction.

        Performs the cross-lane XOR-mask permutation: lane ``L`` reads from
        lane ``(L & 0x1F) ^ xor_mask`` within its 32-lane half (lanes 0-31
        permute independently from lanes 32-63 — wave64 ds_swizzle is
        "intra-32-lane" by design). For our attention kernel this is
        always called inside a single MFMA row-group of 16 lanes which is
        a subset of one 32-lane half, so the intra-half constraint is
        always satisfied.

        Compared to ``ds_bpermute``:
          - **1 LDS instruction** (vs 2: vgpr-addr-compute + ds_bpermute)
          - **No VGPR pressure** for an address vector
          - **Encoded mask** in the immediate offset (vs runtime addr)

        This is the gfx950-friendly equivalent of RDNA's ``v_permlanex16``
        XMASK DPP mode (which is RDNA-only — gfx950 rejects the encoding,
        verified empirically on gfx950). The data sheet calls the
        encoding "FFT mode": ``offset = 0x8000 | (0x1F << 10) | xor_mask``
        which selects ``and_mask=0x1F or_mask=0 xor_mask=k`` —
        ``lane_dst = (lane_src & 0x1F) | 0 ^ k`` = ``lane_src ^ k`` modulo
        32-lane half.

        ``xor_mask`` must be in 1..31. For wider XOR masks that need to
        cross the 32-lane boundary, use ``permlane32_swap`` (only valid
        for full-half exchange, mask=32).
        """
        if data.type.name != "i32":
            raise ValueError("ds_swizzle_xor requires i32 data")
        if not (1 <= xor_mask <= 31):
            raise ValueError(
                f"ds_swizzle_xor xor_mask must be 1..31 (intra-32-lane), got {xor_mask}"
            )
        return self._op(
            "tile.ds_swizzle_xor",
            [data],
            [I32],
            attrs={"xor_mask": int(xor_mask)},
            result_name_hint="sw",
        ).result

    def mov_dpp(
        self,
        data: Value,
        *,
        row_shr: Optional[int] = None,
        row_shl: Optional[int] = None,
        bound_ctrl: bool = False,
    ) -> Value:
        """``v_mov_b32_dpp`` row-shift primitive (gfx9+).

        ``row_shr={1,2,4,8,15}`` shifts each lane's value RIGHT inside
        its 16-lane row-group (lanes 0..15, 16..31, …). ``row_shl`` is
        the symmetric LEFT shift. ``bound_ctrl=True`` zero-fills lanes
        that would shift in from outside the row-group (default: passes
        the consumer's old VGPR through).

        Cycle cost on AMDGPU is ~1 (vs ~3 for ``ds_bpermute``); used by
        AITER / CK Tile cumsum + warp-XOR scan kernels for the inner
        row-shift stage. Currently only one of ``row_shr`` / ``row_shl``
        may be set per call.
        """
        if (row_shr is None) == (row_shl is None):
            raise ValueError(
                "mov_dpp requires exactly one of row_shr / row_shl to be set"
            )
        if row_shr is not None and row_shr not in (1, 2, 4, 8, 15):
            raise ValueError(
                f"mov_dpp row_shr must be in {{1,2,4,8,15}}, got {row_shr}"
            )
        if row_shl is not None and row_shl not in (1, 2, 4, 8, 15):
            raise ValueError(
                f"mov_dpp row_shl must be in {{1,2,4,8,15}}, got {row_shl}"
            )
        if data.type.name != "i32":
            raise ValueError("mov_dpp requires i32 data")
        attrs = {"bound_ctrl": bool(bound_ctrl)}
        if row_shr is not None:
            attrs["row_shr"] = int(row_shr)
        else:
            attrs["row_shl"] = int(row_shl)
        return self._op(
            "tile.mov_dpp",
            [data],
            [I32],
            attrs=attrs,
            result_name_hint="dpp",
        ).result

    def dpp_xor(self, data: Value, xor_mask: int) -> Value:
        """``v_mov_b32_dpp`` ``row_xmask`` — cross-lane XOR permute on the
        **VALU** (NOT the LDS port).

        Lane ``L`` reads ``data`` from lane ``(L & ~0xF) | ((L & 0xF) ^
        xor_mask)`` — i.e. an XOR butterfly *within* each 16-lane DPP row.
        ``xor_mask`` must be in 1..15 so the partner stays in the same row.

        This is the RDNA (gfx11/gfx12) replacement for the gfx9/gfx95
        :meth:`ds_swizzle_xor` softmax-reduction butterfly. ``ds_swizzle``
        is an LDS-bus op that costs ``lgkmcnt`` and serialises on the LDS
        port; ``row_xmask`` runs entirely in the VALU pipeline, and LLVM's
        DPP-combine folds a single-use ``update.dpp`` + ``fmax``/``fadd``
        into one ``v_max_f32_dpp`` / ``v_add_f32_dpp``. Lowered through
        ``llvm.amdgcn.update.dpp.i32`` with ``dpp_ctrl = 0x160 | mask``
        (the AMDGPU ``DPP_ROW_XMASK`` encoding). Data must be ``i32``."""
        if data.type.name != "i32":
            raise ValueError("dpp_xor requires i32 data")
        if not (1 <= xor_mask <= 15):
            raise ValueError(
                f"dpp_xor xor_mask must be 1..15 (intra-16-lane row), got {xor_mask}"
            )
        return self._op(
            "tile.dpp_xor",
            [data],
            [I32],
            attrs={"xor_mask": int(xor_mask)},
            result_name_hint="dppx",
        ).result

    def ds_bpermute_b64(self, addr: Value, data: Value) -> Value:
        """Packed 64-bit ``ds_bpermute`` — single LDS op for paired
        ``(val, idx)`` cross-lane shuffles (gfx9+).

        ``addr`` is the per-lane source-lane index (same encoding as
        :meth:`ds_bpermute`). ``data`` must be ``i64`` (typically built
        from ``vec_concat(i32, i32)`` or a bitcast of two ``i32`` lanes
        of payload). Returns the permuted ``i64``; consumer can split
        back into the ``(val, idx)`` pair.

        Halves the LDS-op count vs two separate ``ds_bpermute`` calls
        in topk-softmax / argmax butterflies (``_wave_argmax_butterfly``).
        """
        if addr.type.name != "i32":
            raise ValueError("ds_bpermute_b64 requires i32 addr")
        if data.type.name != "i64":
            raise ValueError("ds_bpermute_b64 requires i64 data")
        return self._op(
            "tile.ds_bpermute_b64",
            [addr, data],
            [I64],
            result_name_hint="bp64",
        ).result

    def permlane32_swap(self, lo: Value, hi: Value):
        """`__builtin_amdgcn_permlane32_swap(lo, hi)` — swap the values
        of two registers across the 32-lane half boundary.

        After the swap: lane ``L`` in [0..31] gets the input ``hi[L+32]``
        in its ``lo`` register and ``lo[L]`` in its ``hi``; lane ``L`` in
        [32..63] gets the symmetric swap. Used for the FINAL stage of a
        wave64 softmax reduction after in-lane (or intra-32-lane) max
        chains: combines the per-row max from lanes [0..31] with the
        per-row max from lanes [32..63] in a SINGLE VALU instruction.

        Triton's long-prefill kernel uses exactly this pattern (16
        chained ``v_max3`` + 1 ``v_permlane32_swap_b32`` + 1 ``v_max3``
        per row-reduction). We use it for the same purpose in our
        rewritten 32x32x16 MFMA path.

        Returns the two swapped values as a tuple ``(new_lo, new_hi)``;
        both inputs must be ``i32`` (bitcast f32 → i32 first if needed).
        """
        if lo.type.name != "i32" or hi.type.name != "i32":
            raise ValueError("permlane32_swap requires i32 operands")
        op = self._op(
            "tile.permlane32_swap",
            [lo, hi],
            [I32, I32],
            result_name_hint="psw",
        )
        return op.results[0], op.results[1]

    def perm_b32(self, src0: Value, src1: Value, sel: Value) -> Value:
        """`__builtin_amdgcn_perm(src0, src1, sel)` = ``v_perm_b32`` — an
        in-lane byte-select across two source VGPRs (NO cross-lane, NO
        ``lgkmcnt``, pure VALU). The 8 bytes ``[src1 (bytes 0..3), src0
        (bytes 4..7)]`` are addressed by the four nibbles... err, four
        bytes of ``sel``: result byte ``i`` = byte ``(sel >> (8*i)) &
        0xFF`` of the concatenated 8-byte source (selector value 0..7
        picks a source byte; >=8 yields a sign/zero pattern per the ISA).

        This is the op CK's ``transpose_vectors`` uses for the in-register
        2x2 (f16) / 4x4 (i8) operand transpose at LDS-store time. We expose
        it so the gfx942 conflict-free-V store path can reshape V from its
        B-natural ``[HD,T+pad]`` layout into the MFMA-A operand register
        arrangement WITHOUT the ``ds_swizzle``/``warp_shuffle_xor`` L2
        serialization wall.

        All three operands must be ``i32`` (bitcast/pack f16 pairs to i32
        first). Returns the permuted ``i32``.
        """
        if src0.type.name != "i32" or src1.type.name != "i32" or sel.type.name != "i32":
            raise ValueError("perm_b32 requires i32 operands")
        return self._op(
            "tile.perm_b32",
            [src0, src1, sel],
            [I32],
            result_name_hint="perm",
        ).result

    def permlanex16(self, v: Value) -> Value:
        """`__builtin_amdgcn_permlanex16(old, v, 0x76543210, 0xfedcba98,
        false, true)` — swap each lane's value with its ``lane ^ 16``
        partner within a 32-lane group.

        This is a permute-NETWORK VALU op (NOT an LDS/ds_bpermute op): the
        two 32-bit selector immediates (``0x76543210`` for lanes 0..15,
        ``0xfedcba98`` for lanes 16..31) request, for each destination
        lane, the source lane ``L ^ 16``. ``bound_ctrl=true`` writes every
        destination lane, so the ``old`` operand is a don't-care (we reuse
        ``v``); full warps are active so ``fi=false`` is enough for both
        halves to see each other (matches CK's gfx11 usage exactly).

        This is the cheap cross-half vehicle CK's gfx11 FMHA pipelines use
        for the C→A transpose (`PermuteWarpGemmCToA`); a single VALU op,
        no LDS round-trip, no barrier. Operand must be ``i32``.
        """
        if v.type.name != "i32":
            raise ValueError("permlanex16 requires an i32 operand")
        return self._op(
            "tile.permlanex16",
            [v],
            [I32],
            result_name_hint="plx16",
        ).result

    def byte_perm(self, a: Value, b: Value, sel: int) -> Value:
        """`__builtin_amdgcn_perm(a, b, sel)` — the ``v_perm_b32`` byte
        shuffle: build a 32-bit result by selecting four bytes out of the
        eight source bytes ``{b.b0,b.b1,b.b2,b.b3, a.b0,a.b1,a.b2,a.b3}``
        (indices 0..3 = ``b`` LSB-first, 4..7 = ``a`` LSB-first).

        ``sel`` is a packed i32: result byte ``k`` (k=0 is LSB) is the
        source byte numbered ``(sel >> 8*k) & 0xff``. E.g.
        ``perm(0x11223344, 0x55667788, 0x05010400) == 0x33774488``.

        Used to interleave even/odd k0 byte-codes when assembling the
        fused conv1 A-fragment from the conv0 accumulator. ``a`` and ``b``
        must be ``i32``; ``sel`` is a compile-time constant.
        """
        if a.type.name != "i32" or b.type.name != "i32":
            raise ValueError("byte_perm requires i32 operands")
        return self._op(
            "tile.byte_perm",
            [a, b],
            [I32],
            attrs={"sel": int(sel) & 0xFFFFFFFF},
            result_name_hint="bperm",
        ).result

    def lane_id(self) -> Value:
        """`@llvm.amdgcn.mbcnt.hi(-1, @llvm.amdgcn.mbcnt.lo(-1, 0))` — the
        wave64 lane index (0..63) for the current thread.

        This is equivalent to `threadIdx.x % 64` when the workgroup has at
        most one wave, but is more direct (a single VALU op) for kernels
        that want a true lane index regardless of workgroup size.
        """
        return self._op("tile.lane_id", [], [I32], result_name_hint="lane").result

    def bitcast(self, v: Value, target: Type) -> Value:
        """Bitcast a scalar to another type of the same size."""
        return self._op(
            "arith.bitcast",
            [v],
            [target],
            attrs={"target": target.name},
            result_name_hint="bc",
        ).result

    def warp_shuffle_xor(self, v: Value, lane_xor: int) -> Value:
        """Cross-lane shuffle: lane `l` gets `v` from lane `l ^ lane_xor`.

        For **intra-32-lane XOR** (``lane_xor in 1..31``), uses
        :meth:`ds_swizzle_xor` — a single LDS instruction with the
        permute pattern encoded in the immediate offset. No VGPR
        address-computation pair (xor + shl) needed and the LDS unit
        completes in fewer cycles.

        For wave-wide swaps (``lane_xor in {32}``) — not used by the
        16-lane row-group butterfly attention does, but listed for
        completeness — falls back to ``ds_bpermute`` (the swizzle unit
        only permutes within a 32-lane half; cross-half needs
        permlane32_swap or ds_bpermute).

        Works for any 32-bit scalar `v` (f32, i32). For half/bfloat,
        bitcast to i32 via a 2-element vector first.
        """
        if 1 <= lane_xor <= 31:
            # Emits `ds_swizzle` (XOR pattern in the immediate offset), NOT
            # `ds_bpermute`: 1 LDS op, no addr-compute. This is the path the
            # attention softmax reduction's 16-lane MFMA row-group butterfly
            # actually takes (lane_xor in 1..16). ds_bpermute is only the
            # wave-wide (lane_xor >= 32) fallback below.
            if v.type.name == "f32":
                v_i = self.bitcast(v, I32)
                r = self.ds_swizzle_xor(v_i, int(lane_xor))
                return self.bitcast(r, F32)
            if v.type.name == "i32":
                return self.ds_swizzle_xor(v, int(lane_xor))
            raise ValueError(f"warp_shuffle_xor: unsupported type {v.type.name}")
        # Wave-wide XOR (lane_xor >= 32): swizzle only does intra-32-lane,
        # so fall back to ds_bpermute. Kept available so callers that
        # need 64-lane XOR (e.g. cross-wave reductions) still work.
        lane = self.lane_id()
        xor_const = self.const_i32(int(lane_xor))
        addr = self._op(
            "arith.xor",
            [lane, xor_const],
            [I32],
            result_name_hint="lxor",
        ).result
        addr_shl = self._op(
            "arith.shl",
            [addr, self.const_i32(2)],
            [I32],
            result_name_hint="laddr",
        ).result
        if v.type.name == "f32":
            v_i = self.bitcast(v, I32)
            r = self.ds_bpermute(addr_shl, v_i)
            return self.bitcast(r, F32)
        if v.type.name == "i32":
            return self.ds_bpermute(addr_shl, v)
        raise ValueError(f"warp_shuffle_xor: unsupported type {v.type.name}")

    def vop2_f32_dpp_xor(self, v: Value, xor_mask: int, mnemonic: str) -> Value:
        """One **fused** VALU reduce step: ``<mnemonic>_dpp d, v, v
        row_xmask:mask`` — the cross-lane XOR shuffle *and* the f32 combine in
        a single instruction (``v_max_f32_dpp`` / ``v_add_f32_dpp``).

        This is the fused form of a softmax row-reduce stage. LLVM's
        GCNDPPCombine pass refuses to fold a ``row_xmask`` ``v_mov_b32_dpp``
        into the consuming ``v_max``/``v_add`` (verified on gfx1250: the mov
        survives), so we emit the fused op directly via inline asm — halving
        the VALU op count of the reduction vs the unfused
        :meth:`warp_shuffle_xor_dpp` + ``fmax``/``fadd``.

        The DPP modifier applies to the first source, so ``d = mnemonic(
        shuffle(v), v)`` — correct for the commutative max/add reduction.
        ``convergent`` (reads other lanes), ``bound_ctrl:1`` (partner always
        in-row for mask 1..15). Operand + result are f32."""
        if v.type.name != "f32":
            raise ValueError("vop2_f32_dpp_xor requires f32 data")
        if not (1 <= xor_mask <= 15):
            raise ValueError(f"vop2_f32_dpp_xor xor_mask must be 1..15, got {xor_mask}")
        if mnemonic not in ("v_max_f32", "v_add_f32"):
            raise ValueError(
                f"vop2_f32_dpp_xor mnemonic must be v_max_f32/v_add_f32, "
                f"got {mnemonic!r}"
            )
        tmpl = (
            f"{mnemonic}_dpp $0, $1, $1 "
            f"row_xmask:{xor_mask} row_mask:0xf bank_mask:0xf bound_ctrl:1"
        )
        return self.inline_asm(
            tmpl,
            "=v,v",
            [v],
            result_type=F32,
            sideeffect=False,
            convergent=True,
            result_name_hint="dppr",
        )

    def warp_shuffle_xor_dpp(self, v: Value, lane_xor: int) -> Value:
        """VALU XOR shuffle (lane ``l`` gets ``v`` from lane ``l ^ lane_xor``)
        via :meth:`dpp_xor` ``row_xmask`` — the RDNA replacement for the
        :meth:`warp_shuffle_xor` ``ds_swizzle`` path.

        ``lane_xor`` must be in 1..15 (the XOR partner stays inside the
        16-lane DPP row, which is exactly the WMMA 16-lane softmax
        row-group). Runs on the VALU (no LDS port / ``lgkmcnt``); LLVM's
        DPP-combine folds the single-use result into the consuming
        ``v_max_f32``/``v_add_f32`` to give one ``v_*_f32_dpp`` per stage.
        Handles f32 by bitcasting through i32 (matching
        :meth:`warp_shuffle_xor`)."""
        if not (1 <= lane_xor <= 15):
            raise ValueError(
                f"warp_shuffle_xor_dpp lane_xor must be 1..15, got {lane_xor}"
            )
        if v.type.name == "f32":
            v_i = self.bitcast(v, I32)
            r = self.dpp_xor(v_i, int(lane_xor))
            return self.bitcast(r, F32)
        if v.type.name == "i32":
            return self.dpp_xor(v, int(lane_xor))
        raise ValueError(f"warp_shuffle_xor_dpp: unsupported type {v.type.name}")

    def ds_read_tr16_b64(
        self, smem: Value, *indices: Value, dtype: Type = F16
    ) -> Value:
        """`ds_read_b64_tr_b16` — wave64 transpose-read of a 16x16 fp16 tile
        from LDS, returning the MFMA B-operand layout directly.

        Semantics (gfx950 wave64):
        - The LDS region at `smem[indices..., 0]` is interpreted as a
        16-row x 16-column matrix of fp16 (row-major, 32 bytes per row,
        256 bytes total).
        - After the read, lane `l = 16 * k_chunk + n` (k_chunk in 0..3,
        n in 0..15) holds 4 fp16 values:
        tile[k_chunk*4 + 0, n],
        tile[k_chunk*4 + 1, n],
        tile[k_chunk*4 + 2, n],
        tile[k_chunk*4 + 3, n]
        - Exactly the per-lane B operand of `v_mfma_f32_16x16x16_f16`.

        Use case: PV gemm where `V[T, HD]` is in LDS row-major and we want
        `B[k_chunk*4 + 0..3, n]` per lane without 4 strided `ds_read_u16`.
        """
        if not indices:
            raise ValueError("ds_read_tr16_b64 needs at least one index")
        return self._op(
            "tile.ds_read_tr16_b64",
            [smem, *indices],
            [VectorType(dtype, 4)],
            attrs={"rank": len(indices), "elem_type": dtype.name},
            result_name_hint="tr16",
        ).result

    def ds_read_tr16_b128(
        self, smem: Value, *indices: Value, dtype: Type = F16
    ) -> Value:
        """``ds_read_b128_tr_b16`` — gfx950 wide transpose-read.

        Same shape as :meth:`ds_read_tr16_b64` but reads 16 bytes per
        lane instead of 8. Per-lane result is ``<8 x f16>`` (or bf16),
        delivering the MFMA B-operand layout for the K-packed atoms
        (``f16_16x16x32`` / ``bf16_16x16x32``) in a single LDS op
        instead of two paired ``b64`` reads.

        Used by transpose phase-2 and the PV-V LDS read in
        ``attention_tiled_2d`` 32x32 path: 8 scalar ``ds_read_u16``
        per lane → 1 transposed read.
        """
        if not indices:
            raise ValueError("ds_read_tr16_b128 needs at least one index")
        return self._op(
            "tile.ds_read_tr16_b128",
            [smem, *indices],
            [VectorType(dtype, 8)],
            attrs={"rank": len(indices), "elem_type": dtype.name},
            result_name_hint="tr16w",
        ).result

    def ds_read_tr_b8(
        self, smem: Value, *indices: Value, dtype: Type = FP8E4M3
    ) -> Value:
        """`ds_read_b64_tr_b8` — gfx950 transpose-read for 8-bit LDS tiles.

        Returns ``<8 x fp8e4m3>`` (or ``<8 x bf8e5m2>``) per lane: the
        operand shape needed by ``mfma_f32_16x16x32_fp8``. This is the
        8-bit sibling of :meth:`ds_read_tr16_b64`, exposed through inline
        asm in the LLVM lowering because ROCm 7.0's public intrinsic list
        only exposes the b16 variant.
        """
        if dtype.name not in ("fp8e4m3", "bf8e5m2", "i8"):
            raise ValueError(
                f"ds_read_tr_b8 expects fp8/bf8/i8 dtype, got {dtype.name}"
            )
        if not indices:
            raise ValueError("ds_read_tr_b8 needs at least one index")
        return self._op(
            "tile.ds_read_tr_b8",
            [smem, *indices],
            [VectorType(dtype, 8)],
            attrs={"dtype": dtype.name},
            result_name_hint="tr8",
        ).result

    # ----- LDS pointer arithmetic (for per-wave async-LDS bases) -----

    def smem_addr_of(self, smem: Value) -> Value:
        """The base i64 LDS address of an `smem_alloc` allocation.

        Equivalent to `__builtin_amdgcn_get_local_pointer(smem)`
        followed by `(uintptr_t)ptr`. Returns an i64 so caller can do
        scalar address arithmetic.
        """
        return self._op(
            "tile.smem_addr_of",
            [smem],
            [I64],
            result_name_hint="lds_addr",
        ).result

    def smem_ptr_add(self, lds_addr: Value, byte_off: Value) -> Value:
        """Compute `lds_addr + byte_off` and return an i64 LDS address.

        Used to derive a per-wave base for `async_buffer_load_lds` so
        every wave writes into its own LDS region (the intrinsic always
        writes lane-contiguously, so wave-disambiguation has to live
        in the base).
        """
        return self._op(
            "tile.smem_ptr_add",
            [lds_addr, byte_off],
            [I64],
            result_name_hint="lds_addr",
        ).result

    def async_buffer_load_lds_addr(
        self,
        rsrc: Value,
        lds_addr: Value,
        voffset: Value,
        soffset: Value,
        dwords: int,
        coherency: int = 0,
    ) -> None:
        """Variant of `async_buffer_load_lds` that takes a raw i64 LDS
        address instead of a typed `smem<...>` value. This is the form
        that lets you pass a per-wave-offset LDS pointer to the
        intrinsic.

        Args:
        coherency: AMDGPU buffer-load AUX bits (0..3). One of
        :data:`CACHE_ALL` (0, default), :data:`CACHE_GLOBAL`
        (1, GLC set — skip L2), :data:`CACHE_STREAM` (2, SLC
        set), :data:`NON_TEMPORAL` (3, GLC|SLC). The AUX field
        lives in the last argument of
        ``llvm.amdgcn.raw.ptr.buffer.load[.lds]`` and biases
        the L1/L2 caching policy. ``CACHE_STREAM`` is the
        right hint for one-shot streaming loads in a
        ping-pong pipeline that won't reuse the data.
        """
        if dwords not in (1, 3, 4):
            raise ValueError(
                f"async_buffer_load_lds_addr dwords must be 1, 3, or 4 (got {dwords})"
            )
        if coherency not in (0, 1, 2, 3):
            raise ValueError(f"coherency must be 0..3 (got {coherency})")
        self._op(
            "tile.async_buffer_load_lds_addr",
            [rsrc, lds_addr, voffset, soffset],
            attrs={"dwords": int(dwords), "aux": int(coherency)},
        )

    # ----- scheduler / synchronisation hints -----

    def sync(self) -> None:
        self._op("tile.sync")

    def s_barrier_bare(self) -> None:
        """Bare workgroup barrier: ``s_barrier`` with NO implicit waitcnt.

        Unlike :meth:`sync` (which prepends ``s_waitcnt vmcnt(0) lgkmcnt(0)``)
        and :meth:`sync_lds_only` (which prepends ``lgkmcnt(0)``), this emits
        ONLY ``llvm.amdgcn.s.barrier()``. The caller controls the wait
        counters explicitly. This is required by the warp-specialized
        producer/consumer pipeline (``wsp3``), where the per-iteration
        rendezvous must NOT drain the producers' in-flight async global->LDS
        loads (those are gated by an explicit ``s_waitcnt(vmcnt=0)`` placed
        by the caller just before this barrier). Named/split barriers
        (``s.barrier.signal``/``wait``) ICE the gfx950 backend, so a bare
        full-CTA barrier is the mechanism (the warp-specialized reference pattern).
        """
        self._op("tile.s_barrier_bare")

    def sync_half_block(self, half_selector: Value) -> None:
        """Half-block barrier: only the waves where ``half_selector``
        is non-zero participate in the workgroup barrier.

        Emits the AMDGPU idiom
        ``if (selector) __builtin_amdgcn_s_barrier()``. The ping-pong
        / interwave pattern partitions an N-wave block into two
        halves (typically ``stagger = warpid() / (N/2)``), and
        half-block barriers let one half synchronise on each of two
        independently-progressing pipelines without forcing the whole
        block to converge.

        Caveats:
        * The non-participating waves must NOT reach this point — the
        caller is responsible for ensuring all waves either enter
        this barrier or the matching companion barrier (e.g. on the
        ``stagger=0`` half). If only some waves reach an unmatched
        half-block barrier the HW will deadlock.
        * Always pair with a full :meth:`sync` at the start and end
        of the cluster so the two halves rejoin cleanly.

        Parameters
        ----------
        half_selector
        i32 SSA. Non-zero -> this wave participates in the barrier.
        Typical use: ``b.cmp_ne(stagger, b.const_i32(0))`` where
        ``stagger = warpid() / (num_warps / 2)``.
        """
        self._op("tile.sync_half_block", [half_selector])

    def sync_lds_only(self) -> None:
        """Workgroup barrier that drains LDS ops but NOT VMEM.

        Emits ``s_waitcnt lgkmcnt(0)`` followed by ``s_barrier`` — the
        canonical CK Tile ``block_sync_lds`` pattern. Use this in
        async-DMA pipelines where an in-flight ``raw_ptr_buffer_load_lds``
        (a VMEM op) must keep streaming while the consumer waits for
        prior ``ds_read``/``ds_write`` to settle.

        Versus :meth:`sync`: this skips ``vmcnt(0)`` so the next iter's
        async load stays in flight across the barrier, which is the
        whole point of the ping-pong overlap.
        """
        self._op("tile.sync_lds_only")

    def s_waitcnt(
        self, *, vmcnt: int = -1, lgkmcnt: int = -1, expcnt: int = -1
    ) -> None:
        """Insert an `s_waitcnt`. Pass -1 to leave a counter alone (no
        wait); pass 0 to fully drain that counter.

        Usage in the compv4 pipeline:
        - after issuing the async DRAM->LDS loads for the next K-tile,
        insert `s_waitcnt(vmcnt=0)` *just before* the MFMAs that
        consume the freshly-arrived data;
        - after issuing `ds_read`s, `s_waitcnt(lgkmcnt=0)` ensures
        the LDS data is in registers before MFMA.

        AMDGPU bit encoding is handled by the lowerers. For gfx950 the
        important detail is that VMCNT is 6 bits split across bits [3:0]
        and [15:14], so values such as ``vmcnt=16`` are valid partial
        waits and must not be masked down to zero. We default the not-set
        counters to their max value (no wait).
        """
        self._op(
            "tile.s_waitcnt",
            attrs={"vmcnt": int(vmcnt), "lgkmcnt": int(lgkmcnt), "expcnt": int(expcnt)},
        )

    def s_wait_asynccnt(self, n: int = 0) -> None:
        """Wait until at most ``n`` gfx1250 async global->LDS copies remain
        outstanding (``s_wait_asynccnt n``).

        gfx1250 (GFX12 / gfx1250) tracks ``global_load_async_to_lds`` /
        ``global_store_async_from_lds`` completion on a DEDICATED ``ASYNCcnt``
        counter — separate from ``loadcnt`` (VMEM register loads) and ``dscnt``
        (LDS ds_read/ds_write). This gives true partial-drain control for an
        async-DMA ping-pong: issue the next tile's copies, then
        ``s_wait_asynccnt(n_next)`` to drain the *current* tile while the next
        tile's copies keep streaming. ``n=0`` fully drains.

        On non-gfx1250 backends this is a no-op (the counter does not exist);
        callers must only emit it on the gfx1250 async-to-LDS path.
        """
        self._op("tile.s_wait_asynccnt", attrs={"n": int(n)})

    def global_load_async_to_lds(
        self,
        src_ptr: Value,
        src_index: Value,
        lds_smem: Value,
        lds_indices,
        *,
        width_bytes: int,
        coherency: int = 0,
        offset_bytes: int = 0,
    ) -> None:
        """gfx1250 async global->LDS copy (``global_load_async_to_lds_b{32,64,128}``).

        Each lane copies ``width_bytes`` (4/8/16) from
        ``src_ptr[src_index]`` (a ``global`` element offset) into
        ``lds_smem[*lds_indices]`` (a typed LDS element slot). Unlike the gfx9
        ``buffer_load_lds`` family — which is NOT selectable on gfx1250 — each
        lane writes its own explicit LDS address, so no ``M0`` lane-contiguous
        base trick is needed.

        Completion is tracked on the ASYNC counter; drain with
        :meth:`s_wait_asynccnt` before reading the staged LDS. ``coherency`` is
        the gfx12 cachepolicy immediate (bits[0:2]=th, bits[3:4]=scope); 0 is
        the default, 2 (``CACHE_STREAM``/SLC) suits one-shot streaming loads.
        """
        if width_bytes not in (4, 8, 16):
            raise ValueError(
                f"global_load_async_to_lds width_bytes must be 4, 8, or 16 "
                f"(got {width_bytes})"
            )
        if coherency not in (0, 1, 2, 3):
            raise ValueError(f"coherency must be 0..3 (got {coherency})")
        self._op(
            "tile.global_load_async_to_lds",
            [src_ptr, src_index, lds_smem, *lds_indices],
            attrs={
                "width_bytes": int(width_bytes),
                "cpol": int(coherency),
                "offset_bytes": int(offset_bytes),
            },
        )

    def iglp_opt(self, level: int = 0) -> None:
        """`__builtin_amdgcn_iglp_opt(level)`.

        Asks the AMDGPU post-RA scheduler to apply a canned instruction
        interleaving for the enclosing loop (MFMA / ds_read / ds_write /
        VMEM). ``level`` selects the pattern (0 = GEMM MFMA-interleave,
        1 = attention-style). Placed once at the top of the main loop body;
        it owns the loop schedule, so manual ``sched_barrier`` /
        ``sched_group_barrier`` hints should be suppressed when it is used.
        """
        self._op("tile.iglp_opt", attrs={"level": int(level)})

    def sched_barrier(self, mask: int = 0) -> None:
        """`__builtin_amdgcn_sched_barrier(mask)`.

        Caps instruction reordering across this point. mask=0 means
        "schedule nothing across this barrier"; non-zero mask allows
        instructions of the specified classes to cross.
        """
        self._op("tile.sched_barrier", attrs={"mask": int(mask)})

    def sched_group_barrier(self, mask: int, count: int, group: int = 0) -> None:
        """`__builtin_amdgcn_sched_group_barrier(mask, count, group)`.

        Tells the scheduler to place `count` instructions of the class
        described by `mask` at this position. Used inside CK's compv4
        HotLoopScheduler to deterministically interleave MFMA, LDS
        reads/writes, and VMEM reads.

        AMD mask bits:
        0x008 = MFMA
        0x020 = VMEM read
        0x040 = VMEM write
        0x100 = DS read
        0x200 = DS write
        """
        self._op(
            "tile.sched_group_barrier",
            attrs={"mask": int(mask), "count": int(count), "group": int(group)},
        )

    def s_setprio(self, level: int) -> None:
        """`__builtin_amdgcn_s_setprio(level)`. level in 0..3."""
        if level < 0 or level > 3:
            raise ValueError("s_setprio level must be in 0..3")
        self._op("tile.s_setprio", attrs={"level": int(level)})

    # ----- buffer resources + async DRAM->LDS -----

    def global_ptr_add(self, ptr: Value, byte_off: Value) -> Value:
        """Return ``ptr + byte_off`` as a new global pointer (same type).

        Lowers to ``getelementptr inbounds i8, ptr addrspace(1) ptr,
        i64 byte_off`` (the offset is zero-extended to i64 if needed).
        This is the enabler for 64-bit paged-KV addressing: by folding a
        block's ``physical_block * stride`` (which overflows the i32 buffer
        voffset above a ~2 GiB cache) into a per-block 64-bit buffer
        *base*, the remaining within-block voffset stays small. Without
        this the tiled attention kernel silently corrupts loads once the
        paged cache exceeds 2 GiB (~65 K bf16 / ~131 K fp8 blocks).
        """
        return self._op(
            "tile.global_ptr_add",
            [ptr, byte_off],
            [ptr.type],
            result_name_hint="gptr",
        ).result

    def buffer_rsrc(self, ptr: Value, num_bytes: Value) -> Value:
        """Build an AMDGPU 128-bit buffer resource descriptor.

        Wraps `@llvm.amdgcn.make.buffer.rsrc.p1(ptr, stride=0,
        num_records=num_bytes, flags=0)`. The returned token is opaque
        (typed `<4 x i32>` here for printing; LLVM internally treats it
        as `ptr addrspace(8)`) and is consumed by `buffer_load_vN` and
        `async_buffer_load_lds`.
        """
        return self._op(
            "tile.buffer_rsrc",
            [ptr, num_bytes],
            [VectorType(I32, 4)],
            result_name_hint="rsrc",
        ).result

    def buffer_load_vN(
        self, rsrc: Value, voffset: Value, soffset: Value, dtype: "Type", n: int
    ) -> Value:
        """Dtype-generic vectorised `raw_ptr_buffer_load` of N elements.

        Supported dtypes and widths:
          - f16 / bf16 (2-byte): n in {2, 4, 8} → dwords = n // 2
          - f32 / i32  (4-byte): n in {1, 2, 4} → dwords = n

        The raw `<dwords x i32>` payload is bitcast to `<n x dtype>` so
        the returned Value has the correct IR type for the smem store.
        Bounds-checked: an out-of-range voffset returns 0 (the
        runbook §6.1 lever for tail-safe loads).
        """
        _elem_bytes = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4}
        eb = _elem_bytes.get(dtype.name)
        if eb is None:
            raise ValueError(f"buffer_load_vN: unsupported dtype {dtype.name!r}")
        dwords = (n * eb) // 4
        if dwords not in (1, 2, 4):
            raise ValueError(
                f"buffer_load_vN: n={n} dtype={dtype.name} → dwords={dwords} "
                "must be 1, 2, or 4"
            )
        return self._op(
            "tile.buffer_load_vN",
            [rsrc, voffset, soffset],
            [VectorType(dtype, n)],
            attrs={"dwords": int(dwords), "elem_type": dtype.name},
            result_name_hint=f"bl{n}",
        ).result

    def buffer_load_vN_f16(
        self, rsrc: Value, voffset: Value, soffset: Value, dwords: int
    ) -> Value:
        """Vectorised `raw_ptr_buffer_load`. dwords in {1,2,4}; each
        dword is two halves. Bounds-checked: an out-of-range voffset
        returns 0 (the runbook §6.1 lever for tail-safe loads).
        """
        if dwords not in (1, 2, 4):
            raise ValueError(f"buffer_load dwords must be 1, 2, or 4 (got {dwords})")
        halves = dwords * 2
        return self._op(
            "tile.buffer_load_vN_f16",
            [rsrc, voffset, soffset],
            [VectorType(F16, halves)],
            attrs={"dwords": int(dwords)},
            result_name_hint=f"bl{halves}",
        ).result

    def buffer_load(
        self, rsrc: Value, voffset: Value, soffset: Value, dtype: "Type"
    ) -> Value:
        """Dtype-generic scalar buffer load (single element, OOB-clamped).

        Supported dtypes: f16, bf16 (2-byte → i16 intrinsic),
        f32 / i32 (4-byte → i32 intrinsic).
        """
        _elem_bytes = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4}
        if dtype.name not in _elem_bytes:
            raise ValueError(f"buffer_load: unsupported dtype {dtype.name!r}")
        return self._op(
            "tile.buffer_load",
            [rsrc, voffset, soffset],
            [dtype],
            attrs={"elem_type": dtype.name},
            result_name_hint="bl1",
        ).result

    def buffer_load_f16(self, rsrc: Value, voffset: Value, soffset: Value) -> Value:
        """Scalar f16 buffer load via `raw_ptr_buffer_load_u16` + bitcast.

        Used by the convolution kernel's epilogue and any path that
        wants a per-lane single-half load via buffer descriptor
        (the OOB-clamping protection).
        """
        return self._op(
            "tile.buffer_load_f16",
            [rsrc, voffset, soffset],
            [F16],
            result_name_hint="bl1",
        ).result

    def buffer_store_vN_f16(
        self, rsrc: Value, voffset: Value, soffset: Value, value: Value, dwords: int
    ) -> None:
        """Vectorised `raw_ptr_buffer_store`. dwords in {1,2,4}; each
        dword is two halves. Out-of-range voffsets are *silently
        dropped* by the AMD buffer rsrc — the runbook §6.2 lever
        ("vectorise the epilogue") for tail-safe stores.
        """
        if dwords not in (1, 2, 4):
            raise ValueError(f"buffer_store dwords must be 1, 2, or 4 (got {dwords})")
        self._op(
            "tile.buffer_store_vN_f16",
            [rsrc, voffset, soffset, value],
            attrs={"dwords": int(dwords)},
        )

    def buffer_store_f16(
        self, rsrc: Value, voffset: Value, soffset: Value, value: Value
    ) -> None:
        """Single-half buffer store, OOB-clamped. The epilogue path
        for per-lane direct stores (4 halves per accumulator slot)
        uses this when the kernel layout doesn't align to a 32-bit
        vector boundary."""
        self._op(
            "tile.buffer_store_f16",
            [rsrc, voffset, soffset, value],
        )

    def zero_vec_f16(self, n: int) -> Value:
        """A `<n x half>` zero constant — the canonical "mask out OOB
        load" value, and the canonical "padding" value for direct
        conv kernels (the boundary cells of a 3x3 input get masked
        through this when the validity predicate flips false)."""
        if n <= 0:
            raise ValueError(f"zero_vec_f16 needs positive n, got {n}")
        return self.zero_vec(F16, n)

    def async_buffer_load_lds(
        self,
        rsrc: Value,
        lds_ptr: Value,
        voffset: Value,
        soffset: Value,
        dwords: int,
        coherency: int = 0,
    ) -> None:
        """Async DRAM->LDS copy via `raw_ptr_buffer_load_lds`.

        dwords in {1, 3, 4} on gfx950 (4, 12, or 16 bytes per lane).
        Each lane writes lane-contiguously into `lds_ptr + lane*dwords*4`
        — see runbook §6.3: swizzle must be expressed in *address
        arithmetic*, not by passing an arbitrary per-lane LDS pointer.

        Completion is signalled via the VMEM counter; consumers must
        place an `s_waitcnt(vmcnt=0)` before reading the LDS.

        ``coherency`` selects AUX-bit cache-coherence hints — see
        :data:`CACHE_ALL` / :data:`CACHE_GLOBAL` /
        :data:`CACHE_STREAM` / :data:`NON_TEMPORAL`.
        """
        if dwords not in (1, 3, 4):
            raise ValueError(
                f"async_buffer_load_lds dwords must be 1, 3, or 4 (got {dwords})"
            )
        if coherency not in (0, 1, 2, 3):
            raise ValueError(f"coherency must be 0..3 (got {coherency})")
        self._op(
            "tile.async_buffer_load_lds",
            [rsrc, lds_ptr, voffset, soffset],
            attrs={"dwords": int(dwords), "aux": int(coherency)},
        )

    def global_load_lds(
        self,
        src_ptr: Value,
        byte_off: Value,
        lds_addr: Value,
        size_bytes: int,
        coherency: int = 0,
    ) -> None:
        """Direct DRAM->LDS DMA via ``llvm.amdgcn.global.load.lds``.

        This is the *flat/global* (non-buffer-descriptor) sibling of
        :meth:`async_buffer_load_lds`. It bypasses the VGPR round-trip
        entirely: each lane issues ``global_load_lds_dword{,x4}`` which
        streams ``size_bytes`` directly from ``src_ptr + byte_off`` (a
        ``ptr addrspace(1)``) into the wave-uniform LDS address
        ``lds_addr`` (an i64, biased per-wave by the caller). No buffer
        resource descriptor is required.

        Parameters
        ----------
        src_ptr
            Global (``addrspace(1)``) base pointer.
        byte_off
            i32 per-lane byte offset into ``src_ptr``; folded as an i8
            GEP so the lane's source address is ``src_ptr + byte_off``.
        lds_addr
            i64 LDS (``addrspace(3)``) destination address. The intrinsic
            writes lane-contiguously starting here, so multi-wave kernels
            MUST bias this by ``wave_id * wave_bytes`` (see
            :meth:`smem_ptr_add`) or waves stomp each other.
        size_bytes
            Bytes per lane: 1, 2, 4 (``global_load_lds_dword``) or — on
            gfx950 — 12 / 16 (``global_load_lds_dwordx3/x4``). 16-byte is
            the wide direct-to-LDS path the pyisa reference uses.
        coherency
            AUX-bit cache-coherence hint (0..3): :data:`CACHE_ALL`,
            :data:`CACHE_GLOBAL`, :data:`CACHE_STREAM`,
            :data:`NON_TEMPORAL`. ``CACHE_ALL`` is the right hint for
            reused weights so they stay resident in L2.

        Completion is signalled via the VMEM counter, exactly like
        :meth:`async_buffer_load_lds`; consumers must place an
        ``s_waitcnt(vmcnt=0)`` before reading the LDS. This only WINS
        when coupled to software-prefetch (the next-tile DMA must be in
        flight during the current MFMAs); issued alone on a
        barrier-bound loop it regresses.
        """
        if size_bytes not in (1, 2, 4, 12, 16):
            raise ValueError(
                f"global_load_lds size_bytes must be 1, 2, 4, 12, or 16 "
                f"(got {size_bytes})"
            )
        if coherency not in (0, 1, 2, 3):
            raise ValueError(f"coherency must be 0..3 (got {coherency})")
        self._op(
            "tile.global_load_lds",
            [src_ptr, byte_off, lds_addr],
            attrs={"size_bytes": int(size_bytes), "aux": int(coherency)},
        )

    # ----- f32 LDS ops (cshuffle epilogue) -----

    def smem_alloc_f32(
        self, shape: Sequence[int], name_hint: str = "smem_f32"
    ) -> Value:
        """`smem_alloc` specialised to f32 — used by the cshuffle
        epilogue to LDS-stage the accumulators before wide global
        stores."""
        return self.smem_alloc(F32, shape, name_hint=name_hint)

    def smem_store_vN_f32(
        self, smem: Value, indices: Sequence[Value], value: Value, n: int
    ) -> None:
        if n not in (1, 2, 4):
            raise ValueError(f"smem_store_vN_f32 n must be 1, 2, or 4 (got {n})")
        self._op(
            "tile.smem_store_vN_f32",
            [smem, *indices, value],
            attrs={"rank": len(indices), "elem_type": "f32", "vec": n},
        )

    def smem_load_vN_f32(self, smem: Value, *indices, n: int = 0) -> Value:
        if n not in (1, 2, 4):
            raise ValueError(f"smem_load_vN_f32 n must be 1, 2, or 4 (got {n})")
        if not indices:
            raise ValueError("smem_load_vN_f32 needs at least one index")
        return self._op(
            "tile.smem_load_vN_f32",
            [smem, *indices],
            [VectorType(F32, n)],
            attrs={"elem_type": "f32", "vec": n, "rank": len(indices)},
            result_name_hint=f"av{n}f32",
        ).result

    # ----- packed f32->f16 conversion + wide global store -----

    def vec_trunc_f32_to_f16(self, v: Value) -> Value:
        """Element-wise `fptrunc <N x float> -> <N x half>` — used by
        the f16-output cshuffle epilogue to pack one MFMA accumulator
        vector into a half vector before LDS write."""
        return self.vec_cast_f32_to(v, F16)

    def vec_cast_f32_to(self, v: Value, target: Type) -> Value:
        """Element-wise `fptrunc <N x float> -> <N x target>`.

        Supports f16 and bf16 output vectors.
        """
        if not isinstance(v.type, VectorType) or v.type.elem.name != "f32":
            raise ValueError("vec_cast_f32_to expects <N x f32>")
        if target.name not in ("f16", "bf16"):
            raise ValueError(f"vec_cast_f32_to unsupported target {target.name}")
        return self._op(
            "vector.trunc_f32_to",
            [v],
            [VectorType(target, v.type.count)],
            attrs={"target": target.name},
            result_name_hint=f"vh{v.type.count}",
        ).result

    def global_store_vN_f16(
        self,
        ptr: Value,
        idx: Value,
        value: Value,
        n: int,
        *,
        align: Optional[int] = None,
    ) -> None:
        """Vectorised `<N x half>` global store — the runbook §6.2 lever
        ("Vectorizing the epilogue is often the single largest
        optimization for kernels that already have a good main loop.").
        """
        self.global_store_vN(ptr, idx, value, n, align=align)

    def global_store_vN(
        self,
        ptr: Value,
        idx: Value,
        value: Value,
        n: int,
        *,
        align: Optional[int] = None,
    ) -> None:
        """Vectorised global store of N consecutive elements.

        Supports the full element-type catalog the LLVM lowering already
        emits: ``f16`` / ``bf16`` / ``i16`` (2-byte), ``f32`` / ``i32``
        (4-byte), ``i8`` / ``fp8e4m3`` / ``bf8e5m2`` (1-byte). Lowers to
        a single ``store <N x elem>`` and AMDGPU coalesces into one
        ``global_store_dwordxN`` transaction.
        """
        if n not in (1, 2, 4, 8, 16):
            raise ValueError(f"global_store_vN n must be 1, 2, 4, 8, or 16 (got {n})")
        elem_name = (
            value.type.elem.name
            if isinstance(value.type, VectorType)
            else value.type.name
        )
        if elem_name in ("f16", "bf16", "i16"):
            elem_bytes = 2
            if n == 16:
                raise ValueError(f"global_store_vN n=16 not supported for {elem_name}")
        elif elem_name in ("f32", "i32"):
            elem_bytes = 4
            if n == 16:
                raise ValueError(f"global_store_vN n=16 not supported for {elem_name}")
        elif elem_name in ("i8", "fp8e4m3", "bf8e5m2"):
            elem_bytes = 1
        else:
            raise ValueError(
                "global_store_vN supports f16/bf16/i16/f32/i32/i8/fp8e4m3/bf8e5m2, "
                f"got {elem_name}"
            )
        self._op(
            "memref.global_store_vN",
            [ptr, idx, value],
            attrs={
                "elem_type": elem_name,
                "vec": n,
                "align": int(align or (n * elem_bytes)),
            },
        )

    # ----- atomics (for split-K) -----

    def global_atomic_add_f32(self, ptr: Value, idx: Value, value: Value) -> None:
        """`atomicrmw fadd <ptr addrspace(1)>, float seq_cst` — the
        kernel-side primitive for split-K accumulation. Runbook §4.1
        flags this as the cost the algorithm has to pay; whether it's
        worth it depends on the K/MN ratio.
        """
        self._op(
            "memref.global_atomic_add_f32",
            [ptr, idx, value],
        )

    def store_f16(self, ptr: Value, idx: Value, value: Value) -> None:
        self._op(
            "memref.global_store",
            [ptr, idx, value],
            attrs={"elem_type": "f16", "align": 2},
        )

    def store(
        self,
        value: Value,
        ptr: Value,
        idx: Optional[Value] = None,
        *,
        align: Optional[int] = None,
    ) -> None:
        """Store a scalar ``value`` of any type to ``ptr[idx]`` (default idx 0).

        Unlike :meth:`store_f16` (hardcoded ``half``), this derives the element
        type from ``value`` and lowers through ``memref.global_store_typed``.
        """
        if idx is None:
            idx = self.const_i32(0)
        attrs: Dict[str, Any] = {}
        if align is not None:
            attrs["align"] = int(align)
        self._op("memref.global_store_typed", [ptr, idx, value], attrs=attrs)

    def ret(self) -> None:
        self._op("cf.return")

    def vec_extract(self, v: Value, i: int) -> Value:
        elem_t = v.type.elem if isinstance(v.type, VectorType) else v.type
        return self._op(
            "vector.extract",
            [v],
            [elem_t],
            attrs={"index": int(i)},
            result_name_hint="e",
        ).result

    def vec_pack(self, components: Sequence[Value], elem: Type) -> Value:
        """Pack N scalars into `<N x elem>` via insertelement chain."""
        n = len(components)
        if n == 0:
            raise ValueError("vec_pack needs at least one component")
        for c in components:
            if c.type != elem:
                raise ValueError(f"vec_pack expected {elem.name}, got {c.type.name}")
        return self._op(
            "vector.pack",
            list(components),
            [VectorType(elem, n)],
            attrs={"elem": elem.name, "vec": n},
            result_name_hint="vp",
        ).result

    def vec_concat(self, a: Value, b: Value) -> Value:
        """Concatenate two equal-typed vectors into a double-width vector."""
        if not isinstance(a.type, VectorType) or not isinstance(b.type, VectorType):
            raise ValueError("vec_concat needs vector inputs")
        if a.type.elem != b.type.elem:
            raise ValueError("vec_concat element types must match")
        n = a.type.count + b.type.count
        return self._op(
            "vector.concat",
            [a, b],
            [VectorType(a.type.elem, n)],
            attrs={"elem": a.type.elem.name, "vec": n},
            result_name_hint="vc",
        ).result

    def vec_insert(self, v: Value, scalar: Value, i: int) -> Value:
        if not isinstance(v.type, VectorType):
            raise ValueError("vec_insert expects vector")
        if scalar.type != v.type.elem:
            raise ValueError("vec_insert scalar type mismatch")
        return self._op(
            "vector.insert",
            [v, scalar],
            [v.type],
            attrs={"index": int(i)},
            result_name_hint="vi",
        ).result

    # ----- control flow -----

    def scf_for(self, lower: Value, upper: Value, step: Value, iv_name: str = "k0"):
        body = Region("body")
        iv = Value(name=f"%{iv_name}", type=lower.type)
        op = Op(
            name="scf.for",
            operands=[lower, upper, step],
            attrs={"iv": iv.name, "iv_type": lower.type.name},
            regions=[body],
        )
        iv.op = op
        self._emit(op)
        return _ForBuilder(self, op, iv, body, [])

    def scf_for_iter(
        self,
        lower: Value,
        upper: Value,
        step: Value,
        iter_args: Sequence[Tuple[str, Value]],
        iv_name: str = "k0",
        unroll: bool = False,
        elide_trailing_barrier: bool = True,
    ) -> "_ForBuilder":
        """Create a scf.for loop with iteration arguments.

        Args:
        lower: Loop lower bound
        upper: Loop upper bound
        step: Loop step
        iter_args: Sequence of (name, init_value) for iteration variables
        iv_name: Induction variable name (default: "k0")
        unroll: Loop unrolling hint (default: False)
        - False: emit normal loop
        - True: fully unroll if trip count is compile-time constant
        elide_trailing_barrier: Phase 4 optimization (default: True)
        - True: automatically elide trailing sync() in non-final iterations
        - False: preserve all barriers (for manually optimized kernels)
        """
        body = Region("body")
        iv = Value(name=f"%{iv_name}", type=lower.type)
        iter_vars: List[Value] = []
        iter_inits: List[Value] = []
        iter_meta: List[Dict[str, Any]] = []
        for arg_name, init in iter_args:
            v = Value(name=f"%{arg_name}", type=init.type)
            iter_vars.append(v)
            iter_inits.append(init)
            iter_meta.append({"name": v.name, "type": init.type.name})
        results = [Value(self._fresh("for"), v.type) for v in iter_vars]
        op = Op(
            name="scf.for",
            operands=[lower, upper, step, *iter_inits],
            attrs={
                "iv": iv.name,
                "iv_type": lower.type.name,
                "iter_args": iter_meta,
                "num_iter_args": len(iter_args),
                "unroll": unroll,
                "elide_trailing_barrier": elide_trailing_barrier,
            },
            results=results,
            regions=[body],
        )
        for r in results:
            r.op = op
        iv.op = op
        for v in iter_vars:
            v.op = op
        self._emit(op)
        return _ForBuilder(self, op, iv, body, iter_vars)

    def scf_yield(self, *values: Value) -> None:
        self._op("scf.yield", list(values), [], attrs={"num": len(values)})

    def static_if(
        self,
        cond: bool,
        then_body: Callable[[], None],
        else_body: Optional[Callable[[], None]] = None,
    ) -> None:
        """Python-time branch for compile-time decisions.

        Unlike Python `if value:` on an SSA `Value`, this API requires a real
        host boolean. Passing a runtime `Value` raises immediately so kernels do
        not accidentally mix host control flow with device control flow.
        """
        if isinstance(cond, Value):
            raise TypeError(
                "static_if expects a Python bool, not an SSA Value; use scf_if for runtime control flow"
            )
        if bool(cond):
            then_body()
        elif else_body is not None:
            else_body()

    def scf_if(self, cond: Value):
        """Runtime branch. Prefer static_if for Python-time decisions."""
        then_r = Region("then")
        op = Op(name="scf.if", operands=[cond], regions=[then_r])
        self._emit(op)
        return _IfBuilder(self, op, then_r)


PURE_OP_NAMES = {
    "arith.constant",
    "arith.constant_vec",
    "arith.add",
    "arith.sub",
    "arith.mul",
    "arith.div",
    "arith.mod",
    "arith.fadd",
    "arith.fsub",
    "arith.fmul",
    "arith.fdiv",
    "arith.fneg",
    "arith.cmp",
    "arith.fcmp",
    "arith.select",
    "arith.fmax",
    "arith.fmin",
    "arith.select",
    "arith.and",
    "arith.or",
    "arith.not",
    "arith.smax",
    "arith.smin",
    "arith.zext",
    "arith.sext",
    "arith.trunc",
    "arith.trunc_f32_to_f16",
    "arith.cast_to_f32",
    "arith.cast_f32_to",
    "arith.sitofp_f32",
    "arith.cvt_fp8_to_f32",
    "math.exp2",
    "math.log2",
    "math.rcp",
    "math.rcp_fast",
    "math.sqrt",
    "math.rsqrt",
    "math.tanh",
    "vector.extract",
    "vector.trunc_f32_to_f16",
    "vector.trunc_f32_to",
    "vector.bitcast",
    "vector.add",
    "vector.mul",
    "vector.and",
    "vector.or",
    "vector.shl",
    "vector.lshr",
    "vector.smax",
    "vector.smin",
    "vector.max",
    "vector.cmp",
    "vector.trunc",
    "vector.sext",
    "vector.sum",
    "vector.reduce_max",
    "vector.splat",
    "vector.select",
    "vector.pack",
    "vector.concat",
    "vector.insert",
    "gpu.thread_id",
    "gpu.block_id",
    "tile.readfirstlane",
    "tile.pin_sgpr",
    "tile.wave_all",
    "tile.wave_any",
    "tile.wave_ballot",
    "tile.sync_half_block",
    "tile.smem_addr_of",
    "tile.smem_ptr_add",
    "tile.lane_id",
    "tile.ds_bpermute",
    "tile.ds_read_tr16_b64",
    "arith.bitcast",
    "arith.xor",
    "arith.shl",
    "arith.lshr",
    "arith.umul_hi_i32",
    "arith.cvt_bf8_to_f32",
    "arith.cvt_f32_to_fp8",
    "arith.cvt_f32_to_bf8",
    "arith.cvt_f32_to_i8_sat",
    "arith.rint_f32",
    "arith.cvt_pk_f32_fp8x4",
    "arith.cvt_pk_f32_bf8x4",
    "arith.cvt_pk_fp8_f32x4",
    "arith.cvt_pk_bf8_f32x4",
    "arith.cvt_scalef32_pk_f32_fp8",
    "arith.cvt_scalef32_pk_f32_bf8",
    "arith.cvt_scalef32_pk_fp8_f32",
    "arith.cvt_scalef32_pk_bf8_f32",
    "tile.ds_read_tr_b8",
    "tile.ds_swizzle_xor",
    "tile.dpp_xor",
    "tile.permlane32_swap",
    "tile.perm_b32",
    "tile.permlanex16",
    "tile.byte_perm",
}


def is_pure_op_name(name: str) -> bool:
    return name in PURE_OP_NAMES


class _ForBuilder:
    def __init__(
        self,
        parent: IRBuilder,
        op: Op,
        iv: Value,
        body: Region,
        iter_vars: List[Value],
    ) -> None:
        self._parent = parent
        self.op = op
        self.iv = iv
        self.body = body
        self.iter_vars = iter_vars

    def __enter__(self):
        self._parent.push_region(self.body)
        if self.iter_vars:
            return self.iv, list(self.iter_vars)
        return self.iv

    def __exit__(self, exc_type, exc, tb) -> None:
        self._parent.pop_region()

    @property
    def results(self) -> List[Value]:
        return list(self.op.results)


class _IfBuilder:
    def __init__(self, parent: IRBuilder, op: Op, then_region: Region) -> None:
        self._parent = parent
        self.op = op
        self._then = then_region

    def __enter__(self) -> "_IfBuilder":
        self._parent.push_region(self._then)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._parent.pop_region()
