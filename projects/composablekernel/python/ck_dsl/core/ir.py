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
I32 = Type("i32")
I64 = Type("i64")
BF16 = Type("bf16")
F16 = Type("f16")
F32 = Type("f32")
FP8E4M3 = Type("fp8e4m3")


# AMDGPU buffer-load AUX-byte cache-coherency hints. The AUX field of
# the ``raw_ptr_buffer_load[_lds]`` intrinsics encodes the GLC and SLC
# bits that bias the load's L1/L2 caching policy. Pass one of these
# constants as the ``coherency`` argument to
# ``async_buffer_load_lds_addr`` / ``buffer_load_vN_*``.
CACHE_ALL = 0  # Cache at all levels (default).
CACHE_GLOBAL = 1  # GLC set — skip L2; useful for one-shot loads.
CACHE_STREAM = 2  # SLC set — streaming hint (don't evict useful lines).
NON_TEMPORAL = 3  # GLC + SLC — bypass cache hierarchy entirely.


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
            "ck_dsl SSA Value cannot be used as a Python bool. "
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

    def rcp(self, a: Value) -> Value:
        return self._op("math.rcp", [a], [a.type], result_name_hint="rcp").result

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

    def zext(self, v: Value, target: Type) -> Value:
        return self._op("arith.zext", [v], [target], result_name_hint="zx").result

    def sext(self, v: Value, target: Type) -> Value:
        return self._op("arith.sext", [v], [target], result_name_hint="sx").result

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
        if elem in (F16, BF16):
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
        """Vectorised global load of N consecutive 16-bit values.

        Supports `f16` and `bf16` (N in {2,4,8}).
        """
        if dtype.name not in ("f16", "bf16"):
            raise ValueError(f"global_load_vN supports f16/bf16, got {dtype.name}")
        if n not in (2, 4, 8):
            raise ValueError(f"unsupported vector width for global_load_vN: {n}")
        return self._op(
            "memref.global_load_vN",
            [ptr, idx],
            [VectorType(dtype, n)],
            attrs={"elem_type": dtype.name, "vec": n, "align": int(align or (n * 2))},
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

    def vector_max(self, a: Value, b: Value) -> Value:
        return self.vector_binary("max", a, b)

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
        """Vectorised LDS store of N consecutive 16-bit values.

        Supports scalar 16-bit stores (`n=1`) and vector stores for f16/bf16.
        """
        if n == 1:
            # Single-element store; route through the scalar `tile.smem_store`.
            self._op(
                "tile.smem_store",
                [smem, *indices, value],
                attrs={"rank": len(indices), "elem_type": value.type.name},
            )
            return
        if n not in (2, 4, 8):
            raise ValueError(f"unsupported vector width for smem_store_vN: {n}")
        if not isinstance(value.type, VectorType):
            raise ValueError("smem_store_vN expects vector value for n > 1")
        self._op(
            "tile.smem_store_vN",
            [smem, *indices, value],
            attrs={"rank": len(indices), "elem_type": value.type.elem.name, "vec": n},
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
        """LDS load of <N x dtype> for 16-bit f16/bf16 values."""
        if dtype.name not in ("f16", "bf16"):
            raise ValueError(f"smem_load_vN supports f16/bf16, got {dtype.name}")
        if n not in (1, 2, 4, 8):
            raise ValueError(f"unsupported vector width for smem_load_vN: {n}")
        if not indices:
            raise ValueError("smem_load_vN needs at least one index")
        return self._op(
            "tile.smem_load_vN",
            [smem, *indices],
            [VectorType(dtype, n)],
            attrs={"elem_type": dtype.name, "vec": n, "rank": len(indices)},
            result_name_hint=f"av{n}",
        ).result

    def mfma_f32_16x16x16_f16(self, a: Value, b: Value, c: Value) -> Value:
        return self._op(
            "tile.mfma_f32_16x16x16_f16",
            [a, b, c],
            [VectorType(F32, 4)],
            result_name_hint="acc",
        ).result

    def mfma_f32_16x16x32_f16(self, a: Value, b: Value, c: Value) -> Value:
        """MFMA with K=32 per atom (gfx950 only).

        A and B per-lane operands are `<8 x half>` (vs `<4 x half>` for
        16x16x16); the accumulator stays `<4 x float>`. This halves the
        K-loop trip count for the same per-warp tile, and the wider
        operand load amortises the address arithmetic per MFMA.
        """
        return self._op(
            "tile.mfma_f32_16x16x32_f16",
            [a, b, c],
            [VectorType(F32, 4)],
            result_name_hint="acc",
        ).result

    def mfma_f32_16x16x16_bf16(self, a: Value, b: Value, c: Value) -> Value:
        return self._op(
            "tile.mfma_f32_16x16x16_bf16",
            [a, b, c],
            [VectorType(F32, 4)],
            result_name_hint="acc",
        ).result

    def mfma_f32_16x16x32_bf16(self, a: Value, b: Value, c: Value) -> Value:
        return self._op(
            "tile.mfma_f32_16x16x32_bf16",
            [a, b, c],
            [VectorType(F32, 4)],
            result_name_hint="acc",
        ).result

    def mfma_f32_32x32x8_f16(self, a: Value, b: Value, c: Value) -> Value:
        """The 32x32x8 f16 MFMA atom — the default warp-tile every CK
        Tile dispatcher config from `default_config.json` uses on wave64.

        Wave64 layout: A is `<4 x half>`, B is `<4 x half>` per lane,
        accumulator is `<16 x float>` per lane (32*32 = 1024 outputs /
        64 lanes). One MFMA per K-step produces 16 floats per lane that
        we then truncate to f16 for the GEMM epilogue.
        """
        return self._op(
            "tile.mfma_f32_32x32x8_f16",
            [a, b, c],
            [VectorType(F32, 16)],
            result_name_hint="acc",
        ).result

    def mfma_f32_32x32x16_f16(self, a: Value, b: Value, c: Value) -> Value:
        """The 32x32x16 f16 MFMA atom (gfx950 only).

        Per lane: A `<8 x half>`, B `<8 x half>`, acc `<16 x float>`.
        Doubles K per atom over 32x32x8, halving the K-loop trip count
        for the same per-warp tile and the same accumulator footprint.
        """
        return self._op(
            "tile.mfma_f32_32x32x16_f16",
            [a, b, c],
            [VectorType(F32, 16)],
            result_name_hint="acc",
        ).result

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
        return self._op(
            "tile.mfma_f32_4x4x4_f16",
            [a, b, c],
            [VectorType(F32, 4)],
            result_name_hint="acc",
        ).result

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
        """
        if addr.type.name != "i32" or data.type.name != "i32":
            raise ValueError("ds_bpermute requires i32 addr + i32 data")
        return self._op(
            "tile.ds_bpermute",
            [addr, data],
            [I32],
            result_name_hint="bp",
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

        Wraps `ds_bpermute` with the standard `(lane ^ xor) << 2` address.
        Works for any 32-bit scalar `v` (f32, i32). For half/bfloat, bitcast
        to i32 via a 2-element vector first.
        """
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
        """Vectorised `<N x 16-bit>` global store for f16/bf16."""
        if n not in (1, 2, 4, 8):
            raise ValueError(f"global_store_vN n must be 1, 2, 4, or 8 (got {n})")
        elem_name = (
            value.type.elem.name
            if isinstance(value.type, VectorType)
            else value.type.name
        )
        if elem_name not in ("f16", "bf16"):
            raise ValueError(f"global_store_vN supports f16/bf16, got {elem_name}")
        self._op(
            "memref.global_store_vN",
            [ptr, idx, value],
            attrs={"elem_type": elem_name, "vec": n, "align": int(align or (n * 2))},
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
    "arith.zext",
    "arith.sext",
    "arith.trunc_f32_to_f16",
    "arith.cast_to_f32",
    "arith.cast_f32_to",
    "arith.sitofp_f32",
    "arith.cvt_fp8_to_f32",
    "math.exp2",
    "math.rcp",
    "math.sqrt",
    "math.rsqrt",
    "math.tanh",
    "vector.extract",
    "vector.trunc_f32_to_f16",
    "vector.trunc_f32_to",
    "vector.bitcast",
    "vector.add",
    "vector.mul",
    "vector.max",
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
