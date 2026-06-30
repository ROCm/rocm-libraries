# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lower CK DSL IR to a HIP `__global__` kernel body.

This walks the IR tree and emits C++ statements per operation. It is
deliberately *not* an f-string template: each statement maps from a
specific IR op kind and its operands. New ops require explicit handler
entries here.

The lowering target is a HIP function whose signature is built from the
IR's `KernelDef.params`. Body locals are named after IR SSA values
(stripping the leading `%`).
"""

from __future__ import annotations

from typing import List, Optional

from .ir import (
    KernelDef,
    Op,
    PtrType,
    Region,
    SmemType,
    Value,
    VectorType,
)


_HIP_TYPE = {
    "i1": "bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int",
    "i64": "int64_t",
    "f16": "fp16",
    "bf16": "bf16",
    "f32": "float",
    "fp8e4m3": "fp8e4m3",
    "bf8e5m2": "bf8e5m2",
}


def _type_to_hip(t) -> str:
    if isinstance(t, PtrType):
        if t.space in ("global", "lds"):
            return f"{_type_to_hip(t.pointee)}*"
    if isinstance(t, VectorType):
        # Naming convention matches the prologue's typedefs:
        # ``f16xN``, ``bf16xN``, ``f32xN``, ``i32xN``, ``i8xN`` for N>=1.
        elem = t.elem.name
        if elem == "f16":
            return f"f16x{t.count}"
        if elem == "bf16":
            return f"bf16x{t.count}"
        if elem == "f32":
            return f"f32x{t.count}"
        if elem == "i32":
            return f"i32x{t.count}"
        if elem == "i16":
            return f"i16x{t.count}"
        if elem == "i8":
            return f"i8x{t.count}"
        if elem == "fp8e4m3":
            return f"i8x{t.count}"  # fp8e4m3 is stored as bytes
        if elem == "bf8e5m2":
            return f"i8x{t.count}"  # bf8e5m2 is stored as bytes
        if elem == "i1":
            # i1 vectors materialise per-element predicates; lower as
            # ``boolxN`` from the prologue. The AMDGPU backend folds
            # these into VCC / s_mov_b64 mask ops the same way the
            # direct LLVM IR path does.
            return f"boolx{t.count}"
    if isinstance(t, SmemType):
        # smem allocations expand into __shared__ arrays at the top of
        # the kernel; the value at the use site is a typed pointer-ish.
        return f"{_type_to_hip(t.elem)}*"
    return _HIP_TYPE[t.name]


# Compilable-HIP prologue. Pasted at the top of every lowered source so
# the typedefs the handlers reference (``fp16``, ``bf16``, ``fNxM``)
# resolve, and the AMDGCN builtins we emit (``__builtin_amdgcn_*``,
# ``__hexp2f``, etc.) are available. The prologue is plain C++ that
# any modern hipcc / amdclang understands; no <hip/hip_runtime.h>
# dependency beyond what hipcc auto-injects for ``__global__`` kernels.
HIP_PROLOGUE = """\
// === rocke lower_hip prologue (auto-generated) ===
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>
#include <stdint.h>

using fp16 = _Float16;
#if defined(__BF16__) || defined(__bfloat16)
using bf16 = __bf16;
#else
using bf16 = __bf16;
#endif
using fp8e4m3 = signed char; // raw byte storage; converted via amdgcn intrinsics
using bf8e5m2 = signed char; // raw byte storage; e5m2 variant (different cvt intrinsic)

// AMDGPU vector typedefs via Clang's ext_vector_type. Names match the
// fNxM convention used throughout the handlers below.
#define _ROCKE_VEC(elem_t, name, n) \\
 using name##n = elem_t __attribute__((ext_vector_type(n)))
_ROCKE_VEC(fp16, f16x, 1); _ROCKE_VEC(fp16, f16x, 2); _ROCKE_VEC(fp16, f16x, 4);
_ROCKE_VEC(fp16, f16x, 8); _ROCKE_VEC(fp16, f16x, 16);
_ROCKE_VEC(bf16, bf16x, 1); _ROCKE_VEC(bf16, bf16x, 2); _ROCKE_VEC(bf16, bf16x, 4);
_ROCKE_VEC(bf16, bf16x, 8); _ROCKE_VEC(bf16, bf16x, 16);
_ROCKE_VEC(float, f32x, 1); _ROCKE_VEC(float, f32x, 2); _ROCKE_VEC(float, f32x, 4);
_ROCKE_VEC(float, f32x, 8); _ROCKE_VEC(float, f32x, 16);
_ROCKE_VEC(int, i32x, 1); _ROCKE_VEC(int, i32x, 2); _ROCKE_VEC(int, i32x, 3);
_ROCKE_VEC(int, i32x, 4); _ROCKE_VEC(int, i32x, 8);
_ROCKE_VEC(int16_t, i16x, 1); _ROCKE_VEC(int16_t, i16x, 2);
_ROCKE_VEC(int16_t, i16x, 4); _ROCKE_VEC(int16_t, i16x, 8);
_ROCKE_VEC(int8_t, i8x, 1); _ROCKE_VEC(int8_t, i8x, 2);
_ROCKE_VEC(int8_t, i8x, 4); _ROCKE_VEC(int8_t, i8x, 8); _ROCKE_VEC(int8_t, i8x, 16);
_ROCKE_VEC(bool, boolx, 2); _ROCKE_VEC(bool, boolx, 4); _ROCKE_VEC(bool, boolx, 8);
_ROCKE_VEC(bool, boolx, 16);
#undef _ROCKE_VEC

// Buffer-resource descriptor opaque type. ``__builtin_amdgcn_make_buffer_rsrc``
// returns this; the ``_ptr_`` family of buffer-load / store builtins takes
// it as the first argument. Although the IR uses ``<4 x i32>`` to model the
// 128-bit descriptor, at the C++ level we use the opaque builtin type so
// type checking lines up with the intrinsics.
using rsrc_t = __amdgpu_buffer_rsrc_t;

// LLVM intrinsics that clang 20 does NOT expose as ``__builtin_amdgcn_*``
// builtins (or whose builtins reject the size values we need). We declare
// them as ``__device__ extern "C"`` with an ``__asm`` mangling that names
// the LLVM intrinsic directly; clang lowers the call through the AMDGPU
// backend the same way it would the missing builtin. The ``__device__``
// attribute is required so HIP allows the call from a ``__global__``
// kernel context.
typedef short i16x4_raw __attribute__((ext_vector_type(4)));
__device__ extern "C" i16x4_raw _llvm_amdgcn_ds_read_tr16_b64(
 const __attribute__((address_space(3))) void*)
 __asm("llvm.amdgcn.ds.read.tr16.b64");
typedef short i16x8_raw __attribute__((ext_vector_type(8)));
__device__ extern "C" i16x8_raw _llvm_amdgcn_ds_read_tr16_b128(
 const __attribute__((address_space(3))) void*)
 __asm("llvm.amdgcn.ds.read.tr16.b128");
// ``__builtin_amdgcn_raw_ptr_buffer_load_lds`` restricts the size arg to
// {1, 2, 4} bytes; the LLVM intrinsic itself accepts {1, 2, 4, 12, 16},
// which is what async-DMA pipelines (compv4 / split-KV attention) need.
// Calling the intrinsic directly bypasses the builtin's validation.
__device__ extern "C" void _llvm_amdgcn_raw_ptr_buffer_load_lds(
 __amdgpu_buffer_rsrc_t,
 __attribute__((address_space(3))) void*,
 int /*size_bytes*/,
 int /*voffset*/,
 int /*soffset*/,
 int /*offset_imm*/,
 int /*aux_imm*/)
 __asm("llvm.amdgcn.raw.ptr.buffer.load.lds");
"""


def _name(v: Value) -> str:
    return v.name[1:] if v.name.startswith("%") else v.name


def _f32_literal(val: float) -> str:
    """Format a Python float for C++ float literal context.

    Special-cases ``inf`` / ``-inf`` / ``nan`` because Python's
    ``repr(float('inf'))`` is ``'inf'`` which would emit ``"inff"``
    (invalid C++). Instead emit the standard ``<math.h>`` macros.
    """
    import math

    if math.isnan(val):
        return "((float)NAN)"
    if math.isinf(val):
        return "((float)-INFINITY)" if val < 0 else "((float)INFINITY)"
    return f"{val}f"


def _encode_waitcnt_gfx9_10(vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
    """Encode `s_waitcnt` for the gfx9/gfx10 layout used by gfx950.

    VMCNT is 6 bits split between bits [3:0] and [15:14]. EXPCNT is
    bits [6:4]. LGKMCNT is bits [11:8] on gfx950. This mirrors the
    LLVM lowerer and keeps the HIP debug printer from silently turning
    partial waits such as `vmcnt=16` into full waits.
    """

    vm_b = 0x3F if vmcnt < 0 else min(max(vmcnt, 0), 0x3F)
    ec_b = 0x7 if expcnt < 0 else min(max(expcnt, 0), 0x7)
    lk_b = 0xF if lgkmcnt < 0 else min(max(lgkmcnt, 0), 0xF)
    return (vm_b & 0xF) | (ec_b << 4) | (lk_b << 8) | (((vm_b >> 4) & 0x3) << 14)


def _encode_waitcnt_gfx11(vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
    """Encode `s_waitcnt` for the RDNA3 (gfx11) layout used by gfx1151.

    The gfx11 ``s_waitcnt`` field layout differs from the gfx9/gfx10 split
    that :func:`_encode_waitcnt_gfx9_10` produces: the counters are
    contiguous, there is no split VMCNT, and LGKMCNT is 6 bits (not 4):

    * ``expcnt``  -> bits ``[2:0]``  (3 bits, max 7)
    * ``lgkmcnt`` -> bits ``[9:4]``  (6 bits, max 63)
    * ``vmcnt``   -> bits ``[15:10]`` (6 bits, max 63)

    This mirrors :func:`rocke.core.lower_llvm._encode_waitcnt_gfx11` (the
    layout was read off the ROCm AMDGPU assembler on a gfx1151 node). ``-1``
    means "no wait" and is encoded as the architectural maximum; explicit
    values clamp to the field maximum rather than wrapping, so a partial
    wait never silently becomes a full LDS/VMEM drain on wave32.
    """

    vm_b = 0x3F if vmcnt < 0 else min(max(vmcnt, 0), 0x3F)
    ec_b = 0x7 if expcnt < 0 else min(max(expcnt, 0), 0x7)
    lk_b = 0x3F if lgkmcnt < 0 else min(max(lgkmcnt, 0), 0x3F)
    return (ec_b & 0x7) | ((lk_b & 0x3F) << 4) | ((vm_b & 0x3F) << 10)


# Default arch for the HIP path. gfx950 is the historical CDNA target whose
# emitted source is the byte-identical baseline; callers that don't pass an
# arch (e.g. the in-tree coverage tests) keep getting exactly that output.
_DEFAULT_HIP_ARCH = "gfx950"


class _Lowerer:
    def __init__(self, kernel: KernelDef, *, arch: Optional[str] = None) -> None:
        self.kernel = kernel
        self.lines: List[str] = []
        self.smem_decls: List[str] = []
        self._indent = 1
        self._smem_counter = 0
        # Resolve the architecture seam. The HIP path has no separate ISA
        # backend class (it emits ``__builtin_amdgcn_*`` directly), so the
        # ``ArchTarget`` hardware facts drive every arch-keyed decision:
        # the ``s_waitcnt`` encoding, whether the MMA family is MFMA (CDNA)
        # or WMMA (RDNA/wave32), and whether ``ds_read_*_tr_*`` is available.
        from .arch import ArchTarget

        self.arch = ArchTarget.from_gfx(arch or _DEFAULT_HIP_ARCH)

    # -------------------- arch seam --------------------

    def _encode_waitcnt(self, vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
        """Arch-aware ``s_waitcnt`` immediate.

        CDNA (gfx9/gfx10) uses the split-VMCNT layout; RDNA gfx11 (gfx1151)
        uses the contiguous layout. Selection keys off the resolved
        :class:`ArchTarget` ``target_family`` so CDNA output is unchanged.
        """
        if self.arch.target_family in ("gfx11_rdna", "gfx12_rdna"):
            return _encode_waitcnt_gfx11(vmcnt, expcnt, lgkmcnt)
        return _encode_waitcnt_gfx9_10(vmcnt, expcnt, lgkmcnt)

    def _emit(self, text: str) -> None:
        self.lines.append(" " * self._indent + text)

    def _push_indent(self) -> None:
        self._indent += 1

    def _pop_indent(self) -> None:
        self._indent -= 1

    def lower_op(self, op: Op) -> None:
        method = getattr(self, f"_op_{op.name.replace('.', '_')}", None)
        if method is None:
            raise NotImplementedError(f"no HIP lowering for op {op.name!r}")
        method(op)

    def lower_region(self, region: Region) -> None:
        for op in region.ops:
            self.lower_op(op)

    # -------------------- arith --------------------

    def _op_arith_constant(self, op: Op) -> None:
        res = op.result
        ity = op.attrs.get("ity", "i32")
        val = op.attrs["value"]
        cpp_t = _HIP_TYPE[ity]
        if ity in ("f16", "f32"):
            literal = _f32_literal(float(val))
            if ity == "f16":
                self._emit(f"{cpp_t} {_name(res)} = (fp16){literal};")
            else:
                self._emit(f"{cpp_t} {_name(res)} = {literal};")
        else:
            self._emit(f"{cpp_t} {_name(res)} = {val};")

    def _op_arith_constant_vec(self, op: Op) -> None:
        res = op.result
        fill = op.attrs.get("fill", 0.0)
        if not isinstance(res.type, VectorType):
            raise NotImplementedError("constant_vec result must be a vector")
        count = res.type.count
        cpp_t = _type_to_hip(res.type)
        elem_name = res.type.elem.name
        if elem_name in ("f16", "bf16", "f32"):
            item = _f32_literal(float(fill))
        else:
            item = str(int(fill))
        items = ", ".join(item for _ in range(count))
        self._emit(f"{cpp_t} {_name(res)} = {{{items}}};")

    def _binary(self, op: Op, c_op: str) -> None:
        a, b = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = {_name(a)} {c_op} {_name(b)};"
        )

    def _op_arith_add(self, op: Op) -> None:
        self._binary(op, "+")

    def _op_arith_sub(self, op: Op) -> None:
        self._binary(op, "-")

    def _op_arith_mul(self, op: Op) -> None:
        self._binary(op, "*")

    def _op_arith_div(self, op: Op) -> None:
        self._binary(op, "/")

    def _op_arith_mod(self, op: Op) -> None:
        self._binary(op, "%")

    def _op_arith_cmp(self, op: Op) -> None:
        pred = op.attrs.get("pred", "lt")
        c_op = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}[
            pred
        ]
        a, b = op.operands
        self._emit(f"bool {_name(op.result)} = {_name(a)} {c_op} {_name(b)};")

    def _op_arith_select(self, op: Op) -> None:
        cond, lhs, rhs = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = {_name(cond)} ? {_name(lhs)} : {_name(rhs)};"
        )

    def _op_arith_and(self, op: Op) -> None:
        a, b = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = {_name(a)} & {_name(b)};"
        )

    def _op_arith_or(self, op: Op) -> None:
        a, b = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = {_name(a)} | {_name(b)};"
        )

    def _op_arith_smax(self, op: Op) -> None:
        a, b = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
            f"({_name(a)} > {_name(b)} ? {_name(a)} : {_name(b)});"
        )

    def _op_arith_smin(self, op: Op) -> None:
        a, b = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
            f"({_name(a)} < {_name(b)} ? {_name(a)} : {_name(b)});"
        )

    def _op_arith_zext(self, op: Op) -> None:
        (v,) = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = ({_type_to_hip(op.result.type)}){_name(v)};"
        )

    def _op_arith_sext(self, op: Op) -> None:
        (v,) = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = ({_type_to_hip(op.result.type)}){_name(v)};"
        )

    def _op_arith_trunc(self, op: Op) -> None:
        (v,) = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = ({_type_to_hip(op.result.type)}){_name(v)};"
        )

    def _op_arith_trunc_f32_to_f16(self, op: Op) -> None:
        (v,) = op.operands
        self._emit(f"fp16 {_name(op.result)} = (fp16){_name(v)};")

    # -------------------- gpu --------------------

    def _op_gpu_thread_id(self, op: Op) -> None:
        axis = op.attrs.get("axis", "x")
        self._emit(f"int {_name(op.result)} = (int)threadIdx.{axis};")

    def _op_gpu_block_id(self, op: Op) -> None:
        axis = op.attrs.get("axis", "x")
        self._emit(f"int {_name(op.result)} = (int)blockIdx.{axis};")

    # -------------------- memory --------------------

    def _op_tile_smem_alloc(self, op: Op) -> None:
        st = op.result.type
        assert isinstance(st, SmemType)
        dims = "][".join(str(d) for d in st.shape)
        elem = _HIP_TYPE[st.elem.name]
        nice = _name(op.result)
        decl = f"    __shared__ {elem} {nice}_storage[{dims}];"
        self.smem_decls.append(decl)
        # The IR uses the value as a typed token; record the storage name
        # and shape so subsequent ops can index it.
        op.attrs.setdefault("_storage", f"{nice}_storage")
        op.attrs.setdefault("_shape", list(st.shape))

    def _op_memref_global_load(self, op: Op) -> None:
        ptr, idx = op.operands
        self._emit(f"fp16 {_name(op.result)} = {_name(ptr)}[{_name(idx)}];")

    def _op_memref_global_store(self, op: Op) -> None:
        ptr, idx, val = op.operands
        self._emit(f"{_name(ptr)}[{_name(idx)}] = {_name(val)};")

    def _op_memref_global_load_vN(self, op: Op) -> None:
        ptr, idx = op.operands
        vec = int(op.attrs["vec"])
        elem_name = op.attrs.get("elem_type", "f16")
        prefix = {"f16": "f16x", "bf16": "bf16x"}.get(elem_name, "f16x")
        self._emit(
            f"{prefix}{vec} {_name(op.result)} = "
            f"*reinterpret_cast<const {prefix}{vec}*>({_name(ptr)} + {_name(idx)});"
        )

    def _op_tile_smem_store_vN(self, op: Op) -> None:
        smem = op.operands[0]
        value = op.operands[-1]
        indices = op.operands[1:-1]
        vec = int(op.attrs["vec"])
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem store_vN before smem_alloc was lowered")
        idx_str = "][".join(_name(i) for i in indices)
        elem_name = op.attrs.get("elem_type", "f16")
        prefix = {
            "f16": "f16x",
            "bf16": "bf16x",
            "f32": "f32x",
            "i32": "i32x",
            "i16": "i16x",
            "i8": "i8x",
            "fp8e4m3": "i8x",
            "bf8e5m2": "i8x",
        }.get(elem_name, "f16x")
        self._emit(
            f"*reinterpret_cast<{prefix}{vec}*>(&{storage}[{idx_str}]) = {_name(value)};"
        )

    def _op_tile_smem_store(self, op: Op) -> None:
        smem = op.operands[0]
        value = op.operands[-1]
        indices = op.operands[1:-1]
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem store before smem_alloc was lowered")
        idx_str = "][".join(_name(i) for i in indices)
        self._emit(f"{storage}[{idx_str}] = {_name(value)};")

    def _op_tile_smem_load_v4(self, op: Op) -> None:
        smem, row, col = op.operands
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem load before smem_alloc was lowered")
        nice = _name(op.result)
        self._emit(f"f16x4 {nice};")
        for i in range(4):
            self._emit(f"{nice}[{i}] = {storage}[{_name(row)}][{_name(col)} + {i}];")

    def _op_tile_mma(self, op: Op) -> None:
        """Lower the target-neutral ``tile.mma`` op for the HIP backend.

        The HIP source path has no ISA backend (it emits ``__builtin_amdgcn_*``
        directly), so the ``op_id`` is mapped back to the concrete
        ``tile.<op_id>`` op and dispatched through the existing per-op handler.
        This keeps the HIP emission identical to the legacy ISA-named path while
        the IRBuilder helpers route through :meth:`IRBuilder.mma`.
        """
        op_id = op.attrs["op_id"]
        legacy = Op(
            name=f"tile.{op_id}",
            operands=list(op.operands),
            results=list(op.results),
            attrs={k: v for k, v in op.attrs.items() if k != "op_id"},
            loc=op.loc,
        )
        self.lower_op(legacy)

    def _op_tile_mfma_f32_16x16x4_f32(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = __builtin_amdgcn_mfma_f32_16x16x4f32("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_32x32x2_f32(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x16 {_name(op.result)} = __builtin_amdgcn_mfma_f32_32x32x2f32("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_16x16x16_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = __builtin_amdgcn_mfma_f32_16x16x16f16("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_16x16x32_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = __builtin_amdgcn_mfma_f32_16x16x32_f16("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_32x32x8_bf16(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x16 {_name(op.result)} = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_32x32x8_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x16 {_name(op.result)} = __builtin_amdgcn_mfma_f32_32x32x8f16("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_32x32x16_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x16 {_name(op.result)} = __builtin_amdgcn_mfma_f32_32x32x16_f16("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_4x4x4_f16(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = __builtin_amdgcn_mfma_f32_4x4x4f16("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    # ---- WMMA f16 (RDNA3/3.5, gfx11, wave32) ----
    #
    # The RDNA half of the neutral-MMA contract. A WMMA ``op_id`` reaches this
    # handler only through ``_op_tile_mma`` (it rebuilds ``tile.<op_id>``). The
    # hardware-verified wave32 builtin on gfx1151 (Strix Halo) is
    # ``__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(<16 x half> a,
    # <16 x half> b, <8 x float> c) -> <8 x float>`` -- the same fragment
    # layout the LLVM WMMA path emits (A/B per-lane ``f16x16``, accumulator
    # ``f32x8``). It is gated to RDNA targets: MFMA-only CDNA targets have no
    # WMMA instruction, so emitting it there would mis-compile, and the
    # historical CDNA HIP source must stay MFMA. The third ``w32`` argument is
    # the wave-mode / opsel flag (false => no input/output modifiers), matching
    # clang's two-operand-flag convention for the gfx11 WMMA builtin.

    def _op_tile_wmma_f32_16x16x16_f16(self, op: Op) -> None:
        self._require_wmma_arch("wmma_f32_16x16x16_f16")
        a, b, c = op.operands
        self._emit(
            f"f32x8 {_name(op.result)} = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32("
            f"{_name(a)}, {_name(b)}, {_name(c)});"
        )

    def _op_tile_wmma_gfx12_f32_16x16x16_f16(self, op: Op) -> None:
        # RDNA4 builtin: distinct ``_gfx12`` suffix, 8-wide operands.
        self._require_wmma_arch("wmma_gfx12_f32_16x16x16_f16")
        a, b, c = op.operands
        self._emit(
            f"f32x8 {_name(op.result)} = "
            f"__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12("
            f"{_name(a)}, {_name(b)}, {_name(c)});"
        )

    def _op_tile_wmma_gfx1250_f32_16x16x32_f16(self, op: Op) -> None:
        # gfx1250 builtin: K=32, 16-wide f16 operands, 8-operand form:
        # (negA, A, negB, B, fmt, C, reuseA, reuseB).
        self._require_wmma_arch("wmma_gfx1250_f32_16x16x32_f16")
        a, b, c = op.operands
        self._emit(
            f"f32x8 {_name(op.result)} = "
            f"__builtin_amdgcn_wmma_f32_16x16x32_f16("
            f"false, {_name(a)}, false, {_name(b)}, (int16_t)0, "
            f"{_name(c)}, false, false);"
        )

    def _op_tile_wmma_gfx1250_f32_16x16x64_fp8_fp8(self, op: Op) -> None:
        self._emit_wmma_gfx1250_fp8(op, "fp8_fp8")

    def _op_tile_wmma_gfx1250_f32_16x16x64_fp8_bf8(self, op: Op) -> None:
        self._emit_wmma_gfx1250_fp8(op, "fp8_bf8")

    def _op_tile_wmma_gfx1250_f32_16x16x64_bf8_fp8(self, op: Op) -> None:
        self._emit_wmma_gfx1250_fp8(op, "bf8_fp8")

    def _op_tile_wmma_gfx1250_f32_16x16x64_bf8_bf8(self, op: Op) -> None:
        self._emit_wmma_gfx1250_fp8(op, "bf8_bf8")

    def _emit_wmma_gfx1250_fp8(self, op: Op, ab: str) -> None:
        # gfx1250 K=64 FP8/BF8 builtin: A/B are <8 x i32> (32 low-bit
        # bytes per lane), 6-operand form (A, B, fmt, C, reuseA, reuseB).
        self._require_wmma_arch(f"wmma_gfx1250_f32_16x16x64_{ab}")
        a, b, c = op.operands
        self._emit(
            f"f32x8 {_name(op.result)} = "
            f"__builtin_amdgcn_wmma_f32_16x16x64_{ab}("
            f"{_name(a)}, {_name(b)}, (int16_t)0, {_name(c)}, false, false);"
        )

    def _op_tile_wmma_gfx1250_f32_16x16x32_bf16(self, op: Op) -> None:
        # Same gfx1250 K=32 ABI as the f16 form, but operands are true bf16 vectors.
        self._require_wmma_arch("wmma_gfx1250_f32_16x16x32_bf16")
        a, b, c = op.operands
        self._emit(
            f"f32x8 {_name(op.result)} = "
            f"__builtin_amdgcn_wmma_f32_16x16x32_bf16("
            f"false, {_name(a)}, false, {_name(b)}, (int16_t)0, "
            f"{_name(c)}, false, false);"
        )

    # ---- WMMA iu8 (RDNA3/3.5, gfx11, wave32) — integer matrix engine ----
    #
    # The integer twin of the f16 WMMA handler. Per-lane A/B operands are
    # ``i32x4`` (16 int8 packed 4-per-i32) and the accumulator / result are
    # ``i32x8``. The builtin's two leading bool flags select each operand's
    # *signedness* (``true`` => signed; verified on gfx11 — passing ``false``
    # makes the unit compute the *unsigned* dot product), and the trailing bool
    # is the clamp flag (``false`` => exact i32 wrap). This mirrors the verified
    # LLVM-IR emission in ``core/isa/backend.py`` (``_emit_wmma_int``):
    # ``llvm.amdgcn.wmma.i32.16x16x16.iu8(i1 1, A, i1 1, B, C, i1 0)``.

    def _op_tile_wmma_i32_16x16x16_iu8(self, op: Op) -> None:
        self._require_wmma_arch("wmma_i32_16x16x16_iu8")
        a, b, c = op.operands
        self._emit(
            f"i32x8 {_name(op.result)} = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32("
            f"true, {_name(a)}, true, {_name(b)}, {_name(c)}, false);"
        )

    # ---- WMMA iu4 (RDNA3/3.5, gfx11, wave32) — int4 matrix engine ----
    #
    # The int4 sibling of the iu8 handler. Per-lane A/B operands are ``i32x2``
    # (16 signed int4 packed 8-per-i32); accumulator / result are ``i32x8``.
    # Same flag convention as iu8: leading bools select signed operands, the
    # trailing bool is the (off) clamp. Mirrors the LLVM emission
    # ``llvm.amdgcn.wmma.i32.16x16x16.iu4(i1 1, A, i1 1, B, C, i1 0)``.
    def _op_tile_wmma_i32_16x16x16_iu4(self, op: Op) -> None:
        self._require_wmma_arch("wmma_i32_16x16x16_iu4")
        a, b, c = op.operands
        self._emit(
            f"i32x8 {_name(op.result)} = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32("
            f"true, {_name(a)}, true, {_name(b)}, {_name(c)}, false);"
        )

    def _require_wmma_arch(self, op_id: str) -> None:
        """Reject a WMMA op on a target whose ISA has no WMMA instruction.

        WMMA is an RDNA/gfx11 instruction; CDNA/MFMA targets must never emit
        it. We key off the resolved :class:`ArchTarget` MMA catalog so the
        gate is data-driven (a future RDNA target with WMMA passes
        automatically once its catalog row exists)."""
        if self.arch.mma.by_op_id(op_id) is None:
            raise NotImplementedError(
                f"WMMA op {op_id!r} is not available on {self.arch.gfx} "
                f"(WMMA is an RDNA/gfx11 instruction; this is a "
                f"{self.arch.family.upper()} target). The MFMA atoms are the "
                f"matrix path on CDNA."
            )

    def _require_ds_read_tr(self, op_id: str) -> None:
        """Reject an LDS transpose-read on a target without ``ds_read_*_tr_*``.

        ``ds_read_b{64,128}_tr_b16`` is a gfx950-class instruction; gfx942 and
        the RDNA gfx1151 target do not have it (``memory.has_ds_read_tr`` is
        False). Gating data-driven off the :class:`ArchTarget` keeps the
        gfx950 emission unchanged while preventing a silent mis-compile on the
        other targets."""
        if not self.arch.memory.has_ds_read_tr:
            raise NotImplementedError(
                f"transpose LDS read {op_id!r} is not available on "
                f"{self.arch.gfx} (no ds_read_*_tr_* on this target); it is a "
                f"gfx950-class instruction."
            )

    # ---- FP8 / BF8 MFMA (gfx940+) ----
    #
    # IR operand types are ``<8 x fp8e4m3>`` / ``<8 x bf8e5m2>`` --
    # both map to ``i8x8`` in the HIP prologue. On ROCm clang the FP8 /
    # BF8 MFMA builtins take the packed 64-bit operand as ``long``. We
    # ``__builtin_memcpy`` the byte vector into that scalar type so the
    # generated source preserves the raw bits without relying on aliasing.

    def _emit_fp8_mfma(self, op: Op, *, out_vec: int, builtin: str) -> None:
        a, b, c = op.operands
        a_pk = f"{_name(op.result)}_a_pk"
        b_pk = f"{_name(op.result)}_b_pk"
        out_t = f"f32x{out_vec}"
        self._emit(
            f"long {a_pk}; __builtin_memcpy(&{a_pk}, &{_name(a)}, sizeof({a_pk}));"
        )
        self._emit(
            f"long {b_pk}; __builtin_memcpy(&{b_pk}, &{_name(b)}, sizeof({b_pk}));"
        )
        self._emit(
            f"{out_t} {_name(op.result)} = {builtin}("
            f"{a_pk}, {b_pk}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_16x16x32_fp8(self, op: Op) -> None:
        self._emit_fp8_mfma(
            op, out_vec=4, builtin="__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8"
        )

    def _op_tile_mfma_f32_16x16x32_bf8(self, op: Op) -> None:
        self._emit_fp8_mfma(
            op, out_vec=4, builtin="__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8"
        )

    def _op_tile_mfma_f32_32x32x16_fp8(self, op: Op) -> None:
        self._emit_fp8_mfma(
            op, out_vec=16, builtin="__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8"
        )

    def _op_tile_mfma_f32_32x32x16_bf8(self, op: Op) -> None:
        self._emit_fp8_mfma(
            op, out_vec=16, builtin="__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8"
        )

    def _op_vector_bitcast(self, op: Op) -> None:
        (v,) = op.operands
        tgt_name = _type_to_hip(op.result.type)
        self._emit(
            f"{tgt_name} {_name(op.result)}; "
            f"__builtin_memcpy(&{_name(op.result)}, &{_name(v)}, sizeof({tgt_name}));"
        )

    def _op_tile_readfirstlane(self, op: Op) -> None:
        (v,) = op.operands
        ty = _type_to_hip(op.result.type)
        self._emit(
            f"{ty} {_name(op.result)} = __builtin_amdgcn_readfirstlane({_name(v)});"
        )

    def _op_tile_pin_sgpr(self, op: Op) -> None:
        # AMDGPU idiom: `asm volatile("" : "+s"(x))` keeps x in an
        # SGPR across uses. We emit a fresh variable initialised
        # from the input then apply the constraint to that variable.
        (v,) = op.operands
        ty = _type_to_hip(op.result.type)
        self._emit(f"{ty} {_name(op.result)} = {_name(v)};")
        self._emit(f'asm volatile("" : "+s"({_name(op.result)}));')

    def _op_tile_wave_ballot(self, op: Op) -> None:
        # HIP exposes `__ballot(int pred)` which on AMD wave64 returns
        # a 64-bit lane mask.
        (pred,) = op.operands
        self._emit(f"int64_t {_name(op.result)} = __ballot({_name(pred)});")

    def _op_tile_wave_all(self, op: Op) -> None:
        # `__all(pred)` returns 1 iff every active lane's pred is non-zero.
        (pred,) = op.operands
        self._emit(f"int32_t {_name(op.result)} = __all({_name(pred)});")

    def _op_tile_wave_any(self, op: Op) -> None:
        (pred,) = op.operands
        self._emit(f"int32_t {_name(op.result)} = __any({_name(pred)});")

    def _op_tile_smem_addr_of(self, op: Op) -> None:
        # The SSA value ``smem`` is the result of a ``tile.smem_alloc``,
        # which materialises a ``__shared__`` array named
        # ``<name>_storage`` (see ``_op_tile_smem_alloc``). The SSA value
        # name itself is NOT declared in the body, so we must convert
        # through the storage symbol.
        (smem,) = op.operands
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem_addr_of before smem_alloc was lowered")
        self._emit(f"int64_t {_name(op.result)} = (int64_t)(&{storage}[0]);")

    def _op_tile_smem_ptr_add(self, op: Op) -> None:
        base, off = op.operands
        self._emit(f"int64_t {_name(op.result)} = {_name(base)} + {_name(off)};")

    def _op_tile_global_ptr_add(self, op: Op) -> None:
        # ptr + byte_off as a new global pointer (byte arithmetic). The
        # result feeds make_buffer_rsrc (which casts to void*), so a char*
        # is the right C++ type. byte_off is i64 -> no 2 GiB overflow.
        ptr, off = op.operands
        self._emit(
            f"const char* {_name(op.result)} = "
            f"(const char*){_name(ptr)} + (int64_t){_name(off)};"
        )

    def _op_tile_buffer_load_vN_f16(self, op: Op) -> None:
        # Lowers to ``__builtin_amdgcn_raw_buffer_load_b{32,64,128}``,
        # which on ROCm 7 / clang 20 takes ``__amdgpu_buffer_rsrc_t`` (aka
        # ``rsrc_t`` in the prologue) and returns the matching i32 /
        # i32x2 / i32x4 raw payload. We then bitcast to ``f16xN`` via
        # memcpy because that's the canonical ABI-safe punning in HIP C++.
        rsrc, voffset, soffset = op.operands
        dwords = int(op.attrs["dwords"])
        halves = dwords * 2
        b_suffix = {1: "_b32", 2: "_b64", 4: "_b128"}[dwords]
        raw_t = "int" if dwords == 1 else f"i32x{dwords}"
        tmp = f"_blraw_{_name(op.result).lstrip('%')}"
        self._emit(
            f"{raw_t} {tmp} = __builtin_amdgcn_raw_buffer_load{b_suffix}("
            f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
        )
        self._emit(
            f"f16x{halves} {_name(op.result)}; "
            f"__builtin_memcpy(&{_name(op.result)}, &{tmp}, {dwords * 4});"
        )

    def _op_tile_buffer_load_f16(self, op: Op) -> None:
        rsrc, voffset, soffset = op.operands
        tmp = f"_bl_{_name(op.result).lstrip('%')}"
        self._emit(
            f"unsigned int {tmp} = (unsigned int)__builtin_amdgcn_raw_buffer_load_b32("
            f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
        )
        # take low 16 bits as half (matches the LLVM lowering's i32 → i16 trunc)
        self._emit(
            f"fp16 {_name(op.result)}; "
            f"unsigned short _u16_{tmp} = (unsigned short)({tmp} & 0xFFFFu); "
            f"__builtin_memcpy(&{_name(op.result)}, &_u16_{tmp}, 2);"
        )

    def _op_tile_buffer_load_vN(self, op: Op) -> None:
        # Dtype-generic vectorised buffer load.
        # Loads <dwords x i32> via __builtin_amdgcn_raw_buffer_load_b{32,64,128},
        # then memcpy-punches to the target vector type (f16xN, bf16xN, f32xN).
        rsrc, voffset, soffset = op.operands
        dwords = int(op.attrs["dwords"])
        elem_type = op.attrs["elem_type"]
        n = op.result.type.count
        # Map our IR elem name → HIP vector typedef prefix
        _HIP_VEC = {"f16": "f16x", "bf16": "bf16x", "f32": "f32x", "i32": "i32x"}
        hip_vec_t = f"{_HIP_VEC[elem_type]}{n}"
        b_suffix = {1: "_b32", 2: "_b64", 4: "_b128"}[dwords]
        raw_t = "int" if dwords == 1 else f"i32x{dwords}"
        tmp = f"_blraw_{_name(op.result).lstrip('%')}"
        self._emit(
            f"{raw_t} {tmp} = __builtin_amdgcn_raw_buffer_load{b_suffix}("
            f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
        )
        self._emit(
            f"{hip_vec_t} {_name(op.result)}; "
            f"__builtin_memcpy(&{_name(op.result)}, &{tmp}, {dwords * 4});"
        )

    def _op_tile_buffer_load(self, op: Op) -> None:
        # Dtype-generic scalar buffer load.
        # 2-byte types (f16, bf16) → __builtin_amdgcn_raw_buffer_load_b16.
        # 4-byte types (f32, i32) → __builtin_amdgcn_raw_buffer_load_b32.
        rsrc, voffset, soffset = op.operands
        elem_type = op.attrs["elem_type"]
        hip_t = _HIP_TYPE[elem_type]
        tmp = f"_bl_{_name(op.result).lstrip('%')}"
        if elem_type in ("f16", "bf16"):
            self._emit(
                f"unsigned short {tmp} = (unsigned short)"
                f"__builtin_amdgcn_raw_buffer_load_b16("
                f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
            )
            self._emit(
                f"{hip_t} {_name(op.result)}; "
                f"__builtin_memcpy(&{_name(op.result)}, &{tmp}, 2);"
            )
        else:
            self._emit(
                f"unsigned int {tmp} = (unsigned int)"
                f"__builtin_amdgcn_raw_buffer_load_b32("
                f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
            )
            self._emit(
                f"{hip_t} {_name(op.result)}; "
                f"__builtin_memcpy(&{_name(op.result)}, &{tmp}, 4);"
            )

    def _op_tile_buffer_store_vN_f16(self, op: Op) -> None:
        # Store ops have no SSA result; use the value operand's name to
        # disambiguate per-call temporaries (multiple store_vN ops in the
        # same block would otherwise redeclare ``_ub_x``).
        rsrc, voffset, soffset, val = op.operands
        dwords = int(op.attrs["dwords"])
        tmp = f"_ub_{_name(val).lstrip('%')}"
        if dwords == 1:
            self._emit(
                f"unsigned int {tmp} = 0; "
                f"__builtin_memcpy(&{tmp}, &{_name(val)}, 4); "
                f"__builtin_amdgcn_raw_buffer_store_b32({tmp}, "
                f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
            )
        else:
            b_suffix = {2: "_b64", 4: "_b128"}[dwords]
            self._emit(
                f"i32x{dwords} {tmp}; "
                f"__builtin_memcpy(&{tmp}, &{_name(val)}, {dwords * 4}); "
                f"__builtin_amdgcn_raw_buffer_store{b_suffix}({tmp}, "
                f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
            )

    def _op_tile_buffer_store_f16(self, op: Op) -> None:
        rsrc, voffset, soffset, val = op.operands
        tmp = f"_u16_{_name(val).lstrip('%')}"
        self._emit(
            f"unsigned short {tmp} = 0; "
            f"__builtin_memcpy(&{tmp}, &{_name(val)}, 2); "
            f"__builtin_amdgcn_raw_buffer_store_b16({tmp}, "
            f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
        )

    def _op_tile_async_buffer_load_lds_addr(self, op: Op) -> None:
        # Call the LLVM intrinsic through the prologue's ``_llvm_amdgcn_*``
        # shim. The builtin form (``__builtin_amdgcn_raw_ptr_buffer_load_lds``)
        # restricts ``size`` to {1, 2, 4}, but the LLVM intrinsic accepts
        # 12 / 16 (i.e. dwords ∈ {1, 3, 4}). compv4 and split-KV attention
        # need the 16-byte form.
        rsrc, lds_addr, voff, soff = op.operands
        dwords = int(op.attrs["dwords"])
        size_bytes = dwords * 4
        self._emit(
            f"_llvm_amdgcn_raw_ptr_buffer_load_lds("
            f"{_name(rsrc)}, "
            f"(__attribute__((address_space(3))) void*)({_name(lds_addr)}), "
            f"{size_bytes}, {_name(voff)}, {_name(soff)}, 0, 0);"
        )

    def _op_tile_sync(self, op: Op) -> None:
        self._emit("__syncthreads();")

    def _op_tile_sync_half_block(self, op: Op) -> None:
        # `if (selector) __builtin_amdgcn_s_barrier();` -- the
        # staggered half-block barrier pattern used by interwave
        # ping-pong kernels to sync only one half of the workgroup.
        (sel,) = op.operands
        self._emit(f"if ({_name(sel)}) {{ __builtin_amdgcn_s_barrier(); }}")

    def _op_tile_sync_lds_only(self, op: Op) -> None:
        # Drain LDS counter (lgkmcnt) but leave VMEM in flight, then
        # the workgroup barrier. Same encoding as ``block_sync_lds`` in
        # ``ck_tile/core/arch/arch.hpp``. Used by the async-DMA
        # ping-pong pipeline so the next iter's ``buffer_load_lds``
        # keeps streaming across this barrier.
        mask = self._encode_waitcnt(vmcnt=-1, expcnt=-1, lgkmcnt=0)
        self._emit(f"__builtin_amdgcn_s_waitcnt({mask});")
        self._emit("__syncthreads();")

    def _op_tile_s_barrier_bare(self, op: Op) -> None:
        # Bare workgroup rendezvous, NO implicit waitcnt -- the caller
        # issues the explicit s_waitcnt counters. Mirrors the LLVM-direct
        # ``call void @llvm.amdgcn.s.barrier()`` (no preceding waitcnt).
        self._emit("__builtin_amdgcn_s_barrier();")

    def _op_tile_s_waitcnt(self, op: Op) -> None:
        vm = int(op.attrs.get("vmcnt", -1))
        lk = int(op.attrs.get("lgkmcnt", -1))
        ec = int(op.attrs.get("expcnt", -1))
        mask = self._encode_waitcnt(vm, ec, lk)
        self._emit(f"__builtin_amdgcn_s_waitcnt({mask});")

    def _op_tile_iglp_opt(self, op: Op) -> None:
        self._emit(f"__builtin_amdgcn_iglp_opt({int(op.attrs.get('level', 0))});")

    def _op_tile_sched_barrier(self, op: Op) -> None:
        self._emit(f"__builtin_amdgcn_sched_barrier({int(op.attrs.get('mask', 0))});")

    def _op_tile_sched_group_barrier(self, op: Op) -> None:
        m = int(op.attrs["mask"])
        c = int(op.attrs["count"])
        g = int(op.attrs.get("group", 0))
        self._emit(f"__builtin_amdgcn_sched_group_barrier({m}, {c}, {g});")

    def _op_tile_s_setprio(self, op: Op) -> None:
        self._emit(f"__builtin_amdgcn_s_setprio({int(op.attrs['level'])});")

    def _op_memref_global_store_vN(self, op: Op) -> None:
        ptr, idx, val = op.operands
        vec = int(op.attrs["vec"])
        self._emit(
            f"*reinterpret_cast<f16x{vec}*>({_name(ptr)} + {_name(idx)}) = "
            f"{_name(val)};"
        )

    def _op_memref_global_atomic_add_f32(self, op: Op) -> None:
        ptr, idx, val = op.operands
        self._emit(f"atomicAdd({_name(ptr)} + {_name(idx)}, {_name(val)});")

    def _op_vector_extract(self, op: Op) -> None:
        (v,) = op.operands
        i = op.attrs["index"]
        elem_t = v.type.elem if isinstance(v.type, VectorType) else v.type
        self._emit(f"{_HIP_TYPE[elem_t.name]} {_name(op.result)} = {_name(v)}[{i}];")

    def _op_vector_trunc_f32_to_f16(self, op: Op) -> None:
        # Legacy op name; the post-merge IR emits ``vector.trunc_f32_to``
        # with a ``target`` attribute. Kept here for back-compat with
        # any callers still emitting the old name.
        (v,) = op.operands
        n = v.type.count if isinstance(v.type, VectorType) else 1
        nice = _name(op.result)
        self._emit(f"f16x{n} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = (fp16){_name(v)}[{i}];")

    # -------------------- arith: float --------------------

    def _op_arith_fadd(self, op: Op) -> None:
        self._binary(op, "+")

    def _op_arith_fsub(self, op: Op) -> None:
        self._binary(op, "-")

    def _op_arith_fmul(self, op: Op) -> None:
        self._binary(op, "*")

    def _op_arith_fdiv(self, op: Op) -> None:
        self._binary(op, "/")

    def _op_arith_fneg(self, op: Op) -> None:
        (v,) = op.operands
        self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = -{_name(v)};")

    def _op_arith_fabs(self, op: Op) -> None:
        (v,) = op.operands
        ty = _type_to_hip(op.result.type)
        # `__builtin_fabsf` lowers to a free `abs` modifier on AMDGPU
        # (exactly what the LLVM `llvm.fabs.f32` intrinsic produces);
        # for f16/bf16 the matching builtin yields the same modifier.
        helper = {
            "f32": "fabsf",
            "f16": "__builtin_fabsf",
            "bf16": "__builtin_fabsf",
        }.get(op.result.type.name, "fabsf")
        self._emit(f"{ty} {_name(op.result)} = ({ty}){helper}((float){_name(v)});")

    def _op_arith_fma(self, op: Op) -> None:
        a, b, c = op.operands
        ty = _type_to_hip(op.result.type)
        # `fmaf(a, b, c)` lowers to `v_fma_f32` on AMDGPU for f32; for
        # f16/bf16 we promote to f32, fma, and demote — semantically
        # identical to the IR-level `arith.fma` and matches the
        # `llvm.fmuladd` lowering.
        self._emit(
            f"{ty} {_name(op.result)} = ({ty})fmaf("
            f"(float){_name(a)}, (float){_name(b)}, (float){_name(c)});"
        )

    def _op_arith_fmax3(self, op: Op) -> None:
        a, b, c = op.operands
        ty = _type_to_hip(op.result.type)
        # Two ternary ops in one statement; on AMDGPU the chain folds
        # into a single `v_max3_f32`.
        self._emit(
            f"{ty} {_name(op.result)} = "
            f"(({_name(b)} > {_name(c)}) ? {_name(b)} : {_name(c)});"
        )
        self._emit(
            f"{_name(op.result)} = "
            f"({_name(a)} > {_name(op.result)}) ? {_name(a)} : {_name(op.result)};"
        )

    def _op_arith_fmin3(self, op: Op) -> None:
        a, b, c = op.operands
        ty = _type_to_hip(op.result.type)
        self._emit(
            f"{ty} {_name(op.result)} = "
            f"(({_name(b)} < {_name(c)}) ? {_name(b)} : {_name(c)});"
        )
        self._emit(
            f"{_name(op.result)} = "
            f"({_name(a)} < {_name(op.result)}) ? {_name(a)} : {_name(op.result)};"
        )

    def _op_arith_fmax(self, op: Op) -> None:
        a, b = op.operands
        # Ternary works for fp16/bf16/f32 in C++ and folds to v_max on AMDGPU.
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
            f"({_name(a)} > {_name(b)}) ? {_name(a)} : {_name(b)};"
        )

    def _op_arith_fmin(self, op: Op) -> None:
        a, b = op.operands
        self._emit(
            f"{_type_to_hip(op.result.type)} {_name(op.result)} = "
            f"({_name(a)} < {_name(b)}) ? {_name(a)} : {_name(b)};"
        )

    def _op_arith_fcmp(self, op: Op) -> None:
        pred = op.attrs["pred"]
        a, b = op.operands
        # IEEE ordered predicates: ``a OP b`` evaluates to true only when
        # neither operand is NaN AND the relation holds. Ordered comparisons
        # ``< <= > >= == !=`` in C++ on float types return false when either
        # operand is NaN, which matches the LLVM ordered-predicate semantics.
        # ``ord`` / ``uno`` (NaN-test only) need explicit isnan calls.
        op_map = {
            "olt": "<",
            "ole": "<=",
            "ogt": ">",
            "oge": ">=",
            "oeq": "==",
            "one": "!=",
        }
        if pred in op_map:
            self._emit(
                f"bool {_name(op.result)} = "
                f"(!isnan(float({_name(a)})) && !isnan(float({_name(b)})) "
                f"&& ({_name(a)} {op_map[pred]} {_name(b)}));"
            )
        elif pred == "ord":
            self._emit(
                f"bool {_name(op.result)} = "
                f"(!isnan(float({_name(a)})) && !isnan(float({_name(b)})));"
            )
        elif pred == "uno":
            self._emit(
                f"bool {_name(op.result)} = "
                f"(isnan(float({_name(a)})) || isnan(float({_name(b)})));"
            )
        else:
            raise NotImplementedError(f"unknown fcmp predicate {pred!r}")

    # -------------------- arith: math (transcendentals) --------------------
    #
    # The strategy for f16/bf16 is "promote, compute, demote" via the
    # standard library f32 entry points. clang on AMDGPU folds the
    # promote-compute-demote sequence into ``__builtin_amdgcn_*`` calls
    # the same way the direct-LLVM path does, so the lowered C++ ends up
    # at the same ISA after `-O3` (the only cost is at -O0 debug-mode).
    #
    # For f32, ``exp2f`` / ``__exp2f``, ``sqrtf``, ``tanhf`` are HIP
    # device-runtime math entry points; ``__builtin_amdgcn_rcpf`` and
    # ``__builtin_amdgcn_rsqf`` are direct hardware reciprocal /
    # reciprocal-sqrt builtins (single ISA op).

    def _math1(
        self, op: Op, fn_f32: str, *, prefer_amdgcn_builtin: bool = False
    ) -> None:
        (v,) = op.operands
        tname = op.result.type.name
        cpp_t = _type_to_hip(op.result.type)
        # ``__builtin_amdgcn_*`` only takes float. For non-f32 types,
        # round-trip via float so the math runs at f32 precision.
        if tname == "f32":
            self._emit(f"{cpp_t} {_name(op.result)} = {fn_f32}({_name(v)});")
        else:
            self._emit(
                f"{cpp_t} {_name(op.result)} = ({cpp_t}){fn_f32}((float){_name(v)});"
            )

    def _op_math_exp2(self, op: Op) -> None:
        # ``__exp2f`` is HIP's device runtime exp2 entry point; for fp16/bf16
        # we promote to f32 first.
        self._math1(op, "exp2f")

    def _op_math_log2(self, op: Op) -> None:
        # ``log2f`` is HIP's device runtime base-2 log entry point; for
        # fp16/bf16 we promote to f32 first (same promote-compute-demote
        # shape as exp2 above).
        self._math1(op, "log2f")

    def _op_math_rcp(self, op: Op) -> None:
        # AMDGPU has a hardware reciprocal; emit the builtin directly for
        # f32, promote-compute-demote for f16/bf16.
        (v,) = op.operands
        tname = op.result.type.name
        cpp_t = _type_to_hip(op.result.type)
        if tname == "f32":
            self._emit(
                f"{cpp_t} {_name(op.result)} = __builtin_amdgcn_rcpf({_name(v)});"
            )
        else:
            self._emit(
                f"{cpp_t} {_name(op.result)} = "
                f"({cpp_t})__builtin_amdgcn_rcpf((float){_name(v)});"
            )

    def _op_math_rcp_fast(self, op: Op) -> None:
        # Identical to math.rcp on HIP: the builtin already lowers to v_rcp_f32.
        (v,) = op.operands
        tname = op.result.type.name
        cpp_t = _type_to_hip(op.result.type)
        if tname == "f32":
            self._emit(
                f"{cpp_t} {_name(op.result)} = __builtin_amdgcn_rcpf({_name(v)});"
            )
        else:
            self._emit(
                f"{cpp_t} {_name(op.result)} = "
                f"({cpp_t})__builtin_amdgcn_rcpf((float){_name(v)});"
            )

    def _op_math_sqrt(self, op: Op) -> None:
        self._math1(op, "sqrtf")

    def _op_math_rsqrt(self, op: Op) -> None:
        # AMDGPU's reciprocal-sqrt builtin (single ISA op on gfx9+).
        (v,) = op.operands
        tname = op.result.type.name
        cpp_t = _type_to_hip(op.result.type)
        if tname == "f32":
            self._emit(
                f"{cpp_t} {_name(op.result)} = __builtin_amdgcn_rsqf({_name(v)});"
            )
        else:
            self._emit(
                f"{cpp_t} {_name(op.result)} = "
                f"({cpp_t})__builtin_amdgcn_rsqf((float){_name(v)});"
            )

    def _op_math_tanh(self, op: Op) -> None:
        self._math1(op, "tanhf")

    # -------------------- arith: casts and bitcast --------------------

    def _op_arith_cast_to_f32(self, op: Op) -> None:
        # Element-promote f16/bf16 -> f32. A C-style ``(float)`` cast on
        # an fp16 / bf16 scalar is the canonical lowering and folds to the
        # right cvt instruction in amdclang.
        (v,) = op.operands
        self._emit(f"float {_name(op.result)} = (float){_name(v)};")

    def _op_arith_cast_f32_to(self, op: Op) -> None:
        # f32 -> {f16, bf16}. The IR pins the target via ``target`` attr.
        (v,) = op.operands
        cpp_t = _type_to_hip(op.result.type)
        self._emit(f"{cpp_t} {_name(op.result)} = ({cpp_t}){_name(v)};")

    def _op_arith_sitofp_f32(self, op: Op) -> None:
        # i32 -> f32. C-style cast is sufficient.
        (v,) = op.operands
        self._emit(f"float {_name(op.result)} = (float){_name(v)};")

    def _op_arith_cvt_fp8_to_f32(self, op: Op) -> None:
        # AMDGPU's per-byte fp8e4m3 -> f32 builtin.
        (v,) = op.operands
        self._emit(
            f"float {_name(op.result)} = __builtin_amdgcn_cvt_f32_fp8("
            f"(unsigned int)(unsigned char){_name(v)}, 0);"
        )

    def _op_arith_cvt_bf8_to_f32(self, op: Op) -> None:
        # AMDGPU's per-byte bf8e5m2 -> f32 builtin.
        (v,) = op.operands
        self._emit(
            f"float {_name(op.result)} = __builtin_amdgcn_cvt_f32_bf8("
            f"(unsigned int)(unsigned char){_name(v)}, 0);"
        )

    def _op_arith_cvt_f32_to_fp8(self, op: Op) -> None:
        # AMDGPU's packed f32 -> fp8e4m3 builtin. We feed one live f32
        # plus a +0.0f filler into the second slot, select word-lane 0,
        # and extract the low byte as the single-element result.
        (v,) = op.operands
        tmp = f"{_name(op.result)}_pk"
        self._emit(
            f"unsigned int {tmp} = __builtin_amdgcn_cvt_pk_fp8_f32("
            f"{_name(v)}, 0.0f, 0u, false);"
        )
        self._emit(f"fp8e4m3 {_name(op.result)} = (fp8e4m3)({tmp} & 0xffu);")

    def _op_arith_cvt_f32_to_bf8(self, op: Op) -> None:
        # AMDGPU's packed f32 -> bf8e5m2 builtin (e5m2 sibling).
        (v,) = op.operands
        tmp = f"{_name(op.result)}_pk"
        self._emit(
            f"unsigned int {tmp} = __builtin_amdgcn_cvt_pk_bf8_f32("
            f"{_name(v)}, 0.0f, 0u, false);"
        )
        self._emit(f"bf8e5m2 {_name(op.result)} = (bf8e5m2)({tmp} & 0xffu);")

    def _op_arith_cvt_pk_f32_fp8x4(self, op: Op) -> None:
        """Packed <4 x fp8e4m3> -> <4 x f32> via two
        ``__builtin_amdgcn_cvt_pk_f32_fp8`` calls.

        e4m3 sibling of :meth:`_op_arith_cvt_pk_f32_bf8x4`; mirrors the
        LLVM lowering's two-packed-cvt + concat shape but in HIP
        builtin form so the debug source compiles with hipcc. The
        result-vector C++ type is taken from
        :func:`_type_to_hip` (``f32x4``) — the historical
        ``vec<f32x4>`` spelling was not a real C++ type and broke
        ``hipcc --genco`` on the attention kernels that consume this
        cvt (HIP_LOWERINGS_REPORT.md §4a).
        """
        (v,) = op.operands
        nice = _name(op.result)
        packed = f"{nice}_p"
        lo = f"{nice}_lo"
        hi = f"{nice}_hi"
        res_t = _type_to_hip(op.result.type)
        pair_t = _type_to_hip(VectorType(op.result.type.elem, 2))
        self._emit(
            f"unsigned int {packed}; "
            f"__builtin_memcpy(&{packed}, &{_name(v)}, sizeof({packed}));"
        )
        self._emit(f"{pair_t} {lo} = __builtin_amdgcn_cvt_pk_f32_fp8({packed}, false);")
        self._emit(f"{pair_t} {hi} = __builtin_amdgcn_cvt_pk_f32_fp8({packed}, true);")
        self._emit(f"{res_t} {nice};")
        self._emit(f"{nice}[0] = {lo}.x;")
        self._emit(f"{nice}[1] = {lo}.y;")
        self._emit(f"{nice}[2] = {hi}.x;")
        self._emit(f"{nice}[3] = {hi}.y;")

    def _op_arith_cvt_pk_f32_bf8x4(self, op: Op) -> None:
        """e5m2 sibling of :meth:`_op_arith_cvt_pk_f32_fp8x4`."""
        (v,) = op.operands
        nice = _name(op.result)
        packed = f"{nice}_p"
        lo = f"{nice}_lo"
        hi = f"{nice}_hi"
        res_t = _type_to_hip(op.result.type)
        pair_t = _type_to_hip(VectorType(op.result.type.elem, 2))
        self._emit(
            f"unsigned int {packed}; "
            f"__builtin_memcpy(&{packed}, &{_name(v)}, sizeof({packed}));"
        )
        self._emit(f"{pair_t} {lo} = __builtin_amdgcn_cvt_pk_f32_bf8({packed}, false);")
        self._emit(f"{pair_t} {hi} = __builtin_amdgcn_cvt_pk_f32_bf8({packed}, true);")
        self._emit(f"{res_t} {nice};")
        self._emit(f"{nice}[0] = {lo}.x;")
        self._emit(f"{nice}[1] = {lo}.y;")
        self._emit(f"{nice}[2] = {hi}.x;")
        self._emit(f"{nice}[3] = {hi}.y;")

    def _op_arith_cvt_pk_fp8_f32x4(self, op: Op) -> None:
        """Packed <4 x f32> -> <4 x fp8e4m3> via two
        ``__builtin_amdgcn_cvt_pk_fp8_f32`` calls (mirrors the LLVM
        lowering's pack-byte0/1 then byte2/3 shape).

        The intrinsic returns an i32 with the chosen byte pair filled;
        the second call takes the first call's i32 as ``old`` and
        fills bytes 2,3. The final i32 is memcpy'd into the result
        vector. ``_type_to_hip`` maps ``VectorType(FP8E4M3, 4)`` to
        ``i8x4`` (fp8e4m3 is stored as raw bytes in the prologue),
        which is the type the HIP MFMA fp8 builtins also consume.
        """
        (v,) = op.operands
        nice = _name(op.result)
        lo = f"{nice}_lo"
        packed = f"{nice}_p"
        res_t = _type_to_hip(op.result.type)
        self._emit(
            f"unsigned int {lo} = __builtin_amdgcn_cvt_pk_fp8_f32("
            f"{_name(v)}[0], {_name(v)}[1], 0u, false);"
        )
        self._emit(
            f"unsigned int {packed} = __builtin_amdgcn_cvt_pk_fp8_f32("
            f"{_name(v)}[2], {_name(v)}[3], {lo}, true);"
        )
        self._emit(
            f"{res_t} {nice}; __builtin_memcpy(&{nice}, &{packed}, sizeof({nice}));"
        )

    def _op_arith_cvt_pk_bf8_f32x4(self, op: Op) -> None:
        """e5m2 sibling of :meth:`_op_arith_cvt_pk_fp8_f32x4`."""
        (v,) = op.operands
        nice = _name(op.result)
        lo = f"{nice}_lo"
        packed = f"{nice}_p"
        res_t = _type_to_hip(op.result.type)
        self._emit(
            f"unsigned int {lo} = __builtin_amdgcn_cvt_pk_bf8_f32("
            f"{_name(v)}[0], {_name(v)}[1], 0u, false);"
        )
        self._emit(
            f"unsigned int {packed} = __builtin_amdgcn_cvt_pk_bf8_f32("
            f"{_name(v)}[2], {_name(v)}[3], {lo}, true);"
        )
        self._emit(
            f"{res_t} {nice}; __builtin_memcpy(&{nice}, &{packed}, sizeof({nice}));"
        )

    def _op_arith_cvt_pk_i8_f32x4(self, op: Op) -> None:
        """Packed <4 x f32> -> <4 x i8> saturating cvt.

        Per-element ``rintf`` + clamp + cast loop; AMDGPU's pattern
        matcher in hipcc folds the chain into ``v_med3_i32`` plus a
        single ``v_perm_b32`` byte-select on the production path. Same
        semantics as the LLVM lowering's ``llvm.amdgcn.perm`` shape.
        """
        (v,) = op.operands
        nice = _name(op.result)
        res_t = _type_to_hip(op.result.type)
        self._emit(f"{res_t} {nice};")
        for i in range(4):
            r = f"{nice}_r{i}"
            ai = f"{nice}_i{i}"
            self._emit(f"float {r} = rintf({_name(v)}[{i}]);")
            self._emit(f"int {ai} = (int){r};")
            self._emit(f"{ai} = ({ai} < -128) ? -128 : (({ai} > 127) ? 127 : {ai});")
            self._emit(f"{nice}[{i}] = (int8_t){ai};")

    def _op_arith_cvt_f32_to_i8_sat(self, op: Op) -> None:
        # Saturating f32 -> i8 round-to-nearest-even. ``rintf`` honours
        # the current rounding mode (RNE by default), the explicit
        # clamp into [-128, 127] makes the trunc-to-i8 fully defined,
        # and the resulting sequence folds into a ``v_med3_i32`` plus a
        # ``v_cvt_pk_i16_i32`` byte select on the AMDGPU backend.
        (v,) = op.operands
        rounded = f"{_name(op.result)}_r"
        as_i32 = f"{_name(op.result)}_i"
        self._emit(f"float {rounded} = rintf({_name(v)});")
        self._emit(f"int {as_i32} = (int){rounded};")
        self._emit(
            f"int8_t {_name(op.result)} = (int8_t)({as_i32} < -128 ? -128 : "
            f"({as_i32} > 127 ? 127 : {as_i32}));"
        )

    def _op_arith_rint_f32(self, op: Op) -> None:
        (v,) = op.operands
        self._emit(f"float {_name(op.result)} = rintf({_name(v)});")

    def _op_arith_bitcast(self, op: Op) -> None:
        (v,) = op.operands
        tgt = _type_to_hip(op.result.type)
        # __builtin_bit_cast on same-sized types -> single mov in codegen.
        self._emit(
            f"{tgt} {_name(op.result)}; "
            f"__builtin_memcpy(&{_name(op.result)}, &{_name(v)}, sizeof({tgt}));"
        )

    # -------------------- arith: bitwise / int helpers --------------------

    def _op_arith_not(self, op: Op) -> None:
        # ``arith.not`` is bitwise-NOT. For i1 inputs this is logical-not.
        (v,) = op.operands
        self._emit(f"{_type_to_hip(op.result.type)} {_name(op.result)} = ~{_name(v)};")

    def _op_arith_xor(self, op: Op) -> None:
        self._binary(op, "^")

    def _op_arith_shl(self, op: Op) -> None:
        self._binary(op, "<<")

    def _op_arith_lshr(self, op: Op) -> None:
        """``a >> b`` with the unsigned-shift cast pair so HIP doesn't
        emit a signed arithmetic shift.
        """
        a, b = op.operands
        self._emit(
            f"int {_name(op.result)} = (int)((unsigned)({_name(a)}) >> {_name(b)});"
        )

    def _op_arith_umul_hi_i32(self, op: Op) -> None:
        """High 32 bits of i32 * i32 via HIP's ``__umulhi`` builtin."""
        a, b = op.operands
        self._emit(
            f"int {_name(op.result)} = "
            f"(int)__umulhi((unsigned){_name(a)}, (unsigned){_name(b)});"
        )

    # -------------------- memref: typed loads / stores / atomics --------------------

    def _op_memref_global_load_typed(self, op: Op) -> None:
        ptr, idx = op.operands
        cpp_t = _type_to_hip(op.result.type)
        self._emit(f"{cpp_t} {_name(op.result)} = {_name(ptr)}[{_name(idx)}];")

    def _op_memref_global_atomic_add(self, op: Op) -> None:
        """Lower ``global_atomic_add`` via HIP's ``atomicAdd``.

        HIP's ``atomicAdd(ptr, val)`` returns the value at the slot
        before the add -- matches our IR contract (LLVM atomicrmw add
        semantics). Supports i32 and f32; the f32 variant uses the
        gfx940+ ``global_atomic_add_f32`` instruction under the hood.
        """
        ptr, idx, val = op.operands
        cpp_t = _type_to_hip(val.type)
        self._emit(
            f"{cpp_t} {_name(op.result)} = atomicAdd(&{_name(ptr)}[{_name(idx)}], "
            f"{_name(val)});"
        )

    def _op_memref_global_atomic_add_pk_bf16(self, op: Op) -> None:
        """Lower the packed-bf16 atomic add via the AMDGPU builtin.

        Emits ``__builtin_amdgcn_global_atomic_fadd_v2bf16``, the HIP
        counterpart of LLVM's ``llvm.amdgcn.global.atomic.fadd.v2bf16``
        intrinsic. The input is a ``bf16x2`` and the result is the
        pre-add value at the slot.
        """
        ptr, idx, val = op.operands
        self._emit(
            f"bf16x2 {_name(op.result)} = "
            f"__builtin_amdgcn_global_atomic_fadd_v2bf16("
            f"{_name(ptr)} + {_name(idx)}, {_name(val)});"
        )

    def _op_tile_mfma_scale_f32_16x16x128_f8f6f4(self, op: Op) -> None:
        """HIP debug shim for P15 MX MFMA scaled.

        Hipcc 20 does not expose a builtin for the scaled MX MFMA;
        emit an inline-asm stub that documents the shape so kernel
        authors can read the HIP source to sanity-check the data
        flow. Production runs go through the LLVM path which has
        the real intrinsic.
        """
        a, b, c, a_scale, b_scale = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = {_name(c)};  // P15 MX MFMA stub:"
            f" scale_a={_name(a_scale)} scale_b={_name(b_scale)}"
            f" mantissa_a={_name(a)} mantissa_b={_name(b)}"
        )

    def _op_tile_mfma_f32_16x16x128_fp4(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = {_name(c)};  // P52 fp4 MFMA stub: "
            f"a={_name(a)} b={_name(b)}"
        )

    def _op_tile_mfma_f32_16x16x96_fp6(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = {_name(c)};  // P52 fp6 MFMA stub: "
            f"a={_name(a)} b={_name(b)}"
        )

    def _op_tile_register_p_from_qk_c(self, op: Op) -> None:
        """HIP debug shim for the P13 register permutation."""
        (qk_c,) = op.operands
        target = op.attrs["target_dtype"]
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(8):
            self._emit(f"{nice}[{i}] = ({target}){_name(qk_c)}[{i}];")

    def _op_memref_cooperative_global_store(self, op: Op) -> None:
        """HIP debug shim for P14 cooperative global store."""
        ptr, addrs, values = op.operands
        n = int(op.attrs["vec"])
        for i in range(n):
            self._emit(f"{_name(ptr)}[{_name(addrs)}[{i}]] = {_name(values)}[{i}];")

    def _op_tile_smem_store_distributed(self, op: Op) -> None:
        """HIP debug shim for P42 distributed LDS publish."""
        smem, values = op.operands
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem_store_distributed before smem_alloc was lowered")
        from ..core.ir import VectorType

        n = values.type.count if isinstance(values.type, VectorType) else 1
        for i in range(n):
            self._emit(f"{storage}[{i}] = {_name(values)}[{i}];")

    def _op_tile_lds_atomic_add(self, op: Op) -> None:
        """Lower ``lds_atomic_add`` via HIP's ``atomicAdd`` on a
        ``__shared__`` slot. The pointer is the LDS storage array
        + the rank-N linear index; ``atomicAdd`` selects the
        ``ds_add_u32`` / ``ds_pk_add_f32`` instruction at codegen.

        HIP_LOWERINGS_REPORT.md §4b: the historical implementation
        passed ``_name(smem)`` (the SSA name like ``lds_hist6``)
        rather than the actual ``__shared__`` array
        (``lds_hist6_storage`` from ``smem.op.attrs["_storage"]``);
        the resulting ``hipcc --compile-hip`` failed on
        ``moe_sort_histogram`` with ``use of undeclared identifier
        'lds_hist6'``. Every other smem op in this file routes
        through ``smem.op.attrs["_storage"]`` (see e.g.
        :meth:`_op_tile_smem_store`).
        """
        smem = op.operands[0]
        indices = list(op.operands[1:-1])
        val = op.operands[-1]
        cpp_t = _type_to_hip(val.type)
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("lds_atomic_add before smem_alloc was lowered")
        idx_expr = "][".join(_name(i) for i in indices)
        self._emit(
            f"{cpp_t} {_name(op.result)} = atomicAdd(&{storage}[{idx_expr}], "
            f"{_name(val)});"
        )

    def _op_memref_global_store_typed(self, op: Op) -> None:
        ptr, idx, val = op.operands
        self._emit(f"{_name(ptr)}[{_name(idx)}] = {_name(val)};")

    def _op_memref_global_store_vN(self, op: Op) -> None:
        ptr, idx, val = op.operands
        n = int(op.attrs["vec"])
        elem_name = op.attrs.get("elem_type", "f16")
        prefix = {"f16": "f16x", "bf16": "bf16x"}.get(elem_name, "f16x")
        self._emit(
            f"*reinterpret_cast<{prefix}{n}*>({_name(ptr)} + {_name(idx)}) = "
            f"{_name(val)};"
        )

    def _op_memref_global_atomic_add_f32(self, op: Op) -> None:
        ptr, idx, val = op.operands
        self._emit(f"atomicAdd({_name(ptr)} + {_name(idx)}, {_name(val)});")

    # -------------------- tile: LDS vector load / store --------------------

    def _op_tile_smem_load_vN(self, op: Op) -> None:
        smem = op.operands[0]
        indices = op.operands[1:]
        n = int(op.attrs["vec"])
        elem_name = op.attrs.get("elem_type", "f16")
        prefix = {"f16": "f16x", "bf16": "bf16x"}.get(elem_name, "f16x")
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem load_vN before smem_alloc was lowered")
        idx_str = "][".join(_name(i) for i in indices)
        self._emit(
            f"{prefix}{n} {_name(op.result)} = "
            f"*reinterpret_cast<const {prefix}{n}*>(&{storage}[{idx_str}]);"
        )

    def _op_tile_smem_store_vN_f32(self, op: Op) -> None:
        # The IR types the value as ``VectorType(F32, n)`` even for n=1,
        # so we always emit the ``f32xN`` vector store (the prologue
        # provides ``f32x1``). For n>1 the reinterpret-cast lets the
        # backend coalesce into ``ds_write_b{64,128}``.
        smem = op.operands[0]
        value = op.operands[-1]
        indices = op.operands[1:-1]
        n = int(op.attrs["vec"])
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem store_vN_f32 before smem_alloc was lowered")
        idx_str = "][".join(_name(i) for i in indices)
        self._emit(
            f"*reinterpret_cast<f32x{n}*>(&{storage}[{idx_str}]) = {_name(value)};"
        )

    def _op_tile_smem_load_vN_f32(self, op: Op) -> None:
        smem = op.operands[0]
        indices = op.operands[1:]
        n = int(op.attrs["vec"])
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("smem load_vN_f32 before smem_alloc was lowered")
        idx_str = "][".join(_name(i) for i in indices)
        self._emit(
            f"f32x{n} {_name(op.result)} = "
            f"*reinterpret_cast<const f32x{n}*>(&{storage}[{idx_str}]);"
        )

    # -------------------- tile: CDNA-specific primitives --------------------

    def _op_tile_buffer_rsrc(self, op: Op) -> None:
        # AMDGPU 128-bit buffer-resource descriptor over a global pointer.
        # The IR types this as ``<4 x i32>`` (a 128-bit token) but at the
        # C++ level the builtin uses the opaque ``__amdgpu_buffer_rsrc_t``
        # (aliased as ``rsrc_t`` in the prologue), which is also what the
        # ``raw_buffer_load/store`` family takes as its first argument.
        # Using the opaque type keeps the bitcast pun out of the user code.
        #
        # Signature on ROCm 7 amdclang:
        # ``__builtin_amdgcn_make_buffer_rsrc(void*, short stride,
        # int num_records, int flags)``
        #
        # The flags word is the rsrc DWORD3: it encodes
        # ``TYPE=BUFFER_RESOURCE, DATA_FORMAT=32-bit dword, NUM_FORMAT=UINT``.
        # Without it the AMDGPU compiler can lower buffer loads to
        # "unbounded" single-dword loads (no bounds check) which then
        # produce mismatching output for shapes whose OOB lanes need to
        # see zero. Value matches the LLVM-direct path
        # (``lower_llvm._op_tile_buffer_rsrc`` -> ``i32 159744`` =
        # ``0x00027000``) and CK Tile's
        # ``__builtin_amdgcn_make_buffer_rsrc(p, 0, bytes, 0x00027000)``
        # in ``cktile_fixed_lean_kernel.hpp``.
        ptr, num_bytes = op.operands
        self._emit(
            f"rsrc_t {_name(op.result)} = "
            f"__builtin_amdgcn_make_buffer_rsrc("
            f"(void*){_name(ptr)}, /*stride=*/(short)0, "
            f"/*num_records=*/(int){_name(num_bytes)}, "
            f"/*flags=*/(int)0x00027000);"
        )

    def _op_tile_async_buffer_load_lds(self, op: Op) -> None:
        # Typed-LDS variant: the second operand is a ``smem<...>`` value.
        # Materialise an LDS pointer from the ``__shared__`` storage and
        # hand it to the same intrinsic as the addr variant.
        rsrc, lds_val, voff, soff = op.operands
        dwords = int(op.attrs["dwords"])
        aux = int(op.attrs.get("aux", 0))
        size_bytes = dwords * 4
        storage = lds_val.op.attrs.get("_storage") if lds_val.op else None
        if storage is None:
            raise RuntimeError("async_buffer_load_lds before smem_alloc was lowered")
        # Same builtin-vs-intrinsic distinction as the addr variant above:
        # call the LLVM intrinsic via the prologue's ``_llvm_amdgcn_*``
        # shim so size=12 / size=16 don't trip clang's builtin validator.
        self._emit(
            f"_llvm_amdgcn_raw_ptr_buffer_load_lds("
            f"{_name(rsrc)}, "
            f"(__attribute__((address_space(3))) void*)&{storage}[0], "
            f"{size_bytes}, {_name(voff)}, {_name(soff)}, 0, {aux});"
        )

    def _op_tile_ds_bpermute(self, op: Op) -> None:
        addr, data = op.operands
        self._emit(
            f"int {_name(op.result)} = "
            f"__builtin_amdgcn_ds_bpermute({_name(addr)}, {_name(data)});"
        )

    def _op_tile_ds_bpermute_b64(self, op: Op) -> None:
        """Synthesise the 64-bit ``ds_bpermute`` from two 32-bit calls.

        Mirrors the LLVM lowering's split: take low/high 32 bits, run
        ``__builtin_amdgcn_ds_bpermute`` on each, recombine. AMDGPU
        gfx9 has no native 64-bit ``ds_bpermute``; the IR-level op
        keeps the kernel author's intent visible for future
        backends.
        """
        addr, data = op.operands
        nice = _name(op.result)
        lo = f"{nice}_lo"
        hi = f"{nice}_hi"
        plo = f"{nice}_plo"
        phi = f"{nice}_phi"
        self._emit(f"int {lo} = (int)((uint64_t){_name(data)} & 0xffffffffu);")
        self._emit(f"int {hi} = (int)((uint64_t){_name(data)} >> 32);")
        self._emit(f"int {plo} = __builtin_amdgcn_ds_bpermute({_name(addr)}, {lo});")
        self._emit(f"int {phi} = __builtin_amdgcn_ds_bpermute({_name(addr)}, {hi});")
        self._emit(
            f"int64_t {nice} = ((int64_t)(uint32_t){phi} << 32) | (uint32_t){plo};"
        )

    def _op_tile_mov_dpp(self, op: Op) -> None:
        """``v_mov_b32_dpp`` row-shift via the AMDGPU update_dpp builtin."""
        (data,) = op.operands
        bound_ctrl = bool(op.attrs.get("bound_ctrl", False))
        if "row_shr" in op.attrs:
            shift = int(op.attrs["row_shr"])
            dpp_ctrl = 0x110 | (shift & 0xF)
        else:
            shift = int(op.attrs["row_shl"])
            dpp_ctrl = 0x100 | (shift & 0xF)
        self._emit(
            f"int {_name(op.result)} = __builtin_amdgcn_update_dpp("
            f"{_name(data)}, {_name(data)}, {dpp_ctrl}, 15, 15, "
            f"{1 if bound_ctrl else 0});"
        )

    def _op_tile_dpp_xor(self, op: Op) -> None:
        """``v_mov_b32_dpp`` ``row_xmask`` XOR butterfly (VALU, not LDS).

        ``dpp_ctrl = 0x160 | mask`` (AMDGPU ``DPP_ROW_XMASK``). See
        :meth:`_op_tile_dpp_xor` in ``lower_llvm.py`` for the rationale.
        """
        (data,) = op.operands
        mask = int(op.attrs["xor_mask"])
        dpp_ctrl = 0x160 | (mask & 0xF)
        self._emit(
            f"int {_name(op.result)} = __builtin_amdgcn_update_dpp("
            f"{_name(data)}, {_name(data)}, {dpp_ctrl}, 15, 15, 1);"
        )

    def _op_tile_ds_swizzle_xor(self, op: Op) -> None:
        """``ds_swizzle_b32`` XOR butterfly via SWAP-mode encoding.

        See :meth:`_op_tile_ds_swizzle_xor` in ``lower_llvm.py`` for the
        encoding derivation. ``offset = (xor_mask << 10) | 0x1F``.
        """
        xor_mask = int(op.attrs["xor_mask"])
        offset = (xor_mask << 10) | 0x1F
        (data,) = op.operands
        self._emit(
            f"int {_name(op.result)} = "
            f"__builtin_amdgcn_ds_swizzle({_name(data)}, {offset});"
        )

    def _op_tile_permlane32_swap(self, op: Op) -> None:
        """``v_permlane32_swap_b32`` via inline asm.

        clang doesn't expose a direct builtin for the swap-style permlane;
        use inline asm. The instruction takes two i32 register operands
        and swaps their values across the wave64 32-lane halves.
        """
        lo_in, hi_in = op.operands
        r0, r1 = op.results
        self._emit(f"int {_name(r0)} = {_name(lo_in)};")
        self._emit(f"int {_name(r1)} = {_name(hi_in)};")
        self._emit(
            f'asm volatile("v_permlane32_swap_b32 %0, %1" : '
            f'"+v"({_name(r0)}), "+v"({_name(r1)}));'
        )

    def _op_tile_permlanex16(self, op: Op) -> None:
        """``v_permlanex16_b32`` swap with the ``lane ^ 16`` partner.

        Selector immediates ``0x76543210``/``0xfedcba98`` request source
        lane ``L ^ 16`` for every destination lane; ``fi=true`` reads
        across EXEC so both 16-lane halves see each other. ``old`` is a
        don't-care (all lanes overwritten) so we reuse the input.
        """
        (v,) = op.operands
        self._emit(
            f"int {_name(op.result)} = __builtin_amdgcn_permlanex16("
            f"{_name(v)}, {_name(v)}, 0x76543210u, 0xfedcba98u, false, true);"
        )

    def _op_tile_byte_perm(self, op: Op) -> None:
        """``v_perm_b32`` byte shuffle via ``__builtin_amdgcn_perm``."""
        a, b = op.operands
        sel = int(op.attrs["sel"]) & 0xFFFFFFFF
        self._emit(
            f"int {_name(op.result)} = __builtin_amdgcn_perm("
            f"{_name(a)}, {_name(b)}, {sel}u);"
        )

    def _op_tile_lane_id(self, op: Op) -> None:
        # Wave64 lane index: ``mbcnt.hi(-1, mbcnt.lo(-1, 0))``. The result
        # is a per-lane i32 in [0, 64).
        self._emit(
            f"int {_name(op.result)} = "
            f"__builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));"
        )

    def _op_tile_perm_b32(self, op: Op) -> None:
        """``v_perm_b32`` — in-lane byte select across two VGPRs (pure VALU).

        ``__builtin_amdgcn_perm(src0, src1, sel)`` takes two i32 sources and
        an i32 byte-selector and returns the permuted i32. No cross-lane, no
        LDS, no ``lgkmcnt``.
        """
        src0, src1, sel = op.operands
        self._emit(
            f"int {_name(op.result)} = "
            f"__builtin_amdgcn_perm({_name(src0)}, {_name(src1)}, {_name(sel)});"
        )

    def _op_tile_ds_read_tr16_b64(self, op: Op) -> None:
        # ``ds_read_b64_tr_b16`` -- wave64 transpose-read of a 16x16 tile.
        # AMD's HIP headers expose this as ``__builtin_amdgcn_ds_read_tr16_b64``
        # taking a ``__local`` pointer. We materialise that pointer from
        # the typed smem storage.
        self._require_ds_read_tr("ds_read_tr16_b64")
        smem = op.operands[0]
        indices = op.operands[1:]
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("ds_read_tr16_b64 before smem_alloc was lowered")
        idx_str = "][".join(_name(i) for i in indices)
        elem = op.attrs.get("elem_type", "f16")
        vec_prefix = {"f16": "f16x", "bf16": "bf16x"}.get(elem, "f16x")
        # Clang 20 does not expose ``ds.read.tr16.b64`` as a builtin -- call
        # the LLVM intrinsic through the prologue's ``_llvm_amdgcn_*`` shim
        # and bitcast the ``<4 x i16>`` raw result to the matching half /
        # bfloat vector. (Matches lower_llvm.py's emit pattern.)
        nice = _name(op.result)
        raw_tmp = f"_trraw_{nice.lstrip('%')}"
        self._emit(
            f"i16x4_raw {raw_tmp} = _llvm_amdgcn_ds_read_tr16_b64("
            f"(const __attribute__((address_space(3))) void*)&{storage}[{idx_str}]);"
        )
        self._emit(f"{vec_prefix}4 {nice}; __builtin_memcpy(&{nice}, &{raw_tmp}, 8);")

    def _op_tile_ds_read_tr16_b128(self, op: Op) -> None:
        # ``ds_read_b128_tr_b16`` -- wide gfx950 transpose-read.
        # Same shape as the b64 variant, ``<8 x i16>`` per lane.
        self._require_ds_read_tr("ds_read_tr16_b128")
        smem = op.operands[0]
        indices = op.operands[1:]
        storage = smem.op.attrs.get("_storage")
        if storage is None:
            raise RuntimeError("ds_read_tr16_b128 before smem_alloc was lowered")
        idx_str = "][".join(_name(i) for i in indices)
        elem = op.attrs.get("elem_type", "f16")
        vec_prefix = {"f16": "f16x", "bf16": "bf16x"}.get(elem, "f16x")
        nice = _name(op.result)
        raw_tmp = f"_trraw_{nice.lstrip('%')}"
        self._emit(
            f"i16x8_raw {raw_tmp} = _llvm_amdgcn_ds_read_tr16_b128("
            f"(const __attribute__((address_space(3))) void*)&{storage}[{idx_str}]);"
        )
        self._emit(f"{vec_prefix}8 {nice}; __builtin_memcpy(&{nice}, &{raw_tmp}, 16);")

    def _op_tile_mfma_f32_16x16x16_bf16(self, op: Op) -> None:
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_16x16x32_bf16(self, op: Op) -> None:
        # gfx950 K-packed bf16 atom.
        a, b, c = op.operands
        self._emit(
            f"f32x4 {_name(op.result)} = __builtin_amdgcn_mfma_f32_16x16x32_bf16("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    def _op_tile_mfma_f32_32x32x16_bf16(self, op: Op) -> None:
        # gfx950 long-prefill atom used by CK Tile/Triton.
        a, b, c = op.operands
        self._emit(
            f"f32x16 {_name(op.result)} = __builtin_amdgcn_mfma_f32_32x32x16_bf16("
            f"{_name(a)}, {_name(b)}, {_name(c)}, 0, 0, 0);"
        )

    # -------------------- vector helpers --------------------

    def _op_vector_concat(self, op: Op) -> None:
        a, b = op.operands
        res_t = _type_to_hip(op.result.type)
        n_a = a.type.count
        n_b = b.type.count
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n_a):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}];")
        for i in range(n_b):
            self._emit(f"{nice}[{n_a + i}] = {_name(b)}[{i}];")

    def _op_vector_insert(self, op: Op) -> None:
        # ``vector.insert(v, scalar, i)`` -> v with v[i] = scalar.
        v, scalar = op.operands
        i = int(op.attrs["index"])
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice} = {_name(v)};")
        self._emit(f"{nice}[{i}] = {_name(scalar)};")

    def _op_vector_pack(self, op: Op) -> None:
        # ``vector.pack`` packs N scalars into <N x elem> via insertelement chain.
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i, comp in enumerate(op.operands):
            self._emit(f"{nice}[{i}] = {_name(comp)};")

    def _op_vector_splat(self, op: Op) -> None:
        (scalar,) = op.operands
        n = int(op.attrs["vec"])
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(scalar)};")

    def _op_vector_select(self, op: Op) -> None:
        mask, lhs, rhs = op.operands
        res_t = _type_to_hip(op.result.type)
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        scalar_mask = not isinstance(mask.type, VectorType)
        for i in range(n):
            cond = _name(mask) if scalar_mask else f"{_name(mask)}[{i}]"
            self._emit(f"{nice}[{i}] = {cond} ? {_name(lhs)}[{i}] : {_name(rhs)}[{i}];")

    def _op_vector_sum(self, op: Op) -> None:
        (v,) = op.operands
        n = v.type.count if isinstance(v.type, VectorType) else 1
        elem_t = v.type.elem.name if isinstance(v.type, VectorType) else v.type.name
        cpp_t = _HIP_TYPE[elem_t]
        nice = _name(op.result)
        self._emit(f"{cpp_t} {nice} = {_name(v)}[0];")
        for i in range(1, n):
            self._emit(f"{nice} = {nice} + {_name(v)}[{i}];")

    def _op_vector_add(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}] + {_name(bb)}[{i}];")

    def _op_vector_mul(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}] * {_name(bb)}[{i}];")

    def _op_vector_sub(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}] - {_name(bb)}[{i}];")

    def _op_vector_and(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}] & {_name(bb)}[{i}];")

    def _op_vector_or(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}] | {_name(bb)}[{i}];")

    def _op_vector_shl(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}] << {_name(bb)}[{i}];")

    def _op_vector_lshr(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(
                f"{nice}[{i}] = ((uint32_t){_name(a)}[{i}]) >> {_name(bb)}[{i}];"
            )

    def _op_vector_smax(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(
                f"{nice}[{i}] = ({_name(a)}[{i}] > {_name(bb)}[{i}]) ? "
                f"{_name(a)}[{i}] : {_name(bb)}[{i}];"
            )

    def _op_vector_smin(self, op: Op) -> None:
        a, bb = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(
                f"{nice}[{i}] = ({_name(a)}[{i}] < {_name(bb)}[{i}]) ? "
                f"{_name(a)}[{i}] : {_name(bb)}[{i}];"
            )

    def _op_vector_cmp(self, op: Op) -> None:
        a, bb = op.operands
        pred = op.attrs.get("pred", "lt")
        cmap = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = {_name(a)}[{i}] {cmap[pred]} {_name(bb)}[{i}];")

    def _op_vector_trunc(self, op: Op) -> None:
        (v,) = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        elem_cpp = _type_to_hip(op.result.type.elem)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = ({elem_cpp}){_name(v)}[{i}];")

    def _op_vector_sext(self, op: Op) -> None:
        (v,) = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        elem_cpp = _type_to_hip(op.result.type.elem)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = ({elem_cpp}){_name(v)}[{i}];")

    def _op_vector_fma(self, op: Op) -> None:
        a, bb, cc = op.operands
        n = op.result.type.count if isinstance(op.result.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(
                f"{nice}[{i}] = fmaf((float){_name(a)}[{i}], "
                f"(float){_name(bb)}[{i}], (float){_name(cc)}[{i}]);"
            )

    def _op_vector_reduce_max(self, op: Op) -> None:
        (v,) = op.operands
        n = v.type.count if isinstance(v.type, VectorType) else 1
        elem_t = v.type.elem.name if isinstance(v.type, VectorType) else v.type.name
        cpp_t = _HIP_TYPE[elem_t]
        nice = _name(op.result)
        self._emit(f"{cpp_t} {nice} = {_name(v)}[0];")
        for i in range(1, n):
            self._emit(
                f"{nice} = ({_name(v)}[{i}] > {nice}) ? {_name(v)}[{i}] : {nice};"
            )

    def _op_vector_trunc_f32_to(self, op: Op) -> None:
        # ``vector.trunc_f32_to`` (target carried by attr) -- the modern
        # replacement for ``vector.trunc_f32_to_f16``. Same per-element
        # cast loop, but the target type is read from the result.
        (v,) = op.operands
        n = v.type.count if isinstance(v.type, VectorType) else 1
        res_t = _type_to_hip(op.result.type)
        elem_cpp = _HIP_TYPE[op.attrs.get("target", "f16")]
        nice = _name(op.result)
        self._emit(f"{res_t} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = ({elem_cpp}){_name(v)}[{i}];")

    # -------------------- control flow: function-level --------------------

    def _op_cf_return(self, op: Op) -> None:
        self._emit("return;")

    # -------------------- control flow --------------------

    def _op_scf_for(self, op: Op) -> None:
        num_iter = op.attrs.get("num_iter_args", 0)
        lower = op.operands[0]
        upper = op.operands[1]
        step = op.operands[2]
        iter_inits = op.operands[3 : 3 + num_iter]
        iter_meta = op.attrs.get("iter_args", [])
        iv_name = op.attrs["iv"][1:]
        iv_ty = _HIP_TYPE[op.attrs["iv_type"]]

        def cpp_type_for(type_name: str) -> str:
            if type_name.startswith("vec<f32x"):
                inner = type_name[len("vec<f32x") : -1]
                return f"f32x{int(inner)}"
            if type_name.startswith("vec<f16x"):
                inner = type_name[len("vec<f16x") : -1]
                return f"f16x{int(inner)}"
            return _HIP_TYPE.get(type_name, "auto")

        # Declare for-op results in the enclosing scope so subsequent uses see them.
        for meta, result in zip(iter_meta, op.results):
            self._emit(f"{cpp_type_for(meta['type'])} {_name(result)};")

        # Inner C++ block: iter_args, the loop, and assignment back to results.
        self._emit("{")
        self._push_indent()
        for meta, init in zip(iter_meta, iter_inits):
            self._emit(
                f"{cpp_type_for(meta['type'])} {meta['name'][1:]} = {_name(init)};"
            )
        self._emit(
            f"for({iv_ty} {iv_name} = {_name(lower)}; {iv_name} < {_name(upper)}; {iv_name} += {_name(step)}) {{"
        )
        self._push_indent()
        self.lower_region(op.regions[0])
        self._pop_indent()
        self._emit("}")
        for meta, result in zip(iter_meta, op.results):
            self._emit(f"{_name(result)} = {meta['name'][1:]};")
        self._pop_indent()
        self._emit("}")

    def _op_scf_yield(self, op: Op) -> None:
        # Map each yielded value to the corresponding iter_arg variable.
        # We find the enclosing scf.for by walking up the lowering's region
        # stack; here we just rely on positional order in the for op's
        # iter_args attr captured at parent time. To make this robust, the
        # builder records iter_args; the lowering matches them positionally.
        # In practice we assume scf.yield is the last op in the for body.
        parent_for = _find_enclosing_for(self.kernel.body, op)
        if parent_for is None:
            raise RuntimeError("scf.yield without enclosing scf.for")
        meta = parent_for.attrs.get("iter_args", [])
        if len(op.operands) != len(meta):
            raise RuntimeError(
                f"scf.yield: {len(op.operands)} values vs {len(meta)} iter_args"
            )
        for m, v in zip(meta, op.operands):
            self._emit(f"{m['name'][1:]} = {_name(v)};")

    def _op_scf_if(self, op: Op) -> None:
        (cond,) = op.operands
        self._emit(f"if({_name(cond)}) {{")
        self._push_indent()
        self.lower_region(op.regions[0])
        self._pop_indent()
        self._emit("}")


def _find_enclosing_for(region: Region, target: Op) -> Optional[Op]:
    for op in region.ops:
        if op.name == "scf.for":
            for r in op.regions:
                if target in r.ops:
                    return op
                found = _find_enclosing_for(r, target)
                if found is not None:
                    return found
        else:
            for r in op.regions:
                found = _find_enclosing_for(r, target)
                if found is not None:
                    return found
    return None


def lower_kernel_to_hip(
    kernel: KernelDef,
    *,
    launch_bounds: Optional[int] = None,
    include_prologue: bool = True,
    arch: Optional[str] = None,
) -> str:
    """Return a compilable HIP source for ``kernel``.

    The output is:
    1. The :data:`HIP_PROLOGUE` (typedefs + ``<hip/hip_runtime.h>``
    include + AMDGPU vector typedefs). Disable with
    ``include_prologue=False`` when you want only the body text
    (e.g. for embedding into a larger TU that already has these
    typedefs).
    2. The kernel's ``__global__`` signature, derived from
    :attr:`KernelDef.params`. Pointer params get ``__restrict__``;
    ``__launch_bounds__`` is taken from
    :attr:`KernelDef.max_workgroup_size` unless the caller
    overrides via ``launch_bounds``.
    3. The lowered kernel body, including any ``__shared__`` declarations
    emitted by ``tile.smem_alloc`` ops.

    Mirrors what CK Tile templates expand to after the
    instantiator runs: a single ``__global__`` function with explicit
    inline assembly / builtin calls. Useful for human inspection,
    diff-ing against a hand-written CK Tile kernel, and as a debug
    target for the rocke IR.

    ``arch`` selects the architecture seam that drives the arch-keyed
    decisions: the ``s_waitcnt`` encoding (gfx9/10 split-VMCNT vs gfx11
    contiguous), the MMA-builtin family (MFMA on CDNA, WMMA C++ builtins on
    RDNA/wave32 gfx1151), and whether ``ds_read_*_tr_*`` is available. It
    defaults to :data:`_DEFAULT_HIP_ARCH` (``gfx950``) so existing callers and
    the byte-identical CDNA baseline are preserved.
    """

    if launch_bounds is None:
        launch_bounds = kernel.max_workgroup_size

    sig_args = []
    for param in kernel.params:
        t = _type_to_hip(param.type)
        if "*" in t:
            sig_args.append(f"{t} __restrict__ {param.name}")
        else:
            sig_args.append(f"{t} {param.name}")
    signature = ", ".join(sig_args)
    # ``extern "C"`` keeps the kernel symbol name unmangled in the
    # emitted device ELF. Without it, hipcc / clang's C++ name mangler
    # produces something like ``_Z<len><name>P<arg-types>`` which the
    # run_manifest runner cannot look up (it keys off the IR's
    # ``KernelDef.name``).
    head = (
        f'extern "C" __global__ __launch_bounds__({int(launch_bounds)})\n'
        f"void {kernel.name}({signature})\n{{"
    )

    lowerer = _Lowerer(kernel, arch=arch)
    lowerer.lower_region(kernel.body)

    smem_block = "\n".join(lowerer.smem_decls)
    body_block = "\n".join(lowerer.lines)

    parts: List[str] = []
    if include_prologue:
        parts.append(HIP_PROLOGUE)
    parts.append(head)
    if smem_block:
        parts.append(smem_block)
    parts.append(body_block)
    parts.append("}")
    return "\n".join(parts)
