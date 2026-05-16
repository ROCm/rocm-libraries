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
    "i32": "int",
    "i64": "int64_t",
    "f16": "fp16",
    "bf16": "bf16",
    "f32": "float",
    "fp8e4m3": "fp8e4m3",
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
        if elem == "i8":
            return f"i8x{t.count}"
        if elem == "fp8e4m3":
            return f"i8x{t.count}"  # fp8 is stored as bytes
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
// === ck_dsl lower_hip prologue (auto-generated) ===
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
using fp8e4m3 = signed char;  // raw byte storage; converted via amdgcn intrinsics

// AMDGPU vector typedefs via Clang's ext_vector_type. Names match the
// fNxM convention used throughout the handlers below.
#define _CKDSL_VEC(elem_t, name, n) \\
    using name##n = elem_t __attribute__((ext_vector_type(n)))
_CKDSL_VEC(fp16,  f16x, 1); _CKDSL_VEC(fp16,  f16x, 2); _CKDSL_VEC(fp16,  f16x, 4);
_CKDSL_VEC(fp16,  f16x, 8); _CKDSL_VEC(fp16,  f16x, 16);
_CKDSL_VEC(bf16,  bf16x, 1); _CKDSL_VEC(bf16,  bf16x, 2); _CKDSL_VEC(bf16,  bf16x, 4);
_CKDSL_VEC(bf16,  bf16x, 8); _CKDSL_VEC(bf16,  bf16x, 16);
_CKDSL_VEC(float, f32x, 1); _CKDSL_VEC(float, f32x, 2); _CKDSL_VEC(float, f32x, 4);
_CKDSL_VEC(float, f32x, 8); _CKDSL_VEC(float, f32x, 16);
_CKDSL_VEC(int,   i32x, 1); _CKDSL_VEC(int,   i32x, 2); _CKDSL_VEC(int,   i32x, 3);
_CKDSL_VEC(int,   i32x, 4); _CKDSL_VEC(int,   i32x, 8);
_CKDSL_VEC(int8_t, i8x,  4); _CKDSL_VEC(int8_t, i8x,  8); _CKDSL_VEC(int8_t, i8x, 16);
_CKDSL_VEC(bool,  boolx, 2); _CKDSL_VEC(bool,  boolx, 4); _CKDSL_VEC(bool,  boolx, 8);
_CKDSL_VEC(bool,  boolx, 16);
#undef _CKDSL_VEC

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


class _Lowerer:
    def __init__(self, kernel: KernelDef) -> None:
        self.kernel = kernel
        self.lines: List[str] = []
        self.smem_decls: List[str] = []
        self._indent = 1
        self._smem_counter = 0

    def _emit(self, text: str) -> None:
        self.lines.append("    " * self._indent + text)

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
        elem_name = res.type.elem.name
        cpp_t = f"f{16 if elem_name == 'f16' else 32}x{count}"
        items = ", ".join(f"{fill}f" for _ in range(count))
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
        self._emit(
            f"f16x{vec} {_name(op.result)} = "
            f"*reinterpret_cast<const f16x{vec}*>({_name(ptr)} + {_name(idx)});"
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
        self._emit(
            f"*reinterpret_cast<f16x{vec}*>(&{storage}[{idx_str}]) = {_name(value)};"
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
        mask = _encode_waitcnt_gfx9_10(vmcnt=-1, expcnt=-1, lgkmcnt=0)
        self._emit(f"__builtin_amdgcn_s_waitcnt({mask});")
        self._emit("__syncthreads();")

    def _op_tile_s_waitcnt(self, op: Op) -> None:
        vm = int(op.attrs.get("vmcnt", -1))
        lk = int(op.attrs.get("lgkmcnt", -1))
        ec = int(op.attrs.get("expcnt", -1))
        mask = _encode_waitcnt_gfx9_10(vm, ec, lk)
        self._emit(f"__builtin_amdgcn_s_waitcnt({mask});")

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

    # -------------------- memref: typed loads / stores / atomics --------------------

    def _op_memref_global_load_typed(self, op: Op) -> None:
        ptr, idx = op.operands
        cpp_t = _type_to_hip(op.result.type)
        self._emit(f"{cpp_t} {_name(op.result)} = {_name(ptr)}[{_name(idx)}];")

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
        #   ``__builtin_amdgcn_make_buffer_rsrc(void*, short stride,
        #                                       int num_records, int flags)``
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

    def _op_tile_lane_id(self, op: Op) -> None:
        # Wave64 lane index: ``mbcnt.hi(-1, mbcnt.lo(-1, 0))``. The result
        # is a per-lane i32 in [0, 64).
        self._emit(
            f"int {_name(op.result)} = "
            f"__builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));"
        )

    def _op_tile_ds_read_tr16_b64(self, op: Op) -> None:
        # ``ds_read_b64_tr_b16`` -- wave64 transpose-read of a 16x16 tile.
        # AMD's HIP headers expose this as ``__builtin_amdgcn_ds_read_tr16_b64``
        # taking a ``__local`` pointer. We materialise that pointer from
        # the typed smem storage.
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
        for i in range(n):
            self._emit(
                f"{nice}[{i}] = {_name(mask)}[{i}] ? "
                f"{_name(lhs)}[{i}] : {_name(rhs)}[{i}];"
            )

    def _op_vector_sum(self, op: Op) -> None:
        (v,) = op.operands
        n = v.type.count if isinstance(v.type, VectorType) else 1
        elem_t = v.type.elem.name if isinstance(v.type, VectorType) else v.type.name
        cpp_t = _HIP_TYPE[elem_t]
        nice = _name(op.result)
        self._emit(f"{cpp_t} {nice} = {_name(v)}[0];")
        for i in range(1, n):
            self._emit(f"{nice} = {nice} + {_name(v)}[{i}];")

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
    target for the ck_dsl IR.
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

    lowerer = _Lowerer(kernel)
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
