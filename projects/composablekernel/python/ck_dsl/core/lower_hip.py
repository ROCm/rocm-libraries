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
    "i32": "int",
    "i64": "int64_t",
    "f16": "fp16",
    "f32": "float",
}


def _type_to_hip(t) -> str:
    if isinstance(t, PtrType):
        if t.space == "global":
            return f"{_type_to_hip(t.pointee)}*"
        if t.space == "lds":
            return f"{_type_to_hip(t.pointee)}*"
    if isinstance(t, VectorType):
        if t.elem.name == "f16":
            return f"f16x{t.count}"
        if t.elem.name == "f32":
            return f"f32x{t.count}"
        if t.elem.name == "i32":
            return f"i32x{t.count}"
    if isinstance(t, SmemType):
        # smem allocations expand into __shared__ arrays at the top of
        # the kernel; the value at the use site is a typed pointer-ish.
        return f"{_type_to_hip(t.elem)}*"
    return _HIP_TYPE[t.name]


def _name(v: Value) -> str:
    return v.name[1:] if v.name.startswith("%") else v.name


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
        if ity == "f16":
            self._emit(f"{cpp_t} {_name(res)} = (fp16){val}f;")
        elif ity == "f32":
            self._emit(f"{cpp_t} {_name(res)} = {val}f;")
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
        (smem,) = op.operands
        self._emit(f"int64_t {_name(op.result)} = (int64_t)({_name(smem)});")

    def _op_tile_smem_ptr_add(self, op: Op) -> None:
        base, off = op.operands
        self._emit(f"int64_t {_name(op.result)} = {_name(base)} + {_name(off)};")

    def _op_tile_buffer_load_vN_f16(self, op: Op) -> None:
        rsrc, voffset, soffset = op.operands
        dwords = int(op.attrs["dwords"])
        halves = dwords * 2
        vec_ty = f"_Float16 __attribute__((ext_vector_type({halves})))"
        i32_ty = (
            f"int __attribute__((ext_vector_type({dwords})))" if dwords > 1 else "int"
        )
        tmp = f"_blraw_{_name(op.result).lstrip('%')}"
        self._emit(
            f"{i32_ty} {tmp} = __builtin_amdgcn_raw_ptr_buffer_load"
            f"{'_b64' if dwords == 2 else '_b128' if dwords == 4 else '_b32'}"
            f"({_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
        )
        self._emit(
            f"{vec_ty} {_name(op.result)}; "
            f"__builtin_memcpy(&{_name(op.result)}, &{tmp}, {dwords * 4});"
        )

    def _op_tile_buffer_load_f16(self, op: Op) -> None:
        rsrc, voffset, soffset = op.operands
        self._emit(
            f"unsigned int _bl_{_name(op.result).lstrip('%')} = "
            f"__builtin_amdgcn_raw_ptr_buffer_load_b32("
            f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
        )
        # take low 16 bits as half
        self._emit(
            f"_Float16 {_name(op.result)}; "
            f"unsigned short _u16 = (unsigned short)(_bl_{_name(op.result).lstrip('%')} & 0xFFFF); "
            f"__builtin_memcpy(&{_name(op.result)}, &_u16, 2);"
        )

    def _op_tile_buffer_store_vN_f16(self, op: Op) -> None:
        rsrc, voffset, soffset, val = op.operands
        dwords = int(op.attrs["dwords"])
        if dwords == 1:
            self._emit(
                f"unsigned int _ub = 0; "
                f"__builtin_memcpy(&_ub, &{_name(val)}, 4); "
                f"__builtin_amdgcn_raw_ptr_buffer_store_b32(_ub, "
                f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
            )
        else:
            self._emit(
                f"int __attribute__((ext_vector_type({dwords}))) _ub; "
                f"__builtin_memcpy(&_ub, &{_name(val)}, {dwords * 4}); "
                f"__builtin_amdgcn_raw_ptr_buffer_store"
                f"{'_b64' if dwords == 2 else '_b128'}(_ub, "
                f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
            )

    def _op_tile_buffer_store_f16(self, op: Op) -> None:
        rsrc, voffset, soffset, val = op.operands
        self._emit(
            f"unsigned short _u16 = 0; "
            f"__builtin_memcpy(&_u16, &{_name(val)}, 2); "
            f"__builtin_amdgcn_raw_ptr_buffer_store_b16(_u16, "
            f"{_name(rsrc)}, {_name(voffset)}, {_name(soffset)}, 0);"
        )

    def _op_tile_async_buffer_load_lds_addr(self, op: Op) -> None:
        rsrc, lds_addr, voff, soff = op.operands
        dwords = int(op.attrs["dwords"])
        size_bytes = dwords * 4
        self._emit(
            f"__builtin_amdgcn_raw_ptr_buffer_load_lds("
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
        (v,) = op.operands
        n = v.type.count if isinstance(v.type, VectorType) else 1
        nice = _name(op.result)
        self._emit(f"f16x{n} {nice};")
        for i in range(n):
            self._emit(f"{nice}[{i}] = (fp16){_name(v)}[{i}];")

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


def lower_kernel_to_hip(kernel: KernelDef, *, launch_bounds: int = 256) -> str:
    """Return the HIP source for the kernel body and its `__global__` signature."""

    sig_args = []
    for param in kernel.params:
        t = _type_to_hip(param.type)
        if "*" in t:
            sig_args.append(f"{t} __restrict__ {param.name}")
        else:
            sig_args.append(f"{t} {param.name}")
    signature = ", ".join(sig_args)
    head = (
        f"__global__ __launch_bounds__({launch_bounds})\n"
        f"void {kernel.name}({signature})\n{{"
    )

    lowerer = _Lowerer(kernel)
    lowerer.lower_region(kernel.body)

    smem_block = "\n".join(lowerer.smem_decls)
    body_block = "\n".join(lowerer.lines)

    parts = [head]
    if smem_block:
        parts.append(smem_block)
    parts.append(body_block)
    parts.append("}")
    return "\n".join(parts)
