// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_lower_llvm_mem.c.c -- BUCKET 3 of the C99 port of
 * rocke.core.lower_llvm: global memref loads/stores/atomics, smem
 * alloc/store/load, lds atomics, smem/global pointer arithmetic, buffer
 * resource descriptors + buffer load/store, async/global DRAM->LDS DMA.
 *
 * Shared plumbing (rocke_ll_emit*, rocke_ll_fresh, rocke_ll_operand, rocke_ll_need,
 * rocke_ll_llvm_type, rocke_ll_smem_*, ...) is DEFINED IN BUCKET 0; this file only
 * calls it through rocke/lower_llvm_internal.h.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm_internal.h"

namespace ckc
{

/* ---------------------------------------------------------------- helpers */

/* Result SSA name for a single-result op (Python op.result.name). */
static const char* ll_res(const rocke_op_t* op)
{
    return op->results[0]->name;
}

/* Element byte size for an smem/store/load alignment, keyed by the scalar
 * type's canonical name. Mirrors the per-handler {..}.get(name, 2) dicts. */
static int ll_elem_bytes(const char* name)
{
    if(!name)
    {
        return 2;
    }
    if(strcmp(name, "i8") == 0 || strcmp(name, "fp8e4m3") == 0 || strcmp(name, "bf8e5m2") == 0)
    {
        return 1;
    }
    if(strcmp(name, "i16") == 0 || strcmp(name, "f16") == 0 || strcmp(name, "bf16") == 0)
    {
        return 2;
    }
    if(strcmp(name, "i32") == 0 || strcmp(name, "f32") == 0)
    {
        return 4;
    }
    if(strcmp(name, "i64") == 0)
    {
        return 8;
    }
    return 2;
}

/* int attr with default (Python op.attrs.get(key, dflt)). */
static int64_t ll_attr_int(const rocke_op_t* op, const char* key, int64_t dflt)
{
    int64_t v = 0;
    if(rocke_attr_get_int(&op->attrs, key, &v))
    {
        return v;
    }
    return dflt;
}

/* str attr with default (Python op.attrs.get(key, dflt)). */
static const char* ll_attr_str(const rocke_op_t* op, const char* key, const char* dflt)
{
    const char* s = rocke_attr_get_str(&op->attrs, key);
    return s ? s : dflt;
}

/* True if a type is a vector. */
static bool ll_is_vec(const rocke_type_t* t)
{
    return t && t->kind == ROCKE_TYPE_VECTOR;
}

/* Emit the leading "i32 0, i32 <idx>, ..." gep index string used by every smem
 * gep. Returns an arena-owned string. indices come from op->operands[lo..hi). */
static const char* ll_smem_gidx(rocke_lower_t* L, const rocke_op_t* op, int lo, int hi)
{
    /* "i32 0" + ", i32 <op>" per index. Build incrementally into the arena. */
    const char* acc = "i32 0";
    int i;
    for(i = lo; i < hi; i++)
    {
        acc = rocke_arena_printf(
            &L->arena, "%s, i32 %s", acc, rocke_ll_operand(L, op->operands[i]));
    }
    return acc;
}

/* ====================================================================== */
/* memref.* global loads / stores / atomics                               */
/* ====================================================================== */

static void op_tile_smem_alloc(rocke_lower_t* L, const rocke_op_t* op)
{
    /* Module-level global emitted at finalize time; nothing inline. */
    (void)L;
    (void)op;
}

static void op_memref_global_load(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const char* gep = rocke_ll_fresh(L, "gep");
    int64_t align = ll_attr_int(op, "align", 2);
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds half, ptr addrspace(1) %s, i32 %s",
                   gep,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    rocke_ll_emitf(
        L, "  %s = load half, ptr addrspace(1) %s, align %lld", ll_res(op), gep, (long long)align);
}

static void op_memref_global_load_typed(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const char* elem_ty = rocke_ll_llvm_type(L, op->results[0]->type);
    const char* gep = rocke_ll_fresh(L, "gep");
    int64_t align = ll_attr_int(op, "align", 1);
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(1) %s, i32 %s",
                   gep,
                   elem_ty,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    rocke_ll_emitf(L,
                   "  %s = load %s, ptr addrspace(1) %s, align %lld",
                   ll_res(op),
                   elem_ty,
                   gep,
                   (long long)align);
}

static void op_memref_global_store(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const rocke_value_t* val = op->operands[2];
    const char* gep = rocke_ll_fresh(L, "gep");
    int64_t align = ll_attr_int(op, "align", 2);
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds half, ptr addrspace(1) %s, i32 %s",
                   gep,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    rocke_ll_emitf(L,
                   "  store half %s, ptr addrspace(1) %s, align %lld",
                   rocke_ll_operand(L, val),
                   gep,
                   (long long)align);
}

static void op_memref_global_store_typed(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const rocke_value_t* val = op->operands[2];
    const char* elem_ty = rocke_ll_llvm_type(L, val->type);
    const char* gep = rocke_ll_fresh(L, "gep");
    int64_t align = ll_attr_int(op, "align", 1);
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(1) %s, i32 %s",
                   gep,
                   elem_ty,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    rocke_ll_emitf(L,
                   "  store %s %s, ptr addrspace(1) %s, align %lld",
                   elem_ty,
                   rocke_ll_operand(L, val),
                   gep,
                   (long long)align);
}

static void op_memref_global_atomic_add(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const rocke_value_t* val = op->operands[2];
    const char* elem_ty = rocke_ll_llvm_type(L, val->type);
    const char* gep = rocke_ll_fresh(L, "gep");
    const char* ordering;
    bool is_f32;
    const char* rmw_op;
    const char* md = "";
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(1) %s, i32 %s",
                   gep,
                   elem_ty,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    ordering = ll_attr_str(op, "ordering", "monotonic");
    is_f32 = (val->type->name && strcmp(val->type->name, "f32") == 0);
    rmw_op = is_f32 ? "fadd" : "add";
    if(is_f32)
    {
        L->needs_fp_atomic_md = true;
        md = ", !amdgpu.no.fine.grained.memory !1"
             ", !amdgpu.no.remote.memory !1"
             ", !amdgpu.ignore.denormal.mode !1";
    }
    rocke_ll_emitf(L,
                   "  %s = atomicrmw %s ptr addrspace(1) %s, %s %s %s%s",
                   ll_res(op),
                   rmw_op,
                   gep,
                   elem_ty,
                   rocke_ll_operand(L, val),
                   ordering,
                   md);
}

static void op_memref_global_atomic_add_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const rocke_value_t* val = op->operands[2];
    const char* gep = rocke_ll_fresh(L, "gep");
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds float, ptr addrspace(1) %s, i32 %s",
                   gep,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    /* Python: tmp = self._fresh("a"); the atomicrmw result goes to this fresh
     * local, NOT to op.result (this op has no result -- has_result == false).
     * Capture the fresh name and use it (using ll_res here would deref a NULL
     * results[0] and crash). */
    const char* tmp = rocke_ll_fresh(L, "a");
    L->needs_fp_atomic_md = true;
    rocke_ll_emitf(L,
                   "  %s = atomicrmw fadd ptr addrspace(1) %s, "
                   "float %s syncscope(\"agent\") monotonic, align 4"
                   ", !amdgpu.no.fine.grained.memory !1"
                   ", !amdgpu.no.remote.memory !1"
                   ", !amdgpu.ignore.denormal.mode !1",
                   tmp,
                   gep,
                   rocke_ll_operand(L, val));
}

static void op_memref_global_atomic_add_pk_bf16(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const rocke_value_t* val = op->operands[2];
    const char* gep;
    rocke_ll_need(L, "global.atomic.fadd.v2bf16");
    gep = rocke_ll_fresh(L, "gep");
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds bfloat, ptr addrspace(1) %s, i32 %s",
                   gep,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    rocke_ll_emitf(L,
                   "  %s = call <2 x bfloat> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1("
                   "ptr addrspace(1) %s, <2 x bfloat> %s)",
                   ll_res(op),
                   gep,
                   rocke_ll_operand(L, val));
}

static void op_memref_global_load_vN(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    int64_t vec = ll_attr_int(op, "vec", 0);
    const char* elem_ty = rocke_ll_llvm_type(L, op->results[0]->type->elem);
    const char* idx_ty = rocke_ll_llvm_type(L, idx->type);
    const char* gep = rocke_ll_fresh(L, "gep");
    int64_t align;
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(1) %s, %s %s",
                   gep,
                   elem_ty,
                   rocke_ll_operand(L, ptr),
                   idx_ty,
                   rocke_ll_operand(L, idx));
    align = ll_attr_int(op, "align", vec * 2);
    rocke_ll_emitf(L,
                   "  %s = load <%lld x %s>, ptr addrspace(1) %s, align %lld",
                   ll_res(op),
                   (long long)vec,
                   elem_ty,
                   gep,
                   (long long)align);
}

static void op_memref_global_store_vN(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* idx = op->operands[1];
    const rocke_value_t* val = op->operands[2];
    int64_t vec = ll_attr_int(op, "vec", 0);
    const char* gep = rocke_ll_fresh(L, "gep");
    const char* elem_ty = ll_is_vec(val->type) ? rocke_ll_llvm_type(L, val->type->elem)
                                               : rocke_ll_llvm_type(L, val->type);
    const char* elem_name = ll_is_vec(val->type) ? val->type->elem->name : val->type->name;
    int elem_bytes = ll_elem_bytes(elem_name);
    int64_t align = vec * elem_bytes;
    const char* ty = rocke_ll_llvm_type(L, val->type);
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(1) %s, i32 %s",
                   gep,
                   elem_ty,
                   rocke_ll_operand(L, ptr),
                   rocke_ll_operand(L, idx));
    rocke_ll_emitf(L,
                   "  store %s %s, ptr addrspace(1) %s, align %lld",
                   ty,
                   rocke_ll_operand(L, val),
                   gep,
                   (long long)align);
}

static void op_memref_cooperative_global_store(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* addrs = op->operands[1];
    const rocke_value_t* values = op->operands[2];
    int64_t n = ll_attr_int(op, "vec", 0);
    const char* elem_ty;
    const char* addr_elem_ty;
    const char* addrs_ty;
    const char* values_ty;
    int64_t i;
    if(!ll_is_vec(values->type))
    {
        rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "cooperative_global_store requires vector values");
    }
    elem_ty = rocke_ll_llvm_type(L, values->type->elem);
    addr_elem_ty = ll_is_vec(addrs->type) ? rocke_ll_llvm_type(L, addrs->type->elem) : "i32";
    addrs_ty = rocke_ll_llvm_type(L, addrs->type);
    values_ty = rocke_ll_llvm_type(L, values->type);
    for(i = 0; i < n; i++)
    {
        const char* ai
            = rocke_ll_fresh(L, rocke_arena_printf(&L->arena, "coop_a%lld", (long long)i));
        const char* vi
            = rocke_ll_fresh(L, rocke_arena_printf(&L->arena, "coop_v%lld", (long long)i));
        const char* gep;
        rocke_ll_emitf(L,
                       "  %s = extractelement %s %s, i32 %lld",
                       ai,
                       addrs_ty,
                       rocke_ll_operand(L, addrs),
                       (long long)i);
        rocke_ll_emitf(L,
                       "  %s = extractelement %s %s, i32 %lld",
                       vi,
                       values_ty,
                       rocke_ll_operand(L, values),
                       (long long)i);
        gep = rocke_ll_fresh(L, rocke_arena_printf(&L->arena, "coop_gep%lld", (long long)i));
        rocke_ll_emitf(L,
                       "  %s = getelementptr inbounds %s, ptr addrspace(1) %s, %s %s",
                       gep,
                       elem_ty,
                       rocke_ll_operand(L, ptr),
                       addr_elem_ty,
                       ai);
        rocke_ll_emitf(L, "  store %s %s, ptr addrspace(1) %s, align 4", elem_ty, vi, gep);
    }
}

/* ====================================================================== */
/* tile.* smem (LDS) stores / loads / atomics                             */
/* ====================================================================== */

static void op_tile_smem_store(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    const rocke_value_t* value = op->operands[op->num_operands - 1];
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* gep = rocke_ll_fresh(L, "gep");
    const char* gidx = ll_smem_gidx(L, op, 1, op->num_operands - 1);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    int align = ll_elem_bytes(value->type->name);
    rocke_ll_emitf(
        L, "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, %s", gep, agg_ty, gname, gidx);
    rocke_ll_emitf(L,
                   "  store %s %s, ptr addrspace(3) %s, align %d",
                   rocke_ll_llvm_type(L, value->type),
                   rocke_ll_operand(L, value),
                   gep,
                   align);
}

static void op_tile_lds_atomic_add(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    const rocke_value_t* val = op->operands[op->num_operands - 1];
    const char* elem_ty = rocke_ll_llvm_type(L, val->type);
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    const char* gep = rocke_ll_fresh(L, "gep");
    const char* gidx = ll_smem_gidx(L, op, 1, op->num_operands - 1);
    const char* ordering = ll_attr_str(op, "ordering", "monotonic");
    const char* rmw_op = (val->type->name && strcmp(val->type->name, "f32") == 0) ? "fadd" : "add";
    rocke_ll_emitf(
        L, "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, %s", gep, agg_ty, gname, gidx);
    rocke_ll_emitf(L,
                   "  %s = atomicrmw %s ptr addrspace(3) %s, %s %s %s",
                   ll_res(op),
                   rmw_op,
                   gep,
                   elem_ty,
                   rocke_ll_operand(L, val),
                   ordering);
}

static void op_tile_smem_store_vN(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    const rocke_value_t* value = op->operands[op->num_operands - 1];
    int64_t vec = ll_attr_int(op, "vec", 0);
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    const char* gep = rocke_ll_fresh(L, "gep");
    const char* gidx = ll_smem_gidx(L, op, 1, op->num_operands - 1);
    const char* elem_ty = rocke_ll_llvm_type(L, value->type->elem);
    int elem_bytes = ll_elem_bytes(value->type->elem->name);
    int64_t align = ll_attr_int(op, "align", vec * elem_bytes);
    rocke_ll_emitf(
        L, "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, %s", gep, agg_ty, gname, gidx);
    rocke_ll_emitf(L,
                   "  store <%lld x %s> %s, ptr addrspace(3) %s, align %lld",
                   (long long)vec,
                   elem_ty,
                   rocke_ll_operand(L, value),
                   gep,
                   (long long)align);
}

static void op_tile_smem_load_v4(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    const rocke_value_t* row = op->operands[1];
    const rocke_value_t* col = op->operands[2];
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    const char* base = rocke_ll_fresh(L, "smem.base");
    const char* elems[4];
    const char* prev;
    int i;
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, "
                   "i32 0, i32 %s, i32 %s",
                   base,
                   agg_ty,
                   gname,
                   rocke_ll_operand(L, row),
                   rocke_ll_operand(L, col));
    for(i = 0; i < 4; i++)
    {
        const char* ep = rocke_ll_fresh(L, "smem.ep");
        const char* ld;
        rocke_ll_emitf(
            L, "  %s = getelementptr inbounds half, ptr addrspace(3) %s, i32 %d", ep, base, i);
        ld = rocke_ll_fresh(L, "smem.ld");
        rocke_ll_emitf(L, "  %s = load half, ptr addrspace(3) %s, align 2", ld, ep);
        elems[i] = ld;
    }
    prev = "undef";
    for(i = 0; i < 4; i++)
    {
        const char* tmp = (i == 3) ? ll_res(op) : rocke_ll_fresh(L, "vec");
        rocke_ll_emitf(
            L, "  %s = insertelement <4 x half> %s, half %s, i32 %d", tmp, prev, elems[i], i);
        prev = tmp;
    }
}

static void op_tile_smem_load_vN(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    int64_t vec = ll_attr_int(op, "vec", 0);
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    const char* base = rocke_ll_fresh(L, "smem.base");
    const char* idx_strs = ll_smem_gidx(L, op, 1, op->num_operands);
    const char* elem_ty = rocke_ll_llvm_type(L, op->results[0]->type->elem);
    /* Python _op_tile_smem_load_vN uses a LOCAL dict that, unlike the other
     * handlers, does NOT list fp8e4m3 / bf8e5m2 -- so they fall to the default
     * 2 (not 1). Replicate that exact dict here rather than the shared
     * ll_elem_bytes (which maps fp8/bf8 -> 1), or the fp8 down-GEMM LDS reads
     * emit `align 16` instead of the Python `align 32`.
     *   {"i8":1,"f16":2,"bf16":2,"i32":4,"f32":4,"i64":8}.get(name, 2) */
    const char* en = op->results[0]->type->elem->name;
    int elem_bytes = 2;
    if(en)
    {
        if(strcmp(en, "i8") == 0)
        {
            elem_bytes = 1;
        }
        else if(strcmp(en, "f16") == 0 || strcmp(en, "bf16") == 0)
        {
            elem_bytes = 2;
        }
        else if(strcmp(en, "i32") == 0 || strcmp(en, "f32") == 0)
        {
            elem_bytes = 4;
        }
        else if(strcmp(en, "i64") == 0)
        {
            elem_bytes = 8;
        }
    }
    int64_t align = vec * elem_bytes;
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, %s",
                   base,
                   agg_ty,
                   gname,
                   idx_strs);
    if(vec == 1)
    {
        const char* scalar = rocke_ll_fresh(L, "smem.s");
        rocke_ll_emitf(L,
                       "  %s = load %s, ptr addrspace(3) %s, align %lld",
                       scalar,
                       elem_ty,
                       base,
                       (long long)align);
        rocke_ll_emitf(L,
                       "  %s = insertelement <1 x %s> undef, %s %s, i32 0",
                       ll_res(op),
                       elem_ty,
                       elem_ty,
                       scalar);
    }
    else
    {
        rocke_ll_emitf(L,
                       "  %s = load <%lld x %s>, ptr addrspace(3) %s, align %lld",
                       ll_res(op),
                       (long long)vec,
                       elem_ty,
                       base,
                       (long long)align);
    }
}

static void op_tile_smem_store_distributed(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    const rocke_value_t* values = op->operands[1];
    int n = ll_is_vec(values->type) ? values->type->count : 1;
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    const char* elem_ty = ll_is_vec(values->type) ? rocke_ll_llvm_type(L, values->type->elem)
                                                  : rocke_ll_llvm_type(L, values->type);
    const char* values_ty = rocke_ll_llvm_type(L, values->type);
    int i;
    for(i = 0; i < n; i++)
    {
        const char* ev = rocke_ll_fresh(L, rocke_arena_printf(&L->arena, "sd_e%d", i));
        const char* gep = rocke_ll_fresh(L, rocke_arena_printf(&L->arena, "sd_gep%d", i));
        rocke_ll_emitf(L,
                       "  %s = extractelement %s %s, i32 %d",
                       ev,
                       values_ty,
                       rocke_ll_operand(L, values),
                       i);
        rocke_ll_emitf(L,
                       "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, i32 0, i32 %d",
                       gep,
                       agg_ty,
                       gname,
                       i);
        rocke_ll_emitf(L, "  store %s %s, ptr addrspace(3) %s, align 2", elem_ty, ev, gep);
    }
}

static void op_tile_smem_store_vN_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    const rocke_value_t* value = op->operands[op->num_operands - 1];
    int64_t vec = ll_attr_int(op, "vec", 0);
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    const char* gep = rocke_ll_fresh(L, "gep");
    const char* gidx = ll_smem_gidx(L, op, 1, op->num_operands - 1);
    int64_t align = vec * 4;
    rocke_ll_emitf(
        L, "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, %s", gep, agg_ty, gname, gidx);
    if(vec == 1)
    {
        if(ll_is_vec(value->type))
        {
            const char* ext = rocke_ll_fresh(L, "v1ext");
            rocke_ll_emitf(L,
                           "  %s = extractelement %s %s, i32 0",
                           ext,
                           rocke_ll_llvm_type(L, value->type),
                           rocke_ll_operand(L, value));
            rocke_ll_emitf(
                L, "  store float %s, ptr addrspace(3) %s, align %lld", ext, gep, (long long)align);
        }
        else
        {
            rocke_ll_emitf(L,
                           "  store float %s, ptr addrspace(3) %s, align %lld",
                           rocke_ll_operand(L, value),
                           gep,
                           (long long)align);
        }
    }
    else
    {
        rocke_ll_emitf(L,
                       "  store <%lld x float> %s, ptr addrspace(3) %s, align %lld",
                       (long long)vec,
                       rocke_ll_operand(L, value),
                       gep,
                       (long long)align);
    }
}

static void op_tile_smem_load_vN_f32(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    int64_t vec = ll_attr_int(op, "vec", 0);
    const rocke_type_t* stype = NULL;
    const char* gname = rocke_ll_smem_global_name(L, smem, &stype);
    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    const char* base = rocke_ll_fresh(L, "smem.base");
    const char* idx_strs = ll_smem_gidx(L, op, 1, op->num_operands);
    int64_t align = vec * 4;
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds %s, ptr addrspace(3) %s, %s",
                   base,
                   agg_ty,
                   gname,
                   idx_strs);
    if(vec == 1)
    {
        const char* scalar = rocke_ll_fresh(L, "smem.s");
        rocke_ll_emitf(L,
                       "  %s = load float, ptr addrspace(3) %s, align %lld",
                       scalar,
                       base,
                       (long long)align);
        rocke_ll_emitf(
            L, "  %s = insertelement <1 x float> undef, float %s, i32 0", ll_res(op), scalar);
    }
    else
    {
        rocke_ll_emitf(L,
                       "  %s = load <%lld x float>, ptr addrspace(3) %s, align %lld",
                       ll_res(op),
                       (long long)vec,
                       base,
                       (long long)align);
    }
}

/* ====================================================================== */
/* tile.* smem / global pointer arithmetic                                */
/* ====================================================================== */

static void op_tile_smem_addr_of(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* smem = op->operands[0];
    const char* gname = rocke_ll_smem_global_name(L, smem, NULL);
    rocke_ll_emitf(L, "  %s = ptrtoint ptr addrspace(3) %s to i64", ll_res(op), gname);
}

static void op_tile_smem_ptr_add(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* base = op->operands[0];
    const rocke_value_t* off = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = add i64 %s, %s",
                   ll_res(op),
                   rocke_ll_operand(L, base),
                   rocke_ll_operand(L, off));
}

static void op_tile_global_ptr_add(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* off = op->operands[1];
    const char* off_ty = rocke_ll_llvm_type(L, off->type);
    const char* off64;
    if(strcmp(off_ty, "i32") == 0)
    {
        off64 = rocke_ll_fresh(L, "goff64");
        rocke_ll_emitf(L, "  %s = zext i32 %s to i64", off64, rocke_ll_operand(L, off));
    }
    else if(strcmp(off_ty, "i64") == 0)
    {
        off64 = rocke_ll_operand(L, off);
    }
    else
    {
        rocke_ll_fail(
            L, ROCKE_ERR_VALUE, "tile.global_ptr_add offset must be i32 or i64, got %s", off_ty);
    }
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds i8, ptr addrspace(1) %s, i64 %s",
                   ll_res(op),
                   rocke_ll_operand(L, ptr),
                   off64);
}

/* ====================================================================== */
/* tile.* buffer resource descriptor + buffer load/store                  */
/* ====================================================================== */

static void op_tile_buffer_rsrc(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* ptr = op->operands[0];
    const rocke_value_t* num_bytes = op->operands[1];
    const char* intrinsic;
    const char* nb_text;
    int word3;
    rocke_ll_need(L, "make.buffer.rsrc.p1");
    if(L->flavor == ROCKE_LLVM_FLAVOR_LLVM22)
    {
        const char* nb_ty = rocke_ll_llvm_type(L, num_bytes->type);
        const char* nb_arg;
        intrinsic = "llvm.amdgcn.make.buffer.rsrc.p8.p1";
        if(strcmp(nb_ty, "i64") == 0)
        {
            nb_arg = rocke_ll_operand(L, num_bytes);
        }
        else if(strcmp(nb_ty, "i32") == 0)
        {
            nb_arg = rocke_ll_fresh(L, "nb64");
            rocke_ll_emitf(L, "  %s = zext i32 %s to i64", nb_arg, rocke_ll_operand(L, num_bytes));
        }
        else
        {
            rocke_ll_fail(
                L, ROCKE_ERR_VALUE, "tile.buffer_rsrc num_bytes must be i32 or i64, got %s", nb_ty);
        }
        nb_text = rocke_arena_printf(&L->arena, "i64 %s", nb_arg);
    }
    else
    {
        intrinsic = "llvm.amdgcn.make.buffer.rsrc.p1";
        nb_text = rocke_arena_printf(&L->arena, "i32 %s", rocke_ll_operand(L, num_bytes));
    }
    word3 = L->backend->buffer_rsrc_word3;
    rocke_ll_emitf(L,
                   "  %s = call ptr addrspace(8) @%s("
                   "ptr addrspace(1) %s, i16 0, %s, i32 %d)",
                   ll_res(op),
                   intrinsic,
                   rocke_ll_operand(L, ptr),
                   nb_text,
                   word3);
}

static void op_tile_buffer_load_vN_f16(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* rsrc = op->operands[0];
    const rocke_value_t* voffset = op->operands[1];
    const rocke_value_t* soffset = op->operands[2];
    int64_t dwords = ll_attr_int(op, "dwords", 0);
    if(dwords == 1)
    {
        const char* tmp;
        rocke_ll_need(L, "raw.ptr.buffer.load.i32");
        tmp = rocke_ll_fresh(L, "bli32");
        rocke_ll_emitf(L,
                       "  %s = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32("
                       "ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                       tmp,
                       rocke_ll_operand(L, rsrc),
                       rocke_ll_operand(L, voffset),
                       rocke_ll_operand(L, soffset));
        rocke_ll_emitf(L, "  %s = bitcast i32 %s to <2 x half>", ll_res(op), tmp);
    }
    else
    {
        const char* intr
            = rocke_arena_printf(&L->arena, "raw.ptr.buffer.load.v%lldi32", (long long)dwords);
        const char* tmp;
        int64_t halves = dwords * 2;
        rocke_ll_need(L, intr);
        tmp = rocke_ll_fresh(L, rocke_arena_printf(&L->arena, "blv%lld", (long long)dwords));
        rocke_ll_emitf(L,
                       "  %s = call <%lld x i32> @llvm.amdgcn.raw.ptr.buffer.load.v%lldi32("
                       "ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                       tmp,
                       (long long)dwords,
                       (long long)dwords,
                       rocke_ll_operand(L, rsrc),
                       rocke_ll_operand(L, voffset),
                       rocke_ll_operand(L, soffset));
        rocke_ll_emitf(L,
                       "  %s = bitcast <%lld x i32> %s to <%lld x half>",
                       ll_res(op),
                       (long long)dwords,
                       tmp,
                       (long long)halves);
    }
}

/* Dtype-generic vectorised buffer load (Python _op_tile_buffer_load_vN):
 * loads <dwords x i32> via the matching intrinsic and bitcasts to <n x elem>.
 * f16/bf16 (2-byte): n = dwords*2; f32/i32 (4-byte): n = dwords. */
static void op_tile_buffer_load_vN(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* rsrc = op->operands[0];
    const rocke_value_t* voffset = op->operands[1];
    const rocke_value_t* soffset = op->operands[2];
    int64_t dwords = ll_attr_int(op, "dwords", 0);
    const char* elem = ll_attr_str(op, "elem_type", "f16");
    const char* llvm_elem;
    int64_t n;
    if(strcmp(elem, "f16") == 0)
    {
        llvm_elem = "half";
        n = dwords * 2;
    }
    else if(strcmp(elem, "bf16") == 0)
    {
        llvm_elem = "bfloat";
        n = dwords * 2;
    }
    else if(strcmp(elem, "f32") == 0)
    {
        llvm_elem = "float";
        n = dwords;
    }
    else
    {
        llvm_elem = "i32";
        n = dwords;
    }
    if(dwords == 1)
    {
        const char* tmp;
        rocke_ll_need(L, "raw.ptr.buffer.load.i32");
        tmp = rocke_ll_fresh(L, "bli32");
        rocke_ll_emitf(L,
                       "  %s = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32("
                       "ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                       tmp,
                       rocke_ll_operand(L, rsrc),
                       rocke_ll_operand(L, voffset),
                       rocke_ll_operand(L, soffset));
        rocke_ll_emitf(
            L, "  %s = bitcast i32 %s to <%lld x %s>", ll_res(op), tmp, (long long)n, llvm_elem);
    }
    else
    {
        const char* intr
            = rocke_arena_printf(&L->arena, "raw.ptr.buffer.load.v%lldi32", (long long)dwords);
        const char* tmp;
        rocke_ll_need(L, intr);
        tmp = rocke_ll_fresh(L, rocke_arena_printf(&L->arena, "blv%lld", (long long)dwords));
        rocke_ll_emitf(L,
                       "  %s = call <%lld x i32> @llvm.amdgcn.raw.ptr.buffer.load.v%lldi32("
                       "ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                       tmp,
                       (long long)dwords,
                       (long long)dwords,
                       rocke_ll_operand(L, rsrc),
                       rocke_ll_operand(L, voffset),
                       rocke_ll_operand(L, soffset));
        rocke_ll_emitf(L,
                       "  %s = bitcast <%lld x i32> %s to <%lld x %s>",
                       ll_res(op),
                       (long long)dwords,
                       tmp,
                       (long long)n,
                       llvm_elem);
    }
}

static void op_tile_buffer_load_f16(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* rsrc = op->operands[0];
    const rocke_value_t* voffset = op->operands[1];
    const rocke_value_t* soffset = op->operands[2];
    const char* tmp;
    rocke_ll_need(L, "raw.ptr.buffer.load.i16");
    tmp = rocke_ll_fresh(L, "blu16");
    rocke_ll_emitf(L,
                   "  %s = call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16("
                   "ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                   tmp,
                   rocke_ll_operand(L, rsrc),
                   rocke_ll_operand(L, voffset),
                   rocke_ll_operand(L, soffset));
    rocke_ll_emitf(L, "  %s = bitcast i16 %s to half", ll_res(op), tmp);
}

static void op_tile_buffer_store_vN_f16(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* rsrc = op->operands[0];
    const rocke_value_t* voffset = op->operands[1];
    const rocke_value_t* soffset = op->operands[2];
    const rocke_value_t* val = op->operands[3];
    int64_t dwords = ll_attr_int(op, "dwords", 0);
    int64_t halves = dwords * 2;
    if(dwords == 1)
    {
        const char* bc;
        rocke_ll_need(L, "raw.ptr.buffer.store.i32");
        bc = rocke_ll_fresh(L, "bsbc");
        rocke_ll_emitf(L, "  %s = bitcast <2 x half> %s to i32", bc, rocke_ll_operand(L, val));
        rocke_ll_emitf(L,
                       "  call void @llvm.amdgcn.raw.ptr.buffer.store.i32("
                       "i32 %s, ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                       bc,
                       rocke_ll_operand(L, rsrc),
                       rocke_ll_operand(L, voffset),
                       rocke_ll_operand(L, soffset));
    }
    else
    {
        const char* intr
            = rocke_arena_printf(&L->arena, "raw.ptr.buffer.store.v%lldi32", (long long)dwords);
        const char* bc;
        rocke_ll_need(L, intr);
        bc = rocke_ll_fresh(L, "bsbc");
        rocke_ll_emitf(L,
                       "  %s = bitcast <%lld x half> %s to <%lld x i32>",
                       bc,
                       (long long)halves,
                       rocke_ll_operand(L, val),
                       (long long)dwords);
        rocke_ll_emitf(L,
                       "  call void @llvm.amdgcn.raw.ptr.buffer.store.v%lldi32("
                       "<%lld x i32> %s, ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                       (long long)dwords,
                       (long long)dwords,
                       bc,
                       rocke_ll_operand(L, rsrc),
                       rocke_ll_operand(L, voffset),
                       rocke_ll_operand(L, soffset));
    }
}

static void op_tile_buffer_store_f16(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* rsrc = op->operands[0];
    const rocke_value_t* voffset = op->operands[1];
    const rocke_value_t* soffset = op->operands[2];
    const rocke_value_t* val = op->operands[3];
    const char* bc;
    rocke_ll_need(L, "raw.ptr.buffer.store.i16");
    bc = rocke_ll_fresh(L, "bs1");
    rocke_ll_emitf(L, "  %s = bitcast half %s to i16", bc, rocke_ll_operand(L, val));
    rocke_ll_emitf(L,
                   "  call void @llvm.amdgcn.raw.ptr.buffer.store.i16("
                   "i16 %s, ptr addrspace(8) %s, i32 %s, i32 %s, i32 0)",
                   bc,
                   rocke_ll_operand(L, rsrc),
                   rocke_ll_operand(L, voffset),
                   rocke_ll_operand(L, soffset));
}

/* ====================================================================== */
/* tile.* async / global DRAM->LDS DMA                                    */
/* ====================================================================== */

static void op_tile_async_buffer_load_lds_addr(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* rsrc = op->operands[0];
    const rocke_value_t* lds_addr = op->operands[1];
    const rocke_value_t* voff = op->operands[2];
    const rocke_value_t* soff = op->operands[3];
    int64_t dwords = ll_attr_int(op, "dwords", 0);
    int64_t size_bytes = dwords * 4;
    int64_t aux = ll_attr_int(op, "aux", 0);
    const char* ptr_name;
    rocke_ll_need(L, "raw.ptr.buffer.load.lds");
    ptr_name = rocke_ll_fresh(L, "lds_ptr");
    rocke_ll_emitf(
        L, "  %s = inttoptr i64 %s to ptr addrspace(3)", ptr_name, rocke_ll_operand(L, lds_addr));
    rocke_ll_emitf(L,
                   "  call void @llvm.amdgcn.raw.ptr.buffer.load.lds("
                   "ptr addrspace(8) %s, ptr addrspace(3) %s, i32 %lld, i32 %s, i32 %s, "
                   "i32 0, i32 %lld)",
                   rocke_ll_operand(L, rsrc),
                   ptr_name,
                   (long long)size_bytes,
                   rocke_ll_operand(L, voff),
                   rocke_ll_operand(L, soff),
                   (long long)aux);
}

static void op_tile_async_buffer_load_lds(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* rsrc = op->operands[0];
    const rocke_value_t* lds_ptr = op->operands[1];
    const rocke_value_t* voffset = op->operands[2];
    const rocke_value_t* soffset = op->operands[3];
    int64_t dwords = ll_attr_int(op, "dwords", 0);
    int64_t bytes_per_lane = dwords * 4;
    int64_t aux = ll_attr_int(op, "aux", 0);
    rocke_ll_need(L, "raw.ptr.buffer.load.lds");
    rocke_ll_emitf(L,
                   "  call void @llvm.amdgcn.raw.ptr.buffer.load.lds("
                   "ptr addrspace(8) %s, ptr addrspace(3) %s, i32 %lld, i32 %s, i32 %s, "
                   "i32 0, i32 %lld)",
                   rocke_ll_operand(L, rsrc),
                   rocke_ll_operand(L, lds_ptr),
                   (long long)bytes_per_lane,
                   rocke_ll_operand(L, voffset),
                   rocke_ll_operand(L, soffset),
                   (long long)aux);
}

static void op_tile_global_load_lds(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_value_t* src_ptr = op->operands[0];
    const rocke_value_t* byte_off = op->operands[1];
    const rocke_value_t* lds_addr = op->operands[2];
    int64_t size_bytes = ll_attr_int(op, "size_bytes", 0);
    int64_t aux = ll_attr_int(op, "aux", 0);
    const char* gep;
    const char* ptr_name;
    rocke_ll_need(L, "global.load.lds");
    gep = rocke_ll_fresh(L, "gld_src");
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds i8, ptr addrspace(1) %s, i32 %s",
                   gep,
                   rocke_ll_operand(L, src_ptr),
                   rocke_ll_operand(L, byte_off));
    ptr_name = rocke_ll_fresh(L, "lds_ptr");
    rocke_ll_emitf(
        L, "  %s = inttoptr i64 %s to ptr addrspace(3)", ptr_name, rocke_ll_operand(L, lds_addr));
    rocke_ll_emitf(L,
                   "  call void @llvm.amdgcn.global.load.lds("
                   "ptr addrspace(1) %s, ptr addrspace(3) %s, i32 %lld, i32 0, i32 %lld)",
                   gep,
                   ptr_name,
                   (long long)size_bytes,
                   (long long)aux);
}

/* ====================================================================== */
/* Registration hook                                                      */
/* ====================================================================== */

void rocke_ll_register_mem(void)
{
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_ALLOC, op_tile_smem_alloc);

    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_LOAD, op_memref_global_load);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED, op_memref_global_load_typed);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, op_memref_global_load_vN);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_STORE, op_memref_global_store);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED, op_memref_global_store_typed);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_STORE_VN, op_memref_global_store_vN);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD, op_memref_global_atomic_add);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32, op_memref_global_atomic_add_f32);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16,
                         op_memref_global_atomic_add_pk_bf16);
    rocke_ll_set_handler(ROCKE_OP_MEMREF_COOPERATIVE_GLOBAL_STORE,
                         op_memref_cooperative_global_store);

    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_STORE, op_tile_smem_store);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_STORE_VN, op_tile_smem_store_vN);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_STORE_VN_F32, op_tile_smem_store_vN_f32);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_STORE_DISTRIBUTED, op_tile_smem_store_distributed);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_LOAD_V4, op_tile_smem_load_v4);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_LOAD_VN, op_tile_smem_load_vN);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_LOAD_VN_F32, op_tile_smem_load_vN_f32);
    rocke_ll_set_handler(ROCKE_OP_TILE_LDS_ATOMIC_ADD, op_tile_lds_atomic_add);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_ADDR_OF, op_tile_smem_addr_of);
    rocke_ll_set_handler(ROCKE_OP_TILE_SMEM_PTR_ADD, op_tile_smem_ptr_add);
    rocke_ll_set_handler(ROCKE_OP_TILE_GLOBAL_PTR_ADD, op_tile_global_ptr_add);

    rocke_ll_set_handler(ROCKE_OP_TILE_BUFFER_RSRC, op_tile_buffer_rsrc);
    rocke_ll_set_handler(ROCKE_OP_TILE_BUFFER_LOAD_VN_F16, op_tile_buffer_load_vN_f16);
    rocke_ll_set_handler(ROCKE_OP_TILE_BUFFER_LOAD_VN, op_tile_buffer_load_vN);
    rocke_ll_set_handler(ROCKE_OP_TILE_BUFFER_LOAD_F16, op_tile_buffer_load_f16);
    rocke_ll_set_handler(ROCKE_OP_TILE_BUFFER_STORE_VN_F16, op_tile_buffer_store_vN_f16);
    rocke_ll_set_handler(ROCKE_OP_TILE_BUFFER_STORE_F16, op_tile_buffer_store_f16);

    rocke_ll_set_handler(ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS, op_tile_async_buffer_load_lds);
    rocke_ll_set_handler(ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS_ADDR,
                         op_tile_async_buffer_load_lds_addr);
    rocke_ll_set_handler(ROCKE_OP_TILE_GLOBAL_LOAD_LDS, op_tile_global_load_lds);
}

} /* namespace ckc */
