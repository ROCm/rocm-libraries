// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_hip_mem.c (bucket 2) -- memory / LDS / buffer / async / atomics handlers
 * for the C99 port of ck_dsl.core.lower_hip.
 *
 * Faithful translation of the Python _Lowerer._op_* methods for the memory
 * family: shared-memory (LDS) alloc/store/load (scalar, vN, vN_f32,
 * distributed), global load/store (scalar, typed, vN), buffer resource
 * descriptors + raw buffer load/store, async DRAM->LDS, LDS pointer arithmetic,
 * and all atomics (global i32/f32/pk_bf16, LDS).
 *
 * Binds strictly to ckc/ir.h (frozen IR) and the shared helpers declared in
 * ckc/lower_hip_internal.h (defined in bucket 0). Shared helpers (ckc_h_emit*,
 * ckc_h_name, ckc_h_type_to_hip, ckc_h_hip_scalar, ckc_h_vec_prefix,
 * ckc_h_smem_set_storage/_storage, ckc_h_fail, ckc_h_live) are NOT defined here.
 */
#include "ckc/lower_hip_internal.h"

#include <stdint.h>
#include <string.h>

namespace ckc
{

/* ------------------------------------------------------------------ helpers */

/* Join the names of `vals[0..n)` with "][" into an arena string, matching the
 * Python `"][".join(_name(i) for i in indices)` idiom. Returns "" for n==0. */
static const char* mem_idx_join(ckc_h_lowerer_t* lw, ckc_value_t* const* vals, int n)
{
    ckc_strbuf_t sb;
    const char* out;
    int i;
    if(n <= 0)
    {
        return "";
    }
    ckc_strbuf_init(&sb, 0);
    for(i = 0; i < n; i++)
    {
        if(i > 0)
        {
            ckc_strbuf_append(&sb, "][");
        }
        ckc_strbuf_append(&sb, ckc_h_name(lw, vals[i]));
    }
    out = ckc_arena_strdup(&lw->b->arena, ckc_strbuf_cstr(&sb));
    ckc_strbuf_free(&sb);
    return out ? out : "";
}

/* Resolve the `_storage` symbol for a tile.smem_alloc result Value via the
 * lowerer side table. NULL on a miss (the Python "before smem_alloc was
 * lowered" RuntimeError case). */
static const char* mem_storage_of(ckc_h_lowerer_t* lw, const ckc_value_t* smem)
{
    return ckc_h_smem_storage(lw, smem);
}

/* Fetch an int64 attr, defaulting to `dflt` when absent (Python attrs.get). */
static int64_t mem_attr_int(const ckc_op_t* op, const char* key, int64_t dflt)
{
    int64_t v;
    if(ckc_attr_get_int(&op->attrs, key, &v))
    {
        return v;
    }
    return dflt;
}

/* Fetch a string attr, defaulting to `dflt` when absent. */
static const char* mem_attr_str(const ckc_op_t* op, const char* key, const char* dflt)
{
    const char* s = ckc_attr_get_str(&op->attrs, key);
    return s ? s : dflt;
}

/* ================================ LDS alloc =============================== */

/* Python _op_tile_smem_alloc */
static ckc_status_t _op_tile_smem_alloc(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    const ckc_value_t* res;
    const ckc_type_t* st;
    const char* elem;
    const char* nice;
    char* storage;
    ckc_strbuf_t dims;
    ckc_strbuf_t decl;
    int i;

    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_alloc: missing result");
    }
    res = op->results[0];
    st = res->type;
    if(!st || st->kind != CKC_TYPE_SMEM)
    {
        return ckc_h_fail(lw, CKC_ERR_TYPE, "tile.smem_alloc: result is not an smem type");
    }

    elem = ckc_h_hip_scalar(st->elem ? st->elem->name : "");
    if(!elem)
    {
        return ckc_h_fail(lw,
                          CKC_ERR_KEY,
                          "tile.smem_alloc: no HIP type for elem '%s'",
                          st->elem ? st->elem->name : "(null)");
    }
    nice = ckc_h_name(lw, res);

    /* dims = "][".join(str(d) for d in st.shape) */
    ckc_strbuf_init(&dims, 0);
    for(i = 0; i < st->rank; i++)
    {
        if(i > 0)
        {
            ckc_strbuf_append(&dims, "][");
        }
        ckc_strbuf_appendf(&dims, "%d", st->shape[i]);
    }

    ckc_strbuf_init(&decl, 0);
    ckc_strbuf_appendf(
        &decl, "    __shared__ %s %s_storage[%s];", elem, nice, ckc_strbuf_cstr(&dims));
    ckc_h_emit_smem_decl(lw, ckc_strbuf_cstr(&decl));
    ckc_strbuf_free(&decl);
    ckc_strbuf_free(&dims);

    /* Record "<name>_storage" on the side table (Python op.attrs["_storage"]). */
    storage = ckc_arena_printf(&lw->b->arena, "%s_storage", nice);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_OOM, "tile.smem_alloc: OOM");
    }
    ckc_h_smem_set_storage(lw, res, storage);
    return lw->status;
}

/* =============================== LDS stores ============================== */

/* Python _op_tile_smem_store */
static ckc_status_t _op_tile_smem_store(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *smem, *value;
    const char *storage, *idx_str;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_store: too few operands");
    }
    smem = op->operands[0];
    value = op->operands[op->num_operands - 1];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "smem store before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    ckc_h_emitf(lw, "%s[%s] = %s;", storage, idx_str, ckc_h_name(lw, value));
    return lw->status;
}

/* Python _op_tile_smem_store_vN */
static ckc_status_t _op_tile_smem_store_vN(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *smem, *value;
    const char *storage, *idx_str, *prefix, *elem_name;
    int64_t vec;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_store_vN: too few operands");
    }
    smem = op->operands[0];
    value = op->operands[op->num_operands - 1];
    vec = mem_attr_int(op, "vec", 0);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "smem store_vN before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = ckc_h_vec_prefix(elem_name, /*full_map=*/true);
    ckc_h_emitf(lw,
                "*reinterpret_cast<%s%lld*>(&%s[%s]) = %s;",
                prefix,
                (long long)vec,
                storage,
                idx_str,
                ckc_h_name(lw, value));
    return lw->status;
}

/* Python _op_tile_smem_store_vN_f32 */
static ckc_status_t _op_tile_smem_store_vN_f32(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *smem, *value;
    const char *storage, *idx_str;
    int64_t n;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_store_vN_f32: too few operands");
    }
    smem = op->operands[0];
    value = op->operands[op->num_operands - 1];
    n = mem_attr_int(op, "vec", 0);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "smem store_vN_f32 before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    ckc_h_emitf(lw,
                "*reinterpret_cast<f32x%lld*>(&%s[%s]) = %s;",
                (long long)n,
                storage,
                idx_str,
                ckc_h_name(lw, value));
    return lw->status;
}

/* Python _op_tile_smem_store_distributed (P42 debug shim) */
static ckc_status_t _op_tile_smem_store_distributed(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *smem, *values;
    const char* storage;
    int n, i;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_store_distributed: too few operands");
    }
    smem = op->operands[0];
    values = op->operands[1];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(
            lw, CKC_ERR_VALUE, "smem_store_distributed before smem_alloc was lowered");
    }
    n = (values->type && values->type->kind == CKC_TYPE_VECTOR) ? values->type->count : 1;
    for(i = 0; i < n; i++)
    {
        ckc_h_emitf(lw, "%s[%d] = %s[%d];", storage, i, ckc_h_name(lw, values), i);
    }
    return lw->status;
}

/* =============================== LDS loads =============================== */

/* Python _op_tile_smem_load_v4 */
static ckc_status_t _op_tile_smem_load_v4(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *smem, *row, *col;
    const char *storage, *nice;
    int i;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_load_v4: bad operand/result count");
    }
    smem = op->operands[0];
    row = op->operands[1];
    col = op->operands[2];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "smem load before smem_alloc was lowered");
    }
    nice = ckc_h_name(lw, op->results[0]);
    ckc_h_emitf(lw, "f16x4 %s;", nice);
    for(i = 0; i < 4; i++)
    {
        ckc_h_emitf(lw,
                    "%s[%d] = %s[%s][%s + %d];",
                    nice,
                    i,
                    storage,
                    ckc_h_name(lw, row),
                    ckc_h_name(lw, col),
                    i);
    }
    return lw->status;
}

/* Python _op_tile_smem_load_vN */
static ckc_status_t _op_tile_smem_load_vN(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t* smem;
    const char *storage, *idx_str, *prefix, *elem_name, *res;
    int64_t n;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 1 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_load_vN: bad operand/result count");
    }
    smem = op->operands[0];
    n = mem_attr_int(op, "vec", 0);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = ckc_h_vec_prefix(elem_name, /*full_map=*/false);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "smem load_vN before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 1);
    res = ckc_h_name(lw, op->results[0]);
    ckc_h_emitf(lw,
                "%s%lld %s = *reinterpret_cast<const %s%lld*>(&%s[%s]);",
                prefix,
                (long long)n,
                res,
                prefix,
                (long long)n,
                storage,
                idx_str);
    return lw->status;
}

/* Python _op_tile_smem_load_vN_f32 */
static ckc_status_t _op_tile_smem_load_vN_f32(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t* smem;
    const char *storage, *idx_str, *res;
    int64_t n;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 1 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_load_vN_f32: bad operand/result count");
    }
    smem = op->operands[0];
    n = mem_attr_int(op, "vec", 0);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "smem load_vN_f32 before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 1);
    res = ckc_h_name(lw, op->results[0]);
    ckc_h_emitf(lw,
                "f32x%lld %s = *reinterpret_cast<const f32x%lld*>(&%s[%s]);",
                (long long)n,
                res,
                (long long)n,
                storage,
                idx_str);
    return lw->status;
}

/* ========================= LDS pointer arithmetic ======================== */

/* Python _op_tile_smem_addr_of */
static ckc_status_t _op_tile_smem_addr_of(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t* smem;
    const char* storage;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 1 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_addr_of: bad operand/result count");
    }
    smem = op->operands[0];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "smem_addr_of before smem_alloc was lowered");
    }
    ckc_h_emitf(lw, "int64_t %s = (int64_t)(&%s[0]);", ckc_h_name(lw, op->results[0]), storage);
    return lw->status;
}

/* Python _op_tile_smem_ptr_add */
static ckc_status_t _op_tile_smem_ptr_add(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *base, *off;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.smem_ptr_add: bad operand/result count");
    }
    base = op->operands[0];
    off = op->operands[1];
    ckc_h_emitf(lw,
                "int64_t %s = %s + %s;",
                ckc_h_name(lw, op->results[0]),
                ckc_h_name(lw, base),
                ckc_h_name(lw, off));
    return lw->status;
}

/* =============================== LDS atomics ============================= */

/* Python _op_tile_lds_atomic_add */
static ckc_status_t _op_tile_lds_atomic_add(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *smem, *val;
    const char *storage, *idx_expr, *cpp_t;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.lds_atomic_add: bad operand/result count");
    }
    smem = op->operands[0];
    val = op->operands[op->num_operands - 1];
    cpp_t = ckc_h_type_to_hip(lw, val->type);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "lds_atomic_add before smem_alloc was lowered");
    }
    idx_expr = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    ckc_h_emitf(lw,
                "%s %s = atomicAdd(&%s[%s], %s);",
                cpp_t,
                ckc_h_name(lw, op->results[0]),
                storage,
                idx_expr,
                ckc_h_name(lw, val));
    return lw->status;
}

/* ============================= global load ============================== */

/* Python _op_memref_global_load */
static ckc_status_t _op_memref_global_load(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_load: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    ckc_h_emitf(lw,
                "fp16 %s = %s[%s];",
                ckc_h_name(lw, op->results[0]),
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, idx));
    return lw->status;
}

/* Python _op_memref_global_load_typed */
static ckc_status_t _op_memref_global_load_typed(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx;
    const char* cpp_t;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_load_typed: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    cpp_t = ckc_h_type_to_hip(lw, op->results[0]->type);
    ckc_h_emitf(lw,
                "%s %s = %s[%s];",
                cpp_t,
                ckc_h_name(lw, op->results[0]),
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, idx));
    return lw->status;
}

/* Python _op_memref_global_load_vN */
static ckc_status_t _op_memref_global_load_vN(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx;
    const char *prefix, *elem_name, *res;
    int64_t vec;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_load_vN: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    vec = mem_attr_int(op, "vec", 0);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = ckc_h_vec_prefix(elem_name, /*full_map=*/false);
    res = ckc_h_name(lw, op->results[0]);
    ckc_h_emitf(lw,
                "%s%lld %s = *reinterpret_cast<const %s%lld*>(%s + %s);",
                prefix,
                (long long)vec,
                res,
                prefix,
                (long long)vec,
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, idx));
    return lw->status;
}

/* ============================= global store ============================= */

/* Python _op_memref_global_store */
static ckc_status_t _op_memref_global_store(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx, *val;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_store: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    ckc_h_emitf(lw, "%s[%s] = %s;", ckc_h_name(lw, ptr), ckc_h_name(lw, idx), ckc_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_store_typed */
static ckc_status_t _op_memref_global_store_typed(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx, *val;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_store_typed: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    ckc_h_emitf(lw, "%s[%s] = %s;", ckc_h_name(lw, ptr), ckc_h_name(lw, idx), ckc_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_store_vN */
static ckc_status_t _op_memref_global_store_vN(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx, *val;
    const char *prefix, *elem_name;
    int64_t n;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_store_vN: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    n = mem_attr_int(op, "vec", 0);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = ckc_h_vec_prefix(elem_name, /*full_map=*/false);
    ckc_h_emitf(lw,
                "*reinterpret_cast<%s%lld*>(%s + %s) = %s;",
                prefix,
                (long long)n,
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, idx),
                ckc_h_name(lw, val));
    return lw->status;
}

/* ================================ atomics =============================== */

/* Python _op_memref_global_atomic_add */
static ckc_status_t _op_memref_global_atomic_add(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx, *val;
    const char* cpp_t;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_atomic_add: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    cpp_t = ckc_h_type_to_hip(lw, val->type);
    ckc_h_emitf(lw,
                "%s %s = atomicAdd(&%s[%s], %s);",
                cpp_t,
                ckc_h_name(lw, op->results[0]),
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, idx),
                ckc_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_atomic_add_f32 */
static ckc_status_t _op_memref_global_atomic_add_f32(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx, *val;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.global_atomic_add_f32: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    ckc_h_emitf(lw,
                "atomicAdd(%s + %s, %s);",
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, idx),
                ckc_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_atomic_add_pk_bf16 */
static ckc_status_t _op_memref_global_atomic_add_pk_bf16(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *idx, *val;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return ckc_h_fail(
            lw, CKC_ERR_VALUE, "memref.global_atomic_add_pk_bf16: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    ckc_h_emitf(lw,
                "bf16x2 %s = __builtin_amdgcn_global_atomic_fadd_v2bf16("
                "%s + %s, %s);",
                ckc_h_name(lw, op->results[0]),
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, idx),
                ckc_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_cooperative_global_store (P14 debug shim) */
static ckc_status_t _op_memref_cooperative_global_store(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *addrs, *values;
    int64_t n;
    int i;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "memref.cooperative_global_store: too few operands");
    }
    ptr = op->operands[0];
    addrs = op->operands[1];
    values = op->operands[2];
    n = mem_attr_int(op, "vec", 0);
    for(i = 0; i < (int)n; i++)
    {
        ckc_h_emitf(lw,
                    "%s[%s[%d]] = %s[%d];",
                    ckc_h_name(lw, ptr),
                    ckc_h_name(lw, addrs),
                    i,
                    ckc_h_name(lw, values),
                    i);
    }
    return lw->status;
}

/* ========================= global pointer arith ========================= */

/* Python _op_tile_global_ptr_add */
static ckc_status_t _op_tile_global_ptr_add(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *off;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.global_ptr_add: bad operand/result count");
    }
    ptr = op->operands[0];
    off = op->operands[1];
    ckc_h_emitf(lw,
                "const char* %s = (const char*)%s + (int64_t)%s;",
                ckc_h_name(lw, op->results[0]),
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, off));
    return lw->status;
}

/* ======================= buffer resource descriptor ===================== */

/* Python _op_tile_buffer_rsrc */
static ckc_status_t _op_tile_buffer_rsrc(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *ptr, *num_bytes;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.buffer_rsrc: bad operand/result count");
    }
    ptr = op->operands[0];
    num_bytes = op->operands[1];
    ckc_h_emitf(lw,
                "rsrc_t %s = __builtin_amdgcn_make_buffer_rsrc("
                "(void*)%s, /*stride=*/(short)0, "
                "/*num_records=*/(int)%s, "
                "/*flags=*/(int)0x00027000);",
                ckc_h_name(lw, op->results[0]),
                ckc_h_name(lw, ptr),
                ckc_h_name(lw, num_bytes));
    return lw->status;
}

/* ============================== buffer load ============================= */

/* Python _op_tile_buffer_load_f16 */
static ckc_status_t _op_tile_buffer_load_f16(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *rsrc, *voffset, *soffset;
    const char *res, *tmp;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.buffer_load_f16: bad operand/result count");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    res = ckc_h_name(lw, op->results[0]); /* without leading '%' */
    /* tmp = f"_bl_{name}" -- name already has the '%' stripped by ckc_h_name. */
    tmp = ckc_arena_printf(&lw->b->arena, "_bl_%s", res);
    ckc_h_emitf(lw,
                "unsigned int %s = (unsigned int)"
                "__builtin_amdgcn_raw_buffer_load_b32(%s, %s, %s, 0);",
                tmp,
                ckc_h_name(lw, rsrc),
                ckc_h_name(lw, voffset),
                ckc_h_name(lw, soffset));
    ckc_h_emitf(lw,
                "fp16 %s; unsigned short _u16_%s = (unsigned short)(%s & 0xFFFFu); "
                "__builtin_memcpy(&%s, &_u16_%s, 2);",
                res,
                tmp,
                tmp,
                res,
                tmp);
    return lw->status;
}

/* Python _op_tile_buffer_load_vN_f16 */
static ckc_status_t _op_tile_buffer_load_vN_f16(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *rsrc, *voffset, *soffset;
    const char *res, *b_suffix, *raw_t, *tmp;
    int64_t dwords;
    long long halves;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.buffer_load_vN_f16: bad operand/result count");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    dwords = mem_attr_int(op, "dwords", 0);
    halves = (long long)dwords * 2;
    if(dwords == 1)
    {
        b_suffix = "_b32";
    }
    else if(dwords == 2)
    {
        b_suffix = "_b64";
    }
    else if(dwords == 4)
    {
        b_suffix = "_b128";
    }
    else
    {
        return ckc_h_fail(
            lw, CKC_ERR_KEY, "tile.buffer_load_vN_f16: unsupported dwords=%lld", (long long)dwords);
    }
    if(dwords == 1)
    {
        raw_t = "int";
    }
    else
    {
        raw_t = ckc_arena_printf(&lw->b->arena, "i32x%lld", (long long)dwords);
    }
    res = ckc_h_name(lw, op->results[0]);
    tmp = ckc_arena_printf(&lw->b->arena, "_blraw_%s", res);
    ckc_h_emitf(lw,
                "%s %s = __builtin_amdgcn_raw_buffer_load%s(%s, %s, %s, 0);",
                raw_t,
                tmp,
                b_suffix,
                ckc_h_name(lw, rsrc),
                ckc_h_name(lw, voffset),
                ckc_h_name(lw, soffset));
    ckc_h_emitf(lw,
                "f16x%lld %s; __builtin_memcpy(&%s, &%s, %lld);",
                halves,
                res,
                res,
                tmp,
                (long long)dwords * 4);
    return lw->status;
}

/* Python _op_tile_buffer_load_vN (dtype-generic): raw buffer load + memcpy into
 * the <n x elem> result. f16/bf16 (2-byte): n = dwords*2; f32/i32: n = dwords. */
static ckc_status_t _op_tile_buffer_load_vN(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *rsrc, *voffset, *soffset;
    const char *res, *b_suffix, *raw_t, *tmp, *elem, *prefix;
    int64_t dwords, n;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.buffer_load_vN: bad operand/result count");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    dwords = mem_attr_int(op, "dwords", 0);
    elem = mem_attr_str(op, "elem_type", "f16");
    prefix = ckc_h_vec_prefix(elem, /*full_map=*/false);
    n = (strcmp(elem, "f16") == 0 || strcmp(elem, "bf16") == 0) ? dwords * 2 : dwords;
    if(dwords == 1)
    {
        b_suffix = "_b32";
    }
    else if(dwords == 2)
    {
        b_suffix = "_b64";
    }
    else if(dwords == 4)
    {
        b_suffix = "_b128";
    }
    else
    {
        return ckc_h_fail(
            lw, CKC_ERR_KEY, "tile.buffer_load_vN: unsupported dwords=%lld", (long long)dwords);
    }
    raw_t = (dwords == 1) ? "int" : ckc_arena_printf(&lw->b->arena, "i32x%lld", (long long)dwords);
    res = ckc_h_name(lw, op->results[0]);
    tmp = ckc_arena_printf(&lw->b->arena, "_blraw_%s", res);
    ckc_h_emitf(lw,
                "%s %s = __builtin_amdgcn_raw_buffer_load%s(%s, %s, %s, 0);",
                raw_t,
                tmp,
                b_suffix,
                ckc_h_name(lw, rsrc),
                ckc_h_name(lw, voffset),
                ckc_h_name(lw, soffset));
    ckc_h_emitf(lw,
                "%s%lld %s; __builtin_memcpy(&%s, &%s, %lld);",
                prefix,
                (long long)n,
                res,
                res,
                tmp,
                (long long)dwords * 4);
    return lw->status;
}

/* ============================== buffer store ============================ */

/* Python _op_tile_buffer_store_f16 */
static ckc_status_t _op_tile_buffer_store_f16(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *rsrc, *voffset, *soffset, *val;
    const char *vname, *tmp;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.buffer_store_f16: too few operands");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    val = op->operands[3];
    vname = ckc_h_name(lw, val);
    tmp = ckc_arena_printf(&lw->b->arena, "_u16_%s", vname);
    ckc_h_emitf(lw,
                "unsigned short %s = 0; __builtin_memcpy(&%s, &%s, 2); "
                "__builtin_amdgcn_raw_buffer_store_b16(%s, %s, %s, %s, 0);",
                tmp,
                tmp,
                vname,
                tmp,
                ckc_h_name(lw, rsrc),
                ckc_h_name(lw, voffset),
                ckc_h_name(lw, soffset));
    return lw->status;
}

/* Python _op_tile_buffer_store_vN_f16 */
static ckc_status_t _op_tile_buffer_store_vN_f16(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *rsrc, *voffset, *soffset, *val;
    const char *vname, *b_suffix, *tmp;
    int64_t dwords;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.buffer_store_vN_f16: too few operands");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    val = op->operands[3];
    dwords = mem_attr_int(op, "dwords", 0);
    vname = ckc_h_name(lw, val);
    tmp = ckc_arena_printf(&lw->b->arena, "_ub_%s", vname);
    if(dwords == 1)
    {
        ckc_h_emitf(lw,
                    "unsigned int %s = 0; __builtin_memcpy(&%s, &%s, 4); "
                    "__builtin_amdgcn_raw_buffer_store_b32(%s, %s, %s, %s, 0);",
                    tmp,
                    tmp,
                    vname,
                    tmp,
                    ckc_h_name(lw, rsrc),
                    ckc_h_name(lw, voffset),
                    ckc_h_name(lw, soffset));
    }
    else if(dwords == 2 || dwords == 4)
    {
        b_suffix = (dwords == 2) ? "_b64" : "_b128";
        ckc_h_emitf(lw,
                    "i32x%lld %s; __builtin_memcpy(&%s, &%s, %lld); "
                    "__builtin_amdgcn_raw_buffer_store%s(%s, %s, %s, %s, 0);",
                    (long long)dwords,
                    tmp,
                    tmp,
                    vname,
                    (long long)dwords * 4,
                    b_suffix,
                    tmp,
                    ckc_h_name(lw, rsrc),
                    ckc_h_name(lw, voffset),
                    ckc_h_name(lw, soffset));
    }
    else
    {
        return ckc_h_fail(lw,
                          CKC_ERR_KEY,
                          "tile.buffer_store_vN_f16: unsupported dwords=%lld",
                          (long long)dwords);
    }
    return lw->status;
}

/* ============================ async DRAM->LDS =========================== */

/* Python _op_tile_async_buffer_load_lds_addr */
static ckc_status_t _op_tile_async_buffer_load_lds_addr(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *rsrc, *lds_addr, *voff, *soff;
    int64_t dwords, size_bytes;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.async_buffer_load_lds_addr: too few operands");
    }
    rsrc = op->operands[0];
    lds_addr = op->operands[1];
    voff = op->operands[2];
    soff = op->operands[3];
    dwords = mem_attr_int(op, "dwords", 0);
    size_bytes = dwords * 4;
    ckc_h_emitf(lw,
                "_llvm_amdgcn_raw_ptr_buffer_load_lds(%s, "
                "(__attribute__((address_space(3))) void*)(%s), "
                "%lld, %s, %s, 0, 0);",
                ckc_h_name(lw, rsrc),
                ckc_h_name(lw, lds_addr),
                (long long)size_bytes,
                ckc_h_name(lw, voff),
                ckc_h_name(lw, soff));
    return lw->status;
}

/* Python _op_tile_async_buffer_load_lds (typed-LDS variant) */
static ckc_status_t _op_tile_async_buffer_load_lds(ckc_h_lowerer_t* lw, const ckc_op_t* op)
{
    ckc_value_t *rsrc, *lds_val, *voff, *soff;
    const char* storage;
    int64_t dwords, aux, size_bytes;
    if(!ckc_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "tile.async_buffer_load_lds: too few operands");
    }
    rsrc = op->operands[0];
    lds_val = op->operands[1];
    voff = op->operands[2];
    soff = op->operands[3];
    dwords = mem_attr_int(op, "dwords", 0);
    aux = mem_attr_int(op, "aux", 0);
    size_bytes = dwords * 4;
    storage = mem_storage_of(lw, lds_val);
    if(!storage)
    {
        return ckc_h_fail(lw, CKC_ERR_VALUE, "async_buffer_load_lds before smem_alloc was lowered");
    }
    ckc_h_emitf(lw,
                "_llvm_amdgcn_raw_ptr_buffer_load_lds(%s, "
                "(__attribute__((address_space(3))) void*)&%s[0], "
                "%lld, %s, %s, 0, %lld);",
                ckc_h_name(lw, rsrc),
                storage,
                (long long)size_bytes,
                ckc_h_name(lw, voff),
                ckc_h_name(lw, soff),
                (long long)aux);
    return lw->status;
}

/* ============================ registration table ======================== */

/* Bucket 2 handler table. Terminated by CKC_OP_INVALID. The Python
 * `tile.global_load_lds` op (CKC_OP_TILE_GLOBAL_LOAD_LDS) has no HIP handler in
 * lower_hip.py (it is an LLVM-path op), so it is intentionally absent here and
 * falls through to NotImplementedError parity. */
const ckc_h_handler_entry_t* ckc_h_handlers_mem(void)
{
    static const ckc_h_handler_entry_t table[]
        = {/* LDS alloc */
           {CKC_OP_TILE_SMEM_ALLOC, _op_tile_smem_alloc},
           /* LDS stores */
           {CKC_OP_TILE_SMEM_STORE, _op_tile_smem_store},
           {CKC_OP_TILE_SMEM_STORE_VN, _op_tile_smem_store_vN},
           {CKC_OP_TILE_SMEM_STORE_VN_F32, _op_tile_smem_store_vN_f32},
           {CKC_OP_TILE_SMEM_STORE_DISTRIBUTED, _op_tile_smem_store_distributed},
           /* LDS loads */
           {CKC_OP_TILE_SMEM_LOAD_V4, _op_tile_smem_load_v4},
           {CKC_OP_TILE_SMEM_LOAD_VN, _op_tile_smem_load_vN},
           {CKC_OP_TILE_SMEM_LOAD_VN_F32, _op_tile_smem_load_vN_f32},
           /* LDS pointer arithmetic */
           {CKC_OP_TILE_SMEM_ADDR_OF, _op_tile_smem_addr_of},
           {CKC_OP_TILE_SMEM_PTR_ADD, _op_tile_smem_ptr_add},
           /* LDS atomics */
           {CKC_OP_TILE_LDS_ATOMIC_ADD, _op_tile_lds_atomic_add},
           /* global load */
           {CKC_OP_MEMREF_GLOBAL_LOAD, _op_memref_global_load},
           {CKC_OP_MEMREF_GLOBAL_LOAD_TYPED, _op_memref_global_load_typed},
           {CKC_OP_MEMREF_GLOBAL_LOAD_VN, _op_memref_global_load_vN},
           /* global store */
           {CKC_OP_MEMREF_GLOBAL_STORE, _op_memref_global_store},
           {CKC_OP_MEMREF_GLOBAL_STORE_TYPED, _op_memref_global_store_typed},
           {CKC_OP_MEMREF_GLOBAL_STORE_VN, _op_memref_global_store_vN},
           /* atomics */
           {CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD, _op_memref_global_atomic_add},
           {CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32, _op_memref_global_atomic_add_f32},
           {CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16, _op_memref_global_atomic_add_pk_bf16},
           {CKC_OP_MEMREF_COOPERATIVE_GLOBAL_STORE, _op_memref_cooperative_global_store},
           /* global pointer arithmetic + buffer rsrc */
           {CKC_OP_TILE_GLOBAL_PTR_ADD, _op_tile_global_ptr_add},
           {CKC_OP_TILE_BUFFER_RSRC, _op_tile_buffer_rsrc},
           /* buffer load/store */
           {CKC_OP_TILE_BUFFER_LOAD_F16, _op_tile_buffer_load_f16},
           {CKC_OP_TILE_BUFFER_LOAD_VN_F16, _op_tile_buffer_load_vN_f16},
           {CKC_OP_TILE_BUFFER_LOAD_VN, _op_tile_buffer_load_vN},
           {CKC_OP_TILE_BUFFER_STORE_F16, _op_tile_buffer_store_f16},
           {CKC_OP_TILE_BUFFER_STORE_VN_F16, _op_tile_buffer_store_vN_f16},
           /* async DRAM->LDS */
           {CKC_OP_TILE_ASYNC_BUFFER_LOAD_LDS_ADDR, _op_tile_async_buffer_load_lds_addr},
           {CKC_OP_TILE_ASYNC_BUFFER_LOAD_LDS, _op_tile_async_buffer_load_lds},
           {CKC_OP_INVALID, NULL}};
    return table;
}

} /* namespace ckc */
