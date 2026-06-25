// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_hip_mem.c (bucket 2) -- memory / LDS / buffer / async / atomics handlers
 * for the C99 port of rocke.core.lower_hip.
 *
 * Faithful translation of the Python _Lowerer._op_* methods for the memory
 * family: shared-memory (LDS) alloc/store/load (scalar, vN, vN_f32,
 * distributed), global load/store (scalar, typed, vN), buffer resource
 * descriptors + raw buffer load/store, async DRAM->LDS, LDS pointer arithmetic,
 * and all atomics (global i32/f32/pk_bf16, LDS).
 *
 * Binds strictly to rocke/ir.h (frozen IR) and the shared helpers declared in
 * rocke/lower_hip_internal.h (defined in bucket 0). Shared helpers (rocke_h_emit*,
 * rocke_h_name, rocke_h_type_to_hip, rocke_h_hip_scalar, rocke_h_vec_prefix,
 * rocke_h_smem_set_storage/_storage, rocke_h_fail, rocke_h_live) are NOT defined here.
 */
#include "rocke/lower_hip_internal.h"

#include <stdint.h>
#include <string.h>

namespace ckc
{

/* ------------------------------------------------------------------ helpers */

/* Join the names of `vals[0..n)` with "][" into an arena string, matching the
 * Python `"][".join(_name(i) for i in indices)` idiom. Returns "" for n==0. */
static const char* mem_idx_join(rocke_h_lowerer_t* lw, rocke_value_t* const* vals, int n)
{
    rocke_strbuf_t sb;
    const char* out;
    int i;
    if(n <= 0)
    {
        return "";
    }
    rocke_strbuf_init(&sb, 0);
    for(i = 0; i < n; i++)
    {
        if(i > 0)
        {
            rocke_strbuf_append(&sb, "][");
        }
        rocke_strbuf_append(&sb, rocke_h_name(lw, vals[i]));
    }
    out = rocke_arena_strdup(&lw->b->arena, rocke_strbuf_cstr(&sb));
    rocke_strbuf_free(&sb);
    return out ? out : "";
}

/* Resolve the `_storage` symbol for a tile.smem_alloc result Value via the
 * lowerer side table. NULL on a miss (the Python "before smem_alloc was
 * lowered" RuntimeError case). */
static const char* mem_storage_of(rocke_h_lowerer_t* lw, const rocke_value_t* smem)
{
    return rocke_h_smem_storage(lw, smem);
}

/* Fetch an int64 attr, defaulting to `dflt` when absent (Python attrs.get). */
static int64_t mem_attr_int(const rocke_op_t* op, const char* key, int64_t dflt)
{
    int64_t v;
    if(rocke_attr_get_int(&op->attrs, key, &v))
    {
        return v;
    }
    return dflt;
}

/* Fetch a string attr, defaulting to `dflt` when absent. */
static const char* mem_attr_str(const rocke_op_t* op, const char* key, const char* dflt)
{
    const char* s = rocke_attr_get_str(&op->attrs, key);
    return s ? s : dflt;
}

/* ================================ LDS alloc =============================== */

/* Python _op_tile_smem_alloc */
static rocke_status_t _op_tile_smem_alloc(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    const rocke_value_t* res;
    const rocke_type_t* st;
    const char* elem;
    const char* nice;
    char* storage;
    rocke_strbuf_t dims;
    rocke_strbuf_t decl;
    int i;

    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_alloc: missing result");
    }
    res = op->results[0];
    st = res->type;
    if(!st || st->kind != ROCKE_TYPE_SMEM)
    {
        return rocke_h_fail(lw, ROCKE_ERR_TYPE, "tile.smem_alloc: result is not an smem type");
    }

    elem = rocke_h_hip_scalar(st->elem ? st->elem->name : "");
    if(!elem)
    {
        return rocke_h_fail(lw,
                            ROCKE_ERR_KEY,
                            "tile.smem_alloc: no HIP type for elem '%s'",
                            st->elem ? st->elem->name : "(null)");
    }
    nice = rocke_h_name(lw, res);

    /* dims = "][".join(str(d) for d in st.shape) */
    rocke_strbuf_init(&dims, 0);
    for(i = 0; i < st->rank; i++)
    {
        if(i > 0)
        {
            rocke_strbuf_append(&dims, "][");
        }
        rocke_strbuf_appendf(&dims, "%d", st->shape[i]);
    }

    rocke_strbuf_init(&decl, 0);
    rocke_strbuf_appendf(
        &decl, "    __shared__ %s %s_storage[%s];", elem, nice, rocke_strbuf_cstr(&dims));
    rocke_h_emit_smem_decl(lw, rocke_strbuf_cstr(&decl));
    rocke_strbuf_free(&decl);
    rocke_strbuf_free(&dims);

    /* Record "<name>_storage" on the side table (Python op.attrs["_storage"]). */
    storage = rocke_arena_printf(&lw->b->arena, "%s_storage", nice);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_OOM, "tile.smem_alloc: OOM");
    }
    rocke_h_smem_set_storage(lw, res, storage);
    return lw->status;
}

/* =============================== LDS stores ============================== */

/* Python _op_tile_smem_store */
static rocke_status_t _op_tile_smem_store(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *smem, *value;
    const char *storage, *idx_str;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_store: too few operands");
    }
    smem = op->operands[0];
    value = op->operands[op->num_operands - 1];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "smem store before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    rocke_h_emitf(lw, "%s[%s] = %s;", storage, idx_str, rocke_h_name(lw, value));
    return lw->status;
}

/* Python _op_tile_smem_store_vN */
static rocke_status_t _op_tile_smem_store_vN(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *smem, *value;
    const char *storage, *idx_str, *prefix, *elem_name;
    int64_t vec;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_store_vN: too few operands");
    }
    smem = op->operands[0];
    value = op->operands[op->num_operands - 1];
    vec = mem_attr_int(op, "vec", 0);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "smem store_vN before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = rocke_h_vec_prefix(elem_name, /*full_map=*/true);
    rocke_h_emitf(lw,
                  "*reinterpret_cast<%s%lld*>(&%s[%s]) = %s;",
                  prefix,
                  (long long)vec,
                  storage,
                  idx_str,
                  rocke_h_name(lw, value));
    return lw->status;
}

/* Python _op_tile_smem_store_vN_f32 */
static rocke_status_t _op_tile_smem_store_vN_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *smem, *value;
    const char *storage, *idx_str;
    int64_t n;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_store_vN_f32: too few operands");
    }
    smem = op->operands[0];
    value = op->operands[op->num_operands - 1];
    n = mem_attr_int(op, "vec", 0);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "smem store_vN_f32 before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    rocke_h_emitf(lw,
                  "*reinterpret_cast<f32x%lld*>(&%s[%s]) = %s;",
                  (long long)n,
                  storage,
                  idx_str,
                  rocke_h_name(lw, value));
    return lw->status;
}

/* Python _op_tile_smem_store_distributed (P42 debug shim) */
static rocke_status_t _op_tile_smem_store_distributed(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *smem, *values;
    const char* storage;
    int n, i;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_store_distributed: too few operands");
    }
    smem = op->operands[0];
    values = op->operands[1];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "smem_store_distributed before smem_alloc was lowered");
    }
    n = (values->type && values->type->kind == ROCKE_TYPE_VECTOR) ? values->type->count : 1;
    for(i = 0; i < n; i++)
    {
        rocke_h_emitf(lw, "%s[%d] = %s[%d];", storage, i, rocke_h_name(lw, values), i);
    }
    return lw->status;
}

/* =============================== LDS loads =============================== */

/* Python _op_tile_smem_load_v4 */
static rocke_status_t _op_tile_smem_load_v4(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *smem, *row, *col;
    const char *storage, *nice;
    int i;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_load_v4: bad operand/result count");
    }
    smem = op->operands[0];
    row = op->operands[1];
    col = op->operands[2];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "smem load before smem_alloc was lowered");
    }
    nice = rocke_h_name(lw, op->results[0]);
    rocke_h_emitf(lw, "f16x4 %s;", nice);
    for(i = 0; i < 4; i++)
    {
        rocke_h_emitf(lw,
                      "%s[%d] = %s[%s][%s + %d];",
                      nice,
                      i,
                      storage,
                      rocke_h_name(lw, row),
                      rocke_h_name(lw, col),
                      i);
    }
    return lw->status;
}

/* Python _op_tile_smem_load_vN */
static rocke_status_t _op_tile_smem_load_vN(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t* smem;
    const char *storage, *idx_str, *prefix, *elem_name, *res;
    int64_t n;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 1 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_load_vN: bad operand/result count");
    }
    smem = op->operands[0];
    n = mem_attr_int(op, "vec", 0);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = rocke_h_vec_prefix(elem_name, /*full_map=*/false);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "smem load_vN before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 1);
    res = rocke_h_name(lw, op->results[0]);
    rocke_h_emitf(lw,
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
static rocke_status_t _op_tile_smem_load_vN_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t* smem;
    const char *storage, *idx_str, *res;
    int64_t n;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 1 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_load_vN_f32: bad operand/result count");
    }
    smem = op->operands[0];
    n = mem_attr_int(op, "vec", 0);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "smem load_vN_f32 before smem_alloc was lowered");
    }
    idx_str = mem_idx_join(lw, &op->operands[1], op->num_operands - 1);
    res = rocke_h_name(lw, op->results[0]);
    rocke_h_emitf(lw,
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
static rocke_status_t _op_tile_smem_addr_of(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t* smem;
    const char* storage;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 1 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_addr_of: bad operand/result count");
    }
    smem = op->operands[0];
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "smem_addr_of before smem_alloc was lowered");
    }
    rocke_h_emitf(lw, "int64_t %s = (int64_t)(&%s[0]);", rocke_h_name(lw, op->results[0]), storage);
    return lw->status;
}

/* Python _op_tile_smem_ptr_add */
static rocke_status_t _op_tile_smem_ptr_add(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *base, *off;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.smem_ptr_add: bad operand/result count");
    }
    base = op->operands[0];
    off = op->operands[1];
    rocke_h_emitf(lw,
                  "int64_t %s = %s + %s;",
                  rocke_h_name(lw, op->results[0]),
                  rocke_h_name(lw, base),
                  rocke_h_name(lw, off));
    return lw->status;
}

/* =============================== LDS atomics ============================= */

/* Python _op_tile_lds_atomic_add */
static rocke_status_t _op_tile_lds_atomic_add(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *smem, *val;
    const char *storage, *idx_expr, *cpp_t;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.lds_atomic_add: bad operand/result count");
    }
    smem = op->operands[0];
    val = op->operands[op->num_operands - 1];
    cpp_t = rocke_h_type_to_hip(lw, val->type);
    storage = mem_storage_of(lw, smem);
    if(!storage)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "lds_atomic_add before smem_alloc was lowered");
    }
    idx_expr = mem_idx_join(lw, &op->operands[1], op->num_operands - 2);
    rocke_h_emitf(lw,
                  "%s %s = atomicAdd(&%s[%s], %s);",
                  cpp_t,
                  rocke_h_name(lw, op->results[0]),
                  storage,
                  idx_expr,
                  rocke_h_name(lw, val));
    return lw->status;
}

/* ============================= global load ============================== */

/* Python _op_memref_global_load */
static rocke_status_t _op_memref_global_load(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "memref.global_load: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    rocke_h_emitf(lw,
                  "fp16 %s = %s[%s];",
                  rocke_h_name(lw, op->results[0]),
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, idx));
    return lw->status;
}

/* Python _op_memref_global_load_typed */
static rocke_status_t _op_memref_global_load_typed(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx;
    const char* cpp_t;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "memref.global_load_typed: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    cpp_t = rocke_h_type_to_hip(lw, op->results[0]->type);
    rocke_h_emitf(lw,
                  "%s %s = %s[%s];",
                  cpp_t,
                  rocke_h_name(lw, op->results[0]),
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, idx));
    return lw->status;
}

/* Python _op_memref_global_load_vN */
static rocke_status_t _op_memref_global_load_vN(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx;
    const char *prefix, *elem_name, *res;
    int64_t vec;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "memref.global_load_vN: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    vec = mem_attr_int(op, "vec", 0);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = rocke_h_vec_prefix(elem_name, /*full_map=*/false);
    res = rocke_h_name(lw, op->results[0]);
    rocke_h_emitf(lw,
                  "%s%lld %s = *reinterpret_cast<const %s%lld*>(%s + %s);",
                  prefix,
                  (long long)vec,
                  res,
                  prefix,
                  (long long)vec,
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, idx));
    return lw->status;
}

/* ============================= global store ============================= */

/* Python _op_memref_global_store */
static rocke_status_t _op_memref_global_store(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx, *val;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "memref.global_store: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    rocke_h_emitf(
        lw, "%s[%s] = %s;", rocke_h_name(lw, ptr), rocke_h_name(lw, idx), rocke_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_store_typed */
static rocke_status_t _op_memref_global_store_typed(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx, *val;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "memref.global_store_typed: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    rocke_h_emitf(
        lw, "%s[%s] = %s;", rocke_h_name(lw, ptr), rocke_h_name(lw, idx), rocke_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_store_vN */
static rocke_status_t _op_memref_global_store_vN(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx, *val;
    const char *prefix, *elem_name;
    int64_t n;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "memref.global_store_vN: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    n = mem_attr_int(op, "vec", 0);
    elem_name = mem_attr_str(op, "elem_type", "f16");
    prefix = rocke_h_vec_prefix(elem_name, /*full_map=*/false);
    rocke_h_emitf(lw,
                  "*reinterpret_cast<%s%lld*>(%s + %s) = %s;",
                  prefix,
                  (long long)n,
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, idx),
                  rocke_h_name(lw, val));
    return lw->status;
}

/* ================================ atomics =============================== */

/* Python _op_memref_global_atomic_add */
static rocke_status_t _op_memref_global_atomic_add(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx, *val;
    const char* cpp_t;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "memref.global_atomic_add: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    cpp_t = rocke_h_type_to_hip(lw, val->type);
    rocke_h_emitf(lw,
                  "%s %s = atomicAdd(&%s[%s], %s);",
                  cpp_t,
                  rocke_h_name(lw, op->results[0]),
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, idx),
                  rocke_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_atomic_add_f32 */
static rocke_status_t _op_memref_global_atomic_add_f32(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx, *val;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "memref.global_atomic_add_f32: too few operands");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    rocke_h_emitf(lw,
                  "atomicAdd(%s + %s, %s);",
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, idx),
                  rocke_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_global_atomic_add_pk_bf16 */
static rocke_status_t _op_memref_global_atomic_add_pk_bf16(rocke_h_lowerer_t* lw,
                                                           const rocke_op_t* op)
{
    rocke_value_t *ptr, *idx, *val;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "memref.global_atomic_add_pk_bf16: bad operand/result count");
    }
    ptr = op->operands[0];
    idx = op->operands[1];
    val = op->operands[2];
    rocke_h_emitf(lw,
                  "bf16x2 %s = __builtin_amdgcn_global_atomic_fadd_v2bf16("
                  "%s + %s, %s);",
                  rocke_h_name(lw, op->results[0]),
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, idx),
                  rocke_h_name(lw, val));
    return lw->status;
}

/* Python _op_memref_cooperative_global_store (P14 debug shim) */
static rocke_status_t _op_memref_cooperative_global_store(rocke_h_lowerer_t* lw,
                                                          const rocke_op_t* op)
{
    rocke_value_t *ptr, *addrs, *values;
    int64_t n;
    int i;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "memref.cooperative_global_store: too few operands");
    }
    ptr = op->operands[0];
    addrs = op->operands[1];
    values = op->operands[2];
    n = mem_attr_int(op, "vec", 0);
    for(i = 0; i < (int)n; i++)
    {
        rocke_h_emitf(lw,
                      "%s[%s[%d]] = %s[%d];",
                      rocke_h_name(lw, ptr),
                      rocke_h_name(lw, addrs),
                      i,
                      rocke_h_name(lw, values),
                      i);
    }
    return lw->status;
}

/* ========================= global pointer arith ========================= */

/* Python _op_tile_global_ptr_add */
static rocke_status_t _op_tile_global_ptr_add(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *off;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.global_ptr_add: bad operand/result count");
    }
    ptr = op->operands[0];
    off = op->operands[1];
    rocke_h_emitf(lw,
                  "const char* %s = (const char*)%s + (int64_t)%s;",
                  rocke_h_name(lw, op->results[0]),
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, off));
    return lw->status;
}

/* ======================= buffer resource descriptor ===================== */

/* Python _op_tile_buffer_rsrc */
static rocke_status_t _op_tile_buffer_rsrc(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *ptr, *num_bytes;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 2 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.buffer_rsrc: bad operand/result count");
    }
    ptr = op->operands[0];
    num_bytes = op->operands[1];
    rocke_h_emitf(lw,
                  "rsrc_t %s = __builtin_amdgcn_make_buffer_rsrc("
                  "(void*)%s, /*stride=*/(short)0, "
                  "/*num_records=*/(int)%s, "
                  "/*flags=*/(int)0x00027000);",
                  rocke_h_name(lw, op->results[0]),
                  rocke_h_name(lw, ptr),
                  rocke_h_name(lw, num_bytes));
    return lw->status;
}

/* ============================== buffer load ============================= */

/* Python _op_tile_buffer_load_f16 */
static rocke_status_t _op_tile_buffer_load_f16(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *rsrc, *voffset, *soffset;
    const char *res, *tmp;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.buffer_load_f16: bad operand/result count");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    res = rocke_h_name(lw, op->results[0]); /* without leading '%' */
    /* tmp = f"_bl_{name}" -- name already has the '%' stripped by rocke_h_name. */
    tmp = rocke_arena_printf(&lw->b->arena, "_bl_%s", res);
    rocke_h_emitf(lw,
                  "unsigned int %s = (unsigned int)"
                  "__builtin_amdgcn_raw_buffer_load_b32(%s, %s, %s, 0);",
                  tmp,
                  rocke_h_name(lw, rsrc),
                  rocke_h_name(lw, voffset),
                  rocke_h_name(lw, soffset));
    rocke_h_emitf(lw,
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
static rocke_status_t _op_tile_buffer_load_vN_f16(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *rsrc, *voffset, *soffset;
    const char *res, *b_suffix, *raw_t, *tmp;
    int64_t dwords;
    long long halves;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "tile.buffer_load_vN_f16: bad operand/result count");
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
        return rocke_h_fail(lw,
                            ROCKE_ERR_KEY,
                            "tile.buffer_load_vN_f16: unsupported dwords=%lld",
                            (long long)dwords);
    }
    if(dwords == 1)
    {
        raw_t = "int";
    }
    else
    {
        raw_t = rocke_arena_printf(&lw->b->arena, "i32x%lld", (long long)dwords);
    }
    res = rocke_h_name(lw, op->results[0]);
    tmp = rocke_arena_printf(&lw->b->arena, "_blraw_%s", res);
    rocke_h_emitf(lw,
                  "%s %s = __builtin_amdgcn_raw_buffer_load%s(%s, %s, %s, 0);",
                  raw_t,
                  tmp,
                  b_suffix,
                  rocke_h_name(lw, rsrc),
                  rocke_h_name(lw, voffset),
                  rocke_h_name(lw, soffset));
    rocke_h_emitf(lw,
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
static rocke_status_t _op_tile_buffer_load_vN(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *rsrc, *voffset, *soffset;
    const char *res, *b_suffix, *raw_t, *tmp, *elem, *prefix;
    int64_t dwords, n;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 3 || op->num_results < 1)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.buffer_load_vN: bad operand/result count");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    dwords = mem_attr_int(op, "dwords", 0);
    elem = mem_attr_str(op, "elem_type", "f16");
    prefix = rocke_h_vec_prefix(elem, /*full_map=*/false);
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
        return rocke_h_fail(
            lw, ROCKE_ERR_KEY, "tile.buffer_load_vN: unsupported dwords=%lld", (long long)dwords);
    }
    raw_t
        = (dwords == 1) ? "int" : rocke_arena_printf(&lw->b->arena, "i32x%lld", (long long)dwords);
    res = rocke_h_name(lw, op->results[0]);
    tmp = rocke_arena_printf(&lw->b->arena, "_blraw_%s", res);
    rocke_h_emitf(lw,
                  "%s %s = __builtin_amdgcn_raw_buffer_load%s(%s, %s, %s, 0);",
                  raw_t,
                  tmp,
                  b_suffix,
                  rocke_h_name(lw, rsrc),
                  rocke_h_name(lw, voffset),
                  rocke_h_name(lw, soffset));
    rocke_h_emitf(lw,
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
static rocke_status_t _op_tile_buffer_store_f16(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *rsrc, *voffset, *soffset, *val;
    const char *vname, *tmp;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.buffer_store_f16: too few operands");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    val = op->operands[3];
    vname = rocke_h_name(lw, val);
    tmp = rocke_arena_printf(&lw->b->arena, "_u16_%s", vname);
    rocke_h_emitf(lw,
                  "unsigned short %s = 0; __builtin_memcpy(&%s, &%s, 2); "
                  "__builtin_amdgcn_raw_buffer_store_b16(%s, %s, %s, %s, 0);",
                  tmp,
                  tmp,
                  vname,
                  tmp,
                  rocke_h_name(lw, rsrc),
                  rocke_h_name(lw, voffset),
                  rocke_h_name(lw, soffset));
    return lw->status;
}

/* Python _op_tile_buffer_store_vN_f16 */
static rocke_status_t _op_tile_buffer_store_vN_f16(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *rsrc, *voffset, *soffset, *val;
    const char *vname, *b_suffix, *tmp;
    int64_t dwords;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.buffer_store_vN_f16: too few operands");
    }
    rsrc = op->operands[0];
    voffset = op->operands[1];
    soffset = op->operands[2];
    val = op->operands[3];
    dwords = mem_attr_int(op, "dwords", 0);
    vname = rocke_h_name(lw, val);
    tmp = rocke_arena_printf(&lw->b->arena, "_ub_%s", vname);
    if(dwords == 1)
    {
        rocke_h_emitf(lw,
                      "unsigned int %s = 0; __builtin_memcpy(&%s, &%s, 4); "
                      "__builtin_amdgcn_raw_buffer_store_b32(%s, %s, %s, %s, 0);",
                      tmp,
                      tmp,
                      vname,
                      tmp,
                      rocke_h_name(lw, rsrc),
                      rocke_h_name(lw, voffset),
                      rocke_h_name(lw, soffset));
    }
    else if(dwords == 2 || dwords == 4)
    {
        b_suffix = (dwords == 2) ? "_b64" : "_b128";
        rocke_h_emitf(lw,
                      "i32x%lld %s; __builtin_memcpy(&%s, &%s, %lld); "
                      "__builtin_amdgcn_raw_buffer_store%s(%s, %s, %s, %s, 0);",
                      (long long)dwords,
                      tmp,
                      tmp,
                      vname,
                      (long long)dwords * 4,
                      b_suffix,
                      tmp,
                      rocke_h_name(lw, rsrc),
                      rocke_h_name(lw, voffset),
                      rocke_h_name(lw, soffset));
    }
    else
    {
        return rocke_h_fail(lw,
                            ROCKE_ERR_KEY,
                            "tile.buffer_store_vN_f16: unsupported dwords=%lld",
                            (long long)dwords);
    }
    return lw->status;
}

/* ============================ async DRAM->LDS =========================== */

/* Python _op_tile_async_buffer_load_lds_addr */
static rocke_status_t _op_tile_async_buffer_load_lds_addr(rocke_h_lowerer_t* lw,
                                                          const rocke_op_t* op)
{
    rocke_value_t *rsrc, *lds_addr, *voff, *soff;
    int64_t dwords, size_bytes;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "tile.async_buffer_load_lds_addr: too few operands");
    }
    rsrc = op->operands[0];
    lds_addr = op->operands[1];
    voff = op->operands[2];
    soff = op->operands[3];
    dwords = mem_attr_int(op, "dwords", 0);
    size_bytes = dwords * 4;
    rocke_h_emitf(lw,
                  "_llvm_amdgcn_raw_ptr_buffer_load_lds(%s, "
                  "(__attribute__((address_space(3))) void*)(%s), "
                  "%lld, %s, %s, 0, 0);",
                  rocke_h_name(lw, rsrc),
                  rocke_h_name(lw, lds_addr),
                  (long long)size_bytes,
                  rocke_h_name(lw, voff),
                  rocke_h_name(lw, soff));
    return lw->status;
}

/* Python _op_tile_async_buffer_load_lds (typed-LDS variant) */
static rocke_status_t _op_tile_async_buffer_load_lds(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_value_t *rsrc, *lds_val, *voff, *soff;
    const char* storage;
    int64_t dwords, aux, size_bytes;
    if(!rocke_h_live(lw))
    {
        return lw->status;
    }
    if(op->num_operands < 4)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "tile.async_buffer_load_lds: too few operands");
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
        return rocke_h_fail(
            lw, ROCKE_ERR_VALUE, "async_buffer_load_lds before smem_alloc was lowered");
    }
    rocke_h_emitf(lw,
                  "_llvm_amdgcn_raw_ptr_buffer_load_lds(%s, "
                  "(__attribute__((address_space(3))) void*)&%s[0], "
                  "%lld, %s, %s, 0, %lld);",
                  rocke_h_name(lw, rsrc),
                  storage,
                  (long long)size_bytes,
                  rocke_h_name(lw, voff),
                  rocke_h_name(lw, soff),
                  (long long)aux);
    return lw->status;
}

/* ============================ registration table ======================== */

/* Bucket 2 handler table. Terminated by ROCKE_OP_INVALID. The Python
 * `tile.global_load_lds` op (ROCKE_OP_TILE_GLOBAL_LOAD_LDS) has no HIP handler in
 * lower_hip.py (it is an LLVM-path op), so it is intentionally absent here and
 * falls through to NotImplementedError parity. */
const rocke_h_handler_entry_t* rocke_h_handlers_mem(void)
{
    static const rocke_h_handler_entry_t table[]
        = {/* LDS alloc */
           {ROCKE_OP_TILE_SMEM_ALLOC, _op_tile_smem_alloc},
           /* LDS stores */
           {ROCKE_OP_TILE_SMEM_STORE, _op_tile_smem_store},
           {ROCKE_OP_TILE_SMEM_STORE_VN, _op_tile_smem_store_vN},
           {ROCKE_OP_TILE_SMEM_STORE_VN_F32, _op_tile_smem_store_vN_f32},
           {ROCKE_OP_TILE_SMEM_STORE_DISTRIBUTED, _op_tile_smem_store_distributed},
           /* LDS loads */
           {ROCKE_OP_TILE_SMEM_LOAD_V4, _op_tile_smem_load_v4},
           {ROCKE_OP_TILE_SMEM_LOAD_VN, _op_tile_smem_load_vN},
           {ROCKE_OP_TILE_SMEM_LOAD_VN_F32, _op_tile_smem_load_vN_f32},
           /* LDS pointer arithmetic */
           {ROCKE_OP_TILE_SMEM_ADDR_OF, _op_tile_smem_addr_of},
           {ROCKE_OP_TILE_SMEM_PTR_ADD, _op_tile_smem_ptr_add},
           /* LDS atomics */
           {ROCKE_OP_TILE_LDS_ATOMIC_ADD, _op_tile_lds_atomic_add},
           /* global load */
           {ROCKE_OP_MEMREF_GLOBAL_LOAD, _op_memref_global_load},
           {ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED, _op_memref_global_load_typed},
           {ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, _op_memref_global_load_vN},
           /* global store */
           {ROCKE_OP_MEMREF_GLOBAL_STORE, _op_memref_global_store},
           {ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED, _op_memref_global_store_typed},
           {ROCKE_OP_MEMREF_GLOBAL_STORE_VN, _op_memref_global_store_vN},
           /* atomics */
           {ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD, _op_memref_global_atomic_add},
           {ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32, _op_memref_global_atomic_add_f32},
           {ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16, _op_memref_global_atomic_add_pk_bf16},
           {ROCKE_OP_MEMREF_COOPERATIVE_GLOBAL_STORE, _op_memref_cooperative_global_store},
           /* global pointer arithmetic + buffer rsrc */
           {ROCKE_OP_TILE_GLOBAL_PTR_ADD, _op_tile_global_ptr_add},
           {ROCKE_OP_TILE_BUFFER_RSRC, _op_tile_buffer_rsrc},
           /* buffer load/store */
           {ROCKE_OP_TILE_BUFFER_LOAD_F16, _op_tile_buffer_load_f16},
           {ROCKE_OP_TILE_BUFFER_LOAD_VN_F16, _op_tile_buffer_load_vN_f16},
           {ROCKE_OP_TILE_BUFFER_LOAD_VN, _op_tile_buffer_load_vN},
           {ROCKE_OP_TILE_BUFFER_STORE_F16, _op_tile_buffer_store_f16},
           {ROCKE_OP_TILE_BUFFER_STORE_VN_F16, _op_tile_buffer_store_vN_f16},
           /* async DRAM->LDS */
           {ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS_ADDR, _op_tile_async_buffer_load_lds_addr},
           {ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS, _op_tile_async_buffer_load_lds},
           {ROCKE_OP_INVALID, NULL}};
    return table;
}

} /* namespace ckc */
