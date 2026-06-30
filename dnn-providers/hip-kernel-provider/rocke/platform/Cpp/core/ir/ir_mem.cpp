// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ir_ir_mem.c -- bucket "ir_mem" of the C99 port of rocke.core.ir.
 *
 * Covers: gpu ids, scalar/vectorised global loads & stores (+ masked load),
 * all atomics (global / lds / packed-bf16 / f32), the vector.* element-wise op
 * family, vec extract/insert/pack/concat, and vec bitcast/trunc/cast.
 *
 * Binds strictly to rocke/ir.h (the frozen contract). All shared plumbing
 * (rocke_i_*) is defined in bucket 0; here we only call it.
 */

#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_internal.h"

/* ============================== gpu ids ================================= */

rocke_value_t* rocke_b_thread_id_x(rocke_ir_builder_t* b)
{
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "axis", "x");
    return rocke_i_op1(b, ROCKE_OP_GPU_THREAD_ID, NULL, 0, rocke_i32(), &a, "tid");
}

static rocke_value_t* rocke_i_block_id_axis(rocke_ir_builder_t* b, const char* axis)
{
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "axis", axis);
    return rocke_i_op1(b, ROCKE_OP_GPU_BLOCK_ID, NULL, 0, rocke_i32(), &a, "bid");
}

rocke_value_t* rocke_b_block_id_x(rocke_ir_builder_t* b)
{
    return rocke_i_block_id_axis(b, "x");
}
rocke_value_t* rocke_b_block_id_y(rocke_ir_builder_t* b)
{
    return rocke_i_block_id_axis(b, "y");
}
rocke_value_t* rocke_b_block_id_z(rocke_ir_builder_t* b)
{
    return rocke_i_block_id_axis(b, "z");
}

/* ============================ global loads ============================== */

rocke_value_t* rocke_b_global_load(rocke_ir_builder_t* b,
                                   rocke_value_t* ptr,
                                   rocke_value_t* idx,
                                   const rocke_type_t* dtype,
                                   int align)
{
    rocke_value_t* ops[2];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !idx || !dtype)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "global_load: null operand/dtype");
    if(align <= 0)
        align = 1;
    ops[0] = ptr;
    ops[1] = idx;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", dtype->name);
    rocke_attr_set_int(b, &a, "align", (int64_t)align);
    return rocke_i_op1(b, ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED, ops, 2, dtype, &a, "gl");
}

rocke_value_t* rocke_b_global_load_f16(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align)
{
    rocke_value_t* ops[2];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !idx)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "global_load_f16: null operand");
    if(align <= 0)
        align = 2;
    ops[0] = ptr;
    ops[1] = idx;
    a = rocke_i_attrs(b);
    rocke_attr_set_int(b, &a, "align", (int64_t)align);
    return rocke_i_op1(b, ROCKE_OP_MEMREF_GLOBAL_LOAD, ops, 2, rocke_f16(), &a, "gl");
}

rocke_value_t* rocke_b_global_load_f32(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align)
{
    return rocke_b_global_load(b, ptr, idx, rocke_f32(), align <= 0 ? 4 : align);
}

rocke_value_t* rocke_b_global_load_i32(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align)
{
    return rocke_b_global_load(b, ptr, idx, rocke_i32(), align <= 0 ? 4 : align);
}

rocke_value_t* rocke_b_global_load_i64(rocke_ir_builder_t* b,
                                       rocke_value_t* ptr,
                                       rocke_value_t* idx,
                                       int align)
{
    return rocke_b_global_load(b, ptr, idx, rocke_i64(), align <= 0 ? 8 : align);
}

rocke_value_t* rocke_b_global_load_bf16(rocke_ir_builder_t* b,
                                        rocke_value_t* ptr,
                                        rocke_value_t* idx,
                                        int align)
{
    return rocke_b_global_load(b, ptr, idx, rocke_bf16(), align <= 0 ? 2 : align);
}

rocke_value_t* rocke_b_global_load_fp8e4m3(rocke_ir_builder_t* b,
                                           rocke_value_t* ptr,
                                           rocke_value_t* idx,
                                           int align)
{
    return rocke_b_global_load(b, ptr, idx, rocke_fp8e4m3(), align <= 0 ? 1 : align);
}

rocke_value_t* rocke_b_masked_global_load(rocke_ir_builder_t* b,
                                          rocke_value_t* ptr,
                                          rocke_value_t* idx,
                                          rocke_value_t* mask,
                                          rocke_value_t* other,
                                          const rocke_type_t* dtype,
                                          int align)
{
    rocke_value_t *zero, *safe_idx, *loaded;
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !idx || !mask || !other || !dtype)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "masked_global_load: null operand/dtype");
    if(!rocke_i_type_is(idx->type, "i32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "masked_global_load expects i32 index for clamp-safe load");
    /* safe_idx = select(mask, idx, const_i32(0)); these live in other buckets. */
    zero = rocke_b_const_i32(b, 0);
    safe_idx = rocke_b_select(b, mask, idx, zero);
    loaded = rocke_b_global_load(b, ptr, safe_idx, dtype, align);
    return rocke_b_select(b, mask, loaded, other);
}

void rocke_b_global_store(
    rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* idx, rocke_value_t* value, int align)
{
    rocke_value_t* ops[3];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "global_store: null operand");
        return;
    }
    if(align <= 0)
        align = 1;
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", value->type->name);
    rocke_attr_set_int(b, &a, "align", (int64_t)align);
    (void)rocke_i_op0(b, ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED, ops, 3, &a);
}

rocke_value_t* rocke_b_global_load_vN(rocke_ir_builder_t* b,
                                      rocke_value_t* ptr,
                                      rocke_value_t* idx,
                                      const rocke_type_t* dtype,
                                      int n,
                                      int align)
{
    rocke_value_t* ops[2];
    rocke_attr_map_t a;
    const rocke_type_t* vt;
    int elem_bytes;
    const char* en;
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !idx || !dtype)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "global_load_vN: null operand/dtype");
    en = dtype->name;
    if(rocke_i_type_is(dtype, "f16") || rocke_i_type_is(dtype, "bf16")
       || rocke_i_type_is(dtype, "i16"))
    {
        elem_bytes = 2;
        if(n != 2 && n != 4 && n != 8 && n != 16)
            return (rocke_value_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "unsupported vector width for global_load_vN: %d", n);
    }
    else if(rocke_i_type_is(dtype, "f32") || rocke_i_type_is(dtype, "i32"))
    {
        elem_bytes = 4;
        if(n != 2 && n != 4 && n != 8)
            return (rocke_value_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "unsupported vector width for %s global_load_vN: %d", en, n);
    }
    else if(rocke_i_type_is(dtype, "fp8e4m3") || rocke_i_type_is(dtype, "bf8e5m2")
            || rocke_i_type_is(dtype, "i8"))
    {
        elem_bytes = 1;
        if(n != 2 && n != 4 && n != 8 && n != 16)
            return (rocke_value_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "unsupported vector width for %s global_load_vN: %d", en, n);
    }
    else
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "global_load_vN supports f16/bf16/i16/f32/i32/fp8e4m3/bf8e5m2/i8, got %s",
            en);
    }
    vt = rocke_vector_type(b, dtype, n);
    if(!vt)
        return NULL;
    ops[0] = ptr;
    ops[1] = idx;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", en);
    rocke_attr_set_int(b, &a, "vec", (int64_t)n);
    rocke_attr_set_int(b, &a, "align", (int64_t)(align > 0 ? align : n * elem_bytes));
    {
        char hint[16];
        /* result_name_hint = "gv{n}" */
        int i = 0, val = n, j;
        char tmp[8];
        hint[i++] = 'g';
        hint[i++] = 'v';
        if(val == 0)
        {
            hint[i++] = '0';
        }
        else
        {
            int t = 0;
            while(val > 0)
            {
                tmp[t++] = (char)('0' + val % 10);
                val /= 10;
            }
            for(j = t - 1; j >= 0; --j)
                hint[i++] = tmp[j];
        }
        hint[i] = '\0';
        return rocke_i_op1(b, ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, ops, 2, vt, &a, hint);
    }
}

rocke_value_t* rocke_b_global_load_vN_f16(
    rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* idx, int n, int align)
{
    return rocke_b_global_load_vN(b, ptr, idx, rocke_f16(), n, align);
}

/* ====================== vectorised global stores ======================= */

void rocke_b_global_store_vN(rocke_ir_builder_t* b,
                             rocke_value_t* ptr,
                             rocke_value_t* idx,
                             rocke_value_t* value,
                             int n,
                             int align)
{
    rocke_value_t* ops[3];
    rocke_attr_map_t a;
    const rocke_type_t* et;
    const char* en;
    int elem_bytes;
    if(!rocke_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "global_store_vN: null operand");
        return;
    }
    if(n != 1 && n != 2 && n != 4 && n != 8 && n != 16)
    {
        (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "global_store_vN n must be 1, 2, 4, 8, or 16 (got %d)", n);
        return;
    }
    et = rocke_i_elem_of(value->type); /* vector elem, or scalar type itself */
    en = et->name;
    if(rocke_i_type_is(et, "f16") || rocke_i_type_is(et, "bf16") || rocke_i_type_is(et, "i16"))
    {
        elem_bytes = 2;
        if(n == 16)
        {
            (void)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "global_store_vN n=16 not supported for %s", en);
            return;
        }
    }
    else if(rocke_i_type_is(et, "f32") || rocke_i_type_is(et, "i32"))
    {
        elem_bytes = 4;
        if(n == 16)
        {
            (void)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "global_store_vN n=16 not supported for %s", en);
            return;
        }
    }
    else if(rocke_i_type_is(et, "i8") || rocke_i_type_is(et, "fp8e4m3")
            || rocke_i_type_is(et, "bf8e5m2"))
    {
        elem_bytes = 1;
    }
    else
    {
        (void)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "global_store_vN supports f16/bf16/i16/f32/i32/i8/fp8e4m3/bf8e5m2, got %s",
            en);
        return;
    }
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", en);
    rocke_attr_set_int(b, &a, "vec", (int64_t)n);
    rocke_attr_set_int(b, &a, "align", (int64_t)(align > 0 ? align : n * elem_bytes));
    (void)rocke_i_op0(b, ROCKE_OP_MEMREF_GLOBAL_STORE_VN, ops, 3, &a);
}

void rocke_b_global_store_vN_f16(rocke_ir_builder_t* b,
                                 rocke_value_t* ptr,
                                 rocke_value_t* idx,
                                 rocke_value_t* value,
                                 int n,
                                 int align)
{
    rocke_b_global_store_vN(b, ptr, idx, value, n, align);
}

void rocke_b_store_f16(rocke_ir_builder_t* b,
                       rocke_value_t* ptr,
                       rocke_value_t* idx,
                       rocke_value_t* value)
{
    rocke_value_t* ops[3];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "store_f16: null operand");
        return;
    }
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", "f16");
    rocke_attr_set_int(b, &a, "align", 2);
    (void)rocke_i_op0(b, ROCKE_OP_MEMREF_GLOBAL_STORE, ops, 3, &a);
}

rocke_value_t* rocke_b_zero_vec_f16(rocke_ir_builder_t* b, int n)
{
    if(!rocke_i_live(b))
        return NULL;
    if(n <= 0)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "zero_vec_f16 needs positive n, got %d", n);
    /* rocke_b_zero_vec (elem=f16) lives in bucket 0. */
    return rocke_b_zero_vec(b, rocke_f16(), n);
}

/* ================================ atomics ============================== */

static bool rocke_i_ordering_ok(const char* ordering)
{
    /* {monotonic, acquire, release, acq_rel, seq_cst} */
    return ordering
           && (!strcmp(ordering, "monotonic") || !strcmp(ordering, "acquire")
               || !strcmp(ordering, "release") || !strcmp(ordering, "acq_rel")
               || !strcmp(ordering, "seq_cst"));
}

rocke_value_t* rocke_b_global_atomic_add(rocke_ir_builder_t* b,
                                         rocke_value_t* ptr,
                                         rocke_value_t* idx,
                                         rocke_value_t* value,
                                         const char* ordering)
{
    rocke_value_t* ops[3];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !idx || !value)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "global_atomic_add: null operand");
    if(ordering == NULL)
        ordering = "monotonic";
    if(!rocke_i_type_is(value->type, "i32") && !rocke_i_type_is(value->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "global_atomic_add supports i32 / f32, got %s", value->type->name);
    if(!rocke_i_ordering_ok(ordering))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unknown ordering '%s'", ordering);
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", value->type->name);
    rocke_attr_set_str(b, &a, "ordering", ordering);
    return rocke_i_op1(b, ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD, ops, 3, value->type, &a, "atom_add");
}

rocke_value_t* rocke_b_lds_atomic_add(rocke_ir_builder_t* b,
                                      rocke_value_t* smem,
                                      rocke_value_t* const* indices,
                                      int num_indices,
                                      rocke_value_t* value,
                                      const char* ordering)
{
    rocke_value_t** ops;
    rocke_attr_map_t a;
    int i;
    if(!rocke_i_live(b))
        return NULL;
    if(!smem || !value || (num_indices > 0 && !indices))
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "lds_atomic_add: null operand");
    if(num_indices < 0)
        num_indices = 0;
    if(ordering == NULL)
        ordering = "monotonic";
    if(!rocke_i_type_is(value->type, "i32") && !rocke_i_type_is(value->type, "f32"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "lds_atomic_add supports i32 / f32, got %s", value->type->name);
    if(!rocke_i_ordering_ok(ordering))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unknown ordering '%s'", ordering);
    /* operands = [smem, *indices, value] */
    ops = (rocke_value_t**)rocke_arena_alloc(&b->arena, sizeof(*ops) * (size_t)(num_indices + 2));
    if(!ops)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "lds_atomic_add: OOM");
    ops[0] = smem;
    for(i = 0; i < num_indices; ++i)
        ops[1 + i] = indices[i];
    ops[1 + num_indices] = value;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", value->type->name);
    rocke_attr_set_int(b, &a, "rank", (int64_t)num_indices);
    rocke_attr_set_str(b, &a, "ordering", ordering);
    return rocke_i_op1(
        b, ROCKE_OP_TILE_LDS_ATOMIC_ADD, ops, num_indices + 2, value->type, &a, "lds_atom");
}

rocke_value_t* rocke_b_global_atomic_add_pk_bf16(rocke_ir_builder_t* b,
                                                 rocke_value_t* ptr,
                                                 rocke_value_t* idx,
                                                 rocke_value_t* value,
                                                 const char* ordering)
{
    rocke_value_t* ops[3];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !idx || !value)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "global_atomic_add_pk_bf16: null operand");
    if(ordering == NULL)
        ordering = "monotonic";
    if(!rocke_i_ordering_ok(ordering))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unknown ordering '%s'", ordering);
    if(!rocke_i_is_vector(value->type, "bf16", 2))
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "global_atomic_add_pk_bf16 expects <2 x bf16> input, got %s",
            value->type->name);
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem_type", "bf16");
    rocke_attr_set_int(b, &a, "vec", 2);
    rocke_attr_set_str(b, &a, "ordering", ordering);
    return rocke_i_op1(
        b, ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16, ops, 3, value->type, &a, "atom_bf16");
}

void rocke_b_global_atomic_add_f32(rocke_ir_builder_t* b,
                                   rocke_value_t* ptr,
                                   rocke_value_t* idx,
                                   rocke_value_t* value)
{
    rocke_value_t* ops[3];
    if(!rocke_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "global_atomic_add_f32: null operand");
        return;
    }
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    (void)rocke_i_op0(b, ROCKE_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32, ops, 3, NULL);
}

/* ===================== vector.* element-wise family ===================== */

/* IRBuilder.vector_binary: matching vector operands -> a->type. */
static rocke_value_t* rocke_i_vector_binary(rocke_ir_builder_t* b,
                                            rocke_opcode_t opcode,
                                            rocke_value_t* a,
                                            rocke_value_t* c,
                                            const char* hint)
{
    rocke_value_t* ops[2];
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vector_binary: null operand");
    if(!rocke_i_is_vector(a->type, NULL, -1) || !rocke_type_eq(a->type, c->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_binary expects matching vector operands");
    ops[0] = a;
    ops[1] = c;
    return rocke_i_op1(b, opcode, ops, 2, a->type, NULL, hint);
}

rocke_value_t* rocke_b_vector_add(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_ADD, a, c, "vadd");
}
rocke_value_t* rocke_b_vector_sub(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_SUB, a, c, "vsub");
}
rocke_value_t* rocke_b_vector_mul(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_MUL, a, c, "vmul");
}
rocke_value_t* rocke_b_vector_and(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_AND, a, c, "vand");
}
rocke_value_t* rocke_b_vector_or(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_OR, a, c, "vor");
}
rocke_value_t* rocke_b_vector_shl(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_SHL, a, c, "vshl");
}
rocke_value_t* rocke_b_vector_lshr(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_LSHR, a, c, "vlshr");
}
rocke_value_t* rocke_b_vector_smax(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_SMAX, a, c, "vsmax");
}
rocke_value_t* rocke_b_vector_smin(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_SMIN, a, c, "vsmin");
}
rocke_value_t* rocke_b_vector_max(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    return rocke_i_vector_binary(b, ROCKE_OP_VECTOR_MAX, a, c, "vmax");
}

rocke_value_t*
    rocke_b_vector_fma(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, rocke_value_t* d)
{
    rocke_value_t* ops[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c || !d)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vector_fma: null operand");
    if(!rocke_i_is_vector(a->type, NULL, -1) || !rocke_type_eq(a->type, c->type)
       || !rocke_type_eq(a->type, d->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_fma expects three matching vector operands");
    ops[0] = a;
    ops[1] = c;
    ops[2] = d;
    return rocke_i_op1(b, ROCKE_OP_VECTOR_FMA, ops, 3, a->type, NULL, "vfma");
}

rocke_value_t* rocke_b_vector_sum(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* ops[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vector_sum: null operand");
    if(!rocke_i_is_vector(v->type, NULL, -1))
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vector_sum expects vector");
    ops[0] = v;
    return rocke_i_op1(b, ROCKE_OP_VECTOR_SUM, ops, 1, rocke_i_elem_of(v->type), NULL, "vsum");
}

rocke_value_t* rocke_b_vector_reduce_max(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* ops[1];
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_reduce_max: null operand");
    if(!rocke_i_is_vector(v->type, NULL, -1))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_reduce_max expects vector");
    ops[0] = v;
    return rocke_i_op1(
        b, ROCKE_OP_VECTOR_REDUCE_MAX, ops, 1, rocke_i_elem_of(v->type), NULL, "vmax");
}

rocke_value_t* rocke_b_vector_splat(rocke_ir_builder_t* b, rocke_value_t* scalar, int n)
{
    rocke_value_t* ops[1];
    rocke_attr_map_t a;
    const rocke_type_t* vt;
    if(!rocke_i_live(b))
        return NULL;
    if(!scalar)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vector_splat: null operand");
    vt = rocke_vector_type(b, scalar->type, n);
    if(!vt)
        return NULL;
    ops[0] = scalar;
    a = rocke_i_attrs(b);
    rocke_attr_set_int(b, &a, "vec", (int64_t)n);
    return rocke_i_op1(b, ROCKE_OP_VECTOR_SPLAT, ops, 1, vt, &a, "splat");
}

rocke_value_t* rocke_b_vector_select(rocke_ir_builder_t* b,
                                     rocke_value_t* mask,
                                     rocke_value_t* lhs,
                                     rocke_value_t* rhs)
{
    rocke_value_t* ops[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!mask || !lhs || !rhs)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vector_select: null operand");
    if(!rocke_type_eq(lhs->type, rhs->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_select lhs/rhs type mismatch");
    ops[0] = mask;
    ops[1] = lhs;
    ops[2] = rhs;
    return rocke_i_op1(b, ROCKE_OP_VECTOR_SELECT, ops, 3, lhs->type, NULL, "vsel");
}

rocke_value_t*
    rocke_b_vector_cmp(rocke_ir_builder_t* b, const char* pred, rocke_value_t* a, rocke_value_t* c)
{
    rocke_value_t* ops[2];
    rocke_attr_map_t attr;
    const rocke_type_t* vt;
    char hint[32];
    int i, j;
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !c || !pred)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vector_cmp: null operand/pred");
    if(!rocke_i_is_vector(a->type, NULL, -1) || !rocke_type_eq(a->type, c->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_cmp expects matching vector operands");
    vt = rocke_vector_type(b, rocke_i1(), rocke_i_count_of(a->type));
    if(!vt)
        return NULL;
    ops[0] = a;
    ops[1] = c;
    attr = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attr, "pred", pred);
    /* result_name_hint = "vcmp_{pred}" */
    i = 0;
    hint[i++] = 'v';
    hint[i++] = 'c';
    hint[i++] = 'm';
    hint[i++] = 'p';
    hint[i++] = '_';
    for(j = 0; pred[j] && i < (int)sizeof(hint) - 1; ++j)
        hint[i++] = pred[j];
    hint[i] = '\0';
    return rocke_i_op1(b, ROCKE_OP_VECTOR_CMP, ops, 2, vt, &attr, hint);
}

/* shared "vN{count}" hint formatter for vector_trunc/sext */
static void rocke_i_count_hint(char* buf, const char* prefix, int count)
{
    int i = 0, val = count, j, t;
    char tmp[8];
    while(prefix[i])
    {
        buf[i] = prefix[i];
        ++i;
    }
    if(val <= 0)
    {
        buf[i++] = '0';
        buf[i] = '\0';
        return;
    }
    t = 0;
    while(val > 0)
    {
        tmp[t++] = (char)('0' + val % 10);
        val /= 10;
    }
    for(j = t - 1; j >= 0; --j)
        buf[i++] = tmp[j];
    buf[i] = '\0';
}

rocke_value_t*
    rocke_b_vector_trunc(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* ops[1];
    rocke_attr_map_t a;
    const rocke_type_t* vt;
    char hint[16];
    if(!rocke_i_live(b))
        return NULL;
    if(!v || !target)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_trunc: null operand/target");
    if(!rocke_i_is_vector(v->type, NULL, -1))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_trunc expects vector input");
    vt = rocke_vector_type(b, target, rocke_i_count_of(v->type));
    if(!vt)
        return NULL;
    ops[0] = v;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "target", target->name);
    rocke_i_count_hint(hint, "vtr", rocke_i_count_of(v->type));
    return rocke_i_op1(b, ROCKE_OP_VECTOR_TRUNC, ops, 1, vt, &a, hint);
}

rocke_value_t*
    rocke_b_vector_sext(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* ops[1];
    rocke_attr_map_t a;
    const rocke_type_t* vt;
    char hint[16];
    if(!rocke_i_live(b))
        return NULL;
    if(!v || !target)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_sext: null operand/target");
    if(!rocke_i_is_vector(v->type, NULL, -1))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vector_sext expects vector input");
    vt = rocke_vector_type(b, target, rocke_i_count_of(v->type));
    if(!vt)
        return NULL;
    ops[0] = v;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "target", target->name);
    rocke_i_count_hint(hint, "vsx", rocke_i_count_of(v->type));
    return rocke_i_op1(b, ROCKE_OP_VECTOR_SEXT, ops, 1, vt, &a, hint);
}

/* ===================== vec extract/insert/pack/concat =================== */

rocke_value_t* rocke_b_vec_extract(rocke_ir_builder_t* b, rocke_value_t* v, int i)
{
    rocke_value_t* ops[1];
    rocke_attr_map_t a;
    const rocke_type_t* et;
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vec_extract: null operand");
    et = rocke_i_elem_of(v->type); /* vec elem, or scalar type itself */
    ops[0] = v;
    a = rocke_i_attrs(b);
    rocke_attr_set_int(b, &a, "index", (int64_t)i);
    return rocke_i_op1(b, ROCKE_OP_VECTOR_EXTRACT, ops, 1, et, &a, "e");
}

rocke_value_t*
    rocke_b_vec_insert(rocke_ir_builder_t* b, rocke_value_t* v, rocke_value_t* scalar, int i)
{
    rocke_value_t* ops[2];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    if(!v || !scalar)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vec_insert: null operand");
    if(!rocke_i_is_vector(v->type, NULL, -1))
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vec_insert expects vector");
    if(!rocke_type_eq(scalar->type, rocke_i_elem_of(v->type)))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_insert scalar type mismatch");
    ops[0] = v;
    ops[1] = scalar;
    a = rocke_i_attrs(b);
    rocke_attr_set_int(b, &a, "index", (int64_t)i);
    return rocke_i_op1(b, ROCKE_OP_VECTOR_INSERT, ops, 2, v->type, &a, "vi");
}

rocke_value_t* rocke_b_vec_pack(rocke_ir_builder_t* b,
                                rocke_value_t* const* components,
                                int num_components,
                                const rocke_type_t* elem)
{
    rocke_attr_map_t a;
    const rocke_type_t* vt;
    int i;
    if(!rocke_i_live(b))
        return NULL;
    if(!elem)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vec_pack: null elem");
    if(num_components <= 0 || !components)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_pack needs at least one component");
    for(i = 0; i < num_components; ++i)
    {
        if(!components[i] || !rocke_type_eq(components[i]->type, elem))
            return (rocke_value_t*)rocke_i_set_err(b,
                                                   ROCKE_ERR_VALUE,
                                                   "vec_pack expected %s, got %s",
                                                   elem->name,
                                                   components[i] ? components[i]->type->name
                                                                 : "(null)");
    }
    vt = rocke_vector_type(b, elem, num_components);
    if(!vt)
        return NULL;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "elem", elem->name);
    rocke_attr_set_int(b, &a, "vec", (int64_t)num_components);
    return rocke_i_op1(b, ROCKE_OP_VECTOR_PACK, components, num_components, vt, &a, "vp");
}

rocke_value_t* rocke_b_vec_concat(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb)
{
    rocke_value_t* ops[2];
    rocke_attr_map_t attr;
    const rocke_type_t *elem, *vt;
    int n;
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !bb)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "vec_concat: null operand");
    if(!rocke_i_is_vector(a->type, NULL, -1) || !rocke_i_is_vector(bb->type, NULL, -1))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_concat needs vector inputs");
    elem = rocke_i_elem_of(a->type);
    if(!rocke_type_eq(elem, rocke_i_elem_of(bb->type)))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_concat element types must match");
    n = rocke_i_count_of(a->type) + rocke_i_count_of(bb->type);
    vt = rocke_vector_type(b, elem, n);
    if(!vt)
        return NULL;
    ops[0] = a;
    ops[1] = bb;
    attr = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attr, "elem", elem->name);
    rocke_attr_set_int(b, &attr, "vec", (int64_t)n);
    return rocke_i_op1(b, ROCKE_OP_VECTOR_CONCAT, ops, 2, vt, &attr, "vc");
}

/* =================== vec bitcast / packed f32->fXX ===================== */

rocke_value_t*
    rocke_b_vec_bitcast(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* ops[1];
    rocke_attr_map_t a;
    if(!rocke_i_live(b))
        return NULL;
    if(!v || !target)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_bitcast: null operand/target");
    ops[0] = v;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "target", target->name);
    return rocke_i_op1(b, ROCKE_OP_VECTOR_BITCAST, ops, 1, target, &a, "bc");
}

rocke_value_t*
    rocke_b_vec_cast_f32_to(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* target)
{
    rocke_value_t* ops[1];
    rocke_attr_map_t a;
    const rocke_type_t* vt;
    char hint[16];
    if(!rocke_i_live(b))
        return NULL;
    if(!v || !target)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_cast_f32_to: null operand/target");
    if(!rocke_i_is_vector(v->type, "f32", -1))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_cast_f32_to expects <N x f32>");
    if(!rocke_i_type_is(target, "f16") && !rocke_i_type_is(target, "bf16"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "vec_cast_f32_to unsupported target %s", target->name);
    vt = rocke_vector_type(b, target, rocke_i_count_of(v->type));
    if(!vt)
        return NULL;
    ops[0] = v;
    a = rocke_i_attrs(b);
    rocke_attr_set_str(b, &a, "target", target->name);
    rocke_i_count_hint(hint, "vh", rocke_i_count_of(v->type));
    return rocke_i_op1(b, ROCKE_OP_VECTOR_TRUNC_F32_TO, ops, 1, vt, &a, hint);
}

rocke_value_t* rocke_b_vec_trunc_f32_to_f16(rocke_ir_builder_t* b, rocke_value_t* v)
{
    return rocke_b_vec_cast_f32_to(b, v, rocke_f16());
}
