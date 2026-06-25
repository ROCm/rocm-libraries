// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ir_ir_mem.c -- bucket "ir_mem" of the C99 port of ck_dsl.core.ir.
 *
 * Covers: gpu ids, scalar/vectorised global loads & stores (+ masked load),
 * all atomics (global / lds / packed-bf16 / f32), the vector.* element-wise op
 * family, vec extract/insert/pack/concat, and vec bitcast/trunc/cast.
 *
 * Binds strictly to ckc/ir.h (the frozen contract). All shared plumbing
 * (ckc_i_*) is defined in bucket 0; here we only call it.
 */

#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_internal.h"

/* ============================== gpu ids ================================= */

ckc_value_t* ckc_b_thread_id_x(ckc_ir_builder_t* b)
{
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "axis", "x");
    return ckc_i_op1(b, CKC_OP_GPU_THREAD_ID, NULL, 0, ckc_i32(), &a, "tid");
}

static ckc_value_t* ckc_i_block_id_axis(ckc_ir_builder_t* b, const char* axis)
{
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "axis", axis);
    return ckc_i_op1(b, CKC_OP_GPU_BLOCK_ID, NULL, 0, ckc_i32(), &a, "bid");
}

ckc_value_t* ckc_b_block_id_x(ckc_ir_builder_t* b)
{
    return ckc_i_block_id_axis(b, "x");
}
ckc_value_t* ckc_b_block_id_y(ckc_ir_builder_t* b)
{
    return ckc_i_block_id_axis(b, "y");
}
ckc_value_t* ckc_b_block_id_z(ckc_ir_builder_t* b)
{
    return ckc_i_block_id_axis(b, "z");
}

/* ============================ global loads ============================== */

ckc_value_t* ckc_b_global_load(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, const ckc_type_t* dtype, int align)
{
    ckc_value_t* ops[2];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    if(!ptr || !idx || !dtype)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "global_load: null operand/dtype");
    if(align <= 0)
        align = 1;
    ops[0] = ptr;
    ops[1] = idx;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", dtype->name);
    ckc_attr_set_int(b, &a, "align", (int64_t)align);
    return ckc_i_op1(b, CKC_OP_MEMREF_GLOBAL_LOAD_TYPED, ops, 2, dtype, &a, "gl");
}

ckc_value_t*
    ckc_b_global_load_f16(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align)
{
    ckc_value_t* ops[2];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    if(!ptr || !idx)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "global_load_f16: null operand");
    if(align <= 0)
        align = 2;
    ops[0] = ptr;
    ops[1] = idx;
    a = ckc_i_attrs(b);
    ckc_attr_set_int(b, &a, "align", (int64_t)align);
    return ckc_i_op1(b, CKC_OP_MEMREF_GLOBAL_LOAD, ops, 2, ckc_f16(), &a, "gl");
}

ckc_value_t*
    ckc_b_global_load_f32(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align)
{
    return ckc_b_global_load(b, ptr, idx, ckc_f32(), align <= 0 ? 4 : align);
}

ckc_value_t*
    ckc_b_global_load_i32(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align)
{
    return ckc_b_global_load(b, ptr, idx, ckc_i32(), align <= 0 ? 4 : align);
}

ckc_value_t*
    ckc_b_global_load_i64(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align)
{
    return ckc_b_global_load(b, ptr, idx, ckc_i64(), align <= 0 ? 8 : align);
}

ckc_value_t*
    ckc_b_global_load_bf16(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align)
{
    return ckc_b_global_load(b, ptr, idx, ckc_bf16(), align <= 0 ? 2 : align);
}

ckc_value_t*
    ckc_b_global_load_fp8e4m3(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int align)
{
    return ckc_b_global_load(b, ptr, idx, ckc_fp8e4m3(), align <= 0 ? 1 : align);
}

ckc_value_t* ckc_b_masked_global_load(ckc_ir_builder_t* b,
                                      ckc_value_t* ptr,
                                      ckc_value_t* idx,
                                      ckc_value_t* mask,
                                      ckc_value_t* other,
                                      const ckc_type_t* dtype,
                                      int align)
{
    ckc_value_t *zero, *safe_idx, *loaded;
    if(!ckc_i_live(b))
        return NULL;
    if(!ptr || !idx || !mask || !other || !dtype)
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "masked_global_load: null operand/dtype");
    if(!ckc_i_type_is(idx->type, "i32"))
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "masked_global_load expects i32 index for clamp-safe load");
    /* safe_idx = select(mask, idx, const_i32(0)); these live in other buckets. */
    zero = ckc_b_const_i32(b, 0);
    safe_idx = ckc_b_select(b, mask, idx, zero);
    loaded = ckc_b_global_load(b, ptr, safe_idx, dtype, align);
    return ckc_b_select(b, mask, loaded, other);
}

void ckc_b_global_store(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value, int align)
{
    ckc_value_t* ops[3];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "global_store: null operand");
        return;
    }
    if(align <= 0)
        align = 1;
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", value->type->name);
    ckc_attr_set_int(b, &a, "align", (int64_t)align);
    (void)ckc_i_op0(b, CKC_OP_MEMREF_GLOBAL_STORE_TYPED, ops, 3, &a);
}

ckc_value_t* ckc_b_global_load_vN(ckc_ir_builder_t* b,
                                  ckc_value_t* ptr,
                                  ckc_value_t* idx,
                                  const ckc_type_t* dtype,
                                  int n,
                                  int align)
{
    ckc_value_t* ops[2];
    ckc_attr_map_t a;
    const ckc_type_t* vt;
    int elem_bytes;
    const char* en;
    if(!ckc_i_live(b))
        return NULL;
    if(!ptr || !idx || !dtype)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "global_load_vN: null operand/dtype");
    en = dtype->name;
    if(ckc_i_type_is(dtype, "f16") || ckc_i_type_is(dtype, "bf16") || ckc_i_type_is(dtype, "i16"))
    {
        elem_bytes = 2;
        if(n != 2 && n != 4 && n != 8 && n != 16)
            return (ckc_value_t*)ckc_i_set_err(
                b, CKC_ERR_VALUE, "unsupported vector width for global_load_vN: %d", n);
    }
    else if(ckc_i_type_is(dtype, "f32") || ckc_i_type_is(dtype, "i32"))
    {
        elem_bytes = 4;
        if(n != 2 && n != 4 && n != 8)
            return (ckc_value_t*)ckc_i_set_err(
                b, CKC_ERR_VALUE, "unsupported vector width for %s global_load_vN: %d", en, n);
    }
    else if(ckc_i_type_is(dtype, "fp8e4m3") || ckc_i_type_is(dtype, "bf8e5m2")
            || ckc_i_type_is(dtype, "i8"))
    {
        elem_bytes = 1;
        if(n != 2 && n != 4 && n != 8 && n != 16)
            return (ckc_value_t*)ckc_i_set_err(
                b, CKC_ERR_VALUE, "unsupported vector width for %s global_load_vN: %d", en, n);
    }
    else
    {
        return (ckc_value_t*)ckc_i_set_err(
            b,
            CKC_ERR_VALUE,
            "global_load_vN supports f16/bf16/i16/f32/i32/fp8e4m3/bf8e5m2/i8, got %s",
            en);
    }
    vt = ckc_vector_type(b, dtype, n);
    if(!vt)
        return NULL;
    ops[0] = ptr;
    ops[1] = idx;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", en);
    ckc_attr_set_int(b, &a, "vec", (int64_t)n);
    ckc_attr_set_int(b, &a, "align", (int64_t)(align > 0 ? align : n * elem_bytes));
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
        return ckc_i_op1(b, CKC_OP_MEMREF_GLOBAL_LOAD_VN, ops, 2, vt, &a, hint);
    }
}

ckc_value_t* ckc_b_global_load_vN_f16(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, int n, int align)
{
    return ckc_b_global_load_vN(b, ptr, idx, ckc_f16(), n, align);
}

/* ====================== vectorised global stores ======================= */

void ckc_b_global_store_vN(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value, int n, int align)
{
    ckc_value_t* ops[3];
    ckc_attr_map_t a;
    const ckc_type_t* et;
    const char* en;
    int elem_bytes;
    if(!ckc_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "global_store_vN: null operand");
        return;
    }
    if(n != 1 && n != 2 && n != 4 && n != 8 && n != 16)
    {
        (void)ckc_i_set_err(
            b, CKC_ERR_VALUE, "global_store_vN n must be 1, 2, 4, 8, or 16 (got %d)", n);
        return;
    }
    et = ckc_i_elem_of(value->type); /* vector elem, or scalar type itself */
    en = et->name;
    if(ckc_i_type_is(et, "f16") || ckc_i_type_is(et, "bf16") || ckc_i_type_is(et, "i16"))
    {
        elem_bytes = 2;
        if(n == 16)
        {
            (void)ckc_i_set_err(b, CKC_ERR_VALUE, "global_store_vN n=16 not supported for %s", en);
            return;
        }
    }
    else if(ckc_i_type_is(et, "f32") || ckc_i_type_is(et, "i32"))
    {
        elem_bytes = 4;
        if(n == 16)
        {
            (void)ckc_i_set_err(b, CKC_ERR_VALUE, "global_store_vN n=16 not supported for %s", en);
            return;
        }
    }
    else if(ckc_i_type_is(et, "i8") || ckc_i_type_is(et, "fp8e4m3") || ckc_i_type_is(et, "bf8e5m2"))
    {
        elem_bytes = 1;
    }
    else
    {
        (void)ckc_i_set_err(
            b,
            CKC_ERR_VALUE,
            "global_store_vN supports f16/bf16/i16/f32/i32/i8/fp8e4m3/bf8e5m2, got %s",
            en);
        return;
    }
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", en);
    ckc_attr_set_int(b, &a, "vec", (int64_t)n);
    ckc_attr_set_int(b, &a, "align", (int64_t)(align > 0 ? align : n * elem_bytes));
    (void)ckc_i_op0(b, CKC_OP_MEMREF_GLOBAL_STORE_VN, ops, 3, &a);
}

void ckc_b_global_store_vN_f16(
    ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value, int n, int align)
{
    ckc_b_global_store_vN(b, ptr, idx, value, n, align);
}

void ckc_b_store_f16(ckc_ir_builder_t* b, ckc_value_t* ptr, ckc_value_t* idx, ckc_value_t* value)
{
    ckc_value_t* ops[3];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "store_f16: null operand");
        return;
    }
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", "f16");
    ckc_attr_set_int(b, &a, "align", 2);
    (void)ckc_i_op0(b, CKC_OP_MEMREF_GLOBAL_STORE, ops, 3, &a);
}

ckc_value_t* ckc_b_zero_vec_f16(ckc_ir_builder_t* b, int n)
{
    if(!ckc_i_live(b))
        return NULL;
    if(n <= 0)
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "zero_vec_f16 needs positive n, got %d", n);
    /* ckc_b_zero_vec (elem=f16) lives in bucket 0. */
    return ckc_b_zero_vec(b, ckc_f16(), n);
}

/* ================================ atomics ============================== */

static bool ckc_i_ordering_ok(const char* ordering)
{
    /* {monotonic, acquire, release, acq_rel, seq_cst} */
    return ordering
           && (!strcmp(ordering, "monotonic") || !strcmp(ordering, "acquire")
               || !strcmp(ordering, "release") || !strcmp(ordering, "acq_rel")
               || !strcmp(ordering, "seq_cst"));
}

ckc_value_t* ckc_b_global_atomic_add(ckc_ir_builder_t* b,
                                     ckc_value_t* ptr,
                                     ckc_value_t* idx,
                                     ckc_value_t* value,
                                     const char* ordering)
{
    ckc_value_t* ops[3];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    if(!ptr || !idx || !value)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "global_atomic_add: null operand");
    if(ordering == NULL)
        ordering = "monotonic";
    if(!ckc_i_type_is(value->type, "i32") && !ckc_i_type_is(value->type, "f32"))
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "global_atomic_add supports i32 / f32, got %s", value->type->name);
    if(!ckc_i_ordering_ok(ordering))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "unknown ordering '%s'", ordering);
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", value->type->name);
    ckc_attr_set_str(b, &a, "ordering", ordering);
    return ckc_i_op1(b, CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD, ops, 3, value->type, &a, "atom_add");
}

ckc_value_t* ckc_b_lds_atomic_add(ckc_ir_builder_t* b,
                                  ckc_value_t* smem,
                                  ckc_value_t* const* indices,
                                  int num_indices,
                                  ckc_value_t* value,
                                  const char* ordering)
{
    ckc_value_t** ops;
    ckc_attr_map_t a;
    int i;
    if(!ckc_i_live(b))
        return NULL;
    if(!smem || !value || (num_indices > 0 && !indices))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "lds_atomic_add: null operand");
    if(num_indices < 0)
        num_indices = 0;
    if(ordering == NULL)
        ordering = "monotonic";
    if(!ckc_i_type_is(value->type, "i32") && !ckc_i_type_is(value->type, "f32"))
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "lds_atomic_add supports i32 / f32, got %s", value->type->name);
    if(!ckc_i_ordering_ok(ordering))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "unknown ordering '%s'", ordering);
    /* operands = [smem, *indices, value] */
    ops = (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(*ops) * (size_t)(num_indices + 2));
    if(!ops)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_OOM, "lds_atomic_add: OOM");
    ops[0] = smem;
    for(i = 0; i < num_indices; ++i)
        ops[1 + i] = indices[i];
    ops[1 + num_indices] = value;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", value->type->name);
    ckc_attr_set_int(b, &a, "rank", (int64_t)num_indices);
    ckc_attr_set_str(b, &a, "ordering", ordering);
    return ckc_i_op1(
        b, CKC_OP_TILE_LDS_ATOMIC_ADD, ops, num_indices + 2, value->type, &a, "lds_atom");
}

ckc_value_t* ckc_b_global_atomic_add_pk_bf16(ckc_ir_builder_t* b,
                                             ckc_value_t* ptr,
                                             ckc_value_t* idx,
                                             ckc_value_t* value,
                                             const char* ordering)
{
    ckc_value_t* ops[3];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    if(!ptr || !idx || !value)
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "global_atomic_add_pk_bf16: null operand");
    if(ordering == NULL)
        ordering = "monotonic";
    if(!ckc_i_ordering_ok(ordering))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "unknown ordering '%s'", ordering);
    if(!ckc_i_is_vector(value->type, "bf16", 2))
        return (ckc_value_t*)ckc_i_set_err(
            b,
            CKC_ERR_VALUE,
            "global_atomic_add_pk_bf16 expects <2 x bf16> input, got %s",
            value->type->name);
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem_type", "bf16");
    ckc_attr_set_int(b, &a, "vec", 2);
    ckc_attr_set_str(b, &a, "ordering", ordering);
    return ckc_i_op1(
        b, CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD_PK_BF16, ops, 3, value->type, &a, "atom_bf16");
}

void ckc_b_global_atomic_add_f32(ckc_ir_builder_t* b,
                                 ckc_value_t* ptr,
                                 ckc_value_t* idx,
                                 ckc_value_t* value)
{
    ckc_value_t* ops[3];
    if(!ckc_i_live(b))
        return;
    if(!ptr || !idx || !value)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "global_atomic_add_f32: null operand");
        return;
    }
    ops[0] = ptr;
    ops[1] = idx;
    ops[2] = value;
    (void)ckc_i_op0(b, CKC_OP_MEMREF_GLOBAL_ATOMIC_ADD_F32, ops, 3, NULL);
}

/* ===================== vector.* element-wise family ===================== */

/* IRBuilder.vector_binary: matching vector operands -> a->type. */
static ckc_value_t* ckc_i_vector_binary(
    ckc_ir_builder_t* b, ckc_opcode_t opcode, ckc_value_t* a, ckc_value_t* c, const char* hint)
{
    ckc_value_t* ops[2];
    if(!ckc_i_live(b))
        return NULL;
    if(!a || !c)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_binary: null operand");
    if(!ckc_i_is_vector(a->type, NULL, -1) || !ckc_type_eq(a->type, c->type))
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "vector_binary expects matching vector operands");
    ops[0] = a;
    ops[1] = c;
    return ckc_i_op1(b, opcode, ops, 2, a->type, NULL, hint);
}

ckc_value_t* ckc_b_vector_add(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_ADD, a, c, "vadd");
}
ckc_value_t* ckc_b_vector_sub(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_SUB, a, c, "vsub");
}
ckc_value_t* ckc_b_vector_mul(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_MUL, a, c, "vmul");
}
ckc_value_t* ckc_b_vector_and(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_AND, a, c, "vand");
}
ckc_value_t* ckc_b_vector_or(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_OR, a, c, "vor");
}
ckc_value_t* ckc_b_vector_shl(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_SHL, a, c, "vshl");
}
ckc_value_t* ckc_b_vector_lshr(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_LSHR, a, c, "vlshr");
}
ckc_value_t* ckc_b_vector_smax(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_SMAX, a, c, "vsmax");
}
ckc_value_t* ckc_b_vector_smin(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_SMIN, a, c, "vsmin");
}
ckc_value_t* ckc_b_vector_max(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c)
{
    return ckc_i_vector_binary(b, CKC_OP_VECTOR_MAX, a, c, "vmax");
}

ckc_value_t* ckc_b_vector_fma(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, ckc_value_t* d)
{
    ckc_value_t* ops[3];
    if(!ckc_i_live(b))
        return NULL;
    if(!a || !c || !d)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_fma: null operand");
    if(!ckc_i_is_vector(a->type, NULL, -1) || !ckc_type_eq(a->type, c->type)
       || !ckc_type_eq(a->type, d->type))
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "vector_fma expects three matching vector operands");
    ops[0] = a;
    ops[1] = c;
    ops[2] = d;
    return ckc_i_op1(b, CKC_OP_VECTOR_FMA, ops, 3, a->type, NULL, "vfma");
}

ckc_value_t* ckc_b_vector_sum(ckc_ir_builder_t* b, ckc_value_t* v)
{
    ckc_value_t* ops[1];
    if(!ckc_i_live(b))
        return NULL;
    if(!v)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_sum: null operand");
    if(!ckc_i_is_vector(v->type, NULL, -1))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_sum expects vector");
    ops[0] = v;
    return ckc_i_op1(b, CKC_OP_VECTOR_SUM, ops, 1, ckc_i_elem_of(v->type), NULL, "vsum");
}

ckc_value_t* ckc_b_vector_reduce_max(ckc_ir_builder_t* b, ckc_value_t* v)
{
    ckc_value_t* ops[1];
    if(!ckc_i_live(b))
        return NULL;
    if(!v)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_reduce_max: null operand");
    if(!ckc_i_is_vector(v->type, NULL, -1))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_reduce_max expects vector");
    ops[0] = v;
    return ckc_i_op1(b, CKC_OP_VECTOR_REDUCE_MAX, ops, 1, ckc_i_elem_of(v->type), NULL, "vmax");
}

ckc_value_t* ckc_b_vector_splat(ckc_ir_builder_t* b, ckc_value_t* scalar, int n)
{
    ckc_value_t* ops[1];
    ckc_attr_map_t a;
    const ckc_type_t* vt;
    if(!ckc_i_live(b))
        return NULL;
    if(!scalar)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_splat: null operand");
    vt = ckc_vector_type(b, scalar->type, n);
    if(!vt)
        return NULL;
    ops[0] = scalar;
    a = ckc_i_attrs(b);
    ckc_attr_set_int(b, &a, "vec", (int64_t)n);
    return ckc_i_op1(b, CKC_OP_VECTOR_SPLAT, ops, 1, vt, &a, "splat");
}

ckc_value_t*
    ckc_b_vector_select(ckc_ir_builder_t* b, ckc_value_t* mask, ckc_value_t* lhs, ckc_value_t* rhs)
{
    ckc_value_t* ops[3];
    if(!ckc_i_live(b))
        return NULL;
    if(!mask || !lhs || !rhs)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_select: null operand");
    if(!ckc_type_eq(lhs->type, rhs->type))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_select lhs/rhs type mismatch");
    ops[0] = mask;
    ops[1] = lhs;
    ops[2] = rhs;
    return ckc_i_op1(b, CKC_OP_VECTOR_SELECT, ops, 3, lhs->type, NULL, "vsel");
}

ckc_value_t* ckc_b_vector_cmp(ckc_ir_builder_t* b, const char* pred, ckc_value_t* a, ckc_value_t* c)
{
    ckc_value_t* ops[2];
    ckc_attr_map_t attr;
    const ckc_type_t* vt;
    char hint[32];
    int i, j;
    if(!ckc_i_live(b))
        return NULL;
    if(!a || !c || !pred)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_cmp: null operand/pred");
    if(!ckc_i_is_vector(a->type, NULL, -1) || !ckc_type_eq(a->type, c->type))
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "vector_cmp expects matching vector operands");
    vt = ckc_vector_type(b, ckc_i1(), ckc_i_count_of(a->type));
    if(!vt)
        return NULL;
    ops[0] = a;
    ops[1] = c;
    attr = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attr, "pred", pred);
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
    return ckc_i_op1(b, CKC_OP_VECTOR_CMP, ops, 2, vt, &attr, hint);
}

/* shared "vN{count}" hint formatter for vector_trunc/sext */
static void ckc_i_count_hint(char* buf, const char* prefix, int count)
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

ckc_value_t* ckc_b_vector_trunc(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target)
{
    ckc_value_t* ops[1];
    ckc_attr_map_t a;
    const ckc_type_t* vt;
    char hint[16];
    if(!ckc_i_live(b))
        return NULL;
    if(!v || !target)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_trunc: null operand/target");
    if(!ckc_i_is_vector(v->type, NULL, -1))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_trunc expects vector input");
    vt = ckc_vector_type(b, target, ckc_i_count_of(v->type));
    if(!vt)
        return NULL;
    ops[0] = v;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "target", target->name);
    ckc_i_count_hint(hint, "vtr", ckc_i_count_of(v->type));
    return ckc_i_op1(b, CKC_OP_VECTOR_TRUNC, ops, 1, vt, &a, hint);
}

ckc_value_t* ckc_b_vector_sext(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target)
{
    ckc_value_t* ops[1];
    ckc_attr_map_t a;
    const ckc_type_t* vt;
    char hint[16];
    if(!ckc_i_live(b))
        return NULL;
    if(!v || !target)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_sext: null operand/target");
    if(!ckc_i_is_vector(v->type, NULL, -1))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vector_sext expects vector input");
    vt = ckc_vector_type(b, target, ckc_i_count_of(v->type));
    if(!vt)
        return NULL;
    ops[0] = v;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "target", target->name);
    ckc_i_count_hint(hint, "vsx", ckc_i_count_of(v->type));
    return ckc_i_op1(b, CKC_OP_VECTOR_SEXT, ops, 1, vt, &a, hint);
}

/* ===================== vec extract/insert/pack/concat =================== */

ckc_value_t* ckc_b_vec_extract(ckc_ir_builder_t* b, ckc_value_t* v, int i)
{
    ckc_value_t* ops[1];
    ckc_attr_map_t a;
    const ckc_type_t* et;
    if(!ckc_i_live(b))
        return NULL;
    if(!v)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_extract: null operand");
    et = ckc_i_elem_of(v->type); /* vec elem, or scalar type itself */
    ops[0] = v;
    a = ckc_i_attrs(b);
    ckc_attr_set_int(b, &a, "index", (int64_t)i);
    return ckc_i_op1(b, CKC_OP_VECTOR_EXTRACT, ops, 1, et, &a, "e");
}

ckc_value_t* ckc_b_vec_insert(ckc_ir_builder_t* b, ckc_value_t* v, ckc_value_t* scalar, int i)
{
    ckc_value_t* ops[2];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    if(!v || !scalar)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_insert: null operand");
    if(!ckc_i_is_vector(v->type, NULL, -1))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_insert expects vector");
    if(!ckc_type_eq(scalar->type, ckc_i_elem_of(v->type)))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_insert scalar type mismatch");
    ops[0] = v;
    ops[1] = scalar;
    a = ckc_i_attrs(b);
    ckc_attr_set_int(b, &a, "index", (int64_t)i);
    return ckc_i_op1(b, CKC_OP_VECTOR_INSERT, ops, 2, v->type, &a, "vi");
}

ckc_value_t* ckc_b_vec_pack(ckc_ir_builder_t* b,
                            ckc_value_t* const* components,
                            int num_components,
                            const ckc_type_t* elem)
{
    ckc_attr_map_t a;
    const ckc_type_t* vt;
    int i;
    if(!ckc_i_live(b))
        return NULL;
    if(!elem)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_pack: null elem");
    if(num_components <= 0 || !components)
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "vec_pack needs at least one component");
    for(i = 0; i < num_components; ++i)
    {
        if(!components[i] || !ckc_type_eq(components[i]->type, elem))
            return (ckc_value_t*)ckc_i_set_err(b,
                                               CKC_ERR_VALUE,
                                               "vec_pack expected %s, got %s",
                                               elem->name,
                                               components[i] ? components[i]->type->name
                                                             : "(null)");
    }
    vt = ckc_vector_type(b, elem, num_components);
    if(!vt)
        return NULL;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "elem", elem->name);
    ckc_attr_set_int(b, &a, "vec", (int64_t)num_components);
    return ckc_i_op1(b, CKC_OP_VECTOR_PACK, components, num_components, vt, &a, "vp");
}

ckc_value_t* ckc_b_vec_concat(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* bb)
{
    ckc_value_t* ops[2];
    ckc_attr_map_t attr;
    const ckc_type_t *elem, *vt;
    int n;
    if(!ckc_i_live(b))
        return NULL;
    if(!a || !bb)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_concat: null operand");
    if(!ckc_i_is_vector(a->type, NULL, -1) || !ckc_i_is_vector(bb->type, NULL, -1))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_concat needs vector inputs");
    elem = ckc_i_elem_of(a->type);
    if(!ckc_type_eq(elem, ckc_i_elem_of(bb->type)))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_concat element types must match");
    n = ckc_i_count_of(a->type) + ckc_i_count_of(bb->type);
    vt = ckc_vector_type(b, elem, n);
    if(!vt)
        return NULL;
    ops[0] = a;
    ops[1] = bb;
    attr = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attr, "elem", elem->name);
    ckc_attr_set_int(b, &attr, "vec", (int64_t)n);
    return ckc_i_op1(b, CKC_OP_VECTOR_CONCAT, ops, 2, vt, &attr, "vc");
}

/* =================== vec bitcast / packed f32->fXX ===================== */

ckc_value_t* ckc_b_vec_bitcast(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target)
{
    ckc_value_t* ops[1];
    ckc_attr_map_t a;
    if(!ckc_i_live(b))
        return NULL;
    if(!v || !target)
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_bitcast: null operand/target");
    ops[0] = v;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "target", target->name);
    return ckc_i_op1(b, CKC_OP_VECTOR_BITCAST, ops, 1, target, &a, "bc");
}

ckc_value_t* ckc_b_vec_cast_f32_to(ckc_ir_builder_t* b, ckc_value_t* v, const ckc_type_t* target)
{
    ckc_value_t* ops[1];
    ckc_attr_map_t a;
    const ckc_type_t* vt;
    char hint[16];
    if(!ckc_i_live(b))
        return NULL;
    if(!v || !target)
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "vec_cast_f32_to: null operand/target");
    if(!ckc_i_is_vector(v->type, "f32", -1))
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "vec_cast_f32_to expects <N x f32>");
    if(!ckc_i_type_is(target, "f16") && !ckc_i_type_is(target, "bf16"))
        return (ckc_value_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "vec_cast_f32_to unsupported target %s", target->name);
    vt = ckc_vector_type(b, target, ckc_i_count_of(v->type));
    if(!vt)
        return NULL;
    ops[0] = v;
    a = ckc_i_attrs(b);
    ckc_attr_set_str(b, &a, "target", target->name);
    ckc_i_count_hint(hint, "vh", ckc_i_count_of(v->type));
    return ckc_i_op1(b, CKC_OP_VECTOR_TRUNC_F32_TO, ops, 1, vt, &a, hint);
}

ckc_value_t* ckc_b_vec_trunc_f32_to_f16(ckc_ir_builder_t* b, ckc_value_t* v)
{
    return ckc_b_vec_cast_f32_to(b, v, ckc_f16());
}
