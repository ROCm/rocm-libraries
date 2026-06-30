// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ir_ir_tile.c -- C99 port of the "ir_tile" bucket of rocke.core.ir.
 *
 * Covers tile.* LDS memory (smem alloc/store/load incl. f32 + distributed),
 * the target-neutral tile.mma plus all ISA-named MFMA/WMMA wrappers,
 * tile.inline_asm (single + multi), and register_p_from_qk_c /
 * cooperative_global_store.
 *
 * Binds to the FROZEN IR contract in rocke/ir.h; shared plumbing
 * (rocke_i_op / rocke_i_op1 / rocke_i_op0 / rocke_i_set_err / rocke_i_live /
 * rocke_i_attrs / type helpers) lives in bucket 0 (ir_core.c) and is declared
 * in rocke/ir_internal.h.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/ir_internal.h"

/* ===================================================================== */
/*  target-neutral MMA metadata (mirrors the module-level tables in       */
/*  ir.py: _MMA_C_FRAG_LEN, _MMA_C_INT_OP_IDS, _MMA_RESULT_HINT).         */
/* ===================================================================== */

typedef struct
{
    const char* op_id;
    int c_frag_len;
} rocke_mma_frag_row_t;

/* op_id -> per-lane accumulator/result fragment length. */
static const rocke_mma_frag_row_t ROCKE_MMA_C_FRAG_LEN[] = {
    {"mfma_f32_16x16x4_f32", 4},
    {"mfma_f32_32x32x2_f32", 16},
    {"mfma_f32_16x16x16_f16", 4},
    {"mfma_f32_16x16x32_f16", 4},
    {"mfma_f32_16x16x16_bf16", 4},
    {"mfma_f32_16x16x32_bf16", 4},
    {"mfma_f32_16x16x32_fp8", 4},
    {"mfma_f32_16x16x32_bf8", 4},
    {"mfma_f32_32x32x8_f16", 16},
    {"mfma_f32_32x32x16_f16", 16},
    {"mfma_f32_32x32x8_bf16", 16},
    {"mfma_f32_32x32x16_bf16", 16},
    {"mfma_f32_32x32x16_fp8", 16},
    {"mfma_f32_32x32x16_bf8", 16},
    {"mfma_f32_4x4x4_f16", 4},
    {"mfma_f32_16x16x128_fp4", 4},
    {"mfma_f32_16x16x96_fp6", 4},
    {"mfma_f32_16x16x128_fp8", 4},
    {"mfma_scale_f32_16x16x128_f8f6f4", 4},
    {"wmma_f32_16x16x16_f16", 8},
    {"wmma_f32_16x16x16_bf16", 8},
    {"wmma_i32_16x16x16_iu8", 8},
    {"wmma_i32_16x16x16_iu4", 8},
    {"wmma_gfx12_f32_16x16x16_f16", 8},
    {"wmma_gfx12_f32_16x16x16_bf16", 8},
};

/* op_ids that accumulate in i32 (integer WMMA atoms). */
static const char* const ROCKE_MMA_C_INT_OP_IDS[] = {
    "wmma_i32_16x16x16_iu8",
    "wmma_i32_16x16x16_iu4",
};

/* op_id -> result_name_hint override (default "acc"). */
typedef struct
{
    const char* op_id;
    const char* hint;
} rocke_mma_hint_row_t;

static const rocke_mma_hint_row_t ROCKE_MMA_RESULT_HINT[] = {
    {"mfma_f32_32x32x16_bf16", "acc32"},
    {"mfma_f32_16x16x128_fp4", "acc4"},
    {"mfma_f32_16x16x96_fp6", "acc6"},
    {"mfma_f32_16x16x128_fp8", "acc128"},
    {"mfma_scale_f32_16x16x128_f8f6f4", "mxacc"},
};

/* Returns the c_frag_len for op_id, or -1 if unknown. */
static int rocke_mma_c_frag_len(const char* op_id)
{
    size_t i;
    if(!op_id)
    {
        return -1;
    }
    for(i = 0; i < sizeof(ROCKE_MMA_C_FRAG_LEN) / sizeof(ROCKE_MMA_C_FRAG_LEN[0]); ++i)
    {
        if(strcmp(ROCKE_MMA_C_FRAG_LEN[i].op_id, op_id) == 0)
        {
            return ROCKE_MMA_C_FRAG_LEN[i].c_frag_len;
        }
    }
    return -1;
}

static bool rocke_mma_is_int_acc(const char* op_id)
{
    size_t i;
    if(!op_id)
    {
        return false;
    }
    for(i = 0; i < sizeof(ROCKE_MMA_C_INT_OP_IDS) / sizeof(ROCKE_MMA_C_INT_OP_IDS[0]); ++i)
    {
        if(strcmp(ROCKE_MMA_C_INT_OP_IDS[i], op_id) == 0)
        {
            return true;
        }
    }
    return false;
}

static const char* rocke_mma_result_hint(const char* op_id)
{
    size_t i;
    if(op_id)
    {
        for(i = 0; i < sizeof(ROCKE_MMA_RESULT_HINT) / sizeof(ROCKE_MMA_RESULT_HINT[0]); ++i)
        {
            if(strcmp(ROCKE_MMA_RESULT_HINT[i].op_id, op_id) == 0)
            {
                return ROCKE_MMA_RESULT_HINT[i].hint;
            }
        }
    }
    return "acc";
}

/* element-byte width helper (mirrors smem_store_vN's inline ternary). */
static int rocke_elem_bytes_name(const char* elem_name)
{
    if(!elem_name)
    {
        return 2;
    }
    if(strcmp(elem_name, "i8") == 0 || strcmp(elem_name, "fp8e4m3") == 0
       || strcmp(elem_name, "bf8e5m2") == 0)
    {
        return 1;
    }
    if(strcmp(elem_name, "f32") == 0 || strcmp(elem_name, "i32") == 0)
    {
        return 4;
    }
    return 2;
}

static bool rocke_n_in(int n, const int* allowed, int count)
{
    int i;
    for(i = 0; i < count; ++i)
    {
        if(allowed[i] == n)
        {
            return true;
        }
    }
    return false;
}

/* Build the [smem, *indices, value] operand array in the arena. Returns NULL on
 * OOM (sticky error set). With_value=false omits the trailing value. */
static rocke_value_t** rocke_build_mem_operands(rocke_ir_builder_t* b,
                                                rocke_value_t* smem,
                                                rocke_value_t* const* indices,
                                                int num_indices,
                                                rocke_value_t* value,
                                                bool with_value,
                                                int* out_count)
{
    int extra = with_value ? 1 : 0;
    int n = 1 + num_indices + extra;
    rocke_value_t** ops;
    int i;
    ops = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(*ops));
    if(!ops)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "smem operand array alloc failed");
        return NULL;
    }
    ops[0] = smem;
    for(i = 0; i < num_indices; ++i)
    {
        ops[1 + i] = indices[i];
    }
    if(with_value)
    {
        ops[1 + num_indices] = value;
    }
    *out_count = n;
    return ops;
}

/* ===================================================================== */
/*  LDS (shared memory) -- alloc                                          */
/* ===================================================================== */

rocke_value_t* rocke_b_smem_alloc(rocke_ir_builder_t* b,
                                  const rocke_type_t* elem,
                                  const int* shape,
                                  int rank,
                                  const char* name_hint)
{
    const rocke_type_t* t;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    t = rocke_smem_type(b, elem, shape, rank);
    if(!t)
    {
        return NULL;
    }
    return rocke_i_op1(
        b, ROCKE_OP_TILE_SMEM_ALLOC, NULL, 0, t, NULL, name_hint ? name_hint : "smem");
}

/* ===================================================================== */
/*  LDS stores                                                            */
/* ===================================================================== */

void rocke_b_smem_store_f16(rocke_ir_builder_t* b,
                            rocke_value_t* smem,
                            rocke_value_t* const* indices,
                            int num_indices,
                            rocke_value_t* value)
{
    rocke_value_t** ops;
    int nops = 0;
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
    {
        return;
    }
    ops = rocke_build_mem_operands(b, smem, indices, num_indices, value, true, &nops);
    if(!ops)
    {
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    rocke_attr_set_str(b, &attrs, "elem_type", "f16");
    rocke_i_op0(b, ROCKE_OP_TILE_SMEM_STORE, ops, nops, &attrs);
}

void rocke_b_smem_store_vN(rocke_ir_builder_t* b,
                           rocke_value_t* smem,
                           rocke_value_t* const* indices,
                           int num_indices,
                           rocke_value_t* value,
                           int n)
{
    rocke_value_t** ops;
    int nops = 0;
    rocke_attr_map_t attrs;
    const char* elem_name;
    static const int allowed_8bit[] = {2, 4, 8, 16};
    static const int allowed_other[] = {2, 4, 8};
    int elem_bytes;
    if(!rocke_i_live(b))
    {
        return;
    }
    if(n == 1)
    {
        /* Single-element store; route through scalar tile.smem_store. */
        if(!value)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "smem_store_vN value is NULL");
            return;
        }
        ops = rocke_build_mem_operands(b, smem, indices, num_indices, value, true, &nops);
        if(!ops)
        {
            return;
        }
        attrs = rocke_i_attrs(b);
        rocke_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
        rocke_attr_set_str(b, &attrs, "elem_type", value->type->name);
        rocke_i_op0(b, ROCKE_OP_TILE_SMEM_STORE, ops, nops, &attrs);
        return;
    }
    if(!value || !rocke_i_is_vector(value->type, NULL, -1))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "smem_store_vN expects vector value for n > 1");
        return;
    }
    elem_name = value->type->elem->name;
    {
        bool eight = (strcmp(elem_name, "i8") == 0 || strcmp(elem_name, "fp8e4m3") == 0
                      || strcmp(elem_name, "bf8e5m2") == 0);
        const int* allowed = eight ? allowed_8bit : allowed_other;
        int acount = eight ? 4 : 3;
        if(!rocke_n_in(n, allowed, acount))
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "unsupported vector width for smem_store_vN of %s: %d",
                            elem_name,
                            n);
            return;
        }
    }
    elem_bytes = rocke_elem_bytes_name(elem_name);
    ops = rocke_build_mem_operands(b, smem, indices, num_indices, value, true, &nops);
    if(!ops)
    {
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    rocke_attr_set_str(b, &attrs, "elem_type", elem_name);
    rocke_attr_set_int(b, &attrs, "vec", (int64_t)n);
    rocke_attr_set_int(b, &attrs, "align", (int64_t)(n * elem_bytes));
    rocke_i_op0(b, ROCKE_OP_TILE_SMEM_STORE_VN, ops, nops, &attrs);
}

void rocke_b_smem_store_vN_f16(rocke_ir_builder_t* b,
                               rocke_value_t* smem,
                               rocke_value_t* const* indices,
                               int num_indices,
                               rocke_value_t* value,
                               int n)
{
    /* Thin wrapper over smem_store_vN (Python: same). */
    rocke_b_smem_store_vN(b, smem, indices, num_indices, value, n);
}

/* ===================================================================== */
/*  LDS loads                                                             */
/* ===================================================================== */

rocke_value_t* rocke_b_smem_load_v4_f16(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* row,
                                        rocke_value_t* col)
{
    rocke_value_t* ops[3];
    const rocke_type_t* vt;
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    ops[0] = smem;
    ops[1] = row;
    ops[2] = col;
    vt = rocke_vector_type(b, rocke_f16(), 4);
    if(!vt)
    {
        return NULL;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "elem_type", "f16");
    return rocke_i_op1(b, ROCKE_OP_TILE_SMEM_LOAD_V4, ops, 3, vt, &attrs, "a");
}

rocke_value_t* rocke_b_smem_load_vN(rocke_ir_builder_t* b,
                                    rocke_value_t* smem,
                                    rocke_value_t* const* indices,
                                    int num_indices,
                                    const rocke_type_t* dtype,
                                    int n)
{
    rocke_value_t** ops;
    int nops = 0;
    const rocke_type_t* vt;
    rocke_attr_map_t attrs;
    const char* dn;
    static const int allowed_8bit[] = {1, 2, 4, 8, 16};
    static const int allowed_other[] = {1, 2, 4, 8};
    char hint[16];
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(!dtype)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "smem_load_vN dtype is NULL");
    }
    dn = dtype->name;
    if(!(strcmp(dn, "f16") == 0 || strcmp(dn, "bf16") == 0 || strcmp(dn, "f32") == 0
         || strcmp(dn, "i32") == 0 || strcmp(dn, "fp8e4m3") == 0 || strcmp(dn, "bf8e5m2") == 0
         || strcmp(dn, "i8") == 0))
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "smem_load_vN supports f16 / bf16 / f32 / i32 / fp8e4m3 / "
            "bf8e5m2 / i8, got %s",
            dn);
    }
    {
        bool eight
            = (strcmp(dn, "fp8e4m3") == 0 || strcmp(dn, "bf8e5m2") == 0 || strcmp(dn, "i8") == 0);
        const int* allowed = eight ? allowed_8bit : allowed_other;
        int acount = eight ? 5 : 4;
        if(!rocke_n_in(n, allowed, acount))
        {
            return (rocke_value_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "unsupported vector width %d for smem_load_vN of %s", n, dn);
        }
    }
    if(num_indices <= 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "smem_load_vN needs at least one index");
    }
    vt = rocke_vector_type(b, dtype, n);
    if(!vt)
    {
        return NULL;
    }
    ops = rocke_build_mem_operands(b, smem, indices, num_indices, NULL, false, &nops);
    if(!ops)
    {
        return NULL;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "elem_type", dn);
    rocke_attr_set_int(b, &attrs, "vec", (int64_t)n);
    rocke_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    snprintf(hint, sizeof(hint), "av%d", n);
    return rocke_i_op1(b, ROCKE_OP_TILE_SMEM_LOAD_VN, ops, nops, vt, &attrs, hint);
}

rocke_value_t* rocke_b_smem_load_vN_f16(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* const* indices,
                                        int num_indices,
                                        int n)
{
    return rocke_b_smem_load_vN(b, smem, indices, num_indices, rocke_f16(), n);
}

/* ===================================================================== */
/*  f32 LDS ops (cshuffle epilogue)                                       */
/* ===================================================================== */

rocke_value_t*
    rocke_b_smem_alloc_f32(rocke_ir_builder_t* b, const int* shape, int rank, const char* name_hint)
{
    return rocke_b_smem_alloc(b, rocke_f32(), shape, rank, name_hint ? name_hint : "smem_f32");
}

void rocke_b_smem_store_vN_f32(rocke_ir_builder_t* b,
                               rocke_value_t* smem,
                               rocke_value_t* const* indices,
                               int num_indices,
                               rocke_value_t* value,
                               int n)
{
    rocke_value_t** ops;
    int nops = 0;
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
    {
        return;
    }
    if(!(n == 1 || n == 2 || n == 4))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "smem_store_vN_f32 n must be 1, 2, or 4 (got %d)", n);
        return;
    }
    ops = rocke_build_mem_operands(b, smem, indices, num_indices, value, true, &nops);
    if(!ops)
    {
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    rocke_attr_set_str(b, &attrs, "elem_type", "f32");
    rocke_attr_set_int(b, &attrs, "vec", (int64_t)n);
    rocke_i_op0(b, ROCKE_OP_TILE_SMEM_STORE_VN_F32, ops, nops, &attrs);
}

rocke_value_t* rocke_b_smem_load_vN_f32(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* const* indices,
                                        int num_indices,
                                        int n)
{
    rocke_value_t** ops;
    int nops = 0;
    const rocke_type_t* vt;
    rocke_attr_map_t attrs;
    char hint[16];
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(!(n == 1 || n == 2 || n == 4))
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "smem_load_vN_f32 n must be 1, 2, or 4 (got %d)", n);
    }
    if(num_indices <= 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "smem_load_vN_f32 needs at least one index");
    }
    vt = rocke_vector_type(b, rocke_f32(), n);
    if(!vt)
    {
        return NULL;
    }
    ops = rocke_build_mem_operands(b, smem, indices, num_indices, NULL, false, &nops);
    if(!ops)
    {
        return NULL;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "elem_type", "f32");
    rocke_attr_set_int(b, &attrs, "vec", (int64_t)n);
    rocke_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    snprintf(hint, sizeof(hint), "av%df32", n);
    return rocke_i_op1(b, ROCKE_OP_TILE_SMEM_LOAD_VN_F32, ops, nops, vt, &attrs, hint);
}

/* ===================================================================== */
/*  distributed / cooperative epilogue stores                            */
/* ===================================================================== */

void rocke_b_smem_store_distributed(rocke_ir_builder_t* b,
                                    rocke_value_t* smem,
                                    const rocke_attr_map_t* layout_attrs,
                                    rocke_value_t* values)
{
    rocke_value_t* ops[2];
    if(!rocke_i_live(b))
    {
        return;
    }
    ops[0] = smem;
    ops[1] = values;
    /* attrs = dict(layout_attrs); rocke_i_op0 deep-copies the passed map. */
    rocke_i_op0(b, ROCKE_OP_TILE_SMEM_STORE_DISTRIBUTED, ops, 2, layout_attrs);
}

void rocke_b_cooperative_global_store(rocke_ir_builder_t* b,
                                      rocke_value_t* ptr,
                                      rocke_value_t* addrs,
                                      rocke_value_t* values)
{
    rocke_value_t* ops[3];
    rocke_attr_map_t attrs;
    int64_t vec;
    if(!rocke_i_live(b))
    {
        return;
    }
    ops[0] = ptr;
    ops[1] = addrs;
    ops[2] = values;
    vec = (values && rocke_i_is_vector(values->type, NULL, -1)) ? (int64_t)values->type->count : 1;
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "vec", vec);
    rocke_i_op0(b, ROCKE_OP_MEMREF_COOPERATIVE_GLOBAL_STORE, ops, 3, &attrs);
}

/* ===================================================================== */
/*  target-neutral MMA                                                    */
/* ===================================================================== */

rocke_value_t* rocke_b_mma(rocke_ir_builder_t* b,
                           const char* op_id,
                           rocke_value_t* a,
                           rocke_value_t* bb,
                           rocke_value_t* c,
                           rocke_value_t* const* extra,
                           int num_extra)
{
    int c_frag_len;
    bool is_int_acc;
    const rocke_type_t* c_elem;
    const rocke_type_t* vt;
    const char* hint;
    rocke_attr_map_t attrs;
    rocke_value_t** ops;
    int nops;
    int i;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(!op_id)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "mma op_id is NULL");
    }
    /* C has no MmaOp object; op_id is always a bare string, so the frag length
     * and accumulator element come from the static op_id table (the Python
     * bare-string code path). */
    c_frag_len = rocke_mma_c_frag_len(op_id);
    if(c_frag_len < 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unknown MMA op_id '%s'; pass a known mfma_*/wmma_* op_id", op_id);
    }
    is_int_acc = rocke_mma_is_int_acc(op_id);
    c_elem = is_int_acc ? rocke_i32() : rocke_f32();
    vt = rocke_vector_type(b, c_elem, c_frag_len);
    if(!vt)
    {
        return NULL;
    }
    hint = rocke_mma_result_hint(op_id);

    if(num_extra < 0)
    {
        num_extra = 0;
    }
    nops = 3 + num_extra;
    ops = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)nops * sizeof(*ops));
    if(!ops)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "mma operand array alloc failed");
    }
    ops[0] = a;
    ops[1] = bb;
    ops[2] = c;
    for(i = 0; i < num_extra; ++i)
    {
        ops[3 + i] = extra[i];
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "op_id", op_id);
    return rocke_i_op1(b, ROCKE_OP_TILE_MMA, ops, nops, vt, &attrs, hint);
}

/* ----- ISA-named MMA wrappers (thin wrappers over rocke_b_mma) ----- */

#define ROCKE_MMA_WRAP(fn, opid)                                                      \
    rocke_value_t* fn(                                                                \
        rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb, rocke_value_t* c) \
    {                                                                                 \
        return rocke_b_mma(b, opid, a, bb, c, NULL, 0);                               \
    }

ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x16_f16, "mfma_f32_16x16x16_f16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x32_f16, "mfma_f32_16x16x32_f16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x16_bf16, "mfma_f32_16x16x16_bf16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x32_bf16, "mfma_f32_16x16x32_bf16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x32_fp8, "mfma_f32_16x16x32_fp8")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x32_bf8, "mfma_f32_16x16x32_bf8")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_32x32x8_f16, "mfma_f32_32x32x8_f16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_32x32x8_bf16, "mfma_f32_32x32x8_bf16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_32x32x16_f16, "mfma_f32_32x32x16_f16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_32x32x16_bf16, "mfma_f32_32x32x16_bf16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_32x32x16_fp8, "mfma_f32_32x32x16_fp8")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_32x32x16_bf8, "mfma_f32_32x32x16_bf8")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_4x4x4_f16, "mfma_f32_4x4x4_f16")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x128_fp4, "mfma_f32_16x16x128_fp4")
ROCKE_MMA_WRAP(rocke_b_mfma_f32_16x16x96_fp6, "mfma_f32_16x16x96_fp6")
ROCKE_MMA_WRAP(rocke_b_wmma_f32_16x16x16_f16, "wmma_f32_16x16x16_f16")
ROCKE_MMA_WRAP(rocke_b_wmma_f32_16x16x16_bf16, "wmma_f32_16x16x16_bf16")
ROCKE_MMA_WRAP(rocke_b_wmma_gfx12_f32_16x16x16_f16, "wmma_gfx12_f32_16x16x16_f16")
ROCKE_MMA_WRAP(rocke_b_wmma_gfx12_f32_16x16x16_bf16, "wmma_gfx12_f32_16x16x16_bf16")

#undef ROCKE_MMA_WRAP

rocke_value_t* rocke_b_mfma_scale_f32_16x16x128_f8f6f4(rocke_ir_builder_t* b,
                                                       rocke_value_t* a,
                                                       rocke_value_t* bb,
                                                       rocke_value_t* c,
                                                       rocke_value_t* a_scale,
                                                       rocke_value_t* b_scale)
{
    rocke_value_t* extra[2];
    extra[0] = a_scale;
    extra[1] = b_scale;
    return rocke_b_mma(b, "mfma_scale_f32_16x16x128_f8f6f4", a, bb, c, extra, 2);
}

/* ===================================================================== */
/*  register-fragment reshape (P13)                                       */
/* ===================================================================== */

rocke_value_t* rocke_b_register_p_from_qk_c(rocke_ir_builder_t* b,
                                            rocke_value_t* qk_c,
                                            const rocke_type_t* target_dtype)
{
    rocke_value_t* ops[1];
    const rocke_type_t* vt;
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(!target_dtype
       || !(strcmp(target_dtype->name, "f16") == 0 || strcmp(target_dtype->name, "bf16") == 0))
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "register_p_from_qk_c target must be f16/bf16, got %s",
            target_dtype ? target_dtype->name : "(null)");
    }
    ops[0] = qk_c;
    vt = rocke_vector_type(b, target_dtype, 8);
    if(!vt)
    {
        return NULL;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "target_dtype", target_dtype->name);
    return rocke_i_op1(b, ROCKE_OP_TILE_REGISTER_P_FROM_QK_C, ops, 1, vt, &attrs, "pa");
}

/* ===================================================================== */
/*  inline asm                                                           */
/* ===================================================================== */

rocke_op_t* rocke_b_inline_asm(rocke_ir_builder_t* b,
                               const char* asm_template,
                               const char* constraints,
                               rocke_value_t* const* operands,
                               int num_operands,
                               const rocke_type_t* const* result_types,
                               int num_results,
                               const rocke_inline_asm_opts_t* opts)
{
    rocke_attr_map_t attrs;
    bool sideeffect = true; /* Python default */
    bool convergent = false; /* Python default */
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(opts)
    {
        if(opts->sideeffect_set)
        {
            sideeffect = opts->sideeffect;
        }
        if(opts->convergent_set)
        {
            convergent = opts->convergent;
        }
    }
    if(num_operands < 0)
    {
        num_operands = 0;
    }
    if(num_results < 0)
    {
        num_results = 0;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "template", asm_template ? asm_template : "");
    rocke_attr_set_str(b, &attrs, "constraints", constraints ? constraints : "");
    rocke_attr_set_bool(b, &attrs, "sideeffect", sideeffect);
    rocke_attr_set_bool(b, &attrs, "convergent", convergent);
    return rocke_i_op(b,
                      ROCKE_OP_TILE_INLINE_ASM,
                      operands,
                      num_operands,
                      result_types,
                      num_results,
                      &attrs,
                      NULL,
                      0,
                      "asm",
                      NULL);
}

rocke_op_t* rocke_b_inline_asm_multi(rocke_ir_builder_t* b,
                                     const char* asm_template,
                                     const char* constraints,
                                     rocke_value_t* const* operands,
                                     int num_operands,
                                     const rocke_type_t* const* result_types,
                                     int num_results,
                                     const rocke_inline_asm_opts_t* opts)
{
    /* Python: <=1 result types delegates to inline_asm; >1 emits a single
     * tile.inline_asm op with all N result types (literal-struct return). Both
     * paths funnel through the same op builder here, so a single call with the
     * given result_types reproduces the emission for any N. */
    return rocke_b_inline_asm(
        b, asm_template, constraints, operands, num_operands, result_types, num_results, opts);
}
