/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ir_ir_tile.c -- C99 port of the "ir_tile" bucket of ck_dsl.core.ir.
 *
 * Covers tile.* LDS memory (smem alloc/store/load incl. f32 + distributed),
 * the target-neutral tile.mma plus all ISA-named MFMA/WMMA wrappers,
 * tile.inline_asm (single + multi), and register_p_from_qk_c /
 * cooperative_global_store.
 *
 * Binds to the FROZEN IR contract in ckc/ir.h; shared plumbing
 * (ckc_i_op / ckc_i_op1 / ckc_i_op0 / ckc_i_set_err / ckc_i_live /
 * ckc_i_attrs / type helpers) lives in bucket 0 (ir_core.c) and is declared
 * in ckc/ir_internal.h.
 */
#include <stdio.h>
#include <string.h>

#include "ckc/ir_internal.h"

/* ===================================================================== */
/*  target-neutral MMA metadata (mirrors the module-level tables in       */
/*  ir.py: _MMA_C_FRAG_LEN, _MMA_C_INT_OP_IDS, _MMA_RESULT_HINT).         */
/* ===================================================================== */

typedef struct {
    const char *op_id;
    int         c_frag_len;
} ckc_mma_frag_row_t;

/* op_id -> per-lane accumulator/result fragment length. */
static const ckc_mma_frag_row_t CKC_MMA_C_FRAG_LEN[] = {
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
static const char *const CKC_MMA_C_INT_OP_IDS[] = {
    "wmma_i32_16x16x16_iu8",
    "wmma_i32_16x16x16_iu4",
};

/* op_id -> result_name_hint override (default "acc"). */
typedef struct {
    const char *op_id;
    const char *hint;
} ckc_mma_hint_row_t;

static const ckc_mma_hint_row_t CKC_MMA_RESULT_HINT[] = {
    {"mfma_f32_32x32x16_bf16", "acc32"},
    {"mfma_f32_16x16x128_fp4", "acc4"},
    {"mfma_f32_16x16x96_fp6", "acc6"},
    {"mfma_f32_16x16x128_fp8", "acc128"},
    {"mfma_scale_f32_16x16x128_f8f6f4", "mxacc"},
};

/* Returns the c_frag_len for op_id, or -1 if unknown. */
static int ckc_mma_c_frag_len(const char *op_id) {
    size_t i;
    if (!op_id) {
        return -1;
    }
    for (i = 0; i < sizeof(CKC_MMA_C_FRAG_LEN) / sizeof(CKC_MMA_C_FRAG_LEN[0]);
         ++i) {
        if (strcmp(CKC_MMA_C_FRAG_LEN[i].op_id, op_id) == 0) {
            return CKC_MMA_C_FRAG_LEN[i].c_frag_len;
        }
    }
    return -1;
}

static bool ckc_mma_is_int_acc(const char *op_id) {
    size_t i;
    if (!op_id) {
        return false;
    }
    for (i = 0;
         i < sizeof(CKC_MMA_C_INT_OP_IDS) / sizeof(CKC_MMA_C_INT_OP_IDS[0]);
         ++i) {
        if (strcmp(CKC_MMA_C_INT_OP_IDS[i], op_id) == 0) {
            return true;
        }
    }
    return false;
}

static const char *ckc_mma_result_hint(const char *op_id) {
    size_t i;
    if (op_id) {
        for (i = 0;
             i < sizeof(CKC_MMA_RESULT_HINT) / sizeof(CKC_MMA_RESULT_HINT[0]);
             ++i) {
            if (strcmp(CKC_MMA_RESULT_HINT[i].op_id, op_id) == 0) {
                return CKC_MMA_RESULT_HINT[i].hint;
            }
        }
    }
    return "acc";
}

/* element-byte width helper (mirrors smem_store_vN's inline ternary). */
static int ckc_elem_bytes_name(const char *elem_name) {
    if (!elem_name) {
        return 2;
    }
    if (strcmp(elem_name, "i8") == 0 || strcmp(elem_name, "fp8e4m3") == 0 ||
        strcmp(elem_name, "bf8e5m2") == 0) {
        return 1;
    }
    if (strcmp(elem_name, "f32") == 0 || strcmp(elem_name, "i32") == 0) {
        return 4;
    }
    return 2;
}

static bool ckc_n_in(int n, const int *allowed, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        if (allowed[i] == n) {
            return true;
        }
    }
    return false;
}

/* Build the [smem, *indices, value] operand array in the arena. Returns NULL on
 * OOM (sticky error set). With_value=false omits the trailing value. */
static ckc_value_t **ckc_build_mem_operands(ckc_ir_builder_t *b,
                                            ckc_value_t *smem,
                                            ckc_value_t *const *indices,
                                            int num_indices,
                                            ckc_value_t *value, bool with_value,
                                            int *out_count) {
    int extra = with_value ? 1 : 0;
    int n = 1 + num_indices + extra;
    ckc_value_t **ops;
    int i;
    ops = (ckc_value_t **)ckc_arena_alloc(&b->arena,
                                          (size_t)n * sizeof(*ops));
    if (!ops) {
        ckc_i_set_err(b, CKC_ERR_OOM, "smem operand array alloc failed");
        return NULL;
    }
    ops[0] = smem;
    for (i = 0; i < num_indices; ++i) {
        ops[1 + i] = indices[i];
    }
    if (with_value) {
        ops[1 + num_indices] = value;
    }
    *out_count = n;
    return ops;
}

/* ===================================================================== */
/*  LDS (shared memory) -- alloc                                          */
/* ===================================================================== */

ckc_value_t *ckc_b_smem_alloc(ckc_ir_builder_t *b, const ckc_type_t *elem,
                              const int *shape, int rank,
                              const char *name_hint) {
    const ckc_type_t *t;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    t = ckc_smem_type(b, elem, shape, rank);
    if (!t) {
        return NULL;
    }
    return ckc_i_op1(b, CKC_OP_TILE_SMEM_ALLOC, NULL, 0, t, NULL,
                     name_hint ? name_hint : "smem");
}

/* ===================================================================== */
/*  LDS stores                                                            */
/* ===================================================================== */

void ckc_b_smem_store_f16(ckc_ir_builder_t *b, ckc_value_t *smem,
                          ckc_value_t *const *indices, int num_indices,
                          ckc_value_t *value) {
    ckc_value_t **ops;
    int nops = 0;
    ckc_attr_map_t attrs;
    if (!ckc_i_live(b)) {
        return;
    }
    ops = ckc_build_mem_operands(b, smem, indices, num_indices, value, true,
                                 &nops);
    if (!ops) {
        return;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    ckc_attr_set_str(b, &attrs, "elem_type", "f16");
    ckc_i_op0(b, CKC_OP_TILE_SMEM_STORE, ops, nops, &attrs);
}

void ckc_b_smem_store_vN(ckc_ir_builder_t *b, ckc_value_t *smem,
                         ckc_value_t *const *indices, int num_indices,
                         ckc_value_t *value, int n) {
    ckc_value_t **ops;
    int nops = 0;
    ckc_attr_map_t attrs;
    const char *elem_name;
    static const int allowed_8bit[] = {2, 4, 8, 16};
    static const int allowed_other[] = {2, 4, 8};
    int elem_bytes;
    if (!ckc_i_live(b)) {
        return;
    }
    if (n == 1) {
        /* Single-element store; route through scalar tile.smem_store. */
        if (!value) {
            ckc_i_set_err(b, CKC_ERR_VALUE, "smem_store_vN value is NULL");
            return;
        }
        ops = ckc_build_mem_operands(b, smem, indices, num_indices, value, true,
                                     &nops);
        if (!ops) {
            return;
        }
        attrs = ckc_i_attrs(b);
        ckc_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
        ckc_attr_set_str(b, &attrs, "elem_type", value->type->name);
        ckc_i_op0(b, CKC_OP_TILE_SMEM_STORE, ops, nops, &attrs);
        return;
    }
    if (!value || !ckc_i_is_vector(value->type, NULL, -1)) {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "smem_store_vN expects vector value for n > 1");
        return;
    }
    elem_name = value->type->elem->name;
    {
        bool eight = (strcmp(elem_name, "i8") == 0 ||
                      strcmp(elem_name, "fp8e4m3") == 0 ||
                      strcmp(elem_name, "bf8e5m2") == 0);
        const int *allowed = eight ? allowed_8bit : allowed_other;
        int acount = eight ? 4 : 3;
        if (!ckc_n_in(n, allowed, acount)) {
            ckc_i_set_err(b, CKC_ERR_VALUE,
                          "unsupported vector width for smem_store_vN of %s: %d",
                          elem_name, n);
            return;
        }
    }
    elem_bytes = ckc_elem_bytes_name(elem_name);
    ops = ckc_build_mem_operands(b, smem, indices, num_indices, value, true,
                                 &nops);
    if (!ops) {
        return;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    ckc_attr_set_str(b, &attrs, "elem_type", elem_name);
    ckc_attr_set_int(b, &attrs, "vec", (int64_t)n);
    ckc_attr_set_int(b, &attrs, "align", (int64_t)(n * elem_bytes));
    ckc_i_op0(b, CKC_OP_TILE_SMEM_STORE_VN, ops, nops, &attrs);
}

void ckc_b_smem_store_vN_f16(ckc_ir_builder_t *b, ckc_value_t *smem,
                             ckc_value_t *const *indices, int num_indices,
                             ckc_value_t *value, int n) {
    /* Thin wrapper over smem_store_vN (Python: same). */
    ckc_b_smem_store_vN(b, smem, indices, num_indices, value, n);
}

/* ===================================================================== */
/*  LDS loads                                                             */
/* ===================================================================== */

ckc_value_t *ckc_b_smem_load_v4_f16(ckc_ir_builder_t *b, ckc_value_t *smem,
                                    ckc_value_t *row, ckc_value_t *col) {
    ckc_value_t *ops[3];
    const ckc_type_t *vt;
    ckc_attr_map_t attrs;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    ops[0] = smem;
    ops[1] = row;
    ops[2] = col;
    vt = ckc_vector_type(b, ckc_f16(), 4);
    if (!vt) {
        return NULL;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attrs, "elem_type", "f16");
    return ckc_i_op1(b, CKC_OP_TILE_SMEM_LOAD_V4, ops, 3, vt, &attrs, "a");
}

ckc_value_t *ckc_b_smem_load_vN(ckc_ir_builder_t *b, ckc_value_t *smem,
                                ckc_value_t *const *indices, int num_indices,
                                const ckc_type_t *dtype, int n) {
    ckc_value_t **ops;
    int nops = 0;
    const ckc_type_t *vt;
    ckc_attr_map_t attrs;
    const char *dn;
    static const int allowed_8bit[] = {1, 2, 4, 8, 16};
    static const int allowed_other[] = {1, 2, 4, 8};
    char hint[16];
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (!dtype) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_VALUE,
                                            "smem_load_vN dtype is NULL");
    }
    dn = dtype->name;
    if (!(strcmp(dn, "f16") == 0 || strcmp(dn, "bf16") == 0 ||
          strcmp(dn, "f32") == 0 || strcmp(dn, "i32") == 0 ||
          strcmp(dn, "fp8e4m3") == 0 || strcmp(dn, "bf8e5m2") == 0 ||
          strcmp(dn, "i8") == 0)) {
        return (ckc_value_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE,
            "smem_load_vN supports f16 / bf16 / f32 / i32 / fp8e4m3 / "
            "bf8e5m2 / i8, got %s",
            dn);
    }
    {
        bool eight = (strcmp(dn, "fp8e4m3") == 0 ||
                      strcmp(dn, "bf8e5m2") == 0 || strcmp(dn, "i8") == 0);
        const int *allowed = eight ? allowed_8bit : allowed_other;
        int acount = eight ? 5 : 4;
        if (!ckc_n_in(n, allowed, acount)) {
            return (ckc_value_t *)ckc_i_set_err(
                b, CKC_ERR_VALUE,
                "unsupported vector width %d for smem_load_vN of %s", n, dn);
        }
    }
    if (num_indices <= 0) {
        return (ckc_value_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE, "smem_load_vN needs at least one index");
    }
    vt = ckc_vector_type(b, dtype, n);
    if (!vt) {
        return NULL;
    }
    ops = ckc_build_mem_operands(b, smem, indices, num_indices, NULL, false,
                                 &nops);
    if (!ops) {
        return NULL;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attrs, "elem_type", dn);
    ckc_attr_set_int(b, &attrs, "vec", (int64_t)n);
    ckc_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    snprintf(hint, sizeof(hint), "av%d", n);
    return ckc_i_op1(b, CKC_OP_TILE_SMEM_LOAD_VN, ops, nops, vt, &attrs, hint);
}

ckc_value_t *ckc_b_smem_load_vN_f16(ckc_ir_builder_t *b, ckc_value_t *smem,
                                    ckc_value_t *const *indices,
                                    int num_indices, int n) {
    return ckc_b_smem_load_vN(b, smem, indices, num_indices, ckc_f16(), n);
}

/* ===================================================================== */
/*  f32 LDS ops (cshuffle epilogue)                                       */
/* ===================================================================== */

ckc_value_t *ckc_b_smem_alloc_f32(ckc_ir_builder_t *b, const int *shape,
                                  int rank, const char *name_hint) {
    return ckc_b_smem_alloc(b, ckc_f32(), shape, rank,
                            name_hint ? name_hint : "smem_f32");
}

void ckc_b_smem_store_vN_f32(ckc_ir_builder_t *b, ckc_value_t *smem,
                             ckc_value_t *const *indices, int num_indices,
                             ckc_value_t *value, int n) {
    ckc_value_t **ops;
    int nops = 0;
    ckc_attr_map_t attrs;
    if (!ckc_i_live(b)) {
        return;
    }
    if (!(n == 1 || n == 2 || n == 4)) {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "smem_store_vN_f32 n must be 1, 2, or 4 (got %d)", n);
        return;
    }
    ops = ckc_build_mem_operands(b, smem, indices, num_indices, value, true,
                                 &nops);
    if (!ops) {
        return;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    ckc_attr_set_str(b, &attrs, "elem_type", "f32");
    ckc_attr_set_int(b, &attrs, "vec", (int64_t)n);
    ckc_i_op0(b, CKC_OP_TILE_SMEM_STORE_VN_F32, ops, nops, &attrs);
}

ckc_value_t *ckc_b_smem_load_vN_f32(ckc_ir_builder_t *b, ckc_value_t *smem,
                                    ckc_value_t *const *indices,
                                    int num_indices, int n) {
    ckc_value_t **ops;
    int nops = 0;
    const ckc_type_t *vt;
    ckc_attr_map_t attrs;
    char hint[16];
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (!(n == 1 || n == 2 || n == 4)) {
        return (ckc_value_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE,
            "smem_load_vN_f32 n must be 1, 2, or 4 (got %d)", n);
    }
    if (num_indices <= 0) {
        return (ckc_value_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE, "smem_load_vN_f32 needs at least one index");
    }
    vt = ckc_vector_type(b, ckc_f32(), n);
    if (!vt) {
        return NULL;
    }
    ops = ckc_build_mem_operands(b, smem, indices, num_indices, NULL, false,
                                 &nops);
    if (!ops) {
        return NULL;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attrs, "elem_type", "f32");
    ckc_attr_set_int(b, &attrs, "vec", (int64_t)n);
    ckc_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
    snprintf(hint, sizeof(hint), "av%df32", n);
    return ckc_i_op1(b, CKC_OP_TILE_SMEM_LOAD_VN_F32, ops, nops, vt, &attrs,
                     hint);
}

/* ===================================================================== */
/*  distributed / cooperative epilogue stores                            */
/* ===================================================================== */

void ckc_b_smem_store_distributed(ckc_ir_builder_t *b, ckc_value_t *smem,
                                  const ckc_attr_map_t *layout_attrs,
                                  ckc_value_t *values) {
    ckc_value_t *ops[2];
    if (!ckc_i_live(b)) {
        return;
    }
    ops[0] = smem;
    ops[1] = values;
    /* attrs = dict(layout_attrs); ckc_i_op0 deep-copies the passed map. */
    ckc_i_op0(b, CKC_OP_TILE_SMEM_STORE_DISTRIBUTED, ops, 2, layout_attrs);
}

void ckc_b_cooperative_global_store(ckc_ir_builder_t *b, ckc_value_t *ptr,
                                    ckc_value_t *addrs, ckc_value_t *values) {
    ckc_value_t *ops[3];
    ckc_attr_map_t attrs;
    int64_t vec;
    if (!ckc_i_live(b)) {
        return;
    }
    ops[0] = ptr;
    ops[1] = addrs;
    ops[2] = values;
    vec = (values && ckc_i_is_vector(values->type, NULL, -1))
              ? (int64_t)values->type->count
              : 1;
    attrs = ckc_i_attrs(b);
    ckc_attr_set_int(b, &attrs, "vec", vec);
    ckc_i_op0(b, CKC_OP_MEMREF_COOPERATIVE_GLOBAL_STORE, ops, 3, &attrs);
}

/* ===================================================================== */
/*  target-neutral MMA                                                    */
/* ===================================================================== */

ckc_value_t *ckc_b_mma(ckc_ir_builder_t *b, const char *op_id, ckc_value_t *a,
                       ckc_value_t *bb, ckc_value_t *c,
                       ckc_value_t *const *extra, int num_extra) {
    int c_frag_len;
    bool is_int_acc;
    const ckc_type_t *c_elem;
    const ckc_type_t *vt;
    const char *hint;
    ckc_attr_map_t attrs;
    ckc_value_t **ops;
    int nops;
    int i;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (!op_id) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_VALUE,
                                            "mma op_id is NULL");
    }
    /* C has no MmaOp object; op_id is always a bare string, so the frag length
     * and accumulator element come from the static op_id table (the Python
     * bare-string code path). */
    c_frag_len = ckc_mma_c_frag_len(op_id);
    if (c_frag_len < 0) {
        return (ckc_value_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE,
            "unknown MMA op_id '%s'; pass a known mfma_*/wmma_* op_id", op_id);
    }
    is_int_acc = ckc_mma_is_int_acc(op_id);
    c_elem = is_int_acc ? ckc_i32() : ckc_f32();
    vt = ckc_vector_type(b, c_elem, c_frag_len);
    if (!vt) {
        return NULL;
    }
    hint = ckc_mma_result_hint(op_id);

    if (num_extra < 0) {
        num_extra = 0;
    }
    nops = 3 + num_extra;
    ops = (ckc_value_t **)ckc_arena_alloc(&b->arena,
                                          (size_t)nops * sizeof(*ops));
    if (!ops) {
        return (ckc_value_t *)ckc_i_set_err(b, CKC_ERR_OOM,
                                            "mma operand array alloc failed");
    }
    ops[0] = a;
    ops[1] = bb;
    ops[2] = c;
    for (i = 0; i < num_extra; ++i) {
        ops[3 + i] = extra[i];
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attrs, "op_id", op_id);
    return ckc_i_op1(b, CKC_OP_TILE_MMA, ops, nops, vt, &attrs, hint);
}

/* ----- ISA-named MMA wrappers (thin wrappers over ckc_b_mma) ----- */

#define CKC_MMA_WRAP(fn, opid)                                                  \
    ckc_value_t *fn(ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *bb,       \
                    ckc_value_t *c) {                                           \
        return ckc_b_mma(b, opid, a, bb, c, NULL, 0);                           \
    }

CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x16_f16, "mfma_f32_16x16x16_f16")
CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x32_f16, "mfma_f32_16x16x32_f16")
CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x16_bf16, "mfma_f32_16x16x16_bf16")
CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x32_bf16, "mfma_f32_16x16x32_bf16")
CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x32_fp8, "mfma_f32_16x16x32_fp8")
CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x32_bf8, "mfma_f32_16x16x32_bf8")
CKC_MMA_WRAP(ckc_b_mfma_f32_32x32x8_f16, "mfma_f32_32x32x8_f16")
CKC_MMA_WRAP(ckc_b_mfma_f32_32x32x16_f16, "mfma_f32_32x32x16_f16")
CKC_MMA_WRAP(ckc_b_mfma_f32_32x32x16_bf16, "mfma_f32_32x32x16_bf16")
CKC_MMA_WRAP(ckc_b_mfma_f32_32x32x16_fp8, "mfma_f32_32x32x16_fp8")
CKC_MMA_WRAP(ckc_b_mfma_f32_32x32x16_bf8, "mfma_f32_32x32x16_bf8")
CKC_MMA_WRAP(ckc_b_mfma_f32_4x4x4_f16, "mfma_f32_4x4x4_f16")
CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x128_fp4, "mfma_f32_16x16x128_fp4")
CKC_MMA_WRAP(ckc_b_mfma_f32_16x16x96_fp6, "mfma_f32_16x16x96_fp6")
CKC_MMA_WRAP(ckc_b_wmma_f32_16x16x16_f16, "wmma_f32_16x16x16_f16")
CKC_MMA_WRAP(ckc_b_wmma_f32_16x16x16_bf16, "wmma_f32_16x16x16_bf16")
CKC_MMA_WRAP(ckc_b_wmma_gfx12_f32_16x16x16_f16, "wmma_gfx12_f32_16x16x16_f16")
CKC_MMA_WRAP(ckc_b_wmma_gfx12_f32_16x16x16_bf16, "wmma_gfx12_f32_16x16x16_bf16")

#undef CKC_MMA_WRAP

ckc_value_t *ckc_b_mfma_scale_f32_16x16x128_f8f6f4(
    ckc_ir_builder_t *b, ckc_value_t *a, ckc_value_t *bb, ckc_value_t *c,
    ckc_value_t *a_scale, ckc_value_t *b_scale) {
    ckc_value_t *extra[2];
    extra[0] = a_scale;
    extra[1] = b_scale;
    return ckc_b_mma(b, "mfma_scale_f32_16x16x128_f8f6f4", a, bb, c, extra, 2);
}

/* ===================================================================== */
/*  register-fragment reshape (P13)                                       */
/* ===================================================================== */

ckc_value_t *ckc_b_register_p_from_qk_c(ckc_ir_builder_t *b, ckc_value_t *qk_c,
                                        const ckc_type_t *target_dtype) {
    ckc_value_t *ops[1];
    const ckc_type_t *vt;
    ckc_attr_map_t attrs;
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (!target_dtype ||
        !(strcmp(target_dtype->name, "f16") == 0 ||
          strcmp(target_dtype->name, "bf16") == 0)) {
        return (ckc_value_t *)ckc_i_set_err(
            b, CKC_ERR_VALUE,
            "register_p_from_qk_c target must be f16/bf16, got %s",
            target_dtype ? target_dtype->name : "(null)");
    }
    ops[0] = qk_c;
    vt = ckc_vector_type(b, target_dtype, 8);
    if (!vt) {
        return NULL;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attrs, "target_dtype", target_dtype->name);
    return ckc_i_op1(b, CKC_OP_TILE_REGISTER_P_FROM_QK_C, ops, 1, vt, &attrs,
                     "pa");
}

/* ===================================================================== */
/*  inline asm                                                           */
/* ===================================================================== */

ckc_op_t *ckc_b_inline_asm(ckc_ir_builder_t *b, const char *asm_template,
                           const char *constraints,
                           ckc_value_t *const *operands, int num_operands,
                           const ckc_type_t *const *result_types,
                           int num_results,
                           const ckc_inline_asm_opts_t *opts) {
    ckc_attr_map_t attrs;
    bool sideeffect = true;  /* Python default */
    bool convergent = false; /* Python default */
    if (!ckc_i_live(b)) {
        return NULL;
    }
    if (opts) {
        if (opts->sideeffect_set) {
            sideeffect = opts->sideeffect;
        }
        if (opts->convergent_set) {
            convergent = opts->convergent;
        }
    }
    if (num_operands < 0) {
        num_operands = 0;
    }
    if (num_results < 0) {
        num_results = 0;
    }
    attrs = ckc_i_attrs(b);
    ckc_attr_set_str(b, &attrs, "template", asm_template ? asm_template : "");
    ckc_attr_set_str(b, &attrs, "constraints", constraints ? constraints : "");
    ckc_attr_set_bool(b, &attrs, "sideeffect", sideeffect);
    ckc_attr_set_bool(b, &attrs, "convergent", convergent);
    return ckc_i_op(b, CKC_OP_TILE_INLINE_ASM, operands, num_operands,
                    result_types, num_results, &attrs, NULL, 0, "asm", NULL);
}

ckc_op_t *ckc_b_inline_asm_multi(ckc_ir_builder_t *b, const char *asm_template,
                                 const char *constraints,
                                 ckc_value_t *const *operands, int num_operands,
                                 const ckc_type_t *const *result_types,
                                 int num_results,
                                 const ckc_inline_asm_opts_t *opts) {
    /* Python: <=1 result types delegates to inline_asm; >1 emits a single
     * tile.inline_asm op with all N result types (literal-struct return). Both
     * paths funnel through the same op builder here, so a single call with the
     * given result_types reproduces the emission for any N. */
    return ckc_b_inline_asm(b, asm_template, constraints, operands, num_operands,
                            result_types, num_results, opts);
}
