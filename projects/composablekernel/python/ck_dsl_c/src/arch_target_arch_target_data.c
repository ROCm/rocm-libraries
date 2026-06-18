/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * arch_target_arch_target_data.c -- bucket 0 of the C99 port of
 * ck_dsl.core.arch.target: the frozen SSOT data, the lane-coord IR emitters,
 * the dtype-normalisation core, and the shared lookup tables/helpers declared in
 * ckc/arch_target_internal.h.
 *
 * This file is a faithful, byte-identical translation of:
 *   - normalize_dtype() / _DTYPE_ALIASES                  -> ckc_normalize_dtype
 *   - the _mfma / _wmma lane-coord closures                -> static ckc_lane_coord_fn
 *   - LayoutMap.coord                                      -> ckc_layout_map_coord
 *   - _MMA_FRAGMENT_INFO + _build_mma_op (precomputed)     -> per-op_id static
 *                                                            ckc_layout_map_t +
 *                                                            ckc_mma_op_t[] catalogs
 *   - the embedded core/arch/data/arch_specs.json          -> static
 *                                                            ckc_arch_target_t
 *                                                            singletons + registry
 *
 * The lane-coord emitters reproduce EXACTLY the index arithmetic of the Python
 * closures (same op order, same constants) so the emitted IR is byte-identical.
 * All catalog/target descriptors are static read-only storage (program-lifetime
 * borrows, mirroring the Python @lru_cache singletons); the only arena/IR use is
 * inside the layout-map emitters, which create Values in the supplied builder.
 */
#include "ckc/arch_target_internal.h"

#include <stdio.h>
#include <string.h>

/* =========================================================================
 * dtype normalisation core (shared by both buckets)
 * =========================================================================
 *
 * Mirrors target.py::normalize_dtype: strip + lower(name), then
 * _DTYPE_ALIASES.get(key, key). The canonical RHS values are interned static
 * strings; unknown spellings pass through as the lowercased text in `lowered`.
 */

typedef struct ckc_ati_dtype_alias {
    const char *alias;       /* lowercased key */
    const char *canonical;   /* interned canonical value */
} ckc_ati_dtype_alias_t;

/* Byte-for-byte the _DTYPE_ALIASES map (insertion order is irrelevant: lookup is
 * by exact lowercased key). */
static const ckc_ati_dtype_alias_t k_dtype_aliases[] = {
    {"f16", "fp16"},
    {"half", "fp16"},
    {"fp16", "fp16"},
    {"bf16", "bf16"},
    {"bfloat16", "bf16"},
    {"f32", "fp32"},
    {"float", "fp32"},
    {"fp32", "fp32"},
    {"fp8", "fp8e4m3"},
    {"fp8e4m3", "fp8e4m3"},
    {"bf8", "bf8e5m2"},
    {"bf8e5m2", "bf8e5m2"},
    {"iu8", "iu8"},
    {"iu4", "iu4"},
    {"i8", "i8"},
    {"int8", "i8"},
    {"i4", "i4"},
    {"int4", "i4"},
    {"i32", "i32"},
    {"int32", "i32"},
};
#define K_NUM_DTYPE_ALIASES \
    ((int)(sizeof(k_dtype_aliases) / sizeof(k_dtype_aliases[0])))

/* str.strip(): Python strips ASCII whitespace from both ends. */
static int ckc_ati_is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

const char *ckc_ati_normalize_dtype(const char *name, char *lowered, size_t cap) {
    int i;
    size_t n;
    const char *start;
    const char *end;

    if (name == NULL) {
        /* Python would raise on None.strip(); the C contract requires a result.
         * Treat as empty string (no alias match, passes through as ""). */
        if (lowered != NULL && cap > 0) {
            lowered[0] = '\0';
        }
        return (lowered != NULL && cap > 0) ? lowered : "";
    }

    /* strip(): advance over leading/trailing whitespace */
    start = name;
    while (*start != '\0' && ckc_ati_is_ws(*start)) {
        start++;
    }
    end = start + strlen(start);
    while (end > start && ckc_ati_is_ws(end[-1])) {
        end--;
    }

    /* lower() into the caller buffer (the pass-through result) */
    n = (size_t)(end - start);
    if (lowered == NULL || cap == 0) {
        /* No scratch: caller guarantees a known spelling. Build a small inline
         * copy on a fixed buffer to look up; if unknown we have nowhere to
         * return it, so fall back to "". */
        static char tmp[64];
        size_t m = n < sizeof(tmp) - 1 ? n : sizeof(tmp) - 1;
        for (i = 0; (size_t)i < m; i++) {
            char c = start[i];
            tmp[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
        }
        tmp[m] = '\0';
        for (i = 0; i < K_NUM_DTYPE_ALIASES; i++) {
            if (strcmp(tmp, k_dtype_aliases[i].alias) == 0) {
                return k_dtype_aliases[i].canonical;
            }
        }
        return "";
    }

    if (n > cap - 1) {
        n = cap - 1;
    }
    for (i = 0; (size_t)i < n; i++) {
        char c = start[i];
        lowered[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
    }
    lowered[n] = '\0';

    /* _DTYPE_ALIASES.get(key, key) */
    for (i = 0; i < K_NUM_DTYPE_ALIASES; i++) {
        if (strcmp(lowered, k_dtype_aliases[i].alias) == 0) {
            return k_dtype_aliases[i].canonical;
        }
    }
    return lowered;
}

const char *ckc_normalize_dtype(const char *name, char *scratch, size_t scratch_cap) {
    return ckc_ati_normalize_dtype(name, scratch, scratch_cap);
}

/* =========================================================================
 * lane-coord emitters (static ckc_lane_coord_fn)
 * =========================================================================
 *
 * Each fn reproduces, op-for-op, the Python closure of the same name, emitting
 * arith.* through the builder. On a failed/NULL builder the public ckc_b_*
 * helpers are sticky no-ops returning NULL; we additionally short-circuit so the
 * out params are left NULL (matching the "no-op leaving NULLs" contract).
 */

#define CKC_ATI_COORD_GUARD(b, out0, out1)        \
    do {                                          \
        if (out0) *(out0) = NULL;                 \
        if (out1) *(out1) = NULL;                 \
        if ((b) == NULL || (b)->status != CKC_OK) \
            return;                               \
    } while (0)

/* MFMA 16x16 accumulator: slot i -> (m_blk*4 + i, lane % 16). */
static void _mfma_acc_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                            ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *n_in_atom, *m_blk, *row;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    n_in_atom = ckc_b_mod(b, lane, c16);
    m_blk = ckc_b_div(b, lane, c16);
    /* row = b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        ckc_value_t *row_mul = ckc_b_mul(b, m_blk, ckc_b_const_i32(b, 4));
        row = ckc_b_add(b, row_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = row;
    if (out1) *out1 = n_in_atom;
}

/* MFMA 32x32 accumulator:
 *   row = (slot//4)*8 + (lane//32)*4 + (slot%4), col = lane % 32. */
static void _mfma_acc_32x32(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                            ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c32, *n_in_atom, *m_blk, *row;
    int rb, ri;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c32 = ckc_b_const_i32(b, 32);
    n_in_atom = ckc_b_mod(b, lane, c32);
    m_blk = ckc_b_div(b, lane, c32);
    rb = slot / 4;
    ri = slot % 4;
    /* row = b.add(b.add(b.const_i32(rb*8), b.mul(m_blk, b.const_i32(4))),
     *             b.const_i32(ri))
     * Python evaluates each add left-to-right: const(rb*8) first, then the mul
     * (const 4 + mul), then the inner add, then const(ri), then the outer add.
     * C call-arg order is unspecified (GCC: right-to-left); sequence into temps
     * to pin Python source-order. */
    {
        ckc_value_t *c_rb8 = ckc_b_const_i32(b, rb * 8);
        ckc_value_t *blk_mul = ckc_b_mul(b, m_blk, ckc_b_const_i32(b, 4));
        ckc_value_t *inner = ckc_b_add(b, c_rb8, blk_mul);
        row = ckc_b_add(b, inner, ckc_b_const_i32(b, ri));
    }
    if (out0) *out0 = row;
    if (out1) *out1 = n_in_atom;
}

/* MFMA 16x16x16 A operand: (lane % 16, k_blk*4 + slot). */
static void _mfma_a_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                          ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *m_in_atom, *k_blk, *k;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    m_in_atom = ckc_b_mod(b, lane, c16);
    k_blk = ckc_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        ckc_value_t *k_mul = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, 4));
        k = ckc_b_add(b, k_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = m_in_atom;
    if (out1) *out1 = k;
}

/* MFMA 16x16x16 B operand: (k_blk*4 + slot, lane % 16). */
static void _mfma_b_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                          ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *n_in_atom, *k_blk, *k;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    n_in_atom = ckc_b_mod(b, lane, c16);
    k_blk = ckc_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        ckc_value_t *k_mul = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, 4));
        k = ckc_b_add(b, k_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = k;
    if (out1) *out1 = n_in_atom;
}

/* MFMA 32x32x8 A operand: (lane % 32, k_blk*4 + slot). */
static void _mfma_a_32x32x8(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                            ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c32, *m_in_atom, *k_blk, *k;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c32 = ckc_b_const_i32(b, 32);
    m_in_atom = ckc_b_mod(b, lane, c32);
    k_blk = ckc_b_div(b, lane, c32);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        ckc_value_t *k_mul = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, 4));
        k = ckc_b_add(b, k_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = m_in_atom;
    if (out1) *out1 = k;
}

/* MFMA 32x32x8 B operand: (k_blk*4 + slot, lane % 32). */
static void _mfma_b_32x32x8(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                            ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c32, *n_in_atom, *k_blk, *k;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c32 = ckc_b_const_i32(b, 32);
    n_in_atom = ckc_b_mod(b, lane, c32);
    k_blk = ckc_b_div(b, lane, c32);
    /* k = b.add(b.mul(k_blk, b.const_i32(4)), b.const_i32(slot)). Pin the mul
     * (with its const 4) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        ckc_value_t *k_mul = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, 4));
        k = ckc_b_add(b, k_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = k;
    if (out1) *out1 = n_in_atom;
}

/* WMMA 16x16x16 accumulator (wave32): (2*slot + lane//16, lane % 16). */
static void _wmma_acc_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                            ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *col, *half, *row;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    col = ckc_b_mod(b, lane, c16);
    half = ckc_b_div(b, lane, c16);
    row = ckc_b_add(b, ckc_b_const_i32(b, 2 * slot), half);
    if (out0) *out0 = row;
    if (out1) *out1 = col;
}

/* WMMA 16x16x16 A operand (wave32): (lane % 16, K=slot). */
static void _wmma_a_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                          ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *row;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    row = ckc_b_mod(b, lane, c16);
    if (out0) *out0 = row;
    if (out1) *out1 = ckc_b_const_i32(b, slot);
}

/* WMMA 16x16x16 B operand (wave32): (K=slot, lane % 16). */
static void _wmma_b_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                          ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *col;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    col = ckc_b_mod(b, lane, c16);
    if (out0) *out0 = ckc_b_const_i32(b, slot);
    if (out1) *out1 = col;
}

/* iu8 WMMA A operand (wave32): (lane % 16, k_base = 4*slot). */
static void _wmma_a_16x16_iu8(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                              ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *row;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    row = ckc_b_mod(b, lane, c16);
    if (out0) *out0 = row;
    if (out1) *out1 = ckc_b_const_i32(b, 4 * slot);
}

/* iu8 WMMA B operand (wave32): (k_base = 4*slot, lane % 16). */
static void _wmma_b_16x16_iu8(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                              ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *col;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    col = ckc_b_mod(b, lane, c16);
    if (out0) *out0 = ckc_b_const_i32(b, 4 * slot);
    if (out1) *out1 = col;
}

/* iu4 WMMA A operand (wave32): (lane % 16, k_base = 8*slot). */
static void _wmma_a_16x16_iu4(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                              ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *row;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    row = ckc_b_mod(b, lane, c16);
    if (out0) *out0 = row;
    if (out1) *out1 = ckc_b_const_i32(b, 8 * slot);
}

/* iu4 WMMA B operand (wave32): (k_base = 8*slot, lane % 16). */
static void _wmma_b_16x16_iu4(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                              ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *col;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    col = ckc_b_mod(b, lane, c16);
    if (out0) *out0 = ckc_b_const_i32(b, 8 * slot);
    if (out1) *out1 = col;
}

/* RDNA4 WMMA 16x16x16 accumulator (wave32): ((lane//16)*8 + slot, lane % 16). */
static void _wmma_gfx12_acc_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                                  ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *col, *half, *row;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    col = ckc_b_mod(b, lane, c16);
    half = ckc_b_div(b, lane, c16);
    /* row = b.add(b.mul(half, b.const_i32(8)), b.const_i32(slot)). Python
     * evaluates the add's args left-to-right (the mul, with its const 8, emits
     * BEFORE const(slot)). C call-arg order is unspecified (GCC: right-to-left),
     * which would emit const(slot) first and shift later SSA names. Hoist the
     * mul into a temp to pin Python source-order. */
    {
        ckc_value_t *row_mul = ckc_b_mul(b, half, ckc_b_const_i32(b, 8));
        row = ckc_b_add(b, row_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = row;
    if (out1) *out1 = col;
}

/* RDNA4 WMMA 16x16x16 A operand (wave32): (lane % 16, (lane//16)*8 + slot). */
static void _wmma_gfx12_a_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                                ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *row, *k_half, *k;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    row = ckc_b_mod(b, lane, c16);
    k_half = ckc_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_half, b.const_i32(8)), b.const_i32(slot)). Pin the mul
     * (with its const 8) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        ckc_value_t *k_mul = ckc_b_mul(b, k_half, ckc_b_const_i32(b, 8));
        k = ckc_b_add(b, k_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = row;
    if (out1) *out1 = k;
}

/* RDNA4 WMMA 16x16x16 B operand (wave32): ((lane//16)*8 + slot, lane % 16). */
static void _wmma_gfx12_b_16x16(ckc_ir_builder_t *b, ckc_value_t *lane, int slot,
                                ckc_value_t **out0, ckc_value_t **out1) {
    ckc_value_t *c16, *col, *k_half, *k;
    CKC_ATI_COORD_GUARD(b, out0, out1);
    c16 = ckc_b_const_i32(b, 16);
    col = ckc_b_mod(b, lane, c16);
    k_half = ckc_b_div(b, lane, c16);
    /* k = b.add(b.mul(k_half, b.const_i32(8)), b.const_i32(slot)). Pin the mul
     * (with its const 8) to emit before const(slot) to match Python's
     * left-to-right arg evaluation (C call-arg order is unspecified). */
    {
        ckc_value_t *k_mul = ckc_b_mul(b, k_half, ckc_b_const_i32(b, 8));
        k = ckc_b_add(b, k_mul, ckc_b_const_i32(b, slot));
    }
    if (out0) *out0 = k;
    if (out1) *out1 = col;
}

/* =========================================================================
 * LayoutMap.coord -> ckc_layout_map_coord
 * =========================================================================*/

bool ckc_layout_map_coord(const ckc_layout_map_t *m, ckc_ir_builder_t *b,
                          ckc_value_t *lane, int slot,
                          ckc_value_t **out0, ckc_value_t **out1) {
    if (out0) *out0 = NULL;
    if (out1) *out1 = NULL;
    if (m == NULL || m->fn == NULL) {
        return false;
    }
    /* Python LayoutMap.coord: ValueError if not (0 <= slot < frag_len). */
    if (!(slot >= 0 && slot < m->frag_len)) {
        if (b != NULL) {
            const char *role_txt = (m->role == CKC_MMA_ROLE_ACC)  ? "acc"
                                   : (m->role == CKC_MMA_ROLE_A)   ? "a"
                                                                   : "b";
            b->status = CKC_ERR_VALUE;
            snprintf(b->err, CKC_ERR_MSG_CAP,
                     "fragment slot %d out of range [0, %d) for '%s' layout map",
                     slot, m->frag_len, role_txt);
        }
        return false;
    }
    m->fn(b, lane, slot, out0, out1);
    if (b != NULL && b->status != CKC_OK) {
        return false;
    }
    return true;
}

/* =========================================================================
 * Per-op_id LayoutMap statics (the precomputed _build_mma_op output)
 * =========================================================================
 *
 * In Python _build_mma_op constructs, for each catalog row, up to three
 * LayoutMap instances from the op_id's _FragInfo (role + frag_len + wave_size +
 * fn). Because the maps depend ONLY on op_id (not on the per-arch row), we define
 * one static LayoutMap triplet per op_id here and point every arch's catalog rows
 * at them. A map is omitted (left NULL) when the _FragInfo has no fn for that
 * role or a zero frag_len (the `_mk(...) is None` case).
 *
 * Naming: lm_<opidkey>_{a,b,c}. Only op_ids with at least one verified map need a
 * struct; the rest carry NULL layout pointers in the catalog directly.
 */

/* --- mfma_f32_16x16x16_f16 / _bf16: a/b/c all present (frag 4/4/4, wave64) --- */
static const ckc_layout_map_t lm_mfma_16x16x16_a = {
    CKC_MMA_ROLE_A, 4, 64, _mfma_a_16x16};
static const ckc_layout_map_t lm_mfma_16x16x16_b = {
    CKC_MMA_ROLE_B, 4, 64, _mfma_b_16x16};
static const ckc_layout_map_t lm_mfma_16x16x16_c = {
    CKC_MMA_ROLE_ACC, 4, 64, _mfma_acc_16x16};

/* --- mfma_f32_16x16x32_{f16,bf16,fp8,bf8}: only c map (frag 8/8/4, wave64) --- */
static const ckc_layout_map_t lm_mfma_16x16x32_c = {
    CKC_MMA_ROLE_ACC, 4, 64, _mfma_acc_16x16};

/* --- mfma_f32_32x32x8_f16: a/b/c present (frag 4/4/16, wave64) --- */
static const ckc_layout_map_t lm_mfma_32x32x8_a = {
    CKC_MMA_ROLE_A, 4, 64, _mfma_a_32x32x8};
static const ckc_layout_map_t lm_mfma_32x32x8_b = {
    CKC_MMA_ROLE_B, 4, 64, _mfma_b_32x32x8};
static const ckc_layout_map_t lm_mfma_32x32x8_c = {
    CKC_MMA_ROLE_ACC, 16, 64, _mfma_acc_32x32};

/* --- mfma_f32_32x32x16_{f16,bf16,fp8,bf8}: only c map (frag 8/8/16) --- */
static const ckc_layout_map_t lm_mfma_32x32x16_c = {
    CKC_MMA_ROLE_ACC, 16, 64, _mfma_acc_32x32};

/* --- wmma_f32_16x16x16_{f16,bf16}: a/b/c present (frag 16/16/8, wave32) --- */
static const ckc_layout_map_t lm_wmma_16x16x16_a = {
    CKC_MMA_ROLE_A, 16, 32, _wmma_a_16x16};
static const ckc_layout_map_t lm_wmma_16x16x16_b = {
    CKC_MMA_ROLE_B, 16, 32, _wmma_b_16x16};
static const ckc_layout_map_t lm_wmma_16x16x16_c = {
    CKC_MMA_ROLE_ACC, 8, 32, _wmma_acc_16x16};

/* --- wmma_i32_16x16x16_iu8: a/b/c present (frag 4/4/8, wave32) --- */
static const ckc_layout_map_t lm_wmma_iu8_a = {
    CKC_MMA_ROLE_A, 4, 32, _wmma_a_16x16_iu8};
static const ckc_layout_map_t lm_wmma_iu8_b = {
    CKC_MMA_ROLE_B, 4, 32, _wmma_b_16x16_iu8};
static const ckc_layout_map_t lm_wmma_iu8_c = {
    CKC_MMA_ROLE_ACC, 8, 32, _wmma_acc_16x16};

/* --- wmma_i32_16x16x16_iu4: a/b/c present (frag 2/2/8, wave32) --- */
static const ckc_layout_map_t lm_wmma_iu4_a = {
    CKC_MMA_ROLE_A, 2, 32, _wmma_a_16x16_iu4};
static const ckc_layout_map_t lm_wmma_iu4_b = {
    CKC_MMA_ROLE_B, 2, 32, _wmma_b_16x16_iu4};
static const ckc_layout_map_t lm_wmma_iu4_c = {
    CKC_MMA_ROLE_ACC, 8, 32, _wmma_acc_16x16};

/* --- wmma_gfx12_f32_16x16x16_{f16,bf16}: a/b/c present (frag 8/8/8, wave32) --- */
static const ckc_layout_map_t lm_wmma_gfx12_a = {
    CKC_MMA_ROLE_A, 8, 32, _wmma_gfx12_a_16x16};
static const ckc_layout_map_t lm_wmma_gfx12_b = {
    CKC_MMA_ROLE_B, 8, 32, _wmma_gfx12_b_16x16};
static const ckc_layout_map_t lm_wmma_gfx12_c = {
    CKC_MMA_ROLE_ACC, 8, 32, _wmma_gfx12_acc_16x16};

/* =========================================================================
 * Per-arch ckc_mma_op_t[] catalogs (the embedded arch_specs.json mma rows,
 * enriched with frag lengths + layout-map pointers, mirroring _build_mma_op).
 * =========================================================================
 *
 * Frag lengths come from _MMA_FRAGMENT_INFO[op_id]; op_ids absent from that
 * table carry (0,0,0,64) and NULL maps. dtype fields hold the normalised
 * (canonical) catalog keys (interned strings from k_dtype_aliases), exactly as
 * normalize_dtype(o["a"/"b"/"c"]) would produce.
 *
 * Field order of ckc_mma_op_t:
 *   family, a_dtype, b_dtype, c_dtype, m, n, k, op_id,
 *   a_frag_len, b_frag_len, c_frag_len, wave_size, a_layout, b_layout, c_layout
 *
 * NOTE: fp4/fp6 dtypes are not in _DTYPE_ALIASES, so normalize_dtype passes them
 * through as the lowercased spelling "fp4"/"fp6" (Python identity fallthrough).
 */

/* ----------------------------- gfx90a (CDNA2) ---------------------------- */
/* MI200 (Aldebaran): wave64, 64 KB LDS, no async_lds, no fp8/bf8 native MFMA.
 * bf16 32x32x8 is hardware-supported (same MFMA atom dimensions as fp16 32x32x8;
 * bf16 arrived in CDNA2 with the same layout maps as fp16 for those shapes). */
static const ckc_mma_op_t k_mma_gfx90a[] = {
    {"mma", "fp16", "fp16", "fp32", 16, 16, 16, "mfma_f32_16x16x16_f16",
     4, 4, 4, 64, &lm_mfma_16x16x16_a, &lm_mfma_16x16x16_b, &lm_mfma_16x16x16_c},
    {"mma", "fp16", "fp16", "fp32", 32, 32, 8, "mfma_f32_32x32x8_f16",
     4, 4, 16, 64, &lm_mfma_32x32x8_a, &lm_mfma_32x32x8_b, &lm_mfma_32x32x8_c},
    {"mma", "bf16", "bf16", "fp32", 16, 16, 16, "mfma_f32_16x16x16_bf16",
     4, 4, 4, 64, &lm_mfma_16x16x16_a, &lm_mfma_16x16x16_b, &lm_mfma_16x16x16_c},
    {"mma", "bf16", "bf16", "fp32", 32, 32, 8, "mfma_f32_32x32x8_bf16",
     4, 4, 16, 64, &lm_mfma_32x32x8_a, &lm_mfma_32x32x8_b, &lm_mfma_32x32x8_c},
};

/* ----------------------------- gfx942 (CDNA) ----------------------------- */
static const ckc_mma_op_t k_mma_gfx942[] = {
    {"mma", "fp16", "fp16", "fp32", 16, 16, 16, "mfma_f32_16x16x16_f16",
     4, 4, 4, 64, &lm_mfma_16x16x16_a, &lm_mfma_16x16x16_b, &lm_mfma_16x16x16_c},
    {"mma", "fp16", "fp16", "fp32", 32, 32, 8, "mfma_f32_32x32x8_f16",
     4, 4, 16, 64, &lm_mfma_32x32x8_a, &lm_mfma_32x32x8_b, &lm_mfma_32x32x8_c},
    {"mma", "bf16", "bf16", "fp32", 16, 16, 16, "mfma_f32_16x16x16_bf16",
     4, 4, 4, 64, &lm_mfma_16x16x16_a, &lm_mfma_16x16x16_b, &lm_mfma_16x16x16_c},
    {"mma", "fp8e4m3", "fp8e4m3", "fp32", 16, 16, 32, "mfma_f32_16x16x32_fp8",
     8, 8, 4, 64, NULL, NULL, &lm_mfma_16x16x32_c},
    {"mma", "fp8e4m3", "fp8e4m3", "fp32", 32, 32, 16, "mfma_f32_32x32x16_fp8",
     8, 8, 16, 64, NULL, NULL, &lm_mfma_32x32x16_c},
    {"mma", "bf8e5m2", "bf8e5m2", "fp32", 16, 16, 32, "mfma_f32_16x16x32_bf8",
     8, 8, 4, 64, NULL, NULL, &lm_mfma_16x16x32_c},
    {"mma", "bf8e5m2", "bf8e5m2", "fp32", 32, 32, 16, "mfma_f32_32x32x16_bf8",
     8, 8, 16, 64, NULL, NULL, &lm_mfma_32x32x16_c},
};

/* ----------------------------- gfx950 (CDNA) ----------------------------- */
static const ckc_mma_op_t k_mma_gfx950[] = {
    {"mma", "fp16", "fp16", "fp32", 16, 16, 16, "mfma_f32_16x16x16_f16",
     4, 4, 4, 64, &lm_mfma_16x16x16_a, &lm_mfma_16x16x16_b, &lm_mfma_16x16x16_c},
    {"mma", "fp16", "fp16", "fp32", 16, 16, 32, "mfma_f32_16x16x32_f16",
     8, 8, 4, 64, NULL, NULL, &lm_mfma_16x16x32_c},
    {"mma", "fp16", "fp16", "fp32", 32, 32, 8, "mfma_f32_32x32x8_f16",
     4, 4, 16, 64, &lm_mfma_32x32x8_a, &lm_mfma_32x32x8_b, &lm_mfma_32x32x8_c},
    {"mma", "fp16", "fp16", "fp32", 32, 32, 16, "mfma_f32_32x32x16_f16",
     8, 8, 16, 64, NULL, NULL, &lm_mfma_32x32x16_c},
    {"mma", "bf16", "bf16", "fp32", 16, 16, 16, "mfma_f32_16x16x16_bf16",
     4, 4, 4, 64, &lm_mfma_16x16x16_a, &lm_mfma_16x16x16_b, &lm_mfma_16x16x16_c},
    {"mma", "bf16", "bf16", "fp32", 16, 16, 32, "mfma_f32_16x16x32_bf16",
     8, 8, 4, 64, NULL, NULL, &lm_mfma_16x16x32_c},
    {"mma", "fp8e4m3", "fp8e4m3", "fp32", 16, 16, 32, "mfma_f32_16x16x32_fp8",
     8, 8, 4, 64, NULL, NULL, &lm_mfma_16x16x32_c},
    {"mma", "fp8e4m3", "fp8e4m3", "fp32", 32, 32, 16, "mfma_f32_32x32x16_fp8",
     8, 8, 16, 64, NULL, NULL, &lm_mfma_32x32x16_c},
    {"mma", "bf8e5m2", "bf8e5m2", "fp32", 16, 16, 32, "mfma_f32_16x16x32_bf8",
     8, 8, 4, 64, NULL, NULL, &lm_mfma_16x16x32_c},
    {"mma", "bf8e5m2", "bf8e5m2", "fp32", 32, 32, 16, "mfma_f32_32x32x16_bf8",
     8, 8, 16, 64, NULL, NULL, &lm_mfma_32x32x16_c},
    /* fp4/fp6 pass through normalize_dtype unchanged; op_ids carry frag lengths
     * only (no verified maps), per _MMA_FRAGMENT_INFO. */
    {"mma", "fp4", "fp4", "fp32", 16, 16, 128, "mfma_f32_16x16x128_fp4",
     16, 16, 4, 64, NULL, NULL, NULL},
    {"mma", "fp6", "fp6", "fp32", 16, 16, 96, "mfma_f32_16x16x96_fp6",
     12, 12, 4, 64, NULL, NULL, NULL},
};

/* ----------------------------- gfx1151 (RDNA3.5) ------------------------- */
static const ckc_mma_op_t k_mma_gfx1151[] = {
    {"wmma", "fp16", "fp16", "fp32", 16, 16, 16, "wmma_f32_16x16x16_f16",
     16, 16, 8, 32, &lm_wmma_16x16x16_a, &lm_wmma_16x16x16_b, &lm_wmma_16x16x16_c},
    {"wmma", "bf16", "bf16", "fp32", 16, 16, 16, "wmma_f32_16x16x16_bf16",
     16, 16, 8, 32, &lm_wmma_16x16x16_a, &lm_wmma_16x16x16_b, &lm_wmma_16x16x16_c},
    {"wmma", "iu8", "iu8", "i32", 16, 16, 16, "wmma_i32_16x16x16_iu8",
     4, 4, 8, 32, &lm_wmma_iu8_a, &lm_wmma_iu8_b, &lm_wmma_iu8_c},
    {"wmma", "iu4", "iu4", "i32", 16, 16, 16, "wmma_i32_16x16x16_iu4",
     2, 2, 8, 32, &lm_wmma_iu4_a, &lm_wmma_iu4_b, &lm_wmma_iu4_c},
};

/* ----------------------------- gfx1201 (RDNA4) -------------------------- */
static const ckc_mma_op_t k_mma_gfx1201[] = {
    {"wmma", "fp16", "fp16", "fp32", 16, 16, 16, "wmma_gfx12_f32_16x16x16_f16",
     8, 8, 8, 32, &lm_wmma_gfx12_a, &lm_wmma_gfx12_b, &lm_wmma_gfx12_c},
    {"wmma", "bf16", "bf16", "fp32", 16, 16, 16, "wmma_gfx12_f32_16x16x16_bf16",
     8, 8, 8, 32, &lm_wmma_gfx12_a, &lm_wmma_gfx12_b, &lm_wmma_gfx12_c},
};

/* --------------------------- gfx11-generic (RDNA3) ---------------------- */
static const ckc_mma_op_t k_mma_gfx11_generic[] = {
    {"wmma", "fp16", "fp16", "fp32", 16, 16, 16, "wmma_f32_16x16x16_f16",
     16, 16, 8, 32, &lm_wmma_16x16x16_a, &lm_wmma_16x16x16_b, &lm_wmma_16x16x16_c},
    {"wmma", "bf16", "bf16", "fp32", 16, 16, 16, "wmma_f32_16x16x16_bf16",
     16, 16, 8, 32, &lm_wmma_16x16x16_a, &lm_wmma_16x16x16_b, &lm_wmma_16x16x16_c},
    {"wmma", "iu8", "iu8", "i32", 16, 16, 16, "wmma_i32_16x16x16_iu8",
     4, 4, 8, 32, &lm_wmma_iu8_a, &lm_wmma_iu8_b, &lm_wmma_iu8_c},
    {"wmma", "iu4", "iu4", "i32", 16, 16, 16, "wmma_i32_16x16x16_iu4",
     2, 2, 8, 32, &lm_wmma_iu4_a, &lm_wmma_iu4_b, &lm_wmma_iu4_c},
};

/* =========================================================================
 * Per-arch ckc_arch_target_t singletons (the embedded arches[gfx] entries).
 * =========================================================================
 *
 * Field order of ckc_arch_target_t:
 *   gfx, family, target_family, wave_size, lds_capacity_bytes, vmcnt_bits,
 *   mma{ops,num_ops}, memory{has_async_lds,has_ds_read_tr,buffer_load_max_dwords},
 *   limits{max_threads_per_block,vgprs,agprs,sgprs}
 */

#define K_NUM(arr) ((int)(sizeof(arr) / sizeof((arr)[0])))

static const ckc_arch_target_t k_target_gfx90a = {
    "gfx90a", "cdna", "gfx9_mfma", 64, 65536, 4,
    {k_mma_gfx90a, K_NUM(k_mma_gfx90a)},
    {false, false, 4},
    {1024, 512, 256, 102},
};

static const ckc_arch_target_t k_target_gfx942 = {
    "gfx942", "cdna", "gfx9_mfma", 64, 65536, 4,
    {k_mma_gfx942, K_NUM(k_mma_gfx942)},
    {true, false, 4},
    {1024, 512, 256, 102},
};

static const ckc_arch_target_t k_target_gfx950 = {
    "gfx950", "cdna", "gfx950", 64, 163840, 6,
    {k_mma_gfx950, K_NUM(k_mma_gfx950)},
    {true, true, 4},
    {1024, 512, 256, 102},
};

static const ckc_arch_target_t k_target_gfx1151 = {
    "gfx1151", "rdna", "gfx11_rdna", 32, 65536, 6,
    {k_mma_gfx1151, K_NUM(k_mma_gfx1151)},
    {false, false, 4},
    {1024, 256, 0, 106},
};

static const ckc_arch_target_t k_target_gfx1201 = {
    "gfx1201", "rdna", "gfx12_rdna", 32, 65536, 6,
    {k_mma_gfx1201, K_NUM(k_mma_gfx1201)},
    {false, false, 4},
    {1024, 256, 0, 106},
};

static const ckc_arch_target_t k_target_gfx11_generic = {
    "gfx11-generic", "rdna", "gfx11_rdna", 32, 65536, 6,
    {k_mma_gfx11_generic, K_NUM(k_mma_gfx11_generic)},
    {false, false, 4},
    {1024, 256, 0, 106},
};

/* =========================================================================
 * Shared registry + known-arches (sorted by gfx token).
 * =========================================================================
 *
 * Python known_arches() returns tuple(sorted(specs)); sorted() on the gfx
 * strings yields: "gfx11-generic", "gfx1151", "gfx1201", "gfx90a", "gfx942",
 * "gfx950". ('-' (0x2D) < '1' (0x31), so "gfx11-generic" sorts before
 * "gfx1151"; '0' < '4' so "gfx90a" sorts before "gfx942".)
 * The registry is kept in this same sorted order so ckc_known_arches can return
 * ckc_ati_known_arches directly.
 */

const ckc_ati_arch_row_t ckc_ati_arch_registry[] = {
    {"gfx11-generic", &k_target_gfx11_generic},
    {"gfx1151", &k_target_gfx1151},
    {"gfx1201", &k_target_gfx1201},
    {"gfx90a", &k_target_gfx90a},
    {"gfx942", &k_target_gfx942},
    {"gfx950", &k_target_gfx950},
    {NULL, NULL}, /* terminator */
};

const int ckc_ati_arch_registry_len = 6;

const char *const ckc_ati_known_arches[] = {
    "gfx11-generic",
    "gfx1151",
    "gfx1201",
    "gfx90a",
    "gfx942",
    "gfx950",
    NULL,
};
