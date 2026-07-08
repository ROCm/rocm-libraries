// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.instances.common.sparse_attention.c -- C99 port of the LDS
 * bitmap primitives from rocke/instances/common/sparse_attention.py.
 *
 * Ported symbols: _const_i8, _cooperative_iter, _stage_jenga_mask_to_lds,
 * _stage_vsa_bitmap_to_lds, _lds_bitmap_predicate.
 *
 * Each helper reproduces its Python counterpart's rocke_b_* builder-call
 * sequence byte-faithfully (same ops, same order, same operands, same
 * result-name hints). The host-side control structure (chunk loops, static
 * range checks, scf_if scoping) is reproduced exactly so the emitted op
 * stream is identical to the Python.
 *
 * scf_if scoping: Python's ``with b.scf_if(cond):`` maps to
 *   rocke_if_t iff = rocke_b_scf_if(b, cond);
 *   rocke_b_region_enter(b, iff.then_region);
 *   ... body ...
 *   rocke_b_region_leave(b);
 *
 * Python host-side closures (``body(slot)``) map to a function pointer +
 * opaque ``user`` context (the standard closure-emulation idiom).
 *
 * Lifetime: every node is arena-owned (rocke_ir_builder_t.arena). Nothing is
 * freed individually.
 */

#include "rocke/helper_rocke.instances.common.sparse_attention.h"
#include "rocke/ir.h"

/* ------------------------------------------------------------- _const_i8 */

rocke_value_t* rocke_sparse_attn_const_i8(rocke_ir_builder_t* b, int value)
{
    rocke_attr_map_t a;
    const rocke_type_t* rty[1];
    rocke_op_t* op;

    if(b == NULL)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* return b._op("arith.constant", result_types=[I8],
     *              attrs={"value": int(value), "ity": "i8"},
     *              result_name_hint="ci8").result */
    rocke_attr_map_init(&a);
    rocke_attr_set_int(b, &a, "value", (int64_t)value);
    rocke_attr_set_str(b, &a, "ity", "i8");

    rty[0] = rocke_i8();
    op = rocke_b_op(b, ROCKE_OP_ARITH_CONSTANT, NULL, 0, rty, 1, &a, NULL, 0, "ci8", NULL);
    return rocke_op_result(b, op);
}

/* ----------------------------------------------------- _cooperative_iter */

void rocke_sparse_attn_cooperative_iter(rocke_ir_builder_t* b,
                                        rocke_value_t* tid,
                                        int total,
                                        rocke_sparse_attn_slot_body_fn body,
                                        void* user)
{
    int n_chunks;
    int chunk;

    if(b == NULL || body == NULL)
    {
        return;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return;
    }

    /* if total <= 0: return */
    if(total <= 0)
    {
        return;
    }

    /* for chunk in range((total + _BLOCK_SIZE - 1) // _BLOCK_SIZE): */
    n_chunks = (total + ROCKE_SPARSE_ATTN_BLOCK_SIZE - 1) / ROCKE_SPARSE_ATTN_BLOCK_SIZE;
    for(chunk = 0; chunk < n_chunks; ++chunk)
    {
        int base = chunk * ROCKE_SPARSE_ATTN_BLOCK_SIZE;
        rocke_value_t* slot;

        /* slot = tid if base == 0 else b.add(tid, b.const_i32(base)) */
        if(base == 0)
        {
            slot = tid;
        }
        else
        {
            slot = rocke_b_add(b, tid, rocke_b_const_i32(b, (int64_t)base));
        }

        /* if base + _BLOCK_SIZE <= total: body(slot)
         * else: with b.scf_if(slot < total): body(slot) */
        if(base + ROCKE_SPARSE_ATTN_BLOCK_SIZE <= total)
        {
            body(b, slot, user);
        }
        else
        {
            rocke_value_t* in_range = rocke_b_cmp_lt(b, slot, rocke_b_const_i32(b, (int64_t)total));
            rocke_if_t iff = rocke_b_scf_if(b, in_range);
            rocke_b_region_enter(b, iff.then_region);
            body(b, slot, user);
            rocke_b_region_leave(b);
        }
    }
}

/* --------------------------------------------- _stage_jenga_mask_to_lds */

/* Context for the jenga cooperative-copy body. */
typedef struct rocke_sparse_attn_jenga_ctx
{
    rocke_value_t* mask_global;
    rocke_value_t* mask_row_base;
    rocke_value_t* mask_lds;
} rocke_sparse_attn_jenga_ctx_t;

static void rocke_sparse_attn_jenga_body(rocke_ir_builder_t* b, rocke_value_t* slot, void* user)
{
    rocke_sparse_attn_jenga_ctx_t* ctx = (rocke_sparse_attn_jenga_ctx_t*)user;
    rocke_value_t* mask_off;
    rocke_value_t* mask_byte;
    rocke_value_t* idx[1];

    /* mask_off  = b.add(mask_row_base, slot) */
    mask_off = rocke_b_add(b, ctx->mask_row_base, slot);
    /* mask_byte = b.global_load(mask_global, mask_off, I8)  (align=1 default) */
    mask_byte = rocke_b_global_load(b, ctx->mask_global, mask_off, rocke_i8(), 1);
    /* b.smem_store_vN(mask_lds, [slot], mask_byte, 1) */
    idx[0] = slot;
    rocke_b_smem_store_vN(b, ctx->mask_lds, idx, 1, mask_byte, 1);
}

rocke_value_t* rocke_sparse_attn_stage_jenga_mask_to_lds(rocke_ir_builder_t* b,
                                                         rocke_value_t* mask_global,
                                                         rocke_value_t* mask_row_base,
                                                         int num_k_blocks,
                                                         rocke_value_t* tid)
{
    int shape[1];
    rocke_value_t* mask_lds;
    rocke_sparse_attn_jenga_ctx_t ctx;

    if(b == NULL)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* mask_lds = b.smem_alloc(I8, [num_k_blocks], name_hint="jenga_mask") */
    shape[0] = num_k_blocks;
    mask_lds = rocke_b_smem_alloc(b, rocke_i8(), shape, 1, "jenga_mask");

    ctx.mask_global = mask_global;
    ctx.mask_row_base = mask_row_base;
    ctx.mask_lds = mask_lds;

    /* _cooperative_iter(b, tid=tid, total=num_k_blocks, body=_body) */
    rocke_sparse_attn_cooperative_iter(b, tid, num_k_blocks, rocke_sparse_attn_jenga_body, &ctx);
    return mask_lds;
}

/* --------------------------------------------- _stage_vsa_bitmap_to_lds */

/* Context for the VSA zero pass. */
typedef struct rocke_sparse_attn_vsa_zero_ctx
{
    rocke_value_t* bitmap_lds;
    rocke_value_t* zero_i8;
} rocke_sparse_attn_vsa_zero_ctx_t;

static void rocke_sparse_attn_vsa_zero_body(rocke_ir_builder_t* b, rocke_value_t* slot, void* user)
{
    rocke_sparse_attn_vsa_zero_ctx_t* ctx = (rocke_sparse_attn_vsa_zero_ctx_t*)user;
    rocke_value_t* idx[1];
    /* b.smem_store_vN(bitmap_lds, [slot], zero_i8, 1) */
    idx[0] = slot;
    rocke_b_smem_store_vN(b, ctx->bitmap_lds, idx, 1, ctx->zero_i8, 1);
}

/* Context for the VSA scatter pass. */
typedef struct rocke_sparse_attn_vsa_scatter_ctx
{
    rocke_value_t* bitmap_lds;
    rocke_value_t* one_i8;
    rocke_value_t* block_lut;
    rocke_value_t* lut_row_base;
    rocke_value_t* block_count_v;
} rocke_sparse_attn_vsa_scatter_ctx_t;

/* _scatter_body(slot): the per-lane in-range guard + LUT load + LDS scatter.
 * Mirrors the Python nested closure exactly. */
static void rocke_sparse_attn_vsa_scatter_body(rocke_ir_builder_t* b,
                                               rocke_value_t* slot,
                                               rocke_sparse_attn_vsa_scatter_ctx_t* ctx)
{
    rocke_value_t* in_range;
    rocke_if_t iff;

    /* in_range = b.cmp_lt(slot, block_count_v) */
    in_range = rocke_b_cmp_lt(b, slot, ctx->block_count_v);
    /* with b.scf_if(in_range): */
    iff = rocke_b_scf_if(b, in_range);
    rocke_b_region_enter(b, iff.then_region);
    {
        rocke_value_t* slot_off;
        rocke_value_t* lut_val;
        rocke_value_t* idx[1];

        /* slot_off = b.add(lut_row_base, slot) */
        slot_off = rocke_b_add(b, ctx->lut_row_base, slot);
        /* lut_val  = b.global_load_i32(block_lut, slot_off)  (align=4 default) */
        lut_val = rocke_b_global_load_i32(b, ctx->block_lut, slot_off, 4);
        /* b.smem_store_vN(bitmap_lds, [lut_val], one_i8, 1) */
        idx[0] = lut_val;
        rocke_b_smem_store_vN(b, ctx->bitmap_lds, idx, 1, ctx->one_i8, 1);
    }
    rocke_b_region_leave(b);
}

rocke_value_t* rocke_sparse_attn_stage_vsa_bitmap_to_lds(rocke_ir_builder_t* b,
                                                         rocke_value_t* block_lut,
                                                         rocke_value_t* block_count,
                                                         rocke_value_t* q_block_idx,
                                                         rocke_value_t* lut_row_base,
                                                         int num_k_blocks,
                                                         int max_blocks_per_q,
                                                         rocke_value_t* tid)
{
    int shape[1];
    rocke_value_t* bitmap_lds;
    rocke_value_t* zero_i8;
    rocke_value_t* one_i8;
    rocke_value_t* block_count_v;
    rocke_sparse_attn_vsa_zero_ctx_t zero_ctx;
    rocke_sparse_attn_vsa_scatter_ctx_t scatter_ctx;
    int n_chunks;
    int chunk;

    if(b == NULL)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* bitmap_lds = b.smem_alloc(I8, [num_k_blocks], name_hint="vsa_bitmap") */
    shape[0] = num_k_blocks;
    bitmap_lds = rocke_b_smem_alloc(b, rocke_i8(), shape, 1, "vsa_bitmap");
    /* zero_i8 = _const_i8(b, 0); one_i8 = _const_i8(b, 1) */
    zero_i8 = rocke_sparse_attn_const_i8(b, 0);
    one_i8 = rocke_sparse_attn_const_i8(b, 1);

    /* Pass 1: zero the bitmap (cooperative iter over num_k_blocks). */
    zero_ctx.bitmap_lds = bitmap_lds;
    zero_ctx.zero_i8 = zero_i8;
    rocke_sparse_attn_cooperative_iter(
        b, tid, num_k_blocks, rocke_sparse_attn_vsa_zero_body, &zero_ctx);
    /* b.sync() */
    rocke_b_sync(b);

    /* Pass 2: scatter LUT-pointed slots.
     * block_count_v = b.global_load_i32(block_count, q_block_idx) (align=4) */
    block_count_v = rocke_b_global_load_i32(b, block_count, q_block_idx, 4);

    scatter_ctx.bitmap_lds = bitmap_lds;
    scatter_ctx.one_i8 = one_i8;
    scatter_ctx.block_lut = block_lut;
    scatter_ctx.lut_row_base = lut_row_base;
    scatter_ctx.block_count_v = block_count_v;

    /* The static cooperative iter handles max_blocks_per_q > 64 via chunked
     * walks; the per-lane in-range guard handles slot >= block_count_v. This
     * loop is the explicit form Python writes (NOT _cooperative_iter) so the
     * last partial chunk's static-range check wraps the WHOLE _scatter_body
     * (including the global LUT load), matching the Python op stream exactly.
     *
     * for chunk in range((max_blocks_per_q + _BLOCK_SIZE - 1) // _BLOCK_SIZE): */
    n_chunks = (max_blocks_per_q + ROCKE_SPARSE_ATTN_BLOCK_SIZE - 1) / ROCKE_SPARSE_ATTN_BLOCK_SIZE;
    for(chunk = 0; chunk < n_chunks; ++chunk)
    {
        int base = chunk * ROCKE_SPARSE_ATTN_BLOCK_SIZE;
        rocke_value_t* slot;

        /* slot = tid if base == 0 else b.add(tid, b.const_i32(base)) */
        if(base == 0)
        {
            slot = tid;
        }
        else
        {
            slot = rocke_b_add(b, tid, rocke_b_const_i32(b, (int64_t)base));
        }

        /* if base + _BLOCK_SIZE <= max_blocks_per_q: _scatter_body(slot)
         * else: with b.scf_if(slot < max_blocks_per_q): _scatter_body(slot) */
        if(base + ROCKE_SPARSE_ATTN_BLOCK_SIZE <= max_blocks_per_q)
        {
            rocke_sparse_attn_vsa_scatter_body(b, slot, &scatter_ctx);
        }
        else
        {
            rocke_value_t* in_range
                = rocke_b_cmp_lt(b, slot, rocke_b_const_i32(b, (int64_t)max_blocks_per_q));
            rocke_if_t iff = rocke_b_scf_if(b, in_range);
            rocke_b_region_enter(b, iff.then_region);
            rocke_sparse_attn_vsa_scatter_body(b, slot, &scatter_ctx);
            rocke_b_region_leave(b);
        }
    }
    return bitmap_lds;
}

/* --------------------------------------------- _lds_bitmap_predicate */

rocke_value_t* rocke_sparse_attn_lds_bitmap_predicate(rocke_ir_builder_t* b,
                                                      rocke_value_t* bitmap_lds,
                                                      rocke_value_t* k_block_idx)
{
    rocke_value_t* idx[1];
    rocke_value_t* loaded;
    rocke_value_t* byte_v;

    if(b == NULL)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* loaded = b.smem_load_vN(bitmap_lds, k_block_idx, dtype=I8, n=1) */
    idx[0] = k_block_idx;
    loaded = rocke_b_smem_load_vN(b, bitmap_lds, idx, 1, rocke_i8(), 1);
    /* byte_v = b.vec_extract(loaded, 0) */
    byte_v = rocke_b_vec_extract(b, loaded, 0);
    /* return b.cmp_ne(byte_v, _const_i8(b, 0)) */
    return rocke_b_cmp_ne(b, byte_v, rocke_sparse_attn_const_i8(b, 0));
}
