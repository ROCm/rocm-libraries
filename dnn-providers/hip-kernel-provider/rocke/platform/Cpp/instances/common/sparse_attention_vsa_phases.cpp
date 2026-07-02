// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sparse_attention_vsa_phases.c -- C99 port of the VSA build-phase
 * functions of rocke/instances/common/sparse_attention.py
 * (build_vsa_sparse_attention prologue + LDS-staging + predicate closure).
 *
 * This part-file implements the VSA build-phase functions over
 * rocke_vsa_sparse_ctx_t declared in rocke/instance_sparse_attention_internal.h:
 *
 *   Python (sparse_attention.py)              C99 (this TU)
 *   ---------------------------------------   ------------------------------------
 *   _magic_div(b, dividend, divisor)          rocke_sparse_attn_magic_div  (shared)
 *   _declare_vsa_params(kb)  (268-278)        rocke_vsa_declare_params
 *   build_vsa_sparse_attention prologue       rocke_vsa_prologue
 *     (563-587)
 *   stage + sync + tiles_per_block_k          rocke_vsa_stage_bitmap
 *     (589-602)
 *   _vsa_tile_predicate(b, kt) (604-613)      rocke_vsa_tile_predicate
 *
 * Each function reproduces its Python counterpart's rocke_b_* / helper builder-call
 * sequence byte-faithfully: same ops, same order, same operands, same result-name
 * hints. The body-emit / signature peers live in sibling TUs and are bound via
 * the internal header.
 *
 * Lifetime: every node is arena-owned (rocke_ir_builder_t.arena via the embedded
 * FmhaKernelBuilder). Nothing is freed individually.
 */

#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/helper_rocke.instances.common.sparse_attention.h"
#include "rocke/instance_sparse_attention.h"
#include "rocke/instance_sparse_attention_internal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  _magic_div(b, dividend, divisor): dividend // divisor via CK Tile's magic
 *  mul-hi division.
 *
 *  Python:
 *      def _magic_div(b, dividend, divisor):
 *          mult, shift = calculate_magic_numbers(divisor)
 *          return do_magic_division(b, dividend, mult, shift)
 *
 *  Shared by the q_block / k_block index decodes and both predicate closures.
 * ===================================================================== */
rocke_value_t*
    rocke_sparse_attn_magic_div(rocke_ir_builder_t* b, rocke_value_t* dividend, int divisor)
{
    uint64_t mult;
    int shift;

    if(b == NULL)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* mult, shift = calculate_magic_numbers(divisor) */
    if(!rocke_calculate_magic_numbers(b, divisor, &mult, &shift))
    {
        return NULL;
    }
    /* return do_magic_division(b, dividend, mult, shift) */
    return rocke_do_magic_division(b, dividend, mult, shift);
}

/* ===================================================================== *
 *  _declare_vsa_params(kb)  (Python lines 268-278).
 * ===================================================================== */
void rocke_vsa_declare_params(rocke_vsa_sparse_ctx_t* ctx)
{
    rocke_fmha_kernel_builder_t* kb;
    const char* stride_names[4];

    if(ctx == NULL)
    {
        return;
    }
    kb = &ctx->kb;

    /* kb.add_tensor("Q", readonly=True) */
    rocke_fmha_kernel_builder_add_tensor(kb, "Q", NULL, /*readonly=*/true, /*writeonly=*/false, 16);
    /* kb.add_tensor("K", readonly=True) */
    rocke_fmha_kernel_builder_add_tensor(kb, "K", NULL, /*readonly=*/true, /*writeonly=*/false, 16);
    /* kb.add_tensor("V", readonly=True) */
    rocke_fmha_kernel_builder_add_tensor(kb, "V", NULL, /*readonly=*/true, /*writeonly=*/false, 16);
    /* kb.add_tensor("O", readonly=False, writeonly=True) */
    rocke_fmha_kernel_builder_add_tensor(kb, "O", NULL, /*readonly=*/false, /*writeonly=*/true, 16);
    /* kb.add_ptr("block_lut", dtype="i32", readonly=True)  (align=4 default) */
    rocke_fmha_kernel_builder_add_ptr(kb, "block_lut", "i32", /*readonly=*/true, 4);
    /* kb.add_ptr("block_count", dtype="i32", readonly=True) */
    rocke_fmha_kernel_builder_add_ptr(kb, "block_count", "i32", /*readonly=*/true, 4);
    /* kb.add_scalar("scale_log2", "f32") */
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    /* kb.add_scalar("seqlen_q", "i32") */
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    /* kb.add_scalar("seqlen_k", "i32") */
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");
    /* kb.add_strides("q", "k", "v", "o") */
    stride_names[0] = "q";
    stride_names[1] = "k";
    stride_names[2] = "v";
    stride_names[3] = "o";
    rocke_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* ===================================================================== *
 *  build_vsa_sparse_attention prologue  (Python lines 563-587).
 * ===================================================================== */
bool rocke_vsa_prologue(rocke_vsa_sparse_ctx_t* ctx)
{
    char reason[ROCKE_ERR_MSG_CAP];
    const char* kernel_name;
    char name_buf[256];
    rocke_fmha_kernel_builder_t* kb;
    rocke_ir_builder_t* b;
    rocke_value_t* mfma_block_m;

    if(ctx == NULL)
    {
        return false;
    }

    /* ok, why = is_valid_vsa_spec(spec, arch)
     * if not ok: raise ValueError(f"invalid vsa_sparse spec: {why}") */
    reason[0] = '\0';
    if(!rocke_is_valid_vsa_spec(ctx->spec, ctx->arch, reason, sizeof(reason)))
    {
        return false;
    }

    /* s = spec.common */
    ctx->s = ctx->spec->common;

    /* kb = FmhaKernelBuilder(spec.kernel_name(), s) */
    name_buf[0] = '\0';
    if(rocke_vsa_sparse_kernel_name(ctx->spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
    {
        return false;
    }
    kernel_name = name_buf;
    if(rocke_fmha_kernel_builder_init(&ctx->kb, kernel_name, &ctx->s) != ROCKE_OK)
    {
        return false;
    }
    kb = &ctx->kb;

    /* kb.block_size(_BLOCK_SIZE) */
    rocke_fmha_kernel_builder_block_size(kb, ROCKE_SPARSE_ATTN_BLOCK_SIZE);

    /* _declare_vsa_params(kb) */
    rocke_vsa_declare_params(ctx);

    /* kb.decode_grid() */
    rocke_fmha_kernel_builder_decode_grid(kb,
                                          /*num_queries_per_kv=*/-1,
                                          /*has_batch_axis=*/false,
                                          &ctx->q_tile_idx,
                                          &ctx->head_idx,
                                          &ctx->kv_head_idx);

    /* b = kb.builder */
    ctx->b = rocke_fmha_kernel_builder_builder(kb);
    b = ctx->b;

    /* Q = kb.tensor("Q") ... O = kb.tensor("O") */
    ctx->Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
    ctx->K = rocke_fmha_kernel_builder_tensor(kb, "K");
    ctx->V = rocke_fmha_kernel_builder_tensor(kb, "V");
    ctx->O = rocke_fmha_kernel_builder_tensor(kb, "O");
    /* block_lut = kb.ptr("block_lut"); block_count = kb.ptr("block_count") */
    ctx->block_lut = rocke_fmha_kernel_builder_ptr(kb, "block_lut");
    ctx->block_count = rocke_fmha_kernel_builder_ptr(kb, "block_count");
    /* scale_log2 = kb.scalar("scale_log2"); seqlen_k_arg = kb.scalar("seqlen_k") */
    ctx->scale_log2 = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");
    ctx->seqlen_k_arg = rocke_fmha_kernel_builder_scalar(kb, "seqlen_k");

    /* q_tile_idx = kb.q_token; head_idx = kb.head_idx; kv_head_idx = kb.kv_head_idx
     * (already populated by decode_grid above; mirror the field reads). */
    ctx->q_tile_idx = kb->q_token;
    ctx->head_idx = kb->head_idx;
    ctx->kv_head_idx = kb->kv_head_idx;

    /* q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M)) */
    mfma_block_m = rocke_b_const_i32(b, (int64_t)ROCKE_MFMA_ATTN_BLOCK_M);
    ctx->q_tile_base = rocke_b_mul(b, ctx->q_tile_idx, mfma_block_m);
    /* q_block_idx = _magic_div(b, q_tile_base, spec.block_q) */
    ctx->q_block_idx = rocke_sparse_attn_magic_div(b, ctx->q_tile_base, ctx->spec->block_q);
    /* lut_row_base = b.mul(q_block_idx, b.const_i32(spec.max_blocks_per_q)) */
    ctx->lut_row_base = rocke_b_mul(
        b, ctx->q_block_idx, rocke_b_const_i32(b, (int64_t)ctx->spec->max_blocks_per_q));

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  LDS staging + sync + tiles_per_block_k  (Python lines 589-602).
 * ===================================================================== */
void rocke_vsa_stage_bitmap(rocke_vsa_sparse_ctx_t* ctx)
{
    rocke_ir_builder_t* b;

    if(ctx == NULL)
    {
        return;
    }
    b = ctx->b;
    if(b == NULL)
    {
        return;
    }

    /* tid = b.thread_id_x() */
    ctx->tid = rocke_b_thread_id_x(b);

    /* bitmap_lds = _stage_vsa_bitmap_to_lds(
     *     b,
     *     block_lut=block_lut,
     *     block_count=block_count,
     *     q_block_idx=q_block_idx,
     *     lut_row_base=lut_row_base,
     *     num_k_blocks=spec.num_k_blocks,
     *     max_blocks_per_q=spec.max_blocks_per_q,
     *     tid=tid,
     * ) */
    ctx->num_k_blocks = rocke_vsa_sparse_spec_num_k_blocks(ctx->spec);
    ctx->max_blocks_per_q = ctx->spec->max_blocks_per_q;
    ctx->bitmap_lds = rocke_sparse_attn_stage_vsa_bitmap_to_lds(b,
                                                                ctx->block_lut,
                                                                ctx->block_count,
                                                                ctx->q_block_idx,
                                                                ctx->lut_row_base,
                                                                ctx->num_k_blocks,
                                                                ctx->max_blocks_per_q,
                                                                ctx->tid);

    /* b.sync() */
    rocke_b_sync(b);

    /* tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K */
    ctx->tiles_per_block_k = ctx->spec->block_k / ROCKE_MFMA_ATTN_BLOCK_K;
}

/* ===================================================================== *
 *  _vsa_tile_predicate(b, kt)  (Python lines 604-613).
 *
 *  The extra_mask_predicate callback. `user` is a rocke_vsa_sparse_ctx_t*. Matches
 *  rocke_attn_predicate_fn so it can be handed straight to
 *  mfma_attention_fwd_inner_body.
 * ===================================================================== */
rocke_value_t* rocke_vsa_tile_predicate(rocke_ir_builder_t* b, rocke_value_t* kt, void* user)
{
    rocke_vsa_sparse_ctx_t* ctx = (rocke_vsa_sparse_ctx_t*)user;
    rocke_value_t* k_block_idx;

    if(b == NULL || ctx == NULL)
    {
        return NULL;
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* k_block_idx = _magic_div(b, kt, tiles_per_block_k) */
    k_block_idx = rocke_sparse_attn_magic_div(b, kt, ctx->tiles_per_block_k);
    /* return _lds_bitmap_predicate(b, bitmap_lds, k_block_idx) */
    return rocke_sparse_attn_lds_bitmap_predicate(b, ctx->bitmap_lds, k_block_idx);
}
