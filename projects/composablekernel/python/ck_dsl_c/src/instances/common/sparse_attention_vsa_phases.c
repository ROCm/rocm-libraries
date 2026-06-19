/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_sparse_attention_vsa_phases.c -- C99 port of the VSA build-phase
 * functions of ck_dsl/instances/common/sparse_attention.py
 * (build_vsa_sparse_attention prologue + LDS-staging + predicate closure).
 *
 * This part-file implements the VSA build-phase functions over
 * ckc_vsa_sparse_ctx_t declared in ckc/instance_sparse_attention_internal.h:
 *
 *   Python (sparse_attention.py)              C99 (this TU)
 *   ---------------------------------------   ------------------------------------
 *   _magic_div(b, dividend, divisor)          ckc_sparse_attn_magic_div  (shared)
 *   _declare_vsa_params(kb)  (268-278)        ckc_vsa_declare_params
 *   build_vsa_sparse_attention prologue       ckc_vsa_prologue
 *     (563-587)
 *   stage + sync + tiles_per_block_k          ckc_vsa_stage_bitmap
 *     (589-602)
 *   _vsa_tile_predicate(b, kt) (604-613)      ckc_vsa_tile_predicate
 *
 * Each function reproduces its Python counterpart's ckc_b_* / helper builder-call
 * sequence byte-faithfully: same ops, same order, same operands, same result-name
 * hints. The body-emit / signature peers live in sibling TUs and are bound via
 * the internal header.
 *
 * Lifetime: every node is arena-owned (ckc_ir_builder_t.arena via the embedded
 * FmhaKernelBuilder). Nothing is freed individually.
 */

#include "ckc/instance_sparse_attention_internal.h"
#include "ckc/instance_sparse_attention.h"
#include "ckc/ir.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"
#include "ckc/helper_ck_dsl.instances.common.sparse_attention.h"
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h"
#include "ckc/helper_ck_dsl.helpers.transforms.h"

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
ckc_value_t* ckc_sparse_attn_magic_div(ckc_ir_builder_t* b, ckc_value_t* dividend, int divisor)
{
    uint64_t mult;
    int shift;

    if(b == NULL)
    {
        return NULL;
    }
    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }

    /* mult, shift = calculate_magic_numbers(divisor) */
    if(!ckc_calculate_magic_numbers(b, divisor, &mult, &shift))
    {
        return NULL;
    }
    /* return do_magic_division(b, dividend, mult, shift) */
    return ckc_do_magic_division(b, dividend, mult, shift);
}

/* ===================================================================== *
 *  _declare_vsa_params(kb)  (Python lines 268-278).
 * ===================================================================== */
void ckc_vsa_declare_params(ckc_vsa_sparse_ctx_t* ctx)
{
    ckc_fmha_kernel_builder_t* kb;
    const char* stride_names[4];

    if(ctx == NULL)
    {
        return;
    }
    kb = &ctx->kb;

    /* kb.add_tensor("Q", readonly=True) */
    ckc_fmha_kernel_builder_add_tensor(kb, "Q", NULL, /*readonly=*/true, /*writeonly=*/false, 16);
    /* kb.add_tensor("K", readonly=True) */
    ckc_fmha_kernel_builder_add_tensor(kb, "K", NULL, /*readonly=*/true, /*writeonly=*/false, 16);
    /* kb.add_tensor("V", readonly=True) */
    ckc_fmha_kernel_builder_add_tensor(kb, "V", NULL, /*readonly=*/true, /*writeonly=*/false, 16);
    /* kb.add_tensor("O", readonly=False, writeonly=True) */
    ckc_fmha_kernel_builder_add_tensor(kb, "O", NULL, /*readonly=*/false, /*writeonly=*/true, 16);
    /* kb.add_ptr("block_lut", dtype="i32", readonly=True)  (align=4 default) */
    ckc_fmha_kernel_builder_add_ptr(kb, "block_lut", "i32", /*readonly=*/true, 4);
    /* kb.add_ptr("block_count", dtype="i32", readonly=True) */
    ckc_fmha_kernel_builder_add_ptr(kb, "block_count", "i32", /*readonly=*/true, 4);
    /* kb.add_scalar("scale_log2", "f32") */
    ckc_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    /* kb.add_scalar("seqlen_q", "i32") */
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    /* kb.add_scalar("seqlen_k", "i32") */
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");
    /* kb.add_strides("q", "k", "v", "o") */
    stride_names[0] = "q";
    stride_names[1] = "k";
    stride_names[2] = "v";
    stride_names[3] = "o";
    ckc_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* ===================================================================== *
 *  build_vsa_sparse_attention prologue  (Python lines 563-587).
 * ===================================================================== */
bool ckc_vsa_prologue(ckc_vsa_sparse_ctx_t* ctx)
{
    char reason[CKC_ERR_MSG_CAP];
    const char* kernel_name;
    char name_buf[256];
    ckc_fmha_kernel_builder_t* kb;
    ckc_ir_builder_t* b;
    ckc_value_t* mfma_block_m;

    if(ctx == NULL)
    {
        return false;
    }

    /* ok, why = is_valid_vsa_spec(spec, arch)
     * if not ok: raise ValueError(f"invalid vsa_sparse spec: {why}") */
    reason[0] = '\0';
    if(!ckc_is_valid_vsa_spec(ctx->spec, ctx->arch, reason, sizeof(reason)))
    {
        return false;
    }

    /* s = spec.common */
    ctx->s = ctx->spec->common;

    /* kb = FmhaKernelBuilder(spec.kernel_name(), s) */
    name_buf[0] = '\0';
    if(ckc_vsa_sparse_kernel_name(ctx->spec, name_buf, sizeof(name_buf)) != CKC_OK)
    {
        return false;
    }
    kernel_name = name_buf;
    if(ckc_fmha_kernel_builder_init(&ctx->kb, kernel_name, &ctx->s) != CKC_OK)
    {
        return false;
    }
    kb = &ctx->kb;

    /* kb.block_size(_BLOCK_SIZE) */
    ckc_fmha_kernel_builder_block_size(kb, CKC_SPARSE_ATTN_BLOCK_SIZE);

    /* _declare_vsa_params(kb) */
    ckc_vsa_declare_params(ctx);

    /* kb.decode_grid() */
    ckc_fmha_kernel_builder_decode_grid(kb,
                                        /*num_queries_per_kv=*/-1,
                                        /*has_batch_axis=*/false,
                                        &ctx->q_tile_idx,
                                        &ctx->head_idx,
                                        &ctx->kv_head_idx);

    /* b = kb.builder */
    ctx->b = ckc_fmha_kernel_builder_builder(kb);
    b      = ctx->b;

    /* Q = kb.tensor("Q") ... O = kb.tensor("O") */
    ctx->Q = ckc_fmha_kernel_builder_tensor(kb, "Q");
    ctx->K = ckc_fmha_kernel_builder_tensor(kb, "K");
    ctx->V = ckc_fmha_kernel_builder_tensor(kb, "V");
    ctx->O = ckc_fmha_kernel_builder_tensor(kb, "O");
    /* block_lut = kb.ptr("block_lut"); block_count = kb.ptr("block_count") */
    ctx->block_lut   = ckc_fmha_kernel_builder_ptr(kb, "block_lut");
    ctx->block_count = ckc_fmha_kernel_builder_ptr(kb, "block_count");
    /* scale_log2 = kb.scalar("scale_log2"); seqlen_k_arg = kb.scalar("seqlen_k") */
    ctx->scale_log2   = ckc_fmha_kernel_builder_scalar(kb, "scale_log2");
    ctx->seqlen_k_arg = ckc_fmha_kernel_builder_scalar(kb, "seqlen_k");

    /* q_tile_idx = kb.q_token; head_idx = kb.head_idx; kv_head_idx = kb.kv_head_idx
     * (already populated by decode_grid above; mirror the field reads). */
    ctx->q_tile_idx  = kb->q_token;
    ctx->head_idx    = kb->head_idx;
    ctx->kv_head_idx = kb->kv_head_idx;

    /* q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M)) */
    mfma_block_m       = ckc_b_const_i32(b, (int64_t)CKC_MFMA_ATTN_BLOCK_M);
    ctx->q_tile_base   = ckc_b_mul(b, ctx->q_tile_idx, mfma_block_m);
    /* q_block_idx = _magic_div(b, q_tile_base, spec.block_q) */
    ctx->q_block_idx   = ckc_sparse_attn_magic_div(b, ctx->q_tile_base, ctx->spec->block_q);
    /* lut_row_base = b.mul(q_block_idx, b.const_i32(spec.max_blocks_per_q)) */
    ctx->lut_row_base  = ckc_b_mul(
        b, ctx->q_block_idx, ckc_b_const_i32(b, (int64_t)ctx->spec->max_blocks_per_q));

    return ckc_ir_builder_ok(b);
}

/* ===================================================================== *
 *  LDS staging + sync + tiles_per_block_k  (Python lines 589-602).
 * ===================================================================== */
void ckc_vsa_stage_bitmap(ckc_vsa_sparse_ctx_t* ctx)
{
    ckc_ir_builder_t* b;

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
    ctx->tid = ckc_b_thread_id_x(b);

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
    ctx->num_k_blocks     = ckc_vsa_sparse_spec_num_k_blocks(ctx->spec);
    ctx->max_blocks_per_q = ctx->spec->max_blocks_per_q;
    ctx->bitmap_lds       = ckc_sparse_attn_stage_vsa_bitmap_to_lds(b,
                                                              ctx->block_lut,
                                                              ctx->block_count,
                                                              ctx->q_block_idx,
                                                              ctx->lut_row_base,
                                                              ctx->num_k_blocks,
                                                              ctx->max_blocks_per_q,
                                                              ctx->tid);

    /* b.sync() */
    ckc_b_sync(b);

    /* tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K */
    ctx->tiles_per_block_k = ctx->spec->block_k / CKC_MFMA_ATTN_BLOCK_K;
}

/* ===================================================================== *
 *  _vsa_tile_predicate(b, kt)  (Python lines 604-613).
 *
 *  The extra_mask_predicate callback. `user` is a ckc_vsa_sparse_ctx_t*. Matches
 *  ckc_attn_predicate_fn so it can be handed straight to
 *  mfma_attention_fwd_inner_body.
 * ===================================================================== */
ckc_value_t* ckc_vsa_tile_predicate(ckc_ir_builder_t* b, ckc_value_t* kt, void* user)
{
    ckc_vsa_sparse_ctx_t* ctx = (ckc_vsa_sparse_ctx_t*)user;
    ckc_value_t* k_block_idx;

    if(b == NULL || ctx == NULL)
    {
        return NULL;
    }
    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }

    /* k_block_idx = _magic_div(b, kt, tiles_per_block_k) */
    k_block_idx = ckc_sparse_attn_magic_div(b, kt, ctx->tiles_per_block_k);
    /* return _lds_bitmap_predicate(b, bitmap_lds, k_block_idx) */
    return ckc_sparse_attn_lds_bitmap_predicate(b, ctx->bitmap_lds, k_block_idx);
}
