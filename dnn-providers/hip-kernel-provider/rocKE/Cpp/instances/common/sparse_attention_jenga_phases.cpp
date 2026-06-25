// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sparse_attention_jenga_phases.c -- C99 port of the Jenga build phases
 * of ck_dsl/instances/common/sparse_attention.py (build_jenga_sparse_attention).
 *
 * This translation unit implements the Jenga prologue / staging / tile-predicate
 * phase functions over ckc_jenga_sparse_ctx_t (declared in
 * ckc/instance_sparse_attention_internal.h):
 *
 *   ckc_jenga_declare_params  <- Python _declare_jenga_params (lines 256-265)
 *   ckc_jenga_prologue        <- build_jenga_sparse_attention prologue (lines 466-490)
 *   ckc_jenga_stage_mask      <- the LDS stage (lines 492-506)
 *   ckc_jenga_tile_predicate  <- the _jenga_tile_predicate closure (lines 508-511)
 *
 * Each phase reproduces its Python counterpart's builder-call sequence
 * byte-faithfully: same ops, same order, same operands. The host-side control
 * structure is reproduced exactly so the emitted op stream is identical.
 *
 * Closure capture: Python's `_jenga_tile_predicate(b, kt)` captures the
 * enclosing-function locals (mask_lds, tiles_per_block_k); in C those are read
 * from the shared ckc_jenga_sparse_ctx_t handed in via the predicate `user`.
 *
 * Peer phases (ckc_jenga_emit_body, ckc_sparse_attn_magic_div, the validity
 * gate, the spec helpers, and the LDS-bitmap primitives) live in sibling TUs and
 * are reached through the internal / public headers.
 *
 * Lifetime: every node is arena-owned (ckc_fmha_kernel_builder_t's embedded IR
 * builder). Nothing is freed individually here.
 */

#include "ckc/helper_ck_dsl.helpers.mfma_attention.h" /* CKC_MFMA_ATTN_BLOCK_M/K */
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"
#include "ckc/helper_ck_dsl.instances.common.sparse_attention.h"
#include "ckc/instance_sparse_attention.h"
#include "ckc/instance_sparse_attention_internal.h"
#include "ckc/ir.h"

/* ------------------------------------------------------------------ *
 * ckc_jenga_declare_params  --  Python _declare_jenga_params (256-265)
 * ------------------------------------------------------------------ *
 *
 *   kb.add_tensor("Q", readonly=True)
 *   kb.add_tensor("K", readonly=True)
 *   kb.add_tensor("V", readonly=True)
 *   kb.add_tensor("O", readonly=False, writeonly=True)
 *   kb.add_ptr("mask", dtype="i8", readonly=True, align=1)
 *   kb.add_scalar("scale_log2", "f32")
 *   kb.add_scalar("seqlen_q", "i32")
 *   kb.add_scalar("seqlen_k", "i32")
 *   kb.add_strides("q", "k", "v", "o")
 */
void ckc_jenga_declare_params(ckc_jenga_sparse_ctx_t* ctx)
{
    static const char* const stride_names[4] = {"q", "k", "v", "o"};

    if(ctx == NULL)
    {
        return;
    }

    /* kb.add_tensor("Q", readonly=True) -- dtype=None => common.dtype, align=16 */
    ckc_fmha_kernel_builder_add_tensor(&ctx->kb, "Q", NULL, true, false, 16);
    /* kb.add_tensor("K", readonly=True) */
    ckc_fmha_kernel_builder_add_tensor(&ctx->kb, "K", NULL, true, false, 16);
    /* kb.add_tensor("V", readonly=True) */
    ckc_fmha_kernel_builder_add_tensor(&ctx->kb, "V", NULL, true, false, 16);
    /* kb.add_tensor("O", readonly=False, writeonly=True) */
    ckc_fmha_kernel_builder_add_tensor(&ctx->kb, "O", NULL, false, true, 16);
    /* kb.add_ptr("mask", dtype="i8", readonly=True, align=1) */
    ckc_fmha_kernel_builder_add_ptr(&ctx->kb, "mask", "i8", true, 1);
    /* kb.add_scalar("scale_log2", "f32") */
    ckc_fmha_kernel_builder_add_scalar(&ctx->kb, "scale_log2", "f32");
    /* kb.add_scalar("seqlen_q", "i32") */
    ckc_fmha_kernel_builder_add_scalar(&ctx->kb, "seqlen_q", "i32");
    /* kb.add_scalar("seqlen_k", "i32") */
    ckc_fmha_kernel_builder_add_scalar(&ctx->kb, "seqlen_k", "i32");
    /* kb.add_strides("q", "k", "v", "o") */
    ckc_fmha_kernel_builder_add_strides(&ctx->kb, stride_names, 4);
}

/* ------------------------------------------------------------------ *
 * ckc_jenga_prologue  --  build_jenga_sparse_attention prologue (466-490)
 * ------------------------------------------------------------------ *
 *
 *   ok, why = is_valid_jenga_spec(spec, arch)
 *   if not ok: raise ValueError(...)
 *   s = spec.common
 *   kb = FmhaKernelBuilder(spec.kernel_name(), s)
 *   kb.block_size(_BLOCK_SIZE)
 *   _declare_jenga_params(kb)
 *   kb.decode_grid()
 *   b = kb.builder
 *   Q = kb.tensor("Q"); K = kb.tensor("K"); V = kb.tensor("V"); O = kb.tensor("O")
 *   mask = kb.ptr("mask")
 *   scale_log2 = kb.scalar("scale_log2"); seqlen_k_arg = kb.scalar("seqlen_k")
 *   q_tile_idx = kb.q_token; head_idx = kb.head_idx; kv_head_idx = kb.kv_head_idx
 *   q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
 *   q_block_idx = _magic_div(b, q_tile_base, spec.block_q)
 *   mask_row_base = b.mul(q_block_idx, b.const_i32(spec.num_k_blocks))
 */
bool ckc_jenga_prologue(ckc_jenga_sparse_ctx_t* ctx)
{
    char kernel_name[256];
    ckc_status_t st;

    if(ctx == NULL || ctx->spec == NULL)
    {
        return false;
    }

    /* ok, why = is_valid_jenga_spec(spec, arch); if not ok: raise ValueError */
    if(!ckc_is_valid_jenga_spec(ctx->spec, ctx->arch, NULL, 0))
    {
        return false;
    }

    /* s = spec.common */
    ctx->s = ctx->spec->common;

    /* kb = FmhaKernelBuilder(spec.kernel_name(), s) */
    st = ckc_jenga_sparse_kernel_name(ctx->spec, kernel_name, sizeof(kernel_name));
    if(st != CKC_OK)
    {
        return false;
    }
    st = ckc_fmha_kernel_builder_init(&ctx->kb, kernel_name, &ctx->s);
    if(st != CKC_OK)
    {
        return false;
    }

    /* kb.block_size(_BLOCK_SIZE) */
    ckc_fmha_kernel_builder_block_size(&ctx->kb, CKC_SPARSE_ATTN_BLOCK_SIZE);

    /* _declare_jenga_params(kb) */
    ckc_jenga_declare_params(ctx);

    /* kb.decode_grid() */
    ckc_fmha_kernel_builder_decode_grid(&ctx->kb,
                                        -1, /* num_queries_per_kv=None */
                                        false /* has_batch_axis=False */,
                                        &ctx->q_tile_idx,
                                        &ctx->head_idx,
                                        &ctx->kv_head_idx);

    /* b = kb.builder */
    ctx->b = ckc_fmha_kernel_builder_builder(&ctx->kb);

    /* geometry scalars cached on ctx (Python spec.num_k_blocks). */
    ctx->num_k_blocks = ckc_jenga_sparse_spec_num_k_blocks(ctx->spec);

    /* Q = kb.tensor("Q") ... O = kb.tensor("O") */
    ctx->Q = ckc_fmha_kernel_builder_tensor(&ctx->kb, "Q");
    ctx->K = ckc_fmha_kernel_builder_tensor(&ctx->kb, "K");
    ctx->V = ckc_fmha_kernel_builder_tensor(&ctx->kb, "V");
    ctx->O = ckc_fmha_kernel_builder_tensor(&ctx->kb, "O");
    /* mask = kb.ptr("mask") */
    ctx->mask = ckc_fmha_kernel_builder_ptr(&ctx->kb, "mask");
    /* scale_log2 = kb.scalar("scale_log2"); seqlen_k_arg = kb.scalar("seqlen_k") */
    ctx->scale_log2 = ckc_fmha_kernel_builder_scalar(&ctx->kb, "scale_log2");
    ctx->seqlen_k_arg = ckc_fmha_kernel_builder_scalar(&ctx->kb, "seqlen_k");

    /* q_tile_idx/head_idx/kv_head_idx already populated by decode_grid (kb.q_token,
     * kb.head_idx, kb.kv_head_idx). */

    /* q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M)) */
    ctx->q_tile_base
        = ckc_b_mul(ctx->b, ctx->q_tile_idx, ckc_b_const_i32(ctx->b, CKC_MFMA_ATTN_BLOCK_M));
    /* q_block_idx = _magic_div(b, q_tile_base, spec.block_q) */
    ctx->q_block_idx = ckc_sparse_attn_magic_div(ctx->b, ctx->q_tile_base, ctx->spec->block_q);
    /* mask_row_base = b.mul(q_block_idx, b.const_i32(spec.num_k_blocks)) */
    ctx->mask_row_base
        = ckc_b_mul(ctx->b, ctx->q_block_idx, ckc_b_const_i32(ctx->b, ctx->num_k_blocks));

    return ckc_ir_builder_ok(ctx->b);
}

/* ------------------------------------------------------------------ *
 * ckc_jenga_stage_mask  --  the LDS stage (492-506)
 * ------------------------------------------------------------------ *
 *
 *   tid = b.thread_id_x()
 *   mask_lds = _stage_jenga_mask_to_lds(
 *       b, mask_global=mask, mask_row_base=mask_row_base,
 *       num_k_blocks=spec.num_k_blocks, tid=tid)
 *   b.sync()
 *   tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K
 */
void ckc_jenga_stage_mask(ckc_jenga_sparse_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return;
    }

    /* tid = b.thread_id_x() */
    ctx->tid = ckc_b_thread_id_x(ctx->b);

    /* mask_lds = _stage_jenga_mask_to_lds(...) */
    ctx->mask_lds = ckc_sparse_attn_stage_jenga_mask_to_lds(
        ctx->b, ctx->mask, ctx->mask_row_base, ctx->num_k_blocks, ctx->tid);

    /* b.sync() */
    ckc_b_sync(ctx->b);

    /* tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K */
    ctx->tiles_per_block_k = ctx->spec->block_k / CKC_MFMA_ATTN_BLOCK_K;
}

/* ------------------------------------------------------------------ *
 * ckc_jenga_tile_predicate  --  _jenga_tile_predicate(b, kt) (508-511)
 * ------------------------------------------------------------------ *
 *
 *   def _jenga_tile_predicate(b, kt):
 *       k_block_idx = _magic_div(b, kt, tiles_per_block_k)
 *       return _lds_bitmap_predicate(b, mask_lds, k_block_idx)
 *
 * `user` is the capturing ckc_jenga_sparse_ctx_t* (mask_lds + tiles_per_block_k).
 * Signature matches ckc_attn_predicate_fn for the inner-body hook.
 */
ckc_value_t* ckc_jenga_tile_predicate(ckc_ir_builder_t* b, ckc_value_t* kt, void* user)
{
    ckc_jenga_sparse_ctx_t* ctx = (ckc_jenga_sparse_ctx_t*)user;
    ckc_value_t* k_block_idx;

    if(ctx == NULL)
    {
        return NULL;
    }

    /* k_block_idx = _magic_div(b, kt, tiles_per_block_k) */
    k_block_idx = ckc_sparse_attn_magic_div(b, kt, ctx->tiles_per_block_k);
    /* return _lds_bitmap_predicate(b, mask_lds, k_block_idx) */
    return ckc_sparse_attn_lds_bitmap_predicate(b, ctx->mask_lds, k_block_idx);
}
