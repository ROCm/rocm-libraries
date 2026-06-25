/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_sparse_attention_internal.h -- PRIVATE shared state + phase-
 * function contract for the C99 port of build_jenga_sparse_attention and
 * build_vsa_sparse_attention (ck_dsl/instances/common/sparse_attention.py).
 *
 * WHY THIS HEADER EXISTS.
 *   Each Python builder is a prologue that computes a block of enclosing-function
 *   locals (the FmhaKernelBuilder, the param Values, the grid decode, the
 *   sparsity-block index, the LDS bitmap handle, tiles_per_block_k) and then
 *   defines an inner closure -- `_jenga_tile_predicate(b, kt)` /
 *   `_vsa_tile_predicate(b, kt)` -- that CAPTURES those locals and is handed to
 *   mfma_attention_fwd_inner_body as extra_mask_predicate. The predicate is
 *   replayed once per MFMA K-tile from inside the inner body.
 *
 *   In C there is no closure capture. The faithful port turns each predicate
 *   closure into a free function taking the opaque `user` of the
 *   ckc_attn_predicate_fn callback, where `user` points at one shared context
 *   struct (ckc_jenga_sparse_ctx_t / ckc_vsa_sparse_ctx_t) holding EXACTLY the
 *   captured locals (the builder, the LDS bitmap handle, tiles_per_block_k). The
 *   driver populates the ctx in the same order the Python prologue computes its
 *   locals, then calls the phase functions in Python order and threads `&ctx`
 *   through mfma_attention_fwd_inner_body's extra_mask_predicate_user.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c TU binds to.
 *   It is DESIGNED TO BE COMPLETE: every local the Python body shares across the
 *   prologue / predicate closure / inner-body call is a field here. A body agent
 *   implementing a phase MUST be able to read/write only ctx fields and call the
 *   prototypes below WITHOUT editing this header. If a phase genuinely needs a
 *   value not present, that is a design bug to fix here once, deliberately.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `q_tile_base`
 *   -> `ctx->q_tile_base`; Python `mask_row_base` -> `ctx->mask_row_base`). Phase
 *   functions mirror the Python helper / closure names with a `ckc_jenga_` /
 *   `ckc_vsa_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. Included only by the
 * instance_sparse_attention*.c translation units. Public callers use
 * ckc/instance_sparse_attention.h.
 */
#ifndef CKC_INSTANCE_SPARSE_ATTENTION_INTERNAL_H
#define CKC_INSTANCE_SPARSE_ATTENTION_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/helper_ck_dsl.instances.common._fmha_common.h" /* ckc_fmha_kernel_builder_t */
#include "ckc/instance_sparse_attention.h"
#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  _magic_div(b, dividend, divisor): dividend // divisor via CK Tile's magic
 *  mul-hi division. Internal to this instance (the Python module-private
 *  _magic_div). Calls ckc_calculate_magic_numbers(divisor) -> (mult, shift) then
 *  ckc_do_magic_division(b, dividend, mult, shift). Returns the quotient Value or
 *  NULL on a builder error. Used by the q_block / k_block index decodes and the
 *  predicate closures.
 * ===================================================================== */
ckc_value_t* ckc_sparse_attn_magic_div(ckc_ir_builder_t* b, ckc_value_t* dividend, int divisor);

/* ===================================================================== *
 *  ckc_jenga_sparse_ctx_t  --  shared state for build_jenga_sparse_attention.
 *
 *  Field order follows the Python prologue top-to-bottom (lines 466-538) so the
 *  populate routine reads straight against the source.
 * ===================================================================== */
typedef struct ckc_jenga_sparse_ctx
{
    /* ---- inputs / resolved environment -- */
    const ckc_jenga_sparse_spec_t* spec; /* the JengaSparseSpec               */
    const char* arch; /* NULL-normalised "gfx950"          */
    ckc_fmha_common_spec_t s; /* spec->common (Python `s`)         */

    /* ---- the kernel builder + its underlying IR builder -- */
    ckc_fmha_kernel_builder_t kb; /* FmhaKernelBuilder(spec.kernel_name(), s) */
    ckc_ir_builder_t* b; /* kb.builder (== &kb.b)                    */

    /* ---- geometry scalars (host ints) -- */
    int num_k_blocks; /* spec.num_k_blocks (= mask row stride)            */
    int tiles_per_block_k; /* spec.block_k // MFMA_ATTN_BLOCK_K                */

    /* ---- kernel params (Values), in declaration order -- */
    ckc_value_t* Q;
    ckc_value_t* K;
    ckc_value_t* V;
    ckc_value_t* O;
    ckc_value_t* mask; /* kb.ptr("mask") -- i8 MaskBitmap base ptr     */
    ckc_value_t* scale_log2; /* kb.scalar("scale_log2")                      */
    ckc_value_t* seqlen_k_arg; /* kb.scalar("seqlen_k")                        */

    /* ---- grid decode (Values) -- */
    ckc_value_t* q_tile_idx; /* kb.q_token                                  */
    ckc_value_t* head_idx; /* kb.head_idx                                 */
    ckc_value_t* kv_head_idx; /* kb.kv_head_idx                              */

    /* ---- derived sparsity indices (Values) -- */
    ckc_value_t* q_tile_base; /* q_tile_idx * MFMA_ATTN_BLOCK_M              */
    ckc_value_t* q_block_idx; /* _magic_div(q_tile_base, block_q)            */
    ckc_value_t* mask_row_base; /* q_block_idx * num_k_blocks                  */

    /* ---- cooperative-stage state -- */
    ckc_value_t* tid; /* b.thread_id_x()                             */
    ckc_value_t* mask_lds; /* LDS i8 handle from stage_jenga_mask_to_lds  */
    /* (captured by the predicate closure)         */
} ckc_jenga_sparse_ctx_t;

/* ===================================================================== *
 *  ckc_vsa_sparse_ctx_t  --  shared state for build_vsa_sparse_attention.
 *
 *  Field order follows the Python prologue (lines 563-640).
 * ===================================================================== */
typedef struct ckc_vsa_sparse_ctx
{
    /* ---- inputs / resolved environment -- */
    const ckc_vsa_sparse_spec_t* spec; /* the VsaSparseSpec                   */
    const char* arch; /* NULL-normalised "gfx950"            */
    ckc_fmha_common_spec_t s; /* spec->common (Python `s`)           */

    /* ---- the kernel builder + its underlying IR builder -- */
    ckc_fmha_kernel_builder_t kb; /* FmhaKernelBuilder(spec.kernel_name(), s) */
    ckc_ir_builder_t* b; /* kb.builder (== &kb.b)                    */

    /* ---- geometry scalars (host ints) -- */
    int num_k_blocks; /* spec.num_k_blocks (= LDS bitmap length)          */
    int max_blocks_per_q; /* spec.max_blocks_per_q (= LUT row stride)         */
    int tiles_per_block_k; /* spec.block_k // MFMA_ATTN_BLOCK_K                */

    /* ---- kernel params (Values), in declaration order -- */
    ckc_value_t* Q;
    ckc_value_t* K;
    ckc_value_t* V;
    ckc_value_t* O;
    ckc_value_t* block_lut; /* kb.ptr("block_lut")   -- i32 LUT base ptr    */
    ckc_value_t* block_count; /* kb.ptr("block_count") -- i32 count base ptr  */
    ckc_value_t* scale_log2; /* kb.scalar("scale_log2")                      */
    ckc_value_t* seqlen_k_arg; /* kb.scalar("seqlen_k")                        */

    /* ---- grid decode (Values) -- */
    ckc_value_t* q_tile_idx; /* kb.q_token                                  */
    ckc_value_t* head_idx; /* kb.head_idx                                 */
    ckc_value_t* kv_head_idx; /* kb.kv_head_idx                              */

    /* ---- derived sparsity indices (Values) -- */
    ckc_value_t* q_tile_base; /* q_tile_idx * MFMA_ATTN_BLOCK_M              */
    ckc_value_t* q_block_idx; /* _magic_div(q_tile_base, block_q)            */
    ckc_value_t* lut_row_base; /* q_block_idx * max_blocks_per_q              */

    /* ---- cooperative-stage state -- */
    ckc_value_t* tid; /* b.thread_id_x()                             */
    ckc_value_t* bitmap_lds; /* LDS i8 handle from stage_vsa_bitmap_to_lds  */
    /* (captured by the predicate closure)         */
} ckc_vsa_sparse_ctx_t;

/* ===================================================================== *
 *  JENGA PHASE FUNCTIONS -- one per Python prologue stage / closure.
 *  Each phase reads/writes only ctx (+ the builder it carries) and emits IR in
 *  byte-identical Python order.
 * ===================================================================== */

/* Param declaration (Python _declare_jenga_params, lines 256-265): declare
 * Q/K/V/O tensors, the i8 `mask` ptr, scale_log2/seqlen_q/seqlen_k scalars, and
 * the q/k/v/o stride pairs on ctx->kb. */
void ckc_jenga_declare_params(ckc_jenga_sparse_ctx_t* ctx);

/* Prologue (lines 466-490): is_valid_jenga_spec gate, init the FmhaKernelBuilder
 * with the kernel name + common, block_size(64), declare params, decode_grid,
 * bind ctx->b, fetch the Q/K/V/O/mask/scalar param Values + grid coords, and
 * compute q_tile_base / q_block_idx / mask_row_base via _magic_div. Fills the
 * corresponding ctx fields. Returns false (builder/sticky error set) on a
 * rejected spec. */
bool ckc_jenga_prologue(ckc_jenga_sparse_ctx_t* ctx);

/* LDS staging (lines 492-506): tid = thread_id_x(); mask_lds =
 * stage_jenga_mask_to_lds(mask, mask_row_base, num_k_blocks, tid); sync(); and
 * derive tiles_per_block_k. Fills ctx->tid / ctx->mask_lds / ctx->tiles_per_block_k. */
void ckc_jenga_stage_mask(ckc_jenga_sparse_ctx_t* ctx);

/* Closure: _jenga_tile_predicate(b, kt) (lines 508-511). The extra_mask_predicate
 * callback: k_block_idx = _magic_div(kt, tiles_per_block_k); return
 * lds_bitmap_predicate(mask_lds, k_block_idx). `user` is `ckc_jenga_sparse_ctx_t*`.
 * Matches the ckc_attn_predicate_fn signature so it can be passed straight to
 * mfma_attention_fwd_inner_body. */
ckc_value_t* ckc_jenga_tile_predicate(ckc_ir_builder_t* b, ckc_value_t* kt, void* user);

/* Inner-body call + return (lines 513-538): assemble the ckc_mfma_attn_params_t
 * (Q/K/V/O, head_size, seqlen_k, q_tile_base, head/kv_head idx, all q/k/v/o
 * strides from ctx->kb, scale_log2, dtype, mask_mode="none", arch) wiring
 * extra_mask_predicate = ckc_jenga_tile_predicate with user = ctx, run
 * mfma_attention_fwd_inner_body, then b.ret(). Returns ctx->kb.kernel on success,
 * NULL on any builder error. */
ckc_kernel_def_t* ckc_jenga_emit_body(ckc_jenga_sparse_ctx_t* ctx);

/* ===================================================================== *
 *  VSA PHASE FUNCTIONS.
 * ===================================================================== */

/* Param declaration (Python _declare_vsa_params, lines 268-278): declare
 * Q/K/V/O tensors, the i32 `block_lut` + `block_count` ptrs,
 * scale_log2/seqlen_q/seqlen_k scalars, and the q/k/v/o stride pairs on
 * ctx->kb. */
void ckc_vsa_declare_params(ckc_vsa_sparse_ctx_t* ctx);

/* Prologue (lines 563-587): is_valid_vsa_spec gate, init the FmhaKernelBuilder,
 * block_size(64), declare params, decode_grid, bind ctx->b, fetch the
 * Q/K/V/O/block_lut/block_count/scalar param Values + grid coords, and compute
 * q_tile_base / q_block_idx / lut_row_base via _magic_div. Returns false on a
 * rejected spec. */
bool ckc_vsa_prologue(ckc_vsa_sparse_ctx_t* ctx);

/* LDS staging (lines 589-602): tid = thread_id_x(); bitmap_lds =
 * stage_vsa_bitmap_to_lds(block_lut, block_count, q_block_idx, lut_row_base,
 * num_k_blocks, max_blocks_per_q, tid); sync(); and derive tiles_per_block_k.
 * Fills ctx->tid / ctx->bitmap_lds / ctx->tiles_per_block_k. */
void ckc_vsa_stage_bitmap(ckc_vsa_sparse_ctx_t* ctx);

/* Closure: _vsa_tile_predicate(b, kt) (lines 604-613). The extra_mask_predicate
 * callback: k_block_idx = _magic_div(kt, tiles_per_block_k); return
 * lds_bitmap_predicate(bitmap_lds, k_block_idx). `user` is
 * `ckc_vsa_sparse_ctx_t*`. Matches ckc_attn_predicate_fn. */
ckc_value_t* ckc_vsa_tile_predicate(ckc_ir_builder_t* b, ckc_value_t* kt, void* user);

/* Inner-body call + return (lines 615-641): assemble the ckc_mfma_attn_params_t
 * wiring extra_mask_predicate = ckc_vsa_tile_predicate with user = ctx, run
 * mfma_attention_fwd_inner_body, then b.ret(). Returns ctx->kb.kernel on success,
 * NULL on any builder error. */
ckc_kernel_def_t* ckc_vsa_emit_body(ckc_vsa_sparse_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_SPARSE_ATTENTION_INTERNAL_H */
