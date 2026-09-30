/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_sparse_attention_internal.h -- PRIVATE shared state + phase-
 * function contract for the C99 port of build_jenga_sparse_attention and
 * build_vsa_sparse_attention (rocke/instances/common/sparse_attention.py).
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
 *   rocke_attn_predicate_fn callback, where `user` points at one shared context
 *   struct (rocke_jenga_sparse_ctx_t / rocke_vsa_sparse_ctx_t) holding EXACTLY the
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
 *   functions mirror the Python helper / closure names with a `rocke_jenga_` /
 *   `rocke_vsa_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. Included only by the
 * instance_sparse_attention*.c translation units. Public callers use
 * rocke/instance_sparse_attention.h.
 */
#ifndef ROCKE_INSTANCE_SPARSE_ATTENTION_INTERNAL_H
#define ROCKE_INSTANCE_SPARSE_ATTENTION_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h" /* rocke_fmha_kernel_builder_t */
#include "rocke/instance_sparse_attention.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  _magic_div(b, dividend, divisor): dividend // divisor via CK Tile's magic
 *  mul-hi division. Internal to this instance (the Python module-private
 *  _magic_div). Calls rocke_calculate_magic_numbers(divisor) -> (mult, shift) then
 *  rocke_do_magic_division(b, dividend, mult, shift). Returns the quotient Value or
 *  NULL on a builder error. Used by the q_block / k_block index decodes and the
 *  predicate closures.
 * ===================================================================== */
rocke_value_t*
    rocke_sparse_attn_magic_div(rocke_ir_builder_t* b, rocke_value_t* dividend, int divisor);

/* ===================================================================== *
 *  rocke_jenga_sparse_ctx_t  --  shared state for build_jenga_sparse_attention.
 *
 *  Field order follows the Python prologue top-to-bottom (lines 466-538) so the
 *  populate routine reads straight against the source.
 * ===================================================================== */
typedef struct rocke_jenga_sparse_ctx
{
    /* ---- inputs / resolved environment -- */
    const rocke_jenga_sparse_spec_t* spec; /* the JengaSparseSpec               */
    const char* arch; /* NULL-normalised "gfx950"          */
    rocke_fmha_common_spec_t s; /* spec->common (Python `s`)         */

    /* ---- the kernel builder + its underlying IR builder -- */
    rocke_fmha_kernel_builder_t kb; /* FmhaKernelBuilder(spec.kernel_name(), s) */
    rocke_ir_builder_t* b; /* kb.builder (== &kb.b)                    */

    /* ---- geometry scalars (host ints) -- */
    int num_k_blocks; /* spec.num_k_blocks (= mask row stride)            */
    int tiles_per_block_k; /* spec.block_k // MFMA_ATTN_BLOCK_K                */

    /* ---- kernel params (Values), in declaration order -- */
    rocke_value_t* Q;
    rocke_value_t* K;
    rocke_value_t* V;
    rocke_value_t* O;
    rocke_value_t* mask; /* kb.ptr("mask") -- i8 MaskBitmap base ptr     */
    rocke_value_t* scale_log2; /* kb.scalar("scale_log2")                      */
    rocke_value_t* seqlen_k_arg; /* kb.scalar("seqlen_k")                        */

    /* ---- grid decode (Values) -- */
    rocke_value_t* q_tile_idx; /* kb.q_token                                  */
    rocke_value_t* head_idx; /* kb.head_idx                                 */
    rocke_value_t* kv_head_idx; /* kb.kv_head_idx                              */

    /* ---- derived sparsity indices (Values) -- */
    rocke_value_t* q_tile_base; /* q_tile_idx * MFMA_ATTN_BLOCK_M              */
    rocke_value_t* q_block_idx; /* _magic_div(q_tile_base, block_q)            */
    rocke_value_t* mask_row_base; /* q_block_idx * num_k_blocks                  */

    /* ---- cooperative-stage state -- */
    rocke_value_t* tid; /* b.thread_id_x()                             */
    rocke_value_t* mask_lds; /* LDS i8 handle from stage_jenga_mask_to_lds  */
    /* (captured by the predicate closure)         */
} rocke_jenga_sparse_ctx_t;

/* ===================================================================== *
 *  rocke_vsa_sparse_ctx_t  --  shared state for build_vsa_sparse_attention.
 *
 *  Field order follows the Python prologue (lines 563-640).
 * ===================================================================== */
typedef struct rocke_vsa_sparse_ctx
{
    /* ---- inputs / resolved environment -- */
    const rocke_vsa_sparse_spec_t* spec; /* the VsaSparseSpec                   */
    const char* arch; /* NULL-normalised "gfx950"            */
    rocke_fmha_common_spec_t s; /* spec->common (Python `s`)           */

    /* ---- the kernel builder + its underlying IR builder -- */
    rocke_fmha_kernel_builder_t kb; /* FmhaKernelBuilder(spec.kernel_name(), s) */
    rocke_ir_builder_t* b; /* kb.builder (== &kb.b)                    */

    /* ---- geometry scalars (host ints) -- */
    int num_k_blocks; /* spec.num_k_blocks (= LDS bitmap length)          */
    int max_blocks_per_q; /* spec.max_blocks_per_q (= LUT row stride)         */
    int tiles_per_block_k; /* spec.block_k // MFMA_ATTN_BLOCK_K                */

    /* ---- kernel params (Values), in declaration order -- */
    rocke_value_t* Q;
    rocke_value_t* K;
    rocke_value_t* V;
    rocke_value_t* O;
    rocke_value_t* block_lut; /* kb.ptr("block_lut")   -- i32 LUT base ptr    */
    rocke_value_t* block_count; /* kb.ptr("block_count") -- i32 count base ptr  */
    rocke_value_t* scale_log2; /* kb.scalar("scale_log2")                      */
    rocke_value_t* seqlen_k_arg; /* kb.scalar("seqlen_k")                        */

    /* ---- grid decode (Values) -- */
    rocke_value_t* q_tile_idx; /* kb.q_token                                  */
    rocke_value_t* head_idx; /* kb.head_idx                                 */
    rocke_value_t* kv_head_idx; /* kb.kv_head_idx                              */

    /* ---- derived sparsity indices (Values) -- */
    rocke_value_t* q_tile_base; /* q_tile_idx * MFMA_ATTN_BLOCK_M              */
    rocke_value_t* q_block_idx; /* _magic_div(q_tile_base, block_q)            */
    rocke_value_t* lut_row_base; /* q_block_idx * max_blocks_per_q              */

    /* ---- cooperative-stage state -- */
    rocke_value_t* tid; /* b.thread_id_x()                             */
    rocke_value_t* bitmap_lds; /* LDS i8 handle from stage_vsa_bitmap_to_lds  */
    /* (captured by the predicate closure)         */
} rocke_vsa_sparse_ctx_t;

/* ===================================================================== *
 *  JENGA PHASE FUNCTIONS -- one per Python prologue stage / closure.
 *  Each phase reads/writes only ctx (+ the builder it carries) and emits IR in
 *  byte-identical Python order.
 * ===================================================================== */

/* Param declaration (Python _declare_jenga_params, lines 256-265): declare
 * Q/K/V/O tensors, the i8 `mask` ptr, scale_log2/seqlen_q/seqlen_k scalars, and
 * the q/k/v/o stride pairs on ctx->kb. */
void rocke_jenga_declare_params(rocke_jenga_sparse_ctx_t* ctx);

/* Prologue (lines 466-490): is_valid_jenga_spec gate, init the FmhaKernelBuilder
 * with the kernel name + common, block_size(64), declare params, decode_grid,
 * bind ctx->b, fetch the Q/K/V/O/mask/scalar param Values + grid coords, and
 * compute q_tile_base / q_block_idx / mask_row_base via _magic_div. Fills the
 * corresponding ctx fields. Returns false (builder/sticky error set) on a
 * rejected spec. */
bool rocke_jenga_prologue(rocke_jenga_sparse_ctx_t* ctx);

/* LDS staging (lines 492-506): tid = thread_id_x(); mask_lds =
 * stage_jenga_mask_to_lds(mask, mask_row_base, num_k_blocks, tid); sync(); and
 * derive tiles_per_block_k. Fills ctx->tid / ctx->mask_lds / ctx->tiles_per_block_k. */
void rocke_jenga_stage_mask(rocke_jenga_sparse_ctx_t* ctx);

/* Closure: _jenga_tile_predicate(b, kt) (lines 508-511). The extra_mask_predicate
 * callback: k_block_idx = _magic_div(kt, tiles_per_block_k); return
 * lds_bitmap_predicate(mask_lds, k_block_idx). `user` is `rocke_jenga_sparse_ctx_t*`.
 * Matches the rocke_attn_predicate_fn signature so it can be passed straight to
 * mfma_attention_fwd_inner_body. */
rocke_value_t* rocke_jenga_tile_predicate(rocke_ir_builder_t* b, rocke_value_t* kt, void* user);

/* Inner-body call + return (lines 513-538): assemble the rocke_mfma_attn_params_t
 * (Q/K/V/O, head_size, seqlen_k, q_tile_base, head/kv_head idx, all q/k/v/o
 * strides from ctx->kb, scale_log2, dtype, mask_mode="none", arch) wiring
 * extra_mask_predicate = rocke_jenga_tile_predicate with user = ctx, run
 * mfma_attention_fwd_inner_body, then b.ret(). Returns ctx->kb.kernel on success,
 * NULL on any builder error. */
rocke_kernel_def_t* rocke_jenga_emit_body(rocke_jenga_sparse_ctx_t* ctx);

/* ===================================================================== *
 *  VSA PHASE FUNCTIONS.
 * ===================================================================== */

/* Param declaration (Python _declare_vsa_params, lines 268-278): declare
 * Q/K/V/O tensors, the i32 `block_lut` + `block_count` ptrs,
 * scale_log2/seqlen_q/seqlen_k scalars, and the q/k/v/o stride pairs on
 * ctx->kb. */
void rocke_vsa_declare_params(rocke_vsa_sparse_ctx_t* ctx);

/* Prologue (lines 563-587): is_valid_vsa_spec gate, init the FmhaKernelBuilder,
 * block_size(64), declare params, decode_grid, bind ctx->b, fetch the
 * Q/K/V/O/block_lut/block_count/scalar param Values + grid coords, and compute
 * q_tile_base / q_block_idx / lut_row_base via _magic_div. Returns false on a
 * rejected spec. */
bool rocke_vsa_prologue(rocke_vsa_sparse_ctx_t* ctx);

/* LDS staging (lines 589-602): tid = thread_id_x(); bitmap_lds =
 * stage_vsa_bitmap_to_lds(block_lut, block_count, q_block_idx, lut_row_base,
 * num_k_blocks, max_blocks_per_q, tid); sync(); and derive tiles_per_block_k.
 * Fills ctx->tid / ctx->bitmap_lds / ctx->tiles_per_block_k. */
void rocke_vsa_stage_bitmap(rocke_vsa_sparse_ctx_t* ctx);

/* Closure: _vsa_tile_predicate(b, kt) (lines 604-613). The extra_mask_predicate
 * callback: k_block_idx = _magic_div(kt, tiles_per_block_k); return
 * lds_bitmap_predicate(bitmap_lds, k_block_idx). `user` is
 * `rocke_vsa_sparse_ctx_t*`. Matches rocke_attn_predicate_fn. */
rocke_value_t* rocke_vsa_tile_predicate(rocke_ir_builder_t* b, rocke_value_t* kt, void* user);

/* Inner-body call + return (lines 615-641): assemble the rocke_mfma_attn_params_t
 * wiring extra_mask_predicate = rocke_vsa_tile_predicate with user = ctx, run
 * mfma_attention_fwd_inner_body, then b.ret(). Returns ctx->kb.kernel on success,
 * NULL on any builder error. */
rocke_kernel_def_t* rocke_vsa_emit_body(rocke_vsa_sparse_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_SPARSE_ATTENTION_INTERNAL_H */
