/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.attention.h -- C99 port of selected symbols from
 * ck_dsl/helpers/attention.py.
 *
 * Attention-specific IR helpers for unified paged attention: score masking and
 * the wave-XOR cross-lane row reductions used by the online softmax.
 *
 * PORTED SYMBOLS (this phase):
 *   - ckc_causal_mask           (causal_mask)
 *   - ckc_sliding_window_mask   (sliding_window_mask)
 *   - ckc_apply_attention_mask  (apply_attention_mask)
 *   - ckc_safe_inv_l            (safe_inv_l)
 *   - ckc_wave_reduce_stages    (wave_reduce_stages -- pure host-side selector)
 *   - ckc_warp_xor_reduce_sum   (warp_xor_reduce_sum)
 *
 * Each IR-emitting helper reproduces its Python counterpart's ckc_b_* builder-
 * call sequence byte-faithfully (same ops, same order, same operands), binding
 * only to ckc/ir.h's public surface.
 *
 * Lifetime: every emitted node is arena-owned (ckc_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_ATTENTION_H
#define CKC_HELPER_CK_DSL_HELPERS_ATTENTION_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- mask modes */

/* mask_mode = Literal["none", "causal", "sliding_window"]. The string-typed
 * Python parameter is mapped to this enum so callers do not strcmp at every
 * site; ckc_apply_attention_mask switches on it exactly as the Python does. */
typedef enum ckc_attn_mask_mode
{
    CKC_ATTN_MASK_NONE = 0,
    CKC_ATTN_MASK_CAUSAL,
    CKC_ATTN_MASK_SLIDING_WINDOW
} ckc_attn_mask_mode_t;

/* ------------------------------------------------------------------- masks */

/* causal_mask(b, key_pos, context_len, query_pos) analogue.
 *
 * Returns the i1 predicate ``key_pos <= context_len + query_pos`` (keep when
 * true). Emits ``add`` then ``cmp_le``. */
ckc_value_t* ckc_causal_mask(ckc_ir_builder_t* b,
                             ckc_value_t* key_pos,
                             ckc_value_t* context_len,
                             ckc_value_t* query_pos);

/* sliding_window_mask(b, key_pos, context_len, query_pos, sliding_window)
 * analogue.
 *
 * Returns the i1 predicate ``(context_len + query_pos - key_pos) <
 * sliding_window`` (keep when true). Emits ``add`` -> ``sub`` -> ``const_i32``
 * -> ``cmp_lt``. */
ckc_value_t* ckc_sliding_window_mask(ckc_ir_builder_t* b,
                                     ckc_value_t* key_pos,
                                     ckc_value_t* context_len,
                                     ckc_value_t* query_pos,
                                     int sliding_window);

/* apply_attention_mask(...) analogue.
 *
 * Maps ``mask_mode`` to the causal / sliding-window predicate and forces
 * masked-out positions to ``neg_inf`` via ``select`` so the softmax exp
 * collapses to zero. No-op (returns score_log2 unchanged) for
 * CKC_ATTN_MASK_NONE.
 *
 * ``context_len`` may be NULL to default to ``const_i32(0)``; ``neg_inf`` may be
 * NULL to default to ``const_f32(-1e30)`` (matching the Python ``None``
 * defaults, and only materialized for the non-"none" branch, exactly as the
 * Python orders it). Returns NULL (sticky error) on an unknown mask_mode. */
ckc_value_t* ckc_apply_attention_mask(ckc_ir_builder_t* b,
                                      ckc_value_t* score_log2,
                                      ckc_attn_mask_mode_t mask_mode,
                                      ckc_value_t* k_idx,
                                      ckc_value_t* query_pos,
                                      int sliding_window,
                                      ckc_value_t* context_len,
                                      ckc_value_t* neg_inf);

/* ------------------------------------------------------ online-softmax inv-l */

/* safe_inv_l(b, denom) analogue: reciprocal of the online-softmax denominator
 * with a zero guard (rcp(0) -> +inf would poison the output). Emits, in order,
 * ``fcmp oeq denom, 0`` -> ``rcp denom`` -> ``select`` so an all-masked tile
 * yields inv_l == 0. */
ckc_value_t* ckc_safe_inv_l(ckc_ir_builder_t* b, ckc_value_t* denom);

/* ---------------------------------------------- wave row-reduction selector */

/* wave_reduce_stages(wave_size, lanes_per_row) analogue: number of XOR
 * butterfly stages (== log2(lanes_per_row)) to reduce a row across
 * ``lanes_per_row`` lanes. Emits NO IR. ``lanes_per_row`` must be a power of two
 * and must not exceed ``wave_size``; on a violation the builder's sticky error
 * is set and -1 is returned.
 *
 * ``b`` may be NULL (no error sink); in that case a violating input simply
 * returns -1. Pass wave_size=64, lanes_per_row=16 for the standard 16x16 tile
 * (4 stages). */
int ckc_wave_reduce_stages(ckc_ir_builder_t* b, int wave_size, int lanes_per_row);

/* -------------------------------------------------- cross-lane sum reduction */

/* warp_xor_reduce_sum(b, v, stages) analogue: wave64 butterfly sum reduction.
 *
 * Runs ``stages`` XOR-shuffle stages with masks 1, 2, 4, ... (1 << k), combining
 * with ``fadd``. Pass stages=4 for the default 16-lane reduction. Emits, per
 * stage, ``warp_shuffle_xor`` then ``fadd`` -- identical op order to the
 * Python. */
ckc_value_t* ckc_warp_xor_reduce_sum(ckc_ir_builder_t* b, ckc_value_t* v, int stages);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_ATTENTION_H */
