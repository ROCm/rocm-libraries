/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.attention.h -- C99 port of selected symbols from
 * rocke/helpers/attention.py.
 *
 * Attention-specific IR helpers for unified paged attention: score masking and
 * the wave-XOR cross-lane row reductions used by the online softmax.
 *
 * PORTED SYMBOLS (this phase):
 *   - rocke_causal_mask           (causal_mask)
 *   - rocke_sliding_window_mask   (sliding_window_mask)
 *   - rocke_apply_attention_mask  (apply_attention_mask)
 *   - rocke_safe_inv_l            (safe_inv_l)
 *   - rocke_wave_reduce_stages    (wave_reduce_stages -- pure host-side selector)
 *   - rocke_warp_xor_reduce_sum   (warp_xor_reduce_sum)
 *   - rocke_dequant_fp8x8_to_dtype (dequant_fp8x8_to_dtype)
 *
 * Each IR-emitting helper reproduces its Python counterpart's rocke_b_* builder-
 * call sequence byte-faithfully (same ops, same order, same operands), binding
 * only to rocke/ir.h's public surface.
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_H
#define ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- mask modes */

/* mask_mode = Literal["none", "causal", "sliding_window"]. The string-typed
 * Python parameter is mapped to this enum so callers do not strcmp at every
 * site; rocke_apply_attention_mask switches on it exactly as the Python does. */
typedef enum rocke_attn_mask_mode
{
    ROCKE_ATTN_MASK_NONE = 0,
    ROCKE_ATTN_MASK_CAUSAL,
    ROCKE_ATTN_MASK_SLIDING_WINDOW
} rocke_attn_mask_mode_t;

/* ------------------------------------------------------------------- masks */

/* causal_mask(b, key_pos, context_len, query_pos) analogue.
 *
 * Returns the i1 predicate ``key_pos <= context_len + query_pos`` (keep when
 * true). Emits ``add`` then ``cmp_le``. */
rocke_value_t* rocke_causal_mask(rocke_ir_builder_t* b,
                                 rocke_value_t* key_pos,
                                 rocke_value_t* context_len,
                                 rocke_value_t* query_pos);

/* sliding_window_mask(b, key_pos, context_len, query_pos, sliding_window)
 * analogue.
 *
 * Returns the i1 predicate ``(context_len + query_pos - key_pos) <
 * sliding_window`` (keep when true). Emits ``add`` -> ``sub`` -> ``const_i32``
 * -> ``cmp_lt``. */
rocke_value_t* rocke_sliding_window_mask(rocke_ir_builder_t* b,
                                         rocke_value_t* key_pos,
                                         rocke_value_t* context_len,
                                         rocke_value_t* query_pos,
                                         int sliding_window);

/* apply_attention_mask(...) analogue.
 *
 * Maps ``mask_mode`` to the causal / sliding-window predicate and forces
 * masked-out positions to ``neg_inf`` via ``select`` so the softmax exp
 * collapses to zero. No-op (returns score_log2 unchanged) for
 * ROCKE_ATTN_MASK_NONE.
 *
 * ``context_len`` may be NULL to default to ``const_i32(0)``; ``neg_inf`` may be
 * NULL to default to ``const_f32(-1e30)`` (matching the Python ``None``
 * defaults, and only materialized for the non-"none" branch, exactly as the
 * Python orders it). Returns NULL (sticky error) on an unknown mask_mode. */
rocke_value_t* rocke_apply_attention_mask(rocke_ir_builder_t* b,
                                          rocke_value_t* score_log2,
                                          rocke_attn_mask_mode_t mask_mode,
                                          rocke_value_t* k_idx,
                                          rocke_value_t* query_pos,
                                          int sliding_window,
                                          rocke_value_t* context_len,
                                          rocke_value_t* neg_inf);

/* ------------------------------------------------------ online-softmax inv-l */

/* safe_inv_l(b, denom) analogue: reciprocal of the online-softmax denominator
 * with a zero guard (rcp(0) -> +inf would poison the output). Emits, in order,
 * ``fcmp oeq denom, 0`` -> ``rcp denom`` -> ``select`` so an all-masked tile
 * yields inv_l == 0. */
rocke_value_t* rocke_safe_inv_l(rocke_ir_builder_t* b, rocke_value_t* denom);

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
int rocke_wave_reduce_stages(rocke_ir_builder_t* b, int wave_size, int lanes_per_row);

/* -------------------------------------------------- cross-lane sum reduction */

/* warp_xor_reduce_sum(b, v, stages) analogue: wave64 butterfly sum reduction.
 *
 * Runs ``stages`` XOR-shuffle stages with masks 1, 2, 4, ... (1 << k), combining
 * with ``fadd``. Pass stages=4 for the default 16-lane reduction. Emits, per
 * stage, ``warp_shuffle_xor`` then ``fadd`` -- identical op order to the
 * Python. */
rocke_value_t* rocke_warp_xor_reduce_sum(rocke_ir_builder_t* b, rocke_value_t* v, int stages);

/* ------------------------------------------------------ fp8 in-register dequant */

/* dequant_fp8x8_to_dtype(b, fp8_vec, scale, dtype) analogue: in-register dequant
 * of a ``<8 x fp8e4m3>`` to a packed ``<8 x dtype>``.
 *
 * Splits the 8 fp8 inputs into two ``<4 x fp8>`` quads, runs the packed
 * ``cvt_pk_f32_fp8x4`` on each, multiplies every f32 lane by ``scale`` (an
 * UNFUSED explicit ``fmul`` -- NOT the fused E8M0-scale cvt, which would
 * silently truncate non-power-of-two scales), casts to ``dtype`` and re-packs
 * into a ``<8 x dtype>`` ready to feed the bf16/fp16 MFMA.
 *
 * Emits, in the exact order of the Python: 8 ``vec_extract`` + ``vec_pack`` for
 * the lo quad (fp8e4m3), 8 ``vec_extract`` + ``vec_pack`` for the hi quad,
 * ``cvt_pk_f32_fp8x4`` lo then hi, then 8 ``(vec_extract -> fmul -> cast_f32_to)``
 * triples (lo lanes 0..3 then hi lanes 0..3), then the final ``vec_pack(dtype)``.
 *
 * ``dtype`` is the packed-output element type (bf16 or f16). Returns NULL
 * (sticky error) if ``dtype`` is NULL. */
rocke_value_t* rocke_dequant_fp8x8_to_dtype(rocke_ir_builder_t* b,
                                            rocke_value_t* fp8_vec,
                                            rocke_value_t* scale,
                                            const rocke_type_t* dtype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_H */
