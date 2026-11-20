/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_alibi.h -- C port of the
 * ALiBi helpers of library/kernels/common/_attention_alibi.py used by the
 * unified tiled attention kernels.
 *
 *   Python (_attention_alibi.py)                    C (this header)
 *   ----------------------------------------------  ----------------------------
 *   ALIBI_DESIGNS[design]                           rocke_attn_alibi_design_token(design)
 *   alibi_slope(b, head, num_heads=H, slopes=p)     rocke_attn_alibi_slope(b, head, H, p)
 *   alibi_slope_log2(b, head, num_heads=H, ...)     rocke_attn_alibi_slope_log2(b, head, H, p)
 *   alibi_tile_bias(b, s, k0, lk, a, n_sub)         rocke_attn_alibi_tile_bias(b, s, k0, lk, a, n_sub, out)
 *   alibi_split_lane_terms(b, s, lane_key)          rocke_attn_alibi_split_lane_terms(b, s, lane_key, out)
 *   alibi_split_tile_offsets(b, s, k0, a, n_sub)    rocke_attn_alibi_split_tile_offsets(b, s, k0, a, n_sub, out)
 *
 * Only the runtime (i32 value) head count is ported: the tiled kernels read the
 * head count as a kernel argument, and the dense kernels that bake it have no
 * hand-written C++ twin. The emitted op sequence (constants included) matches
 * the Python helper exactly, so callers stay byte-identical with their Python
 * kernels.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_ALIBI_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_ALIBI_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* slope(head) as f32. ``slopes`` is the alibi_slopes_ptr kernel argument: a
 * non-null pointer gives slopes[head] (fp32, one per query head, 0 past
 * ``num_heads``), a null one hipDNN's built-in slope of ``head`` among
 * ``num_heads`` query heads (SdpaSemanticsRegister SEM-021), chosen at run time
 * by a uniform branch. ``num_heads`` is the i32 query head count kernel
 * argument. */
rocke_value_t* rocke_attn_alibi_slope(rocke_ir_builder_t* b,
                                      rocke_value_t* head,
                                      rocke_value_t* num_heads,
                                      rocke_value_t* slopes);

/* The kernel-name token of an ALiBi design ("seed_tile" -> "st", "fma_tile" ->
 * "ft", "seed_split" -> "ss", "fma_split" -> "fs"), NULL for any other string.
 * ``seed``: the QK MFMA accumulator starts at the term in raw units; ``fma``:
 * the term joins the post-MFMA scale op. ``tile``: the whole term per KV tile;
 * ``split``: a loop-invariant per-key part plus a per-row offset per tile folded
 * into the row max and the exp2 offset. */
const char* rocke_attn_alibi_design_token(const char* design);

/* slope(head) * log2(e): the slope in log2-domain score units. */
rocke_value_t* rocke_attn_alibi_slope_log2(rocke_ir_builder_t* b,
                                           rocke_value_t* head,
                                           rocke_value_t* num_heads,
                                           rocke_value_t* slopes);

/* slope * (key - anchor) for one KV tile in the transposed 32x32 MFMA C layout:
 * out[nsub * 16 + i] for the key ``tile_key0 + lane_key_off + nsub * 32 +
 * (i / 4) * 8 + i % 4`` (lane_key_off = (lane / 32) * 4). ``out`` holds
 * n_sub * 16 values, in the units of ``slope``. */
void rocke_attn_alibi_tile_bias(rocke_ir_builder_t* b,
                                rocke_value_t* slope,
                                rocke_value_t* tile_key0,
                                rocke_value_t* lane_key_off,
                                rocke_value_t* anchor,
                                int n_sub,
                                rocke_value_t** out);

/* Split designs: the loop-invariant part slope * (lane_key + c_i) of the term
 * for one 32-key sub-block (c_i as in rocke_attn_alibi_tile_bias); out[16]. */
void rocke_attn_alibi_split_lane_terms(rocke_ir_builder_t* b,
                                       rocke_value_t* slope,
                                       rocke_value_t* lane_key,
                                       rocke_value_t** out);

/* Split designs: the per-row offset slope * min(tile_key0 + 32 n - anchor, 0)
 * of sub-block n, out[n_sub]. The clamp keeps it <= 0 for a sub-block without a
 * visible key, so exp2 stays finite for a row masked in the whole tile. */
void rocke_attn_alibi_split_tile_offsets(rocke_ir_builder_t* b,
                                         rocke_value_t* slope,
                                         rocke_value_t* tile_key0,
                                         rocke_value_t* anchor,
                                         int n_sub,
                                         rocke_value_t** out);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_ALIBI_H */
