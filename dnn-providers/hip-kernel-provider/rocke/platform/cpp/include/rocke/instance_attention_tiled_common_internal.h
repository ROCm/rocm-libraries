// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Emission helpers shared by the gfx942 and gfx950 forward attention kernels.
 * Mirrors library/kernels/common/_attention_tiled_shared.py: each helper here
 * had a byte-identical copy in both arch modules. Not a public caller-owned
 * spec ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H

#include <stdbool.h>

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t, rocke_type_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------- 32x32 C helpers *
 *
 * mfma_32x32_c_row(b, lane, elem_idx): MFMA-local output row for a 32x32x16 C
 * element ``elem_idx`` (0..15). Drives CK Tile's C-warp distribution
 * (C32_DIST.calculate_x) with ys=[elem_idx//4, elem_idx%4], ps=[[lane//32,
 * lane%32]] and returns the row X-coord. ``elem_idx`` outside 0..15 is a Python
 * ValueError: the builder records ROCKE_ERR_VALUE and the call returns NULL. */
rocke_value_t* rocke__mfma_32x32_c_row(rocke_ir_builder_t* b, rocke_value_t* lane, int elem_idx);

/* mfma_32x32_c_col(b, lane, n_tile32=0): MFMA-local output col for 32x32x16 C
 * elements in N-tile ``n_tile32``. Drives C32_DIST.calculate_x with ys=[0,0],
 * ps=[[lane//32, lane%32]] and returns the col X-coord; when n_tile32 != 0,
 * ``n_tile32*32`` is added on top. Returns NULL on a builder error. */
rocke_value_t* rocke__mfma_32x32_c_col(rocke_ir_builder_t* b, rocke_value_t* lane, int n_tile32);

/* Re-entrancy reset for the lazily-built C32_DIST cache. The cache holds a
 * tile distribution arena-allocated off a build's IRBuilder; that builder is
 * freed at the end of the build, so the cached pointer dangles. Call this at the
 * start of every tiled-attention build entry (gfx942 + gfx950) so a fresh build
 * rebuilds the distribution against its own arena instead of reading freed
 * memory. Safe to call when nothing is cached (no-op). */
void rocke_attn2d_c32_dist_reset(void);

/* ------------------------------------------- tiled 2D lane map (TiledLaneMap) *
 *
 * Per-lane row map and register-P helpers of the tiled 2D kernels. Each arch
 * fills one from its build ctx right before a call (pointer copies only, no IR).
 * The lane_* values are the prologue-cached SSA values; a NULL one has not been
 * emitted yet and the helper emits a fresh op in its place. */
typedef struct rocke_attn_tiled_lane_map
{
    rocke_ir_builder_t* b;
    rocke_value_t* lane;
    rocke_value_t* lane_rg; /* lane / 16 */
    rocke_value_t* lane_col; /* lane % 16 */
    rocke_value_t* lane_col_div4;
    rocke_value_t* lane_col_mod4;
    rocke_value_t* lane_rg_is0;
    rocke_value_t* lane_rg_is1;
    rocke_value_t* lane_rg_is2;
    const rocke_type_t* dtype;
    bool use_mfma_32x32;
    bool transposed_qk_32x32;
} rocke_attn_tiled_lane_map_t;

/* in_warp_row(r): lane_rg*4 + (r//4)*16 + r%4. */
rocke_value_t* rocke_attn_tiled_in_warp_row(const rocke_attn_tiled_lane_map_t* lm, int r);
/* state_row(r): softmax state slot r's row in the active layout. */
rocke_value_t* rocke_attn_tiled_state_row(const rocke_attn_tiled_lane_map_t* lm, int r);
/* bit2(v, bit): (v >> bit) & 1. */
rocke_value_t* rocke_attn_tiled_bit2(rocke_ir_builder_t* b, rocke_value_t* v, int bit);
/* select_lane_rg(v0..v3): pick v[lane_rg] for lane_rg in 0..3. */
rocke_value_t* rocke_attn_tiled_select_lane_rg(const rocke_attn_tiled_lane_map_t* lm,
                                               rocke_value_t* v0,
                                               rocke_value_t* v1,
                                               rocke_value_t* v2,
                                               rocke_value_t* v3);
/* permute_p_c_to_a16: C-distribution P (4 f32 regs) -> A16 layout; writes 4
 * values to ``out`` and 4 to ``*out_n`` (when non-NULL). */
void rocke_attn_tiled_permute_p_c_to_a16(const rocke_attn_tiled_lane_map_t* lm,
                                         rocke_value_t* const* p_regs_f32,
                                         int n,
                                         rocke_value_t** out,
                                         int* out_n);
/* pack_p_a16 / pack_p_a32: pack permuted P to the MFMA A operand vector. */
rocke_value_t* rocke_attn_tiled_pack_p_a16(const rocke_attn_tiled_lane_map_t* lm,
                                           rocke_value_t* const* p_regs_f32,
                                           int n);
rocke_value_t* rocke_attn_tiled_pack_p_a32(const rocke_attn_tiled_lane_map_t* lm,
                                           rocke_value_t* const* p_regs0_f32,
                                           rocke_value_t* const* p_regs1_f32,
                                           int n);

/* AccLayout.idx: flat index of the (N-tile, M-atom) PV accumulator. */
static inline int rocke_attn_tiled_acc_idx(int acc_m_atoms, int n, int atom)
{
    return n * acc_m_atoms + atom;
}

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H */
