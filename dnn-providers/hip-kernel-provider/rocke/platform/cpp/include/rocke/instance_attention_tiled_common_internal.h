// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Emission helpers shared by the gfx942 and gfx950 forward attention kernels.
 * Mirrors library/kernels/common/_attention_tiled_shared.py: each helper here
 * had a byte-identical copy in both arch modules. Not a public caller-owned
 * spec ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H

#include <stdbool.h>

#include "rocke/helper_rocke.helpers.distribution.h" /* rocke_tile_distribution_t */
#include "rocke/helper_rocke.helpers.transforms.h" /* rocke_tensor_descriptor_t */
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

/* _C16_DIST = make_static_tile_distribution(
 *                 make_c_warp_dstr_encoding(MfmaAtom.f16_16x16x16()))
 * CK Tile's C-warp distribution of the f16 16x16x16 atom. The C accumulator
 * layout is dtype/arch independent, so both 3D engines build it here for their
 * segment and reduce ctx. Returns NULL on failure; a missing atom records
 * ROCKE_ERR_VALUE on ``b`` (sticky, no throw: the reduce ctx_init runs outside
 * a ckc::guard_builder boundary). */
const rocke_tile_distribution_t* rocke_attn_tiled_make_c16_dist(rocke_ir_builder_t* b);

/* mfma_16x16_c_row(b, lane, reg): MFMA-local output row for a 16x16 C element
 * ``reg`` (0..3), through the arch ctx's C16_DIST (see
 * rocke_attn_tiled_make_c16_dist). ``reg`` outside 0..3 is a Python
 * ValueError: raised as a ckc::Error (ROCKE_ERR_VALUE), which the segment
 * build's ckc::guard_builder boundary records on ``b``. */
rocke_value_t* rocke_attn_tiled_mfma_16x16_c_row(rocke_ir_builder_t* b,
                                                 const rocke_tile_distribution_t* c16_dist,
                                                 rocke_value_t* lane,
                                                 int reg);

/* fast_paged_kv_blocks(kv_tile_idx): physical page ids of the two paged-KV
 * blocks of one KV tile (0 past block_table_max_idx), as SGPR values. */
void rocke_attn_tiled_fast_paged_kv_blocks(rocke_ir_builder_t* b,
                                           rocke_value_t* kv_tile_idx,
                                           rocke_value_t* block_tables,
                                           rocke_value_t* seq_base,
                                           rocke_value_t* block_table_max_idx,
                                           rocke_value_t** out_block0,
                                           rocke_value_t** out_block1);

/* load_page_id_window(rel_page): lane l loads the id of the segment's page
 * rel_page + l; past the sequence's last page it gets 0, never used. */
rocke_value_t* rocke_attn_tiled_load_page_id_window(rocke_ir_builder_t* b,
                                                    rocke_value_t* rel_page,
                                                    rocke_value_t* tid,
                                                    rocke_value_t* block_tables,
                                                    rocke_value_t* seq_base,
                                                    rocke_value_t* seq_pages,
                                                    rocke_value_t* seg_first_page);

/* ------------------------------------- fp8 K/V sync loaders (Fp8KvSyncLoader) *
 *
 * Sync fp8 K/V cache loaders of the tiled 2D kernels. Each arch fills one from
 * its build ctx right before a call (pointer copies only, no IR). Only valid
 * when the spec stores K/V as fp8 (kv_storage_dtype == "fp8e4m3"). */
typedef struct rocke_attn_tiled_fp8_kv_loader
{
    rocke_ir_builder_t* b;
    int T;
    int HD;
    int THREADS;
    rocke_value_t* tid;
    rocke_value_t* key;
    rocke_value_t* value;
    rocke_value_t* k_scale;
    rocke_value_t* v_scale;
    rocke_value_t* k_lds;
    rocke_value_t* v_lds;
    rocke_value_t* kv_head_idx;
    rocke_tensor_descriptor_t* kv_desc; /* paged K/V byte descriptor */
    const rocke_type_t* dtype;
} rocke_attn_tiled_fp8_kv_loader_t;

/* issue_dequant_loads(kv_tile_idx, buf_idx, lds_token): per-thread fp8 ->
 * f32 -> *scale -> working dtype -> LDS. ``is_v`` selects V (value / v_scale /
 * V_lds) instead of K. */
void rocke_attn_tiled_fp8_issue_dequant_loads(const rocke_attn_tiled_fp8_kv_loader_t* ld,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx,
                                              bool is_v);
/* issue_v_mfma_stripe(kv_tile_idx): raw fp8 V -> V_lds[0, col/16, token, col%16]. */
void rocke_attn_tiled_fp8_issue_v_mfma_stripe(const rocke_attn_tiled_fp8_kv_loader_t* ld,
                                              rocke_value_t* kv_tile_idx);

/* dequant_fp8_lds_to_bf16: LDS->LDS dequant fp8 -> f32 * scale -> working
 * dtype, from ``src`` slab ``buf_idx`` to ``dst`` slab 0. No caller today. */
void rocke_attn_tiled_fp8_dequant_lds_to_bf16(const rocke_attn_tiled_fp8_kv_loader_t* ld,
                                              rocke_value_t* src,
                                              rocke_value_t* dst,
                                              rocke_value_t* scale,
                                              rocke_value_t* buf_idx);

/* --------------------------------------- tiled 3D K/V LDS swizzle (KvLdsSwizzle) *
 *
 * Optional 16-byte-chunk XOR swizzle of the tiled 3D K/V LDS rows. With
 * ``swizzle`` off both helpers are identities and emit no IR. */
typedef struct rocke_attn_tiled_kv_swizzle
{
    rocke_ir_builder_t* b;
    int HD;
    bool swizzle;
    int chunks_per_row; /* 16-byte chunks per K/V LDS row (HD / 8) */
} rocke_attn_tiled_kv_swizzle_t;

/* col(row, col): LDS column of K/V element (row, col). */
rocke_value_t* rocke_attn_tiled_kv_swz_col(const rocke_attn_tiled_kv_swizzle_t* s,
                                           rocke_value_t* row,
                                           rocke_value_t* col);
/* src_linear(linear_half): the source element an async copy lane fetches. */
rocke_value_t* rocke_attn_tiled_kv_swz_src_linear(const rocke_attn_tiled_kv_swizzle_t* s,
                                                  rocke_value_t* linear_half);

/* AccLayout.idx: flat index of the (N-tile, M-atom) PV accumulator. */
static inline int rocke_attn_tiled_acc_idx(int acc_m_atoms, int n, int atom)
{
    return n * acc_m_atoms + atom;
}

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H */
