// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Emission helpers shared by the gfx942 and gfx950 forward attention kernels.
 * Mirrors library/kernels/common/_attention_tiled_shared.py: each helper here
 * had a byte-identical copy in both arch modules. Not a public caller-owned
 * spec ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t */

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

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_TILED_COMMON_INTERNAL_H */
