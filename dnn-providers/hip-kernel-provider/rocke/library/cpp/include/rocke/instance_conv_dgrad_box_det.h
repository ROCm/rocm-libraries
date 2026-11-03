/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_dgrad_box_det.h
 *
 * C++ port of library/kernels/common/conv_dgrad_box_det.py
 * (DgradBoxDetReduceSpec + build_dgrad_box_det_reduce): Stage 2 of the dgrad
 * pp_box_det deterministic split-K. One workgroup per split tile (grid x =
 * n_tiles, y = msplit, z = the stage-1 wave under wave_wg) with the stage-1
 * warp grid: each lane loads its own accumulators from the tile's workspace
 * slots in slice order, adds them in f32 in that order, and stores the tile
 * once through the cshuffle epilogue at the dX pixels of the tile's rows.
 *
 * Kernel signature: ws (i32*, readonly), dx (dtype_d*, writeonly),
 * tab_word, ws_word, C, dx_bytes (i32).
 */
#ifndef ROCKE_INSTANCE_CONV_DGRAD_BOX_DET_H
#define ROCKE_INSTANCE_CONV_DGRAD_BOX_DET_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Stage-2 table entry words (Python PBD_ENTRY_WORDS). */
#define ROCKE_PBD_ENTRY_WORDS 8

/* Mirror of Python DgradBoxDetReduceSpec (frozen dataclass). */
typedef struct rocke_dgrad_box_det_reduce_spec
{
    int tile_m;
    int tile_n;
    int warp_m;
    int warp_n;
    const char* dtype_d; /* default "bf16" */
    int store_vec; /* default 8 */
    int m_chunks; /* default 1 */
    int wave_size; /* default 64 */
    int msplit; /* default 1: workgroups per split tile (grid y) */
    int unroll; /* default 1: slices per loop trip */
    bool wave_wg; /* default false: one wave per workgroup (grid z = wave) */
    int slot_bytes; /* default 4 (pp_box_det 1); 2 = bf16 slots (pp_box_det 2) */
    int load_aux; /* default 0: cache-policy aux bits of the slot loads */
    bool pix_pre; /* default false: store rows' dX pixels loaded before the fold */
    const char* name; /* default "conv_dgrad_box_det_reduce" */
} rocke_dgrad_box_det_reduce_spec_t;

rocke_dgrad_box_det_reduce_spec_t rocke_dgrad_box_det_reduce_spec_default(void);

/* block_size: wave_size under wave_wg, else warp_m * warp_n * wave_size. */
int rocke_dgrad_box_det_reduce_block_size(const rocke_dgrad_box_det_reduce_spec_t* spec);

/* Python kernel_name(). */
rocke_status_t rocke_dgrad_box_det_reduce_kernel_name(const rocke_dgrad_box_det_reduce_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap);

/* Build into an initialised builder; NULL on error (the builder's sticky error). */
rocke_kernel_def_t* rocke_build_dgrad_box_det_reduce(rocke_ir_builder_t* b,
                                                     const rocke_dgrad_box_det_reduce_spec_t* spec,
                                                     const char* arch);

/* Init the builder with the kernel name, then build. */
rocke_kernel_def_t* rocke_build_dgrad_box_det_reduce_new(
    rocke_ir_builder_t* b, const rocke_dgrad_box_det_reduce_spec_t* spec, const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DGRAD_BOX_DET_H */
