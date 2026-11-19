// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Bounded native fragment authoring: static rank 1..8, global memory.
 * Mirrors tiling TensorDesc/TensorWindow/TileDesc/Fragment and global verbs.
 * No MMA driver/catalog/solver or LDS policy is ported by this interface.
 * Descriptors/fragments are caller-owned; encoding and SSA values are arena-owned.
 * A window borrows its tensor descriptor for the duration of each verb call.
 */
#ifndef ROCKE_HELPER_TILING_H
#define ROCKE_HELPER_TILING_H

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/storage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_TILING_MAX_RANK 8

typedef struct rocke_tiling_index
{
    rocke_value_t* value; /* NULL selects the compile-time integer. */
    int constant;
} rocke_tiling_index_t;

typedef struct rocke_tiling_tensor_desc
{
    int rank;
    int lengths[ROCKE_TILING_MAX_RANK];
    bool has_length[ROCKE_TILING_MAX_RANK]; /* false explicitly denotes an unknown leading extent */
    int strides[ROCKE_TILING_MAX_RANK];
    const rocke_type_t* dtype;
} rocke_tiling_tensor_desc_t;

typedef struct rocke_tiling_window
{
    const rocke_tiling_tensor_desc_t* tensor;
    rocke_tiling_index_t origin[ROCKE_TILING_MAX_RANK];
    bool has_bound[ROCKE_TILING_MAX_RANK];
    rocke_tiling_index_t bounds[ROCKE_TILING_MAX_RANK];
} rocke_tiling_window_t;

typedef struct rocke_tiling_tile_desc
{
    int rank;
    int shape[ROCKE_TILING_MAX_RANK];
    const rocke_tile_distribution_encoding_t* layout;
    int register_count;
} rocke_tiling_tile_desc_t;

typedef struct rocke_tiling_fragment
{
    rocke_tiling_tile_desc_t tile;
    const rocke_type_t* dtype;
    rocke_value_t* value; /* NULL means not materialized yet. */
} rocke_tiling_fragment_t;

#ifdef __cplusplus
#define ROCKE_TILING_DEFAULT_LENGTHS = nullptr
#else
#define ROCKE_TILING_DEFAULT_LENGTHS
#endif
rocke_status_t rocke_h_make_tensor_desc(rocke_ir_builder_t* b,
                                        rocke_tiling_tensor_desc_t* out,
                                        const int* lengths,
                                        const int* strides,
                                        int rank,
                                        const rocke_type_t* dtype,
                                        const bool* has_length ROCKE_TILING_DEFAULT_LENGTHS);
#undef ROCKE_TILING_DEFAULT_LENGTHS
rocke_status_t rocke_h_make_window(rocke_ir_builder_t* b,
                                   rocke_tiling_window_t* out,
                                   const rocke_tiling_tensor_desc_t* tensor,
                                   const rocke_tiling_index_t* origin,
                                   int origin_rank,
                                   const rocke_tiling_index_t* bounds,
                                   const bool* has_bound);
/* Use a layout from rocke_make_tile_distribution_encoding. The tile shape must
 * match its X extents. Do not mutate a descriptor after construction. */
rocke_status_t rocke_h_make_tile_desc(rocke_ir_builder_t* b,
                                      rocke_tiling_tile_desc_t* out,
                                      const int* shape,
                                      int rank,
                                      const rocke_tile_distribution_encoding_t* layout);
rocke_status_t rocke_h_make_matrix_fragment_desc(rocke_ir_builder_t* b,
                                                 rocke_tiling_tile_desc_t* out,
                                                 const rocke_matrix_fragment_layout_t* layout);
/* Verified fp32 accumulator C or result D. WMMA profile is gfx1250-only. */
rocke_status_t rocke_h_make_mma_accumulator_desc(rocke_ir_builder_t* b,
                                                 rocke_tiling_tile_desc_t* out,
                                                 const rocke_mma_op_t* atom,
                                                 const rocke_arch_target_t* target,
                                                 bool result);
/* Verified native A/B scale map in (free axis, K-group) storage order. */
rocke_status_t rocke_h_make_mma_scale_desc(rocke_ir_builder_t* b,
                                           rocke_tiling_tile_desc_t* out,
                                           const rocke_mma_op_t* atom,
                                           const rocke_arch_target_t* target);
rocke_status_t rocke_h_make_fragment(rocke_ir_builder_t* b,
                                     rocke_tiling_fragment_t* out,
                                     const rocke_tiling_tile_desc_t* tile,
                                     const rocke_type_t* dtype,
                                     rocke_value_t* value);
rocke_status_t
    rocke_h_fill_fragment(rocke_ir_builder_t* b, rocke_tiling_fragment_t* fragment, int scalar);

/* alignment_bytes==0 selects Python None (legacy layout-derived guarantee).
 * A positive power of two caps alignment at the passed global pointer.
 * packed_runs_aligned states that runtime packed run origins/bounds are on
 * complete-word groups. Non-packed dtypes reject that statement. */
rocke_status_t rocke_h_load_fragment(rocke_ir_builder_t* b,
                                     rocke_tiling_fragment_t* out,
                                     rocke_value_t* ptr,
                                     const rocke_tiling_window_t* window,
                                     const rocke_tiling_tile_desc_t* tile,
                                     rocke_value_t* thread,
                                     int alignment_bytes,
                                     bool packed_runs_aligned);
rocke_status_t rocke_h_store_fragment(rocke_ir_builder_t* b,
                                      rocke_value_t* ptr,
                                      const rocke_tiling_window_t* window,
                                      const rocke_tiling_fragment_t* fragment,
                                      rocke_value_t* thread,
                                      int alignment_bytes,
                                      bool packed_runs_aligned);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_TILING_H */
