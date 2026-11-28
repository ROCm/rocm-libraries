// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Internal coordinate machinery; mirrors tiling/emit.py, not an author API. */
#ifndef ROCKE_TILING_INTERNAL_H
#define ROCKE_TILING_INTERNAL_H

#include "rocke/helper_rocke.helpers.tiling.h"

rocke_value_t* rocke_i_tiling_as_value(rocke_ir_builder_t* b, rocke_tiling_index_t index);
bool rocke_i_tiling_constant_index(rocke_tiling_index_t index, int64_t* value);
rocke_tiling_index_t rocke_i_tiling_bound(const rocke_tiling_window_t* window, int axis);
bool rocke_i_tiling_may_overhang(const rocke_tiling_window_t* window, int axis, int extent);
int rocke_i_tiling_register_count(const rocke_tile_distribution_encoding_t* layout,
                                  const int* shape);
void rocke_i_tiling_coordinates(rocke_ir_builder_t* b,
                                const rocke_tiling_tile_desc_t* tile,
                                rocke_value_t* thread,
                                int reg,
                                rocke_value_t** coordinates);
void rocke_i_tiling_positions(rocke_ir_builder_t* b,
                              const rocke_tiling_window_t* window,
                              int tile_rank,
                              rocke_value_t* const* coordinates,
                              rocke_value_t** positions);
rocke_value_t* rocke_i_tiling_address(rocke_ir_builder_t* b,
                                      const rocke_tiling_window_t* window,
                                      int tile_rank,
                                      rocke_value_t* const* positions);
rocke_value_t* rocke_i_tiling_clip_mask(rocke_ir_builder_t* b,
                                        const rocke_tiling_window_t* window,
                                        const rocke_tiling_tile_desc_t* tile,
                                        rocke_value_t* const* positions);
void rocke_i_tiling_register_offsets(const rocke_tiling_tile_desc_t* tile,
                                     int reg,
                                     int64_t* offsets);

bool rocke_i_tiling_can_rebase_global(rocke_value_t* ptr,
                                      const rocke_tiling_window_t* window,
                                      const rocke_tiling_tile_desc_t* tile,
                                      int alignment_bytes,
                                      bool in_bounds);
void rocke_i_tiling_split_window(const rocke_tiling_window_t* window,
                                 rocke_tiling_window_t* runtime_window,
                                 int64_t* constant_origin);
rocke_value_t* rocke_i_tiling_byte_address(rocke_ir_builder_t* b,
                                           const rocke_tiling_window_t* window,
                                           int tile_rank,
                                           rocke_value_t* const* positions);
int64_t rocke_i_tiling_storage_displacement(const rocke_tiling_window_t* window,
                                            const rocke_tiling_tile_desc_t* tile,
                                            const int64_t* coordinates,
                                            int64_t constant_origin);

#endif /* ROCKE_TILING_INTERNAL_H */
