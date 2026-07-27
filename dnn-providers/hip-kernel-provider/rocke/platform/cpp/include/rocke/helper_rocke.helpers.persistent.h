/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.persistent.h -- C99 port of rocke.helpers.persistent.
 *
 * Persistent-kernel pattern helper. A small, constant launch grid of CTAs
 * pulls work-items from a global counter via atomic_add(1) until exhausted,
 * decoupling the launch grid from the problem size.
 *
 * This phase ports ONLY:
 *   build_persistent_counter_init
 *   persistent_tile_for_each
 *
 * The context-manager helper persistent_tile_loop (the Python @contextmanager)
 * is reproduced here as a file-local static inside the .c so that both public
 * entries emit a byte-identical IR op stream. The .c re-creates the exact
 * builder-call sequence of the Python.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_PERSISTENT_H
#define ROCKE_HELPER_ROCKE_HELPERS_PERSISTENT_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Atomic-fetch the first tile id for this CTA from `counter`.
 *
 * Python signature:
 *   build_persistent_counter_init(
 *       b, counter, *, counter_idx=None, increment=1, cooperative=True,
 *       broadcast_slot=None, wave_size=64, block_size=64) -> Value
 *
 * counter is an i32-pointer SSA Value (a host-precleared global slot).
 * counter_idx is an i32 SSA Value indexing that buffer; pass NULL for the
 * Python default (slot 0 -> b.const_i32(0)). broadcast_slot is the optional
 * multi-wave LDS broadcast slot (SmemType i32 [1]); pass NULL for the Python
 * default. increment / wave_size / block_size are compile-time ints
 * (Python defaults 1 / 64 / 64). cooperative selects the broadcast path.
 *
 * Returns the i32 SSA value: the slot's pre-increment value (the tile index
 * this CTA owns first).
 */
rocke_value_t* rocke_build_persistent_counter_init(rocke_ir_builder_t* b,
                                                   rocke_value_t* counter,
                                                   rocke_value_t* counter_idx,
                                                   int increment,
                                                   bool cooperative,
                                                   rocke_value_t* broadcast_slot,
                                                   int wave_size,
                                                   int block_size);

/* Per-iteration body callback for persistent_tile_for_each.
 *
 * Python `body(tile_idx)` is a closure; in C we thread an explicit user-data
 * pointer. tile_idx is the current i32 SSA tile id. Invoked once per
 * iteration, already wrapped in the in_range guard region by the helper.
 */
typedef void (*rocke_persistent_tile_body_fn)(rocke_ir_builder_t* b,
                                              rocke_value_t* tile_idx,
                                              void* user_data);

/* Functional sugar over the persistent-tile loop.
 *
 * Python signature:
 *   persistent_tile_for_each(
 *       b, *, counter, num_tiles, max_iters, body, counter_idx=None,
 *       cooperative=True, wave_size=64, block_size=64) -> None
 *
 * num_tiles is an i32 SSA Value. max_iters is the compile-time worst-case
 * per-CTA iteration count. body/user_data form the per-iteration callback.
 * counter_idx is an i32 SSA Value or NULL (Python default slot 0). wave_size /
 * block_size are compile-time ints (Python defaults 64 / 64).
 */
void rocke_persistent_tile_for_each(rocke_ir_builder_t* b,
                                    rocke_value_t* counter,
                                    rocke_value_t* num_tiles,
                                    int max_iters,
                                    rocke_persistent_tile_body_fn body,
                                    void* user_data,
                                    rocke_value_t* counter_idx,
                                    bool cooperative,
                                    int wave_size,
                                    int block_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_PERSISTENT_H */
