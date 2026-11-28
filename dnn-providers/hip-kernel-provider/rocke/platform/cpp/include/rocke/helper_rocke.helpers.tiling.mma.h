// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_HELPER_ROCKE_HELPERS_TILING_MMA_H
#define ROCKE_HELPER_ROCKE_HELPERS_TILING_MMA_H

#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TileMmaDriver's atom-grid traversal. The plan and operand layouts are
 * supplied by the caller; no instruction selection or waits are emitted. */
typedef struct rocke_tile_mma_grid
{
    int m;
    int n;
    int k;
} rocke_tile_mma_grid_t;

typedef rocke_value_t* (*rocke_tile_mma_atom_fn)(
    rocke_ir_builder_t* b, int mi, int ni, int ki, rocke_value_t* acc, void* user);

/* order is a permutation of "MNK", last axis fastest. Accumulators are M/N
 * row-major. `out` may be `accs` or a separate buffer of `count` entries. */
void rocke_tile_mma_accumulate_atoms(rocke_ir_builder_t* b,
                                     rocke_tile_mma_grid_t grid,
                                     const char* order,
                                     rocke_value_t* const* accs,
                                     size_t count,
                                     rocke_tile_mma_atom_fn emit,
                                     void* user,
                                     rocke_value_t** out);

typedef rocke_value_t* (*rocke_tile_mma_read_fn)(rocke_ir_builder_t* b,
                                                 int ki,
                                                 int atom,
                                                 void* user);
typedef rocke_value_t* (*rocke_tile_mma_issue_fn)(
    rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb, rocke_value_t* acc, void* user);
typedef void (*rocke_tile_mma_reads_fn)(rocke_ir_builder_t* b, int ki, void* user);
typedef void (*rocke_tile_mma_row_fn)(rocke_ir_builder_t* b, int ki, int mi, void* user);

typedef struct rocke_tile_mma_readers
{
    rocke_tile_mma_read_fn read_a;
    rocke_tile_mma_read_fn read_b;
    rocke_tile_mma_issue_fn mma;
    rocke_tile_mma_reads_fn after_reads;
    rocke_tile_mma_row_fn before_row;
    rocke_tile_mma_row_fn after_row;
    void* user;
} rocke_tile_mma_readers_t;

/* Staged, K/M/N order. Read each A row and B column once per K atom, then
 * call after_reads before any MMA. Indexed row hooks bracket each MMA row. */
void rocke_tile_mma_accumulate_staged(rocke_ir_builder_t* b,
                                      rocke_tile_mma_grid_t grid,
                                      rocke_value_t* const* accs,
                                      size_t count,
                                      const rocke_tile_mma_readers_t* readers,
                                      rocke_value_t** out);

#ifdef __cplusplus
}
#endif
#endif
