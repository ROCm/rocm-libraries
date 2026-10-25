/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_counted.h -- C port of
 * library/kernels/common/_attention_bwd_counted.py: the counted in-loop waits
 * of the attention backward q step (loop_sync = "counted").
 *
 *   Python (_attention_bwd_counted.py)  C (this header)
 *   ----------------------------------  ----------------------------------------
 *   COUNTED_ISSUES                      ROCKE_ATTN_BWD_COUNTED_ISSUES / _N_ISSUES
 *   COUNTED_VMCNT_MAX                   ROCKE_ATTN_BWD_COUNTED_VMCNT_MAX
 *   TileChunk                           rocke_attn_bwd_tile_chunk_t
 *   tile_chunks                         rocke_attn_bwd_tile_chunks_count / _tile_chunks
 *   issue_tile_loads                    rocke_attn_bwd_issue_tile_loads
 *   store_tile                          rocke_attn_bwd_store_tile
 *   zero_like                           rocke_attn_bwd_zero_like
 *   StagedStep (+ flat, from_flat)      rocke_attn_bwd_staged_step_t (+ _flat_len,
 *                                       _flat, _from_flat)
 *   CountedWaits (+ methods)            rocke_attn_bwd_counted_waits_t (+ _init,
 *                                       _step_entry, _loads, _atomics, _end_step,
 *                                       _barrier, _free)
 *
 * Python callables and views become a function pointer plus a context pointer:
 * a tile loader is (row, col) -> value, a tile view stores (row, col, value,
 * vec). Sequences are (pointer, count) pairs; every array an entry point fills
 * is sized by the caller (the *_count / *_flat_len functions give the size),
 * so no Python-unbounded length has a fixed-size buffer here. StagedStep does
 * not own its arrays.
 *
 * Errors (Python ValueError / AssertionError) are recorded on the builder's
 * sticky error as ROCKE_ERR_VALUE; entries return false (or NULL) after one.
 * Builder calls are made in Python's order.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_COUNTED_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_COUNTED_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Where the next step's global loads are issued: "early" or "late". */
#define ROCKE_ATTN_BWD_COUNTED_N_ISSUES 2
extern const char* const ROCKE_ATTN_BWD_COUNTED_ISSUES[ROCKE_ATTN_BWD_COUNTED_N_ISSUES];
/* Largest vmcnt a CDNA s_waitcnt encodes (6 bits on gfx942 / gfx950). */
#define ROCKE_ATTN_BWD_COUNTED_VMCNT_MAX 63

/* TileChunk: guard == NULL is Python's None (every thread's chunk is inside
 * the tile). */
typedef struct rocke_attn_bwd_tile_chunk
{
    rocke_value_t* row;
    rocke_value_t* col;
    rocke_value_t* guard;
    int total;
} rocke_attn_bwd_tile_chunk_t;

/* Number of chunks tile_chunks returns for these arguments (the deal rounds),
 * or -1 and a recorded ValueError when vec or threads is 0 (Python raises
 * ZeroDivisionError). */
int rocke_attn_bwd_tile_chunks_count(
    rocke_ir_builder_t* b, int rows, int cols, int threads, int vec);

/* tile_chunks(b, rows=, cols=, tid=, threads=, vec=8): fills out[0..count). */
bool rocke_attn_bwd_tile_chunks(rocke_ir_builder_t* b,
                                int rows,
                                int cols,
                                rocke_value_t* tid,
                                int threads,
                                int vec,
                                rocke_attn_bwd_tile_chunk_t* out,
                                int out_cap);

/* One loader (a Python Callable[[Value, Value], Value]). */
typedef rocke_value_t* (*rocke_attn_bwd_tile_load_fn)(void* ctx,
                                                      rocke_value_t* row,
                                                      rocke_value_t* col);
typedef struct rocke_attn_bwd_tile_loader
{
    rocke_attn_bwd_tile_load_fn fn;
    void* ctx;
} rocke_attn_bwd_tile_loader_t;

/* issue_tile_loads: every load of the tiles, tile by tile, chunk by chunk;
 * fills out[0 .. n_loaders * n_chunks). */
bool rocke_attn_bwd_issue_tile_loads(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_tile_loader_t* loads,
                                     int n_loads,
                                     const rocke_attn_bwd_tile_chunk_t* chunks,
                                     int n_chunks,
                                     rocke_value_t** out,
                                     int out_cap);

/* One LDS view: view.store(b, row, col, value, vec). */
typedef void (*rocke_attn_bwd_tile_store_fn)(void* ctx,
                                             rocke_ir_builder_t* b,
                                             rocke_value_t* row,
                                             rocke_value_t* col,
                                             rocke_value_t* value,
                                             int vec);
typedef struct rocke_attn_bwd_tile_view
{
    rocke_attn_bwd_tile_store_fn fn;
    void* ctx;
} rocke_attn_bwd_tile_view_t;

/* store_tile(b, views, chunks, values, vec=8). */
bool rocke_attn_bwd_store_tile(rocke_ir_builder_t* b,
                               const rocke_attn_bwd_tile_view_t* views,
                               int n_views,
                               const rocke_attn_bwd_tile_chunk_t* chunks,
                               int n_chunks,
                               rocke_value_t* const* values,
                               int n_values,
                               int vec);

/* zero_like(b, types): a zero of each type (f32 scalar or vector); fills
 * out[0 .. n). */
bool rocke_attn_bwd_zero_like(rocke_ir_builder_t* b,
                              const rocke_type_t* const* types,
                              int n,
                              rocke_value_t** out);

/* StagedStep (non-owning): stats then qdo. */
typedef struct rocke_attn_bwd_staged_step
{
    rocke_value_t* const* stats;
    int n_stats;
    rocke_value_t* const* qdo;
    int n_qdo;
} rocke_attn_bwd_staged_step_t;

int rocke_attn_bwd_staged_step_flat_len(const rocke_attn_bwd_staged_step_t* s);
/* flat(): out[0 .. flat_len) = stats then qdo. */
void rocke_attn_bwd_staged_step_flat(const rocke_attn_bwd_staged_step_t* s, rocke_value_t** out);
/* from_flat(values, n_stats): stats = values[:n_stats], qdo = the rest (the
 * Python slices clamp, so n_stats > n_values takes everything as stats). The
 * result points into values. */
rocke_attn_bwd_staged_step_t
    rocke_attn_bwd_staged_step_from_flat(rocke_value_t* const* values, int n_values, int n_stats);

/* CountedWaits. waits (n_waits of them, capacity cap_waits) is owned by the
 * struct; release it with rocke_attn_bwd_counted_waits_free. Do not copy a
 * struct that owns waits. */
typedef struct rocke_attn_bwd_counted_waits
{
    int n_atomics;
    int since_load;
    int issued;
    int* waits;
    int n_waits;
    int cap_waits;
} rocke_attn_bwd_counted_waits_t;

/* CountedWaits.__init__; false (and a recorded ValueError) when n_atomics is
 * outside [0, VMCNT_MAX]. */
bool rocke_attn_bwd_counted_waits_init(rocke_attn_bwd_counted_waits_t* w,
                                       rocke_ir_builder_t* b,
                                       int n_atomics);
void rocke_attn_bwd_counted_waits_step_entry(rocke_attn_bwd_counted_waits_t* w);
void rocke_attn_bwd_counted_waits_loads(rocke_attn_bwd_counted_waits_t* w);
void rocke_attn_bwd_counted_waits_atomics(rocke_attn_bwd_counted_waits_t* w, int n);
/* end_step: Python raises AssertionError when the step issued a different
 * number of atomics; here that is a recorded ROCKE_ERR_VALUE and false. */
bool rocke_attn_bwd_counted_waits_end_step(rocke_attn_bwd_counted_waits_t* w,
                                           rocke_ir_builder_t* b);
/* barrier: s_waitcnt vmcnt(n) lgkmcnt(0), then s_barrier (bare); records n. */
void rocke_attn_bwd_counted_waits_barrier(rocke_attn_bwd_counted_waits_t* w, rocke_ir_builder_t* b);
void rocke_attn_bwd_counted_waits_free(rocke_attn_bwd_counted_waits_t* w);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_COUNTED_H */
