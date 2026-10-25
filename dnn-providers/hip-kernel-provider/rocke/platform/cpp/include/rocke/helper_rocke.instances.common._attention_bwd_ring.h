/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_ring.h -- C port of the
 * attention backward Q / dO prefetch ring
 * (library/kernels/common/_attention_bwd_ring.py): the host-side facts the
 * stage table (_attention_bwd_sched) reads and the ring's emission (the
 * slots, the register and DMA prefetch, one q step's part of it).
 *
 *   Python (_attention_bwd_ring.py)     C (this header)
 *   ---------------------------------   ---------------------------------------
 *   DMA_PIECE_BYTES                     ROCKE_ATTN_BWD_DMA_PIECE_BYTES
 *   RING_VMCNT_MAX                      ROCKE_ATTN_BWD_RING_VMCNT_MAX
 *   DmaBatch (+ __post_init__)          rocke_attn_bwd_dma_batch_t (+ _init)
 *     .pieces                             rocke_attn_bwd_dma_batch_pieces
 *     .instructions                       rocke_attn_bwd_dma_batch_n_instructions /
 *                                         _instruction
 *     .of_wave                            rocke_attn_bwd_dma_batch_of_wave
 *     .min_per_wave                       rocke_attn_bwd_dma_batch_min_per_wave
 *   RingWaits                           rocke_attn_bwd_ring_waits_t
 *   dma_wait_counts                     rocke_attn_bwd_dma_wait_counts
 *   ring_slot_schedule                  rocke_attn_bwd_ring_slot_schedule
 *   dma_lane_source (IR / HostInt)      rocke_attn_bwd_dma_lane_source / _host
 *   PrefetchTile                        rocke_attn_bwd_prefetch_tile_t
 *   LdsStatsView (+ rows, packed_rows)  rocke_attn_bwd_lds_stats_view_t (+ _rows,
 *                                       _packed_rows)
 *   QRing (+ methods)                   rocke_attn_bwd_q_ring_t (+ _init, _<method>)
 *   PrefetchStep (+ methods)            rocke_attn_bwd_prefetch_step_t (+ _<method>)
 *
 * Error model: a Python ValueError of the host functions becomes
 * ROCKE_ERR_VALUE plus str(exc) in a rocke_attn_bwd_err_t (err may be NULL).
 * The emitting functions (dma_lane_source, LdsStatsView, QRing,
 * PrefetchStep) record it on the builder's sticky error through
 * rocke_i_set_err, which raises ckc::Error: call them under
 * ckc::guard_builder, as the kernel entry points do; they return false (or
 * NULL) after an error. Every barrier carries ROCKE_ATTN_BWD_BARRIER_FENCE.
 *
 * Arrays the emitting functions return (the carried registers) are allocated
 * from the builder's arena. A ring borrows its geometry, LDS and stats view; a
 * prefetch step borrows its ring and tiles.
 *
 * Everything in _attention_bwd_ring.py is ported.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_RING_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_RING_H

#include <stdbool.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h" /* rocke_attn_bwd_err_t */
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes of one LDS DMA instruction of the ring (64 lanes x 16 bytes). */
#define ROCKE_ATTN_BWD_DMA_PIECE_BYTES 1024
/* Largest vmcnt a CDNA s_waitcnt encodes. */
#define ROCKE_ATTN_BWD_RING_VMCNT_MAX 63

/* DmaBatch: the DMA instructions that stage one q tile and their wave split.
 * Instruction k is ("q", j) for k < pieces, ("do", j) for k < 2 * pieces, then
 * ("lse2", 0) and ("dsum", 0); wave k mod waves issues it. */
typedef struct rocke_attn_bwd_dma_batch
{
    int block_m;
    int head_size;
    int waves;
    int wave_size; /* Python default 64 */
    int elem_bytes; /* Python default 2 */
} rocke_attn_bwd_dma_batch_t;

/* DmaBatch(block_m, head_size, waves, wave_size, elem_bytes) with its
 * __post_init__ checks. */
rocke_status_t rocke_attn_bwd_dma_batch_init(rocke_attn_bwd_dma_batch_t* out,
                                             int block_m,
                                             int head_size,
                                             int waves,
                                             int wave_size,
                                             int elem_bytes,
                                             rocke_attn_bwd_err_t* err);
int rocke_attn_bwd_dma_batch_pieces(const rocke_attn_bwd_dma_batch_t* d);
/* len(instructions) = 2 * pieces + 2. */
int rocke_attn_bwd_dma_batch_n_instructions(const rocke_attn_bwd_dma_batch_t* d);
/* instructions[k] as (*tensor, *index); false when k is out of range. */
bool rocke_attn_bwd_dma_batch_instruction(const rocke_attn_bwd_dma_batch_t* d,
                                          int k,
                                          const char** tensor,
                                          int* index);
/* of_wave(wave): writes up to cap instruction indices k (into instructions)
 * and returns the total count. */
int rocke_attn_bwd_dma_batch_of_wave(const rocke_attn_bwd_dma_batch_t* d,
                                     int wave,
                                     int* out,
                                     int cap);
int rocke_attn_bwd_dma_batch_min_per_wave(const rocke_attn_bwd_dma_batch_t* d);

/* RingWaits: ROCKE_ATTN_BWD_NONE is None (no wait there). */
typedef struct rocke_attn_bwd_ring_waits
{
    int prologue;
    int step_top;
    int step_end;
} rocke_attn_bwd_ring_waits_t;

/* dma_wait_counts(ring_depth, n_qd, n_a). */
rocke_status_t rocke_attn_bwd_dma_wait_counts(
    int ring_depth, int n_qd, int n_a, rocke_attn_bwd_ring_waits_t* out, rocke_attn_bwd_err_t* err);

/* ring_slot_schedule(ring_depth, path): offsets from the loop counter. */
typedef struct rocke_attn_bwd_ring_slots
{
    int read;
    int fill;
    int ahead;
} rocke_attn_bwd_ring_slots_t;

rocke_status_t rocke_attn_bwd_ring_slot_schedule(int ring_depth,
                                                 const char* path,
                                                 rocke_attn_bwd_ring_slots_t* out,
                                                 rocke_attn_bwd_err_t* err);

/* dma_lane_source(b, layout, piece, lane): (row, col) of the 16-byte vector
 * lane `lane` of DMA piece `piece` must fetch. */
bool rocke_attn_bwd_dma_lane_source(rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_dma_slab_layout_t* layout,
                                    int piece,
                                    rocke_value_t* lane,
                                    rocke_value_t** row,
                                    rocke_value_t** col);
/* The same on host ints (Python's HostInt builder). */
rocke_status_t rocke_attn_bwd_dma_lane_source_host(const rocke_attn_bwd_dma_slab_layout_t* layout,
                                                   int piece,
                                                   int lane,
                                                   int* row,
                                                   int* col,
                                                   rocke_attn_bwd_err_t* err);

/* PrefetchTile: has_slabs false is q_slab = do_slab = None; packed NULL is
 * None (borrowed). */
typedef struct rocke_attn_bwd_prefetch_tile
{
    rocke_value_t* q0;
    rocke_value_t* h;
    bool has_slabs;
    rocke_attn_bwd_row_slab_t q_slab;
    rocke_attn_bwd_row_slab_t do_slab;
    const rocke_attn_bwd_packed_step_t* packed;
} rocke_attn_bwd_prefetch_tile_t;

/* LdsStatsView(lse2, dsum, rows_left); rows_left NULL is None. */
typedef struct rocke_attn_bwd_lds_stats_view
{
    rocke_attn_bwd_lds_view_t lse2;
    rocke_attn_bwd_lds_view_t dsum;
    rocke_value_t* rows_left;
} rocke_attn_bwd_lds_stats_view_t;

/* rows / packed_rows: out_lse2[n_rows], out_dsum[n_rows] (h / q0 unused, as
 * in Python). */
bool rocke_attn_bwd_lds_stats_view_rows(const rocke_attn_bwd_lds_stats_view_t* s,
                                        rocke_ir_builder_t* b,
                                        rocke_value_t* const* rows,
                                        int n_rows,
                                        rocke_value_t** out_lse2,
                                        rocke_value_t** out_dsum);
bool rocke_attn_bwd_lds_stats_view_packed_rows(const rocke_attn_bwd_lds_stats_view_t* s,
                                               rocke_ir_builder_t* b,
                                               rocke_value_t* const* rows,
                                               int n_rows,
                                               rocke_value_t** out_lse2,
                                               rocke_value_t** out_dsum);

/* QRing: the per-CTA state of the ring. Borrows geom, lds and stats. */
typedef struct rocke_attn_bwd_q_ring
{
    const rocke_attn_bwd_tile_geometry_t* geom;
    const char* path;
    int depth;
    const rocke_attn_bwd_bwd_lds_t* lds;
    rocke_value_t* tid;
    rocke_value_t* lane;
    rocke_value_t* wave;
    const rocke_attn_bwd_stats_view_t* stats;
    int n_atomics;
    int stride_bytes;
    rocke_attn_bwd_lds_view_t v_q;
    rocke_attn_bwd_lds_view_t v_do;
    rocke_attn_bwd_ring_slots_t schedule;
    int offsets[4]; /* q, do, lse2, dsum */
    bool has_dma; /* path "dma": the members below are set */
    rocke_attn_bwd_dma_batch_t batch;
    rocke_attn_bwd_ring_waits_t waits;
    rocke_attn_bwd_dma_slab_layout_t layout;
    rocke_value_t* lds_base;
    rocke_value_t* lane_src[2];
} rocke_attn_bwd_q_ring_t;

/* QRing(b, geom, lds, tid=, lane=, wave=, stats=, n_atomics=); emits the
 * CTA-invariant lane terms of the DMA path. */
bool rocke_attn_bwd_q_ring_init(rocke_attn_bwd_q_ring_t* r,
                                rocke_ir_builder_t* b,
                                const rocke_attn_bwd_tile_geometry_t* geom,
                                const rocke_attn_bwd_bwd_lds_t* lds,
                                rocke_value_t* tid,
                                rocke_value_t* lane,
                                rocke_value_t* wave,
                                const rocke_attn_bwd_stats_view_t* stats,
                                int n_atomics);
/* check_step(table) (no IR). */
bool rocke_attn_bwd_q_ring_check_step(const rocke_attn_bwd_q_ring_t* r,
                                      rocke_ir_builder_t* b,
                                      const rocke_attn_bwd_stage_table_t* table);
/* slot_of(b, it, offset): NULL for one slot. */
rocke_value_t* rocke_attn_bwd_q_ring_slot_of(const rocke_attn_bwd_q_ring_t* r,
                                             rocke_ir_builder_t* b,
                                             rocke_value_t* it,
                                             int offset);
/* A ring slot: an int (slot_value NULL) or an i32 value; none true is None. */
typedef struct rocke_attn_bwd_ring_slot
{
    bool none;
    rocke_value_t* value;
    int index;
} rocke_attn_bwd_ring_slot_t;

rocke_attn_bwd_ring_slot_t rocke_attn_bwd_ring_slot_none(void);
rocke_attn_bwd_ring_slot_t rocke_attn_bwd_ring_slot_int(int index);
rocke_attn_bwd_ring_slot_t rocke_attn_bwd_ring_slot_val(rocke_value_t* value);

/* views(b, slot): the Q and dO views of the slot. */
bool rocke_attn_bwd_q_ring_views(const rocke_attn_bwd_q_ring_t* r,
                                 rocke_ir_builder_t* b,
                                 rocke_attn_bwd_ring_slot_t slot,
                                 rocke_attn_bwd_lds_view_t* v_q,
                                 rocke_attn_bwd_lds_view_t* v_do);
/* lds_stats(b, slot, tile). */
bool rocke_attn_bwd_q_ring_lds_stats(const rocke_attn_bwd_q_ring_t* r,
                                     rocke_ir_builder_t* b,
                                     rocke_attn_bwd_ring_slot_t slot,
                                     const rocke_attn_bwd_prefetch_tile_t* tile,
                                     rocke_attn_bwd_lds_stats_view_t* out);
/* issue_dma(b, tile, slot). */
bool rocke_attn_bwd_q_ring_issue_dma(const rocke_attn_bwd_q_ring_t* r,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_prefetch_tile_t* tile,
                                     rocke_attn_bwd_ring_slot_t slot);
/* prologue(b, trip, tiles, name=): *out / *n_out are the carried registers
 * (R = 3 on the register path; else none). */
bool rocke_attn_bwd_q_ring_prologue(const rocke_attn_bwd_q_ring_t* r,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t* trip,
                                    const rocke_attn_bwd_prefetch_tile_t* tiles,
                                    int n_tiles,
                                    const char* name,
                                    rocke_value_t*** out,
                                    int* n_out);
bool rocke_attn_bwd_q_ring_drain(const rocke_attn_bwd_q_ring_t* r, rocke_ir_builder_t* b);
int rocke_attn_bwd_q_ring_n_reg_chunks(const rocke_attn_bwd_q_ring_t* r);
/* load_regs(b, tile): *out / *n_out (arena). */
bool rocke_attn_bwd_q_ring_load_regs(const rocke_attn_bwd_q_ring_t* r,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_prefetch_tile_t* tile,
                                     rocke_value_t*** out,
                                     int* n_out);
/* store_regs(b, regs, slot). */
bool rocke_attn_bwd_q_ring_store_regs(const rocke_attn_bwd_q_ring_t* r,
                                      rocke_ir_builder_t* b,
                                      rocke_value_t* const* regs,
                                      int n_regs,
                                      rocke_attn_bwd_ring_slot_t slot);

/* PrefetchStep(ring, it=, here=, ahead=, carried=). Borrows the ring, the
 * tiles and carried; yields / regs are arena arrays the step fills. */
typedef struct rocke_attn_bwd_prefetch_step
{
    const rocke_attn_bwd_q_ring_t* ring;
    rocke_value_t* it;
    const rocke_attn_bwd_prefetch_tile_t* here;
    const rocke_attn_bwd_prefetch_tile_t* ahead;
    rocke_value_t* const* carried;
    int n_carried;
    rocke_value_t** yields;
    int n_yields;
    rocke_value_t** regs;
    int n_regs;
} rocke_attn_bwd_prefetch_step_t;

rocke_attn_bwd_prefetch_step_t
    rocke_attn_bwd_prefetch_step_make(const rocke_attn_bwd_q_ring_t* ring,
                                      rocke_value_t* it,
                                      const rocke_attn_bwd_prefetch_tile_t* here,
                                      const rocke_attn_bwd_prefetch_tile_t* ahead,
                                      rocke_value_t* const* carried,
                                      int n_carried);
bool rocke_attn_bwd_prefetch_step_barrier(rocke_attn_bwd_prefetch_step_t* p, rocke_ir_builder_t* b);
/* begin(b, step, h, q0, packed, defer_loads=): the slot's Q / dO views and the
 * stats source (Python's stats is always None). */
bool rocke_attn_bwd_prefetch_step_begin(rocke_attn_bwd_prefetch_step_t* p,
                                        rocke_ir_builder_t* b,
                                        bool defer_loads,
                                        rocke_attn_bwd_lds_view_t* v_q,
                                        rocke_attn_bwd_lds_view_t* v_do,
                                        rocke_attn_bwd_lds_stats_view_t* stats_src);
bool rocke_attn_bwd_prefetch_step_issue_loads(rocke_attn_bwd_prefetch_step_t* p,
                                              rocke_ir_builder_t* b);
bool rocke_attn_bwd_prefetch_step_end(rocke_attn_bwd_prefetch_step_t* p, rocke_ir_builder_t* b);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_RING_H */
