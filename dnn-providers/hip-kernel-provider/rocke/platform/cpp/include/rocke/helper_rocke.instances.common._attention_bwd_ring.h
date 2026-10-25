/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_ring.h -- C port of the
 * host-side facts of the attention backward Q / dO prefetch ring
 * (library/kernels/common/_attention_bwd_ring.py) that the stage table
 * (_attention_bwd_sched) reads.
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
 *
 * Error model: a Python ValueError becomes ROCKE_ERR_VALUE plus str(exc) in a
 * rocke_attn_bwd_err_t (err may be NULL).
 *
 * Not ported yet (the body unit owns them): ring_slot_schedule,
 * dma_lane_source, PrefetchTile, LdsStatsView, QRing, PrefetchStep.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_RING_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_RING_H

#include <stdbool.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h" /* rocke_attn_bwd_err_t */

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

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_RING_H */
