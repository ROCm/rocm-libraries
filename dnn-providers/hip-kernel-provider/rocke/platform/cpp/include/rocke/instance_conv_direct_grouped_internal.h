/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_grouped_internal.h -- PRIVATE shared state + phase-
 * function contract for the C99 port of build_direct_conv_16c and
 * build_direct_conv_4c (rocke/instances/common/conv_direct_grouped.py).
 *
 * WHY THIS HEADER EXISTS.
 *   Each Python builder is a long function whose body shares one set of
 *   enclosing-function locals across several nested closures:
 *     16c: issue_dram_load(), store_to_lds(), lds_read_input(),
 *          lds_read_input_k32() -- all capture the builder, the param Values,
 *          every geometry constant, the SSA constants, the LDS smem handles,
 *          the buffer rsrcs, the chunk_meta table, and the A/B/D descriptors.
 *     4c:  no named closures, but the prologue computes a wide block of shared
 *          locals (descriptors, constants, precomputed s_consts, weights) that
 *          the unrolled H-loop body reads on every iteration.
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure into a free function taking a POINTER to one shared context struct
 *   (rocke_dconv_16c_ctx_t / rocke_dconv_4c_ctx_t) that holds EXACTLY the variables
 *   the closures/body share. The driver populates the ctx in the same order the
 *   Python prologue computes its locals, then calls the phase functions in
 *   Python order.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c TU binds
 *   to. It is DESIGNED TO BE COMPLETE: every local the Python body shares across
 *   phases is a field here. A body agent implementing a phase MUST be able to
 *   read/write only ctx fields and call the prototypes below WITHOUT editing
 *   this header. If a phase genuinely needs a value not present, that is a
 *   design bug to fix here once, deliberately.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `q_tile_start`
 *   -> `ctx->q_tile_start`; Python `c_BG_cpg` -> `ctx->c_BG_cpg`). Phase
 *   functions mirror the Python closure names with a `rocke_dconv16c_` /
 *   `rocke_dconv4c_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. Included only by the
 * instance_conv_direct_grouped*.c translation units. Public callers use
 * rocke/instance_conv_direct_grouped.h.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_GROUPED_INTERNAL_H
#define ROCKE_INSTANCE_CONV_DIRECT_GROUPED_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.transforms.h" /* rocke_tensor_descriptor_t */
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  Bound on the per-thread DRAM-load passes (16c).
 *  PASSES = ceil(NUM_VEC4 / THREADS). With THREADS = block_groups*wave
 *  (>= 64) and NUM_VEC4 = (block_q+KW-1)*block_groups*cpg/4, the largest
 *  legal config gives a small handful of passes; 16 is generous headroom.
 * ===================================================================== */
#define ROCKE_DCONV16C_MAX_PASSES 16

/* Bound on the per-block accumulator-tile fan-out (q_subtiles = block_q/16 for
 * 16c; q_tiles_per_wave = block_q/4 for 4c). block_q stays small (<=16 in the
 * covered space); 8 is generous. Each tile holds KH (=3) circular acc slots. */
#define ROCKE_DCONV_MAX_QTILES 8
/* KH for a 3x3 conv; the circular accumulator depth. Sized to the max KH the
 * builders unroll. */
#define ROCKE_DCONV_MAX_ACC_SLOTS 8

/* ===================================================================== *
 *  rocke_dconv_16c_ctx_t  --  shared state for build_direct_conv_16c.
 *
 *  Field order follows the Python prologue top-to-bottom (lines 256-642) so the
 *  populate routine reads straight against the source.
 * ===================================================================== */
typedef struct rocke_dconv_16c_ctx
{
    /* ---- inputs / resolved environment -- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`            */
    const rocke_direct_conv_16c_spec_t* spec; /* the DirectConv16cSpec        */
    const char* arch; /* NULL-normalised "gfx950"     */
    rocke_direct_conv_problem_t p; /* spec->problem (by value)     */

    /* ---- block-geometry scalars (Python all-caps locals) -- */
    int BLOCK_Q; /* spec.block_q                              */
    int BLOCK_GROUPS; /* spec.block_groups                         */
    int WAVE; /* spec.wave_size                            */
    int THREADS; /* spec.threads_per_block                    */
    int LDS_W; /* BLOCK_Q + KW - 1                          */
    int LDS_ROW_FP16; /* LDS_W * BLOCK_GROUPS * cpg                */
    int LOAD_VEC; /* 4                                         */
    int NUM_VEC4; /* LDS_ROW_FP16 / LOAD_VEC                    */
    int PASSES; /* ceil(NUM_VEC4 / THREADS)                  */
    int lds_total_fp16; /* PASSES * THREADS * LOAD_VEC                */
    int q_subtiles; /* BLOCK_Q // 16                             */
    int n_iters; /* H + KH - 1                                */

    /* ---- kernel params (Values) -- */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;

    /* ---- common SSA constants -- */
    rocke_value_t* c0; /* const_i32(0)                         */
    rocke_value_t* c_wave; /* const_i32(WAVE)                      */
    rocke_value_t* c_BG; /* const_i32(BLOCK_GROUPS)              */
    rocke_value_t* c_BQ; /* const_i32(BLOCK_Q)                   */
    rocke_value_t* c_cpg; /* const_i32(cpg)                       */
    rocke_value_t* c_kpg; /* const_i32(kpg)                       */
    rocke_value_t* c_W; /* const_i32(W)                         */
    rocke_value_t* c_BG_cpg; /* const_i32(BLOCK_GROUPS * cpg)        */
    rocke_value_t* c_half_bytes; /* const_i32(2)                         */
    rocke_value_t* oob_sentinel; /* const_i32((1<<31)-1)                 */
    rocke_value_t* fp16x4_zero; /* zero_vec_f16(4)                      */
    rocke_value_t* zero_acc; /* zero_vec_f32(4)                      */

    /* ---- thread / wave / lane decode (SSA) -- */
    rocke_value_t* tid; /* thread_id_x()                        */
    rocke_value_t* wave_id; /* tid / WAVE                           */
    rocke_value_t* lane; /* tid % WAVE                           */
    rocke_value_t* c4; /* lane / 16  (0..3)                    */
    rocke_value_t* q_in_lane; /* lane % 16  (0..15)                   */
    rocke_value_t* s_lane_k32; /* c4 / 2                               */
    rocke_value_t* ch_lane_k32; /* (c4 % 2) * 8                         */
    rocke_value_t* ch_lane_k16; /* c4 * 4                               */

    /* ---- grid / group decode (SSA) -- */
    rocke_value_t* bx; /* block_id_x()                         */
    rocke_value_t* by; /* block_id_y()                         */
    rocke_value_t* n; /* block_id_z()                         */
    rocke_value_t* g_tile; /* by                                   */
    rocke_value_t* g; /* g_tile*BLOCK_GROUPS + wave_id        */
    rocke_value_t* q_tile_start; /* bx * BLOCK_Q                         */

    /* ---- LDS ping-pong buffers + buffer rsrcs -- */
    rocke_value_t* A_smem; /* smem_alloc lds_a [1, lds_total_fp16] */
    rocke_value_t* B_smem; /* smem_alloc lds_b (or == A_smem)      */
    rocke_value_t* a_rsrc; /* buffer_rsrc(A, A_bytes)              */
    rocke_value_t* b_rsrc; /* buffer_rsrc(Bp, B_bytes)            */
    rocke_value_t* d_rsrc; /* buffer_rsrc(D, D_bytes)             */

    /* ---- weight loads (constant across the H-loop) -- */
    const rocke_tensor_descriptor_t* b_desc; /* B[total_k,KH,KW,cpg] naive    */
    rocke_value_t* k_out_val; /* g*kpg + q_in_lane             */
    /* fold_k32=False path: weights[r*KW+s], length KH*KW (<=9).            */
    rocke_value_t* weights[16];
    int n_weights;
    /* fold_k32=True path: per-r folded K=32 (S=0,1) + S=2 promoted to a
     * zero-padded K=32 atom, length KH (<=3). */
    rocke_value_t* weights_k32[8];
    rocke_value_t* weights_s2_k32[8];
    int n_weights_k32; /* == KH when fold_k32 else 0                        */
    /* lane_in_lo_half = cmp_lt(c4, 2); fp16x8_zero = zero_vec_f16(8).
     * Used to zero the upper-16-K (c4 in {2,3}) lanes of the S=2 wide atom. */
    rocke_value_t* lane_in_lo_half;
    rocke_value_t* fp16x8_zero;

    /* ---- per-thread chunk decode table (chunk_desc + chunk_meta) -- */
    const rocke_tensor_descriptor_t* chunk_desc; /* unmerge_magic decode      */
    /* chunk_meta[pass_idx]: one entry per DRAM pass. Mirrors the Python
     * dict {chunk_idx, ch_block, group_in_wg, W_lds, in_bounds, abs_group}. */
    struct
    {
        rocke_value_t* chunk_idx; /* tid + pass_idx*THREADS              */
        rocke_value_t* ch_block; /* decoded ch_block                   */
        rocke_value_t* group_in_wg; /* decoded group_in_wg                */
        rocke_value_t* W_lds; /* decoded W_lds                      */
        rocke_value_t* in_bounds; /* cmp_lt(chunk_idx, NUM_VEC4)         */
        rocke_value_t* abs_group; /* g_tile*BLOCK_GROUPS + group_in_wg   */
    } chunk_meta[ROCKE_DCONV16C_MAX_PASSES];
    int n_chunk_meta; /* == PASSES */

    /* ---- A / D descriptors (built once, reused) -- */
    const rocke_tensor_descriptor_t* a_desc; /* A[N,H,W,total_c] + 2 embeds  */
    const rocke_tensor_descriptor_t* d_desc; /* D[N,H,W,total_k] naive       */

    /* ---- accumulator iter-state across the unrolled H-loop -- *
     * acc_tiles[qt][slot]; q_subtiles x KH. Updated in place by the MFMA
     * phase and the flush/reset phase, exactly like the Python list-of-lists. */
    rocke_value_t* acc_tiles[ROCKE_DCONV_MAX_QTILES][ROCKE_DCONV_MAX_ACC_SLOTS];
} rocke_dconv_16c_ctx_t;

/* ===================================================================== *
 *  rocke_dconv_4c_ctx_t  --  shared state for build_direct_conv_4c.
 *
 *  Field order follows the Python prologue (lines 837-966).
 * ===================================================================== */
typedef struct rocke_dconv_4c_ctx
{
    /* ---- inputs / resolved environment -- */
    rocke_ir_builder_t* b;
    const rocke_direct_conv_4c_spec_t* spec;
    const char* arch;
    rocke_direct_conv_problem_t p; /* spec->problem (by value)             */

    int q_tiles_per_wave; /* block_q // 4                              */
    int n_iters; /* H + KH - 1                                */

    /* ---- kernel params (Values) -- */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;

    /* ---- common SSA constants -- */
    rocke_value_t* c0; /* const_i32(0)                          */
    rocke_value_t* c_W; /* const_i32(W)                          */
    rocke_value_t* c_cpg; /* const_i32(cpg)                        */
    rocke_value_t* c_kpg; /* const_i32(kpg)                        */
    rocke_value_t* c_half_bytes; /* const_i32(2)                          */
    rocke_value_t* oob_sentinel; /* const_i32((1<<31)-1)                  */
    rocke_value_t* fp16x4_zero; /* zero_vec_f16(4)                       */
    rocke_value_t* zero_acc; /* zero_vec_f32(4)                       */

    /* ---- thread / wave / lane decode (SSA) -- */
    rocke_value_t* tid; /* thread_id_x()                         */
    rocke_value_t* wave_id; /* tid / wave_size                       */
    rocke_value_t* lane; /* tid % wave_size                       */
    rocke_value_t* batch; /* lane / 4   (group within wave 0..15)  */
    rocke_value_t* lane_q; /* lane % 4                              */

    /* ---- grid / group decode (SSA) -- */
    rocke_value_t* bx;
    rocke_value_t* by;
    rocke_value_t* n; /* block_id_z()                          */
    rocke_value_t* q_tile_start; /* bx * block_q                          */
    rocke_value_t* group_in_wg; /* wave_id*16 + batch                    */
    rocke_value_t* g; /* by*block_groups + group_in_wg         */

    /* ---- buffer rsrcs -- */
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;

    /* ---- weights (per (r,s) per lane; length KH*KW <= 9) -- */
    const rocke_tensor_descriptor_t* b_desc; /* B[total_k,KH,KW,cpg] naive   */
    rocke_value_t* k_out_val; /* g*kpg + lane_q               */
    rocke_value_t* weights[16];
    int n_weights;

    /* ---- descriptors + precomputed loop-invariant locals -- */
    const rocke_tensor_descriptor_t* a_desc; /* A[N,H,W,total_c] + 2 embeds  */
    const rocke_tensor_descriptor_t* d_desc; /* D[N,H,W,total_k] naive       */
    rocke_value_t* c_val_groupc; /* g * cpg                      */
    rocke_value_t* s_consts[16]; /* const_i32(s) for s in KW     */
    int n_s_consts; /* KW                           */

    /* ---- accumulator iter-state across the unrolled H-loop -- *
     * acc_tiles[qt][slot]; q_tiles_per_wave x KH. */
    rocke_value_t* acc_tiles[ROCKE_DCONV_MAX_QTILES][ROCKE_DCONV_MAX_ACC_SLOTS];
} rocke_dconv_4c_ctx_t;

/* ===================================================================== *
 *  16c PHASE FUNCTIONS -- one per Python closure / prologue stage.
 *  Each phase reads/writes only ctx (+ the builder it carries) and emits IR in
 *  byte-identical Python order.
 * ===================================================================== */

/* Prologue (lines 256-355): validate(), is_valid_spec gate, derive every
 * geometry scalar, declare params, build all SSA constants, decode
 * thread/wave/lane + grid/group, alloc the LDS ping-pong, build buffer rsrcs.
 * Fills the corresponding ctx fields. Returns false (builder error set) on a
 * rejected spec or geometry violation. */
bool rocke_dconv16c_prologue(rocke_dconv_16c_ctx_t* ctx);

/* Weight-load phase (lines 357-415): build b_desc, k_out_val, and load
 * weights/weights_k32/weights_s2_k32 per the fold_k32 branch. */
void rocke_dconv16c_load_weights(rocke_dconv_16c_ctx_t* ctx);

/* Chunk-decode phase (lines 444-473): build chunk_desc (naive + unmerge_magic)
 * and populate chunk_meta[0..PASSES) via unmerge_lower. */
void rocke_dconv16c_build_chunk_meta(rocke_dconv_16c_ctx_t* ctx);

/* Descriptor phase (lines 475-519, 637-641): build a_desc (naive + 2 embeds)
 * and d_desc (naive). */
void rocke_dconv16c_build_descriptors(rocke_dconv_16c_ctx_t* ctx);

/* Closure: issue_dram_load(y_iter_val) (lines 521-562). Emits, for each
 * chunk_meta entry, the OOB-safe DRAM read of one vec4 of A at the given
 * (unshifted) output-row index. Writes up to `out_cap` (vec, lds_idx) pairs
 * into out_vecs[]/out_lds_idx[] and returns the count (== PASSES). */
int rocke_dconv16c_issue_dram_load(rocke_dconv_16c_ctx_t* ctx,
                                   rocke_value_t* y_iter_val,
                                   rocke_value_t** out_vecs,
                                   rocke_value_t** out_lds_idx,
                                   int out_cap);

/* Closure: store_to_lds(loads, lds) (lines 564-566). Stores the (vec, lds_idx)
 * pairs produced by issue_dram_load into the given LDS buffer. */
void rocke_dconv16c_store_to_lds(rocke_dconv_16c_ctx_t* ctx,
                                 rocke_value_t* const* vecs,
                                 rocke_value_t* const* lds_idx,
                                 int n,
                                 rocke_value_t* lds);

/* Closure: lds_read_input(q_subtile, s_const, lds) (lines 570-590). Per-lane
 * <4 x half> read from LDS for the s-th column of the 3-wide input row. */
rocke_value_t* rocke_dconv16c_lds_read_input(rocke_dconv_16c_ctx_t* ctx,
                                             int q_subtile,
                                             int s_const,
                                             rocke_value_t* lds);

/* Closure: lds_read_input_k32(q_subtile, lds) (lines 592-607). Per-lane
 * <8 x half> read for the folded K=32 MFMA. */
rocke_value_t* rocke_dconv16c_lds_read_input_k32(rocke_dconv_16c_ctx_t* ctx,
                                                 int q_subtile,
                                                 rocke_value_t* lds);

/* Closure: lds_read_input_s2_k32(q_subtile, lds). Per-lane <8 x half> read
 * for the S=2 residual promoted to a zero-padded K=32 MFMA (high half
 * zeroed via select(lane_in_lo_half, vec, fp16x8_zero)). */
rocke_value_t* rocke_dconv16c_lds_read_input_s2_k32(rocke_dconv_16c_ctx_t* ctx,
                                                    int q_subtile,
                                                    rocke_value_t* lds);

/* Prologue prefetch (lines 609-616): store_to_lds(issue_dram_load(c0), A_smem)
 * then sync(). Zero-fills row 0 (= -PAD) via the descriptor's embed validity. */
void rocke_dconv16c_prologue_prefetch(rocke_dconv_16c_ctx_t* ctx);

/* The unrolled H-row streaming loop (lines 618-739): for each of n_iters rows,
 * pick the cur/nxt ping-pong buffer, read inputs from cur (fold_k32 or per-s),
 * issue next-row DRAM loads, run the per-(qt,r[,s]) MFMA chain into the circular
 * acc slot, store next-row loads to nxt, sync, then conditionally flush the
 * oldest slot to D and unconditionally reset it. Reads/updates ctx->acc_tiles.
 * Builds and returns the kernel via ctx->b->kernel on success (NULL on error). */
rocke_kernel_def_t* rocke_dconv16c_stream_h_loop(rocke_dconv_16c_ctx_t* ctx);

/* ===================================================================== *
 *  4c PHASE FUNCTIONS.
 * ===================================================================== */

/* Prologue (lines 833-876): validate(), is_valid_spec gate, declare params,
 * build SSA constants, decode thread/wave/lane + grid/group, build buffer rsrcs.
 * Returns false on a rejected spec / geometry violation. */
bool rocke_dconv4c_prologue(rocke_dconv_4c_ctx_t* ctx);

/* Weight-load phase (lines 878-901): build b_desc, k_out_val, and load the
 * KH*KW weights[]. */
void rocke_dconv4c_load_weights(rocke_dconv_4c_ctx_t* ctx);

/* Descriptor + invariant phase (lines 903-965): build a_desc (naive + 2 embeds)
 * and d_desc (naive), seed acc_tiles to zero_acc, precompute c_val_groupc and
 * s_consts[]. */
void rocke_dconv4c_build_descriptors(rocke_dconv_4c_ctx_t* ctx);

/* The unrolled H-row loop (lines 967-1031): for each of n_iters rows, load the
 * per-(qt,s) OOB-safe A inputs, run the per-(qt,r,s) 4x4x4 MFMA chain into the
 * circular acc slot, then conditionally flush the oldest slot to D and
 * unconditionally reset it. Reads/updates ctx->acc_tiles. Returns the kernel
 * (ctx->b->kernel) on success, NULL on error. */
rocke_kernel_def_t* rocke_dconv4c_stream_h_loop(rocke_dconv_4c_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_GROUPED_INTERNAL_H */
