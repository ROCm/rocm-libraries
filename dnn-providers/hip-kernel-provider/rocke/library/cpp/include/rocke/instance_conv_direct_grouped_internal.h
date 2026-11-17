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

#include "rocke/arch_target.h" /* rocke_mma_op_t */
#include "rocke/helper_rocke.helpers.transforms.h" /* rocke_tensor_descriptor_t */
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  rocke_dconv_params_t  --  the direct-conv AOT kernarg block.
 *
 *  Emitted from rocke_conv_direct_arg_names() (instance_conv_abi.h), the
 *  twin of the library's kernels.common.conv_abi.conv_direct_arg_names(). Every
 *  direct-conv variant declares the same block in the same order, so one
 *  emitter serves them all; kernargs are packed positionally from the launch
 *  signature, so the order here IS the ABI.
 *
 *  What stays build-time: the filter extents (KH/KW), stride, PAD and the
 *  per-group channel counts. They shape the unrolled MFMA chain, the LDS row
 *  and the tap offsets -- kernel capabilities, not shape. What becomes a
 *  kernarg is everything that merely sizes the work.
 * ===================================================================== */
typedef struct rocke_dconv_params
{
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;
    rocke_value_t* p_N;
    rocke_value_t* p_Hi;
    rocke_value_t* p_Wi;
    rocke_value_t* p_Ho;
    rocke_value_t* p_Wo;
    rocke_value_t* p_groups;
    rocke_value_t* p_total_c;
    rocke_value_t* p_total_k;
    /* Strides of the tensors bound to the A and D operand slots. Forward runs
     * NHWC -> NHWK; dgrad runs the other way, so the two triples swap
     * layouts and the parameter names follow the tensor, not the slot. */
    rocke_value_t* p_A_stride_n;
    rocke_value_t* p_A_stride_hi;
    rocke_value_t* p_A_stride_wi;
    rocke_value_t* p_D_stride_n;
    rocke_value_t* p_D_stride_ho;
    rocke_value_t* p_D_stride_wo;
    /* wgrad only: X (NHWC) rides the B slot. */
    rocke_value_t* p_B_stride_n;
    rocke_value_t* p_B_stride_hi;
    rocke_value_t* p_B_stride_wi;
} rocke_dconv_params_t;

/* Declare the block. direction is "fwd", "dgrad" or "wgrad"; io_type is the
 * operand element type (rocke_f16() or rocke_bf16(); NULL means f16). Under
 * wgrad D is always the fp32 dW atomic target. */
void rocke_dconv_emit_params(rocke_ir_builder_t* b,
                             rocke_dconv_params_t* params,
                             const char* direction,
                             const rocke_type_t* io_type);

/* A[N,H,W,total_c] with runtime extents + the two conv-spatial embeds.
 * w_upper_0 / w_upper_1 name the column pair ("q_pos"/"W_lds_pos" for the
 * LDS-staged variants, "wo"/"s" for the DRAM-direct 4c one). */
struct rocke_dynamic_tensor_descriptor*
    rocke_dconv_a_descriptor_dynamic(rocke_ir_builder_t* b,
                                     const rocke_dconv_params_t* params,
                                     int pad,
                                     int stride,
                                     const char* w_upper_0,
                                     const char* w_upper_1);

/* D[N,Ho,Wo,total_k] with runtime extents. */
struct rocke_dynamic_tensor_descriptor*
    rocke_dconv_d_descriptor_dynamic(rocke_ir_builder_t* b, const rocke_dconv_params_t* params);

/* _krsc_descriptor_dynamic: [rows, KH, KW, inner] with a runtime innermost
 * extent `inner` -- strides [KH*KW*inner, KW*inner, inner, 1]. */
rocke_tensor_descriptor_t* rocke_dconv_krsc_descriptor_dynamic(rocke_ir_builder_t* b,
                                                               const char* name,
                                                               const char* const* coords,
                                                               rocke_value_t* inner,
                                                               int KH,
                                                               int KW);

/* How many row steps one scf.for body covers: lcm(2, KH) where the variant
 * ping-pongs LDS, KH alone where it reads straight from DRAM. */
int rocke_dconv_row_loop_unroll(int kh, bool lds_ping_pong);

/* ===================================================================== *
 *  fp32 storage / TF32 helpers (the "fp32 / TF32 helpers" section of
 *  conv_direct_grouped.py). The 16-bit paths never call them.
 *
 *   Python                          C++
 *   -----------------------------   ------------------------------------
 *   _storage_dtype(dtype)           rocke_dconv_storage_dtype (NULL: unknown)
 *   _elem_bytes(dtype)              rocke_dconv_elem_bytes
 *   _is_f32_storage(dtype)          rocke_dconv_is_f32_storage
 *   _f32_mma_plan(arch, ...)        rocke_dconv_f32_mma_plan
 *   _load_elems(b, dtype, ...)      rocke_dconv_load_elems
 *   _store_from_f32(b, dtype, ...)  rocke_dconv_store_from_f32
 *   _mma_f32_logical(b, arch, ...)  rocke_dconv_mma_f32_logical
 * ===================================================================== */

/* Problem dtype -> storage dtype: fp16 -> "fp16", bf16 -> "bf16", fp32 and
 * tf32 -> "fp32"; NULL for any other dtype (Python raises there). */
const char* rocke_dconv_storage_dtype(const char* dtype);
/* Bytes per stored element: 4 for fp32 storage, else 2. */
int rocke_dconv_elem_bytes(const char* dtype);
/* True for the fp32-storage dtypes ("fp32" and "tf32"). */
bool rocke_dconv_is_f32_storage(const char* dtype);

/* How one logical m x m MMA with `frag` K-elements per lane is issued: `chunks`
 * physical MMAs on `op` over consecutive op-fragment-wide slices, after
 * zero-padding the fragment by `pad` elements. */
typedef struct rocke_dconv_f32_mma_plan
{
    const rocke_mma_op_t* op;
    int chunks;
    int pad;
} rocke_dconv_f32_mma_plan_t;

/* _f32_mma_plan: false when no m x m atom of the arch serves fp32-storage
 * `dtype` (Python None), or `arch` is unknown. Prefers an exact split, else
 * the narrowest atom wider than the fragment. */
bool rocke_dconv_f32_mma_plan(
    const char* arch, const char* dtype, int m, int frag, rocke_dconv_f32_mma_plan_t* out);

/* _load_elems: n contiguous elements of `dtype` from a buffer. 16-bit dtypes
 * are one vN load of n/2 dwords; fp32 storage is dwordx4 (or narrower) pieces
 * at consecutive byte offsets, concatenated. */
rocke_value_t* rocke_dconv_load_elems(rocke_ir_builder_t* b,
                                      const char* dtype,
                                      rocke_value_t* rsrc,
                                      rocke_value_t* voff,
                                      rocke_value_t* soff,
                                      int n);

/* _store_from_f32: store n fp32 accumulators as a `dtype` output (16-bit:
 * truncate; fp32 storage: as is, in dwordx4 or narrower chunks). */
void rocke_dconv_store_from_f32(rocke_ir_builder_t* b,
                                const char* dtype,
                                rocke_value_t* rsrc,
                                rocke_value_t* voff,
                                rocke_value_t* soff,
                                rocke_value_t* acc,
                                int n);

/* _acc_to_storage: fp32 accumulators in the storage type of `dtype` (16-bit:
 * truncated; fp32 storage: as they are), for a non-global store (an LDS tile). */
rocke_value_t*
    rocke_dconv_acc_to_storage(rocke_ir_builder_t* b, const char* dtype, rocke_value_t* acc);

/* _mma_f32_logical: one logical m x m MMA on <frag x float> operands. The
 * fragment is split into (or zero-padded up to, A and B together) the plan's
 * catalog atom; for "tf32" each piece is reinterpreted as vec<tf32 x N>. */
rocke_value_t* rocke_dconv_mma_f32_logical(rocke_ir_builder_t* b,
                                           const char* arch,
                                           const char* dtype,
                                           int m,
                                           rocke_value_t* a,
                                           rocke_value_t* bv,
                                           rocke_value_t* acc);

/* ===================================================================== *
 *  Bound on the per-thread DRAM-load passes (16c).
 *  PASSES = ceil(NUM_VEC4 / THREADS). With THREADS = block_groups*wave
 *  (>= 64) and NUM_VEC4 = (block_q+KW-1)*block_groups*cpg/4, the largest
 *  legal config gives a small handful of passes; 16 is generous headroom.
 * ===================================================================== */
#define ROCKE_DCONV16C_MAX_PASSES 16

/* Bound on the per-thread DRAM-load passes (32c).
 * cpg=32 and block_q up to 128 with KW=3 gives the worst case:
 *   THREADS=256 (BG=4, wave=64), NUM_VEC4=(128+2)*4*32/4=4160, PASSES=17.
 * ===================================================================== */
#define ROCKE_DCONV32C_MAX_PASSES 17

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
    const rocke_type_t* io_type; /* rocke_f16() or rocke_bf16() depending on p.dtype */

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
    int lds_total_elems; /* PASSES * THREADS * LOAD_VEC                */
    int q_subtiles; /* BLOCK_Q // 16                             */
    int n_iters; /* H + KH - 1                                */

    /* ---- kernel params (Values) -- */
    rocke_dconv_params_t params; /* the AOT kernarg block (ABI order)    */
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
    rocke_value_t* c_W; /* p_Wo -- the output-width store bound  */
    rocke_value_t* c_stride; /* const_i32(p.stride)                  */
    rocke_value_t* c_BG_cpg; /* const_i32(BLOCK_GROUPS * cpg)        */
    rocke_value_t* c_half_bytes; /* const_i32(2)                         */
    rocke_value_t* oob_sentinel; /* const_i32((1<<31)-1)                 */
    rocke_value_t* io_vec4_zero; /* zero_vec_f16(4)                      */
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
    rocke_value_t* A_smem; /* smem_alloc lds_a [1, lds_total_elems] */
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
    const rocke_tensor_descriptor_t* d_desc; /* D[N,Ho,Wo,total_k] naive     */

    /* ---- accumulator iter-state across the unrolled H-loop -- *
     * acc_tiles[qt][slot]; q_subtiles x KH. Updated in place by the MFMA
     * phase and the flush/reset phase, exactly like the Python list-of-lists. */
    rocke_value_t* acc_tiles[ROCKE_DCONV_MAX_QTILES][ROCKE_DCONV_MAX_ACC_SLOTS];
} rocke_dconv_16c_ctx_t;

/* Upper bound on the 4c fused-dgrad LDS staging passes; mirrors
 * DCONV4C_MAX_WL_PASSES in the Python emitter and is enforced by
 * rocke_direct_conv_4c_is_valid_spec. */
#define ROCKE_DCONV4C_MAX_WL_PASSES 32
/* Upper bound on 4c filter taps KH*KW: sizes the 4c ctx weights[] and
 * s_consts[] arrays; mirrors DCONV4C_MAX_TAPS in the Python emitter and is
 * enforced by rocke_direct_conv_4c_is_valid_spec. */
#define ROCKE_DCONV4C_MAX_TAPS 16
/* Most threads in one 4c workgroup (stage_rows with waves_q > 1); mirrors
 * DCONV4C_MAX_THREADS in the Python emitter. */
#define ROCKE_DCONV4C_MAX_THREADS 1024
/* Upper bound on the stage_rows input-row staging passes; mirrors
 * DCONV4C_MAX_ROW_PASSES and is enforced by rocke_direct_conv_4c_is_valid_spec. */
#define ROCKE_DCONV4C_MAX_ROW_PASSES 8

/* LDS geometry of the stage_rows 4c kernel (Python _staged_4c_geometry);
 * element counts are in the io dtype. */
typedef struct rocke_dconv4c_staged_geo
{
    int bc;
    int lds_w;
    int vpc;
    int row_stride;
    int nchunk;
    int passes;
    int row_elems;
    int lds_bytes;
} rocke_dconv4c_staged_geo_t;

void rocke_dconv4c_staged_geometry(const rocke_direct_conv_4c_spec_t* spec,
                                   rocke_dconv4c_staged_geo_t* geo);

/* Python _stage_rows_reject_reason: true (and the reason) when stage_rows /
 * waves_q are illegal on spec. The reason text is the one shared by
 * rocke_direct_conv_4c_validate (prefixed "DirectConv4cSpec ") and
 * rocke_direct_conv_4c_is_valid_spec. */
bool rocke_dconv4c_stage_rows_reject(const rocke_direct_conv_4c_spec_t* spec,
                                     char* reason,
                                     size_t reason_cap);

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
    const rocke_type_t* io_type; /* _io_type(p.dtype): f16 or bf16       */
    int is_bf16; /* p.dtype == "bf16"                         */

    /* ---- kernel params (Values) -- */
    rocke_dconv_params_t params; /* the AOT kernarg block (ABI order)    */
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
    rocke_value_t* io_vec4_zero; /* zero_vec(io_type, 4)                  */
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
    rocke_value_t* weights[ROCKE_DCONV4C_MAX_TAPS];
    int n_weights;

    /* ---- descriptors + precomputed loop-invariant locals -- */
    const rocke_tensor_descriptor_t* a_desc; /* A[N,H,W,total_c] + 2 embeds  */
    const rocke_tensor_descriptor_t* d_desc; /* D[N,H,W,total_k] naive       */
    rocke_value_t* c_val_groupc; /* g * cpg                      */
    rocke_value_t* s_consts[ROCKE_DCONV4C_MAX_TAPS]; /* const_i32(s) for s in KW */
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

/* spec.validate() + is_valid_spec_4c gate shared by both 4c builders: on a
 * rejected spec sets the builder error and returns false. */
bool rocke_dconv4c_check_spec(rocke_dconv_4c_ctx_t* ctx);

/* The whole row-staged 4c kernel (Python _build_direct_conv_4c_staged), run
 * when spec.stage_rows. Checks the spec itself (rocke_dconv4c_check_spec);
 * returns the kernel or NULL with the builder error set. */
rocke_kernel_def_t* rocke_dconv4c_build_staged(rocke_dconv_4c_ctx_t* ctx);

/* The unrolled H-row loop (lines 967-1031): for each of n_iters rows, load the
 * per-(qt,s) OOB-safe A inputs, run the per-(qt,r,s) 4x4x4 MFMA chain into the
 * circular acc slot, then conditionally flush the oldest slot to D and
 * unconditionally reset it. Reads/updates ctx->acc_tiles. Returns the kernel
 * (ctx->b->kernel) on success, NULL on error. */
rocke_kernel_def_t* rocke_dconv4c_stream_h_loop(rocke_dconv_4c_ctx_t* ctx);

/* ===================================================================== *
 *  rocke_dconv_8c_ctx_t  --  shared state for build_direct_conv_8c.
 *
 *  Structurally identical to the 16c ctx except:
 *    - No fold_k32 flag (8c always folds s=0 + s=1 into K=16 and handles s=2
 *      as a zero-padded residual).
 *    - ch_block_dim = 2 (cpg=8 → 2 vec4 blocks of 4 channels per group) in
 *      the LDS chunk descriptor instead of 4.
 *    - weights_main[KH]: one folded K=16 vec4 per filter row (both S positions).
 *    - weights_s2[KH]: one zero-padded K=16 vec4 per filter row (S=2 residual).
 *    - s_lane, ch_lane instead of s_lane_k32/ch_lane_k32/ch_lane_k16.
 *    - lane_in_lo_half selects which K-lanes carry valid S=2 data.
 *    - zero_acc is <4 x float> (same as 16c non-folded path).
 *    - q_subtiles = block_q / 16.
 * ===================================================================== */
typedef struct rocke_dconv_8c_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_conv_8c_spec_t* spec;
    const char* arch;
    rocke_direct_conv_problem_t p;
    const rocke_type_t* io_type; /* rocke_f16() or rocke_bf16() */

    int BLOCK_Q;
    int BLOCK_GROUPS;
    int WAVE;
    int THREADS;
    int LDS_W;
    int LDS_ROW_FP16;
    int LOAD_VEC;
    int NUM_VEC4;
    int PASSES;
    int lds_total_elems;
    int q_subtiles; /* BLOCK_Q / 16 */
    int n_iters;

    rocke_dconv_params_t params; /* the AOT kernarg block (ABI order)    */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;

    rocke_value_t* c0;
    rocke_value_t* c_wave;
    rocke_value_t* c_BG;
    rocke_value_t* c_BQ;
    rocke_value_t* c_cpg;
    rocke_value_t* c_kpg;
    rocke_value_t* c_W;
    rocke_value_t* c_stride; /* const_i32(p.stride)                  */
    rocke_value_t* c_BG_cpg;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* io_vec4_zero;
    rocke_value_t* zero_acc; /* <4 x float> */

    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;
    rocke_value_t* c4; /* lane / 16 */
    rocke_value_t* q_in_lane; /* lane % 16 */
    rocke_value_t* s_lane; /* c4 / 2   (0,0,1,1 for c4 in {0,1,2,3}) */
    rocke_value_t* ch_lane; /* (c4%2)*4 (0,4,0,4) */
    rocke_value_t* lane_in_lo_half; /* cmp_lt(c4, 2) — selects valid S=2 lanes */

    rocke_value_t* bx;
    rocke_value_t* by;
    rocke_value_t* n;
    rocke_value_t* g_tile;
    rocke_value_t* g;
    rocke_value_t* q_tile_start;

    rocke_value_t* A_smem;
    rocke_value_t* B_smem;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;

    const rocke_tensor_descriptor_t* b_desc;
    rocke_value_t* k_out_val;
    rocke_value_t* weights_main[ROCKE_DCONV_MAX_ACC_SLOTS]; /* one per KH (s=0+s=1 folded) */
    rocke_value_t* weights_s2[ROCKE_DCONV_MAX_ACC_SLOTS]; /* one per KH (s=2 residual) */
    int n_weights; /* == KH */

    const rocke_tensor_descriptor_t* chunk_desc;
    struct
    {
        rocke_value_t* chunk_idx;
        rocke_value_t* ch_block;
        rocke_value_t* group_in_wg;
        rocke_value_t* W_lds;
        rocke_value_t* in_bounds;
        rocke_value_t* abs_group;
    } chunk_meta[ROCKE_DCONV16C_MAX_PASSES];
    int n_chunk_meta;

    const rocke_tensor_descriptor_t* a_desc;
    const rocke_tensor_descriptor_t* d_desc;

    rocke_value_t* acc_tiles[ROCKE_DCONV_MAX_QTILES][ROCKE_DCONV_MAX_ACC_SLOTS];
} rocke_dconv_8c_ctx_t;

/* ===================================================================== *
 *  rocke_dconv_32c_ctx_t  --  shared state for build_direct_conv_32c.
 *
 *  Uses mfma_f32_32x32x8_f16: M=32=kpg, N=BLOCK_Q, K=8.
 *  Per (r,s): 4 consecutive MFMA calls (atom_idx=0..3, ch_start=0,8,16,24).
 *  Accumulator: <16 x float> (32*32/64 = 16 slots per lane).
 *  q_subtiles = BLOCK_Q / 32.
 * ===================================================================== */
#define ROCKE_DCONV32C_MAX_ATOMS 4 /* 4 atoms per (r,s) for cpg=32 */
#define ROCKE_DCONV32C_MAX_KH 8
#define ROCKE_DCONV32C_MAX_KW 8

typedef struct rocke_dconv_32c_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_conv_32c_spec_t* spec;
    const char* arch;
    rocke_direct_conv_problem_t p;
    const rocke_type_t* io_type; /* rocke_f16() or rocke_bf16() */

    int BLOCK_Q;
    int BLOCK_GROUPS;
    int WAVE;
    int THREADS;
    int LDS_W;
    int LDS_ROW_FP16;
    int LOAD_VEC;
    int NUM_VEC4;
    int PASSES;
    int N_CH_BLOCKS; /* cpg / 4 = 8 for cpg=32 */
    int lds_total_elems;
    int q_subtiles; /* BLOCK_Q / 32 */
    int n_iters;

    rocke_dconv_params_t params; /* the AOT kernarg block (ABI order)    */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;

    rocke_value_t* c0;
    rocke_value_t* c_wave;
    rocke_value_t* c_BG;
    rocke_value_t* c_BQ;
    rocke_value_t* c_cpg;
    rocke_value_t* c_kpg;
    rocke_value_t* c_W;
    rocke_value_t* c_stride; /* const_i32(p.stride)                  */
    rocke_value_t* c_BG_cpg;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* io_vec4_zero;
    rocke_value_t* zero_acc; /* <16 x float> */

    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;
    rocke_value_t* q_in_lane; /* lane % 32 */
    rocke_value_t* k_blk; /* lane / 32 (0 or 1) */
    rocke_value_t* ch_in_atom; /* k_blk * 4 */

    rocke_value_t* bx;
    rocke_value_t* by;
    rocke_value_t* n;
    rocke_value_t* g_tile;
    rocke_value_t* g;
    rocke_value_t* q_tile_start;

    rocke_value_t* A_smem;
    rocke_value_t* B_smem;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;

    const rocke_tensor_descriptor_t* b_desc;
    rocke_value_t* k_out_val;
    /* weights[r][s][atom]: KH x KW x 4 weight vec4s */
    rocke_value_t* weights[ROCKE_DCONV32C_MAX_KH][ROCKE_DCONV32C_MAX_KW][ROCKE_DCONV32C_MAX_ATOMS];
    int n_weight_r; /* KH */
    int n_weight_s; /* KW */

    const rocke_tensor_descriptor_t* chunk_desc;
    struct
    {
        rocke_value_t* chunk_idx;
        rocke_value_t* ch_block;
        rocke_value_t* group_in_wg;
        rocke_value_t* W_lds;
        rocke_value_t* in_bounds;
        rocke_value_t* abs_group;
    } chunk_meta[ROCKE_DCONV32C_MAX_PASSES];
    int n_chunk_meta;

    const rocke_tensor_descriptor_t* a_desc;
    const rocke_tensor_descriptor_t* d_desc;

    /* acc_tiles[qt][slot]: <16 x float> per (q_subtile, circular slot) */
    rocke_value_t* acc_tiles[ROCKE_DCONV_MAX_QTILES][ROCKE_DCONV_MAX_ACC_SLOTS];
} rocke_dconv_32c_ctx_t;

/* ===================================================================== *
 *  rocke_dconv_dw_ctx_t  --  shared state for build_direct_depthwise.
 *
 *  No LDS, no MFMA.  Each lane owns one channel and accumulates
 *  KH*KW scalar FMA products into a BLOCK_W * KH circular register array.
 * ===================================================================== */
#define ROCKE_DCONV_DW_MAX_BLOCK_W 64
/* ROCKE_DCONV_DW_MAX_KH / _MAX_KW live in the public header: they bound the
 * weight tables below AND are enforced by both validators. */

typedef struct rocke_dconv_dw_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_depthwise_spec_t* spec;
    const char* arch;
    rocke_direct_conv_problem_t p;

    int BLOCK_W;
    int BLOCK_WAVES;
    int WAVE;
    int THREADS;
    int BLOCK_CH;
    int n_iters;
    int Ho; /* output height: (H + 2*PAD - KH) / stride + 1 */
    int Wo; /* output width:  (W + 2*PAD - KW) / stride + 1 */
    int c_stride_dw; /* p.stride (kept as int for flush-loop modulo) */
    int is_bf16; /* 1 when p.dtype == "bf16", 0 otherwise */
    const char* dtype; /* p.dtype (fp32 / tf32: fp32 storage, scalar_load/store) */

    rocke_dconv_params_t params; /* the AOT kernarg block (ABI order)    */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;

    rocke_value_t* c0;
    rocke_value_t* c_wave;
    rocke_value_t* c_W; /* const_i32(Wo) — output width          */
    rocke_value_t* c_groups; /* const_i32(groups)                     */
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* zero_f32;
    rocke_value_t* ch_in_range; /* ch < p.groups — guards partial channel tile */

    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;

    rocke_value_t* bx;
    rocke_value_t* by;
    rocke_value_t* n;
    rocke_value_t* q_tile_start;
    rocke_value_t* ch; /* absolute channel: by*BLOCK_CH + wave_id*WAVE + lane */

    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;

    const rocke_tensor_descriptor_t* a_desc; /* A[N,H,W,C] + 2 embeds */
    const rocke_tensor_descriptor_t* b_desc; /* B[total_k,KH,KW,1] naive */
    const rocke_tensor_descriptor_t* d_desc; /* D[N,Ho,Wo,total_k] naive */

    /* weights_f32[r][s]: KH x KW preloaded f32 scalars */
    rocke_value_t* weights_f32[ROCKE_DCONV_DW_MAX_KH][ROCKE_DCONV_DW_MAX_KW];

    /* acc[w_out][slot]: BLOCK_W x KH f32 circular accumulators */
    rocke_value_t* acc[ROCKE_DCONV_DW_MAX_BLOCK_W][ROCKE_DCONV_MAX_ACC_SLOTS];
} rocke_dconv_dw_ctx_t;

/* ===================================================================== *
 *  rocke_dconv_dwcol_ctx_t  --  shared state for build_direct_depthwise_col.
 *
 *  Column-streamed sibling of the above: the KW axis is a runtime scf.for
 *  whose iter_args carry a BLOCK_H x BLOCK_W accumulator band, so the only
 *  register arrays are KH weights and BLOCK_H*BLOCK_W accumulators -- both
 *  bounded by the validator's live-f32 ceiling (arch VGPRs * 3/8), not by a
 *  small compile-time cap.  KH reaches 31+ in the covered space, so neither
 *  array is a fixed ctx member: the emitter alloca's them per call.
 *
 *  AOT: the image extents, group count and tensor strides come from the
 *  kernarg block; only the filter, PAD, stride and the tile geometry are
 *  build-time, so every bounds guard is emitted against a kernarg.
 * ===================================================================== */
typedef struct rocke_dconv_dwcol_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_depthwise_col_spec_t* spec;
    const char* arch;
    rocke_direct_conv_problem_t p;
    rocke_dconv_params_t params; /* the AOT kernarg block (ABI order)    */

    int BLOCK_H; /* output rows per block (the unrolled row tile)  */
    int BLOCK_W;
    int BLOCK_WAVES;
    int WAVE;
    int THREADS;
    int BLOCK_CH;
    int n_iters; /* (BLOCK_H - 1) * stride + KH                    */
    int ELEM_BYTES; /* 2 for f16/bf16                                 */
    const rocke_type_t* DT; /* dtype_to_ir(spec->dtype)                   */

    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;

    rocke_value_t* c0;
    rocke_value_t* c1; /* emitted SECOND, unlike the preload prologue */
    rocke_value_t* c_wave;
    rocke_value_t* c_W; /* p_Wo kernarg -- output width           */
    rocke_value_t* c_groups; /* p_groups kernarg                      */
    rocke_value_t* c_elem_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* zero_f32;
    rocke_value_t* ch_in_range; /* ch < groups                            */

    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;

    rocke_value_t* bx;
    rocke_value_t* by;
    rocke_value_t* bz;
    rocke_value_t* n; /* bz / n_h_tiles                          */
    rocke_value_t* ho_start; /* (bz % n_h_tiles) * BLOCK_H              */
    rocke_value_t* y_start; /* ho_start * stride: first padded input row */
    rocke_value_t* q_tile_start;
    rocke_value_t* ch; /* absolute channel: by*BLOCK_CH + wave_id*WAVE + lane */

    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;

    const rocke_tensor_descriptor_t* a_desc; /* A[N,H,W,C] runtime + 2 embeds */
    const rocke_tensor_descriptor_t* b_desc; /* B[total_k,KH,KW,1] naive      */
    const rocke_tensor_descriptor_t* d_desc; /* D[N,Ho,Wo,total_k] runtime    */
} rocke_dconv_dwcol_ctx_t;

/* ===================================================================== *
 *  8c PHASE FUNCTIONS
 * ===================================================================== */
bool rocke_dconv8c_prologue(rocke_dconv_8c_ctx_t* ctx);
void rocke_dconv8c_load_weights(rocke_dconv_8c_ctx_t* ctx);
void rocke_dconv8c_build_chunk_meta(rocke_dconv_8c_ctx_t* ctx);
void rocke_dconv8c_build_descriptors(rocke_dconv_8c_ctx_t* ctx);
void rocke_dconv8c_prologue_prefetch(rocke_dconv_8c_ctx_t* ctx);
rocke_kernel_def_t* rocke_dconv8c_stream_h_loop(rocke_dconv_8c_ctx_t* ctx);

/* ===================================================================== *
 *  32c PHASE FUNCTIONS
 * ===================================================================== */
bool rocke_dconv32c_prologue(rocke_dconv_32c_ctx_t* ctx);
void rocke_dconv32c_load_weights(rocke_dconv_32c_ctx_t* ctx);
void rocke_dconv32c_build_chunk_meta(rocke_dconv_32c_ctx_t* ctx);
void rocke_dconv32c_build_descriptors(rocke_dconv_32c_ctx_t* ctx);
void rocke_dconv32c_prologue_prefetch(rocke_dconv_32c_ctx_t* ctx);
rocke_kernel_def_t* rocke_dconv32c_stream_h_loop(rocke_dconv_32c_ctx_t* ctx);

/* ===================================================================== *
 *  rocke_dconv_wgrad_ctx_t  --  shared state for build_direct_conv_wgrad.
 *
 *  The Python body is one long function with six closures (_tr_read, _read_dy,
 *  _read_strip, _lds_run, _issue_delta / _commit_delta, _issue_s_strip /
 *  _commit_s_strip, _row_coords) over a wide block of prologue locals. Every
 *  local those closures share is a field here; field order follows the Python
 *  prologue top-to-bottom so the populate routine reads against the source.
 *
 *  Unlike the forward variants there is NO LDS ping-pong and no chunk-decode
 *  table: the row loop stages one dY tile + one S-row strip per input row into
 *  wave-private LDS partitions, and the epilogue atomically adds the KH*KW
 *  accumulators into an fp32 dW.
 * ===================================================================== */
/* ROCKE_DCONV_WGRAD_MAX_KH / _MAX_KW live in the public header: they bound the
 * ctx arrays below AND are enforced by both validators. */
/* STRIP_PASSES = ceil((WO_BLOCK + KW - 1) / WO_BLOCK) is 2 for every legal
 * (WO_BLOCK >= 16, KW <= 8) combination, and STRIP_PASSES_PER_WAVE <=
 * STRIP_PASSES; 8 is generous headroom. */
#define ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES 8
/* dY load passes per tile: 1 on wave64, 2 on wave32 (WO_BLOCK / 16). */
#define ROCKE_DCONV_WGRAD_MAX_DY_PASSES 4

typedef struct rocke_dconv_wgrad_ctx
{
    /* ---- inputs / resolved environment -- */
    rocke_ir_builder_t* b;
    const rocke_direct_conv_wgrad_spec_t* spec;
    const char* arch; /* NULL-normalised "gfx950" */
    rocke_direct_conv_problem_t p;

    /* ---- geometry scalars (Python all-caps locals) --
     * AOT: no image extent is build-time; Ho/Wo/H ride the kernarg block. */
    int KH;
    int KW;
    int WAVE_K; /* spec.wave_tile_k (16)                      */
    int WAVE_C; /* spec.wave_tile_c (16)                      */
    int WAVES_K;
    int WAVES_C;
    int WAVES_Q;
    int WAVE; /* spec.wave_size                             */
    int THREADS; /* spec.threads_per_block                     */
    int HPB; /* spec.ho_per_block                          */
    int WO_BLOCK; /* spec.mfma_k                                */
    int VEC_CH; /* spec.mfma_k / 4                            */
    /* wave32 (WMMA 16x16x32) splits VEC_CH's three wave64 roles: */
    int W32; /* 1 when WAVE == 32                          */
    int LD_VEC; /* channels per lane per load (8 | VEC_CH)    */
    int FRAG; /* MMA operand elements per lane (16 | VEC_CH) */
    int ROWS_PER_PASS; /* WAVE * LD_VEC / WAVE_K (== WO_BLOCK on w64) */
    int DY_PASSES; /* WO_BLOCK / ROWS_PER_PASS (1 | 2)            */
    int C_FRAG; /* f32 accumulator length per lane (4 | 8)     */
    const char* wmma_op_id; /* wave32 WMMA 16x16x32 atom op_id, else NULL */

    /* dY/X element type; dW stays fp32 (the split-K reduction is fp32 atomics). */
    const rocke_type_t* io_type; /* rocke_f16() or rocke_bf16() per p.dtype */
    int is_bf16; /* 1 when p.dtype == "bf16", 0 otherwise      */

    int STRIP_COLS; /* WO_BLOCK + KW - 1                          */
    int TR_N; /* WAVE_K (LDS row width)                     */
    int TR_K_L; /* WO_BLOCK / 4                               */
    int N_TR_READS; /* VEC_CH / 4                                 */
    int LDS_SIZE_DY; /* WO_BLOCK * ROW_STRIDE                      */
    int STRIP_PASSES;
    int STRIP_GROUPS;
    int STRIP_PASSES_PER_WAVE;
    int STRIP_COLS_PAD;
    int STRIP_PER_Q; /* STRIP_COLS_PAD * ROW_STRIDE                */
    int n_k_tiles;
    int n_c_tiles;

    /* ---- kernel params (Values): the AOT kernarg block, conv_abi order -- */
    rocke_dconv_params_t params;
    rocke_value_t* A; /* dY  ptr<f16, global> */
    rocke_value_t* Bp; /* X   ptr<f16, global> */
    rocke_value_t* D; /* dW  ptr<f32, global> */
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    /* D_bytes is declared to keep the shared 6-arg launch signature but the
     * Python discards the Value (dW is reached by plain global_atomic_add, not
     * a buffer resource), so it is not carried here either. */

    /* ---- common SSA constants -- */
    rocke_value_t* c0;
    rocke_value_t* c_wave;
    rocke_value_t* c_cpg;
    rocke_value_t* c_kpg;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* c_H; /* INPUT height  (p_Hi)    */
    rocke_value_t* c_Ho; /* output height (p_Ho)    */
    rocke_value_t* c_Wo; /* output width  (p_Wo)    */
    rocke_value_t* zero_acc; /* zero_vec_f32(C_FRAG)    */

    /* ---- thread / wave / lane decode -- */
    rocke_value_t* tid;
    rocke_value_t* wave_id;
    rocke_value_t* lane;
    rocke_value_t* c4; /* lane / 16 */
    rocke_value_t* q_in_lane; /* lane % 16 */

    /* ---- grid decode -- */
    rocke_value_t* bx;
    rocke_value_t* by;
    rocke_value_t* bz;
    rocke_value_t* c_n_k_tiles;
    rocke_value_t* c_n_c_tiles;
    rocke_value_t* c_n_wo_tiles; /* ceil(p_Wo / WO_BLOCK)       */
    rocke_value_t* c_n_q_blocks; /* ceil(n_wo_tiles / WAVES_Q)  */
    rocke_value_t* c_tile_idx;
    rocke_value_t* gk_flat;
    rocke_value_t* k_tile_in_group;
    rocke_value_t* group;
    rocke_value_t* n_i;
    rocke_value_t* q_block;
    rocke_value_t* hi_block_start;

    /* ---- wave decomposition (KCQ layout) -- */
    rocke_value_t* wave_q_id;
    rocke_value_t* wave_kc_id;
    rocke_value_t* wave_k_id;
    rocke_value_t* wave_c_id;
    rocke_value_t* wave_k_origin;
    rocke_value_t* wave_c_origin;
    rocke_value_t* wo_tile;
    rocke_value_t* wo_tile_start;
    rocke_value_t* wo_tile_valid;
    rocke_value_t* k_tile_origin;
    rocke_value_t* c_tile_origin;
    rocke_value_t* k_wave_base;

    /* ---- buffer rsrcs (dW: plain atomics on wave64, a resource on wave32) -- */
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc; /* wave32 only: dW buffer atomics, else NULL */

    /* ---- descriptors -- */
    const rocke_tensor_descriptor_t* dy_desc; /* A[N,Ho,Wo,total_k], runtime strides */
    const rocke_tensor_descriptor_t* x_strip_desc; /* B[N,H,W,total_c] + w embed (runtime) */
    const rocke_tensor_descriptor_t* dw_desc; /* D[total_k,KH,KW,cpg] naive    */

    /* ---- LDS tiles -- */
    rocke_value_t* dy_lds;
    rocke_value_t* s_strip_lds;

    /* ---- per-thread loader decomposition -- */
    rocke_value_t* c_lanes_per_sp;
    rocke_value_t* c_ld_sp;
    rocke_value_t* c_ld_ch;
    rocke_value_t* dy_part_idx;
    rocke_value_t* dy_wave_off_f16;
    rocke_value_t* s_strip_part_idx;
    rocke_value_t* s_strip_off_f16;
    rocke_value_t* c_TR_N;
    rocke_value_t* c_WO_BLOCK;
    rocke_value_t* c_STRIP_COLS;
    rocke_value_t* c_rows_per_pass; /* const(ROWS_PER_PASS) on w32, c_WO_BLOCK on w64 */
    rocke_value_t* tr_row;
    rocke_value_t* tr_flat;
    /* fp32 storage (fp32 / tf32): no 32-bit transpose read, so each lane
     * gathers its fragment with VEC_CH ds_read_b32 from rows padded to
     * F32_ROW elements (bank-conflict free), committed in F32_ST_VEC pieces. */
    const char* dtype; /* p.dtype                                   */
    int F32_IO; /* _storage_dtype(p.dtype) == "fp32"         */
    int F32_ROW; /* TR_N + (2 if VEC_CH == 8 else 4)          */
    int F32_ST_VEC; /* 2 if (F32_ROW * 4) % 16 else min(VEC_CH, 4) */
    int ROW_STRIDE; /* F32_ROW if F32_IO else TR_N               */
    rocke_value_t* c_ROW; /* const_i32(F32_ROW) or c_TR_N        */
    rocke_value_t* gather_flat; /* c4 * VEC_CH * F32_ROW + q_in_lane  */

    /* ---- S-strip pass decomposition -- */
    rocke_value_t* strip_pass_base;
    rocke_value_t* strip_cols[ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES];
    rocke_value_t* strip_col_ok[ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES];

    /* ---- iter-state carried across the unrolled row loop -- */
    rocke_value_t* acc[ROCKE_DCONV_WGRAD_MAX_KH][ROCKE_DCONV_WGRAD_MAX_KW];
    rocke_value_t* delta_ring[ROCKE_DCONV_WGRAD_MAX_KH];
    rocke_value_t* pending_dy[ROCKE_DCONV_WGRAD_MAX_DY_PASSES];
    rocke_value_t* pending_x[ROCKE_DCONV_WGRAD_MAX_STRIP_PASSES];
} rocke_dconv_wgrad_ctx_t;

/* ===================================================================== *
 *  wgrad PHASE FUNCTIONS
 * ===================================================================== */

/* Prologue: validate() + is_valid_wgrad_spec gate, every geometry scalar, the
 * params, the SSA constants, thread/wave/grid decode, the three descriptors,
 * the two LDS tiles, the loader decomposition and the S-strip pass columns.
 * Returns false (builder error set) on a rejected spec. */
bool rocke_dconv_wgrad_prologue(rocke_dconv_wgrad_ctx_t* ctx);

/* Ring prologue: pre-load the KH-1 past dY rows into ctx->delta_ring, and seed
 * ctx->acc / the remaining ring slot. */
void rocke_dconv_wgrad_ring_prologue(rocke_dconv_wgrad_ctx_t* ctx);

/* The Python-unrolled loop over the HPB input rows of this block: commit the
 * fragments issued last row, issue the next row's, sync, refresh the ring +
 * read the KW S fragments, then the KH*KW MFMAs. */
void rocke_dconv_wgrad_row_loop(rocke_dconv_wgrad_ctx_t* ctx);

/* Epilogue: KH*KW*4 guarded global_atomic_add into the fp32 dW. Returns the
 * kernel (ctx->b->kernel) on success, NULL on error. */
rocke_kernel_def_t* rocke_dconv_wgrad_epilogue(rocke_dconv_wgrad_ctx_t* ctx);

/* _build_direct_conv_wgrad_image(spec, arch): the image-per-block wave32
 * variant (spec->image_per_block). The caller has already run validate() and
 * is_valid_wgrad_spec. Returns the kernel, or NULL with the builder error set. */
rocke_kernel_def_t* rocke_dconv_wgrad_build_image(rocke_ir_builder_t* b,
                                                  const rocke_direct_conv_wgrad_spec_t* spec,
                                                  const char* arch);

/* _build_direct_conv_wgrad_small_c(spec, arch): the small-channel wave32
 * variant (spec->small_c). The caller has already run validate() and
 * is_valid_wgrad_spec. Returns the kernel, or NULL with the builder error set. */
rocke_kernel_def_t* rocke_dconv_wgrad_build_small_c(rocke_ir_builder_t* b,
                                                    const rocke_direct_conv_wgrad_spec_t* spec,
                                                    const char* arch);

/* ===================================================================== *
 *  Depthwise PHASE FUNCTIONS
 * ===================================================================== */
bool rocke_dconv_dw_prologue(rocke_dconv_dw_ctx_t* ctx);
void rocke_dconv_dw_load_weights(rocke_dconv_dw_ctx_t* ctx);
void rocke_dconv_dw_build_descriptors(rocke_dconv_dw_ctx_t* ctx);
rocke_kernel_def_t* rocke_dconv_dw_stream_h_loop(rocke_dconv_dw_ctx_t* ctx);

/* Channel-packed (cpl > 1) depthwise forward kernels
 * (conv_direct_grouped_build_depthwise_packed.cpp). The row-streaming one
 * validates its spec; the tiled one expects a spec its caller validated. */
bool rocke_dconv_depthwise_cpl_reason(
    int cpl, int groups, const char* dtype, char* reason, size_t reason_cap);
rocke_kernel_def_t* rocke_dconv_build_depthwise_packed(rocke_ir_builder_t* b,
                                                       const rocke_direct_depthwise_spec_t* spec,
                                                       const char* arch);
rocke_kernel_def_t* rocke_dconv_build_depthwise_tiled_packed(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_tiled_spec_t* spec, const char* arch);

/* ===================================================================== *
 *  Depthwise-column PHASE FUNCTIONS
 *
 *  There is no separate load_weights phase: the KH weights of one filter
 *  column depend on the runtime `s` induction variable, so they are loaded
 *  inside the column loop rather than in the prologue.
 * ===================================================================== */
bool rocke_dconv_dwcol_prologue(rocke_dconv_dwcol_ctx_t* ctx);
void rocke_dconv_dwcol_build_descriptors(rocke_dconv_dwcol_ctx_t* ctx);
rocke_kernel_def_t* rocke_dconv_dwcol_col_loop(rocke_dconv_dwcol_ctx_t* ctx);

/* ===================================================================== *
 *  Windowed depthwise helpers (conv_direct_grouped_build_depthwise_dgrad_win.cpp)
 *
 *  The spec helpers (passes, tile width, loads, LDS bytes, w_fold) are also
 *  read by the gfx1250 depthwise forward's spec (conv_direct_dw_fwd_wmma.cpp),
 *  which builds through the windowed Toeplitz builder:
 *
 *    Python (conv_direct_grouped.py)        C++ (here)
 *    ------------------------------------   ---------------------------------
 *    _dw_mfma_passes(KW)                    rocke_dw_mfma_passes
 *    _dw_mfma_tile_w(w_fold)                rocke_dw_mfma_tile_w
 *    _dw_mfma_in_passes(...)                rocke_dw_mfma_in_passes
 *    _dw_mfma_lds_bytes(...)                rocke_dw_mfma_lds_bytes
 *    DW_DGRAD_MFMA_W_FOLDS                  rocke_dw_mfma_w_fold_ok
 *    _dw_win_byte_strides(b, params, ...)   rocke_dw_win_byte_strides
 *    _dw_win_tile(b, p_Hi, rows)            rocke_dw_win_tile
 *    _buf_load_vN(b, dtype, ...)            rocke_dconv_buf_load_vN
 *    _buf_store_vN(b, dtype, ...)           rocke_dconv_buf_store_vN
 *    _trunc_f32(b, dtype, val)              rocke_dconv_trunc_f32
 *    _trunc_f32_elem(b, dtype, val)         rocke_dconv_trunc_f32_elem
 *    _buf_load_half(b, dtype, ...)          rocke_dconv_buf_load_half
 *    the one-trip guarded scf.for_iter      rocke_dw_win_guarded (C++ only)
 *
 *  `bf16` stands for Python's `dtype == "bf16"`. Every helper emits its ops in
 *  the Python order (byte identity).
 * ===================================================================== */

/* 16x16x32 passes per filter row: ceil((KW + 1) / 4). */
int rocke_dw_mfma_passes(int KW);

/* w_fold in DW_DGRAD_MFMA_W_FOLDS (1, 2 or 4). */
bool rocke_dw_mfma_w_fold_ok(int w_fold);

/* Output columns per image of a Toeplitz tile: its 32 columns split over
 * w_fold images (0 for a w_fold below 1, which the validators refuse). */
int rocke_dw_mfma_tile_w(int w_fold);

/* 16-byte loads per thread and window row of the Toeplitz form: a window of
 * tile_w + 4 * passes - 2 columns per image, block_ch / 8 loads per column,
 * w_fold images (0 without threads, which the validators refuse). */
int rocke_dw_mfma_in_passes(int tile_w, int KW, int w_fold, int block_ch, int threads);

/* LDS bytes of the Toeplitz form: double-buffered input window rows and
 * output rows, block_ch + 8 channels per pixel (0 for an empty block). */
long rocke_dw_mfma_lds_bytes(int in_passes, int threads, int block_ch, int w_fold, int tile_w);

/* Byte strides of the A-slot and D-slot tensors from the element-stride
 * kernargs: out = (a_img, a_row, a_px, d_img, d_row, d_px) (dY / dX in the
 * dgrad, X / Y in the forward). */
void rocke_dw_win_byte_strides(rocke_ir_builder_t* b,
                               const rocke_dconv_params_t* prm,
                               rocke_value_t* out[6]);

/* (image_tile, h0) of this block: block z is image_tile * h_tiles + h_tile
 * with h_tiles = ceil(p_Hi / rows). */
void rocke_dw_win_tile(rocke_ir_builder_t* b,
                       rocke_value_t* p_Hi,
                       int rows,
                       rocke_value_t** n_tile,
                       rocke_value_t** h0);

rocke_value_t* rocke_dconv_buf_load_half(rocke_ir_builder_t* b,
                                         bool bf16,
                                         rocke_value_t* rsrc,
                                         rocke_value_t* voff,
                                         rocke_value_t* soff);
rocke_value_t* rocke_dconv_buf_load_vN(rocke_ir_builder_t* b,
                                       bool bf16,
                                       rocke_value_t* rsrc,
                                       rocke_value_t* voff,
                                       rocke_value_t* soff,
                                       int dwords);
void rocke_dconv_buf_store_vN(rocke_ir_builder_t* b,
                              bool bf16,
                              rocke_value_t* rsrc,
                              rocke_value_t* voff,
                              rocke_value_t* soff,
                              rocke_value_t* val,
                              int dwords);
/* Truncate a vector of f32 accumulators to the output dtype. */
rocke_value_t* rocke_dconv_trunc_f32(rocke_ir_builder_t* b, bool bf16, rocke_value_t* val);
/* Truncate one f32 value to the output dtype. */
rocke_value_t* rocke_dconv_trunc_f32_elem(rocke_ir_builder_t* b, bool bf16, rocke_value_t* val);

/* ===================================================================== *
 *  The rest of the fp32 / TF32 storage helpers (conv_direct_grouped_common.cpp;
 *  the storage dtype, element size, atom plan, element load, accumulator
 *  store and logical atom are declared with the AOT plumbing above).
 *
 *  Problem dtype "fp32" is fp32 tensors with exact fp32 math; "tf32" is fp32
 *  tensors with TF32 MMA operands (native XF32 on gfx942, BF16x3 emulation on
 *  gfx950). Element counts, not dwords. See the Python twins of the same
 *  names in conv_direct_grouped.py.
 * ===================================================================== */

/* _io_type: the storage IR type (f16 / bf16 / f32); an unknown dtype sets the
 * builder error and returns NULL. */
const rocke_type_t* rocke_dconv_io_type(rocke_ir_builder_t* b, const char* dtype);
/* _mfma_dtype_flags: the "f32" / "tf32" kernel-name flags. */
void rocke_dconv_mfma_dtype_flags(const char* dtype, int* f32, int* tf32);

/* _store_elems: n contiguous elements already in the storage type (a copy,
 * no conversion); fp32 storage in dwordx4 (or narrower) chunks. */
void rocke_dconv_store_elems(rocke_ir_builder_t* b,
                             const char* dtype,
                             rocke_value_t* rsrc,
                             rocke_value_t* voff,
                             rocke_value_t* soff,
                             rocke_value_t* val,
                             int n);
/* _smem_load_elems / _smem_store_elems: LDS [row, idx], fp32 in <= 4-element
 * pieces. */
rocke_value_t* rocke_dconv_smem_load_elems(rocke_ir_builder_t* b,
                                           const char* dtype,
                                           rocke_value_t* lds,
                                           rocke_value_t* row,
                                           rocke_value_t* idx,
                                           int n);
void rocke_dconv_smem_store_elems(rocke_ir_builder_t* b,
                                  const char* dtype,
                                  rocke_value_t* lds,
                                  rocke_value_t* row,
                                  rocke_value_t* idx,
                                  rocke_value_t* val,
                                  int n);

/* _f32_mma_reason: true (and `why`) when no m x m atom serves `dtype`. */
bool rocke_dconv_f32_mma_reason(
    const char* arch, const char* dtype, int m, int frag, char* why, size_t why_cap);
/* _mma_f32_chain: several logical m x m MMAs into one accumulator; as[i] /
 * bs[i] are the chain's (a, b) fragments in order. Pairs consecutive
 * fragments into one atom where the fragment needs padding but twice it is
 * an exact atom. */
rocke_value_t* rocke_dconv_mma_f32_chain(rocke_ir_builder_t* b,
                                         const char* arch,
                                         const char* dtype,
                                         int m,
                                         rocke_value_t* const* as,
                                         rocke_value_t* const* bs,
                                         int n,
                                         rocke_value_t* acc);

/* ===================================================================== *
 *  Scalar-FMA element I/O (conv_direct_grouped_common.cpp): fp16 / bf16, or
 *  fp32 storage for "fp32" and "tf32" (both exact fp32 math, one binary).
 * ===================================================================== */

/* _CPL_16BIT_ONLY: the channel-packed (cpl > 1) depthwise kernels move 16-bit
 * channel pairs; fp32 / tf32 storage runs one channel per lane. */
#define ROCKE_DCONV_CPL_16BIT_ONLY "cpl > 1 packs 16-bit channel pairs; fp32 and tf32 need cpl=1"

/* _scalar_dtype_reason: true (and `why`) when `kernel` cannot take `dtype`. */
bool rocke_dconv_scalar_dtype_reason(const char* kernel,
                                     const char* dtype,
                                     char* why,
                                     size_t why_cap);
/* _scalar_name_flags(dtype)["f32"]: fp32 / tf32 storage. */
int rocke_dconv_scalar_f32_flag(const char* dtype);
/* _scalar_load: one stored element, unwidened. */
rocke_value_t* rocke_dconv_scalar_load(rocke_ir_builder_t* b,
                                       const char* dtype,
                                       rocke_value_t* rsrc,
                                       rocke_value_t* voff,
                                       rocke_value_t* soff);
/* _scalar_store: one f32 accumulator as the output element type. */
void rocke_dconv_scalar_store(rocke_ir_builder_t* b,
                              const char* dtype,
                              rocke_value_t* rsrc,
                              rocke_value_t* voff,
                              rocke_value_t* soff,
                              rocke_value_t* acc);

/* _dw_col_storage: the storage IR type of a column-kernel dtype; "tf32" (any
 * case) resolves to f32, anything else through dtype_to_ir (NULL when it
 * cannot). */
const rocke_type_t* rocke_dconv_dw_col_storage(const char* dtype);

/* One staged pixel chunk of a window row: its LDS column `px`, channel
 * offset `ch8`, and the guarded global byte offset `g_off`. */
typedef struct rocke_dw_win_plan
{
    rocke_value_t* px;
    rocke_value_t* ch8;
    rocke_value_t* g_off;
} rocke_dw_win_plan_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus
#include <string>
#include <vector>

/* A one-trip (or zero-trip) scf.for_iter guarding `body` on the block-uniform
 * `ok`, carrying `accs` (updated in place to the loop results). The iter args
 * are named `<prefix>_y<y>_i<i>`, the induction variable `<prefix>_y<y>`.
 * `body` maps the carried values to the yielded ones; it is a template
 * parameter so each caller's lambda is called directly. */
template <typename Body>
inline void rocke_dw_win_guarded(rocke_ir_builder_t* b,
                                 rocke_value_t* ok,
                                 rocke_value_t* c0,
                                 rocke_value_t* c1,
                                 std::vector<rocke_value_t*>& accs,
                                 const char* prefix,
                                 int y,
                                 bool elide_trailing_barrier,
                                 Body body)
{
    rocke_value_t* trip = rocke_b_select(b, ok, c1, c0);
    std::vector<std::string> names(accs.size());
    std::vector<rocke_iter_arg_t> iargs(accs.size());
    for(size_t i = 0; i < accs.size(); ++i)
    {
        names[i] = std::string(prefix) + "_y" + std::to_string(y) + "_i" + std::to_string(i);
        iargs[i].name = names[i].c_str();
        iargs[i].init = accs[i];
    }
    const std::string iv = std::string(prefix) + "_y" + std::to_string(y);
    rocke_for_t loop = rocke_b_scf_for_iter(b,
                                            c0,
                                            trip,
                                            c1,
                                            iargs.data(),
                                            (int)iargs.size(),
                                            iv.c_str(),
                                            /*unroll=*/false,
                                            elide_trailing_barrier);
    if(!rocke_ir_builder_ok(b))
        return;
    rocke_b_region_enter(b, loop.body);
    std::vector<rocke_value_t*> carried(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
    std::vector<rocke_value_t*> out = body(carried);
    rocke_b_scf_yield(b, out.data(), (int)out.size());
    rocke_b_region_leave(b);
    for(size_t i = 0; i < accs.size(); ++i)
        accs[i] = loop.op->results[i];
}
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_GROUPED_INTERNAL_H */
