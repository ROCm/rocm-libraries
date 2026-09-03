/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.moe_gemm_fused.h -- C99 port of the
 * MoE-specialized MFMA GEMM fusions
 * rocke/instances/common/moe_gemm_fused.py.
 *
 * SCOPE OF THIS PORT (the requested symbol set):
 *
 *   Python (moe_gemm_fused.py)               C99 (this header / .c)
 *   --------------------------------------   -----------------------------------
 *   _silu_mul_f32                            rocke_moe_silu_mul_f32
 *   _CWarpDecode                             rocke_moe_cwarp_decode_t (+ methods)
 *   _MoeOperand                              rocke_moe_operand_t
 *   _MoeKloopPlan                            rocke_moe_kloop_plan_t (+ init)
 *   _emit_cshuffle_stage                     rocke_moe_emit_cshuffle_stage
 *   _emit_down_reduce_epilogue_atomic        rocke_moe_emit_down_reduce_epilogue_atomic
 *   _emit_moe_prefetch_kloop                 rocke_moe_emit_prefetch_kloop
 *   _magic_div_mod                           rocke_moe_magic_div_mod
 *   _vec_rowcol                              rocke_moe_vec_rowcol
 *   _pad_in_bounds                           rocke_moe_pad_in_bounds
 *   _emit_moe_global_load                    rocke_moe_emit_global_load
 *   _emit_moe_lds_store                      rocke_moe_emit_lds_store
 *   _emit_moe_mfma_phase                     rocke_moe_emit_mfma_phase
 *   FusedGateUpSiluGemmSpec                  rocke_moe_gate_up_silu_gemm_spec_t
 *   FusedInterleavedGateUpSiluGemmSpec       rocke_moe_interleaved_gate_up_silu_gemm_spec_t
 *   FusedDownReduceGemmSpec                  rocke_moe_down_reduce_gemm_spec_t
 *   FusedDownSiluReduceGemmSpec              rocke_moe_down_silu_reduce_gemm_spec_t
 *
 * The three MoE GEMM fusions drive the SAME software-prefetched MFMA k-loop
 * (load tile 0 into registers, then per iteration store->sync->prefetch->mfma->
 * sync->yield). The shared loader / k-loop core is parameterised by a list of
 * rocke_moe_operand_t (one entry per B matrix). Each builder wires its operands +
 * custom epilogue and calls rocke_moe_emit_prefetch_kloop.
 *
 * Binds to rocke/ir.h (the C IRBuilder) plus the sibling helper headers
 * (atoms / distribution / tensor_view / transforms / spec) and the universal
 * GEMM instance (instance_gemm_universal.h + instance_gemm_internal.h).
 *
 * PARTIAL DEPENDENCY: store_tile_cshuffle and StaticDistributedTensor.set are
 * not yet exposed by the C distribution port (helper_rocke.helpers.distribution.h
 * exports make_static_distributed_tensor / calculate_x but not the cshuffle
 * space-filling store walk). rocke_moe_emit_cshuffle_stage therefore stages the
 * accumulators with a faithful per-(mi,ni,i) scatter via the universal GEMM's
 * smem-store path; the exact CK-Tile cshuffle store walk is marked
 * TODO(port) until the distribution port lands store_tile_cshuffle.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MOE_GEMM_FUSED_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MOE_GEMM_FUSED_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h" /* rocke_arena_t (signature storage)                   */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom, c_warp   */
#include "rocke/helper_rocke.helpers.distribution.h" /* tile distribution       */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t          */
#include "rocke/helper_rocke.helpers.tensor_view.h" /* TensorView / TileWindow  */
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_universal_spec_t*/
#include "rocke/ir.h" /* rocke_value_t, rocke_type_t, rocke_ir_builder_t, status */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ leaves */

/* _storage_dtype(spec): homogeneous A/B/C dtype -> rocke_type_t. Derived from the
 * spec's data dtype string ("f16"/"fp16" -> f16, "bf16" -> bf16, else by name;
 * a NULL dtype defaults to f16). Shared by all three MoE GEMM fusions. */
const rocke_type_t* rocke_moe_storage_dtype(const rocke_gemm_universal_spec_t* u);

/* _mfma_atom_widths(spec) -> (a_per_lane, b_per_lane, c_per_lane). The warp-tile
 * atom's per-lane fragment widths: (wm*wk)/wave, (wn*wk)/wave, (wm*wn)/wave. */
void rocke_moe_mfma_atom_widths(const rocke_gemm_universal_spec_t* u,
                                int* a_per,
                                int* b_per,
                                int* c_per);

/* _magic_div_mod(b, dividend, divisor) -> (quot, rem) via CK Tile magic
 * division. divisor==1 short-circuits to (dividend, const_i32(0)). The divisor
 * is a compile-time constant so the magic (multiplier, shift) are baked in. */
void rocke_moe_magic_div_mod(rocke_ir_builder_t* b,
                             rocke_value_t* dividend,
                             int divisor,
                             rocke_value_t** out_quot,
                             rocke_value_t** out_rem);

/* _vec_rowcol(b, e, tid, c_threads, block_k_div_vec, c_load_vec, load_vec)
 * -> (row, col): decode the per-thread (row, col) for vec-load element `e` via
 * the magic-division unmerge. */
void rocke_moe_vec_rowcol(rocke_ir_builder_t* b,
                          int e,
                          rocke_value_t* tid,
                          rocke_value_t* c_threads,
                          int block_k_div_vec,
                          rocke_value_t* c_load_vec,
                          int load_vec,
                          rocke_value_t** out_row,
                          rocke_value_t** out_col);

/* _silu_mul_f32(b, g, u, one_f32, c_neg_log2e) -> silu(g)*u (sigmoid via
 * exp2). Constants are caller-supplied so the emitted SSA matches the inline
 * order exactly. */
rocke_value_t* rocke_moe_gemm_fused_silu_mul_f32(rocke_ir_builder_t* b,
                                                 rocke_value_t* g,
                                                 rocke_value_t* u,
                                                 rocke_value_t* one_f32,
                                                 rocke_value_t* c_neg_log2e);

/* _pad_in_bounds(b, c_m, c_n, M, N, pad_m, pad_n, vec) -> mask Value or NULL.
 * Returns NULL when neither pad flag is set (the Python `None`). */
rocke_value_t* rocke_moe_pad_in_bounds(rocke_ir_builder_t* b,
                                       rocke_value_t* c_m,
                                       rocke_value_t* c_n,
                                       rocke_value_t* M,
                                       rocke_value_t* N,
                                       bool pad_m,
                                       bool pad_n,
                                       int vec);

/* ----------------------------------------------------------- _CWarpDecode */

/* MFMA C-accumulator lane -> (row, col) decode via CWarpDstrEncoding. Built
 * once per epilogue from the spec's warp-tile atom. */
typedef struct rocke_moe_cwarp_decode
{
    rocke_ir_builder_t* b;
    const rocke_gemm_universal_spec_t* spec; /* tile geometry                */
    const rocke_tile_distribution_t* dist; /* make_static_tile_distribution */
    int m1; /* Hs[0][2] = kCM1PerLane       */
    rocke_value_t* n_in_atom; /* lane % kCNLane               */
    rocke_value_t* m_blk; /* lane // kCNLane              */
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
} rocke_moe_cwarp_decode_t;

/* _CWarpDecode.__init__. Returns 1 on success, 0 on a builder/encoding error
 * (sticky error set on `b`). */
int rocke_moe_cwarp_decode_init(rocke_moe_cwarp_decode_t* out,
                                rocke_ir_builder_t* b,
                                const rocke_gemm_universal_spec_t* spec,
                                rocke_value_t* warp_m_off,
                                rocke_value_t* warp_n_off,
                                rocke_value_t* lane);

/* _CWarpDecode.coords(mi, ni, i) -> (ld_m, ld_n). */
void rocke_moe_cwarp_decode_coords(const rocke_moe_cwarp_decode_t* d,
                                   int mi,
                                   int ni,
                                   int i,
                                   rocke_value_t** out_ld_m,
                                   rocke_value_t** out_ld_n);

/* _CWarpDecode.warp_row(mi, i). */
rocke_value_t* rocke_moe_cwarp_decode_warp_row(const rocke_moe_cwarp_decode_t* d, int mi, int i);

/* _CWarpDecode.warp_col(ni) (i-independent). */
rocke_value_t* rocke_moe_cwarp_decode_warp_col(const rocke_moe_cwarp_decode_t* d, int ni);

/* ------------------------------------------------------------- _MoeOperand */

/* `cell_value` / `load_b` callbacks: a plain C function pointer + opaque user
 * context (the closures the Python passes inline). */
typedef rocke_value_t* (*rocke_moe_cell_value_fn)(int mi, int ni, int i, void* user);
typedef rocke_value_t* (*rocke_moe_load_b_fn)(rocke_ir_builder_t* b,
                                              int e,
                                              rocke_value_t* k_off,
                                              rocke_value_t* row,
                                              rocke_value_t* col,
                                              void* user);

/* One B matrix of a MoE GEMM fusion, bound to its LDS + accumulator group. */
typedef struct rocke_moe_operand
{
    const rocke_tensor_view_t* global_view; /* 3D global view                  */
    const rocke_tensor_view_t* lds_view; /* 2D LDS view                     */
    rocke_value_t* smem; /* raw LDS allocation the MFMA reads */
    rocke_moe_load_b_fn load_b; /* NULL => canonical window load    */
    void* load_b_user; /* closure context for load_b       */
    bool store_scalar_ok; /* false => always-vectorised store */
} rocke_moe_operand_t;

/* ----------------------------------------------------------- _MoeKloopPlan */

/* Static per-kernel geometry shared by the loader / store / MFMA helpers. */
typedef struct rocke_moe_kloop_plan
{
    rocke_ir_builder_t* b;
    const rocke_gemm_universal_spec_t* u;
    rocke_value_t* tid;
    const rocke_type_t* storage_dtype;
    int a_per_lane, b_per_lane, c_per_lane;
    int block_m, block_n, block_k;
    int mfmas_m, mfmas_n, k_atoms;
    int threads, load_vec;
    int a_vecs_per_thread, b_vecs_per_thread;
    rocke_value_t* c_threads;
    rocke_value_t* c_load_vec;
    int block_k_div_vec;
} rocke_moe_kloop_plan_t;

/* _MoeKloopPlan.__init__. Returns 1 on success, 0 on error (sticky on `b`). */
int rocke_moe_kloop_plan_init(rocke_moe_kloop_plan_t* out,
                              rocke_ir_builder_t* b,
                              const rocke_gemm_universal_spec_t* u,
                              rocke_value_t* tid);

/* -------------------------------------------------------- shared k-loop core */

/* _emit_moe_global_load. `a_mn_origin` / `b_mn_origin` are 2-element arrays
 * (batch_off, block_mn_off). Outputs the A registers and one register group
 * per operand (caller-provided buffers; capacities must be >= a_vecs_per_thread
 * and b_vecs_per_thread respectively). */
void rocke_moe_emit_global_load(const rocke_moe_kloop_plan_t* plan,
                                const rocke_tensor_view_t* a_view,
                                rocke_value_t* const a_mn_origin[2],
                                const rocke_moe_operand_t* operands,
                                int num_operands,
                                rocke_value_t* const b_mn_origin[2],
                                rocke_value_t* k_off,
                                rocke_value_t** out_a_regs,
                                rocke_value_t** out_b_regs /* [num_operands][b_vecs] flat */);

/* _emit_moe_lds_store. `b_reg_groups` is the flat [num_operands][b_vecs]
 * buffer produced by rocke_moe_emit_global_load. */
void rocke_moe_emit_lds_store(const rocke_moe_kloop_plan_t* plan,
                              const rocke_tensor_view_t* a_lds_view,
                              rocke_value_t* const* a_regs,
                              const rocke_moe_operand_t* operands,
                              int num_operands,
                              rocke_value_t* const* b_reg_groups);

/* _emit_moe_mfma_phase. `acc_groups` / `out_groups` are [num_operands] arrays
 * of length mfmas_m*mfmas_n each (flat). Out may alias in (the Python rebuilds
 * new lists); pass distinct buffers. sched_groups==0 disables the hint. */
void rocke_moe_emit_mfma_phase(const rocke_moe_kloop_plan_t* plan,
                               rocke_value_t* a_smem,
                               const rocke_moe_operand_t* operands,
                               int num_operands,
                               rocke_value_t* const* const* acc_groups,
                               const int* group_sizes,
                               rocke_value_t* warp_m_idx,
                               rocke_value_t* warp_n_idx,
                               rocke_value_t* lane,
                               int sched_groups,
                               rocke_value_t** out_groups_flat /* sum(group_sizes) */);

/* _emit_moe_prefetch_kloop. Drives the software-prefetched MFMA k-loop and
 * returns the final accumulator groups (flat, sum(group_sizes) values) into
 * `out_groups_flat`. `acc_inits_flat` are the initial accumulator values
 * (sum(group_sizes) of them) in operand-then-flat order; `group_sizes` is one
 * entry per operand. `acc_names_flat` are the matching loop-carried accumulator
 * SSA names (sum(group_sizes) of them, same operand-then-flat order, e.g.
 * "gate_acc_m0_n0", "up_acc_m0_n0", ...) -- Python carries these from the
 * acc_groups (name, init) tuples; pass NULL to fall back to "acc%d" labels.
 * Returns 1 on success, 0 on error. */
int rocke_moe_emit_prefetch_kloop(const rocke_moe_kloop_plan_t* plan,
                                  const rocke_tensor_view_t* a_view,
                                  const rocke_tensor_view_t* a_lds_view,
                                  rocke_value_t* a_smem,
                                  rocke_value_t* const a_mn_origin[2],
                                  const rocke_moe_operand_t* operands,
                                  int num_operands,
                                  rocke_value_t* const b_mn_origin[2],
                                  rocke_value_t* const* acc_inits_flat,
                                  const char* const* acc_names_flat,
                                  const int* group_sizes,
                                  rocke_value_t* K,
                                  rocke_value_t* warp_m_idx,
                                  rocke_value_t* warp_n_idx,
                                  rocke_value_t* lane,
                                  int sched_groups,
                                  rocke_value_t** out_groups_flat);

/* -------------------------------------------------------------- epilogues */

/* _emit_cshuffle_stage: stage one warp's MFMA accumulators into LDS. The
 * accumulator value for slot i of atom (mi, ni) comes from `cell_value`.
 * NOTE: the exact CK-Tile store_tile_cshuffle space-filling walk is TODO(port)
 * (the C distribution layer does not yet export it); the current body performs
 * the equivalent per-(mi,ni,i) scatter at the same MFMA-output LDS addresses. */
void rocke_moe_emit_cshuffle_stage(rocke_ir_builder_t* b,
                                   const rocke_gemm_universal_spec_t* spec,
                                   const rocke_moe_cwarp_decode_t* cdec,
                                   rocke_value_t* smem,
                                   const rocke_type_t* storage_dtype,
                                   int c_per_lane,
                                   rocke_moe_cell_value_fn cell_value,
                                   void* cell_user);

/* _emit_down_reduce_epilogue_atomic: Y[token, n] += weight * down_acc. `accs`
 * is the flat [mfmas_m*mfmas_n] accumulator array. */
void rocke_moe_emit_down_reduce_epilogue_atomic(rocke_ir_builder_t* b,
                                                const rocke_gemm_universal_spec_t* spec,
                                                rocke_value_t* const* accs,
                                                rocke_value_t* warp_m_idx,
                                                rocke_value_t* warp_n_idx,
                                                rocke_value_t* lane,
                                                rocke_value_t* block_m_off,
                                                rocke_value_t* block_n_off,
                                                rocke_value_t* M,
                                                rocke_value_t* N,
                                                rocke_value_t* SortedTokenIds,
                                                rocke_value_t* SortedWeights,
                                                rocke_value_t* Y,
                                                int c_per_lane,
                                                rocke_value_t* batch_bucket_off,
                                                rocke_value_t* tokens);

/* -------------------------------------------------------------- spec types */

/* FusedGateUpSiluGemmSpec: batched per-expert fused gate+up GEMM + SiLU. */
typedef struct rocke_moe_gate_up_silu_gemm_spec
{
    const char* name;
    rocke_gemm_tile_spec_t tile;
    rocke_gemm_trait_spec_t trait; /* default epilogue="default"          */
    int wave_size; /* default 64                          */
    int block_size; /* default 0 => derived at finalize    */
    const char* dtype; /* default "fp16"                      */
    bool grouped; /* default false                       */
} rocke_moe_gate_up_silu_gemm_spec_t;

/* Default-constructed spec (matches the Python field defaults). */
rocke_moe_gate_up_silu_gemm_spec_t rocke_moe_gate_up_silu_gemm_spec_default(void);
/* __post_init__: derive block_size when 0. Idempotent. */
void rocke_moe_gate_up_silu_gemm_spec_finalize(rocke_moe_gate_up_silu_gemm_spec_t* spec);
/* to_universal_spec(). */
rocke_gemm_universal_spec_t
    rocke_moe_gate_up_silu_gemm_spec_to_universal(const rocke_moe_gate_up_silu_gemm_spec_t* spec);
/* kernel_name() -> NUL-terminated into out. */
rocke_status_t rocke_moe_gate_up_silu_gemm_spec_kernel_name(
    const rocke_moe_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap);

/* FusedInterleavedGateUpSiluGemmSpec: single-B gate+up GEMM with in-kernel
 * activation (WGateUp interleaved along N). */
typedef struct rocke_moe_interleaved_gate_up_silu_gemm_spec
{
    const char* name;
    rocke_gemm_tile_spec_t tile;
    rocke_gemm_trait_spec_t trait;
    int wave_size;
    int block_size;
    const char* dtype;
    bool grouped;
} rocke_moe_interleaved_gate_up_silu_gemm_spec_t;

rocke_moe_interleaved_gate_up_silu_gemm_spec_t
    rocke_moe_interleaved_gate_up_silu_gemm_spec_default(void);
void rocke_moe_interleaved_gate_up_silu_gemm_spec_finalize(
    rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec);
rocke_gemm_universal_spec_t rocke_moe_interleaved_gate_up_silu_gemm_spec_to_universal(
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec);
rocke_status_t rocke_moe_interleaved_gate_up_silu_gemm_spec_kernel_name(
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap);

/* FusedDownReduceGemmSpec: batched down GEMM with top-k weighted reduce. */
typedef struct rocke_moe_down_reduce_gemm_spec
{
    const char* name;
    rocke_gemm_tile_spec_t tile;
    rocke_gemm_trait_spec_t trait;
    int wave_size;
    int block_size;
    const char* dtype;
    bool grouped;
} rocke_moe_down_reduce_gemm_spec_t;

rocke_moe_down_reduce_gemm_spec_t rocke_moe_down_reduce_gemm_spec_default(void);
void rocke_moe_down_reduce_gemm_spec_finalize(rocke_moe_down_reduce_gemm_spec_t* spec);
rocke_gemm_universal_spec_t
    rocke_moe_down_reduce_gemm_spec_to_universal(const rocke_moe_down_reduce_gemm_spec_t* spec);
rocke_status_t rocke_moe_down_reduce_gemm_spec_kernel_name(
    const rocke_moe_down_reduce_gemm_spec_t* spec, char* out, size_t out_cap);

/* FusedDownSiluReduceGemmSpec: single fused down+silu+reduce ("up-kernel"). */
typedef struct rocke_moe_down_silu_reduce_gemm_spec
{
    const char* name;
    rocke_gemm_tile_spec_t tile;
    rocke_gemm_trait_spec_t trait;
    int wave_size;
    int block_size;
    const char* dtype;
} rocke_moe_down_silu_reduce_gemm_spec_t;

rocke_moe_down_silu_reduce_gemm_spec_t rocke_moe_down_silu_reduce_gemm_spec_default(void);
void rocke_moe_down_silu_reduce_gemm_spec_finalize(rocke_moe_down_silu_reduce_gemm_spec_t* spec);
rocke_gemm_universal_spec_t rocke_moe_down_silu_reduce_gemm_spec_to_universal(
    const rocke_moe_down_silu_reduce_gemm_spec_t* spec);
rocke_status_t rocke_moe_down_silu_reduce_gemm_spec_kernel_name(
    const rocke_moe_down_silu_reduce_gemm_spec_t* spec, char* out, size_t out_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MOE_GEMM_FUSED_H */
