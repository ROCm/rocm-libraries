/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.instances.common.moe_gemm_fused.h -- C99 port of the
 * MoE-specialized MFMA GEMM fusions
 * ck_dsl/instances/common/moe_gemm_fused.py.
 *
 * SCOPE OF THIS PORT (the requested symbol set):
 *
 *   Python (moe_gemm_fused.py)               C99 (this header / .c)
 *   --------------------------------------   -----------------------------------
 *   _silu_mul_f32                            ckc_moe_silu_mul_f32
 *   _CWarpDecode                             ckc_moe_cwarp_decode_t (+ methods)
 *   _MoeOperand                              ckc_moe_operand_t
 *   _MoeKloopPlan                            ckc_moe_kloop_plan_t (+ init)
 *   _emit_cshuffle_stage                     ckc_moe_emit_cshuffle_stage
 *   _emit_down_reduce_epilogue_atomic        ckc_moe_emit_down_reduce_epilogue_atomic
 *   _emit_moe_prefetch_kloop                 ckc_moe_emit_prefetch_kloop
 *   _magic_div_mod                           ckc_moe_magic_div_mod
 *   _vec_rowcol                              ckc_moe_vec_rowcol
 *   _pad_in_bounds                           ckc_moe_pad_in_bounds
 *   _emit_moe_global_load                    ckc_moe_emit_global_load
 *   _emit_moe_lds_store                      ckc_moe_emit_lds_store
 *   _emit_moe_mfma_phase                     ckc_moe_emit_mfma_phase
 *   FusedGateUpSiluGemmSpec                  ckc_moe_gate_up_silu_gemm_spec_t
 *   FusedInterleavedGateUpSiluGemmSpec       ckc_moe_interleaved_gate_up_silu_gemm_spec_t
 *   FusedDownReduceGemmSpec                  ckc_moe_down_reduce_gemm_spec_t
 *   FusedDownSiluReduceGemmSpec              ckc_moe_down_silu_reduce_gemm_spec_t
 *
 * The three MoE GEMM fusions drive the SAME software-prefetched MFMA k-loop
 * (load tile 0 into registers, then per iteration store->sync->prefetch->mfma->
 * sync->yield). The shared loader / k-loop core is parameterised by a list of
 * ckc_moe_operand_t (one entry per B matrix). Each builder wires its operands +
 * custom epilogue and calls ckc_moe_emit_prefetch_kloop.
 *
 * Binds to ckc/ir.h (the C IRBuilder) plus the sibling helper headers
 * (atoms / distribution / tensor_view / transforms / spec) and the universal
 * GEMM instance (instance_gemm_universal.h + instance_gemm_internal.h).
 *
 * PARTIAL DEPENDENCY: store_tile_cshuffle and StaticDistributedTensor.set are
 * not yet exposed by the C distribution port (helper_ck_dsl.helpers.distribution.h
 * exports make_static_distributed_tensor / calculate_x but not the cshuffle
 * space-filling store walk). ckc_moe_emit_cshuffle_stage therefore stages the
 * accumulators with a faithful per-(mi,ni,i) scatter via the universal GEMM's
 * smem-store path; the exact CK-Tile cshuffle store walk is marked
 * TODO(port) until the distribution port lands store_tile_cshuffle.
 */
#ifndef CKC_HELPER_CK_DSL_INSTANCES_COMMON_MOE_GEMM_FUSED_H
#define CKC_HELPER_CK_DSL_INSTANCES_COMMON_MOE_GEMM_FUSED_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h" /* ckc_arena_t (signature storage)                   */
#include "ckc/helper_ck_dsl.helpers.atoms.h" /* ckc_mfma_atom, c_warp   */
#include "ckc/helper_ck_dsl.helpers.distribution.h" /* tile distribution       */
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t          */
#include "ckc/helper_ck_dsl.helpers.tensor_view.h" /* TensorView / TileWindow  */
#include "ckc/instance_gemm_universal.h" /* ckc_gemm_universal_spec_t*/
#include "ckc/ir.h" /* ckc_value_t, ckc_type_t, ckc_ir_builder_t, status */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ leaves */

/* _storage_dtype(spec): homogeneous A/B/C dtype -> ckc_type_t. Derived from the
 * spec's data dtype string ("f16"/"fp16" -> f16, "bf16" -> bf16, else by name;
 * a NULL dtype defaults to f16). Shared by all three MoE GEMM fusions. */
const ckc_type_t* ckc_moe_storage_dtype(const ckc_gemm_universal_spec_t* u);

/* _mfma_atom_widths(spec) -> (a_per_lane, b_per_lane, c_per_lane). The warp-tile
 * atom's per-lane fragment widths: (wm*wk)/wave, (wn*wk)/wave, (wm*wn)/wave. */
void ckc_moe_mfma_atom_widths(const ckc_gemm_universal_spec_t* u,
                              int* a_per,
                              int* b_per,
                              int* c_per);

/* _magic_div_mod(b, dividend, divisor) -> (quot, rem) via CK Tile magic
 * division. divisor==1 short-circuits to (dividend, const_i32(0)). The divisor
 * is a compile-time constant so the magic (multiplier, shift) are baked in. */
void ckc_moe_magic_div_mod(ckc_ir_builder_t* b,
                           ckc_value_t* dividend,
                           int divisor,
                           ckc_value_t** out_quot,
                           ckc_value_t** out_rem);

/* _vec_rowcol(b, e, tid, c_threads, block_k_div_vec, c_load_vec, load_vec)
 * -> (row, col): decode the per-thread (row, col) for vec-load element `e` via
 * the magic-division unmerge. */
void ckc_moe_vec_rowcol(ckc_ir_builder_t* b,
                        int e,
                        ckc_value_t* tid,
                        ckc_value_t* c_threads,
                        int block_k_div_vec,
                        ckc_value_t* c_load_vec,
                        int load_vec,
                        ckc_value_t** out_row,
                        ckc_value_t** out_col);

/* _silu_mul_f32(b, g, u, one_f32, c_neg_log2e) -> silu(g)*u (sigmoid via
 * exp2). Constants are caller-supplied so the emitted SSA matches the inline
 * order exactly. */
ckc_value_t* ckc_moe_gemm_fused_silu_mul_f32(ckc_ir_builder_t* b,
                                             ckc_value_t* g,
                                             ckc_value_t* u,
                                             ckc_value_t* one_f32,
                                             ckc_value_t* c_neg_log2e);

/* _pad_in_bounds(b, c_m, c_n, M, N, pad_m, pad_n, vec) -> mask Value or NULL.
 * Returns NULL when neither pad flag is set (the Python `None`). */
ckc_value_t* ckc_moe_pad_in_bounds(ckc_ir_builder_t* b,
                                   ckc_value_t* c_m,
                                   ckc_value_t* c_n,
                                   ckc_value_t* M,
                                   ckc_value_t* N,
                                   bool pad_m,
                                   bool pad_n,
                                   int vec);

/* ----------------------------------------------------------- _CWarpDecode */

/* MFMA C-accumulator lane -> (row, col) decode via CWarpDstrEncoding. Built
 * once per epilogue from the spec's warp-tile atom. */
typedef struct ckc_moe_cwarp_decode
{
    ckc_ir_builder_t* b;
    const ckc_gemm_universal_spec_t* spec; /* tile geometry                */
    const ckc_tile_distribution_t* dist; /* make_static_tile_distribution */
    int m1; /* Hs[0][2] = kCM1PerLane       */
    ckc_value_t* n_in_atom; /* lane % kCNLane               */
    ckc_value_t* m_blk; /* lane // kCNLane              */
    ckc_value_t* warp_m_off;
    ckc_value_t* warp_n_off;
} ckc_moe_cwarp_decode_t;

/* _CWarpDecode.__init__. Returns 1 on success, 0 on a builder/encoding error
 * (sticky error set on `b`). */
int ckc_moe_cwarp_decode_init(ckc_moe_cwarp_decode_t* out,
                              ckc_ir_builder_t* b,
                              const ckc_gemm_universal_spec_t* spec,
                              ckc_value_t* warp_m_off,
                              ckc_value_t* warp_n_off,
                              ckc_value_t* lane);

/* _CWarpDecode.coords(mi, ni, i) -> (ld_m, ld_n). */
void ckc_moe_cwarp_decode_coords(const ckc_moe_cwarp_decode_t* d,
                                 int mi,
                                 int ni,
                                 int i,
                                 ckc_value_t** out_ld_m,
                                 ckc_value_t** out_ld_n);

/* _CWarpDecode.warp_row(mi, i). */
ckc_value_t* ckc_moe_cwarp_decode_warp_row(const ckc_moe_cwarp_decode_t* d, int mi, int i);

/* _CWarpDecode.warp_col(ni) (i-independent). */
ckc_value_t* ckc_moe_cwarp_decode_warp_col(const ckc_moe_cwarp_decode_t* d, int ni);

/* ------------------------------------------------------------- _MoeOperand */

/* `cell_value` / `load_b` callbacks: a plain C function pointer + opaque user
 * context (the closures the Python passes inline). */
typedef ckc_value_t* (*ckc_moe_cell_value_fn)(int mi, int ni, int i, void* user);
typedef ckc_value_t* (*ckc_moe_load_b_fn)(
    ckc_ir_builder_t* b, int e, ckc_value_t* k_off, ckc_value_t* row, ckc_value_t* col, void* user);

/* One B matrix of a MoE GEMM fusion, bound to its LDS + accumulator group. */
typedef struct ckc_moe_operand
{
    const ckc_tensor_view_t* global_view; /* 3D global view                  */
    const ckc_tensor_view_t* lds_view; /* 2D LDS view                     */
    ckc_value_t* smem; /* raw LDS allocation the MFMA reads */
    ckc_moe_load_b_fn load_b; /* NULL => canonical window load    */
    void* load_b_user; /* closure context for load_b       */
    bool store_scalar_ok; /* false => always-vectorised store */
} ckc_moe_operand_t;

/* ----------------------------------------------------------- _MoeKloopPlan */

/* Static per-kernel geometry shared by the loader / store / MFMA helpers. */
typedef struct ckc_moe_kloop_plan
{
    ckc_ir_builder_t* b;
    const ckc_gemm_universal_spec_t* u;
    ckc_value_t* tid;
    const ckc_type_t* storage_dtype;
    int a_per_lane, b_per_lane, c_per_lane;
    int block_m, block_n, block_k;
    int mfmas_m, mfmas_n, k_atoms;
    int threads, load_vec;
    int a_vecs_per_thread, b_vecs_per_thread;
    ckc_value_t* c_threads;
    ckc_value_t* c_load_vec;
    int block_k_div_vec;
} ckc_moe_kloop_plan_t;

/* _MoeKloopPlan.__init__. Returns 1 on success, 0 on error (sticky on `b`). */
int ckc_moe_kloop_plan_init(ckc_moe_kloop_plan_t* out,
                            ckc_ir_builder_t* b,
                            const ckc_gemm_universal_spec_t* u,
                            ckc_value_t* tid);

/* -------------------------------------------------------- shared k-loop core */

/* _emit_moe_global_load. `a_mn_origin` / `b_mn_origin` are 2-element arrays
 * (batch_off, block_mn_off). Outputs the A registers and one register group
 * per operand (caller-provided buffers; capacities must be >= a_vecs_per_thread
 * and b_vecs_per_thread respectively). */
void ckc_moe_emit_global_load(const ckc_moe_kloop_plan_t* plan,
                              const ckc_tensor_view_t* a_view,
                              ckc_value_t* const a_mn_origin[2],
                              const ckc_moe_operand_t* operands,
                              int num_operands,
                              ckc_value_t* const b_mn_origin[2],
                              ckc_value_t* k_off,
                              ckc_value_t** out_a_regs,
                              ckc_value_t** out_b_regs /* [num_operands][b_vecs] flat */);

/* _emit_moe_lds_store. `b_reg_groups` is the flat [num_operands][b_vecs]
 * buffer produced by ckc_moe_emit_global_load. */
void ckc_moe_emit_lds_store(const ckc_moe_kloop_plan_t* plan,
                            const ckc_tensor_view_t* a_lds_view,
                            ckc_value_t* const* a_regs,
                            const ckc_moe_operand_t* operands,
                            int num_operands,
                            ckc_value_t* const* b_reg_groups);

/* _emit_moe_mfma_phase. `acc_groups` / `out_groups` are [num_operands] arrays
 * of length mfmas_m*mfmas_n each (flat). Out may alias in (the Python rebuilds
 * new lists); pass distinct buffers. sched_groups==0 disables the hint. */
void ckc_moe_emit_mfma_phase(const ckc_moe_kloop_plan_t* plan,
                             ckc_value_t* a_smem,
                             const ckc_moe_operand_t* operands,
                             int num_operands,
                             ckc_value_t* const* const* acc_groups,
                             const int* group_sizes,
                             ckc_value_t* warp_m_idx,
                             ckc_value_t* warp_n_idx,
                             ckc_value_t* lane,
                             int sched_groups,
                             ckc_value_t** out_groups_flat /* sum(group_sizes) */);

/* _emit_moe_prefetch_kloop. Drives the software-prefetched MFMA k-loop and
 * returns the final accumulator groups (flat, sum(group_sizes) values) into
 * `out_groups_flat`. `acc_inits_flat` are the initial accumulator values
 * (sum(group_sizes) of them) in operand-then-flat order; `group_sizes` is one
 * entry per operand. `acc_names_flat` are the matching loop-carried accumulator
 * SSA names (sum(group_sizes) of them, same operand-then-flat order, e.g.
 * "gate_acc_m0_n0", "up_acc_m0_n0", ...) -- Python carries these from the
 * acc_groups (name, init) tuples; pass NULL to fall back to "acc%d" labels.
 * Returns 1 on success, 0 on error. */
int ckc_moe_emit_prefetch_kloop(const ckc_moe_kloop_plan_t* plan,
                                const ckc_tensor_view_t* a_view,
                                const ckc_tensor_view_t* a_lds_view,
                                ckc_value_t* a_smem,
                                ckc_value_t* const a_mn_origin[2],
                                const ckc_moe_operand_t* operands,
                                int num_operands,
                                ckc_value_t* const b_mn_origin[2],
                                ckc_value_t* const* acc_inits_flat,
                                const char* const* acc_names_flat,
                                const int* group_sizes,
                                ckc_value_t* K,
                                ckc_value_t* warp_m_idx,
                                ckc_value_t* warp_n_idx,
                                ckc_value_t* lane,
                                int sched_groups,
                                ckc_value_t** out_groups_flat);

/* -------------------------------------------------------------- epilogues */

/* _emit_cshuffle_stage: stage one warp's MFMA accumulators into LDS. The
 * accumulator value for slot i of atom (mi, ni) comes from `cell_value`.
 * NOTE: the exact CK-Tile store_tile_cshuffle space-filling walk is TODO(port)
 * (the C distribution layer does not yet export it); the current body performs
 * the equivalent per-(mi,ni,i) scatter at the same MFMA-output LDS addresses. */
void ckc_moe_emit_cshuffle_stage(ckc_ir_builder_t* b,
                                 const ckc_gemm_universal_spec_t* spec,
                                 const ckc_moe_cwarp_decode_t* cdec,
                                 ckc_value_t* smem,
                                 const ckc_type_t* storage_dtype,
                                 int c_per_lane,
                                 ckc_moe_cell_value_fn cell_value,
                                 void* cell_user);

/* _emit_down_reduce_epilogue_atomic: Y[token, n] += weight * down_acc. `accs`
 * is the flat [mfmas_m*mfmas_n] accumulator array. */
void ckc_moe_emit_down_reduce_epilogue_atomic(ckc_ir_builder_t* b,
                                              const ckc_gemm_universal_spec_t* spec,
                                              ckc_value_t* const* accs,
                                              ckc_value_t* warp_m_idx,
                                              ckc_value_t* warp_n_idx,
                                              ckc_value_t* lane,
                                              ckc_value_t* block_m_off,
                                              ckc_value_t* block_n_off,
                                              ckc_value_t* M,
                                              ckc_value_t* N,
                                              ckc_value_t* SortedTokenIds,
                                              ckc_value_t* SortedWeights,
                                              ckc_value_t* Y,
                                              int c_per_lane,
                                              ckc_value_t* batch_bucket_off,
                                              ckc_value_t* tokens);

/* -------------------------------------------------------------- spec types */

/* FusedGateUpSiluGemmSpec: batched per-expert fused gate+up GEMM + SiLU. */
typedef struct ckc_moe_gate_up_silu_gemm_spec
{
    const char* name;
    ckc_gemm_tile_spec_t tile;
    ckc_gemm_trait_spec_t trait; /* default epilogue="default"          */
    int wave_size; /* default 64                          */
    int block_size; /* default 0 => derived at finalize    */
    const char* dtype; /* default "fp16"                      */
    bool grouped; /* default false                       */
} ckc_moe_gate_up_silu_gemm_spec_t;

/* Default-constructed spec (matches the Python field defaults). */
ckc_moe_gate_up_silu_gemm_spec_t ckc_moe_gate_up_silu_gemm_spec_default(void);
/* __post_init__: derive block_size when 0. Idempotent. */
void ckc_moe_gate_up_silu_gemm_spec_finalize(ckc_moe_gate_up_silu_gemm_spec_t* spec);
/* to_universal_spec(). */
ckc_gemm_universal_spec_t
    ckc_moe_gate_up_silu_gemm_spec_to_universal(const ckc_moe_gate_up_silu_gemm_spec_t* spec);
/* kernel_name() -> NUL-terminated into out. */
ckc_status_t ckc_moe_gate_up_silu_gemm_spec_kernel_name(
    const ckc_moe_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap);

/* FusedInterleavedGateUpSiluGemmSpec: single-B gate+up GEMM with in-kernel
 * activation (WGateUp interleaved along N). */
typedef struct ckc_moe_interleaved_gate_up_silu_gemm_spec
{
    const char* name;
    ckc_gemm_tile_spec_t tile;
    ckc_gemm_trait_spec_t trait;
    int wave_size;
    int block_size;
    const char* dtype;
    bool grouped;
} ckc_moe_interleaved_gate_up_silu_gemm_spec_t;

ckc_moe_interleaved_gate_up_silu_gemm_spec_t
    ckc_moe_interleaved_gate_up_silu_gemm_spec_default(void);
void ckc_moe_interleaved_gate_up_silu_gemm_spec_finalize(
    ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec);
ckc_gemm_universal_spec_t ckc_moe_interleaved_gate_up_silu_gemm_spec_to_universal(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec);
ckc_status_t ckc_moe_interleaved_gate_up_silu_gemm_spec_kernel_name(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap);

/* FusedDownReduceGemmSpec: batched down GEMM with top-k weighted reduce. */
typedef struct ckc_moe_down_reduce_gemm_spec
{
    const char* name;
    ckc_gemm_tile_spec_t tile;
    ckc_gemm_trait_spec_t trait;
    int wave_size;
    int block_size;
    const char* dtype;
    bool grouped;
} ckc_moe_down_reduce_gemm_spec_t;

ckc_moe_down_reduce_gemm_spec_t ckc_moe_down_reduce_gemm_spec_default(void);
void ckc_moe_down_reduce_gemm_spec_finalize(ckc_moe_down_reduce_gemm_spec_t* spec);
ckc_gemm_universal_spec_t
    ckc_moe_down_reduce_gemm_spec_to_universal(const ckc_moe_down_reduce_gemm_spec_t* spec);
ckc_status_t ckc_moe_down_reduce_gemm_spec_kernel_name(const ckc_moe_down_reduce_gemm_spec_t* spec,
                                                       char* out,
                                                       size_t out_cap);

/* FusedDownSiluReduceGemmSpec: single fused down+silu+reduce ("up-kernel"). */
typedef struct ckc_moe_down_silu_reduce_gemm_spec
{
    const char* name;
    ckc_gemm_tile_spec_t tile;
    ckc_gemm_trait_spec_t trait;
    int wave_size;
    int block_size;
    const char* dtype;
} ckc_moe_down_silu_reduce_gemm_spec_t;

ckc_moe_down_silu_reduce_gemm_spec_t ckc_moe_down_silu_reduce_gemm_spec_default(void);
void ckc_moe_down_silu_reduce_gemm_spec_finalize(ckc_moe_down_silu_reduce_gemm_spec_t* spec);
ckc_gemm_universal_spec_t ckc_moe_down_silu_reduce_gemm_spec_to_universal(
    const ckc_moe_down_silu_reduce_gemm_spec_t* spec);
ckc_status_t ckc_moe_down_silu_reduce_gemm_spec_kernel_name(
    const ckc_moe_down_silu_reduce_gemm_spec_t* spec, char* out, size_t out_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_INSTANCES_COMMON_MOE_GEMM_FUSED_H */
