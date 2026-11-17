/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_moe_gemm_fused_internal.h -- PRIVATE shared state + phase-function
 * contract for the C99 port of the three MoE GEMM fusion builders
 * (rocke/instances/common/moe_gemm_fused.py).
 *
 * WHY THIS HEADER EXISTS.
 *   Each of the three primary builders (build_moe_gate_up_silu_gemm,
 *   build_moe_interleaved_gate_up_silu_gemm, build_moe_down_reduce_gemm) is a long
 *   function whose body is a prologue (param decls, batched/grouped dispatch,
 *   smem + global/LDS view + accumulator setup, the _MoeKloopPlan + operand list)
 *   followed by an inner closure (_emit_gate_up_compute / emit_compute_and_epilogue
 *   / _emit_down_compute) that the builder calls either directly or inside a
 *   `with b.scf_if(...)` active-tile gate. That closure captures the ENTIRE
 *   enclosing-function local set: the builder, the universal spec, every param
 *   Value (A / W* / Hidden|Y / M,N,K / strides / the optional BlockExpertIds,
 *   SortedTokenIds, SortedWeights, slot_size, tokens), the geometry constants, the
 *   block/warp/lane decomposition, the batched-vs-grouped tile origins
 *   (batch_off_*, block_*_off), the smem handles, the global + LDS TensorViews,
 *   the accumulator iter-arg lists, the _MoeKloopPlan, and the _MoeOperand list
 *   (including the preshuffle `_load_wgateup` closure on the interleaved path).
 *
 *   In C there is no closure capture. The faithful port turns each builder body
 *   into a free-function driver plus a per-family context struct
 *   (rocke_moe_*_build_ctx_t) holding EXACTLY the variables the inner closure shares
 *   with the prologue. The driver populates the ctx in the same order the Python
 *   prologue computes its locals, then calls the compute+epilogue phase
 *   function(s) in Python order (optionally under the scf_if gate).
 *
 *   The four leaf / closure / shared-k-loop helpers (silu, magic-div, vec_rowcol,
 *   pad_in_bounds, _CWarpDecode, _MoeOperand, _MoeKloopPlan, _emit_moe_global_load,
 *   _emit_moe_lds_store, _emit_moe_mfma_phase, _emit_moe_prefetch_kloop,
 *   _emit_cshuffle_stage, _emit_down_reduce_epilogue_atomic) are already ported in
 *   the value-type helper header (helper_rocke.instances.common.moe_gemm_fused.h)
 *   and are NOT redeclared here; the drivers call them. This header adds only the
 *   per-builder ctx structs, the two epilogue closures that the helper header did
 *   NOT carry (gate+up default + interleaved silu), the preshuffle B-load closure,
 *   and the three ctx-init / compute-phase prototypes.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing agent binds to.
 *   It is DESIGNED TO BE COMPLETE: every local/closured variable each Python body
 *   shares across its prologue and its compute closure is a field in the matching
 *   ctx. A body agent implementing a phase .c MUST be able to read/write only ctx
 *   fields and call the prototypes below WITHOUT editing this header. If a phase
 *   genuinely needs a value not present, that is a design bug here to fix once,
 *   deliberately, not patch per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `block_m_off`
 *   -> `ctx->block_m_off`; `batch_off_c` -> `ctx->batch_off_c`). Phase functions
 *   mirror the Python closure / module-helper names with a `rocke_moe_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_moe_gemm_fused_*.c translation units. Public callers use
 * rocke/instance_moe_gemm_fused.h.
 */
#ifndef ROCKE_INSTANCE_MOE_GEMM_FUSED_INTERNAL_H
#define ROCKE_INSTANCE_MOE_GEMM_FUSED_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.tensor_view.h" /* rocke_tensor_view_t              */
#include "rocke/helper_rocke.instances.common.moe_gemm_fused.h" /* spec/leaf/kloop helpers */
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_universal_spec_t              */
#include "rocke/instance_moe_gemm_fused.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-warp accumulator iter-arg count is mfmas_per_warp_m * mfmas_per_warp_n. The
 * gate+up path carries two such groups (gate + up). 64 per group covers any
 * buildable MoE tile (e.g. 32x128 / (1*2 warp, 32x32 atom) = 1x2 = 2 accs). */
#define ROCKE_MOE_MAX_ACCS 64

/* Per-thread coalesced vec-load register count (a_vecs_per_thread /
 * b_vecs_per_thread). For the buildable MoE tile space this is small; 64 is
 * generous headroom. */
#define ROCKE_MOE_MAX_VECS 64

/* ===================================================================== *
 *  GATE+UP+SILU  (dual-B)   build_moe_gate_up_silu_gemm  (lines 710-909)
 * ===================================================================== *
 *
 * Field order follows the Python prologue top-to-bottom. The compute closure is
 * `_emit_gate_up_compute`, called directly (batched) or under
 * `scf_if(expert_idx >= 0)` (grouped). */
typedef struct rocke_moe_gate_up_build_ctx
{
    /* ---- inputs / resolved environment ---- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`            */
    const rocke_moe_gate_up_silu_gemm_spec_t* spec; /* the FusedGateUpSiluGemmSpec  */
    const char* arch; /* `arch` (NULL-normalised)     */
    rocke_gemm_universal_spec_t u; /* spec.to_universal_spec()     */
    const rocke_type_t* storage_dtype; /* _storage_dtype(u)            */
    bool grouped; /* spec.grouped                 */

    /* ---- kernel params (Values) ---- */
    rocke_value_t* A; /* param A     (PtrType storage global)        */
    rocke_value_t* WGate; /* param WGate (rebased by b_base_bytes if grouped) */
    rocke_value_t* WUp; /* param WUp   (rebased by b_base_bytes if grouped) */
    rocke_value_t* Hidden; /* param Hidden (writeonly)                    */
    rocke_value_t* M; /* param M (I32)                               */
    rocke_value_t* N; /* param N (I32)                               */
    rocke_value_t* K; /* param K (I32)                               */
    rocke_value_t* stride_a; /* param stride_a (I32)                        */
    rocke_value_t* stride_b; /* param stride_b (I32)                        */
    rocke_value_t* stride_c; /* param stride_c (I32)                        */
    rocke_value_t* block_expert_ids; /* param BlockExpertIds (grouped only; else NULL) */
    rocke_value_t* expert_idx; /* global_load_i32(block_expert_ids, m_block_idx) */

    /* ---- geometry (aliases of spec.tile) ---- */
    int block_m, block_n, block_k;
    int mfmas_m, mfmas_n;
    int c_per_lane; /* _mfma_atom_widths(u)[2]                     */

    /* ---- common SSA constants ---- */
    rocke_value_t* c0;
    rocke_value_t *c_wave, *c_warps_n, *c_block_m, *c_block_n;

    /* ---- block/warp/lane decomposition ---- */
    rocke_value_t* tid;
    rocke_value_t* warp_id;
    rocke_value_t* warp_m_idx;
    rocke_value_t* warp_n_idx;
    rocke_value_t* lane;

    /* ---- batched-vs-grouped tile origins ---- */
    rocke_value_t* batch_off_a; /* batch_idx*stride_a OR 0 (grouped)        */
    rocke_value_t* batch_off_b; /* batch_idx*stride_b OR 0 (grouped fold)   */
    rocke_value_t* batch_off_c; /* batch_idx*stride_c OR 0 (grouped)        */
    rocke_value_t* block_m_off; /* block_id_y*tile_m (or m_block_idx*tile_m) */
    rocke_value_t* block_n_off; /* block_id_x*tile_n                        */

    /* ---- smem handles ---- */
    rocke_value_t* A_smem;
    rocke_value_t* Bg_smem;
    rocke_value_t* Bu_smem;

    /* ---- accumulator iter-arg lists (gate + up) ---- */
    rocke_value_t* acc_init; /* _emit_zero_acc(b, u)            */
    const char* gate_acc_names[ROCKE_MOE_MAX_ACCS];
    rocke_value_t* gate_acc_inits[ROCKE_MOE_MAX_ACCS];
    const char* up_acc_names[ROCKE_MOE_MAX_ACCS];
    rocke_value_t* up_acc_inits[ROCKE_MOE_MAX_ACCS];
    int num_accs; /* mfmas_m * mfmas_n (per group)   */

    /* ---- global + LDS views ---- */
    rocke_tensor_view_t a_view, wg_view, wu_view; /* 3D global views           */
    rocke_tensor_view_t a_lds_view, bg_lds_view, bu_lds_view; /* 2D LDS views         */

    /* ---- shared k-loop plan + operands ---- */
    rocke_moe_kloop_plan_t plan;
    rocke_moe_operand_t operands[2]; /* [gate, up]                      */
    rocke_value_t* a_mn_origin[2]; /* (batch_off_a, block_m_off)      */
    rocke_value_t* b_mn_origin[2]; /* (batch_off_b, block_n_off)      */

    /* ---- k-loop results (the two accumulator groups, flat per group) ---- */
    rocke_value_t* gate_res[ROCKE_MOE_MAX_ACCS];
    rocke_value_t* up_res[ROCKE_MOE_MAX_ACCS];
} rocke_moe_gate_up_build_ctx_t;

/* ===================================================================== *
 *  INTERLEAVED GATE+UP+SILU (single-B)
 *  build_moe_interleaved_gate_up_silu_gemm  (lines 1144-1391)
 * ===================================================================== *
 *
 * The compute closure is `emit_compute_and_epilogue`, called directly when there
 * is no gate, or under `scf_if(do_work_cond)` (grouped: expert_idx >= 0;
 * active_tile_skip: first sorted token >= 0). */
typedef struct rocke_moe_interleaved_build_ctx
{
    rocke_ir_builder_t* b;
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec;
    const char* arch;
    rocke_gemm_universal_spec_t u;
    const rocke_type_t* storage_dtype;
    bool grouped;
    bool active_tile_skip; /* u.trait.active_tile_skip && !grouped             */
    bool preshuffle_b; /* u.trait.preshuffle_b                             */

    /* ---- kernel params ---- */
    rocke_value_t* A;
    rocke_value_t* WGateUp; /* rebased by b_base_bytes if grouped              */
    rocke_value_t* Hidden;
    rocke_value_t* M;
    rocke_value_t* N; /* logical intermediate I; GEMM N is 2*N           */
    rocke_value_t* K;
    rocke_value_t* stride_a;
    rocke_value_t* stride_b;
    rocke_value_t* stride_c;
    rocke_value_t* block_expert_ids; /* grouped only; else NULL                   */
    rocke_value_t* expert_idx;
    rocke_value_t* sorted_token_ids; /* active_tile_skip only; else NULL          */
    rocke_value_t* slot_size_p; /* active_tile_skip only; else NULL          */

    /* ---- geometry ---- */
    int block_m, block_n, block_k;
    int mfmas_m, mfmas_n;
    int c_per_lane;

    /* ---- SSA constants ---- */
    rocke_value_t* c0;
    rocke_value_t *c_wave, *c_warps_n, *c_block_m, *c_block_n, *c_block_k;

    /* ---- decomposition ---- */
    rocke_value_t *tid, *warp_id, *warp_m_idx, *warp_n_idx, *lane;

    /* ---- tile origins ---- */
    rocke_value_t *batch_off_a, *batch_off_b, *batch_off_c;
    rocke_value_t *block_m_off, *block_n_off;

    /* ---- smem handles (note: C_smem stages the M x 2I gate/up tile) ---- */
    rocke_value_t* A_smem;
    rocke_value_t* B_smem;
    rocke_value_t* C_smem;

    /* ---- single accumulator group ---- */
    rocke_value_t* acc_init;
    const char* acc_names[ROCKE_MOE_MAX_ACCS];
    rocke_value_t* acc_inits[ROCKE_MOE_MAX_ACCS];
    int num_accs;

    /* ---- views ---- */
    rocke_tensor_view_t a_view, b_view;
    rocke_tensor_view_t a_lds_view, b_lds_view;

    /* ---- k-loop plan + the single operand (with optional preshuffle load_b) ----
     * When preshuffle_b, operand.load_b == rocke_moe_interleaved_load_wgateup and
     * operand.load_b_user == this ctx (the closure reads block_n_off / N /
     * batch_off_b / the plan's c_threads/c_load_vec/load_vec). */
    rocke_moe_kloop_plan_t plan;
    rocke_moe_operand_t operand;
    rocke_value_t* a_mn_origin[2];
    rocke_value_t* b_mn_origin[2];

    /* ---- active-tile gate predicate (NULL == unconditional) ---- */
    rocke_value_t* do_work_cond;

    /* ---- k-loop result (single group, flat) ---- */
    rocke_value_t* acc_res[ROCKE_MOE_MAX_ACCS];
} rocke_moe_interleaved_build_ctx_t;

/* ===================================================================== *
 *  DOWN+REDUCE (single-B, atomic)   build_moe_down_reduce_gemm
 *  (lines 1637-1821)
 * ===================================================================== *
 *
 * The compute closure is `_emit_down_compute`, called directly (batched) or under
 * `scf_if(expert_idx >= 0)` (grouped). Note Y is f32 and the metadata params
 * (SortedTokenIds / SortedWeights / slot_size / tokens) are always present. */
typedef struct rocke_moe_down_build_ctx
{
    rocke_ir_builder_t* b;
    const rocke_moe_down_reduce_gemm_spec_t* spec;
    const char* arch;
    rocke_gemm_universal_spec_t u;
    const rocke_type_t* storage_dtype;
    bool grouped;

    /* ---- kernel params ---- */
    rocke_value_t* A;
    rocke_value_t* WDown; /* rebased by b_base_bytes if grouped        */
    rocke_value_t* SortedTokenIds; /* I32 global                                */
    rocke_value_t* SortedWeights; /* F32 global                                */
    rocke_value_t* Y; /* F32 global (atomic target)                */
    rocke_value_t* M;
    rocke_value_t* N;
    rocke_value_t* K;
    rocke_value_t* stride_a;
    rocke_value_t* stride_b;
    rocke_value_t* slot_size;
    rocke_value_t* tokens;
    rocke_value_t* block_expert_ids; /* grouped only; else NULL                   */
    rocke_value_t* expert_idx;

    /* ---- geometry ---- */
    int block_m, block_n, block_k;
    int mfmas_m, mfmas_n;
    int c_per_lane;

    /* ---- SSA constants ---- */
    rocke_value_t* c0_dr; /* Python `c0_dr`                            */
    rocke_value_t *c_wave, *c_warps_n, *c_block_m, *c_block_n;

    /* ---- decomposition ---- */
    rocke_value_t *tid, *warp_id, *warp_m_idx, *warp_n_idx, *lane;

    /* ---- tile origins + bucket base ---- */
    rocke_value_t *batch_off_a, *batch_off_b;
    rocke_value_t* batch_bucket_off; /* batch_idx*slot_size OR 0 (grouped)        */
    rocke_value_t *block_m_off, *block_n_off;

    /* ---- smem handles ---- */
    rocke_value_t* A_smem;
    rocke_value_t* B_smem;

    /* ---- single accumulator group ---- */
    rocke_value_t* acc_init;
    const char* acc_names[ROCKE_MOE_MAX_ACCS];
    rocke_value_t* acc_inits[ROCKE_MOE_MAX_ACCS];
    int num_accs;

    /* ---- views ---- */
    rocke_tensor_view_t a_view, b_view;
    rocke_tensor_view_t a_lds_view, b_lds_view;

    /* ---- k-loop plan + single operand ---- */
    rocke_moe_kloop_plan_t plan;
    rocke_moe_operand_t operand;
    rocke_value_t* a_mn_origin[2];
    rocke_value_t* b_mn_origin[2];

    /* ---- k-loop result (single group, flat) ---- */
    rocke_value_t* acc_res[ROCKE_MOE_MAX_ACCS];
} rocke_moe_down_build_ctx_t;

/* ===================================================================== *
 *  PHASE / CLOSURE FUNCTIONS -- one per Python closure not already in the
 *  value-type helper header. Each reads/writes only its ctx (and the builder it
 *  carries), emitting IR in Python order.
 * ===================================================================== */

/* ----- GATE+UP+SILU ----- */

/* Build prologue (lines 723-866): validate spec, decl params, batched/grouped
 * dispatch, smem + views + accumulators, _MoeKloopPlan + operands. On a rejected
 * spec / any ValueError sets the builder error and returns false. */
bool rocke_moe_gate_up_build_ctx_init(rocke_moe_gate_up_build_ctx_t* ctx,
                                      rocke_ir_builder_t* b,
                                      const rocke_moe_gate_up_silu_gemm_spec_t* spec,
                                      const char* arch);

/* `_emit_gate_up_compute` (lines 868-900): drive the dual-B prefetch k-loop into
 * gate_res / up_res, then call the silu epilogue. */
void rocke_moe_gate_up_emit_compute(rocke_moe_gate_up_build_ctx_t* ctx);

/* `_emit_gate_up_silu_epilogue_default` (lines 912-1016): CShuffle-style epilogue
 * Hidden = silu(gate_acc) * up_acc -- stage the per-lane silu-mul into LDS via
 * the C-warp decode + cshuffle stage, sync, then wide global stores with the pad
 * mask. (Free helper, ctx-independent: takes the explicit Values the Python fn
 * takes; the gate+up driver supplies them from ctx.) */
void rocke_moe_emit_gate_up_silu_epilogue_default(rocke_ir_builder_t* b,
                                                  const rocke_gemm_universal_spec_t* spec,
                                                  rocke_value_t* const* gate_accs,
                                                  rocke_value_t* const* up_accs,
                                                  int num_accs,
                                                  rocke_value_t* warp_m_idx,
                                                  rocke_value_t* warp_n_idx,
                                                  rocke_value_t* lane,
                                                  rocke_value_t* block_m_off,
                                                  rocke_value_t* block_n_off,
                                                  rocke_value_t* M,
                                                  rocke_value_t* N,
                                                  rocke_value_t* Hidden,
                                                  int c_per_lane,
                                                  rocke_value_t* batch_off_c);

/* ----- INTERLEAVED GATE+UP+SILU ----- */

/* Build prologue (lines 1156-1349): validate, decl params (+ grouped /
 * active_tile_skip optionals), even-tile_n check, dispatch, smem + views +
 * accumulators, _MoeKloopPlan, the preshuffle-or-canonical operand, and the
 * do_work_cond gate. Returns false on reject / ValueError (incl. odd tile_n). */
bool rocke_moe_interleaved_build_ctx_init(
    rocke_moe_interleaved_build_ctx_t* ctx,
    rocke_ir_builder_t* b,
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch);

/* `emit_compute_and_epilogue` (lines 1351-1384): drive the single-B prefetch
 * k-loop into acc_res, then call the interleaved silu epilogue. */
void rocke_moe_interleaved_emit_compute(rocke_moe_interleaved_build_ctx_t* ctx);

/* `_load_wgateup` (lines 1308-1322): the preshuffle B per-element load override.
 * Matches rocke_moe_load_b_fn; `user` is the rocke_moe_interleaved_build_ctx_t*. */
rocke_value_t* rocke_moe_interleaved_load_wgateup(rocke_ir_builder_t* b,
                                                  int e,
                                                  rocke_value_t* k_off,
                                                  rocke_value_t* row,
                                                  rocke_value_t* col,
                                                  void* user);

/* `_emit_interleaved_silu_epilogue` (lines 1394-1527): stage the interleaved
 * gate/up tile to LDS via the C-warp decode + cshuffle stage, sync, then for each
 * vec_h chunk wide-read the interleaved pairs, compute silu(gate)*up in f32, pack,
 * and store Hidden with the pad mask. (Free helper; the driver supplies Values.) */
void rocke_moe_emit_interleaved_silu_epilogue(rocke_ir_builder_t* b,
                                              const rocke_gemm_universal_spec_t* spec,
                                              rocke_value_t* const* accs,
                                              int num_accs,
                                              rocke_value_t* C_smem,
                                              rocke_value_t* warp_m_idx,
                                              rocke_value_t* warp_n_idx,
                                              rocke_value_t* lane,
                                              rocke_value_t* block_m_off,
                                              rocke_value_t* block_n_off,
                                              rocke_value_t* M,
                                              rocke_value_t* N,
                                              rocke_value_t* Hidden,
                                              int c_per_lane,
                                              rocke_value_t* batch_off_c);

/* ----- DOWN+REDUCE ----- */

/* Build prologue (lines 1650-1777): validate, decl params (+ grouped optional),
 * batched/grouped dispatch (incl. the bucket base), smem + views + accumulators,
 * _MoeKloopPlan + single operand. Returns false on reject / ValueError. */
bool rocke_moe_down_build_ctx_init(rocke_moe_down_build_ctx_t* ctx,
                                   rocke_ir_builder_t* b,
                                   const rocke_moe_down_reduce_gemm_spec_t* spec,
                                   const char* arch);

/* `_emit_down_compute` (lines 1779-1813): drive the single-B prefetch k-loop into
 * acc_res, then call rocke_moe_emit_down_reduce_epilogue_atomic (value-type helper
 * header) with the ctx's metadata Values. */
void rocke_moe_down_emit_compute(rocke_moe_down_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_GEMM_FUSED_INTERNAL_H */
