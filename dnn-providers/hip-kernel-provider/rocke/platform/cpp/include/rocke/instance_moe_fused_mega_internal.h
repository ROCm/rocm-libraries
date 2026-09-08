/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_moe_fused_mega_internal.h -- PRIVATE shared state + phase /
 * closure-function contract for the C99 port of build_moe_fused_mega_gemm
 * (rocke/instances/common/moe_fused_mega.py, 740 LOC).
 *
 * WHY THIS HEADER EXISTS.
 *   build_moe_fused_mega_gemm() is a SINGLE-launch fused-MoE mega-kernel builder
 *   whose body is one big function that (a) computes a long prologue of shared
 *   locals -- params, tile geometry, the block/thread decode, the per-expert B
 *   byte bases, every LDS allocation, the gate/up + down k-loop plans and their
 *   operands, the accumulator-init lists, and the down-GEMM setup -- and then
 *   (b) defines a nested closure `_emit_body()` that walks STAGE 1..5 in order
 *   and is finally invoked under a `scf_if(expert_idx >= 0)` guard.
 *
 *   The Python relies on lexical capture: `_emit_body` reads ~40 enclosing
 *   locals (the builder, params, geometry constants, LDS bases, views, plans,
 *   operands, accumulator inits, SiLU constants). In C there is no closure
 *   capture. The faithful port turns every shared local into a field of one
 *   build-context struct, rocke_moe_mega_build_ctx_t, and turns `_emit_body` --
 *   and the two file-level STAGE-4 helpers _emit_moe_down_mfma_phase_lds_a /
 *   _emit_moe_down_kloop_lds_a -- into free functions over that ctx (the latter
 *   two also take their own per-call args, mirroring the Python signatures).
 *
 *   The public driver (rocke_build_moe_fused_mega_gemm in
 *   rocke/instance_moe_fused_mega.h) computes the prologue into the ctx in the
 *   exact order the Python build does (validity gates -> builder attrs -> params
 *   -> geometry -> thread decode -> per-expert B bases -> LDS allocs -> views ->
 *   plans/operands -> acc inits -> down setup), then emits the guarded body by
 *   calling rocke_moe_mega_emit_body(ctx) inside the scf_if.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing agent binds
 *   to. It is DESIGNED TO BE COMPLETE: every local the Python body shares across
 *   the prologue and the staged body is a field here. A body agent implementing
 *   a phase .c file MUST be able to read/write only ctx fields and call the
 *   prototypes below WITHOUT editing this header. If a phase genuinely needs a
 *   value not present, that is a design bug in this header to fix once.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `A_smem` ->
 *   ctx->A_smem; `plan_down` -> ctx->plan_down; `c_neg_log2e` ->
 *   ctx->c_neg_log2e). Phase functions mirror the Python names with a
 *   `rocke_moe_mega_` prefix; the reused moe_gemm_fused helpers keep their own
 *   `rocke_moe_*` names (helper header) and are NOT re-declared here.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_moe_fused_mega_*.c translation units. Public callers use
 * rocke/instance_moe_fused_mega.h.
 */
#ifndef ROCKE_INSTANCE_MOE_FUSED_MEGA_INTERNAL_H
#define ROCKE_INSTANCE_MOE_FUSED_MEGA_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.tensor_view.h" /* rocke_tensor_view_t           */
#include "rocke/helper_rocke.instances.common.moe_gemm_fused.h"
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_universal_spec_t   */
#include "rocke/instance_moe_fused_mega.h" /* public spec + build entry  */
#include "rocke/ir.h"
/* rocke_moe_kloop_plan_t / rocke_moe_operand_t / rocke_moe_cwarp_decode_t +
 * the reused _silu_mul_f32 / _emit_cshuffle_stage / prefetch-kloop /
 * down-reduce-epilogue helpers the body calls. */

#ifdef __cplusplus
extern "C" {
#endif

/* Per-warp accumulator count = mfmas_per_warp_m * mfmas_per_warp_n. For the
 * covered mega tile space (tile_m 16/32, tile_n 256/512, warp_tile 16, atom
 * 16x16x32) this is at most (32/16)*(512/16/warp_n)=... well within 64; the
 * gate/up path keeps two such groups (gate + up). 64 matches the gemm/conv
 * internal-header convention and is generous headroom. */
#define ROCKE_MOE_MEGA_MAX_ACCS 64

/* ===================================================================== *
 *  rocke_moe_mega_build_ctx_t
 *
 *  The single shared state object. Holds EVERY enclosing-function local that
 *  the build_moe_fused_mega_gemm body (and its `_emit_body` closure) shares.
 *  Grouped by the Python prologue's computation order so a body agent can find
 *  each field next to the Python statement that produced it.
 * ===================================================================== */
typedef struct rocke_moe_mega_build_ctx
{
    /* =============================================================== *
     * (0) inputs / resolved environment (driver args + validity gate)
     * =============================================================== */
    rocke_ir_builder_t* b; /* the IRBuilder `b`             */
    const rocke_moe_fused_mega_kernel_spec_t* spec; /* the FusedMegaKernelSpec       */
    const char* arch; /* `arch` (NULL-normalised)      */

    /* u_gu = spec.gate_up_universal_spec(); u_down = spec.down_universal_spec().
     * Both validated by is_valid_gemm_spec(...) in the prologue. Owned by the
     * driver scratch storage; the ctx holds pointers the body forwards. */
    const rocke_gemm_universal_spec_t* u_gu; /* gate+up universal spec        */
    const rocke_gemm_universal_spec_t* u_down; /* down universal spec           */

    const rocke_type_t* storage_dtype; /* _storage_dtype(u_gu)          */

    /* =============================================================== *
     * (1) kernel params (BUILD_SPEC Section 3.1) -- one Value per param.
     *     A/WGate/WUp/WDown are REBOUND in the prologue to their per-expert byte
     *     base (the Python `WGate = _b_base(WGate, stride_b_gate)` shadowing);
     *     the fields below hold the CURRENT (post-rebind) Values, matching what
     *     `_emit_body` sees lexically.
     * =============================================================== */
    rocke_value_t* A; /* per-expert? no: A base (batch_off_a==0)         */
    rocke_value_t* WGate; /* per-expert byte base (post _b_base)             */
    rocke_value_t* WUp; /* per-expert byte base (post _b_base)             */
    rocke_value_t* WDown; /* per-expert byte base (post _b_base)             */
    rocke_value_t* SortedTokenIds;
    rocke_value_t* SortedWeights;
    rocke_value_t* BlockExpertIds;
    rocke_value_t* Y;
    rocke_value_t* M;
    rocke_value_t* N; /* = I (inter dim)                                 */
    rocke_value_t* K; /* = H (hidden contraction)                        */
    rocke_value_t* H_out; /* = H (down output)                               */
    rocke_value_t* stride_a; /* Python `_stride_a` (unused but bound)           */
    rocke_value_t* stride_b_gate;
    rocke_value_t* stride_b_up;
    rocke_value_t* stride_b_down;
    rocke_value_t* slot_size; /* Python `_slot_size` (unused but bound)          */
    rocke_value_t* tokens;

    /* =============================================================== *
     * (2) gate+up tile geometry (t = spec.gate_up_tile()) + scalar consts
     * =============================================================== */
    rocke_gemm_tile_spec_t t; /* spec.gate_up_tile()                             */
    int block_m, block_n, block_k; /* t.tile_m / t.tile_n / t.tile_k            */
    int mfmas_m, mfmas_n; /* t.mfmas_per_warp_m / _n                    */
    int c_per_lane; /* _mfma_atom_widths(u_gu)[2]                 */

    rocke_value_t* c_wave; /* const_i32(spec.wave_size)                       */
    rocke_value_t* c_warps_n; /* const_i32(t.warp_n)                             */
    rocke_value_t* c_block_m; /* const_i32(block_m)                              */
    rocke_value_t* c_block_n; /* const_i32(block_n)                              */
    rocke_value_t* c0; /* const_i32(0)                                    */

    /* =============================================================== *
     * (3) block / thread prelude (copy of build_moe_gate_up_silu_gemm)
     * =============================================================== */
    rocke_value_t* tid; /* thread_id_x()                                   */
    rocke_value_t* warp_id; /* tid // wave                                     */
    rocke_value_t* warp_m_idx; /* warp_id // warps_n                              */
    rocke_value_t* warp_n_idx; /* warp_id %  warps_n                              */
    rocke_value_t* lane; /* tid %  wave                                     */
    rocke_value_t* m_block_idx; /* block_id_y()                                    */
    rocke_value_t* expert_idx; /* global_load_i32(BlockExpertIds, m_block_idx)    */

    rocke_value_t* elem_bytes_b; /* const_i64(2) (f16/bf16 storage)                 */

    rocke_value_t* batch_off_a; /* c0                                              */
    rocke_value_t* batch_off_b; /* c0                                              */
    rocke_value_t* block_m_off; /* m_block_idx * c_block_m                         */
    rocke_value_t* gu_n_off; /* block_id_x() * c_block_n (inter slice base)     */

    /* =============================================================== *
     * (4) LDS allocations (raw smem Values)
     * =============================================================== */
    rocke_value_t* A_smem; /* [block_m, block_k]   gate/up shared A           */
    rocke_value_t* Bg_smem; /* [block_n, block_k]   gate B                     */
    rocke_value_t* Bu_smem; /* [block_n, block_k]   up   B                     */
    rocke_value_t* Hidden_smem; /* [block_m, block_n]   PERSISTENT silu(g)*u tile  */

    /* =============================================================== *
     * (5) accumulator-init lists (gate/up). Each is a (name,value) pair group of
     *     length mfmas_m*mfmas_n. The C port stores the shared init Value plus
     *     the count; the body rebuilds the named groups the same way the Python
     *     comprehensions do.
     * =============================================================== */
    rocke_value_t* acc_init; /* _emit_zero_acc(b, u_gu)                         */
    int num_gate_up_accs; /* mfmas_m * mfmas_n (per group)                   */

    /* =============================================================== *
     * (6) global + LDS tensor views (gate/up).  Owned by driver scratch.
     * =============================================================== */
    const rocke_tensor_view_t* a_view; /* make_global_view(A, (1,1,1), (1,K,1))   */
    const rocke_tensor_view_t* wg_view; /* make_global_view(WGate, ...)            */
    const rocke_tensor_view_t* wu_view; /* make_global_view(WUp, ...)              */
    const rocke_tensor_view_t* a_lds_view; /* TensorView over A_smem (lds)            */
    const rocke_tensor_view_t* bg_lds_view;
    const rocke_tensor_view_t* bu_lds_view;

    /* =============================================================== *
     * (7) gate/up k-loop plan + operands + origins + SiLU constants
     * =============================================================== */
    rocke_moe_kloop_plan_t plan; /* _MoeKloopPlan(b, u_gu, tid)             */
    rocke_moe_operand_t operands[2]; /* [gate, up]                              */
    rocke_value_t* a_mn_origin[2]; /* (batch_off_a, block_m_off)              */
    rocke_value_t* b_mn_origin[2]; /* (batch_off_b, gu_n_off)                 */

    rocke_value_t* c_neg_log2e; /* const_f32(-1.4426950408889634)          */
    rocke_value_t* one_f32; /* const_f32(1.0)                          */

    /* =============================================================== *
     * (8) DOWN-GEMM setup (Stage 4/5).  td = spec.down_tile().
     * =============================================================== */
    rocke_gemm_tile_spec_t td; /* spec.down_tile()                        */
    int block_n_down, block_k_down; /* td.tile_n / td.tile_k                   */
    int down_mfmas_m, down_mfmas_n; /* td.mfmas_per_warp_m / _n                */
    rocke_value_t* c_block_n_down; /* const_i32(block_n_down)                 */

    rocke_value_t* Bd_smem; /* [block_n_down, block_k_down] down B     */
    const rocke_tensor_view_t* bd_lds_view;
    const rocke_tensor_view_t* wd_view; /* make_global_view(WDown, (1,1,1),(1,N,1))*/

    rocke_moe_kloop_plan_t plan_down; /* _MoeKloopPlan(b, u_down, tid)           */
    rocke_moe_operand_t down_operand; /* W_down operand                          */
    rocke_value_t* c_down_k; /* const_i32(spec.tile_n_inter) (contract) */
    rocke_value_t* down_acc_init; /* _emit_zero_acc(b, u_down)               */
    int num_down_accs; /* down_mfmas_m * down_mfmas_n (per tile)  */
} rocke_moe_mega_build_ctx_t;

/* Zero-initialise the ctx and run the full Python prologue: validity-gate both
 * universal specs (gate+up and down) -> ValueError mapped to ROCKE_ERR_VALUE with
 * message in b->err on reject; set the builder attrs (max_workgroup_size,
 * waves_per_eu); bind every param; compute geometry + thread decode; rebind the
 * per-expert B byte bases; allocate all LDS; build the views/plans/operands; and
 * stage the down-GEMM setup. The caller owns the storage the const-pointer
 * fields (u_gu/u_down/views) reference. Returns ROCKE_OK or the first error.
 *
 * On success the ctx is COMPLETE and rocke_moe_mega_emit_body(ctx) may be called
 * (the public driver does so inside the scf_if(expert_idx >= 0) guard). */
rocke_status_t rocke_moe_mega_build_ctx_init(rocke_moe_mega_build_ctx_t* ctx,
                                             rocke_ir_builder_t* b,
                                             const rocke_moe_fused_mega_kernel_spec_t* spec,
                                             const char* arch,
                                             const rocke_gemm_universal_spec_t* u_gu,
                                             const rocke_gemm_universal_spec_t* u_down);

/* ===================================================================== *
 *  STAGE-4 file-level helpers (Python module-level functions, NOT closures).
 *  They take their own per-call args mirroring the Python signatures plus the
 *  plan/operand; they read no ctx (kept ctx-free to match the Python which
 *  passes everything explicitly). Declared here because the body
 *  (rocke_moe_mega_emit_body) and any unit test call them.
 * ===================================================================== */

/* _emit_moe_down_mfma_phase_lds_a: one K-tile of down-GEMM MFMAs with A read
 * from the persistent LDS buffer `a_smem` at column `a_col_base + kk*warp_tile_k`
 * (B from operand->smem at the per-tile-local column). `accs` / `out_accs` are
 * flat [mfmas_m*mfmas_n]; out may be a distinct buffer (the Python rebuilds the
 * list). sched_groups==0 disables the scheduler hint. */
void rocke_moe_mega_emit_down_mfma_phase_lds_a(const rocke_moe_kloop_plan_t* plan,
                                               rocke_value_t* a_smem,
                                               rocke_value_t* a_col_base,
                                               const rocke_moe_operand_t* operand,
                                               rocke_value_t* const* accs,
                                               int num_accs,
                                               rocke_value_t* warp_m_idx,
                                               rocke_value_t* warp_n_idx,
                                               rocke_value_t* lane,
                                               int sched_groups,
                                               rocke_value_t** out_accs);

/* _emit_moe_down_kloop_lds_a: software-prefetched down-GEMM k-loop reading A
 * from the persistent LDS `a_smem_persistent`. Only the B operand (W_down) is
 * prefetched (global -> single LDS buffer); the MFMA reads A at the running
 * K-tile origin k0 via the phase above. `b_mn_origin` is the 2-elem
 * (batch_off, output_row_base); `b_k_base` is this TG's inter-slice base added
 * to the global W_down K offset; `K` is the down contraction extent
 * (tile_n_inter). Returns the final flat [num_accs] accumulators in `out_accs`.
 * Returns 1 on success, 0 on error (sticky on the plan's builder). */
int rocke_moe_mega_emit_down_kloop_lds_a(const rocke_moe_kloop_plan_t* plan,
                                         rocke_value_t* a_smem_persistent,
                                         const rocke_moe_operand_t* operand,
                                         rocke_value_t* const* acc_inits,
                                         int num_accs,
                                         rocke_value_t* const b_mn_origin[2],
                                         rocke_value_t* b_k_base,
                                         rocke_value_t* K,
                                         rocke_value_t* warp_m_idx,
                                         rocke_value_t* warp_n_idx,
                                         rocke_value_t* lane,
                                         int sched_groups,
                                         rocke_value_t** out_accs);

/* ===================================================================== *
 *  BODY PHASE FUNCTIONS -- the `_emit_body` closure, split into the staged
 *  phases the Python comment headers delineate. Each reads/writes only ctx
 *  (and the builder it carries) and emits IR in byte-identical Python order.
 *  rocke_moe_mega_emit_body is the single entry the public driver calls inside
 *  the scf_if guard; it invokes the sub-phases in order. The sub-phases are
 *  exposed so they can be unit-tested / re-used and so the bucket split is
 *  explicit; a caller only needs rocke_moe_mega_emit_body.
 * ===================================================================== */

/* STAGE 1: gate + up GEMM -> f32 register groups (100% reuse of the prefetch
 * kloop). Writes the two flat accumulator groups (gate_res, up_res), each of
 * length ctx->num_gate_up_accs, into the caller buffers. */
void rocke_moe_mega_emit_stage1_gate_up(rocke_moe_mega_build_ctx_t* ctx,
                                        rocke_value_t** out_gate_res,
                                        rocke_value_t** out_up_res);

/* STAGE 2: SiLU(gate)*up -> PERSISTENT LDS Hidden_smem (no HBM store). Builds
 * the _CWarpDecode, then drives _emit_cshuffle_stage with a per-(mi,ni,i) cell
 * = cast(silu_mul_f32(gate_res, up_res)). Reads gate_res/up_res from STAGE 1.
 * Emits the trailing b.sync(). */
void rocke_moe_mega_emit_stage2_silu_to_lds(rocke_moe_mega_build_ctx_t* ctx,
                                            rocke_value_t* const* gate_res,
                                            rocke_value_t* const* up_res);

/* STAGE 3: reshape note -- OPTION A is an IDENTITY (no IR). Provided as a
 * no-op phase so the body walks all five Python comment stages in order and so
 * the parity fallback (Option B / load_tile_transpose) has a named hook. */
void rocke_moe_mega_emit_stage3_reshape(rocke_moe_mega_build_ctx_t* ctx);

/* STAGE 4 + 5: DOWN GEMM (Hidden_LDS @ Wdown^T) -> weighted atomic reduce into
 * Y. Loops H_out output tiles (scf_for ho in [0, H_out, block_n_down)); per
 * tile: build the down acc-init group, run rocke_moe_mega_emit_down_kloop_lds_a
 * (A=Hidden_smem, B=W_down at output-row ho / contraction-base gu_n_off),
 * b.sync(), then rocke_moe_emit_down_reduce_epilogue_atomic into Y. */
void rocke_moe_mega_emit_stage45_down_reduce(rocke_moe_mega_build_ctx_t* ctx);

/* `_emit_body`: walk STAGE 1..5 in Python order. STAGE 1 fills local gate/up
 * register groups consumed by STAGE 2; this entry owns those temporaries. The
 * public driver wraps the call in scf_if(expert_idx >= 0). */
void rocke_moe_mega_emit_body(rocke_moe_mega_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_FUSED_MEGA_INTERNAL_H */
