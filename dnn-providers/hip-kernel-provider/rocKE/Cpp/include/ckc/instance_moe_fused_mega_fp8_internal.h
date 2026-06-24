/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_moe_fused_mega_fp8_internal.h -- PRIVATE shared state + phase-
 * function contract for the C99 port of build_moe_fused_mega_gemm_fp8
 * (ck_dsl/instances/common/moe_fused_mega_fp8.py).
 *
 * WHY THIS HEADER EXISTS.
 *   build_moe_fused_mega_gemm_fp8() in Python is a ~575-line function whose body
 *   is (a) a stack of MODULE-LEVEL emitters that each take the builder + a long
 *   keyword-argument list (the gate/up fused K-loop, the DTLA stage/read pair,
 *   the X-DTLA stage/read pair, the down group GEMM, the down atomic reduce, the
 *   Pass-A store-hidden, the silu_mul, the f32-view load/store, plus the two MFMA
 *   dispatchers and the schedule-cadence emitters), and (b) a set of NESTED
 *   CLOSURES inside the builder (`_elem_bytes_b`, `_b_base`, `_scale_base`,
 *   `_select_item`, `_emit_body`) that capture and MUTATE the same enclosing
 *   locals -- the builder, the resolved atom, all 25 (or 28 persistent) param
 *   Values, every derived geometry constant, the SSA constants, the block/warp/
 *   lane decomposition, the LDS handles + the four TensorViews, the DTLA
 *   landing-zone view, and the nonlocal per-work-item selected state
 *   (expert_idx, the rebased weight/scale pointers, block_m_off, gu_n_off).
 *
 *   In C there is no closure capture. The faithful port turns each Python module
 *   emitter and each nested closure into a free function that takes a POINTER to
 *   one shared context struct, ckc_moe_fp8_build_ctx_t, which holds EXACTLY the
 *   set of variables those functions shared/mutated. The driver
 *   (ckc_build_moe_fused_mega_gemm_fp8 in ckc/instance_moe_fused_mega_fp8.h)
 *   populates the ctx in the same order the Python prologue computes its locals,
 *   then calls the phase functions in Python execution order.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing agent binds to.
 *   It is DESIGNED TO BE COMPLETE: every local/closured variable the Python body
 *   shares across phases is a field here. A body agent implementing a phase .c
 *   file MUST be able to read/write only ctx fields and call the prototypes below
 *   WITHOUT editing this header. If a phase genuinely needs a value not present,
 *   that is a design bug in this header to be fixed once, deliberately, not
 *   patched per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `block_m_off` ->
 *   `ctx->block_m_off`; Python `Hidden_smem` -> `ctx->Hidden_smem`). The
 *   per-work-item NONLOCAL mutable cell of `_select_item` keeps the EXACT Python
 *   names (expert_idx, WGate/WUp/WDown, WGateScale/WUpScale/WDownScale,
 *   block_m_off, gu_n_off). The immutable base pointers keep the Python `*0`
 *   suffix (WGate0, WGateScale0, ...). Phase functions mirror the Python
 *   module-emitter / closure names with a `ckc_moe_fp8_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_moe_fused_mega_fp8_*.c translation units. Public callers use
 * ckc/instance_moe_fused_mega_fp8.h.
 */
#ifndef CKC_INSTANCE_MOE_FUSED_MEGA_FP8_INTERNAL_H
#define CKC_INSTANCE_MOE_FUSED_MEGA_FP8_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/instance_moe_fused_mega_fp8.h"           /* spec + levers (public) */
#include "ckc/helper_ck_dsl.helpers.atoms.h"           /* ckc_mfma_atom_t */
#include "ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h" /* ckc_lane_decode_t */
#include "ckc/helper_ck_dsl.helpers.tensor_view.h"     /* ckc_tensor_view_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- *
 * Static capacity bounds.
 *   nni      = mfmas_n (gate/up N atoms per warp). At the canonical geometry
 *              tile_n_inter=256, warp_n=4, warp_tile_n=16 => mfmas_n = 4.
 *   accs     = 2*nni gate/up outer accumulators (the fused K-loop iter-args)
 *              and mfmas_m_down*mfmas_n_down down accumulators. 64 is generous.
 *   DTLA slots = 5 (gate/up x 2 ping-pong + 1 X). 16 covers any geometry. */
#define CKC_MOE_FP8_MAX_NNI 64
#define CKC_MOE_FP8_MAX_ACCS 128
#define CKC_MOE_FP8_MAX_DTLA_SLOTS 16

/* ===================================================================== *
 *  ckc_moe_fp8_dtla_bundle_t
 *
 *  Python `dtla_bundle` dict (built in _emit_body when spec.use_dtla):
 *      {"view", "base", "warp_row_base", "lane", "wave_size", "x_slot"}
 *  Threaded into _emit_fp8_gateup_fused_kloop's `dtla=` and consumed by the
 *  DTLA stage/read helpers. NULL-equivalent = `present == false` (Python None
 *  when use_dtla is False -> the legacy global->VGPR path). x_slot is Python's
 *  Optional[int]: has_x_slot==false encodes None. */
typedef struct ckc_moe_fp8_dtla_bundle
{
    bool present;                  /* false => Python None (use_dtla False)   */
    const ckc_tensor_view_t* view; /* bstage_view (LDS landing zone)          */
    ckc_value_t* base;             /* wave_lds_base (smem_ptr_add result, i64) */
    ckc_value_t* warp_row_base;    /* warp_id * warp_rows (i32)               */
    ckc_value_t* lane;             /* lane (i32)                              */
    int wave_size;                 /* spec.wave_size                          */
    bool has_x_slot;               /* x_slot is not None (i.e. _USE_X_DTLA)   */
    int x_slot;                    /* X_SLOT (4) when has_x_slot              */
} ckc_moe_fp8_dtla_bundle_t;

/* ===================================================================== *
 *  ckc_moe_fp8_build_ctx_t
 *
 *  The single shared state object. Holds every enclosing-function local that the
 *  Python module emitters take as kwargs AND every nonlocal the closures mutate,
 *  grouped by the Python prologue's phases. Field declaration order follows the
 *  Python prologue computation order so the populate routine fills it top-down.
 * ===================================================================== */
typedef struct ckc_moe_fp8_build_ctx
{
    /* ---- inputs / config (driver arguments) ------------------------------ */
    ckc_ir_builder_t* b;                          /* the IRBuilder            */
    const ckc_fused_mega_kernel_spec_fp8_t* spec; /* the frozen spec          */
    const char* arch;                             /* "gfx950"                 */
    bool persistent;                              /* persistent ABI variant   */
    ckc_fused_mega_fp8_levers_t levers;           /* resolved lever bundle    */

    /* ---- resolved atom + cadence (build prologue) ------------------------ */
    const ckc_mfma_atom_t* atom; /* spec.gate_up_atom() (K=128 hero / K=32)   */
    /* cadence: spec.sched_cadence override (NULL => defer to levers.sched_cadence
     * env default). Mirrors the Python local `cadence = spec.sched_cadence`. */
    const char* cadence;

    /* ---- kernel param Values (BUILD_SPEC_FP8 Section 2.7) ---------------- *
     * Immutable kernel params (b.param order). The per-expert REBASED weight /
     * scale pointers are the *0 immutable bases below; the rebased Values live in
     * the per-work-item selected-state block. */
    ckc_value_t* A;
    ckc_value_t* WGate0; /* immutable base (Python WGate0)                    */
    ckc_value_t* WUp0;
    ckc_value_t* WDown0;
    ckc_value_t* AScale;
    ckc_value_t* WGateScale0; /* immutable base (Python WGateScale0)          */
    ckc_value_t* WUpScale0;
    ckc_value_t* WDownScale0;
    ckc_value_t* SortedTokenIds;
    ckc_value_t* SortedWeights;
    ckc_value_t* BlockExpertIds;
    ckc_value_t* Y;
    ckc_value_t* M;
    ckc_value_t* N;     /* = I (inter dim)                                    */
    ckc_value_t* K;     /* = H (hidden contraction)                           */
    ckc_value_t* H_out; /* = H (down output)                                  */
    ckc_value_t* stride_a;
    ckc_value_t* stride_b_gate;
    ckc_value_t* stride_b_up;
    ckc_value_t* stride_b_down;
    ckc_value_t* stride_a_scale;
    ckc_value_t* stride_gate_scale;
    ckc_value_t* stride_up_scale;
    ckc_value_t* stride_down_scale;
    ckc_value_t* stride_gate_scale_e;
    ckc_value_t* stride_up_scale_e;
    ckc_value_t* stride_down_scale_e;
    ckc_value_t* slot_size;
    ckc_value_t* tokens;
    /* Persistent-only params (set when persistent; else NULL). */
    ckc_value_t* p_grid_x;
    ckc_value_t* p_total_work;
    ckc_value_t* p_P;

    /* ---- derived geometry scalars (host ints) --------------------------- */
    int tile_m;          /* spec.tile_m                                      */
    int tile_n;          /* spec.tile_n_inter                               */
    int n_blocks;        /* tile_n // GROUP_K                               */
    int warp_n_cols;     /* tile_n // warp_n                                */
    int warps_per_block; /* GROUP_K // warp_n_cols                          */
    int mfmas_m;         /* spec.mfmas_m                                    */
    int mfmas_n;         /* spec.mfmas_n  (== nni)                          */
    int mfmas_m_down;    /* spec.mfmas_m_down                              */
    int mfmas_n_down;    /* spec.mfmas_n_down                             */
    int n_warps;         /* warp_m * warp_n                               */
    /* DTLA landing-zone geometry. */
    int dtla_slots;  /* 5 if _USE_X_DTLA else 4                           */
    bool has_x_slot; /* X_SLOT is not None                               */
    int x_slot;      /* 4 when has_x_slot                                */
    int dtla_chunks; /* ceil(atom.b_per_lane / DTLA_CHUNK)                */
    int bstage_rows; /* n_warps*dtla_slots*dtla_chunks*wave_size          */

    /* ---- SSA constants (b.const_*; created in prologue op-order) --------- */
    ckc_value_t* c_wave;      /* const_i32(wave_size)                        */
    ckc_value_t* c_warps_n;   /* const_i32(warp_n)                          */
    ckc_value_t* c_block_m;   /* const_i32(tile_m)                          */
    ckc_value_t* c_block_n;   /* const_i32(tile_n)                          */
    ckc_value_t* c0;          /* const_i32(0)                              */
    ckc_value_t* c_neg_log2e; /* const_f32(-1.4426950408889634)            */
    ckc_value_t* one_f32;     /* const_f32(1.0)                            */
    ckc_value_t* c_fp8_max;   /* const_f32(FP8_MAX)                        */
    ckc_value_t* c_floor;     /* const_f32(AMAX_FLOOR)                     */
    ckc_value_t* c_group_k;   /* const_i32(GROUP_K)                        */
    ckc_value_t* c_threads;   /* const_i32(block_size)                     */
    ckc_value_t* c_n_blocks;  /* const_i32(n_blocks)  (Python _c_n_blocks)  */
    ckc_value_t* c_tile_n;    /* const_i32(tile_n)    (Python _c_tile_n)   */

    /* ---- block / thread prelude ----------------------------------------- */
    ckc_value_t* tid;        /* thread_id_x()                             */
    ckc_value_t* warp_id;    /* tid / wave                               */
    ckc_value_t* warp_m_idx; /* warp_id / warps_n                        */
    ckc_value_t* warp_n_idx; /* warp_id % warps_n                        */
    ckc_value_t* lane;       /* tid % wave                               */
    ckc_value_t* warp_m_off; /* warp_m_idx * (mfmas_m * atom.m)           */
    ckc_value_t* warp_n_off; /* warp_n_idx * (mfmas_n * atom.n)           */

    /* ---- byte-base helper state (_elem_bytes_b lazy holder) ------------- *
     * Python `_elem_bytes_b_holder` lazy-creates const_i64(1) on first call so the
     * default op-counter order matches the pre-persistent baseline. Mirror the
     * one-shot lazy cell: NULL until first _elem_bytes_b() call. */
    ckc_value_t* elem_bytes_b; /* const_i64(1) once materialized            */

    /* ---- LDS handles + TensorViews (allocated in prologue) -------------- */
    ckc_value_t* Hidden_smem;      /* smem_alloc FP8E4M3 [tile_m, tile_n]    */
    ckc_value_t* HiddenScale_smem; /* smem_alloc F32 [tile_m, n_blocks]      */
    ckc_value_t* HiddenF32_smem;   /* smem_alloc F32 [tile_m, tile_n]        */
    ckc_value_t* WarpAmax_smem;    /* smem_alloc F32 [n_warps]               */
    ckc_value_t* BStage_smem;      /* smem_alloc FP8E4M3 [bstage_rows, 16]   */
    ckc_tensor_view_t bstage_view; /* over BStage_smem (lds)                 */
    ckc_tensor_view_t f32_view;    /* over HiddenF32_smem (lds)              */
    ckc_tensor_view_t fp8_view;    /* over Hidden_smem (lds)                 */
    ckc_tensor_view_t scale_view;  /* over HiddenScale_smem (lds)            */

    /* ---- lane decode -------------------------------------------------- */
    ckc_lane_decode_t lane_decode; /* decode_mfma_lanes(b, atom, lane)       */

    /* ===== per-work-item SELECTED state (the _select_item nonlocals) ===== *
     * Python initializes these to None at prologue scope, then _select_item
     * (re-)derives them per (by, bx). For the default path they are populated ONCE
     * at the original prelude op-order; for the persistent path re-derived each
     * loop iteration. NULL when not yet selected. */
    ckc_value_t* expert_idx; /* global_load_i32(BlockExpertIds, m_block_idx) */
    ckc_value_t* WGate;      /* _b_base(WGate0, stride_b_gate, expert_idx)   */
    ckc_value_t* WUp;
    ckc_value_t* WDown;
    ckc_value_t* WGateScale; /* _scale_base(WGateScale0, stride_e, expert)   */
    ckc_value_t* WUpScale;
    ckc_value_t* WDownScale;
    ckc_value_t* block_m_off; /* m_block_idx * tile_m                        */
    ckc_value_t* gu_n_off;    /* bx_block * tile_n                           */

    /* ===== per-body DTLA bundle (rebuilt in _emit_body) ================== *
     * Built each _emit_body call when spec.use_dtla; threaded into the gate/up
     * fused K-loop. Not present (present=false) under !use_dtla. */
    ckc_moe_fp8_dtla_bundle_t dtla;
} ckc_moe_fp8_build_ctx_t;

/* ===================================================================== *
 * PHASE FUNCTIONS
 *
 * Every phase reads/writes only ctx fields (+ explicit per-call args that are
 * loop-local in Python, e.g. mi/ni tile bases, the K-loop tag). They mirror the
 * Python module emitters and closures 1:1. All return via the builder's sticky-
 * error model (a dead builder is a no-op); functions that produce per-lane Values
 * return them (or write into caller arrays) as noted.
 * ===================================================================== */

/* ---- scheduling-cadence + MFMA dispatch (Python lines 203-282) -------- */

/* _emit_loop_cadence_hint: emit b.iglp_opt(1) iff effective cadence == "iglp1".
 * `cadence` NULL => ctx->cadence (which itself NULL => levers env default). */
void ckc_moe_fp8_emit_loop_cadence_hint(ckc_moe_fp8_build_ctx_t* ctx, const char* cadence);

/* _emit_sgb_gateup_dtla: compv4-style cadence for the DTLA gate/up loop body
 * (no-op unless effective cadence == "sgb"). */
void ckc_moe_fp8_emit_sgb_gateup_dtla(ckc_moe_fp8_build_ctx_t* ctx,
                                      int n_mfma,
                                      const char* cadence);

/* _emit_sgb_down_group: compv4-style VMEM<->MFMA cadence for the down loop
 * per-group body (no-op unless effective cadence == "sgb"). */
void ckc_moe_fp8_emit_sgb_down_group(ckc_moe_fp8_build_ctx_t* ctx, int n_mfma, const char* cadence);

/* _emit_mfma: one K=128 fp8 MFMA, via mfma_f8f6f4_agpr iff levers.use_asm_agpr_mfma
 * && atom.k==128 && fp8e4m3, else atom.emit. Returns the updated accumulator. */
ckc_value_t* ckc_moe_fp8_emit_mfma(ckc_moe_fp8_build_ctx_t* ctx,
                                   ckc_value_t* a,
                                   ckc_value_t* bb,
                                   ckc_value_t* acc);

/* _emit_mfma_down: down-loop MFMA; routes through the AGPR-source helper iff
 * levers.use_asm_agpr_mfma_down (else delegates to _emit_mfma). */
ckc_value_t* ckc_moe_fp8_emit_mfma_down(ckc_moe_fp8_build_ctx_t* ctx,
                                        ckc_value_t* a,
                                        ckc_value_t* bb,
                                        ckc_value_t* acc);

/* ---- per-expert pointer rebasing closures (Python lines 1710-1772) ---- */

/* _elem_bytes_b: lazy const_i64(1); materializes ctx->elem_bytes_b on first call
 * and returns it. */
ckc_value_t* ckc_moe_fp8_elem_bytes_b(ckc_moe_fp8_build_ctx_t* ctx);

/* _b_base(ptr, stride_b, expert_idx): global_ptr_add(ptr,
 * sext(expert)*sext(stride)*elem_bytes_b). */
ckc_value_t* ckc_moe_fp8_b_base(ckc_moe_fp8_build_ctx_t* ctx,
                                ckc_value_t* ptr,
                                ckc_value_t* stride_b,
                                ckc_value_t* expert_idx);

/* _scale_base(ptr, stride_e, expert_idx): global_ptr_add(ptr,
 * sext(expert)*sext(stride_e)*4). */
ckc_value_t* ckc_moe_fp8_scale_base(ckc_moe_fp8_build_ctx_t* ctx,
                                    ckc_value_t* ptr,
                                    ckc_value_t* stride_e,
                                    ckc_value_t* expert_idx);

/* _select_item(m_block_idx, bx_block): derive the per-(by, bx) work-item state
 * (expert_idx + rebased weight/scale pointers + block_m_off + gu_n_off) into the
 * ctx nonlocal cells. bx_block NULL => emit block_id_x() HERE (default-path
 * op-order); the persistent path passes a decoded bx Value. */
void ckc_moe_fp8_select_item(ckc_moe_fp8_build_ctx_t* ctx,
                             ckc_value_t* m_block_idx,
                             ckc_value_t* bx_block);

/* ---- fp8 operand loads (Python lines 983-1039, 1224-1255) ------------- */

/* _global_load_fp8_vec(ptr, addr, n): coalesced fp8 vector load of n contiguous
 * bytes (splits into 16-wide chunks + vec_concat when n>16). */
ckc_value_t* ckc_moe_fp8_global_load_fp8_vec(ckc_moe_fp8_build_ctx_t* ctx,
                                             ckc_value_t* ptr,
                                             ckc_value_t* addr,
                                             int n);

/* _load_a_fp8 (global, row-major (M,K)) / _load_b_fp8 (global, row-major (K,N)).
 * Return the per-lane fp8 fragment Value. */
ckc_value_t* ckc_moe_fp8_load_a_fp8(ckc_moe_fp8_build_ctx_t* ctx,
                                    ckc_value_t* A,
                                    ckc_value_t* m_tile_base,
                                    ckc_value_t* k_tile_base,
                                    ckc_value_t* K);
ckc_value_t* ckc_moe_fp8_load_b_fp8(ckc_moe_fp8_build_ctx_t* ctx,
                                    ckc_value_t* B,
                                    ckc_value_t* n_tile_base,
                                    ckc_value_t* k_tile_base,
                                    ckc_value_t* N);

/* _load_a_fp8_lds: per-lane fp8 A load for the down GEMM from the LDS-resident
 * Hidden (a_view). */
ckc_value_t* ckc_moe_fp8_load_a_fp8_lds(ckc_moe_fp8_build_ctx_t* ctx,
                                        const ckc_tensor_view_t* a_view,
                                        ckc_value_t* m_tile_base,
                                        ckc_value_t* k_tile_base);

/* ---- DTLA B-operand staging (Python lines 1072-1145) ----------------- */

/* _dtla_stage_b_fp8: issue the direct-to-LDS DMA of one lane's b_per_lane fp8
 * weight bytes into `slot`. */
void ckc_moe_fp8_dtla_stage_b_fp8(ckc_moe_fp8_build_ctx_t* ctx,
                                  ckc_value_t* B,
                                  ckc_value_t* n_tile_base,
                                  ckc_value_t* k_tile_base,
                                  ckc_value_t* N,
                                  const ckc_tensor_view_t* stage_view,
                                  int slot,
                                  ckc_value_t* wave_lds_base,
                                  ckc_value_t* lane,
                                  int wave_size);

/* _dtla_read_b_fp8: read back a lane's fp8 B fragment staged in LDS. */
ckc_value_t* ckc_moe_fp8_dtla_read_b_fp8(ckc_moe_fp8_build_ctx_t* ctx,
                                         const ckc_tensor_view_t* stage_view,
                                         int slot,
                                         ckc_value_t* lane,
                                         ckc_value_t* warp_row_base,
                                         int wave_size);

/* ---- X (A) DTLA staging (Python lines 1159-1216; _USE_X_DTLA path) ---- */

/* _xdtla_stage_a_fp8 / _xdtla_read_a_fp8: A-operand mirror of the B DTLA pair. */
void ckc_moe_fp8_xdtla_stage_a_fp8(ckc_moe_fp8_build_ctx_t* ctx,
                                   ckc_value_t* A,
                                   ckc_value_t* m_tile_base,
                                   ckc_value_t* k_tile_base,
                                   ckc_value_t* K,
                                   const ckc_tensor_view_t* stage_view,
                                   int slot,
                                   ckc_value_t* wave_lds_base,
                                   ckc_value_t* lane,
                                   int wave_size);
ckc_value_t* ckc_moe_fp8_xdtla_read_a_fp8(ckc_moe_fp8_build_ctx_t* ctx,
                                          const ckc_tensor_view_t* stage_view,
                                          int slot,
                                          ckc_value_t* lane,
                                          ckc_value_t* warp_row_base,
                                          int wave_size);

/* ---- STAGE 1a: gate+up fused K-loop (Python lines 504-980) ----------- */

/* _emit_fp8_gateup_group_gemm: LEGACY per-(mi,ni) independent K-loop (the pre-
 * combination-lever path). Returns (gate_dq, up_dq) via out_gate / out_up.
 * (Kept for fidelity; the active path is the fused emitter below.) */
void ckc_moe_fp8_emit_fp8_gateup_group_gemm(ckc_moe_fp8_build_ctx_t* ctx,
                                            ckc_value_t* A,
                                            ckc_value_t* WGate,
                                            ckc_value_t* WUp,
                                            ckc_value_t* AScale,
                                            ckc_value_t* WGateScale,
                                            ckc_value_t* WUpScale,
                                            ckc_value_t* m_tile_base,
                                            ckc_value_t* n_tile_base,
                                            ckc_value_t* K,
                                            ckc_value_t* stride_a_scale,
                                            ckc_value_t* stride_gate_scale,
                                            ckc_value_t* stride_up_scale,
                                            const char* tag,
                                            ckc_value_t** out_gate,
                                            ckc_value_t** out_up);

/* _emit_fp8_gateup_fused_kloop: gate+up fp8 GEMM fused across ALL ni cells of one
 * mi row (the COMBINATION lever: shared A load, register-double-buffered B,
 * wave-pair interleave; optional DTLA + cluster MFMA paths). Writes the per-ni
 * gate/up outer f32 accumulators into out_gate[0..nni) / out_up[0..nni) (caller
 * arrays of length >= nni == ctx->mfmas_n). `dtla` NULL => legacy global->VGPR
 * path. */
void ckc_moe_fp8_emit_fp8_gateup_fused_kloop(ckc_moe_fp8_build_ctx_t* ctx,
                                             ckc_value_t* A,
                                             ckc_value_t* WGate,
                                             ckc_value_t* WUp,
                                             ckc_value_t* AScale,
                                             ckc_value_t* WGateScale,
                                             ckc_value_t* WUpScale,
                                             ckc_value_t* m_tile_base,
                                             ckc_value_t* const* n_tile_bases,
                                             int nni,
                                             ckc_value_t* K,
                                             ckc_value_t* stride_a_scale,
                                             ckc_value_t* stride_gate_scale,
                                             ckc_value_t* stride_up_scale,
                                             const char* tag,
                                             const ckc_moe_fp8_dtla_bundle_t* dtla,
                                             const char* cadence,
                                             ckc_value_t** out_gate,
                                             ckc_value_t** out_up);

/* ---- STAGE 1b: SiLU*up + dyn-quant (Python lines 1481-1546) ---------- */

/* _silu_mul_f32: f32 SwiGLU chain silu(g)*u (sigmoid via exp2 + rcp_fast). */
ckc_value_t* ckc_moe_fp8_silu_mul_f32(ckc_moe_fp8_build_ctx_t* ctx,
                                      ckc_value_t* g,
                                      ckc_value_t* u,
                                      ckc_value_t* one_f32,
                                      ckc_value_t* c_neg_log2e);

/* f32_view_store / f32_view_load: 1-wide f32 LDS access at (row, col). */
void ckc_moe_fp8_f32_view_store(ckc_moe_fp8_build_ctx_t* ctx,
                                const ckc_tensor_view_t* view,
                                ckc_value_t* row,
                                ckc_value_t* col,
                                ckc_value_t* val);
ckc_value_t* ckc_moe_fp8_f32_view_load(ckc_moe_fp8_build_ctx_t* ctx,
                                       const ckc_tensor_view_t* view,
                                       ckc_value_t* row,
                                       ckc_value_t* col);

/* _store_hidden_f32_pass: Pass A -- silu(gate)*up -> f32 LDS scratch + in-register
 * per-lane amax. Reads gate_list/up_list (length mfmas_m*mfmas_n, row-major
 * (mi,ni)). Returns the per-lane partial amax (floored). */
ckc_value_t* ckc_moe_fp8_store_hidden_f32_pass(ckc_moe_fp8_build_ctx_t* ctx,
                                               ckc_value_t* const* gate_list,
                                               ckc_value_t* const* up_list,
                                               const ckc_tensor_view_t* f32_view,
                                               ckc_value_t* warp_m_off,
                                               ckc_value_t* warp_n_off,
                                               ckc_value_t* lane,
                                               int mfmas_m,
                                               int mfmas_n,
                                               ckc_value_t* one_f32,
                                               ckc_value_t* c_neg_log2e,
                                               ckc_value_t* c_floor);

/* ---- STAGE 2: down GEMM + atomic reduce (Python lines 1258-1473) ----- */

/* _emit_fp8_down_group_gemm: down fp8 GEMM for one warp-atom output cell -> the
 * per-lane f32 accumulator (returned). */
ckc_value_t* ckc_moe_fp8_emit_fp8_down_group_gemm(ckc_moe_fp8_build_ctx_t* ctx,
                                                  const ckc_tensor_view_t* a_view,
                                                  ckc_value_t* WDown,
                                                  ckc_value_t* WDownScale,
                                                  ckc_value_t* n_tile_base,
                                                  const ckc_tensor_view_t* scale_view,
                                                  int inter_slice,
                                                  ckc_value_t* inter_full,
                                                  ckc_value_t* inter_blk_base,
                                                  ckc_value_t* stride_down_scale,
                                                  ckc_value_t* m_row_base,
                                                  const char* tag,
                                                  const char* cadence);

/* _emit_down_atomic_reduce: weighted, token-validity-masked atomic-add of
 * down_list (length mfmas_m*mfmas_n) into Y. */
void ckc_moe_fp8_emit_down_atomic_reduce(ckc_moe_fp8_build_ctx_t* ctx,
                                         ckc_value_t* const* down_list,
                                         ckc_value_t* warp_m_off,
                                         ckc_value_t* warp_n_off,
                                         ckc_value_t* lane,
                                         int mfmas_m,
                                         int mfmas_n,
                                         ckc_value_t* block_m_off,
                                         ckc_value_t* ho_off,
                                         ckc_value_t* H_out,
                                         ckc_value_t* SortedTokenIds,
                                         ckc_value_t* SortedWeights,
                                         ckc_value_t* Y,
                                         ckc_value_t* tokens);

/* ---- whole-body driver (Python _emit_body, lines 1850-2093) ---------- */

/* _emit_body: run STAGE 1a (per-mi fused gate/up K-loop) -> Pass A + amax
 * butterfly -> per-block scale broadcast -> Pass C packed quantize -> STAGE 2
 * (per-H_out-tile down GEMM + atomic reduce), using the per-work-item selected
 * state already in ctx (expert_idx / WGate.. / block_m_off / gu_n_off). Builds
 * the per-body DTLA bundle into ctx->dtla when spec.use_dtla. */
void ckc_moe_fp8_emit_body(ckc_moe_fp8_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_MOE_FUSED_MEGA_FP8_INTERNAL_H */
