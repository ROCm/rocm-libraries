/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_gfx942_attention_tiled_3d_internal.h -- PRIVATE shared state +
 * phase-function contract for the C99 port of the gfx942 narrow-atom tiled
 * split-KV 3D attention kernels (ck_dsl/instances/gfx942/attention_tiled_3d.py):
 *
 *   build_unified_attention_3d_tiled       lines 234-969  (segment kernel)
 *   build_unified_attention_reduce_tiled   lines 1002-1133 (reduce kernel)
 *   _mfma_16x16_c_row(b, lane, reg)        lines 79-88     (C-accum row decode)
 *   _strided_v_b_operand(k_iter, ...)      lines 886-896   (PV B reconstruction)
 *   plus the closured load issuers threaded inside the body:
 *     _issue_k_load / _issue_v_load        lines 604-632   (1-DWORD async DMA)
 *     _issue_wide_load                     lines 645-658   (opt b128 sync feed)
 *     _issue_fp8_dequant_loads             lines 669-701   (fp8 -> dtype -> LDS)
 *     _issue_k / _issue_v                  lines 705-719   (dispatch by KV mode)
 *
 * WHY THIS HEADER EXISTS.
 *   build_unified_attention_3d_tiled is one long body that (a) declares ~16
 *   params in a load-bearing spec-dependent order, (b) computes a prologue stack
 *   of grid ids / binary-search seq-idx / cu_q bounds / kv geometry / SSA
 *   constants / LDS allocs / paged-KV descriptor, (c) closes over a pile of
 *   geometry + the K/V LDS bases + the paged-KV descriptor inside FOUR nested
 *   load-issuer closures (_issue_k_load / _issue_wide_load /
 *   _issue_fp8_dequant_loads + the _issue_k/_issue_v dispatchers) and the
 *   per-N _strided_v_b_operand closure, (d) runs the online-softmax scf.for over
 *   [tile_start, tile_end) threading those closures, and (e) emits the early-out
 *   zero-fill block + the guarded segment-workspace epilogue.
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure / phase into a free function taking a POINTER to one shared context
 *   struct, ckc_gfx942_attention_tiled_3d_build_ctx_t, which holds EXACTLY the
 *   set of compile-time geometry + SSA Values + descriptors + LDS handles those
 *   closures and the loop body share. The glue driver zero-inits a ctx,
 *   populates spec/config + ABI params + prologue in Python order, then calls
 *   the phase functions in Python execution order. The reduce kernel is small
 *   and self-contained; it reuses the same ctx (its own field subset) so the
 *   driver and lower-convenience have one ctx type.
 *
 * CONTRACT STABILITY (bucket note).
 *   This is the ONE shared surface every body-implementing .c binds to. It is
 *   DESIGNED TO BE COMPLETE: every local/shared/closured variable the Python
 *   body passes around is a field here. A body agent implementing a phase MUST
 *   read/write only ctx fields and call the prototypes below WITHOUT editing
 *   this header. A genuinely missing value is a header design bug to fix once,
 *   deliberately, not patch per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `cu_q_start` ->
 *   `ctx->cu_q_start`; Python `qk_scale` -> `ctx->qk_scale`). The all-caps
 *   compile-time config constants keep their Python ALL-CAPS names. Phase
 *   functions mirror the Python closure / build-fn names with the
 *   `ckc_gfx942_attention_tiled_3d_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_gfx942_attention_tiled_3d_*.c translation units. Public callers use
 * ckc/instance_gfx942_attention_tiled_3d.h.
 */
#ifndef CKC_INSTANCE_GFX942_ATTENTION_TILED_3D_INTERNAL_H
#define CKC_INSTANCE_GFX942_ATTENTION_TILED_3D_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/arena.h"
#include "ckc/instance_gfx942_attention_tiled_3d.h"
#include "ckc/helper_ck_dsl.helpers.atoms.h"        /* ckc_mfma_atom_t                  */
#include "ckc/helper_ck_dsl.helpers.distribution.h" /* ckc_tile_distribution_t          */
#include "ckc/helper_ck_dsl.helpers.transforms.h"   /* ckc_tensor_descriptor_t          */
/* The five "new helper" symbols this kernel threads (already ported). */
#include "ckc/helper_helper_ck_dsl.helpers.attention.h"
/* warp_xor_reduce_max/sum + masks. */
#include "ckc/helper_ck_dsl.helpers.attention.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * Which kernel a ctx is being populated for.
 * ============================================================ */
typedef enum ckc_gfx942_attn_tiled_3d_kind
{
    CKC_GFX942_ATTN_TILED_3D_SEGMENT = 0, /* build_unified_attention_3d_tiled     */
    CKC_GFX942_ATTN_TILED_3D_REDUCE       /* build_unified_attention_reduce_tiled */
} ckc_gfx942_attn_tiled_3d_kind_t;

/* ===================================================================== *
 *  Compile-time config (Python ALL-CAPS locals derived from the spec).
 *
 *  All host-side ints; NO IR. Faithful port of the config block at the head
 *  of build_unified_attention_3d_tiled (lines 246-273) plus the reduce
 *  kernel's HALFS_PER_THREAD (line 1014).
 * ===================================================================== */
typedef struct ckc_gfx942_attn_tiled_3d_config
{
    int HD;                   /* spec.head_size                                       */
    int T;                    /* spec.tile_size                                      */
    int BS;                   /* spec.block_size                                     */
    int BLOCK_M;              /* spec.block_m (== 16)                               */
    int BLOCK_Q;              /* spec.block_q                                       */
    int NQK;                  /* spec.num_queries_per_kv                            */
    int NUM_KV;               /* spec.num_kv_heads                                  */
    int NUM_QH;               /* spec.num_query_heads                              */
    int NUM_SEG;              /* spec.num_segments                                 */
    int SLIDING_WINDOW;       /* spec.sliding_window                               */
    bool USE_SOFTCAP;         /* spec.has_softcap                                   */
    bool USE_SINKS;           /* spec.use_sinks                                     */
    bool USE_ALIBI;           /* spec.use_alibi                                     */
    bool USE_QQ_BIAS;         /* spec.use_qq_bias                                  */
    bool USE_INVARIANT_HOIST; /* spec.use_invariant_hoist                     */
    bool KV_FP8;              /* kv_storage_dtype == "fp8e4m3"                     */
    int KV_BYTES;             /* 1 if KV_FP8 else 2                                */

    /* narrow-atom loop trip counts (lines 266-271) */
    int QK_K_STEP;  /* 16 */
    int PV_K_STEP;  /* 16 */
    int QK_K_ITERS; /* HD // QK_K_STEP */
    int QK_N_TILES; /* T // MFMA_N */
    int PV_K_ITERS; /* T // PV_K_STEP */
    int PV_N_TILES; /* HD // MFMA_N */

    int THREADS;             /* 64 */
    int binary_search_iters; /* spec.binary_search_iters */

    /* ---- async / wide / fp8 KV feed geometry (lines 549-667) ---- */
    int ASYNC_LDS_DWORDS;   /* 1 */
    int HALVES_PER_LANE;    /* ASYNC_LDS_DWORDS * 2 */
    int KV_HALVES_PER_CALL; /* THREADS * HALVES_PER_LANE */
    int kv_calls_per_tile;  /* (T*HD) // KV_HALVES_PER_CALL */
    int bytes_per_call;     /* KV_HALVES_PER_CALL * 2 */
    int kv_stride_blk_b;    /* BS*NUM_KV*HD*KV_BYTES */
    int kv_stride_tok_b;    /* NUM_KV*HD*KV_BYTES */
    int kv_stride_h_b;      /* HD*KV_BYTES */
    int bytes_per_buf;      /* T*HD*2 */
    int WIDE_ELEMS;         /* 8 */
    bool WIDE_OK;           /* (T*HD) % (THREADS*WIDE_ELEMS) == 0 */
    int wide_chunks_per_thread;
    int fp8_elems_per_chunk;   /* 8 */
    int fp8_total_chunks;      /* (T*HD)//8 */
    int fp8_chunks_per_thread; /* fp8_total_chunks // THREADS */
    bool WIDE_KV;              /* use_wide_kv_load && !KV_FP8 && WIDE_OK */

    /* Q -> LDS feed (lines 438-439) */
    int Q_VECS_PER_ROW;    /* HD // 8 */
    int Q_VECS_PER_THREAD; /* (BLOCK_M*Q_VECS_PER_ROW)//THREADS */
    int bm1_div_nqk;       /* (BLOCK_M-1)//NQK */

    const ckc_type_t* dtype;       /* F16 / BF16                                 */
    const ckc_type_t* kv_io_dtype; /* FP8E4M3 if KV_FP8 else dtype               */

    /* ---- reduce kernel (lines 1013-1057) ---- */
    int HALFS_PER_THREAD; /* HD // THREADS */
    int SEG_PER_LANE;     /* (NUM_SEG+THREADS-1)//THREADS */
} ckc_gfx942_attn_tiled_3d_config_t;

/* Fill *out from spec (segment kernel). Validates spec + arch, dtype-gates
 * fp16/bf16, and derives every constant above incl. the load-feed asserts
 * (Python asserts -> sticky error + false on violation). arch NULL == "gfx942".
 * Returns true on success. */
bool ckc_gfx942_attn_tiled_3d_config_from_spec(ckc_ir_builder_t* b,
                                               const ckc_unified_attention_3d_tiled_spec_t* spec,
                                               const char* arch,
                                               ckc_gfx942_attn_tiled_3d_config_t* out);

/* Fill *out from the reduce spec (HD/NUM_SEG/NUM_QH/dtype/THREADS/
 * HALFS_PER_THREAD/SEG_PER_LANE). arch is accepted but ignored (arch-neutral).
 * Asserts HALFS_PER_THREAD*THREADS == HD (line 1015). Returns true on success. */
bool ckc_gfx942_attn_tiled_3d_reduce_config_from_spec(
    ckc_ir_builder_t* b,
    const ckc_unified_attention_reduce_tiled_spec_t* spec,
    ckc_gfx942_attn_tiled_3d_config_t* out);

/* ===================================================================== *
 *  ckc_gfx942_attention_tiled_3d_build_ctx_t
 *
 *  Single shared state. Every Value / descriptor / LDS handle the Python
 *  body + its closures pass around. Grouped by the Python prologue phases.
 * ===================================================================== */
typedef struct ckc_gfx942_attention_tiled_3d_build_ctx
{
    /* ---------- inputs / configuration ---------- */
    ckc_ir_builder_t* b;                               /* the IRBuilder (Python `b`)           */
    ckc_gfx942_attn_tiled_3d_kind_t kind;              /* segment or reduce                    */
    const ckc_unified_attention_3d_tiled_spec_t* spec; /* segment spec     */
    const ckc_unified_attention_reduce_tiled_spec_t* reduce_spec; /* reduce spec  */
    ckc_gfx942_attn_tiled_3d_config_t cfg; /* derived compile-time config          */
    ckc_kernel_def_t* kernel;              /* == b->kernel; returned by the driver */

    /* C-accumulator warp distribution for the 16x16x16 atom (_C16_DIST, line 74).
     * Built once (make_static_tile_distribution of make_c_warp_dstr_encoding(
     * MfmaAtom.f16_16x16x16())); read by _mfma_16x16_c_row. */
    const ckc_tile_distribution_t* C16_DIST;

    /* ========================= SEGMENT KERNEL ========================= */

    /* ---------- params (lines 281-329, load-bearing order) ---------- */
    ckc_value_t* segm_output_ptr;   /* F32* writeonly                              */
    ckc_value_t* segm_max_ptr;      /* F32* writeonly                              */
    ckc_value_t* segm_expsum_ptr;   /* F32* writeonly                              */
    ckc_value_t* query;             /* dtype* readonly                             */
    ckc_value_t* key;               /* kv_io_dtype* readonly                       */
    ckc_value_t* value;             /* kv_io_dtype* readonly                       */
    ckc_value_t* sinks;             /* dtype* readonly                             */
    ckc_value_t* block_tables;      /* I32* readonly                               */
    ckc_value_t* seq_lens;          /* I32* readonly                               */
    ckc_value_t* alibi_slopes_ptr;  /* F32* readonly                              */
    ckc_value_t* qq_bias_ptr;       /* F32* readonly                              */
    ckc_value_t* cu_q;              /* I32* readonly (query_start_len_ptr)        */
    ckc_value_t* scale_p;           /* F32 scale                                   */
    ckc_value_t* k_scale_p;         /* F32 k_scale                                 */
    ckc_value_t* v_scale_p;         /* F32 v_scale                                 */
    ckc_value_t* softcap_p;         /* F32 softcap                                 */
    ckc_value_t* num_seqs_p;        /* I32 num_seqs                                */
    ckc_value_t* bt_stride_p;       /* I32 block_table_stride                      */
    ckc_value_t* qq_bias_stride0_p; /* I32 qq_bias_stride_0                        */

    /* ---------- grid ids + thread (lines 331-334) ---------- */
    ckc_value_t* q_block_global_idx; /* block_id_x()                              */
    ckc_value_t* kv_head_idx;        /* block_id_y()                              */
    ckc_value_t* seg_idx;            /* block_id_z()                              */
    ckc_value_t* tid;                /* thread_id_x()                             */

    /* ---------- per-sequence geometry (lines 336-357) ---------- */
    ckc_value_t* seq_idx;            /* binary_search_seq_idx(...)                */
    ckc_value_t* cu_q_start;         /* cu_q[seq_idx]                             */
    ckc_value_t* cu_q_stop;          /* cu_q[seq_idx+1]                           */
    ckc_value_t* cur_batch_q_len;    /* cu_q_stop - cu_q_start                    */
    ckc_value_t* q_block_start_idx;  /* cu_q_start//BLOCK_Q + seq_idx             */
    ckc_value_t* q_block_local_idx;  /* q_block_global_idx - q_block_start_idx    */
    ckc_value_t* seq_len;            /* seq_lens[seq_idx]                         */
    ckc_value_t* context_len;        /* seq_len - cur_batch_q_len                 */
    ckc_value_t* qb_start_pos;       /* q_block_local_idx * BLOCK_Q               */
    ckc_value_t* tps;                /* tiles_per_segment = cdiv(seq_len,NUM_SEG*T)*/
    ckc_value_t* seg_start_tile_pos; /* seg_idx*tps*T                            */

    /* ---------- per-segment tile range (lines 470-477) ---------- */
    ckc_value_t* max_seq_prefix_len; /* min(context_len+qb_start_pos+bm1+1, seq_len)*/
    ckc_value_t* num_tiles;          /* cdiv(max_seq_prefix_len, T)               */
    ckc_value_t* tile_start;         /* seg_idx*tps                               */
    ckc_value_t* tile_end;           /* min((seg_idx+1)*tps, num_tiles)           */

    /* ---------- SSA constants (lines 429-435) ---------- */
    ckc_value_t* neg_inf;  /* const_f32(-inf)                           */
    ckc_value_t* zero_f;   /* const_f32(0.0)                            */
    ckc_value_t* one_f;    /* const_f32(1.0)                            */
    ckc_value_t* rcp_ln2;  /* const_f32(1.4426950408889634)             */
    ckc_value_t* qk_scale; /* scale_p * rcp_ln2                         */
    ckc_value_t* sw_const; /* const_i32(SLIDING_WINDOW)                 */
    ckc_value_t* z8;       /* zero_vec(dtype, 8)                        */

    /* ---------- lane decode (lines 482-483) ---------- */
    ckc_value_t* lane_rg;  /* tid // 16                                 */
    ckc_value_t* lane_col; /* tid % 16                                  */

    /* ---------- descriptors (lines 359-373, 570-602) ---------- */
    ckc_tensor_descriptor_t* ml_desc;       /* segm_ml (token, head, seg)          */
    ckc_tensor_descriptor_t* seg_acc_desc;  /* segm_output (token, head, seg, dim)  */
    ckc_tensor_descriptor_t* q_desc;        /* Q (token, head, dim)                 */
    ckc_tensor_descriptor_t* kv_base_desc;  /* _kv_base byte-stride paged base      */
    ckc_tensor_descriptor_t* paged_kv_desc; /* T==BS or BS%T==0 transformed form    */

    /* ---------- LDS allocations (lines 424-427) ---------- */
    ckc_value_t* Q_lds; /* [BLOCK_M, HD]                             */
    ckc_value_t* K_lds; /* [2, T, HD]                                */
    ckc_value_t* V_lds; /* [2, T, HD]                                */
    ckc_value_t* P_lds; /* [BLOCK_M, T]                              */

    /* ---------- async DMA infra (lines 551-567) ---------- */
    ckc_value_t* big_bytes;      /* const_i32(0x7FFF0000)                     */
    ckc_value_t* key_rsrc;       /* buffer_rsrc(key, big_bytes)               */
    ckc_value_t* value_rsrc;     /* buffer_rsrc(value, big_bytes)             */
    ckc_value_t* lane_half_base; /* tid * HALVES_PER_LANE                     */
    ckc_value_t* K_lds_addr;     /* smem_addr_of(K_lds)                       */
    ckc_value_t* V_lds_addr;     /* smem_addr_of(V_lds)                       */
    ckc_value_t* zero_soff;      /* const_i32(0)                             */
    ckc_value_t* seq_base;       /* seq_idx * bt_stride_p                    */

    /* ---------- online-softmax loop init carry (lines 508-536) ---------- *
     * m_inits/l_inits: 4 regs. acc_inits: PV_N_TILES entries. cur_buf_init = 0.
     * The body-implementing .c sizes the acc arrays from cfg.PV_N_TILES; this
     * fixed-cap holds the legal max (HD=256 -> PV_N_TILES = 16). */
    ckc_value_t* m_inits[4];
    ckc_value_t* l_inits[4];
    ckc_value_t* acc_inits[16]; /* PV_N_TILES <= 16                          */
    ckc_value_t* cur_buf_init;  /* const_i32(0)                              */

    /* ---------- invariant-hoist cache (lines 485-506; NULL when off) ---------- *
     * Per-reg (0..3) hoisted row / qp_r / qh_r / row_ok / causal_lim. */
    ckc_value_t* hoist_row[4];
    ckc_value_t* hoist_qp_r[4];
    ckc_value_t* hoist_qh_r[4];
    ckc_value_t* hoist_row_ok[4];
    ckc_value_t* hoist_causal_lim[4];

    /* ---------- loop results (epilogue inputs, lines 915-918) ---------- */
    ckc_value_t* m_final[4];
    ckc_value_t* l_final[4];
    ckc_value_t* acc_final[16]; /* PV_N_TILES <= 16                          */

    /* ========================= REDUCE KERNEL ========================= */

    /* ---------- params (lines 1022-1030) ---------- */
    ckc_value_t* out;          /* dtype* writeonly                          */
    ckc_value_t* seg_out;      /* F32* readonly (segm_output)               */
    ckc_value_t* seg_max;      /* F32* readonly                             */
    ckc_value_t* seg_l;        /* F32* readonly (segm_expsum)               */
    ckc_value_t* red_seq_lens; /* I32* readonly (_seq_lens, unused body)    */

    /* ---------- grid ids (lines 1032-1034) ---------- */
    ckc_value_t* q_token; /* block_id_x()                              */
    ckc_value_t* q_head;  /* block_id_y()                              */
    /* (reduce tid reuses ctx->tid) */

    /* ---------- reduce descriptors (lines 1039-1053) ---------- */
    ckc_tensor_descriptor_t* ml_desc_red;      /* segm_ml                         */
    ckc_tensor_descriptor_t* seg_acc_desc_red; /* segm_output                     */
    ckc_tensor_descriptor_t* out_desc_red;     /* out (token, head, dim)          */

    /* ---------- reduce state (lines 1055-1102) ---------- */
    ckc_value_t* base_ml;        /* ml_desc_red.offset(q_token,q_head,0)      */
    ckc_value_t* factor_lds;     /* smem_alloc_f32([NUM_SEG])                 */
    ckc_value_t* overall_max;    /* wave64_reduce_max(local_max)              */
    ckc_value_t* overall_expsum; /* wave64_reduce_sum(local_den)              */
    ckc_value_t* inv_l;          /* safe reciprocal of overall_expsum         */
} ckc_gfx942_attention_tiled_3d_build_ctx_t;

/* ============================================================ *
 * Shared host-side helper (no IR)
 * ============================================================ */

/* Zero-init the ctx, copy the spec slice + derive cfg + build C16_DIST. On a
 * validation/dtype/arch failure sets b's sticky error and returns false. `kind`
 * picks which spec pointer is consulted (the other stays NULL). arch NULL ==
 * "gfx942". */
bool ckc_gfx942_attention_tiled_3d_ctx_init(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    ckc_ir_builder_t* b,
    ckc_gfx942_attn_tiled_3d_kind_t kind,
    const ckc_unified_attention_3d_tiled_spec_t* spec,
    const ckc_unified_attention_reduce_tiled_spec_t* reduce_spec,
    const char* arch);

/* ============================================================ *
 * Inner IR helpers (Python module-level / nested closures)
 * ============================================================ */

/* _mfma_16x16_c_row(b, lane, reg) (lines 79-88): MFMA-local output row for a
 * 16x16 C element reg (0..3). Drives ctx->C16_DIST.calculate_x with
 * ys=[0, reg], ps=[[lane//16, lane%16]] and returns the row X-coord. reg
 * outside 0..3 is a Python ValueError -> sticky error + NULL. */
ckc_value_t* ckc_gfx942_attention_tiled_3d_mfma_16x16_c_row(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx, ckc_value_t* lane, int reg);

/* _strided_v_b_operand(ctx, k_iter, v_n_col, v_k_chunk_base) (lines 886-896):
 * build the <4 x dtype> PV B-operand from 4 strided V_lds loads reproducing the
 * per-lane (row, col) a 16x16x16 transpose read would deliver. cur_buf is the
 * current double-buffer index Value (loop carry). Returns the packed vector. */
ckc_value_t*
ckc_gfx942_attention_tiled_3d_strided_v_b_operand(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                                  int k_iter,
                                                  ckc_value_t* cur_buf,
                                                  ckc_value_t* v_n_col,
                                                  ckc_value_t* v_k_chunk_base);

/* The four KV load issuers (closures over the descriptor + LDS bases). Each
 * takes the tile-index + double-buffer-index Values. _issue_k/_issue_v dispatch
 * by KV mode (fp8 -> wide -> 1-DWORD async). lds_token in _issue_fp8 / generic
 * issuers selects K vs V via the ctx K/V handles. */
void ckc_gfx942_attention_tiled_3d_issue_k_load(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                                ckc_value_t* kv_tile_idx,
                                                ckc_value_t* buf_idx); /* lines 604-617 */
void ckc_gfx942_attention_tiled_3d_issue_v_load(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                                ckc_value_t* kv_tile_idx,
                                                ckc_value_t* buf_idx); /* lines 619-632 */
/* _issue_wide_load(src, lds, ...): is_value selects src/lds (false=K, true=V). */
void ckc_gfx942_attention_tiled_3d_issue_wide_load(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                                   bool is_value,
                                                   ckc_value_t* kv_tile_idx,
                                                   ckc_value_t* buf_idx); /* 645-658 */
/* _issue_fp8_dequant_loads(..., lds_token): is_value selects K/V + scale.      */
void ckc_gfx942_attention_tiled_3d_issue_fp8_dequant_loads(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
    bool is_value,
    ckc_value_t* kv_tile_idx,
    ckc_value_t* buf_idx); /* 669-701 */
void ckc_gfx942_attention_tiled_3d_issue_k(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                           ckc_value_t* tile_idx,
                                           ckc_value_t* buf_idx); /* lines 705-711 */
void ckc_gfx942_attention_tiled_3d_issue_v(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx,
                                           ckc_value_t* tile_idx,
                                           ckc_value_t* buf_idx); /* lines 713-719 */

/* ============================================================ *
 * Segment-kernel phase functions (Python execution order)
 * ============================================================ */

/* Declare the ~16 params in the load-bearing spec-dependent order (lines
 * 281-329), filling ctx->segm_output_ptr ... ctx->qq_bias_stride0_p. */
void ckc_gfx942_attention_tiled_3d_declare_params(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Prologue (lines 331-483): grid ids, binary-search seq_idx, cu_q bounds /
 * geometry, the early qb_start_pos>=q_len return guard, tps / descriptors / SSA
 * constants / qk_scale / lane decode / per-segment tile range. Fills the
 * corresponding ctx fields. (The seg_start_tile_pos>=seq_len zero-fill early-out
 * is emitted by ckc_..._emit_early_zero_fill below, called from here in order.) */
void ckc_gfx942_attention_tiled_3d_emit_prologue(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Early-out zero-fill block (lines 376-419): under seg_start_tile_pos>=seq_len,
 * write neg_inf/0 into segm_max/expsum and 0 into segm_output, then ret(). */
void ckc_gfx942_attention_tiled_3d_emit_early_zero_fill(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Q -> LDS feed (lines 437-467): vec8 masked global loads into Q_lds, then
 * sync(). */
void ckc_gfx942_attention_tiled_3d_emit_q_to_lds(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Build the loop-carry init (m/l/acc/cur_buf, lines 508-536, 723-724),
 * including the sinks-conditioned m_inits and the invariant-hoist cache
 * (lines 485-506). Fills ctx->m_inits/l_inits/acc_inits/cur_buf_init + hoist_*.
 * Also issues the first K load (_issue_k(tile_start, 0), line 721). */
void ckc_gfx942_attention_tiled_3d_emit_loop_init(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* The online-softmax scf.for over [tile_start, tile_end) (lines 726-912): per-
 * iter buffer swap, QK narrow MFMA + V/next-K prefetch, alibi/softcap/qq_bias/
 * mask, online (m,l) update via warp_xor reductions, P_lds store, PV narrow MFMA
 * with strided-V B-operand, and the carry yield. Stashes final (m,l,acc) into
 * ctx->m_final/l_final/acc_final. */
void ckc_gfx942_attention_tiled_3d_emit_softmax_loop(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Guarded segment-workspace epilogue (lines 914-967): store acc into
 * segm_output under row_ok; under (lane%16==0 && row_ok) store m/l into
 * segm_max/segm_expsum. */
void ckc_gfx942_attention_tiled_3d_emit_epilogue(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* ============================================================ *
 * Reduce-kernel phase functions (arch-neutral)
 * ============================================================ */

/* Declare reduce params (lines 1022-1030) + grid ids + SSA constants +
 * descriptors + base_ml + factor_lds (lines 1032-1058). */
void ckc_gfx942_attention_tiled_3d_reduce_declare_and_prologue(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Pass 1 (lines 1060-1083): per-lane partial max over owned segments, cache
 * (sv,in_rng,sv_safe) + seg_max/seg_l per lane-slot, then wave64_reduce_max ->
 * ctx->overall_max. (The body .c owns the per-slot caches; ctx exposes the
 * cross-pass overall_max.) */
void ckc_gfx942_attention_tiled_3d_reduce_max_pass(ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Pass 2 (lines 1085-1104): per-lane partial expsum + per-segment factor to
 * factor_lds, wave64_reduce_sum -> overall_expsum, inv_l guard, sync(). */
void ckc_gfx942_attention_tiled_3d_reduce_combine_pass(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

/* Pass 3 (lines 1106-1131): per-element scf.for accumulate over segments,
 * normalize by inv_l, cast to dtype, store to out. */
void ckc_gfx942_attention_tiled_3d_reduce_normalize_pass(
    ckc_gfx942_attention_tiled_3d_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_GFX942_ATTENTION_TILED_3D_INTERNAL_H */
