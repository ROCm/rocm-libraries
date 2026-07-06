/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx942_attention_tiled_2d_internal.h -- PRIVATE shared state +
 * phase-function contract for the C99 port of build_unified_attention_2d_tiled
 * (rocke/instances/gfx942/attention_tiled_2d.py, build body lines 1104-5287).
 *
 * WHY THIS HEADER EXISTS.
 *   build_unified_attention_2d_tiled() in Python is a single ~4000-line function
 *   whose body is a deep stack of NESTED CLOSURES: the V-LDS slot/store/load
 *   helpers (_v_t_slot, _v_t_store, _v_t_load, _v_load1), the per-lane row map
 *   (_in_warp_row, _state_row, _bit2, _select_lane_rg), the P permute/pack
 *   helpers (_permute_p_c_to_a16, _pack_p_a16, _pack_p_a32), the accumulator
 *   index helpers (_acc_idx, _acc_get, _acc_final_get), the paged-KV byte-
 *   descriptor closures (_fast_paged_kv_blocks, _fast_paged_kv_voff), the K/V
 *   async-DMA loaders (_issue_k_load_runtime, _issue_k_slice_load_runtime,
 *   _issue_v_load_runtime, _issue_k, _issue_v), the transposed-V store path
 *   (_issue_v_transposed, _cfvst_load_v_regs, _load_token_row_pair,
 *   _cfvst_block_coords, _cfvst_store_v_regs, _issue_v_transposed_store), the
 *   fp8 K/V cache loaders (_issue_kv_fp8_async_load, _dequant_fp8_lds_to_bf16,
 *   _issue_fp8_dequant_loads, _issue_k_fp8_mfma_async, _issue_v_fp8_mfma_async,
 *   _issue_v_fp8_mfma_stripe, _read_k8_mfma_operand), the KV-loop body
 *   (_emit_kv_body) with its inner QK/PV closures (_kslot,
 *   _apply_transposed_pv_regs, _strided_v_b_operand), and the epilogue.
 *
 *   EVERY closure captures the SAME enclosing-function locals: the builder, the
 *   resolved 16x16x16 QK atom, the param Values (output/query/key/value/...,),
 *   every ALL-CAPS geometry constant (HD/T/BS/BLOCK_M/.../the 32x32 + cfv/cfvst
 *   + k1buf umbrella predicates), the grid ids, the wave decomposition, the
 *   seq-idx + cu_q bounds, the smem handles (K_lds/V_lds/P_lds/Q_lds/Acc_lds +
 *   the fp8 staging slabs), the buffer resources + the paged-KV TensorDescriptor,
 *   the transpose-LDS lane formulas, the SSA constants, the per-lane Q VGPR
 *   gather, the online-softmax iter-args, and the LICM-hoisted per-reg
 *   invariants.
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure into a free function that takes a POINTER to one shared context
 *   struct, rocke_gfx942_attn2d_build_ctx_t, holding EXACTLY the set of variables
 *   the closures share. The driver (rocke_build_unified_attention_2d_tiled_scalar
 *   in instance_gfx942_attention_tiled_2d.h) zero-inits the ctx, populates it in
 *   the SAME order the Python prologue computes its locals, then calls the phase
 *   functions in Python execution order.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c binds to.
 *   It is DESIGNED TO BE COMPLETE: every local/closured variable the Python body
 *   shares across phases is a field here. A body agent implementing a phase .c
 *   file MUST be able to read/write only ctx fields and call the prototypes below
 *   WITHOUT editing this header. If a phase genuinely needs a value not present,
 *   that is a design bug in this header to be fixed once, deliberately, not
 *   patched per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python ``K_lds`` ->
 *   ``ctx->K_lds``; Python ``USE_MFMA_32X32`` -> ``ctx->USE_MFMA_32X32``;
 *   Python ``q_block_local_idx`` -> ``ctx->q_block_local_idx``). Phase functions
 *   mirror the Python closure names with a ``rocke_gfx942_attn2d_`` prefix (so they
 *   never clash with the common/ attention_unified / fmha_* families).
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_gfx942_attention_tiled_2d_*.c translation units. Public callers use
 * rocke/instance_gfx942_attention_tiled_2d.h.
 */
#ifndef ROCKE_INSTANCE_GFX942_ATTENTION_TILED_2D_INTERNAL_H
#define ROCKE_INSTANCE_GFX942_ATTENTION_TILED_2D_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_t, rocke_mmaop_t       */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t                     */
#include "rocke/helper_rocke.helpers.attention.h" /* binary_search / softmax reduces     */
#include "rocke/helper_rocke.helpers.transforms.h" /* rocke_tensor_descriptor_t             */
#include "rocke/instance_gfx942_attention_tiled_2d.h" /* spec_t + config_t (via helper hdr) */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 *  Static fan-out bounds (compile-time, generous headroom).
 * ============================================================ *
 *
 * The buildable gfx942 tiled-2D space caps each list below well under these. */
/* Per-lane accumulator slots: ACC_N_TILES * ACC_M_ATOMS. Max is the 32x32 path
 * (16 N-tiles for HD=256/16 * 1) or HD=256 narrow (16 N-tiles * 2 atoms). 64
 * covers any buildable shape. */
#define ROCKE_GFX942_ATTN2D_MAX_ACCS 64
/* Online-softmax iter-args: 2*SOFTMAX_STATE_SLOTS (m,l per slot) + ACCS. */
#define ROCKE_GFX942_ATTN2D_MAX_ITER_ARGS 96
/* Per-lane register slots (REGS_PER_LANE): 4 (M=16), 8 (M=32), 16 (32x32). */
#define ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE 16
/* QK / PV K-iters (HD/16 or T/16); HD<=256, T<=128 -> <=16. */
#define ROCKE_GFX942_ATTN2D_MAX_K_ITERS 16
/* QK / PV N-tiles (T/16 or HD/16); <=16. */
#define ROCKE_GFX942_ATTN2D_MAX_N_TILES 16

/* ============================================================ *
 *  rocke_gfx942_attn2d_build_ctx_t
 *
 *  The single shared state object. Holds every enclosing-function local the
 *  Python closures capture, grouped in the order the Python prologue computes
 *  them (lines 1104-3593), so the populate routine reads top-to-bottom against
 *  the source. Field names mirror the Python locals 1:1 (ALL-CAPS Python
 *  constants stay ALL-CAPS here).
 * ============================================================ */
typedef struct rocke_gfx942_attn2d_build_ctx
{
    /* ---- inputs / resolved environment (lines 1104-1160) ---- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`              */
    const rocke_attention_tiled_2d_spec_t* spec; /* the spec                       */
    const char* arch; /* `arch` (NULL-normalised)       */
    const rocke_archtarget_t* target; /* ArchTarget.from_gfx(arch)      */
    const rocke_type_t* dtype; /* spec.dtype_ir (F16/BF16)       */
    const rocke_mfma_atom_t* qk_atom; /* select_largest_k(...,k_max=16) */

    /* ---- ALL-CAPS geometry constants (lines 1161-1320) ---- */
    int HD, T, BS, N_BLOCKS_PER_TILE;
    int BLOCK_M, BLOCK_Q, NQK, NUM_KV, NUM_QH;
    int SLIDING_WINDOW;
    bool USE_SOFTCAP, USE_SINKS, USE_ALIBI, USE_QQ_BIAS;
    /* transposed-softmax + experimental predicate aliases (spec.*) */
    bool TRANSPOSED_SCALAR_STATE, TRANSPOSED_INVARIANT_HOIST;
    bool TRANSPOSED_MASK_ONCE, TRANSPOSED_HALF_LOCAL_PV;
    bool SKIP_LEGACY_QREG, TRANSPOSED_MASK_LIMIT, GROUPED_KV2;
    bool FAST_PAGED_KV_DESC, I64_KV_ADDR, EARLY_V_SCHEDULE;
    /* fp8 K/V cache predicates */
    bool KV_FP8, FP8_MFMA_QK, FP8_MFMA_PV, FP8_NATIVE_QK;
    bool REGISTER_PV, TRANSPOSED_QK_32X32;
    bool CONFLICT_FREE_V, CONFLICT_FREE_V_STORE, K_SINGLE_BUF;
    bool K_SLICED_RING, K_SLICED_LDSSEQ, USE_IGLP_OPT, USE_GLOBAL_LOAD_LDS_K;
    bool USE_QK_PV_SCHED_GROUP_BARRIER;
    /* env-var diagnostic switches (os.environ reads, lines 1218-1235) */
    bool CFV_SCALAR_READ, CFV_STORE_SCALAR_LOAD, CFV_STORE_SCATTER;
    bool CFV_STORE_PREZERO, CFV_STORE_SEPOFF, CFV_STORE_SPLIT;
    int KV_BYTES;
    const rocke_type_t* kv_io_dtype;
    int kv_cache_aux; /* the aux-bits enum value        */

    /* 32x32 umbrella + QK/PV MFMA geometry (lines 1245-1281) */
    bool USE_MFMA_32X32X8, USE_MFMA_32X32;
    int QK_MFMA_N, QK_K_STEP, PV_K_STEP;
    int QK_K_ITERS, QK_N_TILES, PV_K_ITERS, PV_N_TILES;

    /* threads / wave / async-DMA width (lines 1282-1320) */
    int NUM_WARPS, WAVE, THREADS;
    int ASYNC_LDS_MAX_DWORDS, ASYNC_LDS_MAX_BYTES_PER_LANE;
    /* K/V async-DMA payload width. gfx942 uses dwords=1 (4 bytes/lane = 2 bf16
     * halves); gfx950 (wide ds_read_tr path) uses dwords=4 (16 bytes/lane = 8
     * bf16 halves). KV_DMA_HALVES_PER_LANE drives lane_half_base / per-call /
     * wave-byte stride; KV_DMA_DWORDS is the async_buffer_load_lds_addr dwords
     * arg. Distinct from ASYNC_LDS_MAX_* which still drives the V swizzle. */
    int KV_DMA_HALVES_PER_LANE, KV_DMA_DWORDS;
    int BLOCK_M_PER_WARP, M_ATOMS_PER_WARP, REGS_PER_LANE, SOFTMAX_STATE_SLOTS;

    /* ---- kernel name + attrs (lines 1321-1327) ---- */
    const char* name; /* spec.kernel_name()             */

    /* ---- kernel params (Values, lines 1329-1369) ---- */
    rocke_value_t* output;
    rocke_value_t* query;
    rocke_value_t* key;
    rocke_value_t* value;
    rocke_value_t* sinks;
    rocke_value_t* block_tables;
    rocke_value_t* seq_lens;
    rocke_value_t* alibi_slopes_ptr;
    rocke_value_t* qq_bias_ptr;
    rocke_value_t* cu_q;
    rocke_value_t* scale_p;
    rocke_value_t* k_scale_p;
    rocke_value_t* v_scale_p;
    rocke_value_t* out_scale;
    rocke_value_t* softcap_p;
    rocke_value_t* num_seqs_p;
    rocke_value_t* bt_stride_p;
    rocke_value_t* qq_bias_stride0_p;

    /* ---- grid ids + wave decomposition (lines 1371-1388) ---- */
    rocke_value_t* q_block_global_idx;
    rocke_value_t* kv_head_idx;
    rocke_value_t* tid;
    rocke_value_t* lane;
    rocke_value_t* wave_id; /* NUM_WARPS>1 only               */
    rocke_value_t* wave_row_base;

    /* ---- seq lookup + Q-block geometry (lines 1390-1409) ---- */
    rocke_value_t* seq_idx;
    rocke_value_t* cu_q_start;
    rocke_value_t* cu_q_stop;
    rocke_value_t* cur_batch_q_len;
    rocke_value_t* q_block_start_idx;
    rocke_value_t* q_block_local_idx;
    rocke_value_t* seq_len;
    rocke_value_t* context_len;
    rocke_value_t* qb_start_pos;

    /* ---- Acc_lds stripe geometry (lines 1436-1443) ---- */
    int OUT_STRIPE_COLS, OUT_STRIPES;

    /* ---- LDS layout: dtypes, buffering, sizes (lines 1505-1873) ---- */
    const rocke_type_t* K_LDS_DTYPE;
    const rocke_type_t* V_LDS_DTYPE;
    const rocke_type_t* P_LDS_DTYPE;
    int Q_BYTES;
    int K_SLICE_HD, K_SLICE_SLOTS;
    bool K_SLICED_ACTIVE;
    int K_LDS_ELEM_BYTES, K_BUF_BYTES, K_BUFS, K_TOTAL_BYTES;
    bool Q_DIRECT_GLOBAL, Q_ALIAS_K, Q_USES_DUAL_SLOT;
    int V_BUFS;
    /* V_lds bank-deconflict swizzle */
    bool SWIZZLE_VLDS;
    int V_LDS_PAD, V_LDS_STRIDE, V_GROUP_STRIDE, V_GROUP_SHIFT, V_ROWS_PER_CALL;
    /* transposed-V (cfv/cfvst) layout */
    bool TRANSPOSED_V, TRANSPOSED_V_STORE, V_T_CK_LAYOUT;
    int V_T_PAD, V_T_STRIDE;
    int V_T_KPACK, V_T_PIXELS_PER_ROW, V_T_NPER_ROW;
    int V_T_NGROUPS, V_T_KGROUPS, V_T_GROUP_STRIDE, V_T_CK_SLOTS;
    int N_STRIPES; /* fp8-PV V_lds stripe count      */
    int V_N_SLOTS; /* swizzled flat V_lds slot count */
    int P_LDS_PAD;

    /* ---- smem handles (lines 1532-1873) ---- */
    rocke_value_t* K_lds;
    rocke_value_t* V_lds;
    rocke_value_t* P_lds; /* NULL on the register-P path    */
    rocke_value_t* Q_lds; /* NULL when Q aliases K_lds       */
    rocke_value_t* Acc_lds;
    rocke_value_t* K_fp8_lds; /* fp8 async staging (NULL else)   */
    rocke_value_t* V_fp8_lds;

    /* ---- transpose-LDS lane formulas (lines 1883-1890) ---- *
     * The compile-time TransposeLDSLayout<M=16,K=*,B=1> parameters used by the
     * strided-V B-operand reader and the transposed feeds. */
    int tlds_m, tlds_k, tlds_b;
    /* TransposeLdsReader.bind(b, lane) SSA values (Python layouts.py 280-292),
     * emitted at line 1888 right before qk_scale; cached for the PV reader. */
    rocke_value_t* tlds_lane_div_16;
    rocke_value_t* tlds_lane_div_4_mod_4;
    rocke_value_t* tlds_col;

    /* Per-warp lane decomposition SSA values (Python 2014-2022), emitted once in
     * emit_loop_bounds_and_inits and reused by every consumer. */
    rocke_value_t* lane_rg_v;
    rocke_value_t* lane_col_v;
    rocke_value_t* lane_half32_v;
    rocke_value_t* lane_col32_v;
    /* The Q32 gather RE-EMITS lane_col32 = lane%32 (Python gfx950 2329), which
     * shadows the prologue lane_col32 for the rest of the function. The
     * transposed QK reads THIS value (not the prologue one) for k_row_t. */
    rocke_value_t* lane_col32_q32_v;
    /* The Q32 gather also RE-EMITS lane_half = lane//32 (Python gfx950 line
     * 3503); the non-transposed 32x32 QK reads THIS value for the K column
     * offset. Cache so the QK/softmax TU reuses the same SSA. */
    rocke_value_t* lane_half32_q32_v;
    /* Transposed PV: Python emits v_buf = const_i32(0) + use_hi = cmp_eq(
     * lane_half32, 1) ONCE before the acc-scaling loop (gfx950 3231-3232), then
     * _apply_transposed_pv_regs reuses them. Cache here so the C PV bucket emits
     * them in the same place and the per-n apply calls reuse the same SSA. */
    rocke_value_t* pv_v_buf_v;
    rocke_value_t* pv_use_hi_v;
    rocke_value_t* lane_col_div4_v;
    rocke_value_t* lane_col_mod4_v;
    rocke_value_t* lane_rg_is0_v;
    rocke_value_t* lane_rg_is1_v;
    rocke_value_t* lane_rg_is2_v;

    /* ---- SSA constants (lines 1891-1912) ---- */
    rocke_value_t* c0;
    rocke_value_t* zero_f;
    rocke_value_t* neg_inf_v; /* b.const_f32(-inf)              (1892) */
    rocke_value_t* one_f_v; /* b.const_f32(1.0)               (1894) */
    rocke_value_t* rcp_ln2_v; /* b.const_f32(1.4426950408889634) (1895) */
    rocke_value_t* qk_scale_v; /* derived from scale_p            */
    rocke_value_t* sw_const_v; /* b.const_i32(SLIDING_WINDOW) (1910) */

    /* ---- paged-KV byte descriptor (full transform DAG, lines 2163-2351) ---- */
    rocke_tensor_descriptor_t* q_desc; /* output/query [token,head,dim]   */
    rocke_tensor_descriptor_t* kv_desc; /* paged K/V byte descriptor       */
    rocke_value_t* seq_base; /* to_sgpr_u32(seq_idx*bt_stride)  */
    rocke_value_t* block_table_max_idx; /* to_sgpr_u32(num_seqs*bt_stride) */
    rocke_value_t* kv_block_bytes_c_v; /* const BS*NUM_KV*HD*KV_BYTES (2227) */
    rocke_value_t* lane_half_base_v; /* tid * KV_HALVES_PER_LANE (2232) */
    rocke_value_t* zero_soff_v; /* const_i32(0) soffset (2238)     */
    rocke_value_t* wave_lds_off_i64_v; /* wave_lds_offset_i64 (2254/2263) */
    rocke_value_t* v_wave_lds_off_i64_v; /* v_wave_lds_offset_i64 (2279/87) */
    rocke_value_t* K_lds_addr_v; /* ptrtoint K_lds (2234)           */
    rocke_value_t* V_lds_addr_v; /* ptrtoint V_lds (2235)           */
    rocke_value_t* k_rsrc; /* make_buffer_resource(K)         */
    rocke_value_t* v_rsrc; /* make_buffer_resource(V)         */
    rocke_value_t* out_rsrc;

    /* ---- per-lane Q VGPR gather (lines 3426-3592) ---- *
     * The Q MFMA A-operand registers (one path: the legacy 16x16 gather; the
     * other: the 32x32x16 / direct-global gather). The active list is q_regs
     * (count q_regs_count); the 32x32 path additionally fills q32_regs. */
    rocke_value_t* q_regs[ROCKE_GFX942_ATTN2D_MAX_K_ITERS * ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    int q_regs_count;
    rocke_value_t* q32_regs[ROCKE_GFX942_ATTN2D_MAX_K_ITERS];
    int q32_regs_count;

    /* ---- KV-loop bounds + step (lines 1981-2006) ---- */
    rocke_value_t* tile_start;
    rocke_value_t* tile_end;
    rocke_value_t* max_seq_prefix_len_v; /* cached (Python 1982-1984)        */
    rocke_value_t* kv_step;

    /* ---- online-softmax iter-args (lines 2007-2023) ---- *
     * The scf_for_iter carry: [m_0,l_0, ..., m_{S-1},l_{S-1}, acc_0..acc_{A-1}].
     * iter_args holds the initial carry; iter_args_count == 2*SOFTMAX_STATE_SLOTS
     * + num_accs. ACC indexing: acc lives at iter_args[ml_count + n*ACC_M_ATOMS
     * + atom]; the helpers below resolve that. */
    rocke_value_t* iter_args[ROCKE_GFX942_ATTN2D_MAX_ITER_ARGS];
    /* Per-slot carry name WITHOUT leading '%' ("m0","l0",...,"acc0"/"acc0a0",
     * "cur_buf"). The names are emitted verbatim as the loop's phi-node names, so
     * they must match the Python iter_args tuple names for byte-identity. Each
     * entry points into ctx->iter_args_name_buf. */
    const char* iter_args_names[ROCKE_GFX942_ATTN2D_MAX_ITER_ARGS];
    char iter_args_name_buf[ROCKE_GFX942_ATTN2D_MAX_ITER_ARGS][32]; /* "acc%da%d" worst case = 27 */
    int iter_args_count;
    int ml_count; /* 2 * SOFTMAX_STATE_SLOTS         */
    int ACC_N_TILES; /* PV_N_TILES (epilogue alias)     */
    int ACC_M_ATOMS; /* M_ATOMS_PER_WARP                */

    /* ---- LICM-hoisted per-reg invariants (lines 3609-3700) ---- *
     * The per-(reg) row/pos/head/mask invariants hoisted out of the KV-loop body.
     * Indexed by lane register slot 0..REGS_PER_LANE-1. */
    rocke_value_t* hoist_in_warp_row[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_state_row[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_q_pos[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_q_head[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_row_mask[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];

    /* ---- KV-loop body live carry (set per _emit_kv_body invocation) ---- *
     * The current loop iter var + the unpacked m/l/acc carry the inner QK/mask/
     * softmax/PV closures read and rewrite. ``skip_mask`` is the per-phase flag
     * (the no-SW transposed split runs a full-tile skip_mask=True phase then a
     * boundary phase). out_carry receives the rewritten carry for the next iter. */
    rocke_value_t* kv_tile_iv;
    /* The loop-carried K/V double-buffer slot (carry[ml_count + num_accs]); set
     * by drive_kv_loop alongside kv_tile_iv. The body computes nxt_buf from it
     * (1 - cur_buf, or cur_buf for the single-buffer path). */
    rocke_value_t* cur_buf;
    /* nxt_buf = 1 - cur_buf (or cur_buf when single-buffered). Python emits it at
     * the TOP of the loop body (before tile_off); computed by the front half and
     * reused by the post-QK K issue. */
    rocke_value_t* nxt_buf_v;
    bool skip_mask;
    rocke_value_t* m_cur[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* l_cur[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* acc_cur[ROCKE_GFX942_ATTN2D_MAX_ACCS];
    rocke_value_t* out_carry[ROCKE_GFX942_ATTN2D_MAX_ITER_ARGS];
    int out_carry_count;

    /* ---- final results read by the epilogue (lines 5064-5080) ---- */
    rocke_value_t* l_final[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* acc_final[ROCKE_GFX942_ATTN2D_MAX_ACCS];
    rocke_value_t* rcp_l[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* l_nonzero[ROCKE_GFX942_ATTN2D_MAX_REGS_PER_LANE];
} rocke_gfx942_attn2d_build_ctx_t;

/* ===================================================================== *
 *  DRIVER-INTERNAL ctx population (the build prologue, lines 1104-1912).
 *
 *  Splitting the long prologue out of the public entry keeps the glue TU small.
 *  Reproduces require_tiled_attention_arch(arch) -> dtype gate -> narrow-atom
 *  select -> the whole ALL-CAPS config derivation -> kernel/param decls -> grid /
 *  seq-idx / Q-block geometry -> LDS layout + smem_alloc -> SSA constants. On any
 *  ValueError / NotImplementedError / arch reject it sets the builder error and
 *  returns false. After this returns true the LDS/descriptor/Q-gather/loop phases
 *  can run against the fully-populated ctx.
 * ===================================================================== */
bool rocke_gfx942_attn2d_build_ctx_init(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                        rocke_ir_builder_t* b,
                                        const rocke_attention_tiled_2d_spec_t* spec,
                                        const char* arch);

/* ===================================================================== *
 *  PHASE FUNCTIONS -- one per Python closure / build section.
 *
 *  Each reads/writes only ctx (and the builder it carries) and emits IR in the
 *  byte-identical Python order. Names track the Python source so a reviewer can
 *  diff each .c against its closure.
 * ===================================================================== */

/* ----- per-lane row map + bit helpers (lines 2031-2052) ----- */
rocke_value_t* rocke_gfx942_attn2d_in_warp_row(rocke_gfx942_attn2d_build_ctx_t* ctx, int r);
rocke_value_t* rocke_gfx942_attn2d_state_row(rocke_gfx942_attn2d_build_ctx_t* ctx, int r);
rocke_value_t*
    rocke_gfx942_attn2d_bit2(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* v, int bit);
rocke_value_t* rocke_gfx942_attn2d_select_lane_rg(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* v0,
                                                  rocke_value_t* v1,
                                                  rocke_value_t* v2,
                                                  rocke_value_t* v3);

/* ----- P permute / pack helpers (lines 2053-2146) ----- *
 * _permute_p_c_to_a16: C-distribution P (REGS_PER_LANE f32) -> A16 layout list;
 * writes out (length == in count). _pack_p_a16 / _pack_p_a32: pack to the MFMA
 * A operand vector(s). */
void rocke_gfx942_attn2d_permute_p_c_to_a16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* const* p_regs_f32,
                                            int n,
                                            rocke_value_t** out,
                                            int* out_n);
rocke_value_t* rocke_gfx942_attn2d_pack_p_a16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* const* p_regs_f32,
                                              int n);
rocke_value_t* rocke_gfx942_attn2d_pack_p_a32(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* const* p_regs0_f32,
                                              rocke_value_t* const* p_regs1_f32,
                                              int n);

/* ----- accumulator index helpers (lines 2147-2162) ----- */
int rocke_gfx942_attn2d_acc_idx(const rocke_gfx942_attn2d_build_ctx_t* ctx, int n, int atom);
rocke_value_t* rocke_gfx942_attn2d_acc_get(rocke_gfx942_attn2d_build_ctx_t* ctx, int n, int atom);
rocke_value_t*
    rocke_gfx942_attn2d_acc_final_get(rocke_gfx942_attn2d_build_ctx_t* ctx, int n, int atom);

/* ----- pre-loop: buffer descriptors + tile-0 prefetch (lines 2163-2351) ----- */
void rocke_gfx942_attn2d_emit_preloop(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- fast paged-KV byte descriptor closures (lines 2352-2439) ----- *
 * _fast_paged_kv_blocks(kv_tile_idx) -> (block0, block1); _fast_paged_kv_voff
 * (call, block0, block1) -> per-call voffset. */
void rocke_gfx942_attn2d_fast_paged_kv_blocks(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t** out_block0,
                                              rocke_value_t** out_block1);
rocke_value_t* rocke_gfx942_attn2d_fast_paged_kv_voff(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                      int call,
                                                      rocke_value_t* block0,
                                                      rocke_value_t* block1);

/* ----- V-LDS slot/store/load + flat V load (lines 1756-1830) ----- */
rocke_value_t* rocke_gfx942_attn2d_v_t_slot(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* dim,
                                            rocke_value_t* tok);
void rocke_gfx942_attn2d_v_t_store(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                   rocke_value_t* dim,
                                   rocke_value_t* tok,
                                   rocke_value_t* value,
                                   int n);
rocke_value_t* rocke_gfx942_attn2d_v_t_load(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* dim,
                                            rocke_value_t* tok,
                                            int n);
rocke_value_t* rocke_gfx942_attn2d_v_load1(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                           rocke_value_t* v_buf,
                                           rocke_value_t* v_row,
                                           rocke_value_t* v_n_col);

/* ----- K/V async-DMA loaders (lines 2440-2620, 3347-3425) ----- */
void rocke_gfx942_attn2d_issue_k_load_runtime(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx);
void rocke_gfx942_attn2d_issue_k_slice_load_runtime(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                    rocke_value_t* kv_tile_idx,
                                                    int slice_idx,
                                                    int slot_idx);
void rocke_gfx942_attn2d_issue_v_load_runtime(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx);
void rocke_gfx942_attn2d_issue_k(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx);
void rocke_gfx942_attn2d_issue_v(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx);
/* _read_k8_mfma_operand: per-lane K8 MFMA B operand for the 32x32x8 path. */
rocke_value_t* rocke_gfx942_attn2d_read_k8_mfma_operand(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                        rocke_value_t* kv_tile_idx,
                                                        rocke_value_t* buf_idx,
                                                        int k_iter,
                                                        int n_tile);

/* ----- transposed-V store path (cfv / cfvst, lines 2621-2904) ----- */
void rocke_gfx942_attn2d_issue_v_transposed(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* kv_tile_idx);
/* _cfvst_load_v_regs -> opaque payload (the loaded V register set + coords) the
 * matching store consumes; _load_token_row_pair and _cfvst_block_coords are its
 * inner closures. The payload is arena-owned. */
void* rocke_gfx942_attn2d_cfvst_load_v_regs(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* kv_tile_idx);
rocke_value_t* rocke_gfx942_attn2d_load_token_row_pair(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                       rocke_value_t* t_row,
                                                       rocke_value_t* d0);
void rocke_gfx942_attn2d_cfvst_block_coords(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* blk,
                                            rocke_value_t** out_a,
                                            rocke_value_t** out_b);
void rocke_gfx942_attn2d_cfvst_store_v_regs(rocke_gfx942_attn2d_build_ctx_t* ctx, void* payload);
void rocke_gfx942_attn2d_issue_v_transposed_store(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* kv_tile_idx);

/* ----- fp8 K/V cache loaders (lines 2905-3346) ----- */
void rocke_gfx942_attn2d_issue_kv_fp8_async_load(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx,
                                                 rocke_value_t* buf_idx);
void rocke_gfx942_attn2d_dequant_fp8_lds_to_bf16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* src,
                                                 rocke_value_t* dst,
                                                 rocke_value_t* scale,
                                                 rocke_value_t* buf_idx);
void rocke_gfx942_attn2d_issue_fp8_dequant_loads(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx,
                                                 rocke_value_t* buf_idx);
void rocke_gfx942_attn2d_issue_k_fp8_mfma_async(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                rocke_value_t* kv_tile_idx,
                                                rocke_value_t* buf_idx);
void rocke_gfx942_attn2d_issue_v_fp8_mfma_async(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                rocke_value_t* kv_tile_idx);
void rocke_gfx942_attn2d_issue_v_fp8_mfma_stripe(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx);

/* ----- per-lane Q -> VGPR gather (lines 3426-3592) ----- *
 * Fills ctx->q_regs (+ ctx->q32_regs on the 32x32 path) from the staged Q in LDS
 * (or direct-from-global on the Q_DIRECT_GLOBAL path). Run once before the
 * KV-loop. */
void rocke_gfx942_attn2d_emit_q_gather(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- Q -> LDS cooperative stage (lines 1913-1980) ----- */
void rocke_gfx942_attn2d_emit_q_load(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- LICM hoist of per-reg invariants (lines 3609-3700) ----- *
 * Fills ctx->hoist_* before the loop. */
void rocke_gfx942_attn2d_emit_licm_hoist(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- KV-loop bounds + online-softmax iter-arg carry inits ----- *
 * Ports Python lines 1982-2161 (tile_start/tile_end via max_seq_prefix_len) and
 * the m/l/acc init computation that seeds the scf_for_iter carry. Must run after
 * emit_q_load (so the sink loads land in Python order) and before emit_preloop.
 * Fills ctx->tile_start/tile_end and ctx->iter_args[0 .. ml_count + num_accs)
 * with their names; the trailing cur_buf carry is appended later by
 * emit_loop_step_const (Python line 3606-3607) and kv_step by the same. */
void rocke_gfx942_attn2d_emit_loop_bounds_and_inits(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- tile-0 prefetch + cur_buf carry + kv_step (lines 3591-3689) ----- *
 * Emits _issue_k(tile_start, 0), appends the ("cur_buf", const_i32(0)) carry to
 * iter_args, and records ctx->kv_step. Runs after emit_q_gather, before
 * emit_licm_hoist. */
void rocke_gfx942_attn2d_emit_preloop_prefetch(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- kv_step const (Python 3689) ----- *
 * Emitted after emit_licm_hoist to match Python const emission order. */
void rocke_gfx942_attn2d_emit_kv_step(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- drive the scf_for_iter KV loop (lines 5043-5052) ----- *
 * Builds the loop over [tile_start, tile_end) step kv_step with the named
 * iter_args carry, enters the body region, unpacks the carry into
 * ctx->m_cur/l_cur/acc_cur + sets ctx->kv_tile_iv, runs emit_kv_body, then
 * leaves. Returns the loop handle whose .op->results are the rewritten carry the
 * epilogue consumes. */
rocke_for_t rocke_gfx942_attn2d_drive_kv_loop(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* ----- KV-loop body (_emit_kv_body, lines 3701-5042) ----- *
 * One full KV-tile body: QK MFMA + softcap/mask/softmax + PV MFMA + acc scale,
 * threading the online-softmax carry. Reads ctx->kv_tile_iv / ctx->skip_mask /
 * ctx->m_cur / ctx->l_cur / ctx->acc_cur and writes ctx->out_carry. */
void rocke_gfx942_attn2d_emit_kv_body(rocke_gfx942_attn2d_build_ctx_t* ctx);

/* Re-entrancy reset for the file-scope REGISTER_PV scratch (p_regs_f32_buf) that
 * emit_kv_body uses to carry the in-register P groups from the softmax sub-block
 * to the PV sub-block. The slots hold builder-bound values that dangle once a
 * build's arena is freed; call this at the build entry so each build starts from
 * clean NULL. */
void rocke_gfx942_attn2d_reset_softmax_scratch(void);

/* Inbound softmax-derived state for the PV bucket. The QK/softmax front half
 * fills this and hands it to rocke_gfx942_attn2d_emit_pv_bucket; the PV bucket runs
 * acc *= alpha; acc += P @ V and emits the scf_yield carry. */
typedef struct rocke_gfx942_attn2d_pv_inputs
{
    rocke_value_t* const* alpha_regs; /* SOFTMAX_STATE_SLOTS                       */
    int alpha_count;
    rocke_value_t* const* new_l_vals; /* SOFTMAX_STATE_SLOTS                       */
    rocke_value_t* const* m_new; /* SOFTMAX_STATE_SLOTS                       */
    /* PT32 register groups: pt32[g] is the flat [p_tile*RPL + reg] array for
     * group g; group 0 = current tile, group 1 = GROUPED_KV2 second tile. */
    rocke_value_t* const* pt32_g0;
    rocke_value_t* const* pt32_g1;
    int pt32_count;
    /* p_regs_f32[reg][n] flattened reg*QK_N_TILES + n (REGISTER_PV path). */
    rocke_value_t* const* p_regs_f32;
    int p_regs_f32_stride; /* == QK_N_TILES                                    */
    /* GROUPED_KV2 V re-issue inputs. */
    rocke_value_t* safe_tile1;
    rocke_value_t* nxt_buf;
    rocke_value_t* cur_buf;
} rocke_gfx942_attn2d_pv_inputs_t;

/* PV MFMA + carry yield bucket (Python lines 4540-5041). */
void rocke_gfx942_attn2d_emit_pv_bucket(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                        const rocke_gfx942_attn2d_pv_inputs_t* in);

/* ----- inner QK / PV closures of _emit_kv_body (lines 3868, 4573, 4900) ----- */
/* _kslot(group_idx) -> K_lds slot (sliced-ring slot map). */
int rocke_gfx942_attn2d_kslot(const rocke_gfx942_attn2d_build_ctx_t* ctx, int group_idx);
/* _apply_transposed_pv_regs(acc32, n, p_regs) -> new acc32 (register-P^T PV). */
rocke_value_t* rocke_gfx942_attn2d_apply_transposed_pv_regs(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                            rocke_value_t* acc32,
                                                            int n,
                                                            rocke_value_t* const* p_regs,
                                                            int p_count);
/* _strided_v_b_operand(k_iter) -> the PV V B-operand from strided LDS reads
 * (gfx942 non-transpose path). */
rocke_value_t* rocke_gfx942_attn2d_strided_v_b_operand(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                       int k_iter,
                                                       rocke_value_t* v_n_col,
                                                       rocke_value_t* v_k_chunk_base,
                                                       rocke_value_t* v_buf);

/* ----- epilogue (lines 5054-5287) ----- *
 * Drains outstanding async copies, reads the loop results into ctx->l_final /
 * acc_final / rcp_l / l_nonzero, then emits the normalize + store (the
 * transposed-x8 direct scalar store, the 32x32 Acc_lds stripe store, or the
 * narrow striped vec8 cooperative store). Returns b->kernel on success or NULL
 * with the builder error set. */
rocke_kernel_def_t* rocke_gfx942_attn2d_emit_epilogue(rocke_gfx942_attn2d_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX942_ATTENTION_TILED_2D_INTERNAL_H */
