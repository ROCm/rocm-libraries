/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx950_attention_tiled_2d_internal.h -- PRIVATE shared state +
 * phase-function contract for the C99 port of build_unified_attention_2d_tiled
 * (rocke/instances/gfx950/attention_tiled_2d.py, build body lines 711-3818).
 *
 * WHY THIS HEADER EXISTS.
 *   build_unified_attention_2d_tiled() in Python is a single ~3100-line function
 *   whose body is a deep stack of NESTED CLOSURES, ALL capturing the SAME
 *   enclosing-function locals. In C there is no closure capture; the faithful
 *   port turns each Python closure into a free function taking a POINTER to one
 *   shared context struct, rocke_gfx950_attn2d_build_ctx_t, holding EXACTLY the set
 *   of variables the closures share. The driver
 *   (rocke_gfx950_build_unified_attention_2d_tiled in
 *   instance_gfx950_attention_tiled_2d.h) zero-inits the ctx, populates it in the
 *   SAME order the Python prologue computes its locals, then calls the phase
 *   functions in Python execution order.
 *
 *   The gfx950 closure set (verified against the source) is a strict SUBSET of
 *   the gfx942 sibling:
 *     row map / bit:   _in_warp_row, _state_row, _bit2, _select_lane_rg   (1270-1290)
 *     P permute/pack:  _permute_p_c_to_a16, _pack_p_a16, _pack_p_a32       (1292-1384)
 *     acc index:       _acc_idx (1386), _acc_get (2505), _acc_final_get (3605)
 *     fast paged-KV:   _fast_paged_kv_blocks (1550), _fast_paged_kv_voff (1573)
 *     K/V loaders:     _issue_k_load_runtime (1631), _issue_v_load_runtime (1691),
 *                      _issue_k (2184), _issue_v (2204), _read_k8_mfma_operand (2229)
 *     fp8 K/V cache:   _issue_kv_fp8_async_load (1798), _dequant_fp8_lds_to_bf16 (1870),
 *                      _issue_fp8_dequant_loads (1928), _issue_k_fp8_mfma_async (2050),
 *                      _issue_v_fp8_mfma_async (2106), _issue_v_fp8_mfma_stripe (2143)
 *     KV-loop body:    _emit_kv_body (2494) with inner _apply_transposed_pv_regs (3234)
 *
 *   gfx950 has NONE of gfx942's transposed-V-store family (_issue_v_transposed,
 *   _cfvst_load_v_regs, _load_token_row_pair, _cfvst_block_coords,
 *   _cfvst_store_v_regs, _issue_v_transposed_store), NO K-sliced-ring
 *   (_issue_k_slice_load_runtime, _kslot), and NO _strided_v_b_operand: the wide
 *   atom's CK-Tile LDS transpose reads (ds_read_b64_tr_b16 / ds_read_b64_tr_b8
 *   via TransposeLdsReader, the ctx->pv_tr_reader binding) replace every one of
 *   those gfx942 scalar-reconstruction paths. This header therefore OMITS those
 *   fields/prototypes by design.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing .c binds to. It
 *   is DESIGNED TO BE COMPLETE: every local/closured variable the Python body
 *   shares across phases is a field here. A body agent implementing a phase .c
 *   MUST read/write only ctx fields and call the prototypes below WITHOUT editing
 *   this header. If a phase genuinely needs a value not present, that is a design
 *   bug here to fix once, deliberately, not patch per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python ``K_lds`` ->
 *   ``ctx->K_lds``; ``USE_MFMA_32X32`` -> ``ctx->USE_MFMA_32X32``;
 *   ``q_block_local_idx`` -> ``ctx->q_block_local_idx``). Phase functions mirror
 *   the Python closure names with a ``rocke_gfx950_attn2d_`` prefix (so they never
 *   clash with the gfx942 sibling or the common/ attention_unified / fmha_*
 *   families).
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * instance_gfx950_attention_tiled_2d_*.c translation units. Public callers use
 * rocke/instance_gfx950_attention_tiled_2d.h.
 */
#ifndef ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_INTERNAL_H
#define ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_t, rocke_mmaop_t       */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t                     */
#include "rocke/helper_rocke.helpers.attention.h" /* binary_search / softmax reduces / fp8 dequant */
#include "rocke/helper_rocke.helpers.distribution.h" /* static tile distribution (32x32 C dist)        */
#include "rocke/helper_rocke.helpers.layouts.h" /* rocke_transpose_lds_reader_t (PV reader)         */
#include "rocke/helper_rocke.helpers.transforms.h" /* rocke_tensor_descriptor_t             */
#include "rocke/instance_gfx950_attention_tiled_2d.h" /* spec_t + config_t (via helper hdr) */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 *  Static fan-out bounds (compile-time, generous headroom).
 * ============================================================ */
/* Per-lane accumulator slots: ACC_N_TILES * ACC_M_ATOMS. Max is the 32x32 path
 * (16 per-lane elems for the wide C dist) or HD=256 narrow (16 N-tiles * 2
 * atoms). 64 covers any buildable shape. */
#define ROCKE_GFX950_ATTN2D_MAX_ACCS 64
/* Online-softmax iter-args: 2*SOFTMAX_STATE_SLOTS (m,l per slot) + ACCS + cur_buf. */
#define ROCKE_GFX950_ATTN2D_MAX_ITER_ARGS 96
/* Per-lane register slots (REGS_PER_LANE): 4 (M=16), 8 (M=32), 16 (32x32). */
#define ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE 16
/* QK / PV K-iters (HD/QK_K_STEP or T/PV_K_STEP); HD<=256, T<=128 -> <=16. */
#define ROCKE_GFX950_ATTN2D_MAX_K_ITERS 16
/* QK / PV N-tiles (T/QK_MFMA_N or HD/16); <=16. */
#define ROCKE_GFX950_ATTN2D_MAX_N_TILES 16

/* ============================================================ *
 *  rocke_gfx950_attn2d_build_ctx_t
 *
 *  The single shared state object. Holds every enclosing-function local the
 *  Python closures capture, grouped in the order the Python prologue computes
 *  them (lines 711-1912 prologue, plus the loop/body carry). Field names mirror
 *  the Python locals 1:1 (ALL-CAPS Python constants stay ALL-CAPS here).
 * ============================================================ */
typedef struct rocke_gfx950_attn2d_build_ctx
{
    /* ---- inputs / resolved environment (lines 711-760) ---- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`              */
    const rocke_attention_tiled_2d_spec_t* spec; /* the spec                       */
    const char* arch; /* `arch` (NULL-normalised gfx950)*/
    const rocke_archtarget_t* target; /* ArchTarget.from_gfx(arch)      */
    const rocke_type_t* dtype; /* spec.dtype_ir (F16/BF16)       */
    /* The WIDE-K QK atom: mfma_32x32x16_for_dtype (USE_MFMA_32X32) or
     * mfma_16x16x32_for_dtype otherwise (Python lines 813-814 / qk_atom select).
     * This is the gfx950 divergence from gfx942's narrow select_largest_k. */
    const rocke_mfma_atom_t* qk_atom;

    /* ---- ALL-CAPS geometry constants (lines 754-847) ---- */
    int HD, T, BS, N_BLOCKS_PER_TILE;
    int BLOCK_M, BLOCK_Q, NQK, NUM_KV, NUM_QH;
    int SLIDING_WINDOW;
    bool USE_SOFTCAP, USE_SINKS, USE_ALIBI, USE_QQ_BIAS;
    /* transposed-softmax + experimental predicate aliases (spec.*) */
    bool TRANSPOSED_SCALAR_STATE, TRANSPOSED_INVARIANT_HOIST;
    bool TRANSPOSED_MASK_ONCE, TRANSPOSED_HALF_LOCAL_PV;
    bool SKIP_LEGACY_QREG, TRANSPOSED_MASK_LIMIT, GROUPED_KV2;
    bool FAST_PAGED_KV_DESC, I64_KV_ADDR, EARLY_V_SCHEDULE;
    bool AGPR_ALLOC_ZERO;
    /* K single-buffer (1 K slot) so T=64 fits the 2-WG/CU LDS budget at
     * HD=128. The next-K prefetch is deferred to after the PV-wait barrier. */
    bool K_SINGLE_BUFFER;
    /* Lever-3 (CK-Tile-derived) sched_barrier fence after the QK MFMA cluster. */
    bool USE_SCHED_BARRIER;
    int SCHED_BARRIER_MASK;
    /* fp8 K/V cache predicates */
    bool KV_FP8, FP8_MFMA_QK, FP8_MFMA_PV;
    bool REGISTER_PV, USE_MFMA_32X32, TRANSPOSED_QK_32X32;
    int KV_BYTES;
    const rocke_type_t* kv_io_dtype;
    int kv_cache_aux; /* the aux-bits enum value        */

    /* QK/PV MFMA geometry (lines 813-820) */
    int QK_MFMA_N, QK_K_STEP, PV_K_STEP;
    int QK_K_ITERS, QK_N_TILES, PV_K_ITERS, PV_N_TILES;

    /* threads / wave / async-DMA width (lines 821-838) */
    int NUM_WARPS, WAVE, THREADS;
    int ASYNC_LDS_MAX_DWORDS, ASYNC_LDS_MAX_BYTES_PER_LANE;
    int BLOCK_M_PER_WARP, M_ATOMS_PER_WARP, REGS_PER_LANE, SOFTMAX_STATE_SLOTS;

    /* ---- kernel name + attrs (lines 842-847) ---- */
    const char* name; /* spec.kernel_name()             */

    /* ---- kernel params (Values, lines 849-893) ---- */
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

    /* ---- grid ids + wave decomposition ---- */
    rocke_value_t* q_block_global_idx;
    rocke_value_t* kv_head_idx;
    rocke_value_t* tid;
    rocke_value_t* lane;
    rocke_value_t* wave_id; /* NUM_WARPS>1 only               */
    rocke_value_t* wave_row_base;

    /* ---- seq lookup + Q-block geometry ---- */
    rocke_value_t* seq_idx;
    rocke_value_t* cu_q_start;
    rocke_value_t* cu_q_stop;
    rocke_value_t* cur_batch_q_len;
    rocke_value_t* q_block_start_idx;
    rocke_value_t* q_block_local_idx;
    rocke_value_t* seq_len;
    rocke_value_t* context_len;
    rocke_value_t* qb_start_pos;

    /* ---- Acc_lds stripe geometry (lines 935-960) ---- */
    int OUT_STRIPE_COLS, OUT_STRIPES;
    /* 32x32 epilogue stripe geometry (lines 3658-3717). */
    int OUT_ROW_BASE32, OUT_VEC32;

    /* ---- LDS layout: dtypes, buffering, sizes (lines 930-1124) ---- */
    const rocke_type_t* K_LDS_DTYPE;
    const rocke_type_t* V_LDS_DTYPE;
    const rocke_type_t* P_LDS_DTYPE;
    int Q_BYTES;
    int K_LDS_ELEM_BYTES, K_BUF_BYTES, K_BUFS, K_TOTAL_BYTES;
    bool Q_DIRECT_GLOBAL, Q_ALIAS_K, Q_USES_DUAL_SLOT;
    int V_BUFS;
    /* V_lds bank-deconflict swizzle */
    bool SWIZZLE_VLDS;
    int V_LDS_PAD, V_LDS_STRIDE, V_GROUP_STRIDE, V_GROUP_SHIFT, V_ROWS_PER_CALL;
    int N_STRIPES; /* fp8-PV V_lds stripe count      */
    int V_N_SLOTS; /* swizzled flat V_lds slot count */
    int P_LDS_PAD;

    /* ---- smem handles (lines 1038-1124) ---- */
    rocke_value_t* K_lds;
    rocke_value_t* V_lds;
    rocke_value_t* P_lds; /* NULL on the register-P path    */
    rocke_value_t* Q_lds; /* NULL when Q aliases K_lds       */
    rocke_value_t* Acc_lds;
    rocke_value_t* K_fp8_lds; /* fp8 async staging (NULL else)   */
    rocke_value_t* V_fp8_lds;

    /* ---- PV transpose-LDS reader (CK-Tile ds_read_b64_tr_b16/b8) ---- *
     * The gfx950 wide path reads the PV V B-operand via TransposeLdsReader bound
     * once at line 1128: TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, lane). The
     * unbound descriptor (compile-time) plus the bound SSA values are cached here
     * and reused for every PV atom. This REPLACES gfx942's _strided_v_b_operand /
     * cfv / cfvst families entirely. */
    rocke_transpose_lds_reader_t pv_tr_reader_desc;
    rocke_bound_transpose_lds_reader_t* pv_tr_reader; /* bound view (line 1128)         */

    /* Per-warp lane decomposition SSA values, emitted once in the loop-bounds
     * phase and reused by every consumer (QK / mask / softmax / PV / epilogue). */
    rocke_value_t* lane_rg_v;
    rocke_value_t* lane_col_v;
    rocke_value_t* lane_half32_v;
    rocke_value_t* lane_col32_v;
    rocke_value_t* lane_col_div4_v;
    rocke_value_t* lane_col_mod4_v;
    rocke_value_t* lane_rg_is0_v;
    rocke_value_t* lane_rg_is1_v;
    rocke_value_t* lane_rg_is2_v;

    /* ---- SSA constants ---- */
    rocke_value_t* c0;
    rocke_value_t* zero_f;
    rocke_value_t* neg_inf_v; /* b.const_f32(-inf)               */
    rocke_value_t* one_f_v; /* b.const_f32(1.0)                */
    rocke_value_t* rcp_ln2_v; /* b.const_f32(1.4426950408889634) */
    rocke_value_t* qk_scale_v; /* derived from scale_p            */
    rocke_value_t* pv_fp8_scale_v; /* fdiv(v_scale, 240) when FP8_MFMA_PV (line 1149) */
    rocke_value_t* sw_const_v; /* b.const_i32(SLIDING_WINDOW)     */

    /* ---- paged-KV byte descriptor (full transform DAG, lines 1163-1630) ---- */
    rocke_tensor_descriptor_t* q_desc; /* output/query [token,head,dim]   */
    rocke_tensor_descriptor_t* kv_desc; /* paged K/V byte descriptor       */
    rocke_value_t* seq_base; /* to_sgpr_u32(seq_idx*bt_stride)  */
    rocke_value_t* block_table_max_idx; /* to_sgpr_u32(num_seqs*bt_stride) */
    rocke_value_t* kv_block_bytes_c_v; /* const BS*NUM_KV*HD*KV_BYTES     */
    rocke_value_t* lane_half_base_v; /* tid * KV_HALVES_PER_LANE        */
    rocke_value_t* zero_soff_v; /* const_i32(0) soffset            */
    rocke_value_t* wave_lds_off_i64_v; /* wave_lds_offset_i64             */
    rocke_value_t* v_wave_lds_off_i64_v; /* v_wave_lds_offset_i64           */
    rocke_value_t* K_lds_addr_v; /* ptrtoint K_lds                  */
    rocke_value_t* V_lds_addr_v; /* ptrtoint V_lds                  */
    /* fp8 async loader (KV_FP8 && use_fp8_mfma_qk). Emitted once in the
     * preloop (Python lines 1785-1796) and reused by the issue closures. */
    /* q_gather's locally-recomputed lane_half (= div(lane,32), Python 2328).
     * The non-transposed 32x32 QK (Python 2908) reads THIS value, distinct from
     * the preloop lane_half32 (1255) the transposed path uses. */
    rocke_value_t* lane_half_qg_v;
    /* transposed PV: v_buf=const(0), use_hi=cmp_eq(lane_half32,1). Python emits
     * both ONCE before the per-N apply loop (lines 3231-3232); cache + reuse. */
    rocke_value_t* pv_v_buf_v;
    rocke_value_t* pv_use_hi_v;
    rocke_value_t* lane_fp8_base_v; /* tid * FP8_ELEMS_PER_LANE        */
    rocke_value_t* wave_fp8_off_i64_v; /* wave_fp8_offset_i64             */
    rocke_value_t* K_fp8_lds_addr_v; /* ptrtoint K_fp8_lds              */
    rocke_value_t* V_fp8_lds_addr_v; /* ptrtoint V_fp8_lds              */
    rocke_value_t* k_rsrc; /* make_buffer_resource(K)         */
    rocke_value_t* v_rsrc; /* make_buffer_resource(V)         */
    rocke_value_t* out_rsrc;

    /* ---- per-lane Q VGPR gather ---- *
     * The Q MFMA A-operand registers (legacy 16x16 gather q_regs; the 32x32x16 /
     * direct-global gather additionally fills q32_regs). The active list is
     * q_regs (count q_regs_count). */
    rocke_value_t* q_regs[ROCKE_GFX950_ATTN2D_MAX_K_ITERS * ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    int q_regs_count;
    rocke_value_t* q32_regs[ROCKE_GFX950_ATTN2D_MAX_K_ITERS];
    int q32_regs_count;

    /* ---- KV-loop bounds + step (lines 1981-2006 analogue) ---- */
    rocke_value_t* tile_start;
    rocke_value_t* tile_end;
    rocke_value_t* max_seq_prefix_len_v;
    rocke_value_t* kv_step;

    /* ---- online-softmax iter-args ---- *
     * The scf_for_iter carry: [m_0,l_0, ..., m_{S-1},l_{S-1}, acc_0..acc_{A-1},
     * cur_buf]. iter_args holds the initial carry. ACC indexing: acc lives at
     * iter_args[ml_count + n*ACC_M_ATOMS + atom]; the _acc_idx helper resolves
     * that. The trailing cur_buf is appended by the preloop-prefetch phase. */
    rocke_value_t* iter_args[ROCKE_GFX950_ATTN2D_MAX_ITER_ARGS];
    /* Per-slot carry name WITHOUT leading '%' ("m0","l0",...,"acc0"/"acc0a0",
     * "cur_buf"). Emitted verbatim as the loop's phi-node names for byte-identity
     * with Python iter_args tuple names. Each entry points into iter_args_name_buf. */
    const char* iter_args_names[ROCKE_GFX950_ATTN2D_MAX_ITER_ARGS];
    char iter_args_name_buf[ROCKE_GFX950_ATTN2D_MAX_ITER_ARGS][32]; /* "acc%da%d" worst case = 27 */
    int iter_args_count;
    int ml_count; /* 2 * SOFTMAX_STATE_SLOTS         */
    int ACC_N_TILES; /* PV_N_TILES (epilogue alias)     */
    int ACC_M_ATOMS; /* M_ATOMS_PER_WARP                */

    /* ---- LICM-hoisted per-reg invariants (transposed mask-once path) ---- *
     * Per-(reg) row/pos/head/mask invariants hoisted out of the KV-loop body.
     * Indexed by lane register slot 0..REGS_PER_LANE-1. */
    rocke_value_t* hoist_in_warp_row[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_state_row[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_q_pos[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_q_head[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* hoist_row_mask[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];

    /* ---- transposed-32x32 invariant-hoist state (Python 2446-2480) ---- *
     * Computed once in the LICM hoist when TRANSPOSED_INVARIANT_HOIST is set;
     * NULL otherwise. The transposed-softmax loop body reuses these instead of
     * recomputing per KV tile. */
    rocke_value_t* st_qp_hoist;
    rocke_value_t* st_qh_hoist;
    rocke_value_t* st_row_ok_hoist;
    rocke_value_t* st_causal_lim_hoist;
    rocke_value_t* st_alibi_slope_hoist;

    /* ---- KV-loop body live carry (set per _emit_kv_body invocation) ---- *
     * The current loop iter var + the unpacked m/l/acc carry the inner QK/mask/
     * softmax/PV closures read and rewrite. ``skip_mask`` is the per-phase flag
     * (no-SW transposed split runs a full-tile skip_mask=True phase then a
     * boundary phase). out_carry receives the rewritten carry for the next iter. */
    rocke_value_t* kv_tile_iv;
    rocke_value_t* cur_buf;
    rocke_value_t* nxt_buf_v; /* 1 - cur_buf (or cur_buf single-buffered) */
    bool skip_mask;
    rocke_value_t* m_cur[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* l_cur[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* acc_cur[ROCKE_GFX950_ATTN2D_MAX_ACCS];
    rocke_value_t* out_carry[ROCKE_GFX950_ATTN2D_MAX_ITER_ARGS];
    int out_carry_count;

    /* ---- final results read by the epilogue (lines 3585-3818) ---- */
    rocke_value_t* l_final[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* acc_final[ROCKE_GFX950_ATTN2D_MAX_ACCS];
    rocke_value_t* rcp_l[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
    rocke_value_t* l_nonzero[ROCKE_GFX950_ATTN2D_MAX_REGS_PER_LANE];
} rocke_gfx950_attn2d_build_ctx_t;

/* ===================================================================== *
 *  DRIVER-INTERNAL ctx population (the build prologue, lines 711-1912).
 *
 *  Splitting the long prologue out of the public entry keeps the glue TU small.
 *  Reproduces require_tiled_attention_arch(arch) (748) -> dtype gate (750) ->
 *  WIDE-atom select (813-814) -> the whole ALL-CAPS config derivation -> kernel /
 *  param decls -> grid / seq-idx / Q-block geometry -> LDS layout + smem_alloc ->
 *  PV TransposeLdsReader bind (1128) -> SSA constants. On any ValueError /
 *  NotImplementedError / arch reject it sets the builder error and returns false.
 *  After this returns true the LDS/descriptor/Q-gather/loop phases can run against
 *  the fully-populated ctx.
 * ===================================================================== */
bool rocke_gfx950_attn2d_build_ctx_init(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                        rocke_ir_builder_t* b,
                                        const rocke_attention_tiled_2d_spec_t* spec,
                                        const char* arch);

/* ===================================================================== *
 *  PHASE FUNCTIONS -- one per Python closure / build section.
 *  Each reads/writes only ctx (and the builder it carries) and emits IR in the
 *  byte-identical Python order.
 * ===================================================================== */

/* ----- per-lane row map + bit helpers (lines 1270-1290) ----- */
rocke_value_t* rocke_gfx950_attn2d_in_warp_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r);
rocke_value_t* rocke_gfx950_attn2d_state_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r);
rocke_value_t*
    rocke_gfx950_attn2d_bit2(rocke_gfx950_attn2d_build_ctx_t* ctx, rocke_value_t* v, int bit);
rocke_value_t* rocke_gfx950_attn2d_select_lane_rg(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* v0,
                                                  rocke_value_t* v1,
                                                  rocke_value_t* v2,
                                                  rocke_value_t* v3);

/* ----- P permute / pack helpers (lines 1292-1384) ----- *
 * _permute_p_c_to_a16: C-distribution P (REGS_PER_LANE f32) -> A16 layout list
 * (gfx950 wide 32x32 C-to-A xor shuffles, lines 1292-1351); _pack_p_a16 /
 * _pack_p_a32 pack to the MFMA A operand vector(s). */
void rocke_gfx950_attn2d_permute_p_c_to_a16(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* const* p_regs_f32,
                                            int n,
                                            rocke_value_t** out,
                                            int* out_n);
rocke_value_t* rocke_gfx950_attn2d_pack_p_a16(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* const* p_regs_f32,
                                              int n);
rocke_value_t* rocke_gfx950_attn2d_pack_p_a32(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* const* p_regs0_f32,
                                              rocke_value_t* const* p_regs1_f32,
                                              int n);

/* ----- accumulator index helpers (lines 1386, 2505, 3605) ----- */
int rocke_gfx950_attn2d_acc_idx(const rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom);
rocke_value_t* rocke_gfx950_attn2d_acc_get(rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom);
rocke_value_t*
    rocke_gfx950_attn2d_acc_final_get(rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom);

/* ----- fast paged-KV byte descriptor closures (lines 1550-1630) ----- */
void rocke_gfx950_attn2d_fast_paged_kv_blocks(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t** out_block0,
                                              rocke_value_t** out_block1);
rocke_value_t* rocke_gfx950_attn2d_fast_paged_kv_voff(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                      int call,
                                                      rocke_value_t* block0,
                                                      rocke_value_t* block1);

/* ----- pre-loop: buffer descriptors + tile-0 prefetch (lines 1163-1630) ----- */
void rocke_gfx950_attn2d_emit_preloop(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* ----- K/V async-DMA loaders (lines 1631-2228) ----- */
void rocke_gfx950_attn2d_issue_k_load_runtime(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx);
void rocke_gfx950_attn2d_issue_v_load_runtime(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* kv_tile_idx,
                                              rocke_value_t* buf_idx);
void rocke_gfx950_attn2d_issue_k(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx);
void rocke_gfx950_attn2d_issue_v(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                 rocke_value_t* tile_idx,
                                 rocke_value_t* buf_idx);
/* _read_k8_mfma_operand: per-lane K8 MFMA B operand (line 2229). */
rocke_value_t* rocke_gfx950_attn2d_read_k8_mfma_operand(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                        rocke_value_t* buf_idx,
                                                        rocke_value_t* k_row,
                                                        rocke_value_t* k_off);

/* ----- fp8 K/V cache loaders (lines 1798-2182) ----- */
/* fp8 async-loader CTA-invariant scalars, emitted once in the preloop
 * (Python lines 1785-1796), gated by (KV_FP8 && use_fp8_mfma_qk). */
void rocke_gfx950_attn2d_emit_fp8_async_preloop_setup(rocke_gfx950_attn2d_build_ctx_t* ctx);
void rocke_gfx950_attn2d_issue_kv_fp8_async_load(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx,
                                                 rocke_value_t* buf_idx);
void rocke_gfx950_attn2d_dequant_fp8_lds_to_bf16(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* src,
                                                 rocke_value_t* dst,
                                                 rocke_value_t* scale,
                                                 rocke_value_t* buf_idx);
/* Python _issue_fp8_dequant_loads(kv_tile_idx, buf_idx, lds_token). is_v selects
 * the "V" slab (value src + v_scale + V_lds); false selects "K". */
void rocke_gfx950_attn2d_issue_fp8_dequant_loads(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx,
                                                 rocke_value_t* buf_idx,
                                                 bool is_v);
void rocke_gfx950_attn2d_issue_k_fp8_mfma_async(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                rocke_value_t* kv_tile_idx,
                                                rocke_value_t* buf_idx);
void rocke_gfx950_attn2d_issue_v_fp8_mfma_async(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                rocke_value_t* kv_tile_idx);
void rocke_gfx950_attn2d_issue_v_fp8_mfma_stripe(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                 rocke_value_t* kv_tile_idx);

/* ----- Q -> LDS cooperative stage + per-lane Q -> VGPR gather ----- *
 * emit_q_load stages Q[BLOCK_M, HD] global -> LDS; emit_q_gather fills
 * ctx->q_regs (+ ctx->q32_regs on the 32x32 path) from the staged Q (or
 * direct-from-global on Q_DIRECT_GLOBAL). Run once before the KV-loop. */
void rocke_gfx950_attn2d_emit_q_load(rocke_gfx950_attn2d_build_ctx_t* ctx);
void rocke_gfx950_attn2d_emit_q_gather(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* ----- LICM hoist of per-reg invariants (transposed mask-once path) ----- *
 * Fills ctx->hoist_* before the loop. */
void rocke_gfx950_attn2d_emit_licm_hoist(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* ----- KV-loop bounds + online-softmax iter-arg carry inits ----- *
 * Ports the tile_start/tile_end (via max_seq_prefix_len) and the m/l/acc init
 * that seeds the scf_for_iter carry, plus the per-warp lane-decomposition SSA
 * (ctx->lane_*_v). Fills ctx->tile_start/tile_end and ctx->iter_args[0 ..
 * ml_count + num_accs) with their names; the trailing cur_buf carry is appended
 * by emit_preloop_prefetch and kv_step recorded by emit_kv_step. */
void rocke_gfx950_attn2d_emit_loop_bounds_and_inits(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* ----- tile-0 prefetch + cur_buf carry ----- *
 * Emits _issue_k(tile_start, 0) and appends the ("cur_buf", const_i32(0)) carry
 * to iter_args. Runs after emit_q_gather, before emit_licm_hoist. */
void rocke_gfx950_attn2d_emit_preloop_prefetch(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* ----- kv_step const ----- *
 * Emitted after emit_licm_hoist to match Python const emission order. */
void rocke_gfx950_attn2d_emit_kv_step(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* ----- drive the scf_for_iter KV loop (line 3581) ----- *
 * Builds the loop over [tile_start, tile_end) step kv_step with the named
 * iter_args carry, enters the body region, unpacks the carry into
 * ctx->m_cur/l_cur/acc_cur + ctx->cur_buf + sets ctx->kv_tile_iv, runs
 * emit_kv_body, then leaves. Returns the loop handle whose .op->results are the
 * rewritten carry the epilogue consumes. */
rocke_for_t rocke_gfx950_attn2d_drive_kv_loop(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* ----- KV-loop body (_emit_kv_body, lines 2494-3578) ----- *
 * One full KV-tile body: QK MFMA + softcap/mask/softmax + PV MFMA + acc scale,
 * threading the online-softmax carry. Reads ctx->kv_tile_iv / ctx->skip_mask /
 * ctx->m_cur / ctx->l_cur / ctx->acc_cur and writes ctx->out_carry. */
void rocke_gfx950_attn2d_emit_kv_body(rocke_gfx950_attn2d_build_ctx_t* ctx);

/* Re-entrancy reset for the file-scope REGISTER_PV scratch (p_regs_f32_buf) that
 * emit_kv_body uses to carry the in-register P groups from the softmax sub-block
 * to the PV sub-block. The slots hold builder-bound values that dangle once a
 * build's arena is freed; call this at the build entry so each build starts from
 * clean NULL. */
void rocke_gfx950_attn2d_reset_softmax_scratch(void);

/* Inbound softmax-derived state for the PV bucket. The QK/softmax front half
 * fills this and hands it to rocke_gfx950_attn2d_emit_pv_bucket; the PV bucket runs
 * acc *= alpha; acc += P @ V (via the wide atom + pv_tr_reader transpose reads)
 * and emits the scf_yield carry. */
typedef struct rocke_gfx950_attn2d_pv_inputs
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
    /* clamped next tile index for the deferred K single-buffer prefetch. */
    rocke_value_t* safe_next_tile;
} rocke_gfx950_attn2d_pv_inputs_t;

/* PV MFMA + carry yield bucket (the back half of _emit_kv_body). */
void rocke_gfx950_attn2d_emit_pv_bucket(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                        const rocke_gfx950_attn2d_pv_inputs_t* in);

/* ----- inner PV closure of _emit_kv_body (line 3234) ----- *
 * _apply_transposed_pv_regs(acc32, n, p_regs) -> new acc32 (register-P^T PV via
 * the wide atom + pv_tr_reader transpose reads). */
rocke_value_t* rocke_gfx950_attn2d_apply_transposed_pv_regs(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                            rocke_value_t* acc32,
                                                            int n,
                                                            rocke_value_t* const* p_regs,
                                                            int p_count);

/* ----- epilogue (lines 3585-3818) ----- *
 * Drains outstanding async copies, reads the loop results into ctx->l_final /
 * acc_final / rcp_l / l_nonzero, then emits the normalize + store: the 32x32
 * direct/coalesced Acc_lds scalar store (3658-3717) or the narrow striped vec8
 * cooperative store (3717-3818). Returns b->kernel on success or NULL with the
 * builder error set. */
rocke_kernel_def_t* rocke_gfx950_attn2d_emit_epilogue(rocke_gfx950_attn2d_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_INTERNAL_H */
