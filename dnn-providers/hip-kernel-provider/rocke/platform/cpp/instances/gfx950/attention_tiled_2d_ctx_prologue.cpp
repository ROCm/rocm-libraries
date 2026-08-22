// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_ctx_prologue.c --
 * the BUILD-CONTEXT PROLOGUE bucket of the chunked C99 port of
 * rocke/instances/gfx950/attention_tiled_2d.py (arch gfx950, WIDE-K atoms).
 *
 * Scope (this part-file):
 *   - rocke_gfx950_attn2d_build_ctx_init  (Python build body lines 711-1135):
 *       require_tiled_attention_arch(arch) (748) + the fp16/bf16 dtype gate (750)
 *       + the whole ALL-CAPS config derivation (HD/T/BS/BLOCK_M + QK/PV wide-atom
 *       geometry + the 32x32 / transposed predicate aliases + fp8 K/V cache
 *       predicates), kernel name/attrs + param decls, grid ids + wave
 *       decomposition (feeding emit_preloop / the descriptor DAG), binary-search
 *       seq_idx + cu_q bounds + Q-block geometry + the padding-block early-return,
 *       the LDS-layout decisions (dtypes / buffering / Acc stripe sizing) and
 *       every smem_alloc handle (K_lds/V_lds/P_lds/Q_lds/Acc_lds + the fp8 staging
 *       slabs, lines 930-1124), the PV TransposeLdsReader bind (line 1128), and
 *       the SSA constants (neg_inf/zero/one/rcp_ln2/qk_scale/sw_const, 1132-1136).
 *   - the pure per-lane row/bit closures _in_warp_row / _state_row / _bit2 /
 *       _select_lane_rg (lines 1270-1290).
 *   - the accumulator-index helpers _acc_idx / _acc_get / _acc_final_get
 *       (lines 1386 / 2505 / 3605).
 *
 * The Python closures capture the enclosing-function locals; here they read the
 * shared rocke_gfx950_attn2d_build_ctx_t. The per-warp lane-decomposition scalars
 * (lane_rg / lane_col / lane_col_mod4 / lane_col_div4 / lane_rg_is*) are ctx
 * fields filled by the (peer) emit_loop_bounds_and_inits phase; the row-map
 * closures reuse them when present and otherwise recompute from ctx->lane (the
 * recomputed IR is DCE-identical to Python's shared SSA values).
 *
 * Every IR node is arena-owned (ctx->b->arena). Error model: on any
 * NotImplementedError / arch reject the builder's sticky error is set and
 * ctx_init returns false.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_helper_rocke.helpers.attention.h" /* rocke_binary_search_seq_idx */
#include "rocke/instance_gfx950_attention_tiled_2d_internal.h"

/* ---------------------------------------------------------------- helpers */

static bool rocke_g950a2d_streq(const char* a, const char* c)
{
    if(a == c)
        return true;
    if(!a || !c)
        return false;
    return strcmp(a, c) == 0;
}

/* require_tiled_attention_arch(arch) for the gfx950 WIDE-K path. Faithful inline
 * port of instances/common/attention_arch.require_tiled_attention_arch: a target
 * is admitted when it carries the wide-K 16x16x32 f16 atom (or ds_read_tr) AND
 * the LDS transpose reads (ds_read_b64_tr_b16). Returns true on accept; on reject
 * latches a NotImplementedError on b. */
static bool rocke_g950a2d_require_tiled_attention_arch(rocke_ir_builder_t* b, const char* arch)
{
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        if(b)
        {
            b->status = ROCKE_ERR_NOTIMPL;
            snprintf(b->err, ROCKE_ERR_MSG_CAP, "unknown gfx target: %s", arch ? arch : "(null)");
        }
        return false;
    }
    const rocke_arch_mma_catalog_t* mma = rocke_archtarget_mma(target);
    bool wide_k = rocke_mma_catalog_has_shape(mma, "mma", "f16", "f16", "fp32", 16, 16, 32)
                  || target->memory.has_ds_read_tr;
    if(wide_k && target->memory.has_ds_read_tr)
        return true;
    if(b)
    {
        b->status = ROCKE_ERR_NOTIMPL;
        if(!wide_k)
            snprintf(b->err,
                     ROCKE_ERR_MSG_CAP,
                     "tiled attention requires the wide-K MFMA atoms "
                     "(mfma_f32_16x16x32 / mfma_f32_32x32x16, gfx950); absent on %s",
                     arch ? arch : "(null)");
        else
            snprintf(b->err,
                     ROCKE_ERR_MSG_CAP,
                     "tiled attention requires LDS transpose reads (ds_read_b64_tr_b16) "
                     "for the wide-K path, absent on %s",
                     arch ? arch : "(null)");
    }
    return false;
}

/* ===================================================================== *
 *  rocke_gfx950_attn2d_build_ctx_init -- the prologue (lines 711-1135)
 * ===================================================================== */
bool rocke_gfx950_attn2d_build_ctx_init(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                        rocke_ir_builder_t* b,
                                        const rocke_attention_tiled_2d_spec_t* spec,
                                        const char* arch)
{
    if(ctx == NULL || b == NULL || spec == NULL)
        return false;
    memset(ctx, 0, sizeof(*ctx));

    if(arch == NULL)
        arch = "gfx950"; /* Python default */

    ctx->b = b;
    ctx->spec = spec;
    ctx->arch = arch;

    /* ---- require_tiled_attention_arch(arch) (line 748) ---- */
    if(!rocke_g950a2d_require_tiled_attention_arch(b, arch))
        return false;

    /* ---- dtype gate (lines 750-751) ---- */
    if(!(rocke_g950a2d_streq(spec->dtype, "fp16") || rocke_g950a2d_streq(spec->dtype, "bf16")))
    {
        b->status = ROCKE_ERR_NOTIMPL;
        snprintf(b->err, ROCKE_ERR_MSG_CAP, "tiled 2D kernel supports fp16/bf16");
        return false;
    }
    const rocke_type_t* dtype = rocke_attention_tiled_2d_spec_dtype_ir(spec);
    ctx->dtype = dtype;

    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    ctx->target = target;

    /* ---- ALL-CAPS geometry constants (lines 754-797) ---- */
    const int HD = spec->head_size;
    const int T = rocke_attention_tiled_2d_spec_tile_size_eff(spec);
    const int BS = spec->block_size;
    const int N_BLOCKS_PER_TILE = rocke_attention_tiled_2d_spec_n_blocks_per_tile(spec);
    const int BLOCK_M = rocke_attention_tiled_2d_spec_block_m(spec);
    const int BLOCK_Q = rocke_attention_tiled_2d_spec_block_q(spec);
    const int NQK = rocke_attention_tiled_2d_spec_num_queries_per_kv(spec);
    const int NUM_KV = spec->num_kv_heads;
    const int NUM_QH = spec->num_query_heads;
    const int SLIDING_WINDOW = spec->sliding_window;

    ctx->HD = HD;
    ctx->T = T;
    ctx->BS = BS;
    ctx->N_BLOCKS_PER_TILE = N_BLOCKS_PER_TILE;
    ctx->BLOCK_M = BLOCK_M;
    ctx->BLOCK_Q = BLOCK_Q;
    ctx->NQK = NQK;
    ctx->NUM_KV = NUM_KV;
    ctx->NUM_QH = NUM_QH;
    ctx->SLIDING_WINDOW = SLIDING_WINDOW;
    ctx->USE_SOFTCAP = spec->has_softcap;
    ctx->USE_SINKS = spec->use_sinks;
    ctx->USE_ALIBI = spec->use_alibi;
    ctx->USE_QQ_BIAS = spec->use_qq_bias;

    ctx->TRANSPOSED_SCALAR_STATE = spec->use_transposed_scalar_state;
    ctx->TRANSPOSED_INVARIANT_HOIST = spec->use_transposed_invariant_hoist;
    ctx->TRANSPOSED_MASK_ONCE = spec->use_transposed_mask_once;
    ctx->TRANSPOSED_HALF_LOCAL_PV = spec->use_transposed_half_local_pv;
    ctx->SKIP_LEGACY_QREG = spec->use_mfma32_skip_legacy_qreg;
    ctx->TRANSPOSED_MASK_LIMIT = spec->use_transposed_mask_limit;
    ctx->GROUPED_KV2 = spec->use_grouped_kv2_softmax;
    ctx->FAST_PAGED_KV_DESC = spec->use_fast_paged_kv_desc;
    ctx->I64_KV_ADDR = spec->use_i64_kv_addr;
    ctx->EARLY_V_SCHEDULE = spec->use_early_v_schedule;
    ctx->AGPR_ALLOC_ZERO = spec->use_agpr_alloc_zero;

    /* ---- fp8 K/V cache predicates (lines 783-797) ---- */
    const bool KV_FP8 = rocke_g950a2d_streq(spec->kv_storage_dtype, "fp8e4m3");
    const bool FP8_MFMA_QK = KV_FP8 && spec->use_fp8_mfma_qk;
    const bool FP8_MFMA_PV = KV_FP8 && spec->use_fp8_mfma_pv;
    /* FP8_NATIVE_QK = False (documented LOSE-LOSE dead path, line 793). */
    ctx->KV_FP8 = KV_FP8;
    ctx->FP8_MFMA_QK = FP8_MFMA_QK;
    ctx->FP8_MFMA_PV = FP8_MFMA_PV;

    ctx->REGISTER_PV = spec->use_register_pv;
    ctx->TRANSPOSED_QK_32X32 = spec->use_transposed_qk_32x32;

    const int KV_BYTES = KV_FP8 ? 1 : 2;
    ctx->KV_BYTES = KV_BYTES;
    ctx->kv_io_dtype = KV_FP8 ? rocke_fp8e4m3() : dtype;

    /* ---- QK/PV wide-atom MFMA geometry (lines 799-819) ---- */
    const bool USE_MFMA_32X32 = spec->use_mfma_32x32;
    ctx->USE_MFMA_32X32 = USE_MFMA_32X32;

    const int QK_MFMA_N = USE_MFMA_32X32 ? 32 : 16; /* MFMA_N == 16 */
    const int QK_K_STEP = USE_MFMA_32X32 ? 16 : 32;
    const int PV_K_STEP = (T % 32 == 0) ? 32 : 16;
    const int QK_K_ITERS = HD / QK_K_STEP;
    const int QK_N_TILES = T / QK_MFMA_N;
    const int PV_K_ITERS = T / PV_K_STEP;
    const int PV_N_TILES = HD / 16; /* MFMA_N */
    ctx->QK_MFMA_N = QK_MFMA_N;
    ctx->QK_K_STEP = QK_K_STEP;
    ctx->PV_K_STEP = PV_K_STEP;
    ctx->QK_K_ITERS = QK_K_ITERS;
    ctx->QK_N_TILES = QK_N_TILES;
    ctx->PV_K_ITERS = PV_K_ITERS;
    ctx->PV_N_TILES = PV_N_TILES;

    /* ---- threads / wave (lines 821-839) ---- */
    const int NUM_WARPS = spec->num_warps;
    const int WAVE = 64;
    const int THREADS = NUM_WARPS * WAVE;
    ctx->NUM_WARPS = NUM_WARPS;
    ctx->WAVE = WAVE;
    ctx->THREADS = THREADS;

    const int BLOCK_M_PER_WARP = spec->block_m_per_warp;
    const int M_ATOMS_PER_WARP = BLOCK_M_PER_WARP / 16; /* MFMA_M */
    const int REGS_PER_LANE = rocke_attention_tiled_2d_spec_regs_per_lane(spec);
    const int SOFTMAX_STATE_SLOTS
        = (USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32 && ctx->TRANSPOSED_SCALAR_STATE)
              ? 1
              : REGS_PER_LANE;
    ctx->BLOCK_M_PER_WARP = BLOCK_M_PER_WARP;
    ctx->M_ATOMS_PER_WARP = M_ATOMS_PER_WARP;
    ctx->REGS_PER_LANE = REGS_PER_LANE;
    ctx->SOFTMAX_STATE_SLOTS = SOFTMAX_STATE_SLOTS;

    /* ---- kernel name + attrs (lines 841-847) ---- *
     * The driver inited b with spec.kernel_name(); mirror that name. */
    ctx->name = (b->kernel != NULL) ? b->kernel->name : NULL;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->has_waves_per_eu)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    if(spec->use_agpr_alloc_zero)
    {
        /* (0, 0) -- serialized as the bare-int list l:[ i:0, i:0 ] */
        const int64_t agpr_zero[2] = {0, 0};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "agpr_alloc", agpr_zero, 2);
    }

    /* ---- parameter declarations (lines 849-889) ---- */
    {
        rocke_param_opts_t o;
        const rocke_type_t* ptr_dt = rocke_ptr_type(b, dtype, "global");
        const rocke_type_t* ptr_kvio = rocke_ptr_type(b, ctx->kv_io_dtype, "global");
        const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");
        const rocke_type_t* ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");

        memset(&o, 0, sizeof(o));
        o.noalias = true;
        o.noalias_set = true;
        o.writeonly = true;
        o.writeonly_set = true;
        o.align = 16;
        o.align_set = true;
        ctx->output = rocke_b_param(b, "output_ptr", ptr_dt, &o);

        memset(&o, 0, sizeof(o));
        o.noalias = true;
        o.noalias_set = true;
        o.readonly = true;
        o.readonly_set = true;
        o.align = 16;
        o.align_set = true;
        ctx->query = rocke_b_param(b, "query_ptr", ptr_dt, &o);
        ctx->key = rocke_b_param(b, "key_cache_ptr", ptr_kvio, &o);
        ctx->value = rocke_b_param(b, "value_cache_ptr", ptr_kvio, &o);

        /* sink_ptr: readonly + align 16 (no noalias), Python line 870. */
        memset(&o, 0, sizeof(o));
        o.readonly = true;
        o.readonly_set = true;
        o.align = 16;
        o.align_set = true;
        ctx->sinks = rocke_b_param(b, "sink_ptr", ptr_dt, &o);

        memset(&o, 0, sizeof(o));
        o.readonly = true;
        o.readonly_set = true;
        o.align = 4;
        o.align_set = true;
        ctx->block_tables = rocke_b_param(b, "block_tables_ptr", ptr_i32, &o);
        ctx->seq_lens = rocke_b_param(b, "seq_lens_ptr", ptr_i32, &o);
        ctx->alibi_slopes_ptr = rocke_b_param(b, "alibi_slopes_ptr", ptr_f32, &o);
        ctx->qq_bias_ptr = rocke_b_param(b, "qq_bias_ptr", ptr_f32, &o);
        ctx->cu_q = rocke_b_param(b, "query_start_len_ptr", ptr_i32, &o);

        ctx->scale_p = rocke_b_param(b, "scale", rocke_f32(), NULL);
        ctx->k_scale_p = rocke_b_param(b, "k_scale", rocke_f32(), NULL);
        ctx->v_scale_p = rocke_b_param(b, "v_scale", rocke_f32(), NULL);
        ctx->out_scale = rocke_b_param(b, "out_scale", rocke_f32(), NULL);
        ctx->softcap_p = rocke_b_param(b, "softcap", rocke_f32(), NULL);
        ctx->num_seqs_p = rocke_b_param(b, "num_seqs", rocke_i32(), NULL);
        ctx->bt_stride_p = rocke_b_param(b, "block_table_stride", rocke_i32(), NULL);
        ctx->qq_bias_stride0_p = rocke_b_param(b, "qq_bias_stride_0", rocke_i32(), NULL);
    }

    /* ---- grid ids + wave decomposition (lines 891-904) ---- *
     * gfx950 fixes the grid mapping: kv_head_idx = block_id_x,
     * q_block_global_idx = block_id_y (no q-major-grid branch). */
    ctx->kv_head_idx = rocke_b_block_id_x(b);
    ctx->q_block_global_idx = rocke_b_block_id_y(b);
    ctx->tid = rocke_b_thread_id_x(b);

    if(NUM_WARPS == 1)
    {
        ctx->lane = ctx->tid;
        ctx->wave_id = NULL;
        ctx->wave_row_base = rocke_b_const_i32(b, 0);
    }
    else
    {
        ctx->lane = rocke_b_mod(b, ctx->tid, rocke_b_const_i32(b, WAVE));
        ctx->wave_id = rocke_b_div(b, ctx->tid, rocke_b_const_i32(b, WAVE));
        ctx->wave_row_base = rocke_b_mul(b, ctx->wave_id, rocke_b_const_i32(b, BLOCK_M_PER_WARP));
    }

    /* ---- seq lookup + Q-block geometry (lines 906-925) ---- */
    ctx->seq_idx
        = rocke_binary_search_seq_idx(b,
                                      ctx->cu_q,
                                      ctx->q_block_global_idx,
                                      ctx->num_seqs_p,
                                      BLOCK_Q,
                                      rocke_attention_tiled_2d_spec_binary_search_iters(spec),
                                      false);
    ctx->cu_q_start = rocke_b_global_load_i32(b, ctx->cu_q, ctx->seq_idx, 0);
    ctx->cu_q_stop = rocke_b_global_load_i32(
        b, ctx->cu_q, rocke_b_add(b, ctx->seq_idx, rocke_b_const_i32(b, 1)), 0);
    ctx->cur_batch_q_len = rocke_b_sub(b, ctx->cu_q_stop, ctx->cu_q_start);
    ctx->q_block_start_idx = rocke_b_add(
        b, rocke_b_div(b, ctx->cu_q_start, rocke_b_const_i32(b, BLOCK_Q)), ctx->seq_idx);
    ctx->q_block_local_idx = rocke_b_sub(b, ctx->q_block_global_idx, ctx->q_block_start_idx);
    ctx->seq_len = rocke_b_global_load_i32(b, ctx->seq_lens, ctx->seq_idx, 0);
    ctx->context_len = rocke_b_sub(b, ctx->seq_len, ctx->cur_batch_q_len);

    ctx->qb_start_pos = rocke_b_mul(b, ctx->q_block_local_idx, rocke_b_const_i32(b, BLOCK_Q));
    {
        rocke_if_t f
            = rocke_b_scf_if(b, rocke_b_cmp_ge(b, ctx->qb_start_pos, ctx->cur_batch_q_len));
        rocke_b_region_enter(b, f.then_region);
        rocke_b_ret(b);
        rocke_b_region_leave(b);
    }

    /* ---- Acc_lds stripe geometry (lines 952-959) ---- */
    const int OUT_STRIPE_COLS = (HD <= 64) ? 32 : HD;
    const int OUT_STRIPES = HD / OUT_STRIPE_COLS;
    ctx->OUT_STRIPE_COLS = OUT_STRIPE_COLS;
    ctx->OUT_STRIPES = OUT_STRIPES;

    /* ---- LDS dtypes / buffering / sizes (lines 1021-1037) ---- */
    const rocke_type_t* K_LDS_DTYPE = FP8_MFMA_QK ? rocke_fp8e4m3() : dtype;
    const rocke_type_t* V_LDS_DTYPE = FP8_MFMA_PV ? rocke_fp8e4m3() : dtype;
    const rocke_type_t* P_LDS_DTYPE = FP8_MFMA_PV ? rocke_fp8e4m3() : dtype;
    ctx->K_LDS_DTYPE = K_LDS_DTYPE;
    ctx->V_LDS_DTYPE = V_LDS_DTYPE;
    ctx->P_LDS_DTYPE = P_LDS_DTYPE;

    const int Q_BYTES = BLOCK_M * HD * 2;
    const int K_LDS_ELEM_BYTES = rocke_type_eq(K_LDS_DTYPE, rocke_fp8e4m3()) ? 1 : 2;
    const int K_BUF_BYTES = T * HD * K_LDS_ELEM_BYTES;
    const int K_TOTAL_BYTES = 2 * K_BUF_BYTES; /* K_lds Q-alias region (2 slots worth) */
    ctx->Q_BYTES = Q_BYTES;
    ctx->K_LDS_ELEM_BYTES = K_LDS_ELEM_BYTES;
    ctx->K_BUF_BYTES = K_BUF_BYTES;
    /* K single-buffer -> 1 slot (halves K_lds so T=64 fits 2 WG/CU). */
    ctx->K_BUFS = ctx->K_SINGLE_BUFFER ? 1 : 2;
    ctx->K_TOTAL_BYTES = K_TOTAL_BYTES;

    const bool Q_ALIAS_K = rocke_type_eq(K_LDS_DTYPE, dtype) && (Q_BYTES <= K_TOTAL_BYTES);
    const bool Q_USES_DUAL_SLOT = Q_ALIAS_K && (BLOCK_M > T);
    ctx->Q_ALIAS_K = Q_ALIAS_K;
    ctx->Q_USES_DUAL_SLOT = Q_USES_DUAL_SLOT;

    /* ---- K_lds smem_alloc (line 1038) ---- */
    {
        int shp[3] = {ctx->K_BUFS, T, HD}; /* 1 slot when K single-buffer */
        ctx->K_lds = rocke_b_smem_alloc(b, K_LDS_DTYPE, shp, 3, "Klds");
    }

    /* ---- V single-buffer + V_lds smem_alloc (lines 1039-1060) ---- */
    const int V_BUFS = 1;
    ctx->V_BUFS = V_BUFS;
    if(FP8_MFMA_PV)
    {
        const int N_STRIPES = HD / 16;
        ctx->N_STRIPES = N_STRIPES;
        int shp[4] = {V_BUFS, N_STRIPES, T, 16};
        ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 4, "VldsStripe");
    }
    else
    {
        int shp[3] = {V_BUFS, T, HD};
        ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 3, "Vlds");
    }

    /* ---- P_lds (lines 1061-1076) ----
     * The transposed-32x32 path (USE_MFMA_32X32 && TRANSPOSED_QK_32X32) keeps P
     * entirely in registers: the softmax publish takes the no-op (pass) branch
     * and the PV consumer reads PT32_n from registers, so P_lds is allocated but
     * never written or read -- pure dead LDS (~18 KiB at BLOCK_M=128/T=64).
     * Dropping it is byte-identical for every non-P_lds instruction and frees
     * the occupancy-limiting LDS at HD=128. Mirrors attention_tiled_2d.py
     * P_LDS_DEAD. The legacy 16x16x32 path, the transitional non-transposed
     * 32x32 consumer, and the fp8-MFMA PV quantised-P path all still use P_lds. */
    const bool P_LDS_DEAD = USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32;
    if(!ctx->REGISTER_PV && !P_LDS_DEAD)
    {
        const int P_LDS_PAD = FP8_MFMA_PV ? 16 : 8;
        ctx->P_LDS_PAD = P_LDS_PAD;
        int shp[2] = {BLOCK_M, T + P_LDS_PAD};
        ctx->P_lds = rocke_b_smem_alloc(b, P_LDS_DTYPE, shp, 2, "Plds");
    }
    else
    {
        ctx->P_lds = NULL;
    }

    /* ---- fp8 K/V staging slabs (lines 1095-1097) ---- */
    if(KV_FP8 && spec->use_fp8_mfma_qk)
    {
        int kshp[3] = {2, T, HD};
        int vshp[3] = {1, T, HD};
        ctx->K_fp8_lds = rocke_b_smem_alloc(b, rocke_fp8e4m3(), kshp, 3, "Kfp8lds");
        ctx->V_fp8_lds = rocke_b_smem_alloc(b, rocke_fp8e4m3(), vshp, 3, "Vfp8lds");
    }

    /* ---- Q_lds (lines 1098-1104) ---- */
    if(Q_ALIAS_K)
    {
        ctx->Q_lds = ctx->K_lds; /* reuse K_lds[0] (+K_lds[1]) as Q scratch */
    }
    else
    {
        int shp[2] = {BLOCK_M, HD};
        ctx->Q_lds = rocke_b_smem_alloc(b, dtype, shp, 2, "Qlds");
    }

    /* ---- Acc_lds (line 1113) ---- */
    {
        int shp[2] = {BLOCK_M, OUT_STRIPE_COLS};
        ctx->Acc_lds = rocke_b_smem_alloc(b, dtype, shp, 2, "Aclds");
    }

    /* ---- PV TransposeLdsReader bind (line 1128) ---- *
     * TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, lane). The unbound
     * descriptor (compile-time params) plus the bound per-lane SSA view are
     * cached for reuse by every PV atom. bind() emits, in order:
     *   lane_div_16      = div(lane, 16)
     *   lane_div_4_mod_4 = mod(div(lane, 4), 4)
     *   col              = mul(mod(lane, 4), 4) */
    ctx->pv_tr_reader_desc.K = PV_K_STEP;
    ctx->pv_tr_reader_desc.M = 16;
    ctx->pv_tr_reader = rocke_transpose_lds_reader_bind(b, &ctx->pv_tr_reader_desc, ctx->lane);

    /* ---- SSA constants (lines 1132-1136) ---- *
     * Exact creation order: neg_inf, zero_f, one_f, rcp_ln2, then qk_scale.
     * Cache neg_inf / zero_f / one_f / rcp_ln2 so downstream phases reuse the
     * same SSA values instead of recreating fresh consts (which would shift the
     * SSA counter). */
    ctx->neg_inf_v = rocke_b_const_f32(b, -INFINITY); /* neg_inf (1132) */
    ctx->zero_f = rocke_b_const_f32(b, 0.0); /* zero_f  (1133) */
    ctx->one_f_v = rocke_b_const_f32(b, 1.0); /* one_f   (1134) */
    ctx->rcp_ln2_v = rocke_b_const_f32(b, 1.4426950408889634); /* rcp_ln2 (1135) */
    ctx->qk_scale_v = rocke_b_fmul(b, ctx->scale_p, ctx->rcp_ln2_v); /* qk_scale (1136) */
    /* FP8_NATIVE_QK is False -> no qk_scale *= k_scale fold (lines 1137-1143). */
    /* pv_fp8_scale = fdiv(v_scale, 240.0) when FP8_MFMA_PV (line 1149). Computed
     * here (between qk_scale and sw_const) so its SSA position matches Python; the
     * PV epilogue reuses ctx->pv_fp8_scale_v instead of recomputing it. */
    ctx->pv_fp8_scale_v
        = ctx->FP8_MFMA_PV ? rocke_b_fdiv(b, ctx->v_scale_p, rocke_b_const_f32(b, 240.0)) : NULL;

    /* sw_const = const_i32(SLIDING_WINDOW). Created once in the constants block
     * and reused by the tile-bound + mask regions (peer phases). */
    ctx->sw_const_v = rocke_b_const_i32(b, SLIDING_WINDOW);
    ctx->c0 = ctx->sw_const_v;

    /* ---- acc geometry the loop-bounds / epilogue phases alias (lines 1380-1387) ---- */
    ctx->ACC_N_TILES = USE_MFMA_32X32 ? (HD / 32) : PV_N_TILES;
    ctx->ACC_M_ATOMS = USE_MFMA_32X32 ? 1 : M_ATOMS_PER_WARP;
    ctx->ml_count = 2 * SOFTMAX_STATE_SLOTS;

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Per-lane row map + bit helpers (lines 1270-1290).
 *
 *  Reuse the cached per-warp lane-decomposition SSA values (ctx->lane_rg_v /
 *  ctx->lane_col_*_v), filled by the peer emit_loop_bounds_and_inits phase.
 *  Fall back to a fresh op only if a caller runs before that phase (the emitted
 *  IR is DCE-identical to Python's shared SSA values).
 * ===================================================================== */

static rocke_value_t* rocke_g950a2d_lane_rg(rocke_gfx950_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_rg_v != NULL)
        return ctx->lane_rg_v;
    return rocke_b_div(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 16));
}

rocke_value_t* rocke_gfx950_attn2d_in_warp_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r)
{
    /* atom_idx = r // 4 ; in_atom = r % 4
     * row = lane_rg*4 + (atom_idx*16 + in_atom) */
    int atom_idx = r / 4;
    int in_atom = r % 4;
    /* Python evaluates b.mul(lane_rg, const(4)) BEFORE the trailing const
     * (left-to-right arg order). Bind the mul to a temp so C's unspecified
     * arg-eval order does not allocate the trailing const ahead of the mul. */
    rocke_value_t* rg4
        = rocke_b_mul(ctx->b, rocke_g950a2d_lane_rg(ctx), rocke_b_const_i32(ctx->b, 4));
    return rocke_b_add(ctx->b, rg4, rocke_b_const_i32(ctx->b, atom_idx * 16 + in_atom));
}

rocke_value_t* rocke_gfx950_attn2d_state_row(rocke_gfx950_attn2d_build_ctx_t* ctx, int r)
{
    if(ctx->USE_MFMA_32X32 && !ctx->TRANSPOSED_QK_32X32)
        return rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row(ctx->b, ctx->lane, r);
    return rocke_gfx950_attn2d_in_warp_row(ctx, r);
}

rocke_value_t*
    rocke_gfx950_attn2d_bit2(rocke_gfx950_attn2d_build_ctx_t* ctx, rocke_value_t* v, int bit)
{
    /* Python: b.land(b.lshr(v, const(bit)), b.const_i32(1)) creates the lshr
     * (and its bit const) BEFORE the trailing const(1) (left-to-right). Bind the
     * lshr to a temp so C's right-to-left arg eval does not allocate const(1)
     * ahead of the lshr and shift the SSA ids. */
    rocke_value_t* sh = rocke_b_lshr(ctx->b, v, rocke_b_const_i32(ctx->b, bit));
    return rocke_b_land(ctx->b, sh, rocke_b_const_i32(ctx->b, 1));
}

rocke_value_t* rocke_gfx950_attn2d_select_lane_rg(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* v0,
                                                  rocke_value_t* v1,
                                                  rocke_value_t* v2,
                                                  rocke_value_t* v3)
{
    /* select(lane_rg_is0, v0, select(lane_rg_is1, v1, select(lane_rg_is2, v2, v3))).
     * Reuse the cached lane_rg_is* SSA values when present, else recompute. */
    rocke_value_t* is0 = ctx->lane_rg_is0_v;
    rocke_value_t* is1 = ctx->lane_rg_is1_v;
    rocke_value_t* is2 = ctx->lane_rg_is2_v;
    if(is0 == NULL || is1 == NULL || is2 == NULL)
    {
        rocke_value_t* lane_rg = rocke_g950a2d_lane_rg(ctx);
        is0 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 0));
        is1 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 1));
        is2 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 2));
    }
    return rocke_b_select(
        ctx->b, is0, v0, rocke_b_select(ctx->b, is1, v1, rocke_b_select(ctx->b, is2, v2, v3)));
}

/* ===================================================================== *
 *  Accumulator index helpers (lines 1386, 2505, 3605).
 * ===================================================================== */

int rocke_gfx950_attn2d_acc_idx(const rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom)
{
    return n * ctx->ACC_M_ATOMS + atom;
}

rocke_value_t* rocke_gfx950_attn2d_acc_get(rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom)
{
    /* Reads the unpacked loop-body acc carry (set per _emit_kv_body). */
    return ctx->acc_cur[rocke_gfx950_attn2d_acc_idx(ctx, n, atom)];
}

rocke_value_t*
    rocke_gfx950_attn2d_acc_final_get(rocke_gfx950_attn2d_build_ctx_t* ctx, int n, int atom)
{
    /* Reads the final loop-result acc (set by the epilogue read-back). */
    return ctx->acc_final[rocke_gfx950_attn2d_acc_idx(ctx, n, atom)];
}
