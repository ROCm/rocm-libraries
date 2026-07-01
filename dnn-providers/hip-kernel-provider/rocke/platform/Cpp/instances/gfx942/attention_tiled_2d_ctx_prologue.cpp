// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_gfx942_attention_tiled_2d_ctx_prologue.c --
 * the PROLOGUE / ctx-population bucket of the chunked C99 port of
 * rocke/instances/gfx942/attention_tiled_2d.py (arch gfx942).
 *
 * Implements:
 *   - rocke_gfx942_attn2d_build_ctx_init  (Python build body lines 1104-1912):
 *       require_tiled_attention_arch(arch) + dtype gate + narrow 16x16x16 atom
 *       select, the whole ALL-CAPS config derivation (HD/T/BS/BLOCK_M + geometry
 *       + the 32x32 / cfv / cfvst / k1buf umbrella predicates +
 *       ASYNC_LDS_MAX_DWORDS=1), kernel attrs + param decls, grid ids + wave
 *       decomposition, binary-search seq_idx + cu_q bounds + Q-block geometry +
 *       early-return, the LDS-layout decisions (dtypes / buffering / swizzle /
 *       transposed-V sizing) + every smem_alloc handle, the transpose-LDS lane
 *       formula params, and the SSA constants.
 *   - the pure row-map / bit / acc-index helpers (in_warp_row, state_row, bit2,
 *       select_lane_rg, acc_idx, acc_get, acc_final_get).
 *
 * The closures in Python capture the enclosing-function locals; here they read
 * the shared rocke_gfx942_attn2d_build_ctx_t. The per-warp lane decomposition
 * scalars (lane_rg, lane_col, lane_col32, lane_col_div4, lane_col_mod4,
 * lane_rg_is*) are NOT ctx fields, so the row-map helpers recompute them from
 * ctx->lane on demand -- the IR these emit is dead-code-eliminated identically
 * to Python's CSE of the shared SSA values (same op_id sequence per call site).
 *
 * Every IR node is arena-owned (ctx->b->arena). Error model: on any
 * NotImplementedError / ValueError / arch reject the builder's sticky error is
 * set and ctx_init returns false.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_helper_rocke.helpers.attention.h" /* rocke_binary_search_seq_idx */
#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"

/* ---------------------------------------------------------------- helpers */

static bool rocke_a2d_streq(const char* a, const char* c)
{
    if(a == c)
        return true;
    if(!a || !c)
        return false;
    return strcmp(a, c) == 0;
}

/* The narrow-atom arch gate. Faithful inline port of
 * instances/common/attention_arch.validate_tiled_attention_arch +
 * require_tiled_attention_arch for the gfx942 narrow path: a target is admitted
 * when it has the wide-K + ds_read_tr path (gfx950) OR, restricted to the
 * narrow-tiled-2d arch set ({"gfx942"}), the narrow 16x16x16 f16 AND bf16 atom.
 * Returns true on accept; on reject latches a NotImplementedError on b. */
static bool rocke_a2d_require_tiled_attention_arch(rocke_ir_builder_t* b, const char* arch)
{
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Python: KeyError from ArchTarget.from_gfx -> str(e) reason. */
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
    bool narrow_arch = rocke_a2d_streq(arch, "gfx942");
    bool narrow_k = rocke_mma_catalog_has_shape(mma, "mma", "f16", "f16", "fp32", 16, 16, 16)
                    && rocke_mma_catalog_has_shape(mma, "mma", "bf16", "bf16", "fp32", 16, 16, 16);
    if(narrow_arch && narrow_k)
        return true;
    if(b)
    {
        b->status = ROCKE_ERR_NOTIMPL;
        if(!wide_k)
            snprintf(b->err,
                     ROCKE_ERR_MSG_CAP,
                     "tiled attention requires either the wide-K MFMA atoms "
                     "(mfma_f32_16x16x32 / mfma_f32_32x32x16, gfx950) or, on a narrow "
                     "variant arch (gfx942), the 16x16x16 f16/bf16 atom; neither path is "
                     "available on %s",
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
 *  rocke_gfx942_attn2d_build_ctx_init -- the prologue (lines 1104-1912)
 * ===================================================================== */
bool rocke_gfx942_attn2d_build_ctx_init(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                        rocke_ir_builder_t* b,
                                        const rocke_attention_tiled_2d_spec_t* spec,
                                        const char* arch)
{
    if(ctx == NULL || b == NULL || spec == NULL)
        return false;
    memset(ctx, 0, sizeof(*ctx));

    if(arch == NULL)
        arch = "gfx942"; /* Python default */

    ctx->b = b;
    ctx->spec = spec;
    ctx->arch = arch;

    /* ---- require_tiled_attention_arch(arch) (lines 1139-1141) ---- */
    if(!rocke_a2d_require_tiled_attention_arch(b, arch))
        return false;

    /* ---- dtype gate (lines 1143-1145) ---- */
    if(!(rocke_a2d_streq(spec->dtype, "fp16") || rocke_a2d_streq(spec->dtype, "bf16")))
    {
        b->status = ROCKE_ERR_NOTIMPL;
        snprintf(b->err, ROCKE_ERR_MSG_CAP, "tiled 2D kernel supports fp16/bf16");
        return false;
    }
    const rocke_type_t* dtype = rocke_attention_tiled_2d_spec_dtype_ir(spec);
    ctx->dtype = dtype;

    /* ---- narrow 16x16x16 atom select (lines 1147-1159) ---- */
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    ctx->target = target;
    const char* a_dt = rocke_a2d_streq(spec->dtype, "fp16") ? "f16" : "bf16";
    const rocke_mma_catalog_t* mma = rocke_archtarget_mma(target);
    const rocke_mma_op_t* qk_atom
        = rocke_mma_catalog_select_largest_k(mma, "mma", a_dt, a_dt, "fp32", 16, 16, 16);
    if(qk_atom == NULL || qk_atom->k != 16)
    {
        b->status = ROCKE_ERR_NOTIMPL;
        snprintf(b->err,
                 ROCKE_ERR_MSG_CAP,
                 "gfx942 tiled 2D kernel requires a 16x16x16 %s MFMA atom on %s",
                 a_dt,
                 arch);
        return false;
    }
    /* The ctx field is typed rocke_mfma_atom_t*; the resolved object is an
     * rocke_mma_op_t (the select_largest_k return). Store via cast -- only .k /
     * the op_id are consumed downstream. */
    ctx->qk_atom = (const rocke_mfma_atom_t*)qk_atom;

    /* ---- ALL-CAPS geometry constants (lines 1161-1184) ---- */
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

    /* ---- fp8 K/V cache predicates (lines 1190-1237) ---- */
    const bool KV_FP8 = rocke_a2d_streq(spec->kv_storage_dtype, "fp8e4m3");
    const bool FP8_MFMA_QK = KV_FP8 && spec->use_fp8_mfma_qk;
    const bool FP8_MFMA_PV = KV_FP8 && spec->use_fp8_mfma_pv;
    const bool FP8_NATIVE_QK = false; /* documented dead path */
    ctx->KV_FP8 = KV_FP8;
    ctx->FP8_MFMA_QK = FP8_MFMA_QK;
    ctx->FP8_MFMA_PV = FP8_MFMA_PV;
    ctx->FP8_NATIVE_QK = FP8_NATIVE_QK;

    ctx->REGISTER_PV = spec->use_register_pv;
    ctx->TRANSPOSED_QK_32X32 = spec->use_transposed_qk_32x32;
    ctx->CONFLICT_FREE_V = spec->use_conflict_free_v;
    const bool CONFLICT_FREE_V_STORE = spec->use_conflict_free_v_store;
    ctx->CONFLICT_FREE_V_STORE = CONFLICT_FREE_V_STORE;
    ctx->K_SINGLE_BUF = spec->use_k_single_buffer;
    const bool K_SLICED_RING = spec->use_k_sliced_ring;
    ctx->K_SLICED_RING = K_SLICED_RING;
    ctx->K_SLICED_LDSSEQ = spec->use_k_sliced_ldsseq;
    ctx->USE_IGLP_OPT = spec->use_iglp_opt;
    ctx->USE_GLOBAL_LOAD_LDS_K = spec->use_global_load_lds_k;
    ctx->USE_QK_PV_SCHED_GROUP_BARRIER = spec->use_qk_pv_sched_group_barrier;

    /* env-var diagnostic switches (lines 1218-1235). os.environ defaults "0". */
    ctx->CFV_SCALAR_READ = false;
    ctx->CFV_STORE_SCALAR_LOAD = false;
    ctx->CFV_STORE_SCATTER = false;
    ctx->CFV_STORE_PREZERO = false;
    ctx->CFV_STORE_SEPOFF = false;
    ctx->CFV_STORE_SPLIT = spec->use_conflict_free_v_store_split;

    const int KV_BYTES = KV_FP8 ? 1 : 2;
    ctx->KV_BYTES = KV_BYTES;
    ctx->kv_io_dtype = KV_FP8 ? rocke_fp8e4m3() : dtype;
    /* kv_cache_aux: {"all":ALL,"global":GLOBAL,"stream":STREAM,"nt":NT}. */
    if(rocke_a2d_streq(spec->kv_cache_policy, "all"))
        ctx->kv_cache_aux = (int)ROCKE_CACHE_ALL;
    else if(rocke_a2d_streq(spec->kv_cache_policy, "global"))
        ctx->kv_cache_aux = (int)ROCKE_CACHE_GLOBAL;
    else if(rocke_a2d_streq(spec->kv_cache_policy, "stream"))
        ctx->kv_cache_aux = (int)ROCKE_CACHE_STREAM;
    else
        ctx->kv_cache_aux = (int)ROCKE_NON_TEMPORAL;

    /* ---- 32x32 umbrella + QK/PV MFMA geometry (lines 1253-1281) ---- */
    const bool USE_MFMA_32X32X8 = spec->use_mfma_32x32x8;
    const bool USE_MFMA_32X32 = spec->use_mfma_32x32 || USE_MFMA_32X32X8;
    ctx->USE_MFMA_32X32X8 = USE_MFMA_32X32X8;
    ctx->USE_MFMA_32X32 = USE_MFMA_32X32;

    int QK_MFMA_N, QK_K_STEP, PV_K_STEP, QK_K_ITERS, QK_N_TILES, PV_K_ITERS, PV_N_TILES;
    if(USE_MFMA_32X32)
    {
        QK_MFMA_N = 32;
        QK_K_STEP = USE_MFMA_32X32X8 ? 8 : 16;
        PV_K_STEP = QK_K_STEP;
        QK_K_ITERS = HD / QK_K_STEP;
        QK_N_TILES = T / QK_MFMA_N;
        PV_K_ITERS = T / PV_K_STEP;
        PV_N_TILES = HD / 32;
    }
    else
    {
        QK_MFMA_N = 16; /* MFMA_N */
        QK_K_STEP = 16;
        PV_K_STEP = 16;
        QK_K_ITERS = HD / QK_K_STEP;
        QK_N_TILES = T / QK_MFMA_N;
        PV_K_ITERS = T / PV_K_STEP;
        PV_N_TILES = HD / 16; /* MFMA_N */
    }
    ctx->QK_MFMA_N = QK_MFMA_N;
    ctx->QK_K_STEP = QK_K_STEP;
    ctx->PV_K_STEP = PV_K_STEP;
    ctx->QK_K_ITERS = QK_K_ITERS;
    ctx->QK_N_TILES = QK_N_TILES;
    ctx->PV_K_ITERS = PV_K_ITERS;
    ctx->PV_N_TILES = PV_N_TILES;

    /* ---- threads / wave / async-DMA width (lines 1282-1319) ---- */
    const int NUM_WARPS = spec->num_warps;
    const int WAVE = 64;
    const int THREADS = NUM_WARPS * WAVE;
    ctx->NUM_WARPS = NUM_WARPS;
    ctx->WAVE = WAVE;
    ctx->THREADS = THREADS;

    const int ASYNC_LDS_MAX_DWORDS = 1;
    const int ASYNC_LDS_MAX_BYTES_PER_LANE = ASYNC_LDS_MAX_DWORDS * 4;
    ctx->ASYNC_LDS_MAX_DWORDS = ASYNC_LDS_MAX_DWORDS;
    ctx->ASYNC_LDS_MAX_BYTES_PER_LANE = ASYNC_LDS_MAX_BYTES_PER_LANE;

    /* K/V async-DMA payload width. The gfx950 tiled-2D builder uses dwords=4
     * (16 bytes/lane = 8 bf16 halves: KV_HALVES_PER_CALL = THREADS*8,
     * lane_half_base = tid*8, WAVE_BYTES = WAVE*16); gfx942 uses dwords=1 (4
     * bytes/lane = 2 halves). Keyed off the wide ds_read_tr path (gfx950). */
    if(target->memory.has_ds_read_tr)
    {
        ctx->KV_DMA_HALVES_PER_LANE = 8;
        ctx->KV_DMA_DWORDS = 4;
    }
    else
    {
        ctx->KV_DMA_HALVES_PER_LANE = ASYNC_LDS_MAX_BYTES_PER_LANE / 2;
        ctx->KV_DMA_DWORDS = ASYNC_LDS_MAX_DWORDS;
    }

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

    /* ---- kernel name + attrs (lines 1321-1327) ---- *
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

    /* ---- parameter declarations (lines 1329-1369) ---- */
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

    /* ---- grid ids + wave decomposition (lines 1371-1388) ---- */
    if(spec->use_q_major_grid)
    {
        ctx->q_block_global_idx = rocke_b_block_id_x(b);
        ctx->kv_head_idx = rocke_b_block_id_y(b);
    }
    else
    {
        ctx->kv_head_idx = rocke_b_block_id_x(b);
        ctx->q_block_global_idx = rocke_b_block_id_y(b);
    }
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

    /* ---- seq lookup + Q-block geometry (lines 1390-1409) ---- */
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

    /* ---- Acc_lds stripe geometry (lines 1436-1443) ---- */
    const int OUT_STRIPE_COLS = (HD <= 64) ? 32 : HD;
    const int OUT_STRIPES = HD / OUT_STRIPE_COLS;
    ctx->OUT_STRIPE_COLS = OUT_STRIPE_COLS;
    ctx->OUT_STRIPES = OUT_STRIPES;

    /* ---- LDS dtypes / buffering / sizes (lines 1505-1531) ---- */
    const rocke_type_t* K_LDS_DTYPE = FP8_MFMA_QK ? rocke_fp8e4m3() : dtype;
    const rocke_type_t* V_LDS_DTYPE = FP8_MFMA_PV ? rocke_fp8e4m3() : dtype;
    const rocke_type_t* P_LDS_DTYPE = FP8_MFMA_PV ? rocke_fp8e4m3() : dtype;
    ctx->K_LDS_DTYPE = K_LDS_DTYPE;
    ctx->V_LDS_DTYPE = V_LDS_DTYPE;
    ctx->P_LDS_DTYPE = P_LDS_DTYPE;

    const int Q_BYTES = BLOCK_M * HD * 2;
    const int K_SLICE_HD = 32;
    const int K_SLICE_SLOTS = 3;
    const bool K_SLICED_ACTIVE = K_SLICED_RING && USE_MFMA_32X32X8 && ctx->TRANSPOSED_QK_32X32;
    const int K_LDS_ELEM_BYTES = rocke_type_eq(K_LDS_DTYPE, rocke_fp8e4m3()) ? 1 : 2;
    const int K_BUF_BYTES = (K_SLICED_ACTIVE ? (T * K_SLICE_HD) : (T * HD)) * K_LDS_ELEM_BYTES;
    const int K_BUFS = K_SLICED_ACTIVE ? K_SLICE_SLOTS : (ctx->K_SINGLE_BUF ? 1 : 2);
    const int K_TOTAL_BYTES = K_BUFS * K_BUF_BYTES;
    ctx->Q_BYTES = Q_BYTES;
    ctx->K_SLICE_HD = K_SLICE_HD;
    ctx->K_SLICE_SLOTS = K_SLICE_SLOTS;
    ctx->K_SLICED_ACTIVE = K_SLICED_ACTIVE;
    ctx->K_LDS_ELEM_BYTES = K_LDS_ELEM_BYTES;
    ctx->K_BUF_BYTES = K_BUF_BYTES;
    ctx->K_BUFS = K_BUFS;
    ctx->K_TOTAL_BYTES = K_TOTAL_BYTES;

    const bool Q_DIRECT_GLOBAL = K_SLICED_ACTIVE || spec->use_q_direct_global;
    const bool Q_ALIAS_K
        = (!Q_DIRECT_GLOBAL) && rocke_type_eq(K_LDS_DTYPE, dtype) && (Q_BYTES <= K_TOTAL_BYTES);
    const bool Q_USES_DUAL_SLOT = Q_ALIAS_K && (BLOCK_M > T);
    ctx->Q_DIRECT_GLOBAL = Q_DIRECT_GLOBAL;
    ctx->Q_ALIAS_K = Q_ALIAS_K;
    ctx->Q_USES_DUAL_SLOT = Q_USES_DUAL_SLOT;

    /* ---- K_lds smem_alloc (lines 1531-1534) ---- */
    if(K_SLICED_ACTIVE)
    {
        int shp[3] = {K_BUFS, T, K_SLICE_HD};
        ctx->K_lds = rocke_b_smem_alloc(b, K_LDS_DTYPE, shp, 3, "KldsS");
    }
    else
    {
        int shp[3] = {K_BUFS, T, HD};
        ctx->K_lds = rocke_b_smem_alloc(b, K_LDS_DTYPE, shp, 3, "Klds");
    }

    const int V_BUFS = 1;
    ctx->V_BUFS = V_BUFS;

    /* ---- V_lds swizzle / transposed-V sizing (lines 1573-1710) ---- */
    const int _V_HALVES_PER_LANE = ASYNC_LDS_MAX_BYTES_PER_LANE / 2;
    const bool _V_ROW_PER_WAVE_CALL = (WAVE * _V_HALVES_PER_LANE) == HD;
    const int V_ROWS_PER_CALL = NUM_WARPS; /* one row per wave per call */

    long _LDS_CAP = 65536;
    if(target != NULL)
        _LDS_CAP = (long)target->lds_capacity_bytes;

    const int _swz_v_rows = (V_ROWS_PER_CALL > 1) ? V_ROWS_PER_CALL : 1;
    const long _swz_extra_bytes = (long)(T / _swz_v_rows) * V_ROWS_PER_CALL * 8 * 2;
    const int _out_stripe_b = (HD <= 64) ? 32 : HD;
    const long _lds_natural = 2L * T * HD * 2 + (long)T * HD * 2 + (long)BLOCK_M * (T + 8) * 2
                              + (long)((BLOCK_M <= 2 * T) ? 0 : (BLOCK_M * HD * 2))
                              + (long)BLOCK_M * _out_stripe_b * 2;
    const bool _swz_fits = (_lds_natural + _swz_extra_bytes) <= _LDS_CAP;

    const int _v_t_pad = 8;
    const long _v_t_extra_bytes = (long)HD * _v_t_pad * 2;
    const bool _v_t_fits = (_lds_natural + _v_t_extra_bytes) <= _LDS_CAP;

    const int _x8_k_slots = (BLOCK_M <= T) ? 1 : 2;
    const long _x8_q_lds = (BLOCK_M <= 2 * T) ? 0 : (long)BLOCK_M * HD * 2;
    const bool _v_t_fits_x8
        = ((long)_x8_k_slots * T * HD * 2 + (long)(T + _v_t_pad) * HD * 2 + _x8_q_lds) <= _LDS_CAP;
    const bool _v_t_fits_eff = CONFLICT_FREE_V_STORE ? _v_t_fits_x8 : _v_t_fits;

    const bool TRANSPOSED_V
        = (ctx->CONFLICT_FREE_V || CONFLICT_FREE_V_STORE) && USE_MFMA_32X32X8
          && ctx->TRANSPOSED_QK_32X32 && !FP8_MFMA_PV && !FP8_MFMA_QK && !KV_FP8
          && !ctx->FAST_PAGED_KV_DESC && rocke_type_eq(V_LDS_DTYPE, dtype) &&
          /* cfv/cfvst is byte-size driven: fp16 and bf16 are
                               * both 2-byte, the perm_b32 2x2 transpose + ds_read_b64
                               * feed are layout-identical, and the K=8 32x32x8 atom is
                               * gfx942-legal for bf16 (the K=16 bf16 atom is gfx950-
                               * only). Mirrors the Python TRANSPOSED_V gate. */
          (rocke_type_eq(dtype, rocke_f16()) || rocke_type_eq(dtype, rocke_bf16()))
          && (HD == 64 || HD == 128) && (HD % 8 == 0) && ((T * HD) % THREADS == 0) && _v_t_fits_eff;
    const bool TRANSPOSED_V_STORE = TRANSPOSED_V && CONFLICT_FREE_V_STORE && (T % 2 == 0);
    ctx->TRANSPOSED_V = TRANSPOSED_V;
    ctx->TRANSPOSED_V_STORE = TRANSPOSED_V_STORE;

    const bool SWIZZLE_VLDS = true /* env HIPDNN_GFX942_SWIZZLE_VLDS default "1" */ && !FP8_MFMA_PV
                              && !FP8_MFMA_QK && !USE_MFMA_32X32 && !ctx->REGISTER_PV
                              && !TRANSPOSED_V && rocke_type_eq(V_LDS_DTYPE, dtype)
                              && _V_ROW_PER_WAVE_CALL && (NUM_WARPS == 1 || NUM_WARPS == 2)
                              && (V_ROWS_PER_CALL != 0 && (T % V_ROWS_PER_CALL) == 0)
                              && (HD == 64 || HD == 128) && !ctx->FAST_PAGED_KV_DESC && _swz_fits;
    ctx->SWIZZLE_VLDS = SWIZZLE_VLDS;

    const int V_LDS_PAD = SWIZZLE_VLDS ? 8 : 0;
    const int V_LDS_STRIDE = HD + V_LDS_PAD;
    const int V_GROUP_STRIDE = V_ROWS_PER_CALL * V_LDS_STRIDE;
    int V_GROUP_SHIFT = 0;
    if(SWIZZLE_VLDS)
    {
        /* V_ROWS_PER_CALL.bit_length() - 1 */
        int v = V_ROWS_PER_CALL;
        int bl = 0;
        while(v > 0)
        {
            bl++;
            v >>= 1;
        }
        V_GROUP_SHIFT = bl - 1;
    }
    ctx->V_LDS_PAD = V_LDS_PAD;
    ctx->V_LDS_STRIDE = V_LDS_STRIDE;
    ctx->V_GROUP_STRIDE = V_GROUP_STRIDE;
    ctx->V_GROUP_SHIFT = V_GROUP_SHIFT;
    ctx->V_ROWS_PER_CALL = V_ROWS_PER_CALL;

    const int V_T_PAD = TRANSPOSED_V ? _v_t_pad : 0;
    const int V_T_STRIDE = T + V_T_PAD;
    ctx->V_T_PAD = V_T_PAD;
    ctx->V_T_STRIDE = V_T_STRIDE;

    const bool V_T_CK_LAYOUT = TRANSPOSED_V && spec->use_conflict_free_v_ck_vlds;
    ctx->V_T_CK_LAYOUT = V_T_CK_LAYOUT;
    const int V_T_KPACK = 8;
    const int V_T_PIXELS_PER_ROW = 64;
    const int V_T_NPER_ROW = V_T_PIXELS_PER_ROW / V_T_KPACK;
    const int V_T_NGROUPS = HD / V_T_NPER_ROW;
    const int V_T_KGROUPS = T / V_T_KPACK;
    const int V_T_GROUP_STRIDE = V_T_PIXELS_PER_ROW + V_T_KPACK;
    const int V_T_CK_SLOTS = V_T_KGROUPS * V_T_NGROUPS * V_T_GROUP_STRIDE;
    ctx->V_T_KPACK = V_T_KPACK;
    ctx->V_T_PIXELS_PER_ROW = V_T_PIXELS_PER_ROW;
    ctx->V_T_NPER_ROW = V_T_NPER_ROW;
    ctx->V_T_NGROUPS = V_T_NGROUPS;
    ctx->V_T_KGROUPS = V_T_KGROUPS;
    ctx->V_T_GROUP_STRIDE = V_T_GROUP_STRIDE;
    ctx->V_T_CK_SLOTS = V_T_CK_SLOTS;

    /* ---- V_lds smem_alloc (lines 1711-1754) ---- */
    if(TRANSPOSED_V)
    {
        if(V_T_CK_LAYOUT)
        {
            int shp[2] = {V_BUFS, V_T_CK_SLOTS};
            ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 2, "VldsTck");
        }
        else
        {
            int shp[3] = {V_BUFS, HD, V_T_STRIDE};
            ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 3, "VldsT");
        }
    }
    else if(FP8_MFMA_PV)
    {
        const int N_STRIPES = HD / 16;
        ctx->N_STRIPES = N_STRIPES;
        int shp[4] = {V_BUFS, N_STRIPES, T, 16};
        ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 4, "VldsStripe");
    }
    else if(SWIZZLE_VLDS)
    {
        const int V_N_GROUPS = T / V_ROWS_PER_CALL;
        const int V_N_SLOTS = V_N_GROUPS * V_GROUP_STRIDE;
        ctx->V_N_SLOTS = V_N_SLOTS;
        int shp[2] = {V_BUFS, V_N_SLOTS};
        ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 2, "Vlds");
    }
    else
    {
        int shp[3] = {V_BUFS, T, HD};
        ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 3, "Vlds");
    }

    /* ---- P_lds (lines 1816-1832) ---- */
    const bool TRANSPOSED_X8 = USE_MFMA_32X32X8 && ctx->TRANSPOSED_QK_32X32;
    if(!ctx->REGISTER_PV && !TRANSPOSED_X8)
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

    /* ---- fp8 K/V staging slabs (lines 1851-1853) ---- */
    if(KV_FP8 && spec->use_fp8_mfma_qk)
    {
        int kshp[3] = {2, T, HD};
        int vshp[3] = {1, T, HD};
        ctx->K_fp8_lds = rocke_b_smem_alloc(b, rocke_fp8e4m3(), kshp, 3, "Kfp8lds");
        ctx->V_fp8_lds = rocke_b_smem_alloc(b, rocke_fp8e4m3(), vshp, 3, "Vfp8lds");
    }

    /* ---- Q_lds (lines 1854-1864) ---- */
    if(Q_DIRECT_GLOBAL)
    {
        ctx->Q_lds = ctx->K_lds;
    }
    else if(Q_ALIAS_K)
    {
        ctx->Q_lds = ctx->K_lds;
    }
    else
    {
        int shp[2] = {BLOCK_M, HD};
        ctx->Q_lds = rocke_b_smem_alloc(b, dtype, shp, 2, "Qlds");
    }

    /* ---- Acc_lds (line 1873) ---- */
    {
        int shp[2] = {BLOCK_M, OUT_STRIPE_COLS};
        ctx->Acc_lds = rocke_b_smem_alloc(b, dtype, shp, 2, "Aclds");
    }

    /* ---- transpose-LDS lane formula params (lines 1883-1889) ---- *
     * TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, lane); B=1. The ctx only
     * carries the compile-time params; the per-lane reader Values are
     * materialized by the consuming phases. */
    ctx->tlds_m = 16;
    ctx->tlds_k = PV_K_STEP;
    ctx->tlds_b = 1;
    /* TransposeLdsReader.bind(b, lane) materializes 3 lane-derived SSA values
     * here (Python layouts.py 289-291), emitted BEFORE qk_scale:
     *   lane_div_16      = div(lane, 16)
     *   lane_div_4_mod_4 = mod(div(lane, 4), 4)
     *   col              = mul(mod(lane, 4), 4) */
    ctx->tlds_lane_div_16 = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, 16));
    {
        /* Match Python's left-to-right value creation exactly:
         *   lane_div_4_mod_4 = mod(div(lane, 4), 4)  -- inner div emitted first;
         *   col              = mul(mod(lane, 4), 4)  -- inner mod emitted first.
         * Sequence via temps so the inner op is numbered before the outer const
         * (C arg-eval order is unspecified and would otherwise reorder them). */
        rocke_value_t* c4a = rocke_b_const_i32(b, 4);
        rocke_value_t* div4 = rocke_b_div(b, ctx->lane, c4a);
        ctx->tlds_lane_div_4_mod_4 = rocke_b_mod(b, div4, rocke_b_const_i32(b, 4));
        rocke_value_t* mod4 = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, 4));
        ctx->tlds_col = rocke_b_mul(b, mod4, rocke_b_const_i32(b, 4));
    }

    /* ---- SSA constants (lines 1892-1896) ---- *
     * Match Python's exact creation order: neg_inf, zero_f, one_f, rcp_ln2,
     * then qk_scale. (The previous port created an i32 0 here that Python does
     * not, and skipped neg_inf/one_f, shifting every later SSA counter.) */
    /* Cache neg_inf / one_f / rcp_ln2 so downstream phases (m_inits, mask)
     * REUSE these exact SSA values instead of recreating fresh consts -- each
     * recreation would allocate an extra %value and shift the counter. Python
     * builds them once here in the constants block. */
    ctx->neg_inf_v = rocke_b_const_f32(b, -INFINITY); /* neg_inf (1892) */
    ctx->zero_f = rocke_b_const_f32(b, 0.0); /* zero_f (1893) */
    ctx->one_f_v = rocke_b_const_f32(b, 1.0); /* one_f (1894) */
    {
        const double rcp_ln2 = 1.4426950408889634;
        ctx->rcp_ln2_v = rocke_b_const_f32(b, rcp_ln2); /* (1895) */
        rocke_value_t* qk_scale = rocke_b_fmul(b, ctx->scale_p, ctx->rcp_ln2_v); /* (1896) */
        if(FP8_NATIVE_QK)
            qk_scale = rocke_b_fmul(b, qk_scale, ctx->k_scale_p);
        ctx->qk_scale_v = qk_scale;
    }
    /* sw_const = b.const_i32(int(SLIDING_WINDOW)) (line 1910). Python creates
     * this once here in the constants block and REUSES it in the tile-bound
     * (line 1994) and mask (lines 4111/4328/4403) regions. The previous port
     * created an unused i32 0 here and a FRESH const in the loop scaffold,
     * allocating an extra SSA value that shifted every later %N. Cache it. */
    ctx->sw_const_v = rocke_b_const_i32(b, ctx->SLIDING_WINDOW);
    ctx->c0 = ctx->sw_const_v;

    /* ---- acc geometry the epilogue aliases (lines 2142-2143) ---- */
    ctx->ACC_N_TILES = USE_MFMA_32X32 ? (HD / 32) : PV_N_TILES;
    ctx->ACC_M_ATOMS = USE_MFMA_32X32 ? 1 : M_ATOMS_PER_WARP;
    ctx->ml_count = 2 * SOFTMAX_STATE_SLOTS;

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Per-lane row map + bit helpers (lines 2031-2052).
 *
 *  These recompute the per-warp lane decomposition (lane_rg / lane_col / ...)
 *  from ctx->lane, mirroring the Python SSA values the closures captured.
 * ===================================================================== */

static rocke_value_t* rocke_a2d_lane_rg(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    /* Reuse the cached SSA value (emitted once in emit_loop_bounds_and_inits).
     * Fall back to a fresh op only if a caller runs before that phase. */
    if(ctx->lane_rg_v != NULL)
        return ctx->lane_rg_v;
    return rocke_b_div(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 16));
}

static rocke_value_t* rocke_a2d_lane_col(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    if(ctx->lane_col_v != NULL)
        return ctx->lane_col_v;
    return rocke_b_mod(ctx->b, ctx->lane, rocke_b_const_i32(ctx->b, 16));
}

rocke_value_t* rocke_gfx942_attn2d_in_warp_row(rocke_gfx942_attn2d_build_ctx_t* ctx, int r)
{
    /* atom_idx = r // 4 ; in_atom = r % 4
     * row = lane_rg*4 + (atom_idx*16 + in_atom) */
    int atom_idx = r / 4;
    int in_atom = r % 4;
    /* Python evaluates b.mul(lane_rg, const(4)) BEFORE the trailing const
     * (left-to-right arg order). Bind the mul to a temp so C's arg-eval order
     * does not allocate the trailing const ahead of the mul and shift it. */
    rocke_value_t* rg4 = rocke_b_mul(ctx->b, rocke_a2d_lane_rg(ctx), rocke_b_const_i32(ctx->b, 4));
    return rocke_b_add(ctx->b, rg4, rocke_b_const_i32(ctx->b, atom_idx * 16 + in_atom));
}

rocke_value_t* rocke_gfx942_attn2d_state_row(rocke_gfx942_attn2d_build_ctx_t* ctx, int r)
{
    if(ctx->USE_MFMA_32X32 && !ctx->TRANSPOSED_QK_32X32)
        return rocke__mfma_32x32_c_row(ctx->b, ctx->lane, r);
    return rocke_gfx942_attn2d_in_warp_row(ctx, r);
}

rocke_value_t*
    rocke_gfx942_attn2d_bit2(rocke_gfx942_attn2d_build_ctx_t* ctx, rocke_value_t* v, int bit)
{
    /* Python: land(lshr(v, const(bit)), const(1)) emits the lshr BEFORE the
     * trailing const(1) (left-to-right). Bind the lshr first so C's right-to-
     * left arg eval does not allocate const(1) ahead of it and shift numbering. */
    rocke_value_t* shifted = rocke_b_lshr(ctx->b, v, rocke_b_const_i32(ctx->b, bit));
    return rocke_b_land(ctx->b, shifted, rocke_b_const_i32(ctx->b, 1));
}

rocke_value_t* rocke_gfx942_attn2d_select_lane_rg(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                                  rocke_value_t* v0,
                                                  rocke_value_t* v1,
                                                  rocke_value_t* v2,
                                                  rocke_value_t* v3)
{
    rocke_value_t* lane_rg = rocke_a2d_lane_rg(ctx);
    rocke_value_t* is0 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 0));
    rocke_value_t* is1 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 1));
    rocke_value_t* is2 = rocke_b_cmp_eq(ctx->b, lane_rg, rocke_b_const_i32(ctx->b, 2));
    return rocke_b_select(
        ctx->b, is0, v0, rocke_b_select(ctx->b, is1, v1, rocke_b_select(ctx->b, is2, v2, v3)));
}

/* ===================================================================== *
 *  Accumulator index helpers (lines 2147-2162, 3714-3715, 5074-5075).
 * ===================================================================== */

int rocke_gfx942_attn2d_acc_idx(const rocke_gfx942_attn2d_build_ctx_t* ctx, int n, int atom)
{
    return n * ctx->ACC_M_ATOMS + atom;
}

rocke_value_t* rocke_gfx942_attn2d_acc_get(rocke_gfx942_attn2d_build_ctx_t* ctx, int n, int atom)
{
    /* Reads the unpacked loop-body acc carry (set per _emit_kv_body). */
    return ctx->acc_cur[rocke_gfx942_attn2d_acc_idx(ctx, n, atom)];
}

rocke_value_t*
    rocke_gfx942_attn2d_acc_final_get(rocke_gfx942_attn2d_build_ctx_t* ctx, int n, int atom)
{
    /* Reads the final loop-result acc (set by the epilogue read-back). */
    return ctx->acc_final[rocke_gfx942_attn2d_acc_idx(ctx, n, atom)];
}

/* ===================================================================== *
 *  P permute / pack helpers (Python lines 2053-2111).
 *
 *  _permute_p_c_to_a16: one 16-col P tile, from 16x16 MFMA-C regs (the QK C
 *  distribution) to the PV-A register layout. The transform is (A,B,C,R) ->
 *  (B,A,R,C): two bit-level transposes swap C with R inside a lane quad, then
 *  lane-field swaps exchange A and B. Faithful transcription of the Python; the
 *  per-lane scalars lane_col_mod4 / lane_rg / lane_col_div4 are recomputed from
 *  ctx->lane (DCE-identical to Python's shared SSA values).
 * ===================================================================== */

void rocke_gfx942_attn2d_permute_p_c_to_a16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                            rocke_value_t* const* p_regs_f32,
                                            int n,
                                            rocke_value_t** out,
                                            int* out_n)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* lane_col = rocke_a2d_lane_col(ctx); /* lane % 16     */
    /* Python uses the prologue-cached lane_col_mod4 / lane_col_div4 (one local
     * each, gfx950 2018-2019); reuse the same SSA values instead of recomputing
     * the mod/div here (which inserted two extra ops and shifted numbering). */
    rocke_value_t* lane_col_mod4 = (ctx->lane_col_mod4_v != NULL)
                                       ? ctx->lane_col_mod4_v
                                       : rocke_b_mod(b, lane_col, rocke_b_const_i32(b, 4));
    rocke_value_t* lane_col_div4 = (ctx->lane_col_div4_v != NULL)
                                       ? ctx->lane_col_div4_v
                                       : rocke_b_div(b, lane_col, rocke_b_const_i32(b, 4));
    rocke_value_t* lane_rg = rocke_a2d_lane_rg(ctx); /* lane / 16     */

    /* The transform is defined for a 4-element register quad. */
    rocke_value_t* vals[4];
    int i;
    for(i = 0; i < 4; ++i)
        vals[i] = (i < n) ? p_regs_f32[i] : NULL;

    /* Two bit-level transposes: swap C (lane_col_mod4) with R (reg index). */
    for(int bit = 0; bit < 2; ++bit)
    {
        rocke_value_t* lane_bit = rocke_gfx942_attn2d_bit2(ctx, lane_col_mod4, bit);
        int reg_bit = 1 << bit;
        rocke_value_t* old[4];
        for(i = 0; i < 4; ++i)
            old[i] = vals[i];
        for(int reg = 0; reg < 4; ++reg)
        {
            rocke_value_t* partner = rocke_b_warp_shuffle_xor(b, old[reg ^ reg_bit], reg_bit);
            rocke_value_t* same_bit
                = rocke_b_cmp_eq(b, lane_bit, rocke_b_const_i32(b, (reg >> bit) & 1));
            vals[reg] = rocke_b_select(b, same_bit, old[reg], partner);
        }
    }

    /* Lane-field swaps: exchange A (lane_rg) and B (lane_col_div4). */
    {
        const int bits[2] = {0, 1};
        const int lane_xors[2] = {20, 40};
        for(int j = 0; j < 2; ++j)
        {
            int bit = bits[j];
            int lane_xor = lane_xors[j];
            rocke_value_t* a_bit = rocke_gfx942_attn2d_bit2(ctx, lane_rg, bit);
            rocke_value_t* b_bit = rocke_gfx942_attn2d_bit2(ctx, lane_col_div4, bit);
            rocke_value_t* swap = rocke_b_cmp_ne(b, a_bit, b_bit);
            for(i = 0; i < 4; ++i)
                vals[i] = rocke_b_select(
                    b, swap, rocke_b_warp_shuffle_xor(b, vals[i], lane_xor), vals[i]);
        }
    }

    for(i = 0; i < 4; ++i)
        out[i] = vals[i];
    if(out_n != NULL)
        *out_n = 4;
}

rocke_value_t* rocke_gfx942_attn2d_pack_p_a16(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* const* p_regs_f32,
                                              int n)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    rocke_value_t* permuted[4];
    rocke_value_t* casted[4];
    int out_n = 0;
    rocke_gfx942_attn2d_permute_p_c_to_a16(ctx, p_regs_f32, n, permuted, &out_n);
    for(int i = 0; i < 4; ++i)
        casted[i] = rocke_b_cast_f32_to(b, permuted[i], dtype);
    return rocke_b_vec_pack(b, casted, 4, dtype);
}

rocke_value_t* rocke_gfx942_attn2d_pack_p_a32(rocke_gfx942_attn2d_build_ctx_t* ctx,
                                              rocke_value_t* const* p_regs0_f32,
                                              rocke_value_t* const* p_regs1_f32,
                                              int n)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dtype = ctx->dtype;
    rocke_value_t* vals0[4];
    rocke_value_t* vals1[4];
    rocke_value_t* lohi[8];
    int dummy = 0;
    rocke_gfx942_attn2d_permute_p_c_to_a16(ctx, p_regs0_f32, n, vals0, &dummy);
    rocke_gfx942_attn2d_permute_p_c_to_a16(ctx, p_regs1_f32, n, vals1, &dummy);
    for(int j = 0; j < 4; ++j)
    {
        rocke_value_t* v0 = vals0[j];
        rocke_value_t* v1 = vals1[j];
        rocke_value_t* v0_x16 = rocke_b_warp_shuffle_xor(b, v0, 16);
        rocke_value_t* v0_x32 = rocke_b_warp_shuffle_xor(b, v0, 32);
        rocke_value_t* v0_x48 = rocke_b_warp_shuffle_xor(b, v0, 48);
        rocke_value_t* v1_x16 = rocke_b_warp_shuffle_xor(b, v1, 16);
        rocke_value_t* v1_x32 = rocke_b_warp_shuffle_xor(b, v1, 32);
        rocke_value_t* v1_x48 = rocke_b_warp_shuffle_xor(b, v1, 48);
        /* lo[j] = select_lane_rg(v0, v0_x48, v1_x32, v1_x16) */
        lohi[j] = rocke_gfx942_attn2d_select_lane_rg(ctx, v0, v0_x48, v1_x32, v1_x16);
        /* hi[j] = select_lane_rg(v0_x16, v0_x32, v1_x48, v1) */
        lohi[4 + j] = rocke_gfx942_attn2d_select_lane_rg(ctx, v0_x16, v0_x32, v1_x48, v1);
    }
    {
        rocke_value_t* casted[8];
        for(int i = 0; i < 8; ++i)
            casted[i] = rocke_b_cast_f32_to(b, lohi[i], dtype);
        return rocke_b_vec_pack(b, casted, 8, dtype);
    }
}

/* ===================================================================== *
 *  LICM hoist of per-reg invariants (Python lines 3609-3651).
 *
 *  Fills ctx->hoist_* before the KV-loop with the per-reg query position /
 *  head / row-validity / causal-limit invariants the QK/softmax bucket reads.
 *  Field-name mapping consumed by the kv_body_qk_softmax bucket:
 *      hoist_q_pos   <- qp_r        hoist_row_mask <- row_ok
 *      hoist_state_row <- causal_lim   hoist_q_head <- alibi slope (USE_ALIBI)
 *  hoist_in_warp_row caches the per-reg row.
 * ===================================================================== */

void rocke_gfx942_attn2d_emit_licm_hoist(rocke_gfx942_attn2d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int reg;
    for(reg = 0; reg < ctx->REGS_PER_LANE; ++reg)
    {
        rocke_value_t* row;
        if(ctx->USE_MFMA_32X32)
            row = rocke_b_add(b, ctx->wave_row_base, rocke__mfma_32x32_c_row(b, ctx->lane, reg));
        else
            row = rocke_b_add(b, ctx->wave_row_base, rocke_gfx942_attn2d_in_warp_row(ctx, reg));

        rocke_value_t* qp_r = rocke_b_add(
            b, ctx->qb_start_pos, rocke_b_div(b, row, rocke_b_const_i32(b, ctx->NQK)));
        /* qh_r = kv_head*NQK + row%NQK (mul before mod). */
        rocke_value_t* qh_mul = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, ctx->NQK));
        rocke_value_t* qh_mod = rocke_b_mod(b, row, rocke_b_const_i32(b, ctx->NQK));
        rocke_value_t* qh_r = rocke_b_add(b, qh_mul, qh_mod);
        /* row_ok = (qp_r < q_len) && (qh_r < NUM_QH) (qp cmp before qh cmp). */
        rocke_value_t* row_ok_pos = rocke_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
        rocke_value_t* row_ok_qh = rocke_b_cmp_lt(b, qh_r, rocke_b_const_i32(b, ctx->NUM_QH));
        rocke_value_t* row_ok = rocke_b_land(b, row_ok_pos, row_ok_qh);
        rocke_value_t* causal_lim = rocke_b_add(b, ctx->context_len, qp_r);

        ctx->hoist_in_warp_row[reg] = row;
        ctx->hoist_q_pos[reg] = qp_r;
        ctx->hoist_q_head[reg] = qh_r;
        ctx->hoist_row_mask[reg] = row_ok;
        ctx->hoist_state_row[reg] = causal_lim;
    }

    if(ctx->USE_ALIBI)
    {
        const rocke_type_t* f32 = rocke_f32();
        for(reg = 0; reg < ctx->REGS_PER_LANE; ++reg)
        {
            rocke_value_t* qh_r = ctx->hoist_q_head[reg];
            rocke_value_t* qh_ok = rocke_b_cmp_lt(b, qh_r, rocke_b_const_i32(b, ctx->NUM_QH));
            rocke_value_t* slope = rocke_b_masked_global_load(
                b, ctx->alibi_slopes_ptr, qh_r, qh_ok, rocke_b_const_f32(b, 0.0), f32, 4);
            /* The softmax bucket reads the per-reg slope from hoist_q_head. */
            ctx->hoist_q_head[reg] = slope;
        }
    }
}
