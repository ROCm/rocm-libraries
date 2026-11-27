// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * The tiled-2D attention spec's default constructor and derived @property
 * accessors, which the gfx942 and gfx950 builders share. See
 * rocke/helper_rocke.instances.common.attention_tiled_2d_spec.h.
 */
#include "rocke/helper_rocke.instances.common.attention_tiled_2d_spec.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------- spec default */

rocke_attention_tiled_2d_spec_t rocke_attention_tiled_2d_spec_default(void)
{
    rocke_attention_tiled_2d_spec_t s;
    memset(&s, 0, sizeof(s));

    /* required fields stay zero/NULL until the caller sets them. */

    /* defaulted fields (mirror the dataclass defaults). */
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.num_seqs = 0;
    s.num_warps = 1;
    s.has_waves_per_eu = false;
    s.waves_per_eu = 0;
    s.kv_storage_dtype = NULL;
    s.use_fp8_mfma_qk = false;
    s.use_fp8_mfma_pv = false;
    s.use_register_pv = false;
    s.use_v_double_buffer = false;
    s.kv_ring_depth = 2;
    s.use_staggered_iter_wait = false;
    s.use_q_reread = false;
    s.use_q_direct_reg = false; /* #79 */
    s.use_sched_barrier = false;
    s.sched_barrier_mask = 0;
    s.use_softmax_mfma_interleave = false; /* #80 */
    s.softmax_interleave_mode = 1;
    s.softmax_interleave_groups = 4;
    s.has_tile_size = false;
    s.tile_size = 0;
    s.block_m_per_warp = 16;
    s.use_mfma_32x32 = false;
    s.use_mfma_32x32x8 = false;
    s.use_transposed_qk_32x32 = false;
    s.use_transposed_scalar_state = false;
    s.use_transposed_invariant_hoist = false;
    s.use_transposed_mask_once = false;
    s.use_transposed_half_local_pv = false;
    s.use_mfma32_skip_legacy_qreg = false;
    s.use_transposed_mask_limit = false;
    s.use_grouped_kv2_softmax = false;
    s.use_fast_paged_kv_desc = false;
    s.use_i64_kv_addr = false;
    s.use_early_v_schedule = false;
    s.use_agpr_alloc_zero = false;
    s.use_conflict_free_v = false;
    s.use_conflict_free_v_store = false;
    s.use_conflict_free_v_store_split = true;
    s.use_conflict_free_v_ck_vlds = true;
    s.use_k_single_buffer = false;
    s.use_k_sliced_ring = false;
    s.ring_depth = 3;
    s.k_slice_hd = 32;
    s.use_k_sliced_ldsseq = false;
    s.use_iglp_opt = false;
    s.use_qk_pv_sched_group_barrier = false;
    s.use_q_direct_global = false;
    s.use_v_hbm_direct = false;
    s.use_k_hbm_direct = false;
    s.kv_cache_policy = "stream";
    s.use_global_load_lds_k = false;
    s.use_q_major_grid = false;
    s.use_qblock_reverse = false;
    s.emit_lse = false;
    s.lse_head_major = false;
    s.dropout = false;
    return s;
}

/* ------------------------------------------------- derived @property bodies */

int rocke_attention_tiled_2d_spec_num_queries_per_kv(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL || s->num_kv_heads == 0)
    {
        return 0;
    }
    return s->num_query_heads / s->num_kv_heads;
}

int rocke_attention_tiled_2d_spec_block_m(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    return s->block_m_per_warp * s->num_warps;
}

int rocke_attention_tiled_2d_spec_regs_per_lane(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    if(s->use_mfma_32x32 || s->use_mfma_32x32x8)
    {
        return 16;
    }
    return s->block_m_per_warp / 4; /* 4 for M=16, 8 for M=32 */
}

int rocke_attention_tiled_2d_spec_block_q(const rocke_attention_tiled_2d_spec_t* s)
{
    int nqk;
    if(s == NULL)
    {
        return 0;
    }
    nqk = rocke_attention_tiled_2d_spec_num_queries_per_kv(s);
    if(nqk == 0)
    {
        return 0;
    }
    return rocke_attention_tiled_2d_spec_block_m(s) / nqk;
}

int rocke_attention_tiled_2d_spec_tile_size_eff(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    return s->has_tile_size ? s->tile_size : s->block_size;
}

int rocke_attention_tiled_2d_spec_n_blocks_per_tile(const rocke_attention_tiled_2d_spec_t* s)
{
    int bs;
    if(s == NULL)
    {
        return 0;
    }
    bs = s->block_size;
    if(bs == 0)
    {
        return 0;
    }
    return rocke_attention_tiled_2d_spec_tile_size_eff(s) / bs;
}

const rocke_type_t* rocke_attention_tiled_2d_spec_dtype_ir(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s != NULL && s->dtype != NULL && strcmp(s->dtype, "fp16") == 0)
    {
        return rocke_f16();
    }
    return rocke_bf16();
}

int rocke_attention_tiled_2d_spec_binary_search_iters(const rocke_attention_tiled_2d_spec_t* s)
{
    int it;
    if(s == NULL || s->num_seqs <= 0)
    {
        return 32;
    }
    /* max(1, ceil(log2(num_seqs + 1))) */
    it = (int)ceil(log2((double)(s->num_seqs + 1)));
    return it < 1 ? 1 : it;
}
