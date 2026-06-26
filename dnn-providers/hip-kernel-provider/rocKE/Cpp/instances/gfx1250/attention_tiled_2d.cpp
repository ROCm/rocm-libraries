// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_attention_tiled_2d.c -- C99 port of
 * rocke/instances/gfx1250/attention_tiled_2d.py.
 *
 * gfx1250 WMMA tiled-2D unified-attention forward: one wave32 CTA per
 * (kv_head, q_block), one paged-KV block per K-loop iteration, fp8e4m3 paged K/V
 * dequantized to bf16, online softmax, P*V via the shared common base. The build
 * op order tracks build_unified_attention_2d_tiled() top-to-bottom; emitted IR is
 * byte-identical to the Python lowerer (args sequenced left-to-right, const_i32
 * never deduped, list comprehensions unrolled in source order).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_attention_tiled_2d.h"
#include "rocke/instance_gfx1250_wmma_attention_common.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_helper_rocke.helpers.attention.h" /* binary_search_seq_idx */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_U2D_BLOCK_SIZE 32
#define ROCKE_U2D_NUM_QUERY_HEADS 64
#define ROCKE_U2D_NUM_KV_HEADS 8
#define ROCKE_U2D_NQK (ROCKE_U2D_NUM_QUERY_HEADS / ROCKE_U2D_NUM_KV_HEADS)
#define ROCKE_U2D_BLOCK_Q (ROCKE_G1250_BLOCK_M / ROCKE_U2D_NQK)

/* 2D phys_block closure: a cached physical_block id (one paged block per tile). */
static rocke_value_t*
    rocke_u2d_phys_block_cached(rocke_ir_builder_t* b, rocke_value_t* tok, void* ctx)
{
    (void)b;
    (void)tok;
    return (rocke_value_t*)ctx;
}

/* qh = add(mul(kv_head_idx, const(NQK)), mod(row, const(NQK))). Python evals the
 * mul before the mod; sequence into temporaries to preserve SSA numbering. */
static rocke_value_t*
    rocke_u2d_qh(rocke_ir_builder_t* b, rocke_value_t* kv_head_idx, rocke_value_t* row, int nqk)
{
    rocke_value_t* m = rocke_b_mul(b, kv_head_idx, rocke_b_const_i32(b, nqk));
    rocke_value_t* md = rocke_b_mod(b, row, rocke_b_const_i32(b, nqk));
    return rocke_b_add(b, m, md);
}

/* land(cmp_lt(q_pos, qlen), cmp_lt(qh, NUM_QH)). Python evals the q_pos cmp
 * first; sequence so the two cmps emit left-to-right. */
static rocke_value_t* rocke_u2d_row_valid(rocke_ir_builder_t* b,
                                          rocke_value_t* q_pos,
                                          rocke_value_t* q_len,
                                          rocke_value_t* qh,
                                          int num_qh)
{
    rocke_value_t* a = rocke_b_cmp_lt(b, q_pos, q_len);
    rocke_value_t* c = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, num_qh));
    return rocke_b_land(b, a, c);
}

rocke_uattn2d_tiled_gfx1250_spec_t rocke_uattn2d_tiled_gfx1250_spec_default(void)
{
    rocke_uattn2d_tiled_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.num_seqs = 0;
    s.num_warps = 1;
    s.waves_per_eu_set = false;
    s.kv_storage_dtype = "fp8e4m3";
    s.tile_size_set = false;
    s.block_m_per_warp = 16;
    s.use_register_p = false;
    return s;
}

int rocke_uattn2d_tiled_gfx1250_num_queries_per_kv(const rocke_uattn2d_tiled_gfx1250_spec_t* spec)
{
    return spec->num_query_heads / spec->num_kv_heads;
}

int rocke_uattn2d_tiled_gfx1250_tile_size_eff(const rocke_uattn2d_tiled_gfx1250_spec_t* spec)
{
    return spec->tile_size_set ? spec->tile_size : spec->block_size;
}

int rocke_uattn2d_tiled_gfx1250_binary_search_iters(const rocke_uattn2d_tiled_gfx1250_spec_t* spec)
{
    int v;
    if(spec->num_seqs <= 0)
    {
        return 32;
    }
    v = (int)ceil(log2((double)(spec->num_seqs + 1)));
    return v < 1 ? 1 : v;
}

rocke_status_t rocke_uattn2d_tiled_gfx1250_kernel_name(
    const rocke_uattn2d_tiled_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char d_part[32];
    char b_part[32];
    char hkv_part[48];
    char sw_part[32];
    const char* parts[8];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(d_part, sizeof(d_part), "d%d", spec->head_size);
    snprintf(b_part, sizeof(b_part), "b%d", spec->block_size);
    snprintf(hkv_part, sizeof(hkv_part), "h%dkv%d", spec->num_query_heads, spec->num_kv_heads);
    if(spec->sliding_window > 0)
    {
        snprintf(sw_part, sizeof(sw_part), "sw%d", spec->sliding_window);
    }
    else
    {
        sw_part[0] = '\0';
    }
    parts[0] = "wmma16x16x32";
    parts[1] = d_part;
    parts[2] = b_part;
    parts[3] = hkv_part;
    parts[4] = spec->dtype;
    parts[5] = "kvfp8e4m3";
    parts[6] = spec->use_sinks ? "sinks" : "";
    parts[7] = sw_part;
    /* Trailing "regp"/"ldsP" is a 9th part (kernel_name_join drops empty ones). */
    {
        const char* allparts[9];
        int i;
        for(i = 0; i < 8; ++i)
        {
            allparts[i] = parts[i];
        }
        allparts[8] = spec->use_register_p ? "regp" : "ldsP";
        return rocke_kernel_name_join(
            "rocke_uattn2d_tiled_gfx1250", allparts, 9, NULL, NULL, 0, out, out_cap, NULL);
    }
}

bool rocke_uattn2d_tiled_gfx1250_supports(const rocke_uattn2d_tiled_gfx1250_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    int nqk;
    char buf[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }
    if(strcmp(arch, "gfx1250") != 0)
    {
        snprintf(
            buf, sizeof(buf), "gfx1250 tiled 2D only supports arch='gfx1250' (got '%s')", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->dtype == NULL || strcmp(spec->dtype, "bf16") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 2D currently supports bf16 Q/O only (got '%s')",
                 spec->dtype ? spec->dtype : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->head_size != ROCKE_G1250_HEAD_SIZE)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 2D currently supports head_size=64 (got %d)",
                 spec->head_size);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->block_size != ROCKE_U2D_BLOCK_SIZE)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 2D currently supports block_size=32 (got %d)",
                 spec->block_size);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    nqk = rocke_uattn2d_tiled_gfx1250_num_queries_per_kv(spec);
    if(nqk != ROCKE_U2D_NQK)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 2D currently supports GQA-8 (got num_queries_per_kv=%d)",
                 nqk);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->use_alibi)
    {
        rocke_spec_set_reason(reason, reason_cap, "gfx1250 tiled 2D does not support ALiBi yet");
        return false;
    }
    if(spec->use_qq_bias)
    {
        rocke_spec_set_reason(reason, reason_cap, "gfx1250 tiled 2D does not support QQ bias yet");
        return false;
    }
    if(spec->kv_storage_dtype == NULL || strcmp(spec->kv_storage_dtype, "fp8e4m3") != 0)
    {
        rocke_spec_set_reason(
            reason, reason_cap, "gfx1250 tiled 2D requires fp8e4m3 paged K/V cache");
        return false;
    }
    if(spec->num_warps != 1 || spec->block_m_per_warp != ROCKE_G1250_BLOCK_M)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 2D v1 is one wave32 CTA (num_warps=%d, block_m_per_warp=%d)",
                 spec->num_warps,
                 spec->block_m_per_warp);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->tile_size_set && spec->tile_size != spec->block_size)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 2D v1 consumes exactly one paged block per iteration "
                 "(tile_size=%d, block_size=%d)",
                 spec->tile_size,
                 spec->block_size);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!rocke_g1250_check_wmma_arch(arch, reason, reason_cap))
    {
        return false;
    }
    /* __post_init__ extras. */
    if(spec->has_softcap)
    {
        rocke_spec_set_reason(reason, reason_cap, "gfx1250 tiled 2D does not support softcap yet");
        return false;
    }
    if(spec->sliding_window < 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "sliding_window must be non-negative");
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "supported by gfx1250 WMMA tiled 2D v1");
    return true;
}

rocke_kernel_def_t* rocke_build_uattn2d_tiled_gfx1250(
    rocke_ir_builder_t* b, const rocke_uattn2d_tiled_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_mma_op_t* op;
        const rocke_layout_map_t* a_map;
        const rocke_layout_map_t* c_map;
        int a_frag;
        int c_frag;
        const rocke_type_t* dtype;
        int HD;
        int BS;
        int NQK;
        int NUM_QH;
        int SLIDING_WINDOW;
        int r;
        int d;
        char reason[ROCKE_ERR_MSG_CAP];

        rocke_value_t* output;
        rocke_value_t* query;
        rocke_value_t* key;
        rocke_value_t* value;
        rocke_value_t* sinks;
        rocke_value_t* block_tables;
        rocke_value_t* seq_lens;
        rocke_value_t* cu_q;
        rocke_value_t* scale;
        rocke_value_t* k_scale;
        rocke_value_t* v_scale;
        rocke_value_t* num_seqs;
        rocke_value_t* block_table_stride;

        rocke_value_t* kv_head_idx;
        rocke_value_t* q_block_global_idx;
        rocke_value_t* lane;
        rocke_value_t* seq_idx;
        rocke_value_t* cu_q_start;
        rocke_value_t* cu_q_stop;
        rocke_value_t* cur_batch_q_len;
        rocke_value_t* q_block_start_idx;
        rocke_value_t* q_block_local_idx;
        rocke_value_t* seq_len;
        rocke_value_t* context_len;
        rocke_value_t* qb_start_pos;
        rocke_value_t* lane_row;
        rocke_value_t* half_k;
        rocke_value_t* col;
        rocke_value_t* neg_inf;
        rocke_value_t* zero_f;
        rocke_value_t* one_f;
        rocke_value_t* rcp_ln2;
        rocke_value_t* qk_scale;
        rocke_value_t* q_pos_for_a;
        rocke_value_t* qh_for_a;
        rocke_value_t* q_valid_for_a;
        rocke_value_t* q_pos_safe;
        rocke_value_t* qh_safe;
        rocke_value_t* q_token;
        rocke_value_t* q_addr_row_base;
        rocke_value_t** q_frags;
        rocke_kv_desc_t kv_desc;
        rocke_value_t* P_lds;
        rocke_value_t* V_lds;
        rocke_value_t** m_inits;
        rocke_value_t* msp_raw;
        rocke_value_t* max_seq_prefix_len;
        rocke_value_t* num_tiles;
        rocke_value_t* tile_start;
        rocke_value_t* tile_end;
        rocke_iter_arg_t* iter_args;
        int n_iter;
        int n_acc;
        int bm1_div_nqk;
        rocke_for_t kloop;
        int pshape[2];
        int vshape[2];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }

        if(!rocke_uattn2d_tiled_gfx1250_supports(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
            return NULL;
        }

        op = rocke_g1250_resolve_wmma(b, arch, &a_map, &c_map, &a_frag, &c_frag);
        if(op == NULL || a_map == NULL || c_map == NULL)
        {
            return NULL;
        }

        dtype = rocke_bf16();
        HD = spec->head_size;
        BS = spec->block_size;
        NQK = rocke_uattn2d_tiled_gfx1250_num_queries_per_kv(spec);
        NUM_QH = spec->num_query_heads;
        SLIDING_WINDOW = spec->sliding_window;
        n_acc = HD / ROCKE_G1250_WMMA_N;

        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ROCKE_G1250_WAVE);
        if(spec->waves_per_eu_set)
        {
            rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
        }

        /* _declare_params(b) */
        {
            rocke_param_opts_t ro16; /* noalias readonly align16 */
            rocke_param_opts_t wo16; /* noalias writeonly align16 */
            rocke_param_opts_t ro4; /* readonly align4 (no noalias) */
            const rocke_type_t* p_bf16 = rocke_ptr_type(b, rocke_bf16(), "global");
            const rocke_type_t* p_fp8 = rocke_ptr_type(b, rocke_fp8e4m3(), "global");
            const rocke_type_t* p_i32 = rocke_ptr_type(b, rocke_i32(), "global");
            const rocke_type_t* p_f32 = rocke_ptr_type(b, rocke_f32(), "global");

            memset(&ro16, 0, sizeof(ro16));
            ro16.noalias = true;
            ro16.noalias_set = true;
            ro16.readonly = true;
            ro16.readonly_set = true;
            ro16.align = 16;
            ro16.align_set = true;
            memset(&wo16, 0, sizeof(wo16));
            wo16.noalias = true;
            wo16.noalias_set = true;
            wo16.writeonly = true;
            wo16.writeonly_set = true;
            wo16.align = 16;
            wo16.align_set = true;
            memset(&ro4, 0, sizeof(ro4));
            ro4.readonly = true;
            ro4.readonly_set = true;
            ro4.align = 4;
            ro4.align_set = true;

            output = rocke_b_param(b, "output_ptr", p_bf16, &wo16);
            query = rocke_b_param(b, "query_ptr", p_bf16, &ro16);
            key = rocke_b_param(b, "key_cache_ptr", p_fp8, &ro16);
            value = rocke_b_param(b, "value_cache_ptr", p_fp8, &ro16);
            {
                rocke_param_opts_t ro16s; /* sinks: readonly align16 (no noalias) */
                memset(&ro16s, 0, sizeof(ro16s));
                ro16s.readonly = true;
                ro16s.readonly_set = true;
                ro16s.align = 16;
                ro16s.align_set = true;
                sinks = rocke_b_param(b, "sink_ptr", p_bf16, &ro16s);
            }
            block_tables = rocke_b_param(b, "block_tables_ptr", p_i32, &ro4);
            seq_lens = rocke_b_param(b, "seq_lens_ptr", p_i32, &ro4);
            (void)rocke_b_param(b, "alibi_slopes_ptr", p_f32, &ro4);
            (void)rocke_b_param(b, "qq_bias_ptr", p_f32, &ro4);
            cu_q = rocke_b_param(b, "query_start_len_ptr", p_i32, &ro4);
            scale = rocke_b_param(b, "scale", rocke_f32(), NULL);
            k_scale = rocke_b_param(b, "k_scale", rocke_f32(), NULL);
            v_scale = rocke_b_param(b, "v_scale", rocke_f32(), NULL);
            (void)rocke_b_param(b, "out_scale", rocke_f32(), NULL);
            (void)rocke_b_param(b, "softcap", rocke_f32(), NULL);
            num_seqs = rocke_b_param(b, "num_seqs", rocke_i32(), NULL);
            block_table_stride = rocke_b_param(b, "block_table_stride", rocke_i32(), NULL);
            (void)rocke_b_param(b, "qq_bias_stride_0", rocke_i32(), NULL);
        }

        kv_head_idx = rocke_b_block_id_x(b);
        q_block_global_idx = rocke_b_block_id_y(b);
        /* lane = mod(thread_id_x(), const(WAVE)) — seq thread_id first */
        {
            rocke_value_t* t = rocke_b_thread_id_x(b);
            lane = rocke_b_mod(b, t, rocke_b_const_i32(b, ROCKE_G1250_WAVE));
        }

        seq_idx = rocke_binary_search_seq_idx(b,
                                              cu_q,
                                              q_block_global_idx,
                                              num_seqs,
                                              ROCKE_U2D_BLOCK_Q,
                                              rocke_uattn2d_tiled_gfx1250_binary_search_iters(spec),
                                              /*per_token=*/false);
        cu_q_start = rocke_b_global_load_i32(b, cu_q, seq_idx, /*align=*/-1);
        cu_q_stop = rocke_b_global_load_i32(
            b, cu_q, rocke_b_add(b, seq_idx, rocke_b_const_i32(b, 1)), /*align=*/-1);
        cur_batch_q_len = rocke_b_sub(b, cu_q_stop, cu_q_start);
        /* q_block_start_idx = add(div(cu_q_start, const(BLOCK_Q)), seq_idx) */
        q_block_start_idx = rocke_b_add(
            b, rocke_b_div(b, cu_q_start, rocke_b_const_i32(b, ROCKE_U2D_BLOCK_Q)), seq_idx);
        q_block_local_idx = rocke_b_sub(b, q_block_global_idx, q_block_start_idx);
        seq_len = rocke_b_global_load_i32(b, seq_lens, seq_idx, /*align=*/-1);
        context_len = rocke_b_sub(b, seq_len, cur_batch_q_len);

        qb_start_pos = rocke_b_mul(b, q_block_local_idx, rocke_b_const_i32(b, ROCKE_U2D_BLOCK_Q));
        /* with b.scf_if(cmp_ge(qb_start_pos, cur_batch_q_len)): b.ret() */
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_ge(b, qb_start_pos, cur_batch_q_len));
            rocke_b_region_enter(b, iff.then_region);
            rocke_b_ret(b);
            rocke_b_region_leave(b);
        }

        lane_row = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));
        /* half_k = a_map.coord(b, lane, 0)[1] */
        {
            rocke_value_t* c0 = NULL;
            rocke_value_t* c1 = NULL;
            rocke_layout_map_coord(a_map, b, lane, 0, &c0, &c1);
            half_k = c1;
        }
        col = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));
        neg_inf = rocke_b_const_f32(b, -1e30);
        zero_f = rocke_b_const_f32(b, 0.0);
        one_f = rocke_b_const_f32(b, 1.0);
        rcp_ln2 = rocke_b_const_f32(b, 1.4426950408889634);
        qk_scale = rocke_b_fmul(b, scale, rcp_ln2);

        /* q_pos_for_a = add(qb_start_pos, div(lane_row, const(NQK))) */
        q_pos_for_a
            = rocke_b_add(b, qb_start_pos, rocke_b_div(b, lane_row, rocke_b_const_i32(b, NQK)));
        /* qh_for_a = add(mul(kv_head_idx, const(NQK)), mod(lane_row, const(NQK))) */
        qh_for_a = rocke_u2d_qh(b, kv_head_idx, lane_row, NQK);
        /* q_valid_for_a = land(cmp_lt(q_pos_for_a, cur_batch_q_len),
         *                      cmp_lt(qh_for_a, const(NUM_QH))) */
        q_valid_for_a = rocke_u2d_row_valid(b, q_pos_for_a, cur_batch_q_len, qh_for_a, NUM_QH);
        q_pos_safe = rocke_b_select(b, q_valid_for_a, q_pos_for_a, rocke_b_const_i32(b, 0));
        qh_safe = rocke_b_select(b, q_valid_for_a, qh_for_a, rocke_b_const_i32(b, 0));

        /* q_token = add(cu_q_start, q_pos_safe) */
        q_token = rocke_b_add(b, cu_q_start, q_pos_safe);
        /* q_addr_row_base = mul(add(mul(q_token, const(NUM_QH)), qh_safe), const(HD))
         * The outer mul's arg1 is a side-effecting add-chain and arg2 is const(HD);
         * sequence so the const(HD) is emitted after the chain (Python order). */
        {
            rocke_value_t* inner
                = rocke_b_add(b, rocke_b_mul(b, q_token, rocke_b_const_i32(b, NUM_QH)), qh_safe);
            q_addr_row_base = rocke_b_mul(b, inner, rocke_b_const_i32(b, HD));
        }

        q_frags
            = (rocke_value_t**)calloc((size_t)(HD / ROCKE_G1250_WMMA_K), sizeof(rocke_value_t*));
        if(q_frags == NULL)
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }
        rocke_g1250_load_q_frags(
            b, query, q_addr_row_base, half_k, q_valid_for_a, HD, a_frag, dtype, q_frags);

        kv_desc.block_size = BS;
        kv_desc.stride_0 = BS * spec->num_kv_heads * HD;
        kv_desc.stride_1 = spec->num_kv_heads * HD;
        kv_desc.stride_2 = HD;
        kv_desc.stride_3 = 1;

        pshape[0] = ROCKE_G1250_BLOCK_M;
        pshape[1] = BS;
        P_lds = spec->use_register_p ? NULL
                                     : rocke_b_smem_alloc(b, dtype, pshape, 2, "Pgfx1250_uattn");
        vshape[0] = BS;
        vshape[1] = HD;
        V_lds = rocke_b_smem_alloc(b, dtype, vshape, 2, "Vgfx1250_uattn");

        /* m_inits/l_inits/acc_inits */
        m_inits = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
        if(m_inits == NULL)
        {
            free(q_frags);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }
        for(r = 0; r < c_frag; ++r)
        {
            rocke_value_t* row_rel = NULL;
            rocke_value_t* qh;
            rocke_value_t* qh_in;
            rocke_layout_map_coord(c_map, b, lane, r, &row_rel, NULL);
            qh = rocke_u2d_qh(b, kv_head_idx, row_rel, NQK);
            qh_in = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, NUM_QH));
            if(spec->use_sinks)
            {
                rocke_value_t* sink_h = rocke_b_global_load(b, sinks, qh, dtype, /*align=*/2);
                rocke_value_t* sink_f = rocke_b_fmul(b, rocke_b_cast_to_f32(b, sink_h), rcp_ln2);
                m_inits[r] = rocke_b_select(b, qh_in, sink_f, neg_inf);
            }
            else
            {
                m_inits[r] = neg_inf;
            }
        }

        /* iter_args: [m0,l0,...,m{c-1},l{c-1}, acc0..acc{n_acc-1}] */
        n_iter = 2 * c_frag + n_acc;
        iter_args = (rocke_iter_arg_t*)calloc((size_t)n_iter, sizeof(rocke_iter_arg_t));
        if(iter_args == NULL)
        {
            free(q_frags);
            free(m_inits);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }
        {
            static const char* const k_mnames[8] = {"m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"};
            static const char* const k_lnames[8] = {"l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"};
            static const char* const k_anames[8]
                = {"acc0", "acc1", "acc2", "acc3", "acc4", "acc5", "acc6", "acc7"};
            for(r = 0; r < c_frag; ++r)
            {
                iter_args[2 * r].name = k_mnames[r];
                iter_args[2 * r].init = m_inits[r];
                iter_args[2 * r + 1].name = k_lnames[r];
                iter_args[2 * r + 1].init = one_f;
            }
            for(d = 0; d < n_acc; ++d)
            {
                iter_args[2 * c_frag + d].name = k_anames[d];
                iter_args[2 * c_frag + d].init = rocke_b_zero_vec_f32(b, c_frag);
            }
        }

        /* tile-range bounding */
        bm1_div_nqk = (ROCKE_G1250_BLOCK_M - 1) / NQK;
        /* msp_raw = add(add(context_len, qb_start_pos), const(bm1_div_nqk+1)).
         * Sequence the inner add before the trailing const. */
        {
            rocke_value_t* cq = rocke_b_add(b, context_len, qb_start_pos);
            msp_raw = rocke_b_add(b, cq, rocke_b_const_i32(b, bm1_div_nqk + 1));
        }
        max_seq_prefix_len
            = rocke_b_select(b, rocke_b_cmp_lt(b, msp_raw, seq_len), msp_raw, seq_len);
        /* num_tiles = div(add(max_seq_prefix_len, const(BS-1)), const(BS)).
         * Sequence the inner add before the divisor const. */
        {
            rocke_value_t* msp1 = rocke_b_add(b, max_seq_prefix_len, rocke_b_const_i32(b, BS - 1));
            num_tiles = rocke_b_div(b, msp1, rocke_b_const_i32(b, BS));
        }
        if(SLIDING_WINDOW > 0)
        {
            rocke_value_t* qpos_hi_raw
                = rocke_b_add(b, qb_start_pos, rocke_b_const_i32(b, bm1_div_nqk));
            rocke_value_t* cur_q_minus1 = rocke_b_sub(b, cur_batch_q_len, rocke_b_const_i32(b, 1));
            rocke_value_t* qpos_hi = rocke_b_select(
                b, rocke_b_cmp_lt(b, qpos_hi_raw, cur_q_minus1), qpos_hi_raw, cur_q_minus1);
            /* first_allowed_key = add(sub(add(context_len, qb_start_pos), const(SW)), const(1))
             * Sequence each trailing const after its side-effecting arg1. */
            rocke_value_t* fak_cq = rocke_b_add(b, context_len, qb_start_pos);
            rocke_value_t* fak_inner = rocke_b_sub(b, fak_cq, rocke_b_const_i32(b, SLIDING_WINDOW));
            rocke_value_t* first_allowed_key = rocke_b_add(b, fak_inner, rocke_b_const_i32(b, 1));
            rocke_value_t* last_allowed_key = rocke_b_add(b, context_len, qpos_hi);
            rocke_value_t* tile_start_raw
                = rocke_b_div(b, first_allowed_key, rocke_b_const_i32(b, BS));
            rocke_value_t* tile_end_div;
            rocke_value_t* tile_end_raw;
            /* tile_start = select(cmp_lt(tile_start_raw, const(0)), const(0), tile_start_raw).
             * Sequence the cmp's const(0) before the select's const(0) operand. */
            {
                rocke_value_t* ts_lt = rocke_b_cmp_lt(b, tile_start_raw, rocke_b_const_i32(b, 0));
                tile_start = rocke_b_select(b, ts_lt, rocke_b_const_i32(b, 0), tile_start_raw);
            }
            tile_end_div = rocke_b_div(b, last_allowed_key, rocke_b_const_i32(b, BS));
            tile_end_raw = rocke_b_add(b, tile_end_div, rocke_b_const_i32(b, 1));
            {
                rocke_value_t* te_lt = rocke_b_cmp_lt(b, tile_end_raw, num_tiles);
                tile_end = rocke_b_select(b, te_lt, tile_end_raw, num_tiles);
            }
        }
        else
        {
            tile_start = rocke_b_const_i32(b, 0);
            tile_end = num_tiles;
        }

        /* kloop = b.scf_for_iter(tile_start, tile_end, const(1), iter_args, iv_name="kt") */
        kloop = rocke_b_scf_for_iter(b,
                                     tile_start,
                                     tile_end,
                                     rocke_b_const_i32(b, 1),
                                     iter_args,
                                     n_iter,
                                     "kt",
                                     /*unroll=*/false,
                                     /*elide_trailing_barrier=*/true);

        rocke_b_region_enter(b, kloop.body);
        {
            rocke_value_t* kt = kloop.iv;
            rocke_value_t** state = kloop.iter_vars;
            rocke_value_t** ms = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** ls = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** new_accs
                = (rocke_value_t**)calloc((size_t)n_acc, sizeof(rocke_value_t*));
            rocke_value_t** new_ms
                = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** new_ls
                = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** ps0 = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** ps1 = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t* tile_base;
            rocke_value_t* physical_block;
            rocke_value_t* scores[2];
            rocke_value_t** yields;
            int yi;

            if(!ms || !ls || !new_accs || !new_ms || !new_ls || !ps0 || !ps1)
            {
                free(ms);
                free(ls);
                free(new_accs);
                free(new_ms);
                free(new_ls);
                free(ps0);
                free(ps1);
                free(q_frags);
                free(m_inits);
                free(iter_args);
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
                return NULL;
            }
            for(r = 0; r < c_frag; ++r)
            {
                ms[r] = state[2 * r];
                ls[r] = state[2 * r + 1];
            }
            for(d = 0; d < n_acc; ++d)
            {
                new_accs[d] = state[2 * c_frag + d];
            }

            /* tile_base = mul(kt, const(BS)) */
            tile_base = rocke_b_mul(b, kt, rocke_b_const_i32(b, BS));
            /* physical_block = global_load_i32(block_tables,
             *                     add(mul(seq_idx, block_table_stride), kt)) */
            physical_block = rocke_b_global_load_i32(
                b,
                block_tables,
                rocke_b_add(b, rocke_b_mul(b, seq_idx, block_table_stride), kt),
                /*align=*/-1);

            /* scores = compute_qk_scores(..., phys_block=cached) */
            rocke_g1250_compute_qk_scores(b,
                                          q_frags,
                                          key,
                                          &kv_desc,
                                          tile_base,
                                          lane_row,
                                          half_k,
                                          kv_head_idx,
                                          BS,
                                          HD,
                                          rocke_fp8e4m3(),
                                          k_scale,
                                          dtype,
                                          c_frag,
                                          rocke_u2d_phys_block_cached,
                                          (void*)physical_block,
                                          /*spacing=*/0,
                                          scores);

            for(r = 0; r < c_frag; ++r)
            {
                rocke_value_t* row_rel = NULL;
                rocke_value_t* col_k = NULL;
                rocke_value_t* q_pos;
                rocke_value_t* qh;
                rocke_value_t* row_valid;
                rocke_value_t* srs[2];
                rocke_value_t* m_new = NULL;
                rocke_value_t* l_new = NULL;
                rocke_value_t* alpha = NULL;
                rocke_value_t* p[2];
                int nsub;

                rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_k);
                q_pos = rocke_b_add(
                    b, qb_start_pos, rocke_b_div(b, row_rel, rocke_b_const_i32(b, NQK)));
                qh = rocke_u2d_qh(b, kv_head_idx, row_rel, NQK);
                row_valid = rocke_u2d_row_valid(b, q_pos, cur_batch_q_len, qh, NUM_QH);
                for(nsub = 0; nsub < 2; ++nsub)
                {
                    /* key_pos = add(add(tile_base, const(nsub*WMMA_N)), col_k) */
                    rocke_value_t* key_pos = rocke_b_add(
                        b,
                        rocke_b_add(b, tile_base, rocke_b_const_i32(b, nsub * ROCKE_G1250_WMMA_N)),
                        col_k);
                    /* score_log2 = fmul(vec_extract(scores[nsub], r), qk_scale) */
                    rocke_value_t* score_log2
                        = rocke_b_fmul(b, rocke_b_vec_extract(b, scores[nsub], r), qk_scale);
                    /* causal_keep = cmp_le(key_pos, add(context_len, q_pos)) */
                    rocke_value_t* causal_keep
                        = rocke_b_cmp_le(b, key_pos, rocke_b_add(b, context_len, q_pos));
                    rocke_value_t* in_seq = rocke_b_cmp_lt(b, key_pos, seq_len);
                    /* keep = land(row_valid, land(in_seq, causal_keep)) */
                    rocke_value_t* keep
                        = rocke_b_land(b, row_valid, rocke_b_land(b, in_seq, causal_keep));
                    if(SLIDING_WINDOW > 0)
                    {
                        /* dist = sub(add(context_len, q_pos), key_pos) */
                        rocke_value_t* dist
                            = rocke_b_sub(b, rocke_b_add(b, context_len, q_pos), key_pos);
                        keep = rocke_b_land(
                            b, keep, rocke_b_cmp_lt(b, dist, rocke_b_const_i32(b, SLIDING_WINDOW)));
                    }
                    srs[nsub] = rocke_b_select(b, keep, score_log2, neg_inf);
                }

                rocke_g1250_softmax_row_update(b,
                                               ms[r],
                                               ls[r],
                                               srs,
                                               neg_inf,
                                               zero_f,
                                               /*use_dpp=*/false,
                                               &m_new,
                                               &l_new,
                                               &alpha,
                                               p);
                new_ms[r] = m_new;
                new_ls[r] = l_new;
                ps0[r] = p[0];
                ps1[r] = p[1];
                for(d = 0; d < n_acc; ++d)
                {
                    rocke_value_t* old = rocke_b_vec_extract(b, new_accs[d], r);
                    new_accs[d]
                        = rocke_b_vec_insert(b, new_accs[d], rocke_b_fmul(b, old, alpha), r);
                }
            }

            if(!spec->use_register_p)
            {
                for(r = 0; r < c_frag; ++r)
                {
                    rocke_value_t* row_rel = NULL;
                    rocke_value_t* col_k = NULL;
                    rocke_value_t* idx[2];
                    rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_k);
                    idx[0] = row_rel;
                    idx[1] = col_k;
                    rocke_b_smem_store_vN(
                        b, P_lds, idx, 2, rocke_b_cast_f32_to(b, ps0[r], dtype), 1);
                    idx[1] = rocke_b_add(b, col_k, rocke_b_const_i32(b, ROCKE_G1250_WMMA_N));
                    rocke_b_smem_store_vN(
                        b, P_lds, idx, 2, rocke_b_cast_f32_to(b, ps1[r], dtype), 1);
                }
            }

            rocke_g1250_stage_v_tile(b,
                                     V_lds,
                                     value,
                                     &kv_desc,
                                     kv_head_idx,
                                     tile_base,
                                     lane,
                                     BS,
                                     HD,
                                     rocke_fp8e4m3(),
                                     v_scale,
                                     dtype,
                                     rocke_u2d_phys_block_cached,
                                     (void*)physical_block);
            rocke_b_sync(b);

            if(spec->use_register_p)
            {
                rocke_g1250_compute_pv_from_probs(b,
                                                  ps0,
                                                  ps1,
                                                  V_lds,
                                                  new_accs,
                                                  a_map,
                                                  lane,
                                                  col,
                                                  a_frag,
                                                  c_frag,
                                                  HD,
                                                  dtype,
                                                  /*v_extra_idx=*/NULL,
                                                  /*spacing=*/0);
            }
            else
            {
                rocke_g1250_compute_pv(b,
                                       P_lds,
                                       V_lds,
                                       new_accs,
                                       a_map,
                                       lane,
                                       lane_row,
                                       col,
                                       a_frag,
                                       c_frag,
                                       HD,
                                       dtype,
                                       /*v_extra_idx=*/NULL,
                                       /*p_extra_idx=*/NULL,
                                       /*spacing=*/0);
            }

            /* yields */
            yields = (rocke_value_t**)calloc((size_t)n_iter, sizeof(rocke_value_t*));
            if(yields == NULL)
            {
                free(ms);
                free(ls);
                free(new_accs);
                free(new_ms);
                free(new_ls);
                free(ps0);
                free(ps1);
                free(q_frags);
                free(m_inits);
                free(iter_args);
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
                return NULL;
            }
            yi = 0;
            for(r = 0; r < c_frag; ++r)
            {
                yields[yi++] = new_ms[r];
                yields[yi++] = new_ls[r];
            }
            for(d = 0; d < n_acc; ++d)
            {
                yields[yi++] = new_accs[d];
            }
            rocke_b_scf_yield(b, yields, n_iter);

            free(ms);
            free(ls);
            free(new_accs);
            free(new_ms);
            free(new_ls);
            free(ps0);
            free(ps1);
            free(yields);
        }
        rocke_b_region_leave(b);

        if(!rocke_ir_builder_ok(b) || kloop.op == NULL || kloop.op->num_results < n_iter)
        {
            free(q_frags);
            free(m_inits);
            free(iter_args);
            return NULL;
        }

        /* Epilogue */
        {
            rocke_value_t** final = kloop.op->results;
            rocke_value_t** ls_final
                = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** accs_final
                = (rocke_value_t**)calloc((size_t)n_acc, sizeof(rocke_value_t*));
            if(!ls_final || !accs_final)
            {
                free(ls_final);
                free(accs_final);
                free(q_frags);
                free(m_inits);
                free(iter_args);
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
                return NULL;
            }
            for(r = 0; r < c_frag; ++r)
            {
                ls_final[r] = final[2 * r + 1];
            }
            for(d = 0; d < n_acc; ++d)
            {
                accs_final[d] = final[2 * c_frag + d];
            }

            for(d = 0; d < n_acc; ++d)
            {
                for(r = 0; r < c_frag; ++r)
                {
                    rocke_value_t* row_rel = NULL;
                    rocke_value_t* col_n = NULL;
                    rocke_value_t* q_pos;
                    rocke_value_t* qh;
                    rocke_value_t* out_valid;
                    rocke_value_t* l_safe;
                    rocke_value_t* inv_l;
                    rocke_value_t* v_f32;
                    rocke_value_t* out_token;
                    rocke_value_t* o_col;
                    rocke_value_t* out_addr;

                    rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_n);
                    q_pos = rocke_b_add(
                        b, qb_start_pos, rocke_b_div(b, row_rel, rocke_b_const_i32(b, NQK)));
                    qh = rocke_u2d_qh(b, kv_head_idx, row_rel, NQK);
                    out_valid = rocke_u2d_row_valid(b, q_pos, cur_batch_q_len, qh, NUM_QH);
                    l_safe = ls_final[r];
                    /* inv_l = select(fcmp oeq(l_safe, zero), zero, rcp(l_safe))
                     * Python evals the fcmp (cond) before rcp; sequence to match. */
                    {
                        rocke_value_t* zmask = rocke_b_fcmp(b, "oeq", l_safe, zero_f);
                        rocke_value_t* rl = rocke_b_rcp(b, l_safe);
                        inv_l = rocke_b_select(b, zmask, zero_f, rl);
                    }
                    v_f32 = rocke_b_fmul(b, rocke_b_vec_extract(b, accs_final[d], r), inv_l);
                    out_token = rocke_b_add(b, cu_q_start, q_pos);
                    o_col = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_G1250_WMMA_N), col_n);
                    /* out_addr = add(mul(add(mul(out_token, const(NUM_QH)), qh), const(HD)), o_col)
                     * Sequence the inner mul-by-HD's const after its add-chain arg1. */
                    {
                        rocke_value_t* inner = rocke_b_add(
                            b, rocke_b_mul(b, out_token, rocke_b_const_i32(b, NUM_QH)), qh);
                        rocke_value_t* scaled = rocke_b_mul(b, inner, rocke_b_const_i32(b, HD));
                        out_addr = rocke_b_add(b, scaled, o_col);
                    }
                    {
                        rocke_if_t iff = rocke_b_scf_if(b, out_valid);
                        rocke_b_region_enter(b, iff.then_region);
                        rocke_b_global_store(
                            b, output, out_addr, rocke_b_cast_f32_to(b, v_f32, dtype), /*align=*/2);
                        rocke_b_region_leave(b);
                    }
                }
            }
            free(ls_final);
            free(accs_final);
        }

        rocke_b_ret(b);

        free(q_frags);
        free(m_inits);
        free(iter_args);

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_uattn2d_tiled_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_uattn2d_tiled_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_uattn2d_tiled_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_uattn2d_tiled_gfx1250(b, spec, arch);
    });
}

rocke_status_t
    rocke_uattn2d_tiled_gfx1250_lower_to_llvm(const rocke_uattn2d_tiled_gfx1250_spec_t* spec,
                                              const char* arch,
                                              rocke_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }

    kernel = rocke_build_uattn2d_tiled_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n = m ? strlen(m) : 0;
            if(m == NULL)
            {
                m = "build_uattn2d_tiled_gfx1250 failed";
                n = strlen(m);
            }
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
