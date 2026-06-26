// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_attention_tiled_3d.cpp -- C99 port of
 * rocke/instances/gfx1250/attention_tiled_3d.py.
 *
 * gfx1250 WMMA split-KV 3D decode attention: segment kernel (split-KV per-
 * segment partial computation) and reduce kernel (merges segment partials).
 * The build op order tracks the Python source top-to-bottom; emitted IR is
 * byte-identical to the Python lowerer (args sequenced left-to-right,
 * const_i32 never deduped).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_attention_tiled_3d.h"
#include "rocke/instance_gfx1250_wmma_attention_common.h"
#include "rocke/ir_internal.h"

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define T3D 32 /* tokens per KV tile */

/* --------------------------------------------------------- helpers */

/* qh = add(mul(kv_head_idx, const(NQK)), mod(row, const(NQK))). */
static rocke_value_t*
    u3d_qh(rocke_ir_builder_t* b, rocke_value_t* kv_head_idx, rocke_value_t* row, int nqk)
{
    rocke_value_t* m = rocke_b_mul(b, kv_head_idx, rocke_b_const_i32(b, nqk));
    rocke_value_t* md = rocke_b_mod(b, row, rocke_b_const_i32(b, nqk));
    return rocke_b_add(b, m, md);
}

/* land(cmp_lt(q_pos, qlen), cmp_lt(qh, NUM_QH)). */
static rocke_value_t* u3d_row_valid(rocke_ir_builder_t* b,
                                    rocke_value_t* q_pos,
                                    rocke_value_t* q_len,
                                    rocke_value_t* qh,
                                    int num_qh)
{
    rocke_value_t* a = rocke_b_cmp_lt(b, q_pos, q_len);
    rocke_value_t* c = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, num_qh));
    return rocke_b_land(b, a, c);
}

/* 3D phys_block closure: per-token block-table lookup. */
typedef struct u3d_phys_ctx
{
    rocke_value_t* block_tables;
    rocke_value_t* seq_idx;
    rocke_value_t* block_table_stride;
    int block_size;
} u3d_phys_ctx_t;

static rocke_value_t* u3d_phys_block(rocke_ir_builder_t* b, rocke_value_t* tok, void* ctx)
{
    u3d_phys_ctx_t* c = (u3d_phys_ctx_t*)ctx;
    rocke_value_t* logical = rocke_b_div(b, tok, rocke_b_const_i32(b, c->block_size));
    return rocke_b_global_load_i32(
        b,
        c->block_tables,
        rocke_b_add(b, rocke_b_mul(b, c->seq_idx, c->block_table_stride), logical),
        -1);
}

/* ============================================================
 * Segment spec default / helpers
 * ============================================================ */

rocke_uattn3d_seg_gfx1250_spec_t rocke_uattn3d_seg_gfx1250_spec_default(void)
{
    rocke_uattn3d_seg_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.num_seqs = 0;
    s.waves_per_eu_set = false;
    s.kv_storage_dtype = NULL; /* bf16 */
    s.tile_size_override_set = false;
    s.use_invariant_hoist = false;
    s.use_wide_kv_load = false;
    s.use_register_p = false;
    s.wmma_spacing = 0;
    s.num_waves = 1;
    s.use_wide_lds_reads = true;
    s.use_dtla_prefetch = false;
    s.use_ds_tr_reads = false;
    s.use_sw_pipeline = false;
    s.ablate_softmax = false;
    s.ablate_pv = false;
    s.use_fused_reduce = false;
    s.use_dpp_softmax = true;
    return s;
}

int rocke_uattn3d_seg_gfx1250_num_queries_per_kv(const rocke_uattn3d_seg_gfx1250_spec_t* spec)
{
    return spec->num_query_heads / spec->num_kv_heads;
}

int rocke_uattn3d_seg_gfx1250_block_q(const rocke_uattn3d_seg_gfx1250_spec_t* spec)
{
    return ROCKE_G1250_BLOCK_M / rocke_uattn3d_seg_gfx1250_num_queries_per_kv(spec);
}

int rocke_uattn3d_seg_gfx1250_binary_search_iters(const rocke_uattn3d_seg_gfx1250_spec_t* spec)
{
    int v;
    if(spec->num_seqs <= 0)
    {
        return 32;
    }
    v = (int)ceil(log2((double)(spec->num_seqs + 1)));
    return v < 32 ? 32 : v;
}

rocke_status_t rocke_uattn3d_seg_gfx1250_kernel_name(const rocke_uattn3d_seg_gfx1250_spec_t* spec,
                                                     char* out,
                                                     size_t out_cap)
{
    char d_part[32];
    char b_part[32];
    char hkv_part[48];
    char kv_part[32];
    char seg_part[32];
    char sw_part[32];
    char nop_part[32];
    char mw_part[32];
    const char* allparts[18];
    int nparts = 0;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(d_part, sizeof(d_part), "d%d", spec->head_size);
    snprintf(b_part, sizeof(b_part), "b%d", spec->block_size);
    snprintf(hkv_part, sizeof(hkv_part), "h%dkv%d", spec->num_query_heads, spec->num_kv_heads);
    if(spec->kv_storage_dtype != NULL && strcmp(spec->kv_storage_dtype, "bf16") != 0)
    {
        snprintf(kv_part, sizeof(kv_part), "kv%s", spec->kv_storage_dtype);
    }
    else
    {
        strcpy(kv_part, "kvbf16");
    }
    snprintf(seg_part, sizeof(seg_part), "seg%d", spec->num_segments);
    if(spec->sliding_window > 0)
    {
        snprintf(sw_part, sizeof(sw_part), "sw%d", spec->sliding_window);
    }
    else
    {
        sw_part[0] = '\0';
    }
    /* __post_init__ forces wmma_spacing>=1 when use_dpp_softmax (default True);
     * mirror that effective value here so the kernel name matches Python. */
    {
        int eff_spacing = spec->wmma_spacing;
        if(spec->use_dpp_softmax && eff_spacing < 1)
        {
            eff_spacing = 1;
        }
        if(eff_spacing > 0)
        {
            snprintf(nop_part, sizeof(nop_part), "nop%d", eff_spacing);
        }
        else
        {
            nop_part[0] = '\0';
        }
    }
    if(spec->num_waves > 1)
    {
        snprintf(mw_part, sizeof(mw_part), "mw%d", spec->num_waves);
    }
    else
    {
        mw_part[0] = '\0';
    }

    allparts[nparts++] = "wmma16x16x32";
    allparts[nparts++] = d_part;
    allparts[nparts++] = b_part;
    allparts[nparts++] = hkv_part;
    allparts[nparts++] = spec->dtype;
    allparts[nparts++] = kv_part;
    allparts[nparts++] = seg_part;
    allparts[nparts++] = spec->use_sinks ? "sinks" : "";
    allparts[nparts++] = sw_part;
    allparts[nparts++] = spec->use_invariant_hoist ? "hoist" : "";
    allparts[nparts++] = spec->use_wide_kv_load ? "wkvb" : "";
    allparts[nparts++] = spec->use_register_p ? "regp" : "ldsP";
    allparts[nparts++] = nop_part;
    allparts[nparts++] = mw_part;
    allparts[nparts++] = spec->use_wide_lds_reads ? "wlds" : "";
    allparts[nparts++] = spec->use_ds_tr_reads ? "dstr" : "";
    allparts[nparts++] = spec->use_dtla_prefetch ? "dtla" : "";
    allparts[nparts++] = spec->use_sw_pipeline ? "swp" : "";
    /* note: "fred" from use_fused_reduce is appended last. */
    {
        const char* final_parts[20];
        int fp = 0;
        int i;
        for(i = 0; i < nparts; ++i)
            final_parts[fp++] = allparts[i];
        final_parts[fp++] = spec->use_fused_reduce ? "fred" : "";
        return rocke_kernel_name_join("rocke_uattn3d_seg_gfx1250",
                                      final_parts,
                                      (size_t)fp,
                                      NULL,
                                      NULL,
                                      0,
                                      out,
                                      out_cap,
                                      NULL);
    }
}

bool rocke_uattn3d_seg_gfx1250_supports(const rocke_uattn3d_seg_gfx1250_spec_t* spec,
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
            buf, sizeof(buf), "gfx1250 tiled 3D only supports arch='gfx1250' (got '%s')", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->dtype == NULL || strcmp(spec->dtype, "bf16") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 3D currently supports bf16 Q/O only (got '%s')",
                 spec->dtype ? spec->dtype : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->head_size != ROCKE_G1250_HEAD_SIZE)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 3D currently supports head_size=64 (got %d)",
                 spec->head_size);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->block_size != 16 && spec->block_size != 32)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 3D supports block_size in {16,32} (got %d)",
                 spec->block_size);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    nqk = rocke_uattn3d_seg_gfx1250_num_queries_per_kv(spec);
    if(nqk != 8)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 3D currently supports GQA-8 (got num_queries_per_kv=%d)",
                 nqk);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->use_alibi)
    {
        rocke_spec_set_reason(reason, reason_cap, "gfx1250 tiled 3D does not support ALiBi yet");
        return false;
    }
    if(spec->use_qq_bias)
    {
        rocke_spec_set_reason(reason, reason_cap, "gfx1250 tiled 3D does not support QQ bias yet");
        return false;
    }
    if(spec->kv_storage_dtype != NULL && strcmp(spec->kv_storage_dtype, "bf16") != 0
       && strcmp(spec->kv_storage_dtype, "fp8e4m3") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "gfx1250 tiled 3D supports bf16/fp8e4m3 KV cache (got '%s')",
                 spec->kv_storage_dtype);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!rocke_g1250_check_wmma_arch(arch, reason, reason_cap))
    {
        return false;
    }
    if(spec->has_softcap)
    {
        rocke_spec_set_reason(reason, reason_cap, "gfx1250 tiled 3D does not support softcap yet");
        return false;
    }
    if(spec->num_segments < 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "num_segments must be >= 1");
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "supported by gfx1250 WMMA tiled 3D v1");
    return true;
}

/* ============================================================
 * build_unified_attention_3d_tiled (segment kernel)
 * ============================================================ */

rocke_kernel_def_t* rocke_build_uattn3d_seg_gfx1250(rocke_ir_builder_t* b,
                                                    const rocke_uattn3d_seg_gfx1250_spec_t* spec,
                                                    const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_mma_op_t* op;
        const rocke_layout_map_t* a_map;
        const rocke_layout_map_t* c_map;
        int a_frag;
        int c_frag;
        const rocke_type_t* dtype;
        const rocke_type_t* kv_dtype;
        int HD, BS, NQK, NUM_QH, NUM_SEG, SLIDING_WINDOW, BLOCK_Q, NUM_WAVES;
        int n_acc;
        int r, d;
        int wmma_spacing;
        char reason[ROCKE_ERR_MSG_CAP];

        rocke_value_t* segm_output;
        rocke_value_t* segm_max;
        rocke_value_t* segm_expsum;
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

        rocke_value_t* q_block_global_idx;
        rocke_value_t* kv_head_idx;
        rocke_value_t* seg_idx;
        rocke_value_t* tid;
        rocke_value_t* lane;
        rocke_value_t* wave_id;
        rocke_value_t* neg_inf;
        rocke_value_t* zero_f;
        rocke_value_t* one_f;
        rocke_value_t* rcp_ln2;
        rocke_value_t* qk_scale;
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
        rocke_value_t* msp_raw;
        rocke_value_t* max_seq_prefix_len;
        rocke_value_t* num_tiles;
        rocke_value_t* tps;
        rocke_value_t* tile_start;
        rocke_value_t* tile_end;
        rocke_value_t* q_pos_for_a;
        rocke_value_t* qh_for_a;
        rocke_value_t* q_valid_for_a;
        rocke_value_t* q_pos_safe;
        rocke_value_t* qh_safe;
        rocke_value_t* q_token;
        rocke_value_t* q_addr_row_base;
        rocke_value_t** q_frags;
        rocke_kv_desc_t kv_desc;
        u3d_phys_ctx_t phys_ctx;
        rocke_value_t* P_lds;
        rocke_value_t* V_lds;
        rocke_value_t** m_inits;
        rocke_value_t** l_inits;
        rocke_iter_arg_t* iter_args;
        int n_iter;
        int bm1_div_nqk;
        rocke_for_t kloop;

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }
        if(!rocke_uattn3d_seg_gfx1250_supports(spec, arch, reason, sizeof(reason)))
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
        kv_dtype = rocke_g1250_kv_storage_ir(spec->kv_storage_dtype);
        if(kv_dtype == NULL)
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported kv_storage_dtype");
            return NULL;
        }
        HD = spec->head_size;
        BS = spec->block_size;
        NQK = rocke_uattn3d_seg_gfx1250_num_queries_per_kv(spec);
        NUM_QH = spec->num_query_heads;
        NUM_SEG = spec->num_segments;
        SLIDING_WINDOW = spec->sliding_window;
        BLOCK_Q = rocke_uattn3d_seg_gfx1250_block_q(spec);
        NUM_WAVES = spec->num_waves;
        wmma_spacing = spec->wmma_spacing;
        /* __post_init__: if use_dpp_softmax && wmma_spacing < 1, force 1 */
        if(spec->use_dpp_softmax && wmma_spacing < 1)
        {
            wmma_spacing = 1;
        }
        n_acc = HD / ROCKE_G1250_WMMA_N;

        rocke_attr_set_int(
            b, &b->kernel->attrs, "max_workgroup_size", ROCKE_G1250_WAVE * NUM_WAVES);
        if(spec->waves_per_eu_set)
        {
            rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
        }

        /* _seg_declare_params */
        {
            rocke_param_opts_t nawo16; /* noalias writeonly align16 */
            rocke_param_opts_t nawo4; /* noalias writeonly align4 */
            rocke_param_opts_t naro16; /* noalias readonly align16 */
            rocke_param_opts_t ro16; /* readonly align16 (no noalias) */
            rocke_param_opts_t ro4; /* readonly align4 */
            const rocke_type_t* p_bf16 = rocke_ptr_type(b, rocke_bf16(), "global");
            const rocke_type_t* p_kv = rocke_ptr_type(b, kv_dtype, "global");
            const rocke_type_t* p_i32 = rocke_ptr_type(b, rocke_i32(), "global");
            const rocke_type_t* p_f32 = rocke_ptr_type(b, rocke_f32(), "global");

            memset(&nawo16, 0, sizeof(nawo16));
            nawo16.noalias = true;
            nawo16.noalias_set = true;
            nawo16.writeonly = true;
            nawo16.writeonly_set = true;
            nawo16.align = 16;
            nawo16.align_set = true;

            memset(&nawo4, 0, sizeof(nawo4));
            nawo4.noalias = true;
            nawo4.noalias_set = true;
            nawo4.writeonly = true;
            nawo4.writeonly_set = true;
            nawo4.align = 4;
            nawo4.align_set = true;

            memset(&naro16, 0, sizeof(naro16));
            naro16.noalias = true;
            naro16.noalias_set = true;
            naro16.readonly = true;
            naro16.readonly_set = true;
            naro16.align = 16;
            naro16.align_set = true;

            memset(&ro16, 0, sizeof(ro16));
            ro16.readonly = true;
            ro16.readonly_set = true;
            ro16.align = 16;
            ro16.align_set = true;

            memset(&ro4, 0, sizeof(ro4));
            ro4.readonly = true;
            ro4.readonly_set = true;
            ro4.align = 4;
            ro4.align_set = true;

            segm_output = rocke_b_param(b, "segm_output_ptr", p_f32, &nawo16);
            segm_max = rocke_b_param(b, "segm_max_ptr", p_f32, &nawo4);
            segm_expsum = rocke_b_param(b, "segm_expsum_ptr", p_f32, &nawo4);
            query = rocke_b_param(b, "query_ptr", p_bf16, &naro16);
            key = rocke_b_param(b, "key_cache_ptr", p_kv, &naro16);
            value = rocke_b_param(b, "value_cache_ptr", p_kv, &naro16);
            sinks = rocke_b_param(b, "sink_ptr", p_bf16, &ro16);
            block_tables = rocke_b_param(b, "block_tables_ptr", p_i32, &ro4);
            seq_lens = rocke_b_param(b, "seq_lens_ptr", p_i32, &ro4);
            (void)rocke_b_param(b, "alibi_slopes_ptr", p_f32, &ro4);
            (void)rocke_b_param(b, "qq_bias_ptr", p_f32, &ro4);
            cu_q = rocke_b_param(b, "query_start_len_ptr", p_i32, &ro4);
            scale = rocke_b_param(b, "scale", rocke_f32(), NULL);
            k_scale = rocke_b_param(b, "k_scale", rocke_f32(), NULL);
            v_scale = rocke_b_param(b, "v_scale", rocke_f32(), NULL);
            (void)rocke_b_param(b, "softcap", rocke_f32(), NULL);
            num_seqs = rocke_b_param(b, "num_seqs", rocke_i32(), NULL);
            block_table_stride = rocke_b_param(b, "block_table_stride", rocke_i32(), NULL);
            (void)rocke_b_param(b, "qq_bias_stride_0", rocke_i32(), NULL);
        }

        q_block_global_idx = rocke_b_block_id_x(b);
        kv_head_idx = rocke_b_block_id_y(b);
        seg_idx = rocke_b_block_id_z(b);
        tid = rocke_b_thread_id_x(b);
        lane = rocke_b_mod(b, tid, rocke_b_const_i32(b, ROCKE_G1250_WAVE));
        wave_id = rocke_b_div(b, tid, rocke_b_const_i32(b, ROCKE_G1250_WAVE));
        (void)wave_id; /* emitted unconditionally (Python parity); used only on the
                        * NUM_WAVES>1 path which the default configs do not select */

        neg_inf = rocke_b_const_f32(b, -1e30);
        zero_f = rocke_b_const_f32(b, 0.0);
        one_f = rocke_b_const_f32(b, 1.0);
        rcp_ln2 = rocke_b_const_f32(b, 1.4426950408889634);
        qk_scale = rocke_b_fmul(b, scale, rcp_ln2);

        seq_idx = rocke_binary_search_seq_idx(b,
                                              cu_q,
                                              q_block_global_idx,
                                              num_seqs,
                                              BLOCK_Q,
                                              rocke_uattn3d_seg_gfx1250_binary_search_iters(spec),
                                              false);
        cu_q_start = rocke_b_global_load_i32(b, cu_q, seq_idx, -1);
        cu_q_stop = rocke_b_global_load_i32(
            b, cu_q, rocke_b_add(b, seq_idx, rocke_b_const_i32(b, 1)), -1);
        cur_batch_q_len = rocke_b_sub(b, cu_q_stop, cu_q_start);
        /* q_block_start_idx = add(div(cu_q_start, const(BLOCK_Q)), seq_idx) */
        q_block_start_idx
            = rocke_b_add(b, rocke_b_div(b, cu_q_start, rocke_b_const_i32(b, BLOCK_Q)), seq_idx);
        q_block_local_idx = rocke_b_sub(b, q_block_global_idx, q_block_start_idx);
        seq_len = rocke_b_global_load_i32(b, seq_lens, seq_idx, -1);
        context_len = rocke_b_sub(b, seq_len, cur_batch_q_len);
        qb_start_pos = rocke_b_mul(b, q_block_local_idx, rocke_b_const_i32(b, BLOCK_Q));

        /* with b.scf_if(cmp_ge(qb_start_pos, cur_batch_q_len)): b.ret() */
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_ge(b, qb_start_pos, cur_batch_q_len));
            rocke_b_region_enter(b, iff.then_region);
            rocke_b_ret(b);
            rocke_b_region_leave(b);
        }

        lane_row = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));
        {
            rocke_value_t* c0 = NULL;
            rocke_value_t* c1 = NULL;
            rocke_layout_map_coord(a_map, b, lane, 0, &c0, &c1);
            half_k = c1;
        }
        col = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));

        bm1_div_nqk = (ROCKE_G1250_BLOCK_M - 1) / NQK;
        /* msp_raw = add(add(context_len, qb_start_pos), const(bm1_div_nqk+1)) */
        {
            rocke_value_t* cq = rocke_b_add(b, context_len, qb_start_pos);
            msp_raw = rocke_b_add(b, cq, rocke_b_const_i32(b, bm1_div_nqk + 1));
        }
        max_seq_prefix_len
            = rocke_b_select(b, rocke_b_cmp_lt(b, msp_raw, seq_len), msp_raw, seq_len);
        /* num_tiles = div(add(max_seq_prefix_len, const(T-1)), const(T)) */
        {
            rocke_value_t* msp1 = rocke_b_add(b, max_seq_prefix_len, rocke_b_const_i32(b, T3D - 1));
            num_tiles = rocke_b_div(b, msp1, rocke_b_const_i32(b, T3D));
        }
        /* tps = div(add(seq_len, const(NUM_SEG*T-1)), const(NUM_SEG*T)) */
        {
            rocke_value_t* sl1 = rocke_b_add(b, seq_len, rocke_b_const_i32(b, NUM_SEG * T3D - 1));
            tps = rocke_b_div(b, sl1, rocke_b_const_i32(b, NUM_SEG * T3D));
        }
        tile_start = rocke_b_mul(b, seg_idx, tps);
        {
            rocke_value_t* te_raw
                = rocke_b_mul(b, rocke_b_add(b, seg_idx, rocke_b_const_i32(b, 1)), tps);
            tile_end = rocke_b_select(b, rocke_b_cmp_lt(b, te_raw, num_tiles), te_raw, num_tiles);
        }
        if(SLIDING_WINDOW > 0)
        {
            /* first_allowed = add(sub(add(context_len, qb_start_pos), const(SW)), const(1)) */
            rocke_value_t* fa_cq = rocke_b_add(b, context_len, qb_start_pos);
            rocke_value_t* fa_sub = rocke_b_sub(b, fa_cq, rocke_b_const_i32(b, SLIDING_WINDOW));
            rocke_value_t* first_allowed = rocke_b_add(b, fa_sub, rocke_b_const_i32(b, 1));
            rocke_value_t* sw_tile_start = rocke_b_div(b, first_allowed, rocke_b_const_i32(b, T3D));
            rocke_value_t* ts2;
            {
                rocke_value_t* lt0 = rocke_b_cmp_lt(b, sw_tile_start, rocke_b_const_i32(b, 0));
                ts2 = rocke_b_select(b, lt0, rocke_b_const_i32(b, 0), sw_tile_start);
            }
            tile_start = rocke_b_select(b, rocke_b_cmp_lt(b, tile_start, ts2), ts2, tile_start);
        }

        /* _write_partials lambda - implemented inline below via goto-like structure.
         * For now, define the function pointer for phys_block closure. */
        phys_ctx.block_tables = block_tables;
        phys_ctx.seq_idx = seq_idx;
        phys_ctx.block_table_stride = block_table_stride;
        phys_ctx.block_size = BS;

        /* ---- Empty segment check: write neutral partials and return ---- */
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_ge(b, tile_start, tile_end));
            rocke_b_region_enter(b, iff.then_region);
            {
                /* neutral_acc = [zero_vec_f32(c_frag) for _ in range(HD // WMMA_N)] */
                rocke_value_t** neutral_acc
                    = (rocke_value_t**)calloc((size_t)n_acc, sizeof(rocke_value_t*));
                for(d = 0; d < n_acc; ++d)
                {
                    neutral_acc[d] = rocke_b_zero_vec_f32(b, c_frag);
                }
                /* _write_partials(neg_inf * c_frag, zero_f * c_frag, neutral_acc) */
                for(r = 0; r < c_frag; ++r)
                {
                    rocke_value_t* row_rel = NULL;
                    rocke_value_t* col_n = NULL;
                    rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_n);
                    rocke_value_t* q_pos = rocke_b_add(
                        b, qb_start_pos, rocke_b_div(b, row_rel, rocke_b_const_i32(b, NQK)));
                    rocke_value_t* qh = u3d_qh(b, kv_head_idx, row_rel, NQK);
                    rocke_value_t* row_valid;
                    {
                        rocke_value_t* a = rocke_b_cmp_lt(b, q_pos, cur_batch_q_len);
                        rocke_value_t* c = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, NUM_QH));
                        row_valid = rocke_b_land(b, a, c);
                    }
                    rocke_value_t* out_token = rocke_b_add(b, cu_q_start, q_pos);
                    /* ml_idx = add(mul(add(mul(out_token, const(NUM_QH)), qh), const(NUM_SEG)), seg_idx) */
                    rocke_value_t* ml_idx;
                    {
                        rocke_value_t* inner = rocke_b_add(
                            b, rocke_b_mul(b, out_token, rocke_b_const_i32(b, NUM_QH)), qh);
                        rocke_value_t* scaled
                            = rocke_b_mul(b, inner, rocke_b_const_i32(b, NUM_SEG));
                        ml_idx = rocke_b_add(b, scaled, seg_idx);
                    }
                    rocke_value_t* so_base = rocke_b_mul(b, ml_idx, rocke_b_const_i32(b, HD));
                    /* m/l written once per row by col-0 lane */
                    {
                        rocke_value_t* ml_cond = rocke_b_land(
                            b, row_valid, rocke_b_cmp_eq(b, col_n, rocke_b_const_i32(b, 0)));
                        rocke_if_t mif = rocke_b_scf_if(b, ml_cond);
                        rocke_b_region_enter(b, mif.then_region);
                        rocke_b_global_store(b, segm_max, ml_idx, neg_inf, 4);
                        rocke_b_global_store(b, segm_expsum, ml_idx, zero_f, 4);
                        rocke_b_region_leave(b);
                    }
                    for(d = 0; d < n_acc; ++d)
                    {
                        rocke_value_t* o_col
                            = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_G1250_WMMA_N), col_n);
                        rocke_if_t oif = rocke_b_scf_if(b, row_valid);
                        rocke_value_t* sidx;
                        rocke_value_t* sval;
                        rocke_b_region_enter(b, oif.then_region);
                        /* store(segm_output, add(so_base, o_col), vec_extract(acc, r)):
                         * Python evals the index (add) before the value (extract). */
                        sidx = rocke_b_add(b, so_base, o_col);
                        sval = rocke_b_vec_extract(b, neutral_acc[d], r);
                        rocke_b_global_store(b, segm_output, sidx, sval, 4);
                        rocke_b_region_leave(b);
                    }
                }
                free(neutral_acc);
            }
            rocke_b_ret(b);
            rocke_b_region_leave(b);
        }

        /* Q fragments */
        q_pos_for_a
            = rocke_b_add(b, qb_start_pos, rocke_b_div(b, lane_row, rocke_b_const_i32(b, NQK)));
        qh_for_a = u3d_qh(b, kv_head_idx, lane_row, NQK);
        q_valid_for_a = u3d_row_valid(b, q_pos_for_a, cur_batch_q_len, qh_for_a, NUM_QH);
        q_pos_safe = rocke_b_select(b, q_valid_for_a, q_pos_for_a, rocke_b_const_i32(b, 0));
        qh_safe = rocke_b_select(b, q_valid_for_a, qh_for_a, rocke_b_const_i32(b, 0));
        q_token = rocke_b_add(b, cu_q_start, q_pos_safe);
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

        /* --- Single-wave path (NUM_WAVES==1, the validated default) --- */

        {
            int pshape[2] = {ROCKE_G1250_BLOCK_M, T3D};
            P_lds = spec->use_register_p ? NULL
                                         : rocke_b_smem_alloc(b, dtype, pshape, 2, "P3d_gfx1250");
        }
        if(spec->use_wide_lds_reads)
        {
            int vshape[2] = {HD, T3D};
            V_lds = rocke_b_smem_alloc(b, dtype, vshape, 2, "V3dT_gfx1250");
        }
        else if(spec->use_wide_kv_load)
        {
            int vshape[3] = {2, T3D, HD};
            V_lds = rocke_b_smem_alloc(b, dtype, vshape, 3, "V3d_gfx1250_dbl");
        }
        else
        {
            int vshape[2] = {T3D, HD};
            V_lds = rocke_b_smem_alloc(b, dtype, vshape, 2, "V3d_gfx1250");
        }

        /* m_inits / l_inits */
        m_inits = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
        l_inits = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
        if(!m_inits || !l_inits)
        {
            free(q_frags);
            free(m_inits);
            free(l_inits);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }
        for(r = 0; r < c_frag; ++r)
        {
            rocke_value_t* row_rel = NULL;
            rocke_layout_map_coord(c_map, b, lane, r, &row_rel, NULL);
            rocke_value_t* qh = u3d_qh(b, kv_head_idx, row_rel, NQK);
            rocke_value_t* qh_in = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, NUM_QH));
            if(spec->use_sinks)
            {
                rocke_value_t* sink_h = rocke_b_global_load(b, sinks, qh, dtype, 2);
                rocke_value_t* sink_f = rocke_b_fmul(b, rocke_b_cast_to_f32(b, sink_h), rcp_ln2);
                rocke_value_t* use_sink
                    = rocke_b_land(b, qh_in, rocke_b_cmp_eq(b, seg_idx, rocke_b_const_i32(b, 0)));
                m_inits[r] = rocke_b_select(b, use_sink, sink_f, neg_inf);
            }
            else
            {
                m_inits[r] = neg_inf;
            }
        }
        /* l_inits */
        for(r = 0; r < c_frag; ++r)
        {
            if(spec->use_sinks)
            {
                l_inits[r] = rocke_b_select(
                    b, rocke_b_cmp_eq(b, seg_idx, rocke_b_const_i32(b, 0)), one_f, zero_f);
            }
            else
            {
                l_inits[r] = one_f;
            }
        }

        /* iter_args */
        n_iter = 2 * c_frag + n_acc;
        iter_args = (rocke_iter_arg_t*)calloc((size_t)n_iter, sizeof(rocke_iter_arg_t));
        if(iter_args == NULL)
        {
            free(q_frags);
            free(m_inits);
            free(l_inits);
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
                iter_args[2 * r + 1].init = l_inits[r];
            }
            for(d = 0; d < n_acc; ++d)
            {
                iter_args[2 * c_frag + d].name = k_anames[d];
                iter_args[2 * c_frag + d].init = rocke_b_zero_vec_f32(b, c_frag);
            }
        }

        /* kloop */
        kloop = rocke_b_scf_for_iter(
            b, tile_start, tile_end, rocke_b_const_i32(b, 1), iter_args, n_iter, "kt", false, true);

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
                free(l_inits);
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

            tile_base = rocke_b_mul(b, kt, rocke_b_const_i32(b, T3D));

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
                                          kv_dtype,
                                          k_scale,
                                          dtype,
                                          c_frag,
                                          u3d_phys_block,
                                          (void*)&phys_ctx,
                                          wmma_spacing,
                                          scores);

            for(r = 0; r < c_frag; ++r)
            {
                rocke_value_t* row_rel = NULL;
                rocke_value_t* col_k = NULL;
                rocke_value_t* q_pos;
                rocke_value_t* qh;
                rocke_value_t* row_valid;
                rocke_value_t* causal_lim;
                rocke_value_t* srs[2];
                rocke_value_t* m_new = NULL;
                rocke_value_t* l_new = NULL;
                rocke_value_t* alpha = NULL;
                rocke_value_t* p[2];
                int nsub;

                rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_k);
                q_pos = rocke_b_add(
                    b, qb_start_pos, rocke_b_div(b, row_rel, rocke_b_const_i32(b, NQK)));
                qh = u3d_qh(b, kv_head_idx, row_rel, NQK);
                row_valid = u3d_row_valid(b, q_pos, cur_batch_q_len, qh, NUM_QH);
                causal_lim = rocke_b_add(b, context_len, q_pos);

                for(nsub = 0; nsub < 2; ++nsub)
                {
                    rocke_value_t* key_pos = rocke_b_add(
                        b,
                        rocke_b_add(b, tile_base, rocke_b_const_i32(b, nsub * ROCKE_G1250_WMMA_N)),
                        col_k);
                    rocke_value_t* score_log2
                        = rocke_b_fmul(b, rocke_b_vec_extract(b, scores[nsub], r), qk_scale);
                    rocke_value_t* causal_keep = rocke_b_cmp_le(b, key_pos, causal_lim);
                    rocke_value_t* in_seq = rocke_b_cmp_lt(b, key_pos, seq_len);
                    rocke_value_t* keep
                        = rocke_b_land(b, row_valid, rocke_b_land(b, in_seq, causal_keep));
                    if(SLIDING_WINDOW > 0)
                    {
                        rocke_value_t* dist = rocke_b_sub(b, causal_lim, key_pos);
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
                                               spec->use_dpp_softmax,
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

            /* V staging + PV GEMM */
            {
                /* v_read = mod(sub(kt, tile_start), const(2)); sequence sub before const. */
                rocke_value_t* v_read_sub = rocke_b_sub(b, kt, tile_start);
                rocke_value_t* v_read = rocke_b_mod(b, v_read_sub, rocke_b_const_i32(b, 2));
                (void)v_read;

                if(spec->use_wide_lds_reads)
                {
                    rocke_g1250_stage_v_tile_transposed(b,
                                                        V_lds,
                                                        value,
                                                        &kv_desc,
                                                        kv_head_idx,
                                                        tile_base,
                                                        lane,
                                                        BS,
                                                        HD,
                                                        kv_dtype,
                                                        v_scale,
                                                        dtype,
                                                        u3d_phys_block,
                                                        (void*)&phys_ctx,
                                                        NULL);
                }
                else if(spec->use_wide_kv_load)
                {
                    rocke_value_t* next_kt = rocke_b_add(b, kt, rocke_b_const_i32(b, 1));
                    rocke_value_t* has_next = rocke_b_cmp_lt(b, next_kt, tile_end);
                    rocke_value_t* v_write
                        = rocke_b_mod(b,
                                      rocke_b_add(b, v_read, rocke_b_const_i32(b, 1)),
                                      rocke_b_const_i32(b, 2));
                    rocke_value_t* tile_base_n = rocke_b_mul(b, next_kt, rocke_b_const_i32(b, T3D));
                    rocke_if_t nif = rocke_b_scf_if(b, has_next);
                    rocke_b_region_enter(b, nif.then_region);
                    rocke_g1250_stage_v_tile_buf(b,
                                                 V_lds,
                                                 v_write,
                                                 value,
                                                 &kv_desc,
                                                 kv_head_idx,
                                                 tile_base_n,
                                                 lane,
                                                 BS,
                                                 HD,
                                                 kv_dtype,
                                                 v_scale,
                                                 dtype,
                                                 u3d_phys_block,
                                                 (void*)&phys_ctx);
                    rocke_b_region_leave(b);
                }
                else
                {
                    rocke_g1250_stage_v_tile(b,
                                             V_lds,
                                             value,
                                             &kv_desc,
                                             kv_head_idx,
                                             tile_base,
                                             lane,
                                             BS,
                                             HD,
                                             kv_dtype,
                                             v_scale,
                                             dtype,
                                             u3d_phys_block,
                                             (void*)&phys_ctx);
                }
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
                                                      spec->use_wide_kv_load ? v_read : NULL,
                                                      wmma_spacing);
                }
                else if(spec->use_wide_lds_reads)
                {
                    rocke_g1250_compute_pv_wide(b,
                                                P_lds,
                                                V_lds,
                                                new_accs,
                                                a_map,
                                                lane,
                                                lane_row,
                                                a_frag,
                                                HD,
                                                dtype,
                                                NULL,
                                                NULL,
                                                wmma_spacing);
                }
                else if(spec->use_ds_tr_reads)
                {
                    rocke_g1250_compute_pv_dstr(b,
                                                P_lds,
                                                V_lds,
                                                new_accs,
                                                a_map,
                                                lane,
                                                lane_row,
                                                a_frag,
                                                HD,
                                                dtype,
                                                NULL,
                                                NULL,
                                                wmma_spacing);
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
                                           spec->use_wide_kv_load ? v_read : NULL,
                                           NULL,
                                           wmma_spacing);
                }
                rocke_b_sync(b);
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
                free(l_inits);
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
            free(l_inits);
            free(iter_args);
            return NULL;
        }

        /* Epilogue: _write_partials(ms_final, ls_final, accs_final) */
        {
            rocke_value_t** final = kloop.op->results;
            for(r = 0; r < c_frag; ++r)
            {
                rocke_value_t* row_rel = NULL;
                rocke_value_t* col_n = NULL;
                rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_n);
                rocke_value_t* q_pos = rocke_b_add(
                    b, qb_start_pos, rocke_b_div(b, row_rel, rocke_b_const_i32(b, NQK)));
                rocke_value_t* qh = u3d_qh(b, kv_head_idx, row_rel, NQK);
                rocke_value_t* row_valid;
                {
                    rocke_value_t* a = rocke_b_cmp_lt(b, q_pos, cur_batch_q_len);
                    rocke_value_t* c = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, NUM_QH));
                    row_valid = rocke_b_land(b, a, c);
                }
                rocke_value_t* out_token = rocke_b_add(b, cu_q_start, q_pos);
                rocke_value_t* ml_idx;
                {
                    rocke_value_t* inner = rocke_b_add(
                        b, rocke_b_mul(b, out_token, rocke_b_const_i32(b, NUM_QH)), qh);
                    rocke_value_t* scaled = rocke_b_mul(b, inner, rocke_b_const_i32(b, NUM_SEG));
                    ml_idx = rocke_b_add(b, scaled, seg_idx);
                }
                rocke_value_t* so_base = rocke_b_mul(b, ml_idx, rocke_b_const_i32(b, HD));
                /* m/l written once per row by col-0 lane */
                {
                    rocke_value_t* ml_cond = rocke_b_land(
                        b, row_valid, rocke_b_cmp_eq(b, col_n, rocke_b_const_i32(b, 0)));
                    rocke_if_t mif = rocke_b_scf_if(b, ml_cond);
                    rocke_b_region_enter(b, mif.then_region);
                    rocke_b_global_store(b, segm_max, ml_idx, final[2 * r], 4);
                    rocke_b_global_store(b, segm_expsum, ml_idx, final[2 * r + 1], 4);
                    rocke_b_region_leave(b);
                }
                for(d = 0; d < n_acc; ++d)
                {
                    rocke_value_t* o_col
                        = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_G1250_WMMA_N), col_n);
                    rocke_if_t oif = rocke_b_scf_if(b, row_valid);
                    rocke_value_t* sidx;
                    rocke_value_t* sval;
                    rocke_b_region_enter(b, oif.then_region);
                    sidx = rocke_b_add(b, so_base, o_col);
                    sval = rocke_b_vec_extract(b, final[2 * c_frag + d], r);
                    rocke_b_global_store(b, segm_output, sidx, sval, 4);
                    rocke_b_region_leave(b);
                }
            }
        }

        rocke_b_ret(b);
        free(q_frags);
        free(m_inits);
        free(l_inits);
        free(iter_args);

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_uattn3d_seg_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_uattn3d_seg_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_uattn3d_seg_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_uattn3d_seg_gfx1250(b, spec, arch);
    });
}

rocke_status_t rocke_uattn3d_seg_gfx1250_lower_to_llvm(const rocke_uattn3d_seg_gfx1250_spec_t* spec,
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

    kernel = rocke_build_uattn3d_seg_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_uattn3d_seg_gfx1250 failed";
            }
            size_t n = strlen(m);
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

/* ============================================================
 * Reduce spec default / helpers
 * ============================================================ */

rocke_uattn3d_reduce_gfx1250_spec_t rocke_uattn3d_reduce_gfx1250_spec_default(void)
{
    rocke_uattn3d_reduce_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.waves_per_eu_set = false;
    return s;
}

rocke_status_t rocke_uattn3d_reduce_gfx1250_kernel_name(
    const rocke_uattn3d_reduce_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char d_part[32];
    char h_part[32];
    char seg_part[32];
    const char* parts[4];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(d_part, sizeof(d_part), "d%d", spec->head_size);
    snprintf(h_part, sizeof(h_part), "h%d", spec->num_query_heads);
    snprintf(seg_part, sizeof(seg_part), "seg%d", spec->num_segments);
    parts[0] = d_part;
    parts[1] = h_part;
    parts[2] = spec->dtype;
    parts[3] = seg_part;
    return rocke_kernel_name_join(
        "rocke_uattn3d_reduce_gfx1250", parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* ============================================================
 * build_unified_attention_reduce_tiled (reduce kernel)
 * ============================================================ */

rocke_kernel_def_t* rocke_build_uattn3d_reduce_gfx1250(
    rocke_ir_builder_t* b, const rocke_uattn3d_reduce_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* dtype;
        int HD, NUM_QH, NUM_SEG, WAVE;
        int n_iter_seg, d_iter;
        int i, s;

        rocke_value_t* output;
        rocke_value_t* segm_output;
        rocke_value_t* segm_max;
        rocke_value_t* segm_expsum;
        rocke_value_t* q_token;
        rocke_value_t* q_head;
        rocke_value_t* lane;
        rocke_value_t* neg_inf;
        rocke_value_t* zero_f;
        rocke_value_t* ml_base;
        rocke_value_t* so_base;
        rocke_value_t* factor_lds;
        rocke_value_t* local_max;
        rocke_value_t* overall_max;
        rocke_value_t* local_sum;
        rocke_value_t* overall;
        rocke_value_t* inv_l;

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }

        dtype = rocke_bf16();
        HD = spec->head_size;
        NUM_QH = spec->num_query_heads;
        NUM_SEG = spec->num_segments;
        WAVE = ROCKE_G1250_WAVE;

        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", WAVE);
        if(spec->waves_per_eu_set)
        {
            rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
        }

        /* params */
        {
            rocke_param_opts_t nawo16;
            rocke_param_opts_t ro16;
            rocke_param_opts_t ro4;
            const rocke_type_t* p_bf16 = rocke_ptr_type(b, rocke_bf16(), "global");
            const rocke_type_t* p_f32 = rocke_ptr_type(b, rocke_f32(), "global");
            const rocke_type_t* p_i32 = rocke_ptr_type(b, rocke_i32(), "global");

            memset(&nawo16, 0, sizeof(nawo16));
            nawo16.noalias = true;
            nawo16.noalias_set = true;
            nawo16.writeonly = true;
            nawo16.writeonly_set = true;
            nawo16.align = 16;
            nawo16.align_set = true;

            memset(&ro16, 0, sizeof(ro16));
            ro16.readonly = true;
            ro16.readonly_set = true;
            ro16.align = 16;
            ro16.align_set = true;

            memset(&ro4, 0, sizeof(ro4));
            ro4.readonly = true;
            ro4.readonly_set = true;
            ro4.align = 4;
            ro4.align_set = true;

            output = rocke_b_param(b, "output_ptr", p_bf16, &nawo16);
            segm_output = rocke_b_param(b, "segm_output_ptr", p_f32, &ro16);
            segm_max = rocke_b_param(b, "segm_max_ptr", p_f32, &ro4);
            segm_expsum = rocke_b_param(b, "segm_expsum_ptr", p_f32, &ro4);
            (void)rocke_b_param(b, "seq_lens_ptr", p_i32, &ro4);
        }

        q_token = rocke_b_block_id_x(b);
        q_head = rocke_b_block_id_y(b);
        {
            rocke_value_t* t = rocke_b_thread_id_x(b);
            lane = rocke_b_mod(b, t, rocke_b_const_i32(b, WAVE));
        }
        neg_inf = rocke_b_const_f32(b, -1e30);
        zero_f = rocke_b_const_f32(b, 0.0);

        /* ml_base = mul(add(mul(q_token, const(NUM_QH)), q_head), const(NUM_SEG)) */
        {
            rocke_value_t* inner
                = rocke_b_add(b, rocke_b_mul(b, q_token, rocke_b_const_i32(b, NUM_QH)), q_head);
            ml_base = rocke_b_mul(b, inner, rocke_b_const_i32(b, NUM_SEG));
        }
        /* so_base = mul(ml_base, const(HD)) -- side-effecting emit for byte-identity */
        so_base = rocke_b_mul(b, ml_base, rocke_b_const_i32(b, HD));
        (void)so_base;

        {
            int fshape[1] = {NUM_SEG};
            factor_lds = rocke_b_smem_alloc(b, rocke_f32(), fshape, 1, "factor3d_gfx1250");
        }

        n_iter_seg = (NUM_SEG + WAVE - 1) / WAVE;

        /* pass 1: overall max */
        local_max = neg_inf;
        for(i = 0; i < n_iter_seg; ++i)
        {
            rocke_value_t* sv = rocke_b_add(b, lane, rocke_b_const_i32(b, i * WAVE));
            rocke_value_t* valid = rocke_b_cmp_lt(b, sv, rocke_b_const_i32(b, NUM_SEG));
            rocke_value_t* s_safe = rocke_b_select(b, valid, sv, rocke_b_const_i32(b, 0));
            rocke_value_t* m
                = rocke_b_global_load(b, segm_max, rocke_b_add(b, ml_base, s_safe), rocke_f32(), 4);
            local_max = rocke_b_fmax(b, local_max, rocke_b_select(b, valid, m, neg_inf));
        }
        overall_max = rocke_wave_reduce_max(b, local_max, WAVE, WAVE);

        /* pass 2: factor[s] + overall denom */
        local_sum = zero_f;
        for(i = 0; i < n_iter_seg; ++i)
        {
            rocke_value_t* sv = rocke_b_add(b, lane, rocke_b_const_i32(b, i * WAVE));
            rocke_value_t* valid = rocke_b_cmp_lt(b, sv, rocke_b_const_i32(b, NUM_SEG));
            rocke_value_t* s_safe = rocke_b_select(b, valid, sv, rocke_b_const_i32(b, 0));
            rocke_value_t* m
                = rocke_b_global_load(b, segm_max, rocke_b_add(b, ml_base, s_safe), rocke_f32(), 4);
            rocke_value_t* l = rocke_b_global_load(
                b, segm_expsum, rocke_b_add(b, ml_base, s_safe), rocke_f32(), 4);
            /* m_finite = land(fcmp oeq(m, m), fcmp ogt(m, neg_inf)) */
            rocke_value_t* m_finite;
            {
                rocke_value_t* oeq = rocke_b_fcmp(b, "oeq", m, m);
                rocke_value_t* ogt = rocke_b_fcmp(b, "ogt", m, neg_inf);
                m_finite = rocke_b_land(b, oeq, ogt);
            }
            /* f = select(m_finite, exp2(fsub(m, overall_max)), zero_f) */
            rocke_value_t* f;
            {
                rocke_value_t* cond = m_finite;
                rocke_value_t* e = rocke_b_exp2(b, rocke_b_fsub(b, m, overall_max));
                f = rocke_b_select(b, cond, e, zero_f);
            }
            f = rocke_b_select(b, valid, f, zero_f);
            {
                rocke_if_t vif = rocke_b_scf_if(b, valid);
                rocke_b_region_enter(b, vif.then_region);
                rocke_value_t* idx[1] = {s_safe};
                rocke_b_smem_store_vN(b, factor_lds, idx, 1, f, 1);
                rocke_b_region_leave(b);
            }
            local_sum = rocke_b_fadd(b, local_sum, rocke_b_fmul(b, l, f));
        }
        overall = rocke_wave_reduce_sum(b, local_sum, WAVE, WAVE);
        /* inv_l = select(fcmp oeq(overall, zero_f), zero_f, rcp(overall)) */
        {
            rocke_value_t* zmask = rocke_b_fcmp(b, "oeq", overall, zero_f);
            rocke_value_t* rl = rocke_b_rcp(b, overall);
            inv_l = rocke_b_select(b, zmask, zero_f, rl);
        }
        rocke_b_sync(b);

        /* pass 3: per-dim acc reduce + normalize + cast */
        d_iter = (HD + WAVE - 1) / WAVE;
        for(i = 0; i < d_iter; ++i)
        {
            rocke_value_t* dv = rocke_b_add(b, lane, rocke_b_const_i32(b, i * WAVE));
            rocke_value_t* d_valid = rocke_b_cmp_lt(b, dv, rocke_b_const_i32(b, HD));
            rocke_value_t* d_safe = rocke_b_select(b, d_valid, dv, rocke_b_const_i32(b, 0));
            rocke_value_t* acc = zero_f;
            for(s = 0; s < NUM_SEG; ++s)
            {
                rocke_value_t* fidx[1] = {rocke_b_const_i32(b, s)};
                rocke_value_t* f = rocke_b_vec_extract(
                    b, rocke_b_smem_load_vN(b, factor_lds, fidx, 1, rocke_f32(), 1), 0);
                /* ov = global_load(segm_output, add(mul(add(ml_base, const(s)), const(HD)), d_safe), F32, 4) */
                rocke_value_t* ov;
                {
                    rocke_value_t* sidx = rocke_b_add(b, ml_base, rocke_b_const_i32(b, s));
                    rocke_value_t* base = rocke_b_mul(b, sidx, rocke_b_const_i32(b, HD));
                    ov = rocke_b_global_load(
                        b, segm_output, rocke_b_add(b, base, d_safe), rocke_f32(), 4);
                }
                acc = rocke_b_fadd(b, acc, rocke_b_fmul(b, ov, f));
            }
            /* out_addr = add(mul(add(mul(q_token, const(NUM_QH)), q_head), const(HD)), d_safe) */
            rocke_value_t* out_addr;
            {
                rocke_value_t* inner
                    = rocke_b_add(b, rocke_b_mul(b, q_token, rocke_b_const_i32(b, NUM_QH)), q_head);
                rocke_value_t* scaled = rocke_b_mul(b, inner, rocke_b_const_i32(b, HD));
                out_addr = rocke_b_add(b, scaled, d_safe);
            }
            {
                rocke_if_t dif = rocke_b_scf_if(b, d_valid);
                rocke_b_region_enter(b, dif.then_region);
                rocke_b_global_store(b,
                                     output,
                                     out_addr,
                                     rocke_b_cast_f32_to(b, rocke_b_fmul(b, acc, inv_l), dtype),
                                     2);
                rocke_b_region_leave(b);
            }
        }
        rocke_b_ret(b);
        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_uattn3d_reduce_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_uattn3d_reduce_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_uattn3d_reduce_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_uattn3d_reduce_gfx1250(b, spec, arch);
    });
}

rocke_status_t
    rocke_uattn3d_reduce_gfx1250_lower_to_llvm(const rocke_uattn3d_reduce_gfx1250_spec_t* spec,
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

    kernel = rocke_build_uattn3d_reduce_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_uattn3d_reduce_gfx1250 failed";
            }
            size_t n = strlen(m);
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
