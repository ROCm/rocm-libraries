// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_wmma_attention_fwd.c -- C99 port of
 * rocke/instances/gfx1250/wmma_attention_fwd.py.
 *
 * Standalone gfx1250 K=32 WMMA FMHA forward: one wave32 per (q_tile, head,
 * batch), BLOCK_M=16 Q rows, BLOCK_K=32 K positions per K-loop iteration, online
 * softmax over 32 k columns, PV via one K=32 WMMA per head_size/16 d-tile. The
 * build op order tracks build_wmma_attention_fwd() top-to-bottom; emitted IR is
 * byte-identical to the Python lowerer (args sequenced left-to-right, const_i32
 * never deduped, list comprehensions unrolled in source order).
 *
 * NOTE: the _WMMA_SPACING experimental knob (ROCKE_GFX1250_WMMA_SPACING) is
 * faithfully mirrored: it defaults to 0, so causal_wmma_spacing() emits nothing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_wmma_attention_fwd.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_WAF_OP_ID "wmma_gfx1250_f32_16x16x32_f16"
#define ROCKE_WAF_BLOCK_M 16
#define ROCKE_WAF_WMMA_N 16
#define ROCKE_WAF_WMMA_K 32
#define ROCKE_WAF_BLOCK_K 32

/* ROCKE_GFX1250_WMMA_SPACING env knob (default 0 => no spacing emitted). */
static int rocke_waf_wmma_spacing(void)
{
    const char* s = getenv("ROCKE_GFX1250_WMMA_SPACING");
    if(s == NULL || s[0] == '\0')
    {
        return 0;
    }
    return atoi(s);
}

rocke_wmma_attention_fwd_gfx1250_spec_t rocke_wmma_attention_fwd_gfx1250_spec_default(void)
{
    rocke_wmma_attention_fwd_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.head_size = 0;
    s.num_query_heads = 0;
    s.num_kv_heads = 0;
    s.dtype = "fp16";
    s.mask_mode = "none";
    s.sliding_window = 0;
    s.name = "rocke_wmma_attention_fwd_gfx1250";
    return s;
}

int rocke_wmma_attention_fwd_gfx1250_kv_heads(const rocke_wmma_attention_fwd_gfx1250_spec_t* spec)
{
    return spec->num_kv_heads != 0 ? spec->num_kv_heads : spec->num_query_heads;
}

int rocke_wmma_attention_fwd_gfx1250_block_size(const rocke_wmma_attention_fwd_gfx1250_spec_t* spec)
{
    (void)spec;
    return 32;
}

rocke_status_t rocke_wmma_attention_fwd_gfx1250_kernel_name(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char h[32];
    char hq[32];
    char hk[32];
    const char* parts[6];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(h, sizeof(h), "H%d", spec->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", spec->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", rocke_wmma_attention_fwd_gfx1250_kv_heads(spec));
    parts[0] = "wmma16x16x32";
    parts[1] = h;
    parts[2] = hq;
    parts[3] = hk;
    parts[4] = "fp16";
    parts[5] = spec->mask_mode;
    return rocke_kernel_name_join(spec->name, parts, 6, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_wmma_attention_fwd_gfx1250_is_valid_spec(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    const rocke_arch_target_t* target;
    const rocke_mma_op_t* op;
    long bytes_lds;
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

    /* __post_init__: dtype fp16 only. */
    if(spec->dtype == NULL || (strcmp(spec->dtype, "fp16") != 0 && strcmp(spec->dtype, "f16") != 0))
    {
        snprintf(buf,
                 sizeof(buf),
                 "WmmaAttentionFwdSpec currently supports fp16 only, got '%s'",
                 spec->dtype ? spec->dtype : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if((spec->head_size % ROCKE_WAF_WMMA_K) != 0)
    {
        snprintf(buf, sizeof(buf), "head_size must be a multiple of %d", ROCKE_WAF_WMMA_K);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->mask_mode == NULL
       || (strcmp(spec->mask_mode, "none") != 0 && strcmp(spec->mask_mode, "causal") != 0))
    {
        snprintf(buf,
                 sizeof(buf),
                 "WMMA FMHA supports mask_mode 'none'/'causal', got '%s'",
                 spec->mask_mode ? spec->mask_mode : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* is_valid_spec body. */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target '%s'", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    op = rocke_archtarget_by_op_id(target, ROCKE_WAF_OP_ID);
    if(op == NULL || op->family == NULL || strcmp(op->family, "wmma") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "WMMA %s atom absent on %s (this kernel targets the gfx1250 16x16x32 WMMA)",
                 ROCKE_WAF_OP_ID,
                 arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(target->wave_size != op->wave_size)
    {
        snprintf(buf,
                 sizeof(buf),
                 "arch wave size %d != WMMA atom wave size %d on %s",
                 target->wave_size,
                 op->wave_size,
                 arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if((spec->head_size % ROCKE_WAF_WMMA_K) != 0)
    {
        snprintf(buf, sizeof(buf), "head_size must be a multiple of %d", ROCKE_WAF_WMMA_K);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* LDS: one 16x32 f16 P-staging tile. */
    bytes_lds = (long)ROCKE_WAF_BLOCK_M * ROCKE_WAF_BLOCK_K * 2;
    if(!rocke_arch_fits_lds(target, bytes_lds))
    {
        snprintf(buf,
                 sizeof(buf),
                 "LDS budget %ld > %d cap on %s",
                 bytes_lds,
                 target->lds_capacity_bytes,
                 arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* The single coord lane->(out0,out1) helper (out1 may be NULL for "[0]"). */
static rocke_value_t* rocke_waf_coord0(rocke_ir_builder_t* b,
                                       const rocke_layout_map_t* m,
                                       rocke_value_t* lane,
                                       int slot)
{
    rocke_value_t* c0 = NULL;
    rocke_layout_map_coord(m, b, lane, slot, &c0, NULL);
    return c0;
}

static void rocke_waf_coord01(rocke_ir_builder_t* b,
                              const rocke_layout_map_t* m,
                              rocke_value_t* lane,
                              int slot,
                              rocke_value_t** out0,
                              rocke_value_t** out1)
{
    rocke_layout_map_coord(m, b, lane, slot, out0, out1);
}

rocke_kernel_def_t* rocke_build_wmma_attention_fwd_gfx1250(
    rocke_ir_builder_t* b, const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_arch_target_t* target;
        const rocke_mma_op_t* op;
        const rocke_layout_map_t* a_map;
        const rocke_layout_map_t* c_map;
        const rocke_type_t* dtype_ir;
        int wave;
        int a_frag;
        int c_frag;
        int H;
        int n_dk;
        int n_pv;
        int spacing;
        int d;
        int r;
        int nsub;
        int j;
        char reason[ROCKE_ERR_MSG_CAP];

        /* params */
        rocke_value_t* Q;
        rocke_value_t* K;
        rocke_value_t* V;
        rocke_value_t* O;
        rocke_value_t* scale_log2;
        rocke_value_t* seqlen_q;
        rocke_value_t* seqlen_k;
        rocke_value_t* stride_q_token;
        rocke_value_t* stride_q_head;
        rocke_value_t* stride_k_token;
        rocke_value_t* stride_k_head;
        rocke_value_t* stride_v_token;
        rocke_value_t* stride_v_head;
        rocke_value_t* stride_o_token;
        rocke_value_t* stride_o_head;

        rocke_value_t* c16;
        rocke_value_t* neg_inf;
        rocke_value_t* zero_f;
        rocke_value_t* q_tile;
        rocke_value_t* head;
        rocke_value_t* batch;
        rocke_value_t* kv_head;
        rocke_value_t* lane;
        rocke_value_t* a_row;
        rocke_value_t* col;
        rocke_value_t* half_k;
        rocke_value_t* q_row0;
        rocke_value_t* batch_row_q;
        rocke_value_t* q_row;
        rocke_value_t* q_addr_row_base;
        rocke_value_t** q_frags;
        rocke_value_t* P_lds;
        rocke_value_t* batch_off_k;
        rocke_value_t* batch_off_v;
        rocke_iter_arg_t* iter_args;
        int n_iter;
        rocke_value_t* c_block_k;
        rocke_value_t* loop_stop;
        rocke_for_t kloop;
        int shape[2];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }

        if(!rocke_wmma_attention_fwd_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            char msg[ROCKE_ERR_MSG_CAP];
            ROCKE_ERR_SNPRINTF(msg, sizeof(msg), "invalid wmma_attention_fwd spec: %s", reason);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
            return NULL;
        }

        target = rocke_archtarget_from_gfx(arch);
        op = rocke_archtarget_by_op_id(target, ROCKE_WAF_OP_ID);
        wave = op->wave_size; /* 32 */
        a_map = rocke_mma_op_a_layout(op, b);
        c_map = rocke_mma_op_c_layout(op, b);
        a_frag = op->a_frag_len; /* 16 */
        c_frag = op->c_frag_len; /* 8 */
        if(a_map == NULL || c_map == NULL)
        {
            return NULL;
        }

        H = spec->head_size;
        n_dk = H / ROCKE_WAF_WMMA_K; /* QK^T d-tiles (op.k=32) */
        n_pv = H / ROCKE_WAF_WMMA_N; /* PV d-tiles (op.n=16) */
        dtype_ir = rocke_f16();
        spacing = rocke_waf_wmma_spacing();

        /* b.kernel.attrs["max_workgroup_size"] = wave */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", wave);

        /* _declare_params(b) */
        {
            rocke_param_opts_t ro;
            rocke_param_opts_t wo;
            const rocke_type_t* ptr_f16 = rocke_ptr_type(b, rocke_f16(), "global");
            memset(&ro, 0, sizeof(ro));
            ro.noalias = true;
            ro.noalias_set = true;
            ro.readonly = true;
            ro.readonly_set = true;
            ro.align = 16;
            ro.align_set = true;
            memset(&wo, 0, sizeof(wo));
            wo.noalias = true;
            wo.noalias_set = true;
            wo.writeonly = true;
            wo.writeonly_set = true;
            wo.align = 16;
            wo.align_set = true;

            Q = rocke_b_param(b, "Q", ptr_f16, &ro);
            K = rocke_b_param(b, "K", ptr_f16, &ro);
            V = rocke_b_param(b, "V", ptr_f16, &ro);
            O = rocke_b_param(b, "O", ptr_f16, &wo);
            scale_log2 = rocke_b_param(b, "scale_log2", rocke_f32(), NULL);
            seqlen_q = rocke_b_param(b, "seqlen_q", rocke_i32(), NULL);
            seqlen_k = rocke_b_param(b, "seqlen_k", rocke_i32(), NULL);
            stride_q_token = rocke_b_param(b, "stride_q_token", rocke_i32(), NULL);
            stride_q_head = rocke_b_param(b, "stride_q_head", rocke_i32(), NULL);
            stride_k_token = rocke_b_param(b, "stride_k_token", rocke_i32(), NULL);
            stride_k_head = rocke_b_param(b, "stride_k_head", rocke_i32(), NULL);
            stride_v_token = rocke_b_param(b, "stride_v_token", rocke_i32(), NULL);
            stride_v_head = rocke_b_param(b, "stride_v_head", rocke_i32(), NULL);
            stride_o_token = rocke_b_param(b, "stride_o_token", rocke_i32(), NULL);
            stride_o_head = rocke_b_param(b, "stride_o_head", rocke_i32(), NULL);
        }

        /* c16 = const(16); neg_inf = const_f32(-1e30); zero_f = const_f32(0.0) */
        c16 = rocke_b_const_i32(b, 16);
        neg_inf = rocke_b_const_f32(b, -1e30);
        zero_f = rocke_b_const_f32(b, 0.0);

        /* q_tile = block_id_x(); head = block_id_y(); batch = block_id_z() */
        q_tile = rocke_b_block_id_x(b);
        head = rocke_b_block_id_y(b);
        batch = rocke_b_block_id_z(b);

        /* kv_head = head if kvh == qh else b.div(head, const(qh // kvh)) */
        {
            int qh = spec->num_query_heads;
            int kvh = rocke_wmma_attention_fwd_gfx1250_kv_heads(spec);
            kv_head = (kvh == qh) ? head : rocke_b_div(b, head, rocke_b_const_i32(b, qh / kvh));
        }

        /* lane = b.mod(b.thread_id_x(), const(wave))
         * Python evals thread_id_x first, then const(wave); sequence to match. */
        {
            rocke_value_t* t = rocke_b_thread_id_x(b);
            lane = rocke_b_mod(b, t, rocke_b_const_i32(b, wave));
        }
        /* a_row = a_map.coord(b, lane, 0)[0] */
        a_row = rocke_waf_coord0(b, a_map, lane, 0);
        /* col = b.mod(lane, c16) */
        col = rocke_b_mod(b, lane, c16);
        /* half_k = a_map.coord(b, lane, 0)[1] */
        {
            rocke_value_t* tmp0 = NULL;
            rocke_value_t* tmp1 = NULL;
            rocke_waf_coord01(b, a_map, lane, 0, &tmp0, &tmp1);
            half_k = tmp1;
        }

        /* q_row0 = mul(q_tile, c16); batch_row_q = mul(batch, seqlen_q) */
        q_row0 = rocke_b_mul(b, q_tile, c16);
        batch_row_q = rocke_b_mul(b, batch, seqlen_q);
        /* q_row = add(add(q_row0, batch_row_q), a_row) */
        q_row = rocke_b_add(b, rocke_b_add(b, q_row0, batch_row_q), a_row);
        /* q_addr_row_base = add(mul(q_row, stride_q_token), mul(head, stride_q_head))
         * Both args are side-effecting muls; sequence L-to-R to match Python. */
        {
            rocke_value_t* qm = rocke_b_mul(b, q_row, stride_q_token);
            rocke_value_t* hm = rocke_b_mul(b, head, stride_q_head);
            q_addr_row_base = rocke_b_add(b, qm, hm);
        }

        /* q_frags: for d in range(n_dk): load this lane's 16 d-elements. */
        q_frags = (rocke_value_t**)calloc((size_t)n_dk, sizeof(rocke_value_t*));
        if(q_frags == NULL)
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }
        for(d = 0; d < n_dk; ++d)
        {
            /* q_addr = add(add(q_addr_row_base, const(d*_WMMA_K)), half_k) */
            rocke_value_t* q_addr = rocke_b_add(
                b,
                rocke_b_add(b, q_addr_row_base, rocke_b_const_i32(b, d * ROCKE_WAF_WMMA_K)),
                half_k);
            q_frags[d] = rocke_b_global_load_vN(b, Q, q_addr, dtype_ir, a_frag, a_frag * 2);
        }

        /* P_lds = b.smem_alloc(dtype_ir, [_BLOCK_M, _BLOCK_K], name_hint="Pgfx1250") */
        shape[0] = ROCKE_WAF_BLOCK_M;
        shape[1] = ROCKE_WAF_BLOCK_K;
        P_lds = rocke_b_smem_alloc(b, dtype_ir, shape, 2, "Pgfx1250");

        /* batch_off_k = mul(mul(batch, seqlen_k), stride_k_token); batch_off_v sib. */
        batch_off_k = rocke_b_mul(b, rocke_b_mul(b, batch, seqlen_k), stride_k_token);
        batch_off_v = rocke_b_mul(b, rocke_b_mul(b, batch, seqlen_k), stride_v_token);

        /* iter_args: [m0,l0,m1,l1,...,m{c_frag-1},l{c_frag-1}, acc0..acc{n_pv-1}] */
        n_iter = 2 * c_frag + n_pv;
        iter_args = (rocke_iter_arg_t*)calloc((size_t)n_iter, sizeof(rocke_iter_arg_t));
        if(iter_args == NULL)
        {
            free(q_frags);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }
        {
            /* Stable storage for the generated iter-arg names. */
            static const char* const k_mnames[8] = {"m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"};
            static const char* const k_lnames[8] = {"l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"};
            static const char* const k_anames[32]
                = {"acc0",  "acc1",  "acc2",  "acc3",  "acc4",  "acc5",  "acc6",  "acc7",
                   "acc8",  "acc9",  "acc10", "acc11", "acc12", "acc13", "acc14", "acc15",
                   "acc16", "acc17", "acc18", "acc19", "acc20", "acc21", "acc22", "acc23",
                   "acc24", "acc25", "acc26", "acc27", "acc28", "acc29", "acc30", "acc31"};
            for(r = 0; r < c_frag; ++r)
            {
                iter_args[2 * r].name = k_mnames[r];
                iter_args[2 * r].init = neg_inf;
                iter_args[2 * r + 1].name = k_lnames[r];
                iter_args[2 * r + 1].init = zero_f;
            }
            for(d = 0; d < n_pv; ++d)
            {
                iter_args[2 * c_frag + d].name = k_anames[d];
                iter_args[2 * c_frag + d].init = rocke_b_zero_vec_f32(b, c_frag);
            }
        }

        /* c_block_k = const(_BLOCK_K); loop_stop = div(seqlen_k, c_block_k) */
        c_block_k = rocke_b_const_i32(b, ROCKE_WAF_BLOCK_K);
        loop_stop = rocke_b_div(b, seqlen_k, c_block_k);
        /* kloop = b.scf_for_iter(const(0), loop_stop, const(1), iter_args, iv_name="kt") */
        kloop = rocke_b_scf_for_iter(b,
                                     rocke_b_const_i32(b, 0),
                                     loop_stop,
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
                = (rocke_value_t**)calloc((size_t)n_pv, sizeof(rocke_value_t*));
            rocke_value_t** new_ms
                = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** new_ls
                = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** ps0 = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** ps1 = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** scores = (rocke_value_t**)calloc(2, sizeof(rocke_value_t*));
            rocke_value_t* k_tile_base;
            rocke_value_t* p_a;
            rocke_value_t** yields;
            int yi;

            if(!ms || !ls || !new_accs || !new_ms || !new_ls || !ps0 || !ps1 || !scores)
            {
                free(ms);
                free(ls);
                free(new_accs);
                free(new_ms);
                free(new_ls);
                free(ps0);
                free(ps1);
                free(scores);
                free(q_frags);
                free(iter_args);
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
                return NULL;
            }

            for(r = 0; r < c_frag; ++r)
            {
                ms[r] = state[2 * r];
                ls[r] = state[2 * r + 1];
            }
            for(d = 0; d < n_pv; ++d)
            {
                new_accs[d] = state[2 * c_frag + d];
            }

            /* k_tile_base = mul(kt, c_block_k) */
            k_tile_base = rocke_b_mul(b, kt, c_block_k);

            /* ---- QK^T: two N-sub-tiles, each summed over d ---- */
            for(nsub = 0; nsub < 2; ++nsub)
            {
                rocke_value_t* score = rocke_b_zero_vec_f32(b, c_frag);
                /* k_row = add(add(k_tile_base, const(nsub*_WMMA_N)), a_row) */
                rocke_value_t* k_row = rocke_b_add(
                    b,
                    rocke_b_add(b, k_tile_base, rocke_b_const_i32(b, nsub * ROCKE_WAF_WMMA_N)),
                    a_row);
                /* k_addr_row_base = add(add(mul(k_row, stride_k_token),
                 *                           mul(kv_head, stride_k_head)), batch_off_k)
                 * Sequence the two inner muls L-to-R. */
                rocke_value_t* k_km = rocke_b_mul(b, k_row, stride_k_token);
                rocke_value_t* k_hm = rocke_b_mul(b, kv_head, stride_k_head);
                rocke_value_t* k_addr_row_base
                    = rocke_b_add(b, rocke_b_add(b, k_km, k_hm), batch_off_k);
                for(d = 0; d < n_dk; ++d)
                {
                    rocke_value_t* k_addr = rocke_b_add(
                        b,
                        rocke_b_add(b, k_addr_row_base, rocke_b_const_i32(b, d * ROCKE_WAF_WMMA_K)),
                        half_k);
                    rocke_value_t* k_frag
                        = rocke_b_global_load_vN(b, K, k_addr, dtype_ir, a_frag, a_frag * 2);
                    score = rocke_b_mma(b, op->op_id, q_frags[d], k_frag, score, NULL, 0);
                    if(spec->mask_mode != NULL && strcmp(spec->mask_mode, "causal") == 0
                       && spacing > 0)
                    {
                        char nops[256];
                        int p = 0;
                        int ni;
                        for(ni = 0; ni < spacing && p < (int)sizeof(nops) - 6; ++ni)
                        {
                            p += snprintf(
                                nops + p, sizeof(nops) - (size_t)p, "%sv_nop", ni == 0 ? "" : "\n");
                        }
                        (void)rocke_b_inline_asm(b, nops, "", NULL, 0, NULL, 0, NULL);
                    }
                }
                scores[nsub] = score;
            }

            /* ---- scale + mask + online softmax over the 32 k columns ---- */
            for(d = 0; d < n_pv; ++d)
            {
                new_accs[d] = new_accs[d]; /* list(accs) copy is identity here */
            }
            for(r = 0; r < c_frag; ++r)
            {
                rocke_value_t* row_rel = NULL;
                rocke_value_t* col_k = NULL;
                rocke_value_t* row_q_pos;
                rocke_value_t* srs0;
                rocke_value_t* srs1;
                rocke_value_t* rm0;
                rocke_value_t* rm1;
                rocke_value_t* row_max;
                rocke_value_t* m_new;
                rocke_value_t* alpha;
                rocke_value_t* p0;
                rocke_value_t* p1;
                rocke_value_t* rs0;
                rocke_value_t* rs1;
                rocke_value_t* row_sum;
                rocke_value_t* l_new;

                rocke_waf_coord01(b, c_map, lane, r, &row_rel, &col_k);
                /* row_q_pos = add(q_row0, row_rel) */
                row_q_pos = rocke_b_add(b, q_row0, row_rel);

                /* nsub 0 then 1: s_r = fmul(vec_extract(scores[nsub], r), scale_log2),
                 * masked. */
                {
                    rocke_value_t* s_r
                        = rocke_b_fmul(b, rocke_b_vec_extract(b, scores[0], r), scale_log2);
                    rocke_value_t* k_col_pos = rocke_b_add(
                        b,
                        rocke_b_add(b, k_tile_base, rocke_b_const_i32(b, 0 * ROCKE_WAF_WMMA_N)),
                        col_k);
                    srs0 = rocke_apply_attention_mask(b,
                                                      s_r,
                                                      (strcmp(spec->mask_mode, "causal") == 0)
                                                          ? ROCKE_ATTN_MASK_CAUSAL
                                                          : ROCKE_ATTN_MASK_NONE,
                                                      k_col_pos,
                                                      row_q_pos,
                                                      spec->sliding_window,
                                                      rocke_b_const_i32(b, 0),
                                                      NULL);
                }
                {
                    rocke_value_t* s_r
                        = rocke_b_fmul(b, rocke_b_vec_extract(b, scores[1], r), scale_log2);
                    rocke_value_t* k_col_pos = rocke_b_add(
                        b,
                        rocke_b_add(b, k_tile_base, rocke_b_const_i32(b, 1 * ROCKE_WAF_WMMA_N)),
                        col_k);
                    srs1 = rocke_apply_attention_mask(b,
                                                      s_r,
                                                      (strcmp(spec->mask_mode, "causal") == 0)
                                                          ? ROCKE_ATTN_MASK_CAUSAL
                                                          : ROCKE_ATTN_MASK_NONE,
                                                      k_col_pos,
                                                      row_q_pos,
                                                      spec->sliding_window,
                                                      rocke_b_const_i32(b, 0),
                                                      NULL);
                }
                /* rm0/rm1 = wave_reduce_max(srs[i], wave, 16); row_max = fmax(rm0, rm1) */
                rm0 = rocke_wave_reduce_max(b, srs0, wave, 16);
                rm1 = rocke_wave_reduce_max(b, srs1, wave, 16);
                row_max = rocke_b_fmax(b, rm0, rm1);
                /* m_new = fmax(ms[r], row_max) */
                m_new = rocke_b_fmax(b, ms[r], row_max);
                /* alpha = exp2(fsub(ms[r], m_new)) */
                alpha = rocke_b_exp2(b, rocke_b_fsub(b, ms[r], m_new));
                /* p0 = exp2(fsub(srs0, m_new)); p1 = exp2(fsub(srs1, m_new)) */
                p0 = rocke_b_exp2(b, rocke_b_fsub(b, srs0, m_new));
                p1 = rocke_b_exp2(b, rocke_b_fsub(b, srs1, m_new));
                /* rs0/rs1 = wave_reduce_sum(p, wave, 16); row_sum = fadd(rs0, rs1) */
                rs0 = rocke_wave_reduce_sum(b, p0, wave, 16);
                rs1 = rocke_wave_reduce_sum(b, p1, wave, 16);
                row_sum = rocke_b_fadd(b, rs0, rs1);
                /* l_new = fadd(fmul(ls[r], alpha), row_sum) */
                l_new = rocke_b_fadd(b, rocke_b_fmul(b, ls[r], alpha), row_sum);
                new_ms[r] = m_new;
                new_ls[r] = l_new;
                ps0[r] = p0;
                ps1[r] = p1;
                for(d = 0; d < n_pv; ++d)
                {
                    /* old = vec_extract(new_accs[d], r);
                     * new_accs[d] = vec_insert(new_accs[d], fmul(old, alpha), r) */
                    rocke_value_t* old = rocke_b_vec_extract(b, new_accs[d], r);
                    new_accs[d]
                        = rocke_b_vec_insert(b, new_accs[d], rocke_b_fmul(b, old, alpha), r);
                }
            }

            /* ---- P staging: acc layout -> 16x32 LDS tile ---- */
            for(r = 0; r < c_frag; ++r)
            {
                rocke_value_t* row_rel = NULL;
                rocke_value_t* col_k = NULL;
                rocke_value_t* sidx[2];
                rocke_waf_coord01(b, c_map, lane, r, &row_rel, &col_k);
                sidx[0] = row_rel;
                sidx[1] = col_k;
                rocke_b_smem_store_vN(
                    b, P_lds, sidx, 2, rocke_b_cast_f32_to(b, ps0[r], dtype_ir), 1);
                sidx[1] = rocke_b_add(b, col_k, c16);
                rocke_b_smem_store_vN(
                    b, P_lds, sidx, 2, rocke_b_cast_f32_to(b, ps1[r], dtype_ir), 1);
            }
            rocke_b_sync(b);

            /* ---- PV: A=P (gfx1250 a-layout over 32 k), B=V (d x k gather) ---- */
            /* p_a = b.zero_vec(dtype_ir, a_frag); for j: insert P[a_row, a_k] */
            p_a = rocke_b_zero_vec(b, dtype_ir, a_frag);
            for(j = 0; j < a_frag; ++j)
            {
                rocke_value_t* a_k = NULL;
                rocke_value_t* sidx[2];
                rocke_value_t* p_v;
                {
                    rocke_value_t* t0 = NULL;
                    rocke_value_t* t1 = NULL;
                    rocke_waf_coord01(b, a_map, lane, j, &t0, &t1);
                    a_k = t1;
                }
                sidx[0] = a_row;
                sidx[1] = a_k;
                p_v = rocke_b_vec_extract(
                    b, rocke_b_smem_load_vN(b, P_lds, sidx, 2, dtype_ir, 1), 0);
                p_a = rocke_b_vec_insert(b, p_a, p_v, j);
            }

            for(d = 0; d < n_pv; ++d)
            {
                /* d_col = add(const(d*_WMMA_N), col) */
                rocke_value_t* d_col
                    = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_WAF_WMMA_N), col);
                rocke_value_t* v_b = rocke_b_zero_vec(b, dtype_ir, a_frag);
                for(j = 0; j < a_frag; ++j)
                {
                    rocke_value_t* v_k = NULL;
                    rocke_value_t* v_row;
                    rocke_value_t* v_row_base;
                    rocke_value_t* v_elem;
                    {
                        rocke_value_t* t0 = NULL;
                        rocke_value_t* t1 = NULL;
                        rocke_waf_coord01(b, a_map, lane, j, &t0, &t1);
                        v_k = t1;
                    }
                    /* v_row = add(k_tile_base, v_k) */
                    v_row = rocke_b_add(b, k_tile_base, v_k);
                    /* v_row_base = add(add(mul(v_row, stride_v_token),
                     *                      mul(kv_head, stride_v_head)), batch_off_v)
                     * Sequence the two inner muls L-to-R. */
                    {
                        rocke_value_t* v_vm = rocke_b_mul(b, v_row, stride_v_token);
                        rocke_value_t* v_hm = rocke_b_mul(b, kv_head, stride_v_head);
                        v_row_base = rocke_b_add(b, rocke_b_add(b, v_vm, v_hm), batch_off_v);
                    }
                    /* v_elem = global_load(V, add(v_row_base, d_col), dtype_ir, align=2) */
                    v_elem = rocke_b_global_load(
                        b, V, rocke_b_add(b, v_row_base, d_col), dtype_ir, /*align=*/2);
                    v_b = rocke_b_vec_insert(b, v_b, v_elem, j);
                }
                new_accs[d] = rocke_b_mma(b, op->op_id, p_a, v_b, new_accs[d], NULL, 0);
            }

            /* yields: [m0,l0,...] + new_accs */
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
                free(scores);
                free(q_frags);
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
            for(d = 0; d < n_pv; ++d)
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
            free(scores);
            free(yields);
        }
        rocke_b_region_leave(b);

        if(!rocke_ir_builder_ok(b) || kloop.op == NULL || kloop.op->num_results < n_iter)
        {
            free(q_frags);
            free(iter_args);
            return NULL;
        }

        /* final = kloop.results; ls_final[r] = final[2r+1]; accs_final = final[2*c_frag:] */
        {
            rocke_value_t** final = kloop.op->results;
            rocke_value_t** ls_final
                = (rocke_value_t**)calloc((size_t)c_frag, sizeof(rocke_value_t*));
            rocke_value_t** accs_final
                = (rocke_value_t**)calloc((size_t)n_pv, sizeof(rocke_value_t*));
            if(!ls_final || !accs_final)
            {
                free(ls_final);
                free(accs_final);
                free(q_frags);
                free(iter_args);
                (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
                return NULL;
            }
            for(r = 0; r < c_frag; ++r)
            {
                ls_final[r] = final[2 * r + 1];
            }
            for(d = 0; d < n_pv; ++d)
            {
                accs_final[d] = final[2 * c_frag + d];
            }

            /* ---- Epilogue: O[q,d] = acc / l (zero-denominator guarded) ---- */
            for(d = 0; d < n_pv; ++d)
            {
                for(r = 0; r < c_frag; ++r)
                {
                    rocke_value_t* row_rel = NULL;
                    rocke_value_t* col_n = NULL;
                    rocke_value_t* l_safe;
                    rocke_value_t* zero_mask;
                    rocke_value_t* inv_l;
                    rocke_value_t* v_f32;
                    rocke_value_t* o_row;
                    rocke_value_t* o_col;
                    rocke_value_t* o_addr;

                    rocke_waf_coord01(b, c_map, lane, r, &row_rel, &col_n);
                    l_safe = ls_final[r];
                    /* zero_mask = fcmp("oeq", l_safe, zero_f) */
                    zero_mask = rocke_b_fcmp(b, "oeq", l_safe, zero_f);
                    /* inv_l = select(zero_mask, zero_f, rcp(l_safe)) */
                    inv_l = rocke_b_select(b, zero_mask, zero_f, rocke_b_rcp(b, l_safe));
                    /* v_f32 = fmul(vec_extract(accs_final[d], r), inv_l) */
                    v_f32 = rocke_b_fmul(b, rocke_b_vec_extract(b, accs_final[d], r), inv_l);
                    /* o_row = add(add(q_row0, batch_row_q), row_rel) */
                    o_row = rocke_b_add(b, rocke_b_add(b, q_row0, batch_row_q), row_rel);
                    /* o_col = add(const(d*_WMMA_N), col_n) */
                    o_col = rocke_b_add(b, rocke_b_const_i32(b, d * ROCKE_WAF_WMMA_N), col_n);
                    /* o_addr = add(add(mul(o_row, stride_o_token),
                     *                  mul(head, stride_o_head)), o_col)
                     * Sequence the two inner muls L-to-R. */
                    {
                        rocke_value_t* o_om = rocke_b_mul(b, o_row, stride_o_token);
                        rocke_value_t* o_hm = rocke_b_mul(b, head, stride_o_head);
                        o_addr = rocke_b_add(b, rocke_b_add(b, o_om, o_hm), o_col);
                    }
                    /* global_store(O, o_addr, cast_f32_to(v_f32, dtype_ir), align=2) */
                    rocke_b_global_store(
                        b, O, o_addr, rocke_b_cast_f32_to(b, v_f32, dtype_ir), /*align=*/2);
                }
            }
            free(ls_final);
            free(accs_final);
        }

        /* b.ret() */
        rocke_b_ret(b);

        free(q_frags);
        free(iter_args);

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_wmma_attention_fwd_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_wmma_attention_fwd_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_wmma_attention_fwd_gfx1250(b, spec, arch);
    });
}

rocke_status_t rocke_wmma_attention_fwd_gfx1250_grid(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec, int seqlen_q, int batch, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(seqlen_q % ROCKE_WAF_BLOCK_M != 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = seqlen_q / ROCKE_WAF_BLOCK_M;
    out[1] = spec->num_query_heads;
    out[2] = batch;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_attention_fwd_gfx1250_lower_to_llvm(
    const rocke_wmma_attention_fwd_gfx1250_spec_t* spec,
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

    kernel = rocke_build_wmma_attention_fwd_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n = m ? strlen(m) : 0;
            if(m == NULL)
            {
                m = "build_wmma_attention_fwd_gfx1250 failed";
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
