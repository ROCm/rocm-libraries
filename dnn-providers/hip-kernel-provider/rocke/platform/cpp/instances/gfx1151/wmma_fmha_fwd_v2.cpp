// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_gfx1151_wmma_fmha_fwd_v2.c -- C99-API port of
 * kernels/gfx1151/wmma_fmha_fwd_v2.py.
 *
 * Byte-identical builder-call sequence vs the Python build_wmma_fmha_fwd_v2:
 * the same kernel attrs, the same parameters in the same order, the same CTA
 * decode and per-(batch, head) buffer resources, then
 * rocke_rdna_attention_fwd_body. Nested Python expressions are hoisted into
 * ordered temporaries (C argument-evaluation order is unspecified).
 */

#include "rocke/instance_gfx1151_wmma_fmha_fwd_v2.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.rdna_attention.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define WMMA_V2_DEFAULT_NAME "rocke_wmma_fmha_fwd_v2"
#define WMMA_V2_MAX_THREADS 1024
#define WMMA_V2_MAX_LDS_BYTES 65536

namespace
{

bool v2_is_bf16(const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    return strcmp(spec->dtype, "bf16") == 0;
}

int v2_dv(const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    return spec->head_dim_v != 0 ? spec->head_dim_v : spec->head_dim;
}

bool v2_pair(const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    if(spec->wave_pair >= 0)
        return spec->wave_pair != 0;
    const int dv = v2_dv(spec);
    return (spec->head_dim > dv ? spec->head_dim : dv) > 128;
}

int v2_bm(const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    return spec->block_m != 0 ? spec->block_m : (v2_pair(spec) ? 64 : 128);
}

bool v2_causal(const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    return strcmp(spec->mask, "causal") == 0;
}

bool v2_fp8(const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    return spec->kv_dtype != NULL && spec->kv_dtype[0] != '\0';
}

bool v2_layout_is(const rocke_wmma_fmha_fwd_v2_spec_t* spec, const char* layout)
{
    return strcmp(spec->layout, layout) == 0;
}

/* WmmaFmhaFwdV2Spec.__post_init__ (as a predicate + reason). */
bool v2_post_init_ok(const rocke_wmma_fmha_fwd_v2_spec_t* spec, char* reason, size_t cap)
{
    char msg[160];
    if(!rocke_scheduler_strategy_is_valid(spec->scheduler_strategy))
    {
        snprintf(msg, sizeof(msg), "unsupported scheduler_strategy '%s'", spec->scheduler_strategy);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(spec->dtype == NULL
       || (strcmp(spec->dtype, "fp16") != 0 && strcmp(spec->dtype, "bf16") != 0))
    {
        snprintf(msg,
                 sizeof(msg),
                 "dtype must be fp16/bf16, got '%s'",
                 spec->dtype ? spec->dtype : "None");
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(spec->mask == NULL || (strcmp(spec->mask, "none") != 0 && strcmp(spec->mask, "causal") != 0))
    {
        snprintf(msg,
                 sizeof(msg),
                 "mask must be none/causal, got '%s'",
                 spec->mask ? spec->mask : "None");
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(spec->bias != NULL && strcmp(spec->bias, "f32") != 0 && strcmp(spec->bias, "q") != 0)
    {
        snprintf(msg, sizeof(msg), "bias must be None, 'f32' or 'q', got '%s'", spec->bias);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(spec->cta_order == NULL
       || (strcmp(spec->cta_order, "hqb") != 0 && strcmp(spec->cta_order, "hbq") != 0))
    {
        snprintf(msg,
                 sizeof(msg),
                 "cta_order must be hqb/hbq, got '%s'",
                 spec->cta_order ? spec->cta_order : "None");
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(spec->head_dim_v < 0 || spec->block_m < 0 || spec->waves_per_eu < 0)
    {
        rocke_spec_set_reason(reason, cap, "head_dim_v, block_m and waves_per_eu must be >= 0");
        return false;
    }
    if(spec->layout == NULL
       || (strcmp(spec->layout, "dense") != 0 && strcmp(spec->layout, "ragged") != 0
           && strcmp(spec->layout, "paged") != 0))
    {
        snprintf(msg,
                 sizeof(msg),
                 "layout must be dense/ragged/paged, got '%s'",
                 spec->layout ? spec->layout : "None");
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const int ps = spec->page_block_size;
    if(strcmp(spec->layout, "paged") == 0)
    {
        if(ps <= 0 || (ps & (ps - 1)) != 0)
        {
            rocke_spec_set_reason(reason, cap, "paged attention needs a power-of-two page size");
            return false;
        }
    }
    else if(ps != 0)
    {
        rocke_spec_set_reason(reason, cap, "page_block_size is only valid for the paged layout");
        return false;
    }
    if(spec->kv_dtype == NULL
       || (strcmp(spec->kv_dtype, "") != 0 && strcmp(spec->kv_dtype, "fp8e4m3") != 0))
    {
        snprintf(msg,
                 sizeof(msg),
                 "kv_dtype must be '' or 'fp8e4m3', got '%s'",
                 spec->kv_dtype ? spec->kv_dtype : "None");
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(spec->wave_pair < -1 || spec->wave_pair > 1)
    {
        rocke_spec_set_reason(reason, cap, "wave_pair must be -1 (auto), 0 or 1");
        return false;
    }
    return true;
}

/* is_gfx12_family(arch); false for an unknown arch. */
bool v2_gfx12(const rocke_archtarget_t* target)
{
    return rocke_archtarget_by_op_id(target, "wmma_gfx12_f32_16x16x16_bf16") != NULL;
}

rocke_rdna_attn_config_t v2_body_config(const rocke_wmma_fmha_fwd_v2_spec_t* spec, bool gfx12)
{
    rocke_rdna_attn_config_t cfg;
    cfg.dtype = spec->dtype;
    cfg.head_dim = spec->head_dim;
    cfg.head_dim_v = v2_dv(spec);
    cfg.block_m = v2_bm(spec);
    cfg.block_n = spec->block_n;
    cfg.causal = v2_causal(spec);
    cfg.gfx12 = gfx12;
    cfg.lazy_rescale = spec->lazy_rescale;
    cfg.k_pad = spec->k_pad;
    cfg.v_pad = spec->v_pad;
    cfg.bias = spec->bias;
    cfg.wave_pair = v2_pair(spec) ? 1 : 0;
    cfg.softcap = spec->use_softcap;
    cfg.sinks = spec->use_sinks;
    cfg.alibi = spec->use_alibi;
    cfg.qq_bias = spec->use_qq_bias;
    cfg.kv_fp8 = v2_fp8(spec);
    cfg.page_block_size = spec->page_block_size;
    return cfg;
}

/* _declare_params: identical names, types, attrs and order. */
void v2_declare_params(rocke_ir_builder_t* b, const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    const rocke_type_t* elem = v2_is_bf16(spec) ? rocke_bf16() : rocke_f16();
    const rocke_type_t* ptr_io = rocke_ptr_type(b, elem, "global");
    const bool fp8 = v2_fp8(spec);
    const rocke_type_t* ptr_kv = fp8 ? rocke_ptr_type(b, rocke_fp8e4m3(), "global") : ptr_io;
    rocke_param_opts_t opts;
    const char* io[] = {"Q", "K", "V", "O"};
    for(int i = 0; i < 4; ++i)
    {
        const bool kv = i == 1 || i == 2;
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.align = kv && fp8 ? 1 : 2;
        opts.align_set = true;
        if(i < 3)
        {
            opts.readonly = true;
            opts.readonly_set = true;
        }
        else
        {
            opts.writeonly = true;
            opts.writeonly_set = true;
        }
        (void)rocke_b_param(b, io[i], kv ? ptr_kv : ptr_io, &opts);
    }
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    (void)rocke_b_param(b, "LSE", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    (void)rocke_b_param(b, "scale_log2", rocke_f32(), NULL);
    const char* scalars[] = {"batch",
                             "seqlen_q",
                             "seqlen_k",
                             "num_query_heads",
                             "num_kv_heads",
                             "num_v_heads",
                             "head_dim_q",
                             "head_dim_v",
                             "bottom_right",
                             "window_left",
                             "window_right",
                             "write_lse"};
    for(const char* name : scalars)
        (void)rocke_b_param(b, name, rocke_i32(), NULL);
    const char* tensors[] = {"q", "k", "v", "o", "lse"};
    const char* axes[] = {"batch", "token", "head"};
    for(const char* t : tensors)
    {
        for(const char* axis : axes)
        {
            char name[32];
            snprintf(name, sizeof(name), "stride_%s_%s", t, axis);
            (void)rocke_b_param(b, name, rocke_i32(), NULL);
        }
    }

    auto ptr = [&](const char* name, const rocke_type_t* pointee, int align) {
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = align;
        opts.align_set = true;
        (void)rocke_b_param(b, name, rocke_ptr_type(b, pointee, "global"), &opts);
    };
    auto scalar = [&](const char* name, const rocke_type_t* type) {
        (void)rocke_b_param(b, name, type, NULL);
    };
    if(spec->use_softcap)
        scalar("softcap", rocke_f32());
    if(spec->use_sinks)
        ptr("sink_ptr", elem, 2);
    if(spec->use_alibi)
        ptr("alibi_slopes_ptr", rocke_f32(), 4);
    if(spec->use_qq_bias)
    {
        ptr("qq_bias_ptr", rocke_f32(), 4);
        scalar("qq_bias_rows", rocke_i32());
        scalar("qq_bias_cols", rocke_i32());
        scalar("qq_bias_stride", rocke_i32());
    }
    if(!v2_layout_is(spec, "dense"))
    {
        ptr("cu_seqlens_q", rocke_i32(), 4);
        if(v2_layout_is(spec, "ragged"))
            ptr("cu_seqlens_k", rocke_i32(), 4);
        else
        {
            ptr("seqused_k", rocke_i32(), 4);
            ptr("block_table", rocke_i32(), 4);
            scalar("block_table_stride", rocke_i32());
            scalar("stride_k_block", rocke_i32());
            scalar("stride_v_block", rocke_i32());
            scalar("num_blocks", rocke_i32());
        }
    }
    if(fp8)
    {
        scalar("k_scale", rocke_f32());
        scalar("v_scale", rocke_f32());
    }
    if(spec->bias != NULL)
    {
        const bool f32 = strcmp(spec->bias, "f32") == 0;
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = f32 ? 4 : 2;
        opts.align_set = true;
        (void)rocke_b_param(
            b, "attn_bias_ptr", rocke_ptr_type(b, f32 ? rocke_f32() : elem, "global"), &opts);
        /* Element strides; the key stride is 1, a 0 stride broadcasts. */
        const char* bias_axes[] = {"bias_stride_b", "bias_stride_h", "bias_stride_q"};
        for(const char* name : bias_axes)
            (void)rocke_b_param(b, name, rocke_i32(), NULL);
    }
}

/* The Python build body after validation; `b` holds the kernel name. */
rocke_status_t
    v2_emit_body(rocke_ir_builder_t* b, const rocke_wmma_fmha_fwd_v2_spec_t* spec, bool gfx12)
{
    const rocke_rdna_attn_config_t cfg = v2_body_config(spec, gfx12);
    const int bm = v2_bm(spec);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", rocke_rdna_attn_threads(&cfg));
    if(spec->scheduler_strategy != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->scheduler_strategy);
    if(spec->waves_per_eu != 0)
    {
        const int64_t wpe[2] = {spec->waves_per_eu, spec->waves_per_eu};
        rocke_attr_set_int_list(b, &b->kernel->attrs, "waves_per_eu", wpe, 2);
    }
    v2_declare_params(b, spec);

    auto ci = [&](int64_t v) { return rocke_b_const_i32(b, v); };
    auto param = [&](const char* name) { return rocke_b_get_param(b, name); };
    rocke_value_t* c0 = ci(0);
    rocke_value_t* c1 = ci(1);
    rocke_value_t* sq = param("seqlen_q");
    rocke_value_t* sk = param("seqlen_k");
    rocke_value_t* hq = param("num_query_heads");

    /* ---- CTA decode: head fastest, then (query tile, batch) in spec order ---- */
    rocke_value_t* bid = rocke_b_block_id_x(b);
    rocke_value_t* head = rocke_b_mod(b, bid, hq);
    rocke_value_t* rest = rocke_b_div(b, bid, hq);
    rocke_value_t* n_qt;
    {
        rocke_value_t* t = ci(bm - 1);
        rocke_value_t* a = rocke_b_add(b, sq, t);
        rocke_value_t* tb = ci(bm);
        n_qt = rocke_b_div(b, a, tb);
    }
    rocke_value_t *qt, *batch;
    if(strcmp(spec->cta_order, "hqb") == 0)
    {
        qt = rocke_b_mod(b, rest, n_qt);
        batch = rocke_b_div(b, rest, n_qt);
    }
    else
    {
        batch = rocke_b_mod(b, rest, param("batch"));
        qt = rocke_b_div(b, rest, param("batch"));
    }
    if(v2_causal(spec))
    {
        rocke_value_t* last = rocke_b_sub(b, n_qt, c1);
        qt = rocke_b_sub(b, last, qt); /* heaviest tiles first (LPT) */
    }
    rocke_value_t* q_start;
    {
        rocke_value_t* tb = ci(bm);
        q_start = rocke_b_mul(b, qt, tb);
    }
    rocke_value_t* kv_head;
    {
        rocke_value_t* g = rocke_b_div(b, hq, param("num_kv_heads"));
        kv_head = rocke_b_div(b, head, g);
    }
    rocke_value_t* v_head;
    {
        /* V may carry its own head count (any Hv dividing Hq). */
        rocke_value_t* g = rocke_b_div(b, hq, param("num_v_heads"));
        v_head = rocke_b_div(b, head, g);
    }

    /* ---- packed layouts: this sequence's rows and lengths ---- */
    rocke_value_t* q_row0 = NULL;
    rocke_value_t* k_row0 = NULL;
    if(!v2_layout_is(spec, "dense"))
    {
        rocke_value_t* nxt = rocke_b_add(b, batch, c1);
        q_row0 = rocke_b_global_load_i32(b, param("cu_seqlens_q"), batch, 4);
        rocke_value_t* q_end = rocke_b_global_load_i32(b, param("cu_seqlens_q"), nxt, 4);
        sq = rocke_b_sub(b, q_end, q_row0);
        rocke_value_t* past = rocke_b_cmp_ge(b, q_start, sq);
        rocke_if_t guard = rocke_b_scf_if(b, past);
        rocke_b_region_enter(b, guard.then_region);
        rocke_b_ret(b); /* past this sequence's queries */
        rocke_b_region_leave(b);
        if(v2_layout_is(spec, "ragged"))
        {
            k_row0 = rocke_b_global_load_i32(b, param("cu_seqlens_k"), batch, 4);
            rocke_value_t* k_end = rocke_b_global_load_i32(b, param("cu_seqlens_k"), nxt, 4);
            sk = rocke_b_sub(b, k_end, k_row0);
        }
        else
            sk = rocke_b_global_load_i32(b, param("seqused_k"), batch, 4);
    }

    /* Resource over one (batch, head) slice: (rows - 1) * token stride +
     * width elements, empty when there are no rows (or not enabled). A
     * packed sequence starts at token `row0` instead of at its batch. */
    auto slice_rsrc = [&](const char* ptr,
                          const char* tensor,
                          rocke_value_t* head_idx,
                          rocke_value_t* rows,
                          rocke_value_t* width,
                          int elem_bytes,
                          rocke_value_t* enabled,
                          rocke_value_t* row0,
                          rocke_value_t** nbytes_out) {
        char sb[32], st[32], sh[32];
        snprintf(sb, sizeof(sb), "stride_%s_batch", tensor);
        snprintf(st, sizeof(st), "stride_%s_token", tensor);
        snprintf(sh, sizeof(sh), "stride_%s_head", tensor);
        rocke_value_t* first
            = row0 == NULL ? rocke_b_mul(b, batch, param(sb)) : rocke_b_mul(b, row0, param(st));
        rocke_value_t* mh = rocke_b_mul(b, head_idx, param(sh));
        rocke_value_t* base = rocke_b_add(b, first, mh);
        rocke_value_t* te = ci(elem_bytes);
        rocke_value_t* bo = rocke_b_mul(b, base, te);
        rocke_value_t* sliced = rocke_b_global_ptr_add(b, param(ptr), bo);
        rocke_value_t* r1 = rocke_b_sub(b, rows, c1);
        rocke_value_t* rs = rocke_b_mul(b, r1, param(st));
        rocke_value_t* extent = rocke_b_add(b, rs, width);
        rocke_value_t* te2 = ci(elem_bytes);
        rocke_value_t* nbytes = rocke_b_mul(b, extent, te2);
        rocke_value_t* live = rocke_b_cmp_gt(b, rows, c0);
        if(enabled != NULL)
            live = rocke_b_land(b, live, enabled);
        nbytes = rocke_b_select(b, live, nbytes, c0);
        if(nbytes_out != NULL)
            *nbytes_out = nbytes;
        return rocke_b_buffer_rsrc(b, sliced, nbytes);
    };

    const int kv_eb = v2_fp8(spec) ? 1 : 2;
    /* Paged K / V: the whole cache at this head (the binding bounds the cache
     * to I32 bytes; dead rows are masked per load). */
    auto page_rsrc = [&](const char* ptr, const char* tensor, rocke_value_t* head_idx) {
        char sh[32];
        snprintf(sh, sizeof(sh), "stride_%s_head", tensor);
        rocke_value_t* mh = rocke_b_mul(b, head_idx, param(sh));
        rocke_value_t* te = ci(kv_eb);
        rocke_value_t* off = rocke_b_mul(b, mh, te);
        rocke_value_t* sliced = rocke_b_global_ptr_add(b, param(ptr), off);
        rocke_value_t* all = ci(0x7FFFFFFF);
        return rocke_b_buffer_rsrc(b, sliced, all);
    };

    rocke_value_t* hdq = param("head_dim_q");
    rocke_value_t* hdv = param("head_dim_v");
    rocke_rdna_attn_params_t p;
    memset(&p, 0, sizeof(p));
    p.q_rsrc = slice_rsrc("Q", "q", head, sq, hdq, 2, NULL, q_row0, NULL);
    if(v2_layout_is(spec, "paged"))
    {
        p.k_rsrc = page_rsrc("K", "k", kv_head);
        p.v_rsrc = page_rsrc("V", "v", v_head);
    }
    else
    {
        p.k_rsrc = slice_rsrc("K", "k", kv_head, sk, hdq, kv_eb, NULL, k_row0, &p.k_bytes);
        p.v_rsrc = slice_rsrc("V", "v", v_head, sk, hdv, kv_eb, NULL, k_row0, &p.v_bytes);
    }
    p.o_rsrc = slice_rsrc("O", "o", head, sq, hdv, 2, NULL, q_row0, NULL);
    {
        rocke_value_t* enabled = rocke_b_cmp_ne(b, param("write_lse"), c0);
        p.lse_rsrc = slice_rsrc("LSE", "lse", head, sq, c1, 4, enabled, q_row0, NULL);
    }
    if(spec->bias != NULL)
    {
        const int eb = strcmp(spec->bias, "f32") == 0 ? 4 : 2;
        rocke_value_t* mb = rocke_b_mul(b, batch, param("bias_stride_b"));
        rocke_value_t* mh = rocke_b_mul(b, head, param("bias_stride_h"));
        rocke_value_t* base = rocke_b_add(b, mb, mh);
        rocke_value_t* te = ci(eb);
        rocke_value_t* bo = rocke_b_mul(b, base, te);
        rocke_value_t* sliced = rocke_b_global_ptr_add(b, param("attn_bias_ptr"), bo);
        rocke_value_t* r1 = rocke_b_sub(b, sq, c1);
        rocke_value_t* rs = rocke_b_mul(b, r1, param("bias_stride_q"));
        rocke_value_t* extent = rocke_b_add(b, rs, sk);
        rocke_value_t* lq = rocke_b_cmp_gt(b, sq, c0);
        rocke_value_t* lk = rocke_b_cmp_gt(b, sk, c0);
        rocke_value_t* live = rocke_b_land(b, lq, lk);
        rocke_value_t* te2 = ci(eb);
        rocke_value_t* nb = rocke_b_mul(b, extent, te2);
        if(eb == 2)
        {
            /* A Q-dtype slice may start or end mid-dword, and the hardware
             * drops a loaded dword that ends past num_records: the resource
             * starts at the dword below the slice and spans whole dwords
             * (bias_lead elements in). */
            rocke_value_t* addr = rocke_b_global_ptr_to_i64(b, sliced);
            rocke_value_t* lo = rocke_b_trunc(b, addr, rocke_i32());
            rocke_value_t* t2 = ci(2);
            rocke_value_t* lead_bytes = rocke_b_land(b, lo, t2);
            rocke_value_t* z = rocke_b_const_i64(b, 0);
            rocke_value_t* lead64 = rocke_b_zext(b, lead_bytes, rocke_i64());
            rocke_value_t* back = rocke_b_sub(b, z, lead64);
            sliced = rocke_b_global_ptr_add(b, sliced, back);
            rocke_value_t* s1 = rocke_b_add(b, nb, lead_bytes);
            rocke_value_t* t3 = ci(3);
            rocke_value_t* s2 = rocke_b_add(b, s1, t3);
            rocke_value_t* tm = ci(-4);
            nb = rocke_b_land(b, s2, tm);
            p.bias_lead = rocke_b_lshr(b, lead_bytes, c1);
        }
        p.bias_bytes = rocke_b_select(b, live, nb, c0);
        p.bias_rsrc = rocke_b_buffer_rsrc(b, sliced, p.bias_bytes);
        p.stride_bias_q = param("bias_stride_q");
    }

    /* ---- score features ---- */
    if(spec->use_qq_bias)
    {
        rocke_value_t* rows = param("qq_bias_rows");
        rocke_value_t* cols = param("qq_bias_cols");
        rocke_value_t* r1 = rocke_b_sub(b, rows, c1);
        rocke_value_t* rs = rocke_b_mul(b, r1, param("qq_bias_stride"));
        rocke_value_t* extent = rocke_b_add(b, rs, cols);
        rocke_value_t* lr = rocke_b_cmp_gt(b, rows, c0);
        rocke_value_t* lc = rocke_b_cmp_gt(b, cols, c0);
        rocke_value_t* live = rocke_b_land(b, lr, lc);
        rocke_value_t* t4 = ci(4);
        rocke_value_t* nb = rocke_b_mul(b, extent, t4);
        rocke_value_t* nbytes = rocke_b_select(b, live, nb, c0);
        p.qq_rsrc = rocke_b_buffer_rsrc(b, param("qq_bias_ptr"), nbytes);
        p.qq_rows = rows;
        p.qq_cols = cols;
        p.qq_stride = param("qq_bias_stride");
    }
    rocke_value_t* log2e = NULL;
    if(spec->use_sinks || spec->use_alibi)
        log2e = rocke_b_const_f32(b, 1.4426950408889634);
    if(spec->use_sinks)
    {
        const rocke_type_t* elem = v2_is_bf16(spec) ? rocke_bf16() : rocke_f16();
        rocke_value_t* raw = rocke_b_global_load(b, param("sink_ptr"), head, elem, 2);
        rocke_value_t* f = rocke_b_cast_to_f32(b, raw);
        p.sink_log2 = rocke_b_fmul(b, f, log2e);
    }
    if(spec->use_alibi)
    {
        rocke_value_t* raw
            = rocke_b_global_load(b, param("alibi_slopes_ptr"), head, rocke_f32(), 4);
        p.alibi_slope_log2 = rocke_b_fmul(b, raw, log2e);
    }
    if(v2_layout_is(spec, "paged"))
    {
        p.table_row = rocke_b_mul(b, batch, param("block_table_stride"));
        p.block_table = param("block_table");
        p.stride_k_block = param("stride_k_block");
        p.stride_v_block = param("stride_v_block");
        p.num_blocks = param("num_blocks");
    }
    if(spec->use_softcap)
        p.softcap = param("softcap");
    if(v2_fp8(spec))
    {
        p.k_scale = param("k_scale");
        p.v_scale = param("v_scale");
    }

    const bool positional = spec->use_alibi || spec->use_qq_bias;
    rocke_value_t* diag = NULL;
    if(v2_causal(spec) || positional)
    {
        rocke_value_t* br = rocke_b_cmp_ne(b, param("bottom_right"), c0);
        rocke_value_t* d = rocke_b_sub(b, sk, sq);
        diag = rocke_b_select(b, br, d, c0);
    }
    if(v2_causal(spec))
    {
        p.diag_offset = diag;
        p.window_left = param("window_left");
        p.window_right = param("window_right");
    }
    if(positional)
        p.context = diag;
    p.stride_q = param("stride_q_token");
    p.stride_k = param("stride_k_token");
    p.stride_v = param("stride_v_token");
    p.stride_o = param("stride_o_token");
    p.stride_lse = param("stride_lse_token");
    p.head_dim_q = hdq;
    p.head_dim_v = hdv;
    p.seqlen_q = sq;
    p.seqlen_k = sk;
    p.q_start = q_start;
    p.scale_log2 = param("scale_log2");
    if(rocke_rdna_attention_fwd_body(b, &cfg, &p) != ROCKE_OK)
        return rocke_ir_builder_status(b);
    rocke_b_ret(b);
    return rocke_ir_builder_status(b);
}

/* Validity with the resolved target; `gfx12_out` receives is_gfx12_family. */
bool v2_valid(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
              const char* arch,
              bool* gfx12_out,
              char* reason,
              size_t cap)
{
    char msg[200];
    if(spec == NULL || arch == NULL)
    {
        rocke_spec_set_reason(reason, cap, "null spec/arch");
        return false;
    }
    if(!v2_post_init_ok(spec, reason, cap))
        return false;
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(msg, sizeof(msg), "unknown gfx target '%s'", arch);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const bool gfx12 = v2_gfx12(target);
    const std::string op_id = std::string(gfx12 ? "wmma_gfx12_f32_16x16x16_" : "wmma_f32_16x16x16_")
                              + (v2_is_bf16(spec) ? "bf16" : "f16");
    if(target->wave_size != 32 || rocke_archtarget_by_op_id(target, op_id.c_str()) == NULL)
    {
        snprintf(msg, sizeof(msg), "%s has no wave32 %s WMMA atom", arch, spec->dtype);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const int threads = rocke_wmma_fmha_fwd_v2_threads(spec);
    if(threads > WMMA_V2_MAX_THREADS)
    {
        snprintf(msg, sizeof(msg), "%d threads per CTA exceed %d", threads, WMMA_V2_MAX_THREADS);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    const rocke_rdna_attn_config_t cfg = v2_body_config(spec, gfx12);
    if(!rocke_rdna_attn_config_validate(&cfg, reason, cap))
        return false;
    if(rocke_rdna_attention_lds_bytes(&cfg) > WMMA_V2_MAX_LDS_BYTES)
    {
        snprintf(msg, sizeof(msg), "LDS tiles exceed %d bytes", WMMA_V2_MAX_LDS_BYTES);
        rocke_spec_set_reason(reason, cap, msg);
        return false;
    }
    if(gfx12_out != NULL)
        *gfx12_out = gfx12;
    return true;
}

} // namespace

rocke_wmma_fmha_fwd_v2_spec_t rocke_wmma_fmha_fwd_v2_spec_default(void)
{
    rocke_wmma_fmha_fwd_v2_spec_t s;
    s.dtype = "bf16";
    s.head_dim = 128;
    s.head_dim_v = 0;
    s.mask = "none";
    s.bias = NULL;
    s.use_softcap = false;
    s.use_sinks = false;
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.layout = "dense";
    s.page_block_size = 0;
    s.kv_dtype = "";
    s.block_m = 0;
    s.block_n = 32;
    s.waves_per_eu = 2;
    s.scheduler_strategy = NULL;
    s.lazy_rescale = true;
    s.k_pad = 8;
    s.v_pad = 8;
    s.wave_pair = -1;
    s.cta_order = "hqb";
    s.name = WMMA_V2_DEFAULT_NAME;
    return s;
}

int rocke_wmma_fmha_fwd_v2_threads(const rocke_wmma_fmha_fwd_v2_spec_t* spec)
{
    return 2 * v2_bm(spec) * (v2_pair(spec) ? 2 : 1);
}

rocke_status_t rocke_wmma_fmha_fwd_v2_kernel_name(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(spec == NULL || out == NULL || !v2_post_init_ok(spec, reason, sizeof(reason)))
        return ROCKE_ERR_VALUE;
    const int dv = v2_dv(spec);
    std::string name = spec->name != NULL ? spec->name : WMMA_V2_DEFAULT_NAME;
    name += v2_is_bf16(spec) ? "_bf16" : "_f16";
    name += "_d" + std::to_string(spec->head_dim);
    if(dv != spec->head_dim)
        name += "v" + std::to_string(dv);
    name += std::string("_") + spec->mask;
    if(spec->bias != NULL)
        name += std::string("_bias") + spec->bias;
    if(spec->use_softcap)
        name += "_softcap";
    if(spec->use_sinks)
        name += "_sinks";
    if(spec->use_alibi)
        name += "_alibi";
    if(spec->use_qq_bias)
        name += "_qqbias";
    if(v2_layout_is(spec, "ragged"))
        name += "_ragged";
    if(v2_layout_is(spec, "paged"))
        name += "_paged" + std::to_string(spec->page_block_size);
    if(v2_fp8(spec))
        name += "_kvfp8";
    name += "_m" + std::to_string(v2_bm(spec)) + "n" + std::to_string(spec->block_n);
    if(v2_pair(spec))
        name += "p";
    if(strcmp(spec->cta_order, "hbq") == 0)
        name += "_hbq";
    if(spec->waves_per_eu != 2)
        name += "_wpe" + std::to_string(spec->waves_per_eu);
    if(spec->scheduler_strategy != NULL)
    {
        std::string sched = spec->scheduler_strategy;
        for(char& ch : sched)
            if(ch == '-')
                ch = '_';
        name += "_sched_" + sched;
    }
    if(!spec->lazy_rescale)
        name += "_eager";
    if(spec->k_pad != 8 || spec->v_pad != 8)
        name += "_pad" + std::to_string(spec->k_pad) + "x" + std::to_string(spec->v_pad);
    if(name.size() + 1 > out_cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, name.c_str(), name.size() + 1);
    return ROCKE_OK;
}

bool rocke_wmma_fmha_fwd_v2_is_valid_spec(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    return v2_valid(spec, arch, NULL, reason, reason_cap);
}

int rocke_wmma_fmha_fwd_v2_lds_bytes(const rocke_wmma_fmha_fwd_v2_spec_t* spec, const char* arch)
{
    if(spec == NULL || arch == NULL)
        return -1;
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
        return -1;
    const rocke_rdna_attn_config_t cfg = v2_body_config(spec, v2_gfx12(target));
    return rocke_rdna_attention_lds_bytes(&cfg);
}

rocke_kernel_def_t* rocke_build_wmma_fmha_fwd_v2(rocke_ir_builder_t* b,
                                                 const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[ROCKE_ERR_MSG_CAP];
        bool gfx12 = false;
        if(b == NULL || spec == NULL)
            return NULL;
        if(!v2_valid(spec, arch, &gfx12, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid wmma_fmha_fwd_v2 spec: %s", reason);
            return NULL;
        }
        if(v2_emit_body(b, spec, gfx12) != ROCKE_OK)
            return NULL;
        return b->kernel;
    });
}

rocke_status_t rocke_wmma_fmha_fwd_v2_grid(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                           int batch,
                                           int seqlen_q,
                                           int num_query_heads,
                                           int grid[3],
                                           int block[3])
{
    if(spec == NULL || grid == NULL || block == NULL)
        return ROCKE_ERR_VALUE;
    const int bm = v2_bm(spec);
    if(bm <= 0)
        return ROCKE_ERR_VALUE;
    const int64_t n_qt = (static_cast<int64_t>(seqlen_q) + bm - 1) / bm;
    const int64_t n = static_cast<int64_t>(batch) * n_qt * num_query_heads;
    if(!(0 < n && n <= 0x7FFFFFFF))
        return ROCKE_ERR_VALUE;
    grid[0] = static_cast<int>(n);
    grid[1] = 1;
    grid[2] = 1;
    block[0] = rocke_wmma_fmha_fwd_v2_threads(spec);
    block[1] = 1;
    block[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_fwd_v2_signature(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                rocke_arena_t* arena,
                                                const rocke_sig_entry_t** out_items,
                                                size_t* out_count)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL
       || !v2_post_init_ok(spec, reason, sizeof(reason)))
        return ROCKE_ERR_VALUE;
    rocke_ir_builder_t b;
    rocke_status_t status = rocke_ir_builder_init(&b, "rocke_wmma_fmha_fwd_v2_sig_probe");
    if(status != ROCKE_OK)
        return status;
    v2_declare_params(&b, spec);
    status = rocke_ir_builder_status(&b);
    if(status != ROCKE_OK)
    {
        rocke_ir_builder_free(&b);
        return status;
    }
    const int count = b.kernel->num_params;
    auto* items = static_cast<rocke_sig_entry_t*>(
        rocke_arena_alloc(arena, static_cast<size_t>(count) * sizeof(rocke_sig_entry_t)));
    if(items == NULL)
    {
        rocke_ir_builder_free(&b);
        return ROCKE_ERR_OOM;
    }
    for(int i = 0; i < count; ++i)
    {
        const rocke_param_t* param = b.kernel->params[i];
        const rocke_type_t* type = param->type;
        if(type->kind == ROCKE_TYPE_PTR)
            status
                = rocke_sig_param(arena, param->name, type->pointee->name, type->space, &items[i]);
        else
            status = rocke_sig_scalar(arena, param->name, type->name, &items[i]);
        if(status != ROCKE_OK)
        {
            rocke_ir_builder_free(&b);
            return status;
        }
    }
    rocke_ir_builder_free(&b);
    *out_items = items;
    *out_count = static_cast<size_t>(count);
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_fmha_fwd_v2_lower_to_llvm(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap)
{
    char name_buf[256];
    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL || arch == NULL)
    {
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: null spec/arch/out");
        return ROCKE_ERR_VALUE;
    }
    if(rocke_wmma_fmha_fwd_v2_kernel_name(spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
    {
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: invalid spec or name too long");
        return ROCKE_ERR_VALUE;
    }
    rocke_ir_builder_t b;
    rocke_status_t st = rocke_ir_builder_init(&b, name_buf);
    if(st != ROCKE_OK)
    {
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: builder init failed");
        return st;
    }
    rocke_kernel_def_t* k = rocke_build_wmma_fmha_fwd_v2(&b, spec, arch);
    if(k == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        rocke_spec_set_reason(err, err_cap, m != NULL ? m : "build_wmma_fmha_fwd_v2 failed");
        st = rocke_ir_builder_status(&b);
        rocke_ir_builder_free(&b);
        return st != ROCKE_OK ? st : ROCKE_ERR_VALUE;
    }
    st = rocke_lower_kernel_to_llvm_ex(k, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
